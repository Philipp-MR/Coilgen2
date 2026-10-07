"""Intrinsic joins follow cylinder facets without changing directed ports."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.geometry import build_cylinder_mesh
from coilgen.connector_bridge_validation import _validate_support
from coilgen.config import CoilGenConfig
from coilgen.joint_cylinder_curves import (
    CYLINDER_TRANSITION_TRIALS,
    _cylinder_chart,
    _split_facets,
    generate_cylindrical_transition_curves,
)
from coilgen.joint_curve_search import cylinder_trial_count
from coilgen.manufacturing import _SurfaceProjector, _prepare_sweep_centerline, conductor_cross_section_points
from coilgen.models import CoilPart, WirePathGeometry


LIMITS = dict(
    maximum_height_m=0.001,
    minimum_radius_m=0.0014,
    maximum_segment_length_m=0.001,
    chord_tolerance_m=1e-5,
    maximum_tangent_angle_deg=2.0,
    maximum_candidates=8,
)


def _projector():
    return _SurfaceProjector(build_cylinder_mesh(0.25, 0.066, 28, 20, 1, 0, 0, 0))


def _ports(projector, *, seam=False):
    chart = _cylinder_chart(projector)
    width = chart.lengths[0]
    s = (-0.7 * width, 0.7 * width) if seam else (0.3 * width, 2.7 * width)
    developed = np.array([[s[0], -0.005, 0], [s[1], 0.005, 0]])
    source = chart.map(developed)
    sides = np.searchsorted(chart.starts, np.mod(developed[:, 0], chart.perimeter), side="right") - 1
    tangents = [chart.directions[side] + 0.1 * chart.axis for side in sides]
    return source, tangents


@pytest.mark.parametrize("seam", [False, True])
def test_intrinsic_join_crosses_facets_on_surface_and_keeps_ports(seam):
    projector = _projector()
    source, tangents = _ports(projector, seam=seam)
    snapshots = source.copy(), [value.copy() for value in tangents], projector.vertices.copy()
    result = generate_cylindrical_transition_curves(source, *tangents, projector, **{**LIMITS, "maximum_height_m": 0})
    assert result.candidates, (result.reason, result.rejected_reasons)
    for curve in result.candidates:
        assert curve.profile_family == "cylinder_surface"
        np.testing.assert_array_equal(curve.points[[0, -1]], source)
        assert np.all(curve.lift_distances_m == 0)
        # Midpoint checks detect the interior chord made by a naive Cartesian
        # join across more than one polygon side, not merely supported vertices.
        midpoint = 0.5 * (curve.points[:-1] + curve.points[1:])
        projected = projector.project(midpoint)[0]
        np.testing.assert_allclose(projected, midpoint, atol=1e-12)
        assert (
            np.max(np.linalg.norm(np.diff(curve.points, axis=0), axis=1)) <= LIMITS["maximum_segment_length_m"] * 1.001
        )
        assert np.all(np.diff(curve.parameters) > 0)
        assert np.all(np.diff(curve.source_fractions) > 0)
        for chord, tangent in (
            (curve.points[1] - curve.points[0], tangents[0]),
            (curve.points[-1] - curve.points[-2], tangents[1]),
        ):
            assert chord @ tangent / np.linalg.norm(chord) / np.linalg.norm(tangent) >= np.cos(np.radians(2.0))
    np.testing.assert_array_equal(source, snapshots[0])
    for value, snapshot in zip(tangents, snapshots[1]):
        np.testing.assert_array_equal(value, snapshot)
    np.testing.assert_array_equal(projector.vertices, snapshots[2])


def test_surface_mapping_inserts_every_edge_even_across_wrap_seam():
    chart = _cylinder_chart(_projector())
    span = chart.lengths[0]
    points = np.array([[-2.3 * span, 0.01, 0], [2.3 * span, -0.01, 0]])
    planar, parameters, fractions = _split_facets(chart, points, [0, 1], [0, 1], 100)
    assert len(planar) == 7
    mapped = chart.map(planar)
    # Each resulting segment lies in an actual polygon side; support creases
    # are retained as boundaries rather than hidden inside a chord.
    middle = (mapped[:-1] + mapped[1:]) / 2
    _cylinder_chart(_projector()).locate(middle)
    np.testing.assert_allclose(parameters, fractions)


def test_intrinsic_join_is_equivariant_under_rigid_transform():
    projector = _projector()
    source, tangents = _ports(projector)
    first = generate_cylindrical_transition_curves(source, *tangents, projector, **LIMITS)
    angle = 0.371
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) @ rotation
    shift = np.array([0.19, -0.42, 0.063])
    transformed = SimpleNamespace(vertices=projector.vertices @ rotation.T + shift, faces=projector.faces.copy())
    second = generate_cylindrical_transition_curves(
        source @ rotation.T + shift,
        *(value @ rotation.T for value in tangents),
        transformed,
        **LIMITS,
        b0_direction=rotation @ np.array([0, 0, 1]),
    )
    assert len(first.candidates) == len(second.candidates) > 0
    for expected, actual in zip(first.candidates, second.candidates):
        np.testing.assert_allclose(actual.points, expected.points @ rotation.T + shift, atol=1e-12)


def test_antipodal_ports_retain_the_monotone_tangent_aligned_half_turn_after_rigid_transform():
    projector = _projector()
    chart = _cylinder_chart(projector)
    first = 0.3 * chart.lengths[0]
    source = chart.map(np.array([[first, -0.005, 0], [first + chart.perimeter / 2, 0.005, 0]]))
    tangents = [chart.directions[0], chart.directions[len(chart.directions) // 2]]
    snapshots = source.copy(), [value.copy() for value in tangents], projector.vertices.copy()
    reservations = []
    result = generate_cylindrical_transition_curves(
        source,
        *tangents,
        projector,
        **{**LIMITS, "maximum_height_m": 0},
        shape_trial_budget=reservations.append,
    )
    assert reservations == [CYLINDER_TRANSITION_TRIALS, CYLINDER_TRANSITION_TRIALS]
    assert result.candidates, (result.reason, result.rejected_reasons)
    curve = result.candidates[0]
    developed, _sides = chart.locate(curve.points)
    circumferential = np.unwrap(2 * np.pi * developed[:, 0] / chart.perimeter) * chart.perimeter / (2 * np.pi)
    assert np.min(np.diff(circumferential)) >= -chart.tolerance
    assert circumferential[-1] - circumferential[0] == pytest.approx(chart.perimeter / 2, abs=chart.tolerance)
    np.testing.assert_array_equal(curve.points[[0, -1]], source)
    assert set(curve.segment_geometry) == {WirePathGeometry.SURFACE}

    rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [2, -1, 4], [3, 4, -1]]))
    shift = np.array([0.19, -0.42, 0.063])
    transformed = SimpleNamespace(vertices=projector.vertices @ rotation.T + shift, faces=projector.faces.copy())
    changed = generate_cylindrical_transition_curves(
        source @ rotation.T + shift,
        *(value @ rotation.T for value in tangents),
        transformed,
        **{**LIMITS, "maximum_height_m": 0},
        b0_direction=rotation @ np.array([0, 0, 1]),
    )
    assert len(changed.candidates) == len(result.candidates)
    for expected, actual in zip(result.candidates, changed.candidates):
        np.testing.assert_allclose(actual.points, expected.points @ rotation.T + shift, atol=1e-12)
    np.testing.assert_array_equal(source, snapshots[0])
    for value, snapshot in zip(tangents, snapshots[1]):
        np.testing.assert_array_equal(value, snapshot)
    np.testing.assert_array_equal(projector.vertices, snapshots[2])


def test_near_antipodal_extra_family_budget_is_reserved_before_generation(monkeypatch):
    import coilgen.joint_cylinder_curves as module

    projector = _projector()
    chart = _cylinder_chart(projector)
    developed = np.array(
        [[0.3 * chart.lengths[0], -0.005, 0], [0.3 * chart.lengths[0] + 0.45 * chart.perimeter, 0.005, 0]]
    )
    source = chart.map(developed)
    sides = np.searchsorted(chart.starts, np.mod(developed[:, 0], chart.perimeter), side="right") - 1
    tangents = [chart.directions[side] for side in sides]
    monkeypatch.setattr(
        module, "generate_joint_transition_curves", lambda *_args, **_kwargs: pytest.fail("unreserved winding")
    )
    reservations = []

    def reject_extra(count):
        reservations.append(count)
        raise RuntimeError("joint_shape_trial_budget_exceeded")

    with pytest.raises(RuntimeError, match="joint_shape_trial_budget_exceeded"):
        generate_cylindrical_transition_curves(
            source,
            *tangents,
            projector,
            **LIMITS,
            shape_trial_budget=reject_extra,
            shape_trials_reserved=True,
        )
    assert reservations == [CYLINDER_TRANSITION_TRIALS]


def test_unsupported_height_tangent_and_missing_support_are_rejected():
    projector = _projector()
    source, tangents = _ports(projector)
    chart = _cylinder_chart(projector)
    normal = np.cross(chart.directions[0], chart.axis)
    raised = source + 0.0001 * normal
    assert not generate_cylindrical_transition_curves(raised, *tangents, projector, **LIMITS).candidates
    off_tangent = tangents[0] + normal
    invalid = generate_cylindrical_transition_curves(source, off_tangent, tangents[1], projector, **LIMITS)
    assert invalid.reason == "cylinder_port_tangent_not_supported"
    missing = SimpleNamespace(vertices=projector.vertices, faces=projector.faces[:-1])
    assert not generate_cylindrical_transition_curves(source, *tangents, missing, **LIMITS).candidates


def test_sample_work_exhaustion_is_explicit():
    projector = _projector()
    source, tangents = _ports(projector)
    result = generate_cylindrical_transition_curves(source, *tangents, projector, **LIMITS, maximum_samples=2)
    assert not result.candidates
    assert result.reason == "cylinder_curve_sample_budget_exceeded"


@pytest.mark.parametrize("lift_sign", [-1.0, 1.0])
def test_chart_arches_obey_lift_support_and_provenance(lift_sign):
    projector = _projector()
    source, tangents = _ports(projector, seam=True)
    reservations = []
    result = generate_cylindrical_transition_curves(
        source, *tangents, projector, **LIMITS, lift_sign=lift_sign, shape_trial_budget=reservations.append
    )
    assert reservations == [CYLINDER_TRANSITION_TRIALS]
    raised = [curve for curve in result.candidates if curve.height_m > 0]
    nominal = [curve for curve in result.candidates if curve.height_m == 0]
    assert nominal and raised, (result.reason, result.rejected_reasons)
    for curve in raised:
        np.testing.assert_array_equal(curve.points[[0, -1]], source)
        assert curve.profile_family == "cylinder_lifted_transition"
        assert len(curve.segment_geometry) == len(curve.points) - 1
        assert WirePathGeometry.FREE_3D in curve.segment_geometry
        assert np.all(lift_sign * curve.lift_distances_m >= 0)
        assert np.max(np.abs(curve.lift_distances_m)) <= LIMITS["maximum_height_m"] + 1e-15
        projected = projector.project(curve.points)[0]
        assert np.max(np.linalg.norm(curve.points - projected, axis=1)) <= LIMITS["maximum_height_m"] + 1e-12
        assert (
            np.max(np.linalg.norm(np.diff(curve.points, axis=0), axis=1)) <= LIMITS["maximum_segment_length_m"] * 1.001
        )
    for curve in nominal:
        assert set(curve.segment_geometry) == {WirePathGeometry.SURFACE}


def test_shape_work_failure_happens_before_curve_generation(monkeypatch):
    import coilgen.joint_cylinder_curves as module

    projector = _projector()
    source, tangents = _ports(projector)
    monkeypatch.setattr(
        module, "generate_joint_transition_curves", lambda *args, **kwargs: pytest.fail("unreserved generation")
    )

    def reject_budget(count):
        assert count == CYLINDER_TRANSITION_TRIALS
        raise RuntimeError("joint_shape_trial_budget_exceeded")

    with pytest.raises(RuntimeError, match="joint_shape_trial_budget_exceeded"):
        generate_cylindrical_transition_curves(source, *tangents, projector, **LIMITS, shape_trial_budget=reject_budget)


@pytest.mark.parametrize("span", [0.05, 0.20])
@pytest.mark.parametrize("search_level", [0, 1, None])
def test_long_axial_return_keeps_compact_terminal_bends_in_the_opening(span, search_level):
    projector = _projector()
    chart = _cylinder_chart(projector)
    circumferential = chart.lengths[0] / 2
    source = chart.map(np.array([[circumferential, -span / 2, 0], [circumferential, span / 2, 0]]))
    incoming, outgoing = -chart.directions[0], chart.directions[0]
    result = generate_cylindrical_transition_curves(
        source,
        incoming,
        outgoing,
        projector,
        **{**LIMITS, "maximum_height_m": 0},
        clearance_score_cap_m=0.005,
        search_level=search_level,
    )
    assert result.candidates, (result.reason, result.rejected_reasons)
    compact = result.candidates[0]
    developed, _sides = chart.locate(compact.points)
    displacement = (developed[:, 0] - circumferential + chart.perimeter / 2) % chart.perimeter - chart.perimeter / 2
    # Increasing the group separation must lengthen the straight axial span,
    # rather than enlarging the end handles until they hit the retained turns.
    assert np.max(np.abs(displacement)) < 2 * LIMITS["minimum_radius_m"]
    straight = np.abs(np.diff(displacement)) < 1e-12
    assert np.sum(np.abs(np.diff(developed[:, 1]))[straight]) > span - 4 * LIMITS["minimum_radius_m"]
    np.testing.assert_array_equal(compact.points[[0, -1]], source)
    assert set(compact.segment_geometry) == {WirePathGeometry.SURFACE}
    assert compact.minimum_radius_m >= LIMITS["minimum_radius_m"]


@pytest.mark.parametrize("search_level", [0, 1, None])
def test_compact_return_can_lift_the_shared_axial_span_without_lifting_terminal_bends(search_level):
    projector = _projector()
    chart = _cylinder_chart(projector)
    circumferential = chart.lengths[0] / 2
    source = chart.map(np.array([[circumferential, -0.1, 0], [circumferential, 0.1, 0]]))
    reservations = []
    result = generate_cylindrical_transition_curves(
        source,
        -chart.directions[0],
        chart.directions[0],
        projector,
        **{**LIMITS, "maximum_height_m": 0.006, "maximum_candidates": 64},
        clearance_score_cap_m=0.005,
        shape_trial_budget=reservations.append,
        search_level=search_level,
    )
    assert reservations[0] == cylinder_trial_count(search_level)
    assert all(value == 1 for value in reservations[1:])
    lifted = [curve for curve in result.candidates if curve.height_m == 0.006]
    assert lifted
    curve = lifted[0]
    np.testing.assert_array_equal(curve.points[[0, -1]], source)
    assert np.max(np.abs(curve.lift_distances_m)) <= 0.006 + 1e-15
    assert WirePathGeometry.SURFACE in curve.segment_geometry
    assert WirePathGeometry.FREE_3D in curve.segment_geometry
    # Nominal terminal arcs remain attached; the return's middle clears the
    # spiral joins as one connected excursion with bounded radial lift.
    surface = np.r_[[value == WirePathGeometry.SURFACE for value in curve.segment_geometry], False]
    assert np.count_nonzero(surface) > 4
    # Mutable nominal arcs smoothly realize the analytic support; their radial
    # correction is bounded by polygon sag and belongs to the same allowance.
    assert np.max(curve.lift_distances_m[surface]) < 0.0005
    assert np.count_nonzero(np.isclose(curve.lift_distances_m, 0.006, rtol=0, atol=1e-15)) > 4


@pytest.mark.parametrize("search_level", [0, 1, None])
def test_real_gy_antipodal_outer_ports_admit_a_lifted_detour_with_compact_directed_bends(search_level):
    # Exact oriented ports from the standard GY outer-loop connection that
    # rejected every single span-scaled cubic. No saved output is required.
    source = np.array(
        [
            [0.001885100603072779, 0.06578760017304246, -0.044555687000676115],
            [-0.0007535254393025267, -0.06591509807346352, -0.0444872972624943],
        ]
    )
    tangents = [
        np.array([-0.001762613602847221, 0.00019859885654245624, 0.00010652821542388169]),
        np.array([-0.0028941887666174733, 0.0003260967569635176, -0.000174917953605698]),
    ]
    projector = _projector()
    original = source.copy(), [value.copy() for value in tangents]
    reservations = []
    result = generate_cylindrical_transition_curves(
        source,
        *tangents,
        projector,
        **{**LIMITS, "maximum_height_m": 0.006, "maximum_candidates": 64},
        clearance_score_cap_m=0.005,
        shape_trial_budget=reservations.append,
        search_level=search_level,
    )
    assert result.candidates, (result.reason, result.rejected_reasons)
    assert reservations[:2] == [cylinder_trial_count(search_level)] * 2
    assert all(count == 1 for count in reservations[2:])
    assert sum(reservations[2:]) <= 200
    lifted = [
        value
        for value in result.candidates
        if value.profile_family == "cylinder_detour_lifted_return" and value.height_m == 0.006
    ]
    assert lifted
    # Both perimeter classes must reach complete-circuit shape/field ranking.
    assert {np.sign(value.control_points_m[-1, 0] - value.control_points_m[0, 0]) for value in lifted} == {-1, 1}
    curve = next(value for value in lifted if np.min(value.points[:, 2]) < source[:, 2].min() - 0.009)
    segments = np.stack((curve.points[:-1], curve.points[1:]), axis=1)
    normal = projector.project_with_face_normals(source[:1])[2][0]
    assert not _validate_support(projector, source[0], segments, normal, 0.006, 1e-8, {})
    np.testing.assert_array_equal(curve.points[[0, -1]], source)
    assert curve.minimum_radius_m >= LIMITS["minimum_radius_m"]
    assert WirePathGeometry.SURFACE in curve.segment_geometry and WirePathGeometry.FREE_3D in curve.segment_geometry
    assert np.max(curve.lift_distances_m) <= 0.006 + 1e-12
    # The smooth radial plateau reserves polygon sag in the same 6 mm
    # nominal-support allowance. Its truthful lift varies with facet position.
    chart = _cylinder_chart(projector)
    axial = (curve.points - chart.origin) @ chart.axis
    radial = curve.points - chart.origin - axial[:, None] * chart.axis
    radii = np.linalg.norm(radial, axis=1)
    assert np.count_nonzero(np.isclose(radii, radii.max(), rtol=0, atol=1e-12)) > 10
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.25, 0.066, 28, 20, 1, 0, 0, 0),
        cross_sectional_points=tuple(
            (0.0007 * np.cos(angle), 0.0007 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ),
        sweep_min_bend_radius=0.0014,
        sweep_max_segment_length=0.001,
        sweep_max_chord_error=1e-5,
    )
    _prepared, _geometry, physical = _prepare_sweep_centerline(
        CoilPart(mesh=build_cylinder_mesh(0.25, 0.066, 28, 20, 1, 0, 0, 0)),
        curve.points,
        list(curve.segment_geometry),
        conductor_cross_section_points(config),
        config,
        closed=False,
    )
    assert physical["sweep_min_resampled_bend_radius_m"] >= 0.0014
    assert physical["sweep_max_curvature_envelope_ratio"] <= 0.5
    assert np.max(np.linalg.norm(np.diff(curve.points, axis=0), axis=1)) <= LIMITS["maximum_segment_length_m"] * 1.001
    for chord, tangent in (
        (curve.points[1] - curve.points[0], tangents[0]),
        (curve.points[-1] - curve.points[-2], tangents[1]),
    ):
        assert chord @ tangent >= np.cos(np.radians(2)) * np.linalg.norm(chord) * np.linalg.norm(tangent)
    np.testing.assert_array_equal(source, original[0])
    for value, saved in zip(tangents, original[1]):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize("coincident", [False, True])
def test_exact_real_gz_cubic_facet_crossings_preserve_physical_bends_and_ports(coincident):
    if coincident:
        source = np.array(
            [
                [0.06643663481912568, -0.005000004272521309, 0.07670242822738253],
                [0.06643663480319616, 0.0050000044136658305, 0.08353568104782892],
            ]
        )
        tangents = [
            np.array([0.00018092907022568272, 0.0016057899118786913, -8.712746174671659e-7]),
            np.array([-0.001087249090596154, 0.009649602571434171, 4.699869771071508e-6]),
        ]
        controls = np.array(
            [
                [0.3100363935722541, 0.07670242822738253, 0],
                [0.3130773946015434, 0.07670078860616336, 0],
                [0.31705867696406803, 0.08353420923154746, 0],
                [0.32009967807920314, 0.08353568104782892, 0],
            ]
        )
    else:
        source = np.array(
            [
                [0.06643663527227336, 0.005000000250453307, -0.030884192842823236],
                [0.06643663510474998, -0.005000001737299392, -0.03826509482939001],
            ]
        )
        tangents = [
            np.array([0.00043566820557336405, -0.003866662270056694, 5.362891576765616e-6]),
            np.array([-0.0007554437363499755, -0.006704748602600608, -7.707132109992243e-6]),
        ]
        controls = np.array(
            [
                [0.32009967388964805, -0.030884192842823236, 0],
                [0.30886779729102937, -0.03086871267282928, 0],
                [0.3212682760621468, -0.038252264926075835, 0],
                [0.31003639612351486, -0.03826509482939001, 0],
            ]
        )
    mesh = build_cylinder_mesh(0.25, 0.067, 28, 20, 1, 0, 0, 0)
    projector = _SurfaceProjector(mesh)
    snapshot = source.copy(), [value.copy() for value in tangents]
    generation = generate_cylindrical_transition_curves(
        source,
        *tangents,
        projector,
        **{**LIMITS, "maximum_height_m": 0.006, "maximum_candidates": 64},
        clearance_score_cap_m=0.0035,
    )
    curve = next(
        value
        for value in generation.candidates
        if value.height_m == 0
        and value.profile_family == "cylinder_surface"
        and np.allclose(value.control_points_m, controls, rtol=0, atol=1e-14)
    )
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.25, 0.067, 28, 20, 1, 0, 0, 0),
        cross_sectional_points=tuple(
            (0.0007 * np.cos(angle), 0.0007 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ),
        sweep_min_bend_radius=0.0014,
        sweep_max_segment_length=0.001,
        sweep_max_chord_error=1e-5,
    )
    prepared, _geometry, physical = _prepare_sweep_centerline(
        CoilPart(mesh=mesh),
        curve.points,
        list(curve.segment_geometry),
        conductor_cross_section_points(config),
        config,
        closed=False,
    )
    assert physical["sweep_min_resampled_bend_radius_m"] >= 0.0014
    assert physical["sweep_max_curvature_envelope_ratio"] <= 0.5
    if coincident:
        assert np.linalg.norm(np.diff(prepared, axis=0), axis=1).min() > 1e-6
    np.testing.assert_array_equal(curve.points[[0, -1]], snapshot[0])
    np.testing.assert_array_equal(source, snapshot[0])
    for actual, prior in zip(tangents, snapshot[1]):
        np.testing.assert_array_equal(actual, prior)


def test_real_gz_raised_compact_return_preserves_physical_surface_port_attachments():
    source = np.array(
        [
            [0.06643663474721395, -0.005000004910715733, 0.10303006723865013],
            [0.06643663482307684, -0.005000004237434161, -0.10304070777974242],
        ]
    )
    tangents = [
        np.array([0.0005362536969139575, 0.004759383197134268, -9.014973498672196e-7]),
        np.array([-1.5460766176839114e-5, -0.00013721809505583965, -7.62282575811346e-8]),
    ]
    mesh = build_cylinder_mesh(0.25, 0.067, 28, 20, 1, 0, 0, 0)
    projector = _SurfaceProjector(mesh)
    generation = generate_cylindrical_transition_curves(
        source,
        *tangents,
        projector,
        **{**LIMITS, "maximum_height_m": 0.006, "maximum_candidates": 64},
        clearance_score_cap_m=0.0035,
    )
    curve = next(
        value
        for value in generation.candidates
        if value.profile_family == "cylinder_compact_lifted_return"
        and value.height_m == 0.006
        and value.minimum_radius_m < 0.0015
    )
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.25, 0.067, 28, 20, 1, 0, 0, 0),
        cross_sectional_points=tuple(
            (0.0007 * np.cos(angle), 0.0007 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ),
        sweep_min_bend_radius=0.0014,
        sweep_max_segment_length=0.001,
        sweep_max_chord_error=1e-5,
    )
    # SURFACE neighbors reproduce the retained-contour attachment: preparation
    # retracts the exact original ports while the raised middle remains FREE.
    prefix = source[0] - 0.002 * tangents[0] / np.linalg.norm(tangents[0])
    suffix = source[-1] + 0.002 * tangents[1] / np.linalg.norm(tangents[1])
    points = np.vstack((prefix, curve.points, suffix))
    _prepared, _geometry, physical = _prepare_sweep_centerline(
        CoilPart(mesh=mesh),
        points,
        [WirePathGeometry.SURFACE, *curve.segment_geometry, WirePathGeometry.SURFACE],
        conductor_cross_section_points(config),
        config,
        closed=False,
    )
    assert physical["sweep_min_resampled_bend_radius_m"] >= 0.0014
    assert physical["sweep_max_curvature_envelope_ratio"] <= 0.5
    assert np.max(curve.lift_distances_m) <= 0.006 + 1e-12
    np.testing.assert_array_equal(curve.points[[0, -1]], source)
