"""Actual frustum developments preserve metric, ports and triangle support."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.developed_frustum import frustum_chart
from coilgen.geometry import build_cylinder_mesh
from coilgen.joint_cylinder_curves import _cylinder_chart, _developable_chart, generate_cylindrical_transition_curves
from coilgen.manufacturing import _SurfaceProjector
from coilgen.models import Mesh, WirePathGeometry


LIMITS = dict(
    maximum_height_m=0.002,
    minimum_radius_m=0.0014,
    maximum_segment_length_m=0.001,
    chord_tolerance_m=1e-5,
    maximum_tangent_angle_deg=2.0,
    maximum_candidates=8,
)


def _projector(*, subdivisions=8, taper=0.04):
    mesh = build_cylinder_mesh(0.3, 0.18, 32, subdivisions, 1, 0, 0, 0)
    mesh.vertices = mesh.vertices.copy()
    axial = mesh.vertices[:, 2]
    radius = 0.18 - taper * (axial + 0.15) / 0.3
    mesh.vertices[:, :2] *= (radius / 0.18)[:, None]
    return _SurfaceProjector(mesh)


def _ports(chart, *, seam=False):
    first_side, last_side = (-1, 0) if seam else (1, 3)
    first_angle = chart.starts[first_side] + 0.3 * chart.angles[first_side]
    last_angle = chart.starts[last_side] + 0.7 * chart.angles[last_side]
    # Locate ports using true physical planes at two axial levels.
    planar = np.array([[np.cos(first_angle), np.sin(first_angle), 0.0], [np.cos(last_angle), np.sin(last_angle), 0.0]])
    sides = np.array([first_side % len(chart.starts), last_side])
    points = chart.apex + np.einsum("nij,nj->ni", chart.jacobians[sides], planar[:, :2])
    axial = (points - chart.origin) @ chart.axis
    desired = np.array([0.11, 0.12])
    apex_height = float((chart.apex - chart.origin) @ chart.axis)
    points = chart.apex + (points - chart.apex) * ((desired - apex_height) / (axial - apex_height))[:, None]
    developed, _ = chart.locate(points)
    tangent = []
    for side, point in zip(sides, developed):
        around = np.array([-point[1], point[0]])
        around /= np.linalg.norm(around)
        outward = point[:2] / np.linalg.norm(point[:2])
        tangent.append(chart.jacobians[side] @ (around + 0.15 * outward))
    return points, tangent


@pytest.mark.parametrize("seam", [False, True])
@pytest.mark.parametrize("taper", [0.04, -0.04])
def test_intrinsic_frustum_curves_keep_directed_ports_and_actual_segment_support(seam, taper):
    projector = _projector(taper=taper)
    chart = _developable_chart(projector)
    source, tangents = _ports(chart, seam=seam)
    vertices = projector.vertices.copy()
    result = generate_cylindrical_transition_curves(source, *tangents, projector, **LIMITS)
    assert result.candidates, (result.reason, result.rejected_reasons)
    for curve in result.candidates:
        assert curve.profile_family == "frustum_surface"
        assert curve.minimum_radius_scope == "smooth_metric_development_screen_only"
        assert set(curve.segment_geometry) == {WirePathGeometry.SURFACE}
        np.testing.assert_array_equal(curve.points[[0, -1]], source)
        np.testing.assert_array_equal(curve.lift_distances_m, np.zeros(len(curve.points)))
        assert curve.minimum_radius_m >= LIMITS["minimum_radius_m"] * 1.005
        assert np.all(np.diff(curve.parameters) > 0)
        assert np.all(np.diff(curve.source_fractions) > 0)
        midpoint = (curve.points[:-1] + curve.points[1:]) / 2
        np.testing.assert_allclose(projector.project(midpoint)[0], midpoint, atol=2e-11)
        assert np.max(np.linalg.norm(np.diff(curve.points, axis=0), axis=1)) <= 0.001001
        for chord, tangent in (
            (curve.points[1] - curve.points[0], tangents[0]),
            (curve.points[-1] - curve.points[-2], tangents[1]),
        ):
            assert chord @ tangent >= np.cos(np.radians(2)) * np.linalg.norm(chord) * np.linalg.norm(tangent)
    np.testing.assert_array_equal(projector.vertices, vertices)


def test_development_is_isometric_with_taper_not_theta_z_scaled():
    chart = frustum_chart(_projector())
    for jacobian in chart.jacobians:
        np.testing.assert_allclose(jacobian.T @ jacobian, np.eye(2), atol=1e-13)
    source, tangents = _ports(chart)
    side, first, developed_tangent = chart.port(source[0], tangents[0], start=True)
    np.testing.assert_allclose(chart.jacobians[side] @ developed_tangent, tangents[0], atol=1e-13)
    target = first + developed_tangent * 1e-4
    mapped, _ = chart.map(np.array([np.r_[first, 0], np.r_[target, 0]]), np.arctan2(first[1], first[0]))
    assert np.linalg.norm(mapped[1] - mapped[0]) == pytest.approx(np.linalg.norm(target - first), abs=1e-13)
    assert chart.period < 2 * np.pi


def test_frustum_curves_are_equivariant_and_ignore_vertex_normal_metadata():
    projector = _projector()
    chart = _developable_chart(projector)
    source, tangents = _ports(chart, seam=True)
    reference = generate_cylindrical_transition_curves(source, *tangents, projector, **LIMITS)
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [2, -1, 4], [3, 4, -1]]))
    translation = np.array([0.19, -0.42, 0.063])
    changed = SimpleNamespace(
        vertices=projector.vertices @ rotation.T + translation,
        faces=projector.faces.copy(),
        vertex_normals=-projector.vertex_normals @ rotation.T,
    )
    result = generate_cylindrical_transition_curves(
        source @ rotation.T + translation,
        *(tangent @ rotation.T for tangent in tangents),
        changed,
        **LIMITS,
        b0_direction=rotation @ np.array([0, 0, 1]),
    )
    assert len(result.candidates) == len(reference.candidates) > 0
    for first, second in zip(reference.candidates, result.candidates):
        np.testing.assert_allclose(second.points, first.points @ rotation.T + translation, atol=3e-11)


def test_coherent_reversed_faces_have_same_geometric_chart():
    projector = _projector()
    first = frustum_chart(projector)
    second = frustum_chart(SimpleNamespace(vertices=projector.vertices, faces=projector.faces[:, [0, 2, 1]]))
    np.testing.assert_allclose(first.apex, second.apex, atol=1e-13)
    np.testing.assert_allclose(first.jacobians, second.jacobians, atol=1e-13)


@pytest.mark.parametrize("fault", ["missing_face", "duplicate_face", "twisted_rim", "nonfinite", "degenerate_face"])
def test_incomplete_nonhomothetic_or_invalid_mesh_is_rejected(fault):
    projector = _projector()
    vertices, faces = projector.vertices.copy(), projector.faces.copy()
    if fault == "missing_face":
        faces = faces[:-1]
    elif fault == "duplicate_face":
        faces = np.vstack((faces, faces[0]))
    elif fault == "twisted_rim":
        selected = vertices[:, 2] > vertices[:, 2].max() - 0.005
        angle = 0.03
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        vertices[selected, :2] = vertices[selected, :2] @ rotation.T
    elif fault == "nonfinite":
        vertices[0, 0] = np.nan
    else:
        faces[0, 1] = faces[0, 0]
    with pytest.raises(ValueError):
        frustum_chart(SimpleNamespace(vertices=vertices, faces=faces))


def test_near_cylinder_and_work_exhaustion_reject_without_approximation():
    with pytest.raises(ValueError, match="ill_conditioned"):
        frustum_chart(_projector(taper=1e-7))
    with pytest.raises(ValueError, match="budget_exceeded"):
        frustum_chart(_projector(), maximum_work=10)
    with pytest.raises(ValueError, match="budget"):
        frustum_chart(_projector(), maximum_sides=8)


def test_uniform_cylinder_dispatch_retains_existing_chart():
    projector = _SurfaceProjector(build_cylinder_mesh(0.3, 0.18, 32, 8, 1, 0, 0, 0))
    expected, actual = _cylinder_chart(projector), _developable_chart(projector)
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual.polygon, expected.polygon)
    np.testing.assert_array_equal(actual.starts, expected.starts)


def test_lifted_ports_off_surface_tangents_rims_and_samples_are_not_silently_repaired():
    projector = _projector()
    chart = _developable_chart(projector)
    source, tangents = _ports(chart)
    side, _point, _tangent = chart.port(source[0], tangents[0], start=True)
    raised = source + 0.0001 * chart.normals[side]
    assert not generate_cylindrical_transition_curves(raised, *tangents, projector, **LIMITS).candidates
    off_tangent = tangents[0] + chart.normals[side]
    result = generate_cylindrical_transition_curves(source, off_tangent, tangents[1], projector, **LIMITS)
    assert result.reason == "frustum_port_tangent_not_supported"
    result = generate_cylindrical_transition_curves(source, *tangents, projector, **LIMITS, maximum_samples=2)
    assert "budget_exceeded" in result.reason
    developed, _ = chart.locate(source)
    developed[:, 2] = 0.001
    with pytest.raises(ValueError, match="height_requires_spatial"):
        chart.map(developed, 0.0)


def test_subdivision_and_vertex_face_permutation_preserve_actual_support_geometry():
    projector = _projector()
    source, tangents = _ports(frustum_chart(projector))
    rng = np.random.default_rng(412)
    permutation = rng.permutation(len(projector.vertices))
    inverse = np.argsort(permutation)
    faces = inverse[projector.faces][rng.permutation(len(projector.faces))]
    changed = SimpleNamespace(vertices=projector.vertices[permutation], faces=faces)
    chart = frustum_chart(changed)
    developed, _ = chart.locate(source)
    angles = np.arctan2(developed[:, 1], developed[:, 0])
    np.testing.assert_allclose(chart.map(developed, angles.mean())[0], source, atol=2e-11)
    result = generate_cylindrical_transition_curves(source, *tangents, changed, **LIMITS)
    assert result.candidates, (result.reason, result.rejected_reasons)
    for curve in result.candidates:
        midpoint = (curve.points[:-1] + curve.points[1:]) / 2
        np.testing.assert_allclose(projector.project(midpoint)[0], midpoint, atol=2e-11)


def test_triangle_interior_subdivision_retains_the_metric_development():
    projector = _projector()
    triangles = projector.vertices[projector.faces]
    centroids = triangles.mean(axis=1)
    centers = np.arange(len(centroids)) + len(projector.vertices)
    faces = np.vstack(
        [np.column_stack((projector.faces[:, i], projector.faces[:, (i + 1) % 3], centers)) for i in range(3)]
    )
    changed = _SurfaceProjector(Mesh(np.vstack((projector.vertices, centroids)), faces))
    first, second = frustum_chart(projector), frustum_chart(changed)
    np.testing.assert_allclose(first.jacobians, second.jacobians, atol=1e-13)
    np.testing.assert_allclose(first.starts, second.starts, atol=1e-13)
    source, tangents = _ports(first)
    result = generate_cylindrical_transition_curves(source, *tangents, changed, **LIMITS)
    assert result.candidates
    for curve in result.candidates:
        midpoint = (curve.points[:-1] + curve.points[1:]) / 2
        np.testing.assert_allclose(changed.project(midpoint)[0], midpoint, atol=2e-11)


def test_oblique_middle_trials_are_reserved_before_any_frustum_generation(monkeypatch):
    import coilgen.joint_frustum_curves as module

    projector = _projector()
    source, tangents = _ports(frustum_chart(projector))
    monkeypatch.setattr(
        module,
        "generate_joint_transition_curves",
        lambda *_args, **_kwargs: pytest.fail("generation exceeded its reserved trial budget"),
    )
    reservations = []

    def stop(count):
        reservations.append(count)
        raise RuntimeError("joint_shape_trial_budget_exceeded")

    with pytest.raises(RuntimeError, match="budget_exceeded"):
        generate_cylindrical_transition_curves(
            source, *tangents, projector, **LIMITS, shape_trial_budget=stop, shape_trials_reserved=True
        )
    assert reservations == [64]
