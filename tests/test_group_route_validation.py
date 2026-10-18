"""Directed coverage and actual manufacturing must both authorize a new route."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from coilgen import group_route_validation as validation
from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.models import CoilPart, InterGroupCut, Mesh, TargetField, WirePath, WireSegmentOwner


@pytest.fixture
def loop_case():
    angle = np.linspace(0.0, 2 * np.pi, 65)
    points = 0.025 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    points[-1] = points[0]
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    owner = WireSegmentOwner("contour_loop", group_index=0, loop_index=0)
    path = WirePath(points, segment_owners=[deepcopy(owner) for _ in range(64)])
    part = CoilPart(mesh=mesh, wire_path=path, contour_step=2.0)
    config = CoilGenConfig(
        skip_normal_shift=True,
        conductor_cross_section_width=0.0005,
        conductor_cross_section_height=0.0004,
        conductor_mounting_mode="surface_outer",
    )
    kwargs = dict(
        protected_segments={index: points[index : index + 2].copy() for index in range(64)},
        candidate_segment_ids=tuple(range(64)),
        group_sequence=(0,),
        required_groups=(0,),
        target_points_m=np.array([[0, 0, 0.01], [0.002, 0, 0.005]]),
        maximum_field_delta_t_per_a=1e-12,
    )
    return part, config, kwargs


def test_real_closed_sweep_accepts_and_returns_isolated_exact_geometry(loop_case):
    part, config, kwargs = loop_case
    original = deepcopy(part)
    result = validation.validate_group_route_candidate(part, part.wire_path, config, **kwargs)
    assert result.accepted, result.reasons
    prepared = result.prepared_part
    assert prepared.manufacturing_diagnostics["manufacturing_ok"]
    assert prepared.manufacturing_diagnostics["exact_collision_completed"]
    assert prepared.manufacturing_diagnostics["exact_intersection_count"] == 0
    np.testing.assert_array_equal(prepared.wire_path.v, part.wire_path.v)
    np.testing.assert_array_equal(
        prepared.fabrication_path.cross_section_m,
        [[-0.00025, -0.0002], [0.00025, -0.0002], [0.00025, 0.0002], [-0.00025, 0.0002]],
    )
    assert not np.array_equal(result.baseline.route_field_per_amp, result.baseline.physical_field_per_amp)
    assert result.diagnostics["physical_maximum_field_delta_t_per_a"] == 0
    np.testing.assert_array_equal(result.physical_field_per_amp, result.baseline.physical_field_per_amp)
    assert not result.physical_field_per_amp.flags.writeable
    assert part.swept_body is original.swept_body is None
    assert part.route_diagnostics == original.route_diagnostics == {}
    assert part.normal_shift_diagnostics == {}
    assert np.shares_memory(prepared.mesh.vertices, part.mesh.vertices)
    assert not prepared.mesh.vertices.flags.writeable
    assert part.mesh.vertices.flags.writeable
    prepared.wire_path.v[1, 2] = 1.0
    prepared.wire_path.segment_owners[0].group_index = 99
    np.testing.assert_array_equal(part.wire_path.v, original.wire_path.v)
    assert part.wire_path.segment_owners[0].group_index == 0
    assert config.skip_normal_shift and not config.skip_sweep


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("missing", "directed_contour_coverage_mismatch"),
        ("duplicate", "directed_contour_coverage_mismatch"),
        ("reverse", "protected_segment_coordinates_or_direction_changed"),
        ("perturb", "protected_segment_coordinates_or_direction_changed"),
        ("owners", "missing_explicit_segment_owners"),
        ("open", "candidate_not_finite_closed_cycle"),
        ("group", "group_cycle_coverage_mismatch"),
    ],
)
def test_provenance_rejects_before_any_preview(loop_case, monkeypatch, defect, reason):
    part, config, kwargs = loop_case
    path = deepcopy(part.wire_path)
    identifiers = list(kwargs["candidate_segment_ids"])
    if defect == "missing":
        identifiers[1] = None
    elif defect == "duplicate":
        identifiers[1] = identifiers[0]
    elif defect == "reverse":
        kwargs["protected_segments"][1] = kwargs["protected_segments"][1][::-1]
    elif defect == "perturb":
        path.v[1, 0] += 1e-12
    elif defect == "owners":
        path.segment_owners = []
    elif defect == "open":
        path.v[-1, 0] += 1e-12
    elif defect == "group":
        kwargs["group_sequence"] = (0, 0)
    kwargs["candidate_segment_ids"] = identifiers
    monkeypatch.setattr(validation, "shift_return_paths", lambda *_: pytest.fail("Invalid structure reached preview"))
    result = validation.validate_group_route_candidate(part, path, config, **kwargs)
    assert not result.accepted and reason in result.reasons
    assert result.prepared_part is None


def test_transition_cuts_must_form_the_declared_directed_group_cycle(loop_case):
    part, _config, kwargs = loop_case
    first = InterGroupCut(np.zeros((2, 3)), np.ones((2, 3)), group1_index=0, group2_index=1)
    second = deepcopy(first)
    args = (part.wire_path, kwargs["candidate_segment_ids"], kwargs["protected_segments"], (0, 1), (0, 1))
    assert "group_transition_cycle_mismatch" in validation._structural_reasons(*args, [first, second])
    second.group1_index, second.group2_index = 1, 0
    assert validation._structural_reasons(*args, [first, second]) == ()


def test_same_invalid_baseline_does_not_waive_candidate_minimum_radius(loop_case):
    part, config, kwargs = loop_case
    config.sweep_min_bend_radius = 0.05
    result = validation.validate_group_route_candidate(part, part.wire_path, config, **kwargs)
    assert not result.accepted
    assert "minimum_bend_radius_violation" in result.reasons
    assert "minimum_bend_radius_violation" in result.baseline.reasons
    assert result.prepared_part is None and part.swept_body is None


def test_same_current_field_delta_rejects_before_candidate_sweep(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    path = deepcopy(part.wire_path)
    path.v[3, 0] += 0.002
    for index in (2, 3):
        del kwargs["protected_segments"][index]
    kwargs["candidate_segment_ids"] = [i if i not in (2, 3) else None for i in range(64)]
    monkeypatch.setattr(validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Field failure reached sweep"))
    result = validation.validate_group_route_candidate(part, path, config, baseline=baseline, **kwargs)
    assert not result.accepted and "field_delta_budget_exceeded" in result.reasons
    assert result.diagnostics["route_maximum_field_delta_t_per_a"] > kwargs["maximum_field_delta_t_per_a"]
    assert result.diagnostics["body_preview_started"] is False


@pytest.mark.parametrize("mutation", ["route", "support", "config", "roi"])
def test_baseline_cache_rejects_same_size_mutations(loop_case, mutation):
    part, config, kwargs = loop_case
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    candidate = deepcopy(part.wire_path)
    if mutation == "route":
        part.wire_path.v[3, 2] += 1e-4
    elif mutation == "support":
        part.mesh.vertices[0, 0] += 1e-4
    elif mutation == "config":
        config.conductor_cross_section_height *= 1.01
    else:
        kwargs["target_points_m"][0, 2] += 1e-4
    result = validation.validate_group_route_candidate(part, candidate, config, baseline=baseline, **kwargs)
    assert not result.accepted and result.reasons == ("stale_baseline",)


def test_reused_baseline_sweeps_only_the_candidate(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    original = validation.create_swept_coil_meshes
    calls = []

    def counted(parts, cfg):
        calls.append(parts[0])
        return original(parts, cfg)

    monkeypatch.setattr(validation, "create_swept_coil_meshes", counted)
    result = validation.validate_group_route_candidate(part, part.wire_path, config, baseline=baseline, **kwargs)
    assert result.accepted and len(calls) == 1


def test_admission_and_exact_pair_budget_fail_closed(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    limits = validation.GroupRouteValidationLimits(maximum_route_points=10)
    monkeypatch.setattr(validation, "_isolated_part", lambda *_: pytest.fail("Oversized route copied"))
    result = validation.validate_group_route_candidate(part, part.wire_path, config, limits=limits, **kwargs)
    assert result.reasons == ("route_point_budget_exceeded",)
    mesh = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.1]]), np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]]))
    report, reason = validation._exact_collision_audit(mesh, replace(limits, maximum_triangle_pairs=1))
    assert reason == "triangle_pair_budget_exceeded" and not report["exact_collision_completed"]


def test_exact_triangle_audit_detects_finite_body_intersection():
    mesh = Mesh(
        np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, -0.5, -1], [0, -0.5, 1], [0, 0.5, 0]]),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    report, reason = validation._exact_collision_audit(mesh, validation.GroupRouteValidationLimits())
    assert reason == "swept_mesh_self_intersection"
    assert report["exact_collision_completed"] and report["exact_intersection_count"] == 1


@pytest.mark.parametrize("overlapping", [False, True])
def test_streamed_triangle_pairs_equal_exhaustive_aabb_pairs(overlapping):
    random = np.random.default_rng(171)
    if overlapping:
        minima = -random.uniform(0.1, 1.0, (129, 3))
        maxima = random.uniform(0.1, 1.0, (129, 3))
    else:
        minima = random.uniform(-1, 1, (129, 3))
        maxima = minima + random.uniform(0.01, 0.4, (129, 3))
    first, second = np.triu_indices(len(minima), 1)
    separation = np.maximum(0.0, np.maximum(minima[first] - maxima[second], minima[second] - maxima[first]))
    near = np.einsum("ij,ij->i", separation, separation) <= 1e-24
    expected = np.column_stack((first[near], second[near]))
    diagnostics = {}
    batches = list(
        validation._triangle_pair_batches(
            minima,
            maxima,
            1e-12,
            validation.GroupRouteValidationLimits(),
            diagnostics,
        )
    )
    actual = np.concatenate(batches) if batches else np.empty((0, 2), dtype=int)
    assert all(len(batch) <= 4096 for batch in batches)
    assert len(actual) == len(expected) == diagnostics["exact_collision_near_pair_count"]
    order = np.lexsort((actual[:, 1], actual[:, 0]))
    np.testing.assert_array_equal(actual[order], expected)
    assert diagnostics["exact_collision_broadphase_pair_count"] >= len(actual)
    if overlapping:
        assert len(batches) > 1


def test_streamed_audit_broadphase_budget_exhaustion_retains_explicit_failure():
    mesh = Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.1]]),
        np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]]),
    )
    limits = validation.GroupRouteValidationLimits(maximum_triangle_broadphase_pairs=1)
    report, reason = validation._exact_collision_audit(mesh, limits)
    assert reason == "triangle_broadphase_pair_budget_exceeded"
    assert not report["exact_collision_completed"]
    assert report["exact_collision_streamed"]
    assert report["exact_collision_broadphase_pair_count"] > report["exact_collision_broadphase_pair_limit"]
    assert report["exact_collision_near_pair_count"] == 0


@pytest.mark.parametrize(
    "extra,second_face,expected",
    [
        ([[0.5, 0.5, -1], [0.5, 0.5, 1]], [0, 3, 4], True),
        ([[-0.5, -0.5, -1], [-0.5, -0.5, 1]], [0, 3, 4], False),
        ([[0.5, 0.2, 0], [0.2, 0.5, 0]], [0, 3, 4], True),
        ([[-1, 0, 0], [0, -1, 0]], [0, 3, 4], False),
        ([[0.5, 1, 0]], [0, 1, 3], True),
        ([[0, -1, 0]], [0, 1, 3], False),
        ([[0, 0, 1]], [0, 1, 3], False),
        ([], [2, 1, 0], True),
    ],
)
def test_incident_triangles_may_share_only_their_declared_vertex_or_edge(extra, second_face, expected):
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], *extra], dtype=float)
    # Non-axis-aligned millimetre geometry exercises plane and ray predicates.
    from coilgen.geometry import rotation_matrix_by_vector

    rotation = rotation_matrix_by_vector(np.array([1.0, 2.0, 3.0]), 0.61)
    mesh = Mesh(0.001 * vertices @ rotation.T + [0.08, -0.02, 0.05], np.array([[0, 1, 2], second_face]))
    report, reason = validation._exact_collision_audit(mesh, validation.GroupRouteValidationLimits())
    assert report["exact_collision_completed"]
    assert report["exact_incident_pair_count"] == 1
    assert report["exact_incident_overlap_count"] == int(expected)
    assert bool(reason) == expected


@pytest.mark.parametrize("change", ["open", "nonfinite", "zero_length"])
def test_prepared_cycle_is_rechecked_after_cleanup_before_fields_or_sweep(loop_case, monkeypatch, change):
    part, config, kwargs = loop_case

    def broken_cleanup(parts, **_options):
        points = parts[0].wire_path.v
        if change == "open":
            points[-1, 2] += 0.001
        elif change == "nonfinite":
            points[2, 1] = np.nan
        else:
            points[2] = points[1]

    monkeypatch.setattr(validation, "cleanup_wire_paths", broken_cleanup)
    monkeypatch.setattr(validation, "_field", lambda *_: pytest.fail("Invalid prepared cycle reached field evaluation"))
    result = validation.validate_group_route_candidate(part, part.wire_path, config, apply_normal_shift=False, **kwargs)
    expected = "prepared_zero_length_segment" if change == "zero_length" else "prepared_not_finite_closed_cycle"
    assert result.reasons == (expected,)
    assert result.prepared_part is None
    assert np.array_equal(part.wire_path.v[0], part.wire_path.v[-1])
    assert np.all(np.isfinite(part.wire_path.v))


def test_final_physical_trace_must_retain_the_electrical_cycle(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    original = validation._sweep_preview

    def open_physical_trace(preview, cfg, limits, progress_callback=None):
        report, reasons = original(preview, cfg, limits, progress_callback)
        trace = preview.fabrication_path
        preview.fabrication_path = replace(trace, closed=False, segment_owners=(), segment_geometry=())
        return report, reasons

    monkeypatch.setattr(validation, "_sweep_preview", open_physical_trace)
    result = validation.validate_group_route_candidate(part, part.wire_path, config, baseline=baseline, **kwargs)
    assert result.reasons == ("prepared_physical_path_not_closed",)
    assert result.prepared_part is None


def test_configured_target_limits_require_context_and_use_total_same_current_field(loop_case):
    part, config, kwargs = loop_case
    config.layout_target_error_rms_limit_T = 1e-10
    rejected = validation.validate_group_route_candidate(part, part.wire_path, config, **kwargs)
    assert rejected.reasons == ("missing_target_field_context",)
    config.layout_target_error_rms_limit_T = 1e-5
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    other = np.full_like(kwargs["target_points_m"], 2e-6)
    target = TargetField(
        kwargs["target_points_m"],
        (baseline.physical_field_per_amp + other) * part.contour_step,
        np.ones(2),
        np.zeros(2, dtype=bool),
        np.ones(2, dtype=int),
    )
    result = validation.validate_group_route_candidate(
        part, part.wire_path, config, baseline=baseline, target_field=target, other_field_per_amp=other, **kwargs
    )
    assert result.accepted, result.reasons
    assert result.diagnostics["physical_candidate_dsv_residual_rms_T"] == 0
    assert result.diagnostics["physical_current_a"] == 2.0
    assert result.diagnostics["physical_current_refitted"] is False


@pytest.mark.parametrize(
    "name",
    [
        "linearity_rms_tolerance_percent",
        "linearity_peak_tolerance_percent",
        "linearity_cross_gradient_tolerance_percent",
        "gradient_strength_tolerance_percent",
    ],
)
def test_configured_gradient_constraints_require_target_context_before_preview(loop_case, monkeypatch, name):
    part, config, kwargs = loop_case
    setattr(config, name, 1.0)
    monkeypatch.setattr(
        validation, "_isolated_part", lambda *_: pytest.fail("Missing gradient context reached preview")
    )
    result = validation.validate_group_route_candidate(part, part.wire_path, config, **kwargs)
    assert result.reasons == ("missing_target_field_context",)


def test_lifting_cannot_move_a_protected_contour(loop_case, monkeypatch):
    part, config, kwargs = loop_case

    def bad_shift(parts, _config):
        parts[0].wire_path.v[3, 2] += 0.001

    monkeypatch.setattr(validation, "shift_return_paths", bad_shift)
    result = validation.validate_group_route_candidate(part, part.wire_path, config, **kwargs)
    assert result.reasons == ("prepared_protected_contour_changed",)
    assert part.wire_path.v[3, 2] == 0


def test_raw_position_revisits_are_not_confused_with_topological_branching(loop_case):
    part, _config, _kwargs = loop_case
    points = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]])
    path = WirePath(points, segment_owners=[deepcopy(part.wire_path.segment_owners[0])] * 5)
    assert validation._structural_reasons(path, [None] * 5, {}, (0,), (0,), ()) == ()
    assert validation._prepared_contour_reasons(path, {}) == ("prepared_revisited_vertex",)


def test_explicit_curves_skip_only_normal_shift(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    config.skip_normal_shift = False
    cleanup = validation.cleanup_wire_paths
    calls = []

    def counted_cleanup(parts, **options):
        calls.append("cleanup")
        return cleanup(parts, **options)

    monkeypatch.setattr(validation, "shift_return_paths", lambda *_: pytest.fail("Explicit curve was shifted again"))
    monkeypatch.setattr(validation, "cleanup_wire_paths", counted_cleanup)
    result = validation.validate_group_route_candidate(part, part.wire_path, config, apply_normal_shift=False, **kwargs)
    assert result.accepted, result.reasons
    assert calls == ["cleanup"]
    assert result.diagnostics["normal_shift_applied"] is False
    assert result.prepared_part.manufacturing_diagnostics["exact_collision_completed"]
    assert result.diagnostics["route_maximum_field_delta_t_per_a"] == 0
    assert result.diagnostics["physical_maximum_field_delta_t_per_a"] == 0


def test_skipping_normal_shift_does_not_waive_field_checks(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    path = deepcopy(part.wire_path)
    path.v[3, 0] += 0.002
    for index in (2, 3):
        del kwargs["protected_segments"][index]
    kwargs["candidate_segment_ids"] = [i if i not in (2, 3) else None for i in range(64)]
    monkeypatch.setattr(validation, "shift_return_paths", lambda *_: pytest.fail("Unexpected shift"))
    monkeypatch.setattr(validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Rejected field reached sweep"))
    result = validation.validate_group_route_candidate(
        part, path, config, baseline=baseline, apply_normal_shift=False, **kwargs
    )
    assert result.reasons == ("field_delta_budget_exceeded",)


def test_route_and_physical_background_fields_are_separate(loop_case):
    part, config, kwargs = loop_case
    config.layout_target_error_rms_limit_T = 1e-5
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    route_other = np.zeros_like(kwargs["target_points_m"])
    physical_other = np.full_like(route_other, 2e-6)
    target = TargetField(
        kwargs["target_points_m"],
        (baseline.physical_field_per_amp + physical_other) * part.contour_step,
        np.ones(2),
        np.zeros(2, dtype=bool),
        np.ones(2, dtype=int),
    )
    result = validation.validate_group_route_candidate(
        part,
        part.wire_path,
        config,
        baseline=baseline,
        target_field=target,
        other_field_per_amp=route_other,
        other_physical_field_per_amp=physical_other,
        **kwargs,
    )
    assert result.accepted, result.reasons
    assert result.diagnostics["physical_candidate_dsv_residual_rms_T"] == 0
    assert result.diagnostics["route_candidate_dsv_residual_rms_T"] > 0
    # The reusable field is per ampere for this part alone, not the whole-coil
    # or operating-current field used by the acceptance contract.
    np.testing.assert_array_equal(result.physical_field_per_amp, baseline.physical_field_per_amp)
    assert not result.physical_field_per_amp.flags.writeable
    np.testing.assert_array_equal(route_other, np.zeros_like(route_other))
    np.testing.assert_array_equal(physical_other, np.full_like(physical_other, 2e-6))


@pytest.mark.parametrize("bad_field", [np.zeros((1, 3)), np.full((2, 3), np.nan), [["bad"]]])
def test_invalid_physical_background_rejects_before_preview(loop_case, monkeypatch, bad_field):
    part, config, kwargs = loop_case
    monkeypatch.setattr(validation, "_isolated_part", lambda *_: pytest.fail("Invalid background reached preview"))
    result = validation.validate_group_route_candidate(
        part, part.wire_path, config, other_physical_field_per_amp=bad_field, **kwargs
    )
    assert result.reasons == ("invalid_other_physical_part_field",)


def test_existing_production_sweep_can_be_reused_without_generation_or_source_mutation(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    validation.create_swept_coil_meshes([part], config)
    diagnostics = deepcopy(part.manufacturing_diagnostics)
    vertices = part.swept_body.vertices.copy()
    centroid = part.fabrication_path.wire_points_m.copy()
    expected = validation._field(centroid, kwargs["target_points_m"])
    monkeypatch.setattr(
        validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Existing baseline was regenerated")
    )
    baseline = validation.prepare_group_route_baseline(
        part, config, kwargs["target_points_m"], reuse_existing_sweep=True
    )
    assert not baseline.reasons
    assert baseline.diagnostics["baseline_sweep_reused"]
    assert baseline.diagnostics["exact_collision_completed"]
    np.testing.assert_array_equal(baseline.physical_field_per_amp, expected)
    assert not baseline.physical_field_per_amp.flags.writeable
    assert part.manufacturing_diagnostics == diagnostics
    np.testing.assert_array_equal(part.swept_body.vertices, vertices)
    np.testing.assert_array_equal(part.fabrication_path.wire_points_m, centroid)


def test_explicit_sweep_reuse_fails_without_existing_physical_geometry(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    monkeypatch.setattr(
        validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Reuse silently generated geometry")
    )
    baseline = validation.prepare_group_route_baseline(
        part, config, kwargs["target_points_m"], reuse_existing_sweep=True
    )
    assert baseline.reasons == ("missing_existing_swept_geometry",)
    assert baseline.physical_field_per_amp is None


@pytest.mark.parametrize("mutation", ["centroid", "body", "body_record", "manufacturing"])
def test_physical_content_changes_invalidate_reused_baseline(loop_case, monkeypatch, mutation):
    part, config, kwargs = loop_case
    validation.create_swept_coil_meshes([part], config)
    baseline = validation.prepare_group_route_baseline(
        part, config, kwargs["target_points_m"], reuse_existing_sweep=True
    )
    if mutation == "centroid":
        points = part.fabrication_path.points_m.copy()
        points[3, 2] += 1e-5
        part.fabrication_path = replace(part.fabrication_path, points_m=points)
    elif mutation == "body":
        part.swept_body.vertices[3, 2] += 1e-5
    elif mutation == "body_record":
        part.swept_bodies[0].diagnostics = {"manufacturing_ok": False}
    else:
        part.manufacturing_diagnostics["manufacturing_ok"] = False
    monkeypatch.setattr(validation, "_isolated_part", lambda *_: pytest.fail("Stale physical data reached preview"))
    result = validation.validate_group_route_candidate(part, part.wire_path, config, baseline=baseline, **kwargs)
    assert result.reasons == ("stale_baseline",)


def test_invalid_reused_baseline_never_waives_candidate_manufacturing_gates(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    config.sweep_min_bend_radius = 0.05
    validation.create_swept_coil_meshes([part], config)
    baseline = validation.prepare_group_route_baseline(
        part, config, kwargs["target_points_m"], reuse_existing_sweep=True
    )
    assert "minimum_bend_radius_violation" in baseline.reasons
    monkeypatch.setattr(validation, "shift_return_paths", lambda *_: pytest.fail("Explicit candidate was shifted"))
    result = validation.validate_group_route_candidate(
        part, part.wire_path, config, baseline=baseline, apply_normal_shift=False, **kwargs
    )
    assert not result.accepted
    assert "minimum_bend_radius_violation" in result.reasons
    assert result.prepared_part is None


def test_direct_validator_cannot_omit_a_declared_longitudinal_budget(loop_case):
    part, config, kwargs = loop_case
    config.joint_group_routing_max_longitudinal_field_change_rel = 0.05
    result = validation.validate_group_route_candidate(part, part.wire_path, config, **kwargs)
    assert result.reasons == ("missing_longitudinal_field_delta_budget",)


def test_direct_validator_without_target_context_keeps_longitudinal_bound_hard(loop_case, monkeypatch):
    part, config, kwargs = loop_case
    config.b_0_direction = (3, 4, 0)
    baseline = validation.prepare_group_route_baseline(part, config, kwargs["target_points_m"])
    monkeypatch.setattr(validation, "_field", lambda *_: baseline.route_field_per_amp + np.array([6e-7, 8e-7, 0]))
    monkeypatch.setattr(validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Field failure reached sweep"))
    kwargs["maximum_field_delta_t_per_a"] = 2e-6
    result = validation.validate_group_route_candidate(
        part,
        part.wire_path,
        config,
        baseline=baseline,
        maximum_longitudinal_field_delta_t_per_a=0.5e-6,
        **kwargs,
    )
    assert result.reasons == ("pointwise_longitudinal_field_delta_budget",)
    assert result.diagnostics["route_maximum_pointwise_longitudinal_delta_t_per_a"] == pytest.approx(1e-6)


def test_validator_passes_longitudinal_budget_to_both_raw_and_physical_field_gates(loop_case, monkeypatch):
    from coilgen import group_route_fields

    part, config, kwargs = loop_case
    points = kwargs["target_points_m"]
    field = TargetField(
        points, np.zeros_like(points), np.ones(len(points)), np.zeros(len(points), bool), np.ones(len(points))
    )
    calls = []

    def assess(_reference, _candidate, _target, _config, _current, vector, **budgets):
        calls.append((vector, budgets))
        return group_route_fields.GroupRouteFieldAssessment(True, (), {})

    monkeypatch.setattr(group_route_fields, "assess_group_route_field", assess)
    result = validation.validate_group_route_candidate(
        part,
        part.wire_path,
        config,
        target_field=field,
        maximum_longitudinal_field_delta_t_per_a=1e-12,
        **kwargs,
    )
    assert result.accepted, result.reasons
    assert calls == [(1e-12, {"maximum_longitudinal_field_delta_t_per_a": 1e-12})] * 2
