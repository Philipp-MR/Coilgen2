"""Regressions for stalled baseline audits and optional compiler fallback."""

from copy import deepcopy
import pickle
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from coilgen import geometry_kernels as kernels, group_route_validation as validation
from coilgen.config import CoilGenConfig
from coilgen.models import CoilPart, Mesh, WirePath, WireSegmentOwner
from coilgen.progress import DesignCancelled


@pytest.fixture
def baseline_case():
    angles = np.linspace(0.0, 2 * np.pi, 33)
    points = 0.025 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    points[-1] = points[0]
    owner = WireSegmentOwner("contour_loop", group_index=0, loop_index=0)
    support = Mesh(
        np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    part = CoilPart(
        mesh=support,
        wire_path=WirePath(points, segment_owners=[deepcopy(owner) for _ in range(32)]),
        contour_step=2.0,
        swept_body=deepcopy(support),
        manufacturing_diagnostics={"manufacturing_ok": True, "mesh_solid_ok": True},
    )
    part.fabrication_path = SimpleNamespace(wire_points_m=points + [0.0, 0.0, 0.0002])
    config = CoilGenConfig(skip_normal_shift=True)
    targets = np.array([[0.0, 0.0, 0.01], [0.002, 0.0, 0.005]])
    return part, config, targets


def test_field_only_baseline_is_immutable_and_does_not_claim_exact_audit(baseline_case, monkeypatch):
    part, config, targets = baseline_case
    part.manufacturing_diagnostics["exact_collision_completed"] = True
    before = pickle.dumps(part, protocol=5)
    monkeypatch.setattr(validation, "_check_swept_geometry", lambda *_: pytest.fail("Field-only preparation audited"))
    monkeypatch.setattr(validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Existing body regenerated"))
    baseline = validation.prepare_group_route_baseline(
        part, config, targets, reuse_existing_sweep=True, audit_geometry=False
    )
    assert baseline.diagnostics["baseline_geometry_audit_deferred"]
    assert not baseline.diagnostics.get("exact_collision_completed", False)
    assert baseline.physical_field_per_amp is not None
    assert not baseline.route_field_per_amp.flags.writeable
    assert not baseline.physical_field_per_amp.flags.writeable
    assert pickle.dumps(part, protocol=5) == before


def test_field_only_and_complete_baselines_have_same_content_key_and_fields(baseline_case, monkeypatch):
    part, config, targets = baseline_case
    deferred = validation.prepare_group_route_baseline(
        part, config, targets, reuse_existing_sweep=True, audit_geometry=False
    )
    calls = []

    def complete(preview, _limits, _progress_callback=None):
        calls.append(preview)
        return {"manufacturing_ok": True, "exact_collision_completed": True}, ()

    monkeypatch.setattr(validation, "_check_swept_geometry", complete)
    certified = validation.prepare_group_route_baseline(part, config, targets, reuse_existing_sweep=True)
    assert len(calls) == 1 and calls[0] is not part
    assert deferred.fingerprint == certified.fingerprint
    np.testing.assert_array_equal(deferred.route_field_per_amp, certified.route_field_per_amp)
    np.testing.assert_array_equal(deferred.physical_field_per_amp, certified.physical_field_per_amp)
    assert certified.diagnostics["exact_collision_completed"]


def test_raw_field_rejection_with_deferred_baseline_never_reaches_sweep_or_exact_audit(baseline_case, monkeypatch):
    part, config, targets = baseline_case
    before = pickle.dumps(part, protocol=5)
    monkeypatch.setattr(validation, "_check_swept_geometry", lambda *_: pytest.fail("Rejected proposal audited"))
    monkeypatch.setattr(validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Rejected proposal swept"))
    baseline = validation.prepare_group_route_baseline(
        part, config, targets, reuse_existing_sweep=True, audit_geometry=False
    )
    candidate = deepcopy(part.wire_path)
    candidate.v += [0.0, 0.0, 0.001]
    result = validation.validate_group_route_candidate(
        part,
        candidate,
        config,
        protected_segments={},
        candidate_segment_ids=(None,) * (len(candidate.v) - 1),
        group_sequence=(0,),
        required_groups=(0,),
        target_points_m=targets,
        maximum_field_delta_t_per_a=0.0,
        baseline=baseline,
        apply_normal_shift=False,
    )
    assert not result.accepted and "field_delta_budget_exceeded" in result.reasons
    assert result.diagnostics["route_maximum_field_delta_t_per_a"] > 0.0
    assert result.prepared_part is None
    assert pickle.dumps(part, protocol=5) == before


@pytest.mark.parametrize("missing", ["swept_body", "fabrication_path"])
def test_field_only_explicit_reuse_requires_existing_physical_geometry(baseline_case, monkeypatch, missing):
    part, config, targets = baseline_case
    setattr(part, missing, None)
    monkeypatch.setattr(validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Missing geometry synthesized"))
    baseline = validation.prepare_group_route_baseline(
        part, config, targets, reuse_existing_sweep=True, audit_geometry=False
    )
    assert baseline.physical_field_per_amp is None
    assert "missing_existing_swept_geometry" in baseline.reasons


def test_deferred_audit_cannot_silently_generate_a_new_baseline(baseline_case, monkeypatch):
    part, config, targets = baseline_case
    monkeypatch.setattr(validation, "create_swept_coil_meshes", lambda *_: pytest.fail("Deferred baseline generated"))
    with pytest.raises(ValueError, match="existing sweep"):
        validation.prepare_group_route_baseline(part, config, targets, audit_geometry=False)


@pytest.mark.parametrize("mutation", ["body", "centroid"])
def test_deferred_field_baseline_rejects_stale_physical_content(baseline_case, monkeypatch, mutation):
    part, config, targets = baseline_case
    baseline = validation.prepare_group_route_baseline(
        part, config, targets, reuse_existing_sweep=True, audit_geometry=False
    )
    if mutation == "body":
        part.swept_body.vertices[0, 2] += 1e-5
    else:
        part.fabrication_path.wire_points_m[3, 2] += 1e-5
    monkeypatch.setattr(validation, "_isolated_part", lambda *_: pytest.fail("Stale deferred reference used"))
    result = validation.validate_group_route_candidate(
        part,
        part.wire_path,
        config,
        protected_segments={},
        candidate_segment_ids=(None,) * (len(part.wire_path.v) - 1),
        group_sequence=(0,),
        required_groups=(0,),
        target_points_m=targets,
        maximum_field_delta_t_per_a=0.0,
        baseline=baseline,
    )
    assert result.reasons == ("stale_baseline",)


def test_failed_production_preview_does_not_run_exact_audit_or_gain_certificate(baseline_case, monkeypatch):
    part, _config, _targets = baseline_case
    part.manufacturing_diagnostics.update(
        manufacturing_ok=False,
        manufacturing_failure_reasons="minimum_bend_radius_violation",
        exact_collision_completed=True,
    )
    monkeypatch.setattr(validation, "_exact_collision_audit", lambda *_: pytest.fail("Known failed body audited"))
    report, reasons = validation._check_swept_geometry(part, validation.GroupRouteValidationLimits())
    assert "minimum_bend_radius_violation" in reasons
    assert not report.get("exact_collision_completed", False)
    assert not report["manufacturing_ok"]


def test_exact_mesh_audit_keeps_collision_threshold_when_optional_compiler_fails(monkeypatch):
    monkeypatch.setattr(kernels, "_NUMBA_FAILURE", None)
    monkeypatch.setattr(kernels, "_NUMBA_WARNING_EMITTED", False)
    calls = []

    def incompatible(vertices):
        calls.append(vertices)
        raise ImportError("Numba needs NumPy 2.3 or less. Got NumPy 2.5.")

    monkeypatch.setattr(kernels, "_compiled_kernel", incompatible)
    triangle = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for height, expected_intersections in ((0.0, 1), (0.9e-12, 1), (1.1e-12, 0)):
            mesh = Mesh(
                np.concatenate((triangle, triangle + [0.0, 0.0, height])),
                np.array([[0, 1, 2], [3, 4, 5]]),
            )
            report, reason = validation._exact_collision_audit(mesh, validation.GroupRouteValidationLimits())
            assert report["exact_collision_completed"]
            assert report["exact_intersection_count"] == expected_intersections
            assert reason == ("swept_mesh_self_intersection" if expected_intersections else "")
    assert calls == [3]
    compatibility_warnings = [item for item in caught if "NumPy 2.5" in str(item.message)]
    assert len(compatibility_warnings) == 1


def test_exact_mesh_audit_progress_counts_real_pairs_and_finishes_without_false_total():
    triangle = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    mesh = Mesh(np.concatenate((triangle, triangle)), np.array([[0, 1, 2], [3, 4, 5]]))
    events = []
    report, reason = validation._exact_collision_audit(
        mesh, validation.GroupRouteValidationLimits(), progress_callback=events.append
    )
    assert report["exact_collision_completed"] and reason == "swept_mesh_self_intersection"
    assert events and events[-1]["status"] == "finished"
    assert all(event["total"] is None for event in events)
    assert all(event["phase"] == "exact body collision audit" for event in events)
    completed = [event["completed"] for event in events]
    assert completed == sorted(completed)
    assert completed[-1] == report["exact_collision_pair_count"] + report["exact_incident_pair_count"] == 1
    assert "building triangle search tree" in events[0]["message"]


@pytest.mark.parametrize("stage", ["baseline", "candidate"])
def test_validation_progress_cancellation_propagates_and_preserves_source(baseline_case, stage):
    part, config, targets = baseline_case
    before = pickle.dumps(part, protocol=5)

    def cancel(_event):
        raise DesignCancelled("cancelled during audit")

    with pytest.raises(DesignCancelled, match="cancelled during audit"):
        if stage == "baseline":
            validation.prepare_group_route_baseline(
                part, config, targets, reuse_existing_sweep=True, progress_callback=cancel
            )
        else:
            baseline = validation.prepare_group_route_baseline(
                part, config, targets, reuse_existing_sweep=True, audit_geometry=False
            )
            validation.validate_group_route_candidate(
                part,
                part.wire_path,
                config,
                protected_segments={},
                candidate_segment_ids=(None,) * (len(part.wire_path.v) - 1),
                group_sequence=(0,),
                required_groups=(0,),
                target_points_m=targets,
                maximum_field_delta_t_per_a=0.0,
                baseline=baseline,
                apply_normal_shift=False,
                progress_callback=cancel,
            )
    assert pickle.dumps(part, protocol=5) == before


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3])
def test_numpy_triangle_fallback_preserves_crossings_degeneracy_and_clearance(scale):
    triangle = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    first = np.repeat(triangle[None], 7, axis=0)
    second = np.array(
        [
            [[0.2, 0.2, -1], [0.2, 0.2, 1], [0.4, 0.2, 0]],
            [[0.1, 0.1, 0], [0.2, 0.1, 0], [0.1, 0.2, 0]],
            triangle + [0, 0, 0.25],
            [[0.2, 0.2, 0.25]] * 3,
            [[2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]],
            triangle + [0, 0, 0.9e-12],
            triangle + [0, 0, 1.1e-12],
        ]
    )
    first *= scale
    second *= scale
    expected = kernels.triangle_pair_distances(first, second, backend="reference").values_m
    actual = kernels.triangle_pair_distances(first, second, backend="numpy")
    assert actual.backend == "numpy_float64"
    np.testing.assert_allclose(actual.values_m, expected, rtol=2e-12, atol=scale * 1e-15)
    np.testing.assert_allclose(actual.values_m[:5], np.array([0, 0, 0.25, 0.25, 1]) * scale, atol=scale * 1e-15)
    np.testing.assert_array_equal(actual.values_m <= 1e-12 * scale, [True, True, False, False, False, True, False])


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3])
def test_numpy_fallback_matches_reference_for_random_and_nearly_parallel_pairs(scale):
    rng = np.random.default_rng(725)
    for vertices, kernel in ((2, kernels.segment_pair_distances), (3, kernels.triangle_pair_distances)):
        first, second = rng.normal(size=(2, 80, vertices, 3)) * scale
        first[0] = first[0, 0]
        second[1] = first[1]
        first[2] = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1e-13, 0]])[:vertices] * scale
        second[2] = first[2] + [0, 0, scale * 1e-9]
        expected = kernel(first, second, backend="reference").values_m
        actual = kernel(first, second, backend="numpy")
        assert actual.backend == "numpy_float64"
        np.testing.assert_allclose(actual.values_m, expected, rtol=2e-12, atol=scale * 1e-14)
