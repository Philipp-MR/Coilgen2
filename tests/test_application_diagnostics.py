"""Spatial issues refer to saved engineering witnesses in the correct geometry."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application.diagnostics import preview_diagnostics, run_problems
from coilgen.models import TargetField
from coilgen.research.storage import file_digest


@pytest.fixture
def problem_record(tmp_path):
    origin = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.0005, 0], [0.1, 0.0005, 0]])
    transverse = np.tile([0, 1, 0], (4, 1))
    normals = np.tile([0, 0, 1], (4, 1))
    centroid = origin + 0.002 * transverse + 0.001 * normals
    path = tmp_path / "part001_body0001_fabrication_frames.npz"
    np.savez(
        path,
        points_m=centroid,
        transverse=transverse,
        normals=normals,
        cross_section_m=np.array([[0, 0], [0.004, 0], [0.004, 0.002], [0, 0.002]]),
        closed=False,
    )
    diagnostics = {
        "manufacturing_failure_reasons": "sweep_clearance_violation;support_surface_penetration;folded_sweep_quad",
        "sweep_ring_count": 4,
        "sweep_capsule_min_segment_a": 0,
        "sweep_capsule_min_segment_b": 2,
        "sweep_capsule_centerline_reference": "mounted_profile_origin",
        "sweep_capsule_min_centerline_distance_m": 0.0005,
        "sweep_required_centerline_clearance_m": 0.001,
        "sweep_support_max_penetration_m": 0.0002,
        "sweep_support_conformity_tolerance_m": 0.00005,
        "sweep_worst_quad_ring": 0,
        "sweep_worst_quad_section": 0,
        "sweep_worst_quad_normal_dot": -1,
    }
    metadata = {
        "config": {"target_field_component": "Bz"},
        "fabrication": {
            "paths": [{"body_id": "part001_body0001", "diagnostics": diagnostics}],
            "failure_reasons": [
                "part001_body0001:sweep_clearance_violation;support_surface_penetration;folded_sweep_quad"
            ],
        },
        "artifacts": [{"name": path.name, "sha256": file_digest(path)}],
        "constraints": [],
        "warnings": [],
    }
    return SimpleNamespace(directory=tmp_path, metadata=metadata, arrays={}, origin=origin)


def test_capsule_witness_uses_saved_profile_origin_not_offset_centroid(problem_record):
    issues = run_problems(problem_record)
    capsule = next(issue for issue in issues if "sweep_clearance" in issue.key)
    np.testing.assert_allclose(capsule.points_m, problem_record.origin, atol=1e-18)
    assert capsule.segments == ((0, 1), (2, 3))
    assert capsule.value == 0.0005 and capsule.limit == 0.001 and capsule.unit == "m"
    assert capsule.body_id == "part001_body0001"
    assert capsule.run_directory == str(Path(problem_record.directory).resolve())
    penetration = next(issue for issue in issues if "penetration" in issue.key)
    assert not penetration.locatable


def test_topology_issue_names_saved_failure_and_connector_without_inventing_location(problem_record):
    body = problem_record.metadata["fabrication"]["paths"][0]
    body["connector_id"] = "connector-7"
    body["diagnostics"].update(
        manufacturing_failure_reasons="route_topology_failure",
        manufacturing_route_topology_failure_reasons="degenerate_opening_cut;surface_geodesic_connector",
        route_group_id=4,
    )
    issue = next(issue for issue in run_problems(problem_record) if "route_topology_failure" in issue.key)
    assert "degenerate_opening_cut" in issue.explanation
    assert "surface_geodesic_connector" in issue.explanation
    assert "connector-7" in issue.explanation and "route_group_id: 4" in issue.explanation
    assert not issue.locatable


@pytest.mark.parametrize("closed,index,expected", ((False, 1, (0, 1, 2)), (True, 0, (3, 0, 1))))
def test_minimum_bend_witness_uses_verified_final_triplet(problem_record, closed, index, expected):
    path = problem_record.directory / problem_record.metadata["artifacts"][0]["name"]
    with np.load(path, allow_pickle=False) as loaded:
        frames = {key: loaded[key] for key in loaded.files}
    frames["closed"] = np.array(closed)
    np.savez(path, **frames)
    problem_record.metadata["artifacts"][0]["sha256"] = file_digest(path)
    diagnostics = problem_record.metadata["fabrication"]["paths"][0]["diagnostics"]
    diagnostics.update(
        manufacturing_failure_reasons="minimum_bend_radius_violation",
        sweep_final_min_bend_point_index=index,
        sweep_final_bend_point_count=4,
        sweep_final_min_bend_radius_m=0.001,
        sweep_required_bend_radius_m=0.006,
    )
    issue = next(issue for issue in run_problems(problem_record) if "minimum_bend_radius_violation" in issue.key)
    np.testing.assert_array_equal(issue.points_m, frames["points_m"][list(expected)])
    assert issue.segments == ((0, 1), (1, 2))
    diagnostics["sweep_final_bend_point_count"] = 3
    issue = next(issue for issue in run_problems(problem_record) if "minimum_bend_radius_violation" in issue.key)
    assert not issue.locatable


@pytest.mark.parametrize("index", (6, 16, 17))
def test_support_witness_uses_declared_ring_major_vertex_and_asymmetric_cap_center(problem_record, index):
    from coilgen.manufacturing import _polygon_centroid

    path = problem_record.directory / problem_record.metadata["artifacts"][0]["name"]
    with np.load(path, allow_pickle=False) as loaded:
        frames = {key: loaded[key] for key in loaded.files}
    profile = np.array([[0, 0], [0.004, 0], [0.006, 0.002], [0, 0.003]])
    frames["cross_section_m"] = profile
    np.savez(path, **frames)
    problem_record.metadata["artifacts"][0]["sha256"] = file_digest(path)
    diagnostics = problem_record.metadata["fabrication"]["paths"][0]["diagnostics"]
    diagnostics.update(
        manufacturing_failure_reasons="support_surface_penetration",
        sweep_support_penetration_vertex_index=index,
        sweep_support_penetration_vertex_count=18,
        sweep_support_vertex_order="ring_major_profile_then_open_cap_centers",
    )
    issue = next(issue for issue in run_problems(problem_record) if "support_surface_penetration" in issue.key)
    ring = index // 4 if index < 16 else 0 if index == 16 else 3
    offset = (profile[index % 4] if index < 16 else profile.mean(axis=0)) - _polygon_centroid(profile)
    expected = frames["points_m"][ring] + offset[0] * frames["transverse"][ring] + offset[1] * frames["normals"][ring]
    np.testing.assert_allclose(issue.points_m, [expected], atol=1e-18)
    assert not issue.segments and issue.dataset == "review_conductor"
    assert np.linalg.norm(profile.mean(axis=0) - _polygon_centroid(profile)) > 0
    for key, value in (
        ("sweep_support_penetration_vertex_count", 16),
        ("sweep_support_vertex_order", "unrecorded"),
        ("sweep_support_penetration_vertex_index", True),
    ):
        previous = diagnostics[key]
        diagnostics[key] = value
        issue = next(issue for issue in run_problems(problem_record) if "support_surface_penetration" in issue.key)
        assert not issue.locatable
        diagnostics[key] = previous
    problem_record.metadata["artifacts"][0]["sha256"] = "0" * 64
    issue = next(issue for issue in run_problems(problem_record) if "support_surface_penetration" in issue.key)
    assert not issue.locatable


def test_fold_witness_reconstructs_recorded_patch_from_asymmetric_profile(problem_record):
    issue = next(issue for issue in run_problems(problem_record) if "folded" in issue.key)
    np.testing.assert_allclose(issue.points_m, [[0, 0, 0], [0.1, 0, 0], [0.1, 0.004, 0], [0, 0.004, 0]], atol=1e-18)
    assert issue.segments == ((0, 1), (1, 2), (2, 3), (3, 0))


@pytest.mark.parametrize("mismatch", ["hash", "ring_count", "reference", "interval"])
def test_missing_or_mismatched_witness_disables_location(problem_record, mismatch):
    diagnostics = problem_record.metadata["fabrication"]["paths"][0]["diagnostics"]
    if mismatch == "hash":
        problem_record.metadata["artifacts"][0]["sha256"] = "0" * 64
    elif mismatch == "ring_count":
        diagnostics["sweep_ring_count"] = 5
    elif mismatch == "reference":
        diagnostics["sweep_capsule_centerline_reference"] = "unspecified"
    else:
        diagnostics["sweep_capsule_min_segment_a"] = 99
    issue = next(issue for issue in run_problems(problem_record) if "sweep_clearance" in issue.key)
    assert issue.locatable is False


def test_failed_bz_field_locates_actual_design_sample_and_excludes_other_components(problem_record):
    target = np.array([[0, 0, 0], [0, 0, 10e-6], [0, 0, 0]])
    actual = target + np.array([[10, 0, 1e-6], [0, 0, 2e-6], [0, 0, 100e-6]])
    points = np.array([[0, 0, 0], [0, 0, 0.01], [0, 0, 0.02]])
    problem_record.arrays = {
        "target_field_T": target,
        "fabricated_field_T": actual,
        "fabricated_points_m": points,
        "target_suppressed": [False, False, True],
        "target_weights": [1, 1, 1],
        "validation_points_m": np.array([[3, 3, 3]]),
    }
    problem_record.metadata["constraints"] = [
        {
            "name": "Fabricated path field error (original current)",
            "passed": False,
            "value": 2e-6,
            "limit": 1e-6,
            "unit": "T",
        }
    ]
    issue = next(issue for issue in run_problems(problem_record) if issue.key.startswith("constraint"))
    assert issue.points_m == ((0.0, 0.0, 0.01),)
    assert "complete sample set" in issue.location_description
    assert issue.dataset == "fabricated"


def test_independent_winding_field_witness_uses_its_own_grid_and_target(problem_record):
    winding_points = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    winding_target = np.array([[0, 0, 1e-6], [0, 0, 3e-6]])
    problem_record.arrays = {
        "winding_validation_points_m": winding_points,
        "winding_validation_target_T": winding_target,
        "winding_validation_field_T": winding_target + [[9, 0, 1e-6], [0, 0, -4e-6]],
        "validation_points_m": np.array([[0.5, 0.5, 0.5]]),
        "validation_target_T": np.zeros((1, 3)),
        "validation_field_T": np.array([[0, 0, 100e-6]]),
    }
    problem_record.metadata["constraints"] = [
        {
            "name": "Independent winding field error (original current)",
            "passed": False,
            "scope": "independent_realized_winding",
        }
    ]
    issue = next(issue for issue in run_problems(problem_record) if issue.key.startswith("constraint"))
    assert issue.points_m == (tuple(winding_points[1]),)
    assert issue.dataset == "winding_validation_residual"
    assert issue.scope == "independent_realized_winding"


@pytest.mark.parametrize(
    "name",
    [
        "Independent winding gradient strength",
        "Independent winding gradient RMS linearity",
        "Independent winding exact-segment integration consistency",
        "Independent winding validation unavailable",
        "Fabricated wire force",
    ],
)
def test_scalar_or_unavailable_contract_has_no_invented_field_residual_witness(problem_record, name):
    problem_record.arrays = {
        "validation_points_m": np.array([[0.5, 0.5, 0.5]]),
        "validation_target_T": np.zeros((1, 3)),
        "validation_field_T": np.array([[0, 0, 100e-6]]),
        "winding_validation_points_m": np.array([[0.1, 0.2, 0.3]]),
        "winding_validation_target_T": np.zeros((1, 3)),
        "winding_validation_field_T": np.array([[0, 0, 100e-6]]),
        "fabricated_points_m": np.array([[0.4, 0.4, 0.4]]),
        "target_field_T": np.zeros((1, 3)),
        "fabricated_field_T": np.array([[0, 0, 100e-6]]),
    }
    problem_record.metadata["constraints"] = [{"name": name, "passed": False, "scope": "independent_realized_winding"}]
    issue = next(issue for issue in run_problems(problem_record) if issue.key.startswith("constraint"))
    assert not issue.locatable
    assert "scalar or unavailable" in issue.location_description


def test_async_preview_preflight_has_actual_world_si_target_without_operator_solve(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from coilgen.gui.viewport import _preview_data

    monkeypatch.setattr(
        "coilgen.physics.calculate_sensitivity_matrix", lambda *_a, **_k: pytest.fail("Operator assembled")
    )
    config = CoilGenConfig(
        coil_mesh_file="create conical frustum mesh",
        conical_frustum_mesh_parameter_list=(0.3, 0.18, 0.14, 8, 4, 1, 0, 0, 0, 0.01, -0.02, 0.03),
        surface_is_cylinder_flag=False,
        field_shape_function="0.002*(2*z*z-x*x-y*y)",
        target_gradient_normalization="none",
        target_region_radius=0.05,
        target_region_resolution=16,
        set_roi_into_mesh_center=True,
    )
    mesh, arrays, _, preflight = _preview_data(config, tmp_path)
    target = preflight["target"]
    assert preflight["coordinate_frame"] == "world"
    assert preflight["geometry"]["vertex_count"] == len(mesh.vertices) == 40
    assert target["sample_count"] == len(arrays["target_points_m"])
    assert target["normalization_scale"] == 1
    assert target["roi_center_policy"] == "support vertex centroid"
    assert target["Bz_min_T"] == pytest.approx(arrays["target_field_T"][:, 2].min())
    assert target["Bz_rms_T"] == pytest.approx(np.sqrt(np.mean(arrays["target_field_T"][:, 2] ** 2)))
    np.testing.assert_allclose(np.mean(arrays["target_points_m"], axis=0), [0.01, -0.02, 0.03], atol=1e-15)


def test_preflight_no_active_samples_has_no_inferred_field_amplitude():
    mesh = SimpleNamespace(
        vertices=np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]), faces=np.array([[0, 1, 2]]), metadata={}
    )
    target = TargetField(
        coords=np.array([[0, 0, 0], [0, 0, 0.01]]),
        b=np.array([[0, 0, 10e-6], [0, 0, -20e-6]]),
        weights=[0, 1],
        suppressed=[False, True],
        target_field_group_inds=[0, 0],
        target_diagnostics={"target_gradient_normalization_method": "off", "target_gradient_normalization_scale": 1.0},
    )
    summary = preview_diagnostics(CoilGenConfig(target_gradient_normalization="off"), mesh, target)["target"]
    assert summary["sample_count"] == 2 and summary["active_sample_count"] == 0
    assert summary["Bz_min_T"] is None and summary["Bz_max_T"] is None and summary["Bz_rms_T"] is None
    assert summary["normalization_method"] == "off" and summary["normalization_scale"] == 1.0
    np.testing.assert_array_equal(target.b[:, 2], [10e-6, -20e-6])


def test_preflight_reports_existing_normalization_without_rescaling():
    mesh = SimpleNamespace(
        vertices=np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]), faces=np.array([[0, 1, 2]]), metadata={}
    )
    target = TargetField(
        coords=np.array([[0, 0, 0], [0, 0, 0.01]]),
        b=np.array([[0, 0, 3e-6], [0, 0, 5e-6]]),
        weights=[1, 3],
        suppressed=[False, False],
        target_field_group_inds=[0, 0],
        target_diagnostics={
            "target_gradient_normalization_method": "peak",
            "target_gradient_normalization_scale": 12.0,
        },
    )
    summary = preview_diagnostics(CoilGenConfig(target_gradient_normalization="peak"), mesh, target)["target"]
    assert summary["normalization_requested"] == "peak" and summary["normalization_scale"] == 12.0
    assert summary["Bz_rms_T"] == pytest.approx(np.sqrt((3e-6**2 + 3 * 5e-6**2) / 4))
    np.testing.assert_array_equal(target.b[:, 2], [3e-6, 5e-6])
