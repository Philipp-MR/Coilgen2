"""Saved field coordinates and reviewed paths must describe their own geometry."""

from types import SimpleNamespace

import numpy as np

from coilgen import CoilGenConfig
from coilgen.models import CoilPart, Mesh, SweptBody
from coilgen.physics import biot_savart_wire_path
from coilgen.research.application import _scientific_arrays, run_design
from coilgen.research.requests import ResearchRequest


def _mesh(offset=0):
    return Mesh(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float64) + offset,
        np.array([[0, 1, 2]], dtype=np.int64),
    )


def _target():
    return SimpleNamespace(coords=np.zeros((1, 3)), b=np.zeros((1, 3)), suppressed=[False], weights=[1.0])


def test_pipeline_sweep_export_preserves_individual_meshes_without_duplicate_aggregate():
    first, second, aggregate = _mesh(0.1), _mesh(0.2), _mesh(9.0)
    part = CoilPart(
        mesh=_mesh(),
        swept_body=aggregate,
        swept_bodies=[SweptBody(first), SweptBody(second)],
        manufacturing_diagnostics={"manufacturing_ok": False},
    )

    arrays = _scientific_arrays([part], _target())

    assert {key for key in arrays if key.startswith("swept_")} == {
        "swept_0_0_vertices_m",
        "swept_0_0_faces",
        "swept_0_1_vertices_m",
        "swept_0_1_faces",
    }
    for index, original in enumerate((first, second)):
        np.testing.assert_array_equal(arrays[f"swept_0_{index}_vertices_m"], original.vertices)
        np.testing.assert_array_equal(arrays[f"swept_0_{index}_faces"], original.faces)
    assert part.manufacturing_diagnostics["manufacturing_ok"] is False


def test_legacy_sweep_aggregate_is_exported_only_when_no_individual_bodies_exist():
    swept = _mesh(0.3)
    parts = [CoilPart(mesh=_mesh()), CoilPart(mesh=_mesh(), swept_body=swept)]

    arrays = _scientific_arrays(parts, _target())

    assert {key for key in arrays if key.startswith("swept_")} == {"swept_1_0_vertices_m", "swept_1_0_faces"}
    np.testing.assert_array_equal(arrays["swept_1_0_vertices_m"], swept.vertices)
    np.testing.assert_array_equal(arrays["swept_1_0_faces"], swept.faces)


def test_review_field_retains_design_grid_and_review_geometry_separate_from_sheet_grid(tmp_path, monkeypatch):
    captured = {}

    def compile_review(result, **kwargs):
        # Simulate a different mounting policy. Its return participates in B
        # while the mounted-conductor trace remains a distinct dataset.
        contour = result.coil_parts[0].contour_lines[0].v
        mounted = contour + np.asarray([0.001, 0.002, 0.003])
        electrical = np.vstack((mounted, mounted[-1] + [0.01, 0.0, 0.0], mounted[0]))
        captured.update(result=result, mounted=mounted, electrical=electrical)
        return SimpleNamespace(
            paths=[SimpleNamespace(points_m=mounted)],
            winding_steps=[{"electrical_points_m": electrical, "drive_feed_designed": False}],
            electrical_metrics={"bodies": [], "total_joule_power_W": 0.0},
            artifacts=[],
            failure_reasons=["synthetic review remains unverified"],
            ready=False,
            summary=lambda: {"ready": False},
        )

    def validate_review(result, fabrication, **kwargs):
        field = result.coil_parts[0].contour_step * biot_savart_wire_path(
            captured["electrical"], result.target_field.coords, worker_count=1
        )
        captured["field"] = field
        return {"checks": [], "unknown_reasons": ["synthetic review remains unverified"], "field_T": field}

    monkeypatch.setattr("coilgen.research.fabrication.compile_fabrication", compile_review)
    monkeypatch.setattr("coilgen.research.application.validate_fabricated_constraints", validate_review)
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=8,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power"},
        constrained_quadratic_fallback_policy="strict",
        target_error_tolerance_percent=5,
        stream_function_smoothness_factor=0,
        turn_spacing_regularization_factor=0,
        levels=6,
        min_point_loop_number=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        progress_flag=False,
        plot_flag=False,
    )
    result = run_design(ResearchRequest.from_config(config, fabricate=True, validation_resolution=5), tmp_path)
    assert result["summary"]["scope"] == "fabrication_review" and result["summary"]["accepted"] is False
    assert "sheet_accepted" in result["summary"]
    assert result["summary"]["fabrication_review_performed"] is True
    assert result["summary"]["pipeline_sweep_checks_passed"] is None
    assert result["wire_fields"]["contour"]["current_refitted"] is False
    assert result["wire_fields"]["routed"]["available"] is False
    with np.load(tmp_path / "scientific_result.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["contour_field_T"], captured["result"].field_by_unconnected_loops)
        assert "routed_field_T" not in saved.files
        np.testing.assert_array_equal(saved["fabricated_points_m"], captured["result"].target_field.coords)
        np.testing.assert_array_equal(saved["fabricated_field_T"], captured["field"])
        np.testing.assert_array_equal(saved["review_conductor_0_points_m"], captured["mounted"])
        np.testing.assert_array_equal(saved["review_electrical_0_points_m"], captured["electrical"])
        assert saved["fabricated_field_T"].shape == saved["fabricated_points_m"].shape
        assert len(saved["validation_points_m"]) != len(saved["fabricated_points_m"])
        assert len(saved["review_electrical_0_points_m"]) > len(saved["review_conductor_0_points_m"])
