"""Opt-in engineering preserves immutable study and realized-field contracts."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application import ProjectDocument, plan_execution
from coilgen.application.evidence import resolve_dataset
from coilgen.application.execution import matches_active_request
from coilgen.application.gradient_review import gradient_review
from coilgen.gradient_engineering import CircuitSpec, DriveSpec, GradientEngineeringSpec, winding_path_sha256
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField
from coilgen.physics import biot_savart_wire_path
from coilgen.pipeline_stages import PipelineStage
from coilgen.research.application import run_design
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import request_identity
from coilgen.scientific_arrays import scientific_arrays


def config():
    return CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        field_shape_function=str(4e-7 * np.sqrt(2) * 2 / 0.1),
        target_gradient_normalization="off",
        target_region_radius=0.01,
        target_error_constraint_metric="rms",
        target_error_rms_limit_T=1e-6,
        research_physics={},
        skip_postprocessing=True,
        skip_sweep=True,
        progress_flag=False,
        plot_flag=False,
    )


def engineering(**changes):
    return GradientEngineeringSpec(**changes).to_dict()


def engineering_project():
    return ProjectDocument.from_config(config()).update_study({"fabricate": True, "engineering": engineering()})


def test_absent_engineering_keeps_legacy_request_and_project_serialization(tmp_path):
    original = ResearchRequest.from_config(config())
    legacy = asdict(original)
    optional = ("engineering", "requirements", "scene", "dynamic_analysis", "array_study")
    for key in optional:
        assert legacy.pop(key) is None
    assert original.to_dict() == json.loads(json.dumps(legacy, default=str))
    assert ResearchRequest.from_dict({**legacy, **dict.fromkeys(optional)}).to_dict() == original.to_dict()
    project = ProjectDocument.from_config(config())
    path = project.save(tmp_path / "default.coilgen.json")
    assert "engineering" not in project.study
    assert "engineering" not in json.loads(path.read_text())["study"]
    assert project.update_study({"engineering": None}) is project


def test_engineering_edit_has_its_own_immutable_revision_and_request_identity(tmp_path):
    baseline = ProjectDocument.from_config(config()).update_study({"fabricate": True})
    authored = engineering()
    changed = baseline.update_study({"engineering": authored})
    assert changed.id == baseline.id and changed.revision == baseline.revision + 1
    assert changed.fingerprint != baseline.fingerprint
    assert changed.model_fingerprint == baseline.model_fingerprint
    authored["independent_winding_validation"] = False
    detached = changed.study
    detached["engineering"]["circuits"].append({"invalid": "mutable caller"})
    assert changed.study["engineering"] == engineering()
    assert "engineering" not in baseline.study
    first = baseline.snapshot(tmp_path / "baseline")
    second = changed.snapshot(tmp_path / "changed")
    first_digest = request_identity(first.to_dict(), asdict(first.to_config()))[0]
    second_digest = request_identity(second.to_dict(), asdict(second.to_config()))[0]
    assert first_digest != second_digest
    assert second.project_fingerprint == changed.fingerprint
    assert second.engineering == changed.study["engineering"]


def test_stale_engineering_plan_cannot_submit_after_study_revision():
    project = engineering_project()
    plan = plan_execution(project)
    changed = project.update_study({"engineering": engineering(independent_winding_validation=False)})
    with pytest.raises(ValueError, match="earlier project revision"):
        plan.require_current(changed)


def test_snapshot_binds_feed_geometry_and_polarity_to_saved_study_revision(tmp_path):
    winding = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
    circuit = CircuitSpec(
        body_id="loop",
        expected_path_sha256=winding_path_sha256(winding),
        polarity=-1,
        supply_points_m=[[0, -0.02, 0], winding[-1]],
        return_points_m=[winding[0], [0, -0.02, 0]],
        source="Measured feed routing",
    )
    authored = engineering(circuits=(circuit,))
    project = ProjectDocument.from_config(config()).update_study({"fabricate": True, "engineering": authored})
    request = project.snapshot(tmp_path / "snapshot")
    authored["circuits"][0]["supply_points_m"] = [[999, 999, 999], [999, 999, 998]]
    detached = request.to_dict()
    detached["engineering"]["circuits"][0]["polarity"] = 1
    saved = request.engineering["circuits"][0]
    assert saved["expected_path_sha256"] == winding_path_sha256(winding)
    assert saved["polarity"] == -1
    np.testing.assert_array_equal(saved["supply_points_m"], circuit.supply_points_m)
    assert request.project_revision == project.revision and request.project_fingerprint == project.fingerprint
    assert request.to_config().field_shape_function == project.config.field_shape_function


def test_nested_engineering_controls_cannot_mutate_a_planned_physics_request():
    project = engineering_project()
    controls = engineering()
    plan = plan_execution(project, engineering=controls)
    controls["independent_winding_validation"] = False
    assert plan.request_controls()["engineering"]["independent_winding_validation"] is True
    with pytest.raises(TypeError):
        plan.controls["engineering"]["independent_winding_validation"] = False
    detached = plan.request_controls()
    detached["engineering"]["circuits"].append({"invalid": "caller edit"})
    assert plan.request_controls()["engineering"]["circuits"] == []


def test_active_engineering_job_must_match_exact_recorded_review_controls():
    project = engineering_project()
    state = {
        "status": "running",
        "project_id": project.id,
        "project_revision": project.revision,
        "project_fingerprint": project.fingerprint,
        **project.study,
    }
    assert matches_active_request(state, project)
    changed = deepcopy(state)
    changed["engineering"]["independent_winding_validation"] = False
    assert not matches_active_request(changed, project)
    state.pop("engineering")
    assert not matches_active_request(state, project)


@pytest.mark.parametrize("changes", [{"fabricate": False}, {"pipeline_stage": "sheet", "fabricate": False}])
def test_engineering_requires_evaluated_fabrication_review(changes):
    with pytest.raises(ValueError, match="evaluated fabrication"):
        ResearchRequest.from_config(config(), engineering=engineering(), **changes)


def test_independent_winding_capability_requires_its_own_coordinates_and_residual_target():
    points = np.array([[0, 0, 0.01], [0, 0, 0.02]])
    arrays = {
        "target_points_m": np.zeros((1, 3)),
        "target_field_T": np.zeros((1, 3)),
        "validation_points_m": points,
        "validation_field_T": np.ones((2, 3)),
        "winding_validation_field_T": np.ones((2, 3)),
    }
    assert not resolve_dataset(arrays, "winding_validation").available
    arrays["winding_validation_points_m"] = points
    assert resolve_dataset(arrays, "winding_validation").available
    assert not resolve_dataset(arrays, "winding_validation_residual").available
    arrays["winding_validation_target_T"] = np.zeros((1, 3))
    assert resolve_dataset(arrays, "winding_validation_residual").invalid
    arrays["winding_validation_target_T"] = np.zeros((2, 3))
    assert resolve_dataset(arrays, "winding_validation_residual").available
    assert resolve_dataset(arrays, "winding_validation_residual", original=True).available


def test_surface_current_density_export_is_face_based_in_A_per_m_without_new_calculation():
    meshes = [
        Mesh(np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], float) + offset, np.array([[0, 1, 2]]))
        for offset in (0, 0.1)
    ]
    parts = [
        CoilPart(mesh=mesh, current_density=np.array([density]))
        for mesh, density in zip(meshes, ([3, 4, 0], [0, 0, 7]))
    ]
    target = TargetField(np.zeros((1, 3)), np.zeros((1, 3)), [1], [False], [1])
    saved = scientific_arrays(parts, target)
    np.testing.assert_array_equal(saved["sheet_current_density_A_per_m"], [[3, 4, 0], [0, 0, 7]])
    np.testing.assert_allclose(
        saved["sheet_current_density_points_m"], [mesh.vertices[mesh.faces].mean(axis=1)[0] for mesh in meshes]
    )
    assert len(saved["sheet_current_density_A_per_m"]) == len(saved["faces"])
    parts[1].current_density = None
    assert "sheet_current_density_A_per_m" not in scientific_arrays(parts, target)
    parts[1].current_density = np.array([[np.nan, 0, 0]])
    assert "sheet_current_density_A_per_m" not in scientific_arrays(parts, target)


@pytest.fixture
def pipeline_fixture(monkeypatch):
    """Keep exact saved wire fields; isolate orchestration from the expensive solve."""
    import coilgen.research.application as application

    path = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float) * 0.1
    points = np.array([[0.002, 0.003, 0.004], [-0.002, -0.003, -0.004]])
    mesh = Mesh(path[:-1].copy(), np.array([[0, 1, 2], [0, 2, 3]]))
    part = CoilPart(
        mesh=mesh,
        contour_step=2,
        contour_lines=[ContourLine(v=path, potential=1)],
        current_density=np.array([[3, 4, 0], [0, 0, 7]]),
    )
    target = TargetField(points, np.tile([0, 0, 4e-7 * np.sqrt(2) * 2 / 0.1], (2, 1)), [1, 1], [False, False], [1, 1])
    per_amp = biot_savart_wire_path(path, points)
    fixture = SimpleNamespace(path=path, part=part, target=target, captured={}, ready=True)

    class Pipeline:
        def __init__(self, executed, **_kwargs):
            fixture.captured["config"] = executed
            self.result = SimpleNamespace(
                input_data=executed,
                target_field=target,
                coil_parts=[part],
                combined_mesh=mesh,
                regularization_diagnostics={
                    "constrained_solver_contract_met": True,
                    "target_gradient_normalization_scale": 1,
                },
                b_field_opt_sf=target.b.copy(),
                field_by_unconnected_loops=2 * per_amp,
                field_by_layout=2 * per_amp,
                field_loops_per1Amp=per_amp,
                field_layout_per1Amp=per_amp,
                needed_current_layout=2,
                timing_summary={},
            )

        def run_stage(self, stage, resumed, **_kwargs):
            assert stage == "evaluated" and resumed is None
            return SimpleNamespace(
                stage=PipelineStage.EVALUATED,
                result=self.result,
                save=lambda destination: Path(destination).write_bytes(b"fake evaluated checkpoint"),
            )

    def compile_review(result, **kwargs):
        fixture.captured["compile_kwargs"] = kwargs
        assert result.coil_parts[0].contour_step == 2
        electrical = {
            "bodies": [
                {
                    "body_id": "loop",
                    "current_A": 2.0,
                    "circuit_complete": True,
                    "resistance_ohm": 0.01,
                    "inductance_H": 1e-6,
                    "length_m": 0.8,
                    "cross_section_area_m2": 1e-6,
                    "joule_power_W": 0.04,
                }
            ],
            "total_joule_power_W": 0.04,
        }
        return SimpleNamespace(
            paths=[SimpleNamespace(points_m=path)],
            winding_steps=[
                {"body_id": "loop", "electrical_points_m": path, "circuit_complete": True, "drive_feed_designed": False}
            ],
            electrical_metrics=electrical,
            artifacts=[],
            failure_reasons=[],
            ready=fixture.ready,
            summary=lambda: {"ready": fixture.ready, "electrical_metrics": electrical},
        )

    monkeypatch.setattr(application, "CoilGenPipeline", Pipeline)
    monkeypatch.setattr("coilgen.research.fabrication.compile_fabrication", compile_review)
    monkeypatch.setattr(
        application,
        "write_design_report",
        lambda _result, destination: Path(destination).write_text("Recorded test report"),
    )
    monkeypatch.setattr(
        application,
        "validate_on_independent_grid",
        lambda result, **_kwargs: {
            "available": True,
            "verified": True,
            "passed": True,
            "value": 0.0,
            "limit": 1e-6,
            "unit": "T",
            "points_m": points + [0, 0, 0.001],
            "field_T": result.target_field.b.copy(),
            "target_T": result.target_field.b.copy(),
        },
    )
    return fixture


def run_fixture(tmp_path, *, review=None):
    return run_design(
        ResearchRequest.from_config(config(), fabricate=True, validation_resolution=3, engineering=review), tmp_path
    )


def test_default_no_engineering_run_retains_existing_field_scope_and_exact_arrays(tmp_path, pipeline_fixture):
    baseline = run_fixture(tmp_path / "default")
    explicit_null = run_fixture(tmp_path / "null", review=None)
    assert baseline["summary"]["accepted"] and baseline["summary"]["sheet_accepted"]
    assert baseline["summary"]["scope"] == "fabrication_review"
    assert baseline["summary"]["construction_released"] is False
    assert "winding_validation" not in baseline and "engineering_review" not in baseline
    assert "circuits" not in pipeline_fixture.captured["compile_kwargs"]
    assert baseline["constraints"] == explicit_null["constraints"]
    with (
        np.load(tmp_path / "default/scientific_result.npz") as first,
        np.load(tmp_path / "null/scientific_result.npz") as second,
    ):
        assert first.files == second.files
        for name in first.files:
            np.testing.assert_array_equal(first[name], second[name])
        assert not any(name.startswith("winding_validation") for name in first.files)


@pytest.mark.parametrize("spacing", [0.0, 0.003, np.inf])
def test_summary_keeps_zero_contour_spacing_and_marks_unmeasured_spacing_unknown(tmp_path, pipeline_fixture, spacing):
    part = pipeline_fixture.part
    part.contour_lines.append(ContourLine(v=pipeline_fixture.path.copy(), potential=2))
    part.contour_distance_matrix = np.array([[0, spacing], [spacing, 0]])
    part.minimal_contour_distance = spacing if np.isfinite(spacing) else 0
    request = ResearchRequest.from_config(
        replace(config(), skip_calculation_min_winding_distance=False), fabricate=True
    )
    summary = run_design(request, tmp_path)["summary"]
    assert summary["minimum_spacing_m"] == (spacing if np.isfinite(spacing) else None)
    assert "surface-geodesic" in summary["minimum_spacing_scope"]


def test_skipped_contour_distance_does_not_show_zero_as_a_measurement(tmp_path, pipeline_fixture):
    part = pipeline_fixture.part
    part.contour_lines.append(ContourLine(v=pipeline_fixture.path.copy(), potential=2))
    part.contour_distance_matrix = np.zeros((2, 2))
    request = ResearchRequest.from_config(replace(config(), skip_calculation_min_winding_distance=True), fabricate=True)
    assert run_design(request, tmp_path)["summary"]["minimum_spacing_m"] is None


def test_run_exports_original_current_independent_winding_arrays_and_scoped_metadata(tmp_path, pipeline_fixture):
    payload = run_fixture(tmp_path, review=engineering())
    validation = payload["winding_validation"]
    assert validation["available"] and validation["passed"]
    assert validation["currents_A"] == {"loop": 2}
    assert validation["current_refitted"] is False
    assert payload["summary"]["accepted"]
    assert pipeline_fixture.captured["compile_kwargs"]["circuits"] == {}
    with np.load(tmp_path / "scientific_result.npz") as saved:
        points = saved["winding_validation_points_m"]
        expected = 2 * biot_savart_wire_path(pipeline_fixture.path, points)
        np.testing.assert_allclose(saved["winding_validation_field_T"], expected, rtol=1e-14, atol=1e-20)
        assert points.shape == saved["winding_validation_target_T"].shape
        assert len(points) != len(saved["target_points_m"])
    assert all(row["scope"] == "independent_realized_winding" for row in validation["checks"])


def test_requested_unavailable_winding_validation_blocks_acceptance(tmp_path, pipeline_fixture, monkeypatch):
    monkeypatch.setattr(
        "coilgen.research.application.validate_winding_on_independent_grid",
        lambda *_a, **_k: {
            "available": False,
            "verified": False,
            "passed": None,
            "reason": "No separate independent dataset",
            "checks": [],
            "unknown_reasons": ["No separate independent dataset"],
        },
    )
    payload = run_fixture(tmp_path, review=engineering())
    assert payload["summary"]["sheet_accepted"]
    assert not payload["summary"]["accepted"]
    row = next(row for row in payload["constraints"] if row["name"] == "Independent winding validation unavailable")
    assert row["passed"] is None and row["scope"] == "independent_realized_winding"
    with np.load(tmp_path / "scientific_result.npz") as saved:
        assert not any(name.startswith("winding_validation") for name in saved.files)


def test_requested_unavailable_drive_blocks_acceptance(tmp_path, pipeline_fixture):
    path = pipeline_fixture.path
    circuit = CircuitSpec(
        body_id="loop",
        expected_path_sha256=winding_path_sha256(path),
        feed_break_segment=0,
        supply_points_m=[[0, -0.2, 0], path[1]],
        return_points_m=[path[0], [0, -0.2, 0]],
        source="Measured routing",
    )
    drive = DriveSpec(
        body_ids=["loop"],
        time_s=[0, 0.001, 0.002],
        currents_A=[[0], [2], [0]],
        current_limits_A=[10],
        voltage_limits_V=[100],
        source="Amplifier waveform",
    )
    payload = run_fixture(tmp_path, review=engineering(circuits=(circuit,), drive=drive))
    assert not payload["engineering_review"]["drive"]["available"]
    assert not payload["summary"]["accepted"]
    row = next(row for row in payload["constraints"] if row["name"] == "Requested drive review unavailable")
    assert row["passed"] is None and row["verified"] is False
    assert pipeline_fixture.captured["compile_kwargs"]["circuits"]["loop"] == circuit


def test_gradient_review_retains_sheet_wire_and_open_circuit_scope_without_acceptance():
    metadata = {
        "summary": {"max_current_density_a_per_m": 100, "minimum_spacing_m": 0.003},
        "winding_validation": {"available": True, "linearity": {"linearity_achieved_gradient_T_per_m": 0.02}},
        "fabrication": {
            "electrical_metrics": {
                "bodies": [
                    {
                        "body_id": "open",
                        "current_A": -2,
                        "cross_section_area_m2": 1e-6,
                        "circuit_complete": False,
                        "inductance_H": 1e-6,
                    }
                ]
            }
        },
    }
    rows = {row.key: row for row in gradient_review(metadata)}
    assert rows["sheet_K"].unit == "A/m" and "sheet" in rows["sheet_K"].scope.lower()
    assert rows["open:wire_J"].value == 2e6 and rows["open:wire_J"].unit == "A/m²"
    assert rows["open:efficiency"].value is None
    assert "partial" in rows["open:inductance_H"].scope
    assert "independent grid" in rows["linearity_achieved_gradient_T_per_m"].scope
    assert not any(hasattr(row, "passed") for row in rows.values())


def test_acceptance_rows_keep_winding_scope_body_and_verdict_independent_of_sheet_verification():
    from coilgen.gui.acceptance import acceptance_report

    report = acceptance_report(
        {
            "summary": {"scope": "fabrication_review"},
            "validation": {"available": False, "verified": False, "passed": None},
            "winding_validation": {"available": True, "verified": True, "passed": True},
            "constraints": [
                {
                    "name": "Independent winding field error (original current)",
                    "body_id": "loop",
                    "scope": "independent_realized_winding",
                    "verified": True,
                    "passed": True,
                    "value": 1e-7,
                    "limit": 1e-6,
                    "effective_limit": 1.01e-6,
                    "unit": "T",
                }
            ],
        }
    )
    row = next(row for row in report.checks if row.key == "constraint:0")
    assert row.stage == "physical" and row.body_id == "loop"
    assert row.requirement == "Independent winding field error (original current)"
    assert row.status == "Passed" and row.observed == 1e-7
    assert row.margin == pytest.approx(9.1e-7)
    assert "independent grid" in row.scope and "original signed current" in row.scope
