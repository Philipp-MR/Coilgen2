"""Strict rejection must leave inspectable evidence without changing solves."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField, WirePath
from coilgen.pipeline import CoilGenPipeline
from coilgen.project import CoilProject
from coilgen.scientist_api import CoilDesignRunOptions, run_coil_design


def _failing_design(monkeypatch):
    points = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.02, 0.02, 0.0]])
    part = CoilPart(mesh=Mesh(points.copy(), np.array([[0, 1, 2]])), wire_path=WirePath(points.copy()))
    part.contour_lines = [ContourLine(points.copy(), potential=2.0, current_orientation=-1.0, turn_multiplicity=3)]

    def prepared(pipeline, state, context):
        state.parts = [part]
        state.target_field = TargetField(
            points.copy(),
            np.zeros_like(points),
            np.array([1.0, 2.0, 3.0]),
            np.array([False, True, True]),
            np.array([1, 2, 2]),
            target_gradient_dbdxyz=np.ones((3, 3, 3)),
            suppression_normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
            suppression_area_weights=np.array([0.1, 0.2, 0.3]),
            target_diagnostics={"suppression_scope": "normal_component_area_weighted"},
        )
        context.stage("prepared geometry")

    def sheet(pipeline, state, context):
        state.parts[0].stream_function = np.array([0.0, 1.0, 2.0])
        context.stage("solved sheet")

    def routed(pipeline, state, context):
        pipeline.config = replace(pipeline.config, gauss_order=7)
        state.parts[0].route_diagnostics = {"failure": "wire_clearance_violation", "minimum_gap_m": 0.0001}
        context.operations((("Validate circuit assembly", 1),))

        def reject():
            raise ValueError("strict clearance rejected")

        context.compute("Validate circuit assembly", reject)

    monkeypatch.setattr(
        "coilgen.pipeline.validate_config", lambda *args, **kwargs: SimpleNamespace(raise_for_errors=lambda: None)
    )
    monkeypatch.setattr(CoilGenPipeline, "_prepare", prepared)
    monkeypatch.setattr(CoilGenPipeline, "_solve_sheet", sheet)
    monkeypatch.setattr(CoilGenPipeline, "_extract_contours", lambda *args: None)
    monkeypatch.setattr(CoilGenPipeline, "_route_and_realize", routed)

    class Design:
        name = "rejected"

        def run(self, *, project_root):
            return CoilGenPipeline(CoilGenConfig(progress_flag=False), project_root).run()

    return Design(), points


def test_scientific_error_type_and_exact_partial_geometry_are_retained(monkeypatch, tmp_path):
    design, points = _failing_design(monkeypatch)
    options = CoilDesignRunOptions(project_root=tmp_path, output_dir=tmp_path, output_prefix="case")
    with pytest.raises(ValueError, match="strict clearance rejected") as caught:
        run_coil_design(design, options)
    report = json.loads((tmp_path / "case_failure_report.json").read_text())
    assert report["execution_status"] == "failed"
    assert report["engineering_acceptance"] == "not_qualified"
    assert report["failed_stage"] == "routed"
    assert report["failed_operation"] == "Validate circuit assembly"
    assert report["config"]["gauss_order"] == 7
    assert report["source_config"]["gauss_order"] == CoilGenConfig().gauss_order
    assert report["parts"][0]["routing"]["minimum_gap_m"] == 0.0001
    contour = report["parts"][0]["contours"][0]
    assert contour["turn_multiplicity"] == 3
    assert contour["current_orientation"] == -1.0
    assert contour["potential_A"] == 2.0
    assert report["target_diagnostics"]["suppression_scope"] == "normal_component_area_weighted"
    with np.load(tmp_path / report["geometry_archive"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["part001_source_wire_points_m"], points)
        np.testing.assert_array_equal(saved["part001_stream_function_A"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(saved[contour["points_array"]], points)
        np.testing.assert_array_equal(saved["target_target_field_group_inds"], [1, 2, 2])
        np.testing.assert_array_equal(saved["target_target_gradient_dbdxyz"], np.ones((3, 3, 3)))
        np.testing.assert_array_equal(saved["target_suppression_normals"], np.tile([0.0, 0.0, 1.0], (3, 1)))
        np.testing.assert_array_equal(saved["target_suppression_area_weights"], [0.1, 0.2, 0.3])
    assert len(caught.value.coilgen_failure_files) == 2


def test_project_stop_on_failure_still_writes_partial_summary(monkeypatch, tmp_path):
    design, _points = _failing_design(monkeypatch)
    project = CoilProject("partial", output_dir=tmp_path, project_root=tmp_path)
    project.channel("bad", design=design)
    with pytest.raises(ValueError, match="strict clearance rejected"):
        project.solve()
    assert project.last_run is not None
    report = json.loads((tmp_path / "partial_project_summary.json").read_text())
    row = report["rows"][0]
    assert row["execution_status"] == "failed"
    assert row["engineering_status"] == "NOT_ASSESSED"
    assert row["first_failure"] == "strict clearance rejected"
    assert len(row["failure_evidence"]) == 2


def test_disabled_reports_do_not_write_artifacts(monkeypatch, tmp_path):
    design, _points = _failing_design(monkeypatch)
    options = CoilDesignRunOptions(project_root=tmp_path, output_dir=tmp_path, save_reports=False)
    with pytest.raises(ValueError):
        run_coil_design(design, options)
    assert not list(tmp_path.iterdir())


def test_partial_summary_write_error_preserves_scientific_failure(monkeypatch, tmp_path):
    design, _points = _failing_design(monkeypatch)
    project = CoilProject("partial", output_dir=tmp_path, project_root=tmp_path)
    project.channel("bad", design=design)

    def fail_summary(*args, **kwargs):
        raise OSError("summary storage unavailable")

    monkeypatch.setattr("coilgen.project.write_project_summary", fail_summary)
    with pytest.raises(ValueError, match="strict clearance rejected") as caught:
        project.solve()
    assert project.last_run is not None
    assert project.last_run.channel_runs[0].exception is caught.value
    assert any("summary storage unavailable" in note for note in caught.value.__notes__)
