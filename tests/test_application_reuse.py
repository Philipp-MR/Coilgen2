"""Real stage snapshots, immutable evidence and source-aware reuse regressions."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.reuse import plan_reuse
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineStage
from test_pipeline_stages import config


def test_boundary_observer_preserves_script_outputs_and_earlier_state():
    settings = config()
    captured = []
    actual = CoilGenPipeline(settings).run_stage("evaluated", checkpoint_callback=captured.append)
    expected = CoilGenPipeline(settings).run_stage("evaluated")
    assert [entry.stage for entry in captured] == list(PipelineStage)
    assert not captured[1].parts[0].contour_lines
    assert captured[2].parts[0].contour_lines
    np.testing.assert_array_equal(actual.result.b_field_opt_sf, expected.result.b_field_opt_sf)
    np.testing.assert_array_equal(actual.result.field_by_layout, expected.result.field_by_layout)
    changed = replace(settings, levels=9)
    reused = CoilGenPipeline(changed).run_stage("contours", captured[1])
    fresh = CoilGenPipeline(changed).run_stage("contours")
    np.testing.assert_array_equal(reused.sf_b_field, captured[1].sf_b_field)
    for first, second in zip(reused.parts[0].contour_lines, fresh.parts[0].contour_lines, strict=True):
        np.testing.assert_array_equal(first.v, second.v)


@pytest.fixture
def run(tmp_path):
    project = ProjectDocument.from_config(config(), name="Reuse test").update_study({"pipeline_stage": "contours"})
    DesignService.run(project, tmp_path / "run")
    return project, DesignService.load_run(tmp_path / "run")


def test_highest_actual_compatible_stage_and_requested_boundary(run):
    project, record = run
    plan = plan_reuse(project, [record])
    assert plan.through("prepared").stage is PipelineStage.PREPARED
    assert plan.through("evaluated").stage is PipelineStage.CONTOURS
    changed = project.update_config({"levels": 9})
    plan = plan_reuse(changed, [record])
    assert plan.through("evaluated").stage is PipelineStage.SHEET
    assert "contours" in plan.stale
    linked = project.update_config({"specific_conductivity_conductor": 3e-8})
    assert plan_reuse(linked, [record]).through("evaluated").stage is PipelineStage.PREPARED
    unrelated = ProjectDocument.from_config(project.config)
    assert not plan_reuse(unrelated, [record]).available


def test_corrupt_checkpoint_is_not_reused_or_presented_as_current(run):
    project, record = run
    checkpoint = record.directory / "checkpoints/sheet.cgstage"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
    plan = plan_reuse(project.update_config({"levels": 10}), [record])
    assert plan.through("evaluated").stage is PipelineStage.PREPARED
    assert any("integrity" in message for message in plan.diagnostics)
    with pytest.raises(ValueError, match="integrity"):
        DesignService.load_run(record.directory)
