"""Execution plans pin input provenance without changing solver requests."""

from unittest.mock import Mock

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectDocument, RunRecord, plan_execution, run_actions
from coilgen.application.execution import ExecutionEvent, matches_active_request, matches_project
from coilgen.application.reuse import ReusableStage, ReusePlan
from coilgen.pipeline_stages import PipelineStage
from coilgen.progress import DesignCancelled, ProgressReporter
from test_application_evidence import _saved_run


def _state(project, **changes):
    return {
        "id": "active",
        "status": "running",
        "project_id": project.id,
        "project_revision": project.revision,
        "project_fingerprint": project.fingerprint,
        **project.study,
        **changes,
    }


def _reuse(tmp_path):
    return ReusePlan((ReusableStage(PipelineStage.SHEET, tmp_path / "sheet.cgstage", tmp_path),), (), ())


def test_plan_keeps_exact_owner_stage_controls_and_actual_checkpoint(tmp_path):
    project = ProjectDocument.new()
    state = _state(project)
    plan = plan_execution(project, states=[state], reuse=_reuse(tmp_path))
    assert plan.active_job_id == "active"
    assert plan.checkpoint.stage == PipelineStage.SHEET
    assert plan.request_controls() == {**project.study, "resume_checkpoint": str(tmp_path / "sheet.cgstage")}
    assert plan_execution(project, pipeline_stage="prepared", reuse=plan.reuse).checkpoint is None
    with pytest.raises(TypeError):
        plan.controls["fabricate"] = True
    assert project.revision == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"project_id": "other"},
        {"project_fingerprint": "earlier"},
        {"pipeline_stage": "sheet"},
        {"validation_resolution": 16},
        {"design_margin": 0.8},
        {"fabricate": True},
        {"kernel_backend": "numpy"},
        {"status": "completed"},
    ],
)
def test_changed_request_cannot_be_the_matching_active_job(changes):
    project = ProjectDocument.new()
    state = _state(project, **changes)
    assert not matches_active_request(state, project)
    assert plan_execution(project, states=[state]).active_job_id is None


@pytest.mark.parametrize("key", ["kernel_backend", "validation_resolution", "design_margin", "fabricate"])
def test_missing_recorded_control_is_unknown_not_assumed_current(key):
    project = ProjectDocument.new()
    state = _state(project)
    state.pop(key)
    assert not matches_active_request(state, project)


@pytest.mark.parametrize(
    "changes",
    [
        {"fabricate": 1},
        {"validation_resolution": 1},
        {"design_margin": 2},
        {"kernel_backend": "unknown"},
        {"project_id": "other"},
        {"project_revision": 7},
        {"project_fingerprint": "wrong"},
    ],
)
def test_invalid_or_foreign_controls_are_rejected_before_deduplication(changes):
    project = ProjectDocument.new()
    with pytest.raises((ValueError, TypeError)):
        plan_execution(project, states=[_state(project)], **changes)


def test_plan_rejects_edit_and_revision_change_before_submission(tmp_path):
    project = ProjectDocument.new()
    plan = DesignService.plan(project)
    jobs = Mock(directory=tmp_path)
    changed = project.update_study({"validation_resolution": 16})
    with pytest.raises(ValueError, match="earlier project revision"):
        DesignService.submit_plan(changed, jobs, plan)
    jobs.list.assert_not_called()
    jobs.submit.assert_not_called()
    renamed = project.rename("Renamed")
    assert renamed.fingerprint == project.fingerprint
    with pytest.raises(ValueError):
        plan.require_current(renamed)


def test_live_ownership_requires_recorded_revision():
    project = ProjectDocument.new()
    state = _state(project)
    state.pop("project_revision")
    assert matches_project(state, project)
    assert not matches_project(state, project, revision=True)
    assert not matches_active_request(state, project)
    assert not matches_active_request(_state(project), project.rename("Renamed"))


@pytest.mark.parametrize("status", ["running", "queued", "cancelling"])
def test_active_job_is_not_a_recovery_source(tmp_path, status):
    actions = run_actions("evaluated", state={"status": status}, reuse=_reuse(tmp_path))
    assert not actions.can_resume
    assert actions.can_cancel == (status != "cancelling")


def test_failed_source_never_enables_recovery_or_viewing(tmp_path):
    actions = run_actions("evaluated", record=object(), reuse=_reuse(tmp_path), source_error="Checksum changed")
    assert not actions.can_resume and not actions.can_view_results
    assert "Checksum changed" in actions.reason
    assert run_actions("evaluated", state={"status": "failed"}, reuse=_reuse(tmp_path)).can_resume


def test_selective_array_access_keeps_existing_copy_contract(tmp_path):
    source = np.arange(6.0).reshape(2, 3)
    record = RunRecord(tmp_path, {}, {"field": source, "other": np.zeros(100)})
    source[0, 0] = 99
    assert record.array("field")[0, 0] == 0
    detached = record.arrays
    detached["field"][0, 0] = 55
    view = record.array("field", copy=False)
    assert view[0, 0] == 0 and not view.flags.writeable
    with pytest.raises(ValueError):
        view[0, 0] = 7
    with pytest.raises(ValueError):
        view.flags.writeable = True
    assert tuple(record.select_arrays(["field"], copy=False)) == ("field",)
    assert record.array_names == ("field", "other")


def test_selective_load_verifies_all_artifacts_but_decompresses_only_requested_arrays(tmp_path, monkeypatch):
    directory = _saved_run(tmp_path)
    accessed = []
    original = np.lib.npyio.NpzFile.__getitem__

    def read(data, name):
        accessed.append(name)
        return original(data, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", read)
    selected = DesignService.load_run(directory, array_names=["target_field_T"])
    assert selected.array_names == ("target_field_T",)
    assert accessed == ["target_field_T"]
    with pytest.raises(KeyError):
        DesignService.load_run(directory, array_names=["uncomputed_field"])
    (directory / "scientific_result.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        DesignService.load_run(directory, array_names=[])


def test_typed_events_preserve_existing_stage_boundaries(monkeypatch, tmp_path):
    project = ProjectDocument.new()
    events = []

    def run(*args, **kwargs):
        reporter = ProgressReporter()
        reporter.report(None, "Field operators")
        reporter.boundary("prepared")
        return {"request_sha256": "verified"}

    monkeypatch.setattr(DesignService, "_run", run)
    result = DesignService.run(project, tmp_path, progress=events.append)
    assert result["request_sha256"] == "verified"
    assert [event.kind for event in events] == ["started", "progress", "stage_completed", "result_available"]
    assert events[1].percent is None and events[1].completed_stage is None
    assert events[2].completed_stage == PipelineStage.PREPARED
    assert events[2].payload["completed_stages"] == ("prepared",)
    with pytest.raises(TypeError):
        events[2].payload["percent"] = 100


def test_cancellation_before_execution_does_not_write_outputs(monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setattr(DesignService, "_run", run)
    events = []
    with pytest.raises(DesignCancelled):
        DesignService.run(ProjectDocument.new(), tmp_path / "output", progress=events.append, cancel_check=lambda: True)
    run.assert_not_called()
    assert not (tmp_path / "output").exists()
    assert [event.kind for event in events] == ["started", "cancelled"]


def test_engine_cancellation_is_reported_and_never_converted_into_success(monkeypatch, tmp_path):
    calls = iter([False, True])
    monkeypatch.setattr(DesignService, "_run", lambda *args, **kwargs: ProgressReporter().report(0, "Computing"))
    events = []
    with pytest.raises(DesignCancelled):
        DesignService.run(ProjectDocument.new(), tmp_path, progress=events.append, cancel_check=lambda: next(calls))
    assert [event.kind for event in events] == ["started", "cancelled"]


def test_observer_errors_do_not_change_a_completed_result_or_mask_original_failure(monkeypatch, tmp_path, caplog):
    def observer(event):
        raise RuntimeError("Display disconnected")

    monkeypatch.setattr(DesignService, "_run", lambda *args, **kwargs: {"ok": True})
    assert DesignService.run(ProjectDocument.new(), tmp_path, progress=observer) == {"ok": True}
    assert "Display disconnected" in caplog.text
    failure = ValueError("Original design failure")
    monkeypatch.setattr(DesignService, "_run", Mock(side_effect=failure))
    with pytest.raises(ValueError) as caught:
        DesignService.run(ProjectDocument.new(), tmp_path, progress=observer)
    assert caught.value is failure


def test_event_construction_does_not_parse_activity_as_a_completed_stage():
    event = ExecutionEvent.from_progress({"message": "prepared complete", "percent": 100})
    assert event.kind == "progress" and event.completed_stage is None


@pytest.mark.parametrize("stage", ["prepared", "sheet"])
def test_service_and_typed_observers_preserve_direct_pipeline_scientific_arrays(tmp_path, stage):
    from coilgen.pipeline import CoilGenPipeline
    from test_pipeline_stages import config

    project = ProjectDocument.from_config(config()).update_study({"pipeline_stage": stage})
    expected = CoilGenPipeline(project.config).run_stage(stage).scientific_arrays()
    events = []
    directory = tmp_path / stage
    DesignService.run(project, directory, threads=1, progress=events.append, cancel_check=lambda: False)
    record = DesignService.load_run(directory)
    for name in (
        "vertices_m",
        "faces",
        "target_points_m",
        "target_field_T",
        "target_weights",
        "target_suppressed",
        "stream_function_A",
        "sheet_field_T",
    ):
        if name in expected:
            np.testing.assert_allclose(record.array(name), expected[name], rtol=1e-12, atol=1e-15)
    assert events[-1].kind == "result_available"
    assert events[-1].completed_stage is None
