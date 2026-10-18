"""Capabilities and visible workflow state agree after model/job transitions."""

from unittest.mock import Mock

import test_workbench_usability as usability
import test_stopped_checkpoint_recovery as recovery
from test_workbench_completed_run_loading import wait as wait_for

from coilgen.application.reuse import ReusableStage, ReusePlan
from coilgen.application.service import RunRecord
from coilgen.pipeline_stages import PipelineStage


application = usability.application
windows = usability.windows
stopped = recovery.stopped


def record(window, state):
    return RunRecord(
        window.jobs_directory / state["id"],
        {**state, "summary": {"accepted": True}, "warnings": [], "constraints": [], "artifacts": []},
        {},
    )


def test_old_completion_preserves_applied_new_revision_plot(windows):
    window = windows()
    state = window.compute()
    saved = record(window, state)
    window.apply_config({"target_gradient_strength": 0.031})
    assert window.live_preview.job_id is None
    window.service.load_run = Mock(return_value=saved)
    window.jobs.states[0].update(status="completed", stage="Complete", progress=1.0)
    window.poll_jobs()
    wait_for(lambda: not window._run_load_tasks)
    assert state["id"] in window.runs
    assert window._selected_run is None
    assert window.viewport.displayed is None


def test_prior_failed_job_does_not_override_successful_cached_boundary(windows, tmp_path):
    window = windows()
    state = window.compute()
    window.live_preview.stop()
    window.states = {
        state["id"]: {**state, "pipeline_stage": "sheet", "status": "failed"},
        "retry": {**state, "id": "retry", "pipeline_stage": "sheet", "status": "completed"},
    }
    window._reuse_plan = ReusePlan((ReusableStage(PipelineStage.SHEET, tmp_path / "sheet.cgstage", tmp_path),), (), ())
    window._refresh_stage_status(refresh_reuse=False)
    assert window.stage_items["sheet"].text(1) == "Cached"


def test_downstream_edit_shows_verified_reusable_stages_without_rechecking_archives(windows, stopped, monkeypatch):
    project, state, root, _directory = stopped
    window = windows(project, directory=root)
    window.jobs.states.append(state)
    window.poll_jobs()
    wait_for(lambda: not window._stopped_reuse_tasks)
    window.apply_config({"levels": project.config.levels + 1})
    wait_for(lambda: not window._stopped_reuse_tasks)
    assert window.design_progress.snapshot.stages[0].status == "reusable"
    assert window.design_progress.snapshot.stages[1].status == "reusable"
    assert window.design_progress.snapshot.stages[2].status == "waiting"
    assert "verified checkpoints reusable" in window.design_progress.snapshot.activity

    import coilgen.gui.workbench as workbench

    monkeypatch.setattr(workbench, "plan_reuse", Mock(side_effect=AssertionError("Repeated archive verification")))
    monkeypatch.setattr(
        workbench, "plan_stopped_reuse", Mock(side_effect=AssertionError("Repeated archive verification"))
    )
    window._refresh_design_progress()
    assert window.design_progress.snapshot.stages[0].status == "reusable"


def test_known_integrity_failure_remains_failed_with_cached_arrays(windows):
    window = windows()
    state = window.compute()
    window.service.load_run = Mock(return_value=record(window, state))
    window.jobs.states[0].update(status="completed", stage="Complete", progress=1.0)
    window.poll_jobs()
    wait_for(lambda: not window._run_load_tasks)
    window.service.load_run.side_effect = ValueError("Artifact checksum changed")
    assert not window.view_results()
    assert "verification failed" in window.execution_label.text()
    assert "verification failed" in window.design_progress.snapshot.activity
    assert all(stage.status == "complete" for stage in window.design_progress.snapshot.stages)
    assert not window.resume_action.isEnabled()


def test_resume_requires_actual_verified_checkpoint(windows, prepared_run):
    window = windows()
    empty = record(window, {"id": "without-checkpoint", "pipeline_stage": "prepared"})
    window.runs["without-checkpoint"] = empty
    window.show_run("without-checkpoint")
    assert not window.resume_action.isEnabled()
    assert "checkpoint" in window.resume_action.toolTip()
    project, saved = prepared_run
    window._set_project(project)
    window.runs["prepared-run"] = saved
    window.show_run("prepared-run")
    assert window.resume_action.isEnabled()
    assert "prepared" in window.resume_action.text()


def test_stopped_job_can_resume_verified_boundary_without_creating_result(windows, stopped):
    project, state, root, directory = stopped
    window = windows(project, directory=root)
    window.jobs.states.append(state)
    window.poll_jobs()
    window._focus_job(state["id"], open_window=False)
    wait_for(lambda: not window._stopped_reuse_tasks)
    assert not window.runs
    assert window._selected_run is None
    assert window.resume_action.isEnabled()
    assert window.resume_action.text() == "Resume from sheet"
    submitted = window.resume_selected_run()
    assert submitted is not None
    assert window.service.submissions[-1][1]["resume_checkpoint"] == str(directory / "checkpoints/sheet.cgstage")
    assert not window.runs


prepared_run = usability.prepared_run
