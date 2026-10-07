"""Qt commits and displays the same immutable application records as scripts."""

import pytest
from coilgen.application import DesignService, RunRecord
from coilgen.research.jobs import JobManager
from test_workbench_usability import application as application, windows as windows
from test_workbench_execution_feedback import _record


def test_combined_model_and_study_edit_is_one_revision_and_one_undo_step(windows):
    window = windows()
    before = window.project
    count = window.undo_stack.count()
    assert window.apply_inputs({"levels": before.config.levels + 1}, {"validation_resolution": 16})
    assert window.project.revision == before.revision + 1
    assert window.project.config.levels == before.config.levels + 1
    assert window.project.study["validation_resolution"] == 16
    assert window.undo_stack.count() == count + 1
    window.undo_stack.undo()
    assert window.project is before


@pytest.mark.parametrize("study", [{"validation_resolution": 1}, {"design_margin": 2}])
def test_rejected_combined_edit_leaves_model_study_and_undo_unchanged(windows, study):
    window = windows()
    before = window.project
    count = window.undo_stack.count()
    assert not window.apply_inputs({"levels": before.config.levels + 1}, study)
    assert window.project is before
    assert window.undo_stack.count() == count


def test_display_does_not_request_a_full_detached_array_copy(windows, tmp_path, monkeypatch):
    window = windows()
    record = _record(window, tmp_path / "saved")
    window.runs["saved"] = record

    def copied(_record):
        raise AssertionError("Display requested a detached copy of every scientific array")

    monkeypatch.setattr(RunRecord, "arrays", property(copied))
    window.show_run("saved")
    assert window.viewport.displayed == record.metadata


@pytest.mark.parametrize("changed_checkpoint", [False, True])
def test_gui_uses_runtime_request_digest_for_active_reuse(windows, tmp_path, monkeypatch, changed_checkpoint):
    # Queue admission checks bytes; checkpoint semantics remain the worker's
    # responsibility. Prevent worker execution in this admission regression.
    monkeypatch.setattr(JobManager, "_monitor", lambda _self: None)
    checkpoint = tmp_path / "first.cgstage"
    checkpoint.write_bytes(b"first checkpoint contents")
    other = tmp_path / "second.cgstage"
    other.write_bytes(b"different checkpoint contents")
    window = windows()
    with JobManager(tmp_path / "runtime") as jobs:
        window.jobs, window.jobs_directory = jobs, jobs.directory
        window.service = DesignService()
        window._runtime_snapshot_taken = False
        existing = DesignService.submit(
            window.project, jobs, **window._request_controls(), resume_checkpoint=str(checkpoint)
        )
        selected = other if changed_checkpoint else checkpoint
        state = window.compute(resume_checkpoint=str(selected))
        assert state is not None
        assert (state["id"] != existing["id"]) == changed_checkpoint
        assert len(jobs.list()) == (2 if changed_checkpoint else 1)
        assert (state["id"] in window._session_job_ids) == changed_checkpoint
