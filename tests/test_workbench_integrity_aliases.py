"""Saved-source health survives run-link aliases and disappears only after revalidation."""

from unittest.mock import Mock

import pytest
import test_workbench_usability as usability
from test_workbench_execution_feedback import _record
from test_workbench_completed_run_loading import wait as wait_for

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.presets import preset_config, preset_study
from coilgen.application.service import RunRecord
from coilgen.gui.problems import ProblemsPanel


application = usability.application
windows = usability.windows


def _source(window, directory):
    record = _record(window, directory)
    metadata = record.metadata
    metadata.update(
        pipeline_stage="evaluated",
        artifacts=[{"name": "request.json", "sha256": "0" * 64}],
        constraints=[
            {
                "name": "field_error",
                "value": 1e-6,
                "limit": 1e-7,
                "passed": False,
                "scope": "continuous_sheet",
                "unit": "T",
            }
        ],
    )
    return RunRecord(record.directory, metadata, record.arrays)


def test_failed_source_is_shared_by_aliases_even_after_the_original_link_is_removed(windows, tmp_path):
    window = windows()
    record = _source(window, tmp_path / "saved-source")
    window.runs.update(job=record, history_alias=record)
    window._mark_run_verification_error("job", "Saved field checksum changed.")
    assert window._run_verification_error("history_alias") == "Saved field checksum changed."
    assert window._run_verification_error(record) == "Saved field checksum changed."
    del window.runs["job"]
    window._job_verification_errors.pop("job")
    window.show_run("history_alias")
    assert window._run_verification_error("history_alias") == "Saved field checksum changed."
    assert window._view_results_id() is None
    assert "verification failed" in window.design_status_label.text().lower()
    assert "verification failed" in window.result_state.text().lower()
    assert not window.acceptance_panel.settings.isEnabled()
    assert not window.acceptance_panel.edit_settings.isEnabled()
    assert not window.problems_panel.navigate.isEnabled()
    assert window._selected_resume_plan().available == ()
    with pytest.raises(ValueError, match="failed verification"):
        window.export_selected_geometry(tmp_path / "export", include_project=False)
    with pytest.raises(ValueError, match="failed verification"):
        window.export_run_artifacts(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="failed verification"):
        window.pin_design_baseline()
    with pytest.raises(ValueError, match="failed verification"):
        window.create_design_variant()
    assert window._portable_assembly().channel(window._active_channel_id).project.saved_runs == ()


def test_legacy_id_error_fixture_still_gates_every_alias(windows, tmp_path):
    window = windows()
    record = _source(window, tmp_path / "legacy-source")
    window.runs.update(job=record, history_alias=record)
    window._job_verification_errors["job"] = "Legacy integrity failure."
    assert window._run_verification_error(record) == "Legacy integrity failure."
    del window.runs["job"]
    window._job_verification_errors.pop("job")
    assert window._run_verification_error("history_alias") == "Legacy integrity failure."


def test_failed_explicit_load_retains_directory_health_without_a_job_or_run_id(windows, tmp_path):
    window = windows()
    directory = tmp_path / "unverified-source"
    window.runs["alias"] = _source(window, directory)
    window.service.load_run = Mock(side_effect=ValueError("Original request checksum changed."))
    with pytest.raises(ValueError, match="checksum"):
        window.load_run(directory, id_="failed-load")
    window._job_verification_errors.pop("failed-load")
    assert window._run_verification_error("alias") == "Original request checksum changed."


def test_successful_explicit_load_clears_only_the_verified_directory_health(windows, tmp_path):
    window = windows()
    record = _source(window, tmp_path / "reverified-source")
    other = _source(window, tmp_path / "still-failed-source")
    window.runs.update(job=record, history_alias=record, other=other)
    window._mark_run_verification_error("job", "Selected source failed.")
    window._job_verification_errors["history_alias"] = "Another link to the failed source."
    window._mark_run_verification_error("other", "Other source still failed.")
    window.service.load_run = Mock(return_value=record)
    window.load_run(record.directory, id_="reverified", select=False)
    assert window._run_verification_error("job") == ""
    assert window._run_verification_error("history_alias") == ""
    assert "job" not in window._job_verification_errors and "history_alias" not in window._job_verification_errors
    assert window._run_verification_error("other") == "Other source still failed."


def test_poll_failure_is_retained_for_a_later_history_alias(windows):
    window = windows()
    state = window.compute()
    window.service.load_run = Mock(side_effect=ValueError("Saved mesh checksum changed."))
    window.jobs.states[0].update(status="completed")
    window.poll_jobs()
    wait_for(lambda: not window._run_load_tasks)
    window.runs["later-alias"] = _source(window, window.jobs.directory / state["id"])
    window._job_verification_errors.pop(state["id"])
    assert window._run_verification_error("later-alias") == "ValueError: Saved mesh checksum changed."


@pytest.mark.parametrize("action", ["artifacts", "resume"])
def test_action_reverification_discovers_shared_source_failure(windows, tmp_path, action):
    window = windows()
    record = _source(window, tmp_path / "action-source")
    window.runs.update(selected=record, history_alias=record)
    window.show_run("selected")
    window.service.load_run = Mock(side_effect=ValueError("Saved source checksum changed."))
    if action == "artifacts":
        with pytest.raises(ValueError, match="checksum"):
            window.export_run_artifacts(tmp_path / "artifacts")
    else:
        assert window._selected_resume_plan(refresh=True).available == ()
    assert window._run_verification_error("history_alias") == "Saved source checksum changed."


def test_displayed_problem_navigation_is_disabled_when_source_later_fails(windows, tmp_path, monkeypatch):
    window = windows()
    record = _source(window, tmp_path / "displayed-source")
    window.runs["selected"] = record
    window.show_run("selected")
    assert window.problems_panel.navigate.isEnabled()
    set_run = Mock(wraps=window.problems_panel.set_run)
    monkeypatch.setattr(window.problems_panel, "set_run", set_run)
    window._mark_run_verification_error("selected", "Source checksum changed.")
    window._refresh_guidance()
    assert not window.problems_panel.navigate.isEnabled()
    assert "checksum" in window.problems_panel.navigate.toolTip()
    calls = set_run.call_count
    window._refresh_guidance()
    assert set_run.call_count == calls
    assert isinstance(window.problems_panel, ProblemsPanel)


def test_identical_independent_channel_fingerprint_never_claims_current_channel_result(windows, tmp_path):
    first = ProjectDocument.from_config(preset_config(), name="First channel").update_study(preset_study())
    second = first.duplicate("Second channel")
    assert first.fingerprint == second.fingerprint and first.id != second.id
    window = windows(ProjectAssembly.new("Independent copies", (first, second)))
    record = _source(window, tmp_path / "other-channel-source")
    metadata = record.metadata
    metadata.update(project_id=second.id, project_revision=second.revision)
    record = RunRecord(record.directory, metadata, record.arrays)
    window.runs["other-channel"] = record
    window.show_run("other-channel")
    assert window.result_state.text().startswith("Another channel")
    assert "Current model/study" not in window.result_state.text()
    assert not window.acceptance_panel.edit_settings.isEnabled()
