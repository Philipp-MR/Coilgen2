"""The design plan keeps channel evidence and user-controlled views separate."""

from dataclasses import replace

import pytest
import test_workbench_usability as usability

from PySide6.QtCore import Qt

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.assembly import ChannelDocument
from coilgen.application.bundles import RunArchive
from coilgen.application.presets import preset_config
from coilgen.application.service import RunRecord


application = usability.application
windows = usability.windows
prepared_run = usability.prepared_run

_STAGES = ("prepared", "sheet", "contours", "routed", "evaluated")


def _assembly():
    project = ProjectDocument.from_config(preset_config(), name="Gx").update_study(
        {"pipeline_stage": "evaluated", "fabricate": False}
    )
    return replace(
        ProjectAssembly.new("Independent channel feedback"),
        channels=tuple(
            ChannelDocument(axis, name, owned)
            for axis, name, owned in (
                ("gx", "Gx", project),
                ("gy", "Gy", project.duplicate("Gy")),
                ("gz", "Gz", project.duplicate("Gz")),
            )
        ),
    )


def _state(channel, status, *, completed=(), progress=None, active_stage=None):
    project = channel.project
    return {
        "id": f"job_{channel.id}",
        "name": channel.name,
        "status": status,
        "pipeline_stage": project.study["pipeline_stage"],
        "project_id": project.id,
        "project_fingerprint": project.fingerprint,
        "project_model_fingerprint": project.model_fingerprint,
        "project_revision": project.revision,
        "created_at": "2026-10-09T12:00:00+00:00",
        "updated_at": "2026-10-09T12:00:30+00:00",
        "completed_stages": list(completed),
        "progress": progress,
        "active_stage": active_stage,
        "stage": "Current numerical operation",
    }


def _record(window, state):
    return RunRecord(
        window.jobs_directory / state["id"],
        {
            **{
                key: state[key]
                for key in (
                    "name",
                    "pipeline_stage",
                    "project_id",
                    "project_fingerprint",
                    "project_model_fingerprint",
                    "project_revision",
                )
            },
            "job": dict(state),
            "summary": {"scope": "evaluated", "accepted": False},
            "constraints": [],
            "wire_fields": {},
            "fabrication": None,
            "warnings": [],
            "artifacts": [],
        },
        {},
    )


def _poll(window, states):
    window.jobs.states = states
    for state in states:
        if state["status"] == "completed":
            window.runs[state["id"]] = _record(window, state)
            window._loaded_ids.add(state["id"])
    window.poll_jobs()


def test_follow_indicated_worker_selects_its_channel_without_opening_settings_or_submitting(windows):
    window = windows(_assembly())
    channel = window.assembly.channels[1]
    state = _state(channel, "running", completed=("prepared",), progress=0.42, active_stage="sheet")
    _poll(window, [state])
    settings = window.findChild(type(window.design_plan_dock), "settings_dock")
    settings.hide()
    assert window.design_progress.worker_channel.channel_id == channel.id
    assert window.live_preview.job_id is None
    assert not window.design_progress.follow_button.isChecked()

    window._set_live_follow(True)

    assert window._active_channel_id == channel.id
    assert window.live_preview.job_id == state["id"]
    assert window.live_preview.following
    assert window.design_progress.follow_button.isChecked()
    assert settings.isHidden()
    assert not window.service.submissions
    window._set_live_follow(False)
    assert not window.live_preview.following
    assert window.live_preview.job_id == state["id"]


def test_follow_other_channel_retains_invalid_model_draft(windows, monkeypatch):
    window = windows(_assembly())
    active = window._active_channel_id
    state = _state(window.assembly.channels[1], "running", progress=0.42)
    _poll(window, [state])
    monkeypatch.setattr(window, "apply_settings", lambda: False)

    window._set_live_follow(True)

    assert window._active_channel_id == active
    assert window.channel_selector.currentData() == active
    assert window.live_preview.job_id is None
    assert not window.design_progress.follow_button.isChecked()
    assert not window.service.submissions


def test_rejected_saved_source_without_runtime_job_is_flagged_in_channel_plan(windows):
    window = windows(_assembly())
    channel = window.assembly.channels[0]
    state = _state(channel, "completed", completed=_STAGES, progress=1)
    record = _record(window, state)
    window.runs["saved-run-alias"] = record
    window._mark_run_verification_error("saved-run-alias", "Saved checksum changed", directory=record.directory)

    window._refresh_design_plan()

    plan = window.design_progress.plan.channels[0]
    assert plan.verification == "failed"
    assert "verification failed" in plan.activity.lower()
    assert not any(stage.status == "complete" for stage in plan.stages)


def _channels(window):
    return {channel.channel_id: channel for channel in window.design_progress.plan.channels}


def test_plan_is_a_right_dock_with_persisted_user_visibility(windows):
    window = windows(_assembly())
    dock = window.design_plan_dock
    assert dock.objectName() == "design_plan_dock"
    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert dock.isAncestorOf(window.design_progress)
    assert not dock.isHidden()
    assert window._workbench_view_state()["design_plan_panel_visible"] is True

    dock.close()
    hidden = window._workbench_view_state()
    assert hidden["design_plan_panel_visible"] is False
    window.poll_jobs()
    assert dock.isHidden()

    restored = windows(_assembly())
    restored._restore_panel_layout(hidden)
    assert restored.design_plan_dock.isHidden()
    restored.poll_jobs()
    assert restored.design_plan_dock.isHidden()
    restored.design_plan_dock.toggleViewAction().trigger()
    assert not restored.design_plan_dock.isHidden()


def test_submit_opens_plan_without_opening_jobs_and_poll_respects_close(windows):
    window = windows()
    window.design_plan_dock.close()
    assert not window.evidence.any_open()

    state = window.compute()

    assert state is not None
    assert not window.design_plan_dock.isHidden()
    assert not window.evidence.any_open()
    window.design_plan_dock.close()
    window.jobs.states[0].update(status="running", active_stage="sheet", completed_stages=["prepared"], progress=0.42)
    window.poll_jobs()
    assert window.design_plan_dock.isHidden()
    assert not window.evidence.any_open()


def test_system_plan_shows_each_job_after_channel_switch_and_job_selection(windows):
    assembly = _assembly()
    assert len({channel.project.fingerprint for channel in assembly.channels}) == 1
    window = windows(assembly)
    states = [
        _state(assembly.channel("gx"), "completed", completed=_STAGES, progress=1),
        _state(assembly.channel("gy"), "running", completed=("prepared",), progress=0.42, active_stage="sheet"),
        _state(assembly.channel("gz"), "queued", progress=0),
    ]
    _poll(window, states)
    plan = window.design_progress.plan
    channels = _channels(window)
    assert [channels[key].status for key in ("gx", "gy", "gz")] == ["completed", "running", "queued"]
    assert channels["gx"].verification == "verified"
    assert all(stage.status == "complete" for stage in channels["gx"].stages)
    assert channels["gy"].percent == 42
    assert [stage.status for stage in channels["gy"].stages] == ["complete", "current", "waiting", "waiting", "waiting"]
    assert not channels["gy"].stages[1].inferred
    assert all(stage.status == "waiting" for stage in channels["gz"].stages)
    assert plan.completed_channels == 1 and plan.running_channels == 1 and plan.queued_channels == 1

    window.channel_selector.setCurrentIndex(window.channel_selector.findData("gy"))
    assert window._active_channel_id == "gy"
    assert _channels(window)["gy"].selected
    assert _channels(window)["gy"].percent == 42
    before = [(channel.channel_id, channel.status, channel.percent) for channel in window.design_progress.plan.channels]
    window._focus_job(states[0]["id"], open_window=False)
    after = [(channel.channel_id, channel.status, channel.percent) for channel in window.design_progress.plan.channels]
    assert before == after
    assert window._active_channel_id == "gy"
    assert not window.service.submissions


def test_disabled_channel_and_different_study_boundaries_keep_their_own_plans(windows):
    assembly = _assembly()
    gy = assembly.channel("gy")
    assembly = assembly.update_channel("gy", project=gy.project.update_study({"pipeline_stage": "sheet"}))
    assembly = assembly.update_channel("gz", enabled=False)
    window = windows(assembly)

    plan, channels = window.design_progress.plan, _channels(window)
    assert channels["gx"].requested_stage == "evaluated"
    assert channels["gy"].requested_stage == "sheet"
    assert [stage.status for stage in channels["gy"].stages] == ["waiting", "waiting", "skipped", "skipped", "skipped"]
    assert not channels["gz"].enabled
    assert "excluded" in channels["gz"].activity
    assert plan.requested_count == 7 and plan.enabled_channels == 2 and plan.disabled_channels == 1
    assert window.assembly == assembly


def test_applied_edit_marks_previous_completion_stale_without_erasing_other_channels(windows):
    assembly = _assembly()
    window = windows(assembly)
    states = [_state(channel, "completed", completed=_STAGES, progress=1) for channel in assembly.channels]
    _poll(window, states)
    assert window.design_progress.plan.completed_channels == 3

    original = window.project
    window._draft_controls["radius_mm"].setValue(172)
    assert window.apply_settings()

    channels = _channels(window)
    assert window.project.revision == original.revision + 1
    assert channels["gx"].status == "stale"
    assert channels["gx"].completed_count == 0
    assert all(stage.status == "stale" for stage in channels["gx"].stages)
    assert channels["gy"].completed_count == 5 and channels["gz"].completed_count == 5
    assert window.design_progress.plan.completed_channels == 2
    assert len(window.runs) == 3
    assert not window.service.submissions


def test_old_revision_and_other_owner_never_mark_a_current_live_stage_complete(windows):
    assembly = _assembly()
    window = windows(assembly)
    gx = assembly.channel("gx")
    old = _state(gx, "running", completed=("prepared", "sheet"), progress=0.81, active_stage="contours")
    old["project_revision"] -= 1
    foreign = {**old, "id": "other_owner", "project_id": "unrelated_project"}
    _poll(window, [old, foreign])

    channels = _channels(window)
    assert all(channel.completed_count == 0 for channel in channels.values())
    assert all(channel.percent is None for channel in channels.values())
    assert all(stage.status == "waiting" for channel in channels.values() for stage in channel.stages)
    assert len(window.states) == 2


def test_unvisited_channel_portable_evidence_is_shown_without_loading_arrays(windows, prepared_run):
    project, record = prepared_run
    archive = RunArchive.capture(record)
    saved = project.with_results((archive,))
    fresh = project.duplicate("Fresh independent channel")
    assembly = ProjectAssembly.new("Portable plan", (fresh, saved))
    window = windows(assembly)

    channels = _channels(window)
    current = assembly.channels[0].id
    archived = assembly.channels[1].id
    assert channels[current].completed_count == 0
    assert channels[archived].status == "completed"
    assert channels[archived].verification == "verified"
    assert channels[archived].completed_count == 1
    assert channels[archived].stages[0].status == "complete"
    assert window.project.id == fresh.id
    assert not any(run.metadata.get("project_id") == saved.id for run in window.runs.values())
    assert not window.service.submissions
    assert not window.jobs.states


@pytest.mark.parametrize(
    "status, stopped_stage", [("failed", "failed"), ("cancelled", "stopped"), ("interrupted", "stopped")]
)
def test_terminal_channel_retains_earlier_stages_and_leaves_unfinished_tail_grey(windows, status, stopped_stage):
    assembly = _assembly()
    window = windows(assembly)
    state = _state(
        assembly.channel("gy"), status, completed=("prepared", "sheet"), progress=0.45, active_stage="contours"
    )
    state["error"] = "Stopped during contour generation"
    _poll(window, [state])

    channel = _channels(window)["gy"]
    assert channel.status == status
    assert [stage.status for stage in channel.stages] == ["complete", "complete", stopped_stage, "waiting", "waiting"]
    assert channel.completed_count == 2
    assert channel.percent is None and not channel.busy
    assert window.design_progress.plan.completed_channels == 0


def test_runtime_failure_keeps_recorded_stages_without_claiming_live_work(windows, monkeypatch):
    assembly = _assembly()
    window = windows(assembly)
    _poll(
        window,
        [_state(assembly.channel("gy"), "running", completed=("prepared",), progress=0.42, active_stage="sheet")],
    )
    monkeypatch.setattr(window.jobs, "list", lambda: (_ for _ in ()).throw(OSError("Status unavailable")))
    window.poll_jobs()

    channel = _channels(window)["gy"]
    assert channel.status == "unavailable"
    assert channel.percent is None and not channel.busy
    assert [stage.status for stage in channel.stages] == ["complete", "waiting", "waiting", "waiting", "waiting"]
    assert "Status unavailable" in channel.tooltip


def test_system_submission_opens_plan_and_keeps_all_channels_visible(windows):
    assembly = _assembly()
    window = windows(assembly)
    window.service.submit_assembly = lambda supplied, jobs: {
        channel.id: window.service.submit(channel.project, jobs, **channel.project.study)
        for channel in supplied.channels
        if channel.enabled
    }
    window.design_plan_dock.close()

    submitted = window.compute_channels()

    assert set(submitted) == {"gx", "gy", "gz"}
    assert not window.design_plan_dock.isHidden()
    assert not window.evidence.any_open()
    assert len(window.service.submissions) == 3
    assert window.design_progress.plan.queued_channels == 3
    assert all(channel.status == "queued" for channel in _channels(window).values())
