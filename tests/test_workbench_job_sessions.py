"""Job ownership and execution dates survive polling without rewriting evidence."""

from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
import test_workbench_usability as usability

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor, QPalette

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.reuse import ReusableStage, ReusePlan
from coilgen.pipeline_stages import PipelineStage
from coilgen.application.presets import preset_config, preset_study
from coilgen.gui import workbench
from coilgen.gui.colors import _contrast
from coilgen.gui.job_history import format_job_timestamp
from coilgen.gui.readiness import ReadinessReport


application = usability.application


class SessionService(usability.Service):
    def submit(self, project, jobs, **controls):
        state = super().submit(project, jobs, **controls)
        stamp = datetime.now(timezone.utc).isoformat()
        state.update(created_at=stamp, updated_at=stamp)
        jobs.states[-1].update(state)
        return state

    def submit_assembly(self, assembly, jobs):
        return {
            channel.id: self.submit(channel.project, jobs, pipeline_stage=channel.project.study["pipeline_stage"])
            for channel in assembly.channels
            if channel.enabled
        }


@pytest.fixture(autouse=True)
def ready_environment(monkeypatch):
    monkeypatch.setattr(workbench, "check_readiness", lambda *_args, **_kwargs: ReadinessReport())


@pytest.fixture
def session_windows(application, tmp_path):
    created = []

    def make(project=None, states=(), *, queue=None, lazy=False):
        project = project or ProjectDocument.from_config(preset_config()).update_study(preset_study())
        queue = queue or usability.Queue(tmp_path / f"jobs-{len(created)}")
        queue.states = deepcopy(list(states))
        queue.close = Mock()
        instance = workbench.CoilGenWorkbench(
            project,
            jobs=None if lazy else queue,
            jobs_directory=queue.directory,
            service=SessionService(),
            viewport_factory=usability.Preview,
        )
        instance.timer.stop()
        created.append(instance)
        return instance, queue

    yield make
    for instance in created:
        instance._draft_dirty = False
        instance.undo_stack.setClean()
        instance.close()
        instance.undo_stack.clear()
        instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _state(id_, created="2026-10-02T08:09:10+00:00", **changes):
    value = {
        "id": id_,
        "name": f"Trial {id_}",
        "status": "failed",
        "stage": "Stopped",
        "project_revision": 7,
        "progress": 0.2,
        "created_at": created,
        "updated_at": "2026-10-04T14:15:16+00:00",
    }
    value.update(changes)
    return value


def _column(window, header):
    return next(
        column
        for column in range(window.jobs_table.columnCount())
        if window.jobs_table.horizontalHeaderItem(column).text() == header
    )


def _ids(window):
    return [
        window.jobs_table.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in range(window.jobs_table.rowCount())
    ]


def _item(window, id_, header):
    return window.jobs_table.item(_ids(window).index(id_), _column(window, header))


def _filter(window, scope):
    index = window.jobs_filter.findData(scope)
    assert index >= 0
    window.jobs_filter.setCurrentIndex(index)


def test_launch_snapshot_keeps_earlier_active_and_failed_jobs_historical(session_windows):
    states = [_state("old-active", status="running"), _state("old-failed")]
    window, queue = session_windows(states=states)
    assert window._session_started_at.tzinfo is not None
    assert window._session_started_at.utcoffset().total_seconds() == 0
    assert window._previous_job_ids == {"old-active", "old-failed"}
    assert not window._session_job_ids
    for state in states:
        assert _item(window, state["id"], "Session").text() == "Earlier session"
    assert "Window opened" in window.jobs_context.text()
    assert format_job_timestamp(window._session_started_at.isoformat()).text in window.jobs_context.text()
    assert queue.states == states


def test_submission_date_is_distinct_from_today_updated_at_and_revision(session_windows):
    states = [_state("two-days-old"), _state("one-day-old", "2026-10-03T08:09:10+00:00")]
    window, queue = session_windows(states=states)
    submitted = _column(window, "Submitted (local)")
    for state in states:
        item = _item(window, state["id"], "Submitted (local)")
        expected = format_job_timestamp(state["created_at"])
        assert item.text() == expected.text
        assert state["created_at"] in item.toolTip()
        assert "UTC:" in item.toolTip()
        assert _item(window, state["id"], "Revision").text() == "7"
    assert window.jobs_table.item(0, submitted).text() != window.jobs_table.item(1, submitted).text()
    assert queue.states == states


def test_exact_utc_instants_control_date_order_with_unknown_timestamps_last(session_windows):
    window, _ = session_windows(
        states=(
            _state("unknown", None),
            _state("older", "2026-10-04T08:00:00+05:30"),
            _state("newer", "2026-10-04T02:30:01Z"),
            _state("bad", "bad timestamp"),
        )
    )
    ids = _ids(window)
    assert ids.index("newer") < ids.index("older")
    assert max(ids.index("older"), ids.index("newer")) < min(ids.index("unknown"), ids.index("bad"))
    assert _item(window, "unknown", "Submitted (local)").text() == "Not recorded"
    assert _item(window, "bad", "Elapsed").text() == "—"


def test_successful_submission_and_cached_resume_are_current_session(session_windows, tmp_path):
    window, queue = session_windows(states=(_state("historical"),))
    window._refresh_stage_status = Mock()
    checkpoint = tmp_path / "saved-sheet.cgstage"
    window._reuse_plan = ReusePlan((ReusableStage(PipelineStage.SHEET, checkpoint, checkpoint.parent),), (), ())
    state = window.compute()
    assert state["id"] in window._session_job_ids
    assert window.service.submissions[-1][1]["resume_checkpoint"] == str(checkpoint)
    assert _item(window, state["id"], "Session").text() == "This session"
    assert _item(window, "historical", "Session").text() == "Earlier session"
    assert queue.states[0]["created_at"] == "2026-10-02T08:09:10+00:00"


def test_reusing_matching_earlier_active_job_does_not_claim_its_submission(session_windows):
    project = ProjectDocument.from_config(preset_config()).update_study(preset_study())
    prior = _state(
        "already-running",
        status="running",
        project_id=project.id,
        project_fingerprint=project.fingerprint,
        project_revision=project.revision,
        **project.study,
    )
    window, _ = session_windows(project, (prior,))
    state = window.compute()
    assert state["id"] == prior["id"]
    assert not window.service.submissions
    assert not window._session_job_ids
    assert _item(window, prior["id"], "Session").text() == "Earlier session"


@pytest.mark.parametrize("difference", ["missing_revision", "missing_backend", "different_validation"])
def test_unproven_or_different_earlier_active_request_does_not_replace_new_submission(session_windows, difference):
    project = ProjectDocument.from_config(preset_config()).update_study(preset_study())
    prior = _state(
        "earlier-request",
        status="running",
        project_id=project.id,
        project_fingerprint=project.fingerprint,
        project_revision=project.revision,
        **project.study,
    )
    if difference == "missing_revision":
        prior.pop("project_revision")
    elif difference == "missing_backend":
        prior.pop("kernel_backend")
    else:
        prior["validation_resolution"] += 1
    window, _ = session_windows(project, (prior,))
    state = window.compute()
    assert state["id"] != prior["id"]
    assert len(window.service.submissions) == 1
    assert window._session_job_ids == {state["id"]}
    assert _item(window, prior["id"], "Session").text() == "Earlier session"


def test_all_independent_channel_submissions_belong_to_current_session(session_windows):
    projects = tuple(
        ProjectDocument.from_config(preset_config(), name=f"Channel {index}").update_study(preset_study())
        for index in range(2)
    )
    assembly = ProjectAssembly.new("Independent studies", projects)
    window, _ = session_windows(assembly, (_state("historical"),))
    states = window.compute_channels()
    assert len(states) == 2
    ids = {state["id"] for state in states.values()}
    assert ids == window._session_job_ids
    assert all(_item(window, id_, "Session").text() == "This session" for id_ in ids)


def test_lazy_runtime_snapshots_persisted_ids_before_submitting(session_windows, monkeypatch):
    window, queue = session_windows(states=(_state("persisted"),), lazy=True)
    monkeypatch.setattr(workbench, "JobManager", lambda *_args, **_kwargs: queue)
    state = window.compute()
    assert window._previous_job_ids == {"persisted"}
    assert window._session_job_ids == {state["id"]}
    assert _item(window, "persisted", "Session").text() == "Earlier session"


def test_external_jobs_are_distinct_and_historical_filter_keeps_both_sources(session_windows):
    window, queue = session_windows(states=(_state("persisted"),))
    own = window.compute()
    queue.states.append(_state("external", "2026-10-04T20:00:00+00:00"))
    window.poll_jobs()
    assert _item(window, "external", "Session").text() == "Other session"
    for count in ("This session: 1", "Earlier: 1", "Other: 1"):
        assert count in window.jobs_context.text()
    _filter(window, "previous")
    assert set(_ids(window)) == {"persisted", "external"}
    _filter(window, "current")
    assert _ids(window) == [own["id"]]
    assert "Earlier: 1" in window.jobs_context.text()
    assert "Other: 1" in window.jobs_context.text()
    _filter(window, "all")
    assert set(_ids(window)) == {"persisted", "external", own["id"]}
    assert set(window.states) == set(_ids(window))
    assert len(queue.states) == 3


def test_active_jobs_sort_first_then_current_session_then_historical_dates(session_windows):
    window, queue = session_windows(states=(_state("old-active", status="running"), _state("old-terminal")))
    own = window.compute()
    queue.states[-1].update(status="failed")
    queue.states.append(_state("later-external", "2026-10-06T12:00:00Z"))
    window.poll_jobs()
    assert _ids(window)[0] == "old-active"
    assert _ids(window)[1] == own["id"]
    assert _ids(window).index("later-external") < _ids(window).index("old-terminal")


def test_poll_and_sort_preserve_selected_job_identity_and_filter_is_a_view(session_windows):
    window, queue = session_windows(states=(_state("first"), _state("second", "2026-10-03T10:00:00Z")))
    window._focus_job("first")
    assert window._selected_job_id() == "first"
    before = deepcopy(window.states)
    queue.states.reverse()
    queue.states.append(_state("new-external", status="running"))
    window.poll_jobs()
    assert window._selected_job_id() == "first"
    _filter(window, "previous")
    assert window._selected_job_id() == "first"
    _filter(window, "current")
    assert not _ids(window)
    assert set(window.states) == {"first", "second", "new-external"}
    assert all(window.states[id_] == state for id_, state in before.items())
    _filter(window, "all")
    window._focus_job("first")
    window.poll_jobs()
    assert window._selected_job_id() == "first"


def test_reopening_window_resets_session_ownership_without_retiming_jobs(session_windows):
    original, queue = session_windows(states=(_state("old"),))
    own = original.compute()
    created = queue.states[-1]["created_at"]
    reopened, _ = session_windows(original.project, queue.states)
    assert not reopened._session_job_ids
    assert reopened._previous_job_ids == {"old", own["id"]}
    assert _item(reopened, own["id"], "Session").text() == "Earlier session"
    assert _item(reopened, own["id"], "Submitted (local)").text() == format_job_timestamp(created).text


@pytest.mark.parametrize("dark", (False, True))
def test_native_1280_layout_keeps_session_date_and_execution_readable(session_windows, application, dark):
    saved = QPalette(application.palette())
    palette = QPalette(saved)
    base = QColor("#202124" if dark else "#ffffff")
    text = QColor("#f1f3f4" if dark else "#202124")
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.AlternateBase):
        palette.setColor(role, base)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text):
        palette.setColor(role, text)
    application.setPalette(palette)
    try:
        window, _ = session_windows(states=(_state("historical"),))
        window.compute()
        window.resize(1280, 900)
        window.show()
        window._show_dock("evidence_dock")
        window.evidence.setCurrentWidget(window.jobs_table)
        application.processEvents()
        window.poll_jobs()
        header = window.jobs_table.horizontalHeader()
        viewport = window.jobs_table.viewport().width()
        for title in ("Session", "Submitted (local)", "Execution"):
            column = _column(window, title)
            assert header.sectionViewportPosition(column) >= 0
            assert header.sectionViewportPosition(column) + header.sectionSize(column) <= viewport
            for row in range(window.jobs_table.rowCount()):
                item = window.jobs_table.item(row, column)
                brush = item.foreground()
                foreground = (
                    window.jobs_table.palette().color(QPalette.ColorRole.Text)
                    if brush.style() == Qt.BrushStyle.NoBrush
                    else brush.color()
                )
                assert _contrast(foreground, window.jobs_table.palette().color(QPalette.ColorRole.Base)) >= 4.5
    finally:
        application.setPalette(saved)
