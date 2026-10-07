"""Graphical progress reports recorded computation, never inferred success."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

from coilgen.gui.design_progress import DesignProgress, build_design_progress


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def progress(app):
    widget = DesignProgress()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def native_font(app):
    font_id = -1
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if not QFontDatabase.families() and font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
    font = QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9) if font_id >= 0 else app.font()
    yield font
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)


def _statuses(snapshot):
    return {stage.key: stage.status for stage in snapshot.stages}


def test_idle_has_requested_pending_stages_and_explicit_short_study_boundary():
    snapshot = build_design_progress("sheet")
    assert _statuses(snapshot) == {
        "prepared": "waiting",
        "sheet": "waiting",
        "contours": "skipped",
        "routed": "skipped",
        "evaluated": "skipped",
    }
    assert snapshot.percent is None
    assert snapshot.follow_available is False
    assert snapshot.busy is False
    assert "await computation" in snapshot.activity


def test_verified_reusable_boundaries_are_distinct_from_waiting_and_completed():
    snapshot = build_design_progress("evaluated", reusable=("prepared", "sheet"))
    assert [stage.status for stage in snapshot.stages] == ["reusable", "reusable", "waiting", "waiting", "waiting"]
    assert "verified checkpoints reusable" in snapshot.activity
    assert "not a newly completed stage" in snapshot.stages[1].tooltip
    assert not any(stage.viewed for stage in snapshot.stages)
    queued = build_design_progress("contours", {"status": "queued"}, reusable=("prepared", "sheet"))
    assert [stage.status for stage in queued.stages] == ["reusable", "reusable", "waiting", "skipped", "skipped"]
    completed = build_design_progress("sheet", completed=("prepared",), reusable=("prepared", "sheet", "contours"))
    assert [stage.status for stage in completed.stages] == ["complete", "reusable", "skipped", "skipped", "skipped"]


def test_cached_compatibility_applies_to_applied_inputs_when_drafts_are_pending():
    snapshot = build_design_progress("sheet", reusable=("prepared",), pending=True)
    assert "changes pending" in snapshot.activity
    assert "Pending drafts are not included" in snapshot.tooltip
    assert build_design_progress("sheet", reusable="prepared").stages[0].status == "waiting"


@pytest.mark.parametrize("status", ["queued", "running", "completed", "failed", "cancelled", "interrupted"])
def test_terminal_status_percent_and_free_text_cannot_verify_stage_completion(status):
    snapshot = build_design_progress(
        "evaluated", {"status": status, "stage": "Complete; all stages accepted", "progress": 1.0}
    )
    assert not any(stage.status == "complete" for stage in snapshot.stages)
    assert snapshot.percent == 100
    assert "does not verify stage completion" in snapshot.tooltip


def test_partial_authoritative_stages_do_not_fill_unrecorded_earlier_or_later_stages():
    snapshot = build_design_progress("evaluated", {"status": "running"}, completed=("sheet",))
    assert _statuses(snapshot) == {
        "prepared": "current",
        "sheet": "complete",
        "contours": "waiting",
        "routed": "waiting",
        "evaluated": "waiting",
    }
    assert snapshot.stages[0].inferred
    assert "Next requested stage without recorded completion" in snapshot.stages[0].tooltip
    assert "does not establish engineering acceptance" in snapshot.stages[1].tooltip
    assert "Optional live display availability and final result integrity" in snapshot.stages[1].tooltip
    assert "recorded computation" in snapshot.tooltip


def test_structured_operation_identity_governs_current_stage_without_marking_prior_complete():
    snapshot = build_design_progress("evaluated", {"status": "running"}, active_stage="routed")
    assert snapshot.stages[3].status == "current"
    assert snapshot.stages[3].inferred is False
    assert all(stage.status == "waiting" for stage in snapshot.stages[:3])


@pytest.mark.parametrize(
    "status, expected",
    [
        ("queued", "waiting"),
        ("running", "current"),
        ("cancelling", "current"),
        ("failed", "failed"),
        ("cancelled", "stopped"),
        ("interrupted", "stopped"),
    ],
)
def test_status_does_not_erase_verified_earlier_evidence(status, expected):
    snapshot = build_design_progress("contours", {"status": status}, completed=("prepared", "sheet"))
    assert [stage.status for stage in snapshot.stages] == ["complete", "complete", expected, "skipped", "skipped"]
    assert snapshot.follow_available == (status in {"queued", "running", "cancelling"})


def test_completed_execution_and_verified_completed_stages_are_distinct():
    worker_done = build_design_progress("sheet", {"status": "completed", "progress": 1})
    verifying = build_design_progress("sheet", {"status": "completed", "progress": 1}, verification="pending")
    verified = build_design_progress(
        "sheet", {"status": "completed"}, verification="verified", completed=("prepared", "sheet")
    )
    failed = build_design_progress("sheet", {"status": "completed"}, verification="failed")
    assert "stage evidence unverified" in worker_done.activity
    assert worker_done.tone == "warning"
    assert verifying.busy
    assert "verifying saved evidence" in verifying.activity
    assert verified.tone == "info"
    assert "review engineering checks" in verified.activity
    assert [stage.status for stage in verified.stages] == ["complete", "complete", "skipped", "skipped", "skipped"]
    assert failed.tone == "error"
    assert not any(stage.status == "complete" for stage in failed.stages)


@pytest.mark.parametrize("value", [None, "0.5", True, float("nan"), float("inf"), -0.1, 1.1, {}])
def test_unknown_or_invalid_legacy_percentage_remains_indeterminate_for_active_jobs(value):
    snapshot = build_design_progress("evaluated", {"status": "running", "progress": value})
    assert snapshot.percent is None
    assert snapshot.busy
    assert "worker activity unrecorded" in snapshot.activity


@pytest.mark.parametrize("value, expected", [(0, 0), (0.275, 28), (1, 100)])
def test_only_finite_fraction_progress_is_presented(value, expected):
    snapshot = build_design_progress("evaluated", {"status": "running", "progress": value})
    assert snapshot.percent == expected
    assert snapshot.busy is False


def test_provenance_pending_edits_and_plotted_stage_are_explicit_and_not_mutated():
    state = {
        "status": "running",
        "name": "Conical Z2",
        "project_revision": 7,
        "progress": 0.2,
        "stage": "Connecting loops",
    }
    before = dict(state)
    snapshot = build_design_progress(
        "evaluated",
        state,
        completed=("prepared", "sheet", "contours"),
        preview_stage="contours",
        pending=True,
        following=False,
    )
    assert snapshot.provenance == "Conical Z2 · revision 7"
    assert "model edits pending" in snapshot.activity
    assert "Plot following is paused" in snapshot.tooltip
    assert snapshot.stages[2].viewed
    assert "Currently shown in the plot" in snapshot.stages[2].tooltip
    assert not snapshot.stages[3].viewed
    assert state == before


def test_unknown_completed_stages_and_unverified_preview_do_not_create_evidence():
    snapshot = build_design_progress("sheet", completed=("invented", "evaluated", {}, None), preview_stage="prepared")
    assert [stage.status for stage in snapshot.stages][:2] == ["waiting", "waiting"]
    assert not any(stage.viewed for stage in snapshot.stages)
    assert not any(stage.status == "complete" for stage in build_design_progress("sheet", completed="prepared").stages)


def test_multiline_worker_errors_remain_one_compact_activity_row():
    snapshot = build_design_progress(
        "sheet", {"status": "failed", "name": "Shim\n system", "error": "Solver failed\n\nconditioning exceeds limit"}
    )
    assert "\n" not in snapshot.activity
    assert "conditioning exceeds limit" in snapshot.tooltip


@pytest.mark.parametrize("requested", ["invalid", None])
def test_unknown_requested_boundary_is_rejected(requested):
    with pytest.raises(ValueError, match="Unknown design stage"):
        build_design_progress(requested)


def test_unknown_verification_state_is_rejected():
    with pytest.raises(ValueError, match="Unknown evidence verification"):
        build_design_progress("sheet", verification="accepted")


def test_only_follow_is_interactive_and_programmatic_updates_do_not_emit(progress):
    events = []
    progress.follow_requested.connect(events.append)
    progress.set_state("evaluated", {"status": "running"}, following=True)
    assert not progress.follow_button.isHidden()
    assert progress.follow_button.isChecked()
    assert events == []
    progress.show()
    QTest.mouseClick(progress.follow_button, Qt.MouseButton.LeftButton)
    assert events == [False]
    assert progress.findChildren(QToolButton) == [progress.follow_button]
    assert not progress.findChildren(QPushButton)
    assert all(chip.focusPolicy() == Qt.FocusPolicy.NoFocus for chip in progress.chips.values())
    progress.set_state("sheet", {"status": "completed"}, completed=("prepared", "sheet"), verification="verified")
    assert progress.follow_button.isHidden()
    assert events == [False]


def test_widget_busy_and_known_progress_are_distinct(progress):
    progress.set_state("evaluated", {"status": "running", "progress": None})
    assert progress.progress_bar.maximum() == 0
    progress.set_state("evaluated", {"status": "running", "progress": 0.63})
    assert progress.progress_bar.maximum() == 100
    assert progress.progress_bar.value() == 63
    progress.set_state("evaluated", {"status": "completed", "progress": 1}, verification="pending")
    assert progress.progress_bar.maximum() == 0
    progress.set_state("evaluated")
    assert progress.progress_bar.isHidden()


def test_cached_chip_has_explicit_accessible_state_and_no_completed_badge(progress):
    progress.set_state("evaluated", reusable=("prepared", "sheet"))
    chip = progress.chips["sheet"]
    assert chip.property("stageStatus") == "reusable"
    assert "checkpoint available for reuse" in chip.accessibleName()
    assert "not a newly completed stage" in chip.toolTip()
    assert progress.chips["contours"].property("stageStatus") == "waiting"


@pytest.mark.parametrize("width", [400, 800])
def test_compact_layout_keeps_all_chips_and_follow_visible_without_overflow(app, progress, native_font, width):
    progress.setFont(native_font)
    progress.set_state(
        "evaluated",
        {
            "status": "running",
            "name": "A very long reproducible conical second-order shim design",
            "project_revision": 100,
            "progress": 0.23,
        },
        completed=("prepared", "sheet"),
        preview_stage="sheet",
    )
    progress.resize(width, progress.sizeHint().height())
    progress.show()
    app.processEvents()
    assert progress.minimumSizeHint().width() <= width
    assert progress.height() <= 90
    assert progress.follow_button.geometry().right() < width
    assert all(chip.geometry().right() < width and chip.width() >= 30 for chip in progress.chips.values())
    assert progress.activity_label.full_text in progress.activity_label.toolTip()
    assert progress.chips["sheet"].property("stageStatus") == "complete"
    assert progress.chips["sheet"].property("plottedStage") is True
    assert not progress.grab().isNull()
