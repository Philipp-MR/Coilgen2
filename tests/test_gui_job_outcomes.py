"""Observed execution outcomes remain actionable without claiming acceptance."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from coilgen.gui.job_outcomes import JobOutcomeNotice, build_job_outcome


def state(id_="gy-job", **changes):
    return {"id": id_, "status": "completed", "name": "Gradient Gy", "project_revision": 4, **changes}


def record(**changes):
    return SimpleNamespace(
        metadata={
            "pipeline_stage": "evaluated",
            "summary": {"accepted": True, "sheet_accepted": True, "scope": "fabrication_review"},
            "pipeline_sweeps": {"checks_passed": True},
            "fabrication": {"ready": True},
            "constraints": [],
            **changes,
        }
    )


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["coilgen-job-outcomes-test"])


@pytest.fixture
def native_font(application):
    original = QFont(application.font())
    path = Path("C:/Windows/Fonts/segoeui.ttf")
    font_id = QFontDatabase.addApplicationFont(str(path)) if path.is_file() else -1
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else ()
    font = QFont(families[0], 9) if families else original
    application.setFont(font)
    yield font
    application.setFont(original)
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)


@pytest.fixture
def notice(application, native_font):
    widget = JobOutcomeNotice()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("status", ("queued", "running", "cancelling", "unknown"))
def test_only_terminal_states_have_outcomes(status):
    assert build_job_outcome(state(status=status)) is None


def test_worker_failure_is_distinct_from_any_supplied_engineering_result():
    outcome = build_job_outcome(state(status="failed", error="Solver conditioning limit exceeded"), record())
    assert outcome.tone == "error"
    assert outcome.title.startswith("Execution failed")
    assert "Gradient Gy · revision 4" in outcome.title
    assert outcome.detail == "Solver conditioning limit exceeded"
    assert outcome.action_label == "Inspect failure"
    assert "checks passed" not in outcome.title


@pytest.mark.parametrize("status", ("cancelled", "interrupted"))
def test_stopped_execution_retains_only_recorded_stages_without_certifying_resume(status):
    outcome = build_job_outcome(
        state(status=status, completed_stages=["prepared", "sheet", "sheet", "invented", {}, None])
    )
    assert outcome.tone == "warning"
    assert outcome.action_label == "View job"
    assert "Earlier recorded stages retained: Prepare, Sheet." in outcome.detail
    assert "must be verified before resuming" in outcome.detail
    assert "Contours" not in outcome.detail


def test_completed_worker_without_verified_evidence_remains_explicitly_unverified():
    outcome = build_job_outcome(state())
    assert "evidence unverified" in outcome.title
    assert outcome.tone == "warning"
    assert outcome.action_label == "View job"
    assert "does not establish field or conductor acceptance" in outcome.detail


def test_verification_failure_takes_precedence_over_cached_success_metadata():
    outcome = build_job_outcome(state(), record(), "Saved artifact checksum changed")
    assert outcome.tone == "error"
    assert "Execution finished" in outcome.title
    assert "evidence verification failed" in outcome.title
    assert outcome.action_label == "Inspect failure"
    assert outcome.detail == "Saved artifact checksum changed"


def test_engineering_failures_count_exact_recorded_constraint_verdicts():
    outcome = build_job_outcome(
        state(),
        record(constraints=[{"passed": False}, {"passed": True}, {"passed": None}, {"passed": "false"}]),
    )
    assert "Execution finished · engineering checks failed" in outcome.title
    assert outcome.tone == "error"
    assert outcome.action_label == "Review checks"
    assert "1 failed, 2 unverified, 4 total" in outcome.detail
    assert "Saved sheet checks: Passed" in outcome.detail
    assert "pipeline conductor geometry: Passed" in outcome.detail
    assert "fabrication review: Passed" in outcome.detail
    assert "do not establish construction release" in outcome.detail


def test_failed_conductor_geometry_cannot_borrow_a_passing_sheet_or_invent_problem_count():
    outcome = build_job_outcome(state(), record(pipeline_sweeps={"checks_passed": False}))
    assert "engineering checks failed" in outcome.title
    assert outcome.tone == "error"
    assert "Saved sheet checks: Passed" in outcome.detail
    assert "pipeline conductor geometry: Failed" in outcome.detail
    assert "Recorded constraints" not in outcome.detail


def test_sparse_engineering_evidence_never_becomes_a_passed_result():
    outcome = build_job_outcome(state(), SimpleNamespace(metadata={}))
    assert outcome.tone == "warning"
    assert outcome.action_label == "Review checks"
    assert "engineering checks incomplete" in outcome.title
    assert "sheet checks: Unverified" in outcome.detail


def test_unverified_constraint_remains_visible_even_when_other_scopes_pass():
    outcome = build_job_outcome(state(), record(constraints=[{"passed": None}]))
    assert outcome.tone == "warning"
    assert "engineering checks incomplete" in outcome.title
    assert "0 failed, 1 unverified, 1 total" in outcome.detail


def test_verified_partial_study_finishes_without_claiming_downstream_acceptance():
    outcome = build_job_outcome(
        state(), record(pipeline_stage="prepared", summary={}, pipeline_sweeps={}, fabrication=None)
    )
    assert "Prepare study" in outcome.title
    assert outcome.tone == "info"
    assert outcome.action_label == "View result"
    assert "sheet checks: Unverified" in outcome.detail


def test_known_saved_passes_remain_scoped_and_keep_channel_provenance():
    outcome = build_job_outcome(state(channel_name="Gx", name="Trial", project_revision=7), record())
    assert "saved checks passed in scope" in outcome.title
    assert "Gx · Trial · revision 7" in outcome.title
    assert outcome.action_label == "View result"
    assert "do not establish construction release" in outcome.detail


@pytest.mark.parametrize("revision", (None, True, "4", -1))
def test_invalid_revision_is_explicitly_unrecorded(revision):
    assert "revision unrecorded" in build_job_outcome(state(project_revision=revision)).title


def test_terminal_notice_requires_its_actual_job_id():
    with pytest.raises(ValueError, match="recorded job ID"):
        build_job_outcome(state(id_=None))


def test_publishing_and_updating_notice_never_emit_a_navigation_action(notice):
    actions = []
    notice.action_requested.connect(actions.append)
    first = build_job_outcome(state())
    notice.publish(first)
    assert not notice.isHidden()
    assert notice.selector.isHidden()
    assert notice.selected == first
    updated = build_job_outcome(state(), record(), "Checksum failure")
    notice.publish(updated)
    assert notice.outcomes == (updated,)
    assert notice.label.full_text == updated.title
    assert notice.label.property("semanticTone") == "error"
    assert not actions
    notice.action_button.click()
    assert actions == ["gy-job"]


def test_later_completion_cannot_hide_an_unresolved_failure_and_each_outcome_is_selectable(notice):
    error = build_job_outcome(state("failed-job", status="failed", error="No convergence"))
    success = build_job_outcome(state("completed-job"), record())
    notice.publish(error)
    notice.publish(success)
    assert not notice.selector.isHidden()
    assert notice.selected == error
    notice.selector.setCurrentIndex(notice.selector.findData("completed-job"))
    assert notice.selected == success
    actions = []
    notice.action_requested.connect(actions.append)
    notice.action_button.click()
    assert actions == ["completed-job"]
    notice.publish(build_job_outcome(state("another-success"), record()))
    assert notice.selected == success
    assert error in notice.outcomes


def test_same_or_less_severe_outcome_updates_preserve_deliberate_selection(notice):
    first = build_job_outcome(state("first"), record())
    second = build_job_outcome(state("second"), record())
    notice.publish(first)
    notice.publish(second)
    notice.selector.setCurrentIndex(notice.selector.findData("second"))
    notice.publish(first)
    assert notice.selected == second
    updated_second = build_job_outcome(state("second", name="Renamed result"), record())
    notice.publish(updated_second)
    assert notice.selected == updated_second
    notice.publish(build_job_outcome(state("third"), record()))
    assert notice.selected == updated_second
    warning = build_job_outcome(state("stopped", status="cancelled"))
    notice.publish(warning)
    assert notice.selected == warning
    error = build_job_outcome(state("failure", status="failed"))
    notice.publish(error)
    assert notice.selected == error
    notice.publish(build_job_outcome(state("other-failure", status="failed")))
    assert notice.selected == error


def test_dismissal_is_per_job_and_does_not_reappear_on_an_update(notice):
    error = build_job_outcome(state("failed-job", status="failed"))
    success = build_job_outcome(state("completed-job"), record())
    notice.publish(error)
    notice.publish(success)
    notice.dismiss_button.click()
    assert notice.outcomes == (success,)
    assert notice.selected == success
    notice.publish(error)
    assert notice.outcomes == (success,)
    notice.dismiss_selected()
    assert notice.isHidden()
    assert not notice.outcomes
    notice.clear()
    notice.publish(error)
    assert notice.selected == error


def test_notice_remains_persistent_without_a_timer_and_preserves_full_text_in_tooltip(notice, application):
    outcome = build_job_outcome(state(name="Long design name " * 30), record())
    notice.publish(outcome)
    notice.resize(420, notice.sizeHint().height())
    notice.show()
    application.processEvents()
    assert not notice.findChildren(QTimer)
    assert notice.minimumSizeHint().width() <= 420
    assert outcome.title in notice.label.toolTip()
    assert outcome.detail in notice.action_button.toolTip()
    assert notice.action_button.geometry().right() < notice.width()


@pytest.mark.parametrize("width,multiple", ((420, False), (480, True), (640, True)))
def test_compact_notice_keeps_outcome_channel_and_revision_distinct(notice, application, width, multiple):
    outcome = build_job_outcome(
        state(name="Biplanar Y gradient (Gy)", project_revision=6),
        record(pipeline_sweeps={"checks_passed": False}),
    )
    notice.publish(outcome)
    if multiple:
        notice.publish(build_job_outcome(state("other", name="Other channel"), record()))
    notice.resize(width, notice.sizeHint().height())
    notice.show()
    application.processEvents()
    lines = notice.label.text().splitlines()
    assert len(lines) == 2
    assert lines[0] == "Finished · checks failed"
    assert "Biplanar" in lines[1] and "(Gy)" in lines[1]
    assert lines[1].endswith("revision 6")
    assert notice.height() <= 70
    assert outcome.title in notice.label.toolTip()
    assert outcome.detail in notice.label.toolTip()


@pytest.mark.parametrize("failed", (False, True))
def test_unloaded_project_adapts_only_visible_action_and_keeps_saved_outcome(notice, failed):
    saved = record(pipeline_sweeps={"checks_passed": not failed})
    outcome = build_job_outcome(state(), saved)
    notice.publish(outcome)
    notice.set_result_context(False)
    assert notice.selected == outcome
    assert notice.outcomes == (outcome,)
    assert notice.action_button.text() == "View job"
    assert "Open the owning project" in notice.action_button.toolTip()
    assert notice.label.property("semanticTone") == outcome.tone
    actions = []
    notice.action_requested.connect(actions.append)
    notice.action_button.click()
    assert actions == [outcome.job_id]
    notice.set_result_context(True)
    assert notice.selected == outcome
    assert notice.action_button.text() == outcome.action_label


def test_missing_project_context_does_not_change_worker_failure_action(notice):
    outcome = build_job_outcome(state(status="failed"))
    notice.publish(outcome)
    notice.set_result_context(False)
    assert notice.action_button.text() == "Inspect failure"
