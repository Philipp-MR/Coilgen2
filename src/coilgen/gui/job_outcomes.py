"""Persistent outcomes of observed executions, separate from engineering acceptance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QWidget

from ..application.diagnostics import run_verdicts
from .colors import apply_tone
from .icons import icon
from .run_metadata import read_run_metadata


_TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
_PRIORITY = {"error": 0, "warning": 1, "info": 2, "success": 3, "neutral": 4}
_STAGES = {
    "prepared": "Prepare",
    "sheet": "Sheet",
    "contours": "Contours",
    "routed": "Winding",
    "evaluated": "Evaluate",
}


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job_id: str
    title: str
    detail: str
    tone: str
    action_label: str
    status: str
    caption: str = ""
    source_name: str = ""
    revision_text: str = ""


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def build_job_outcome(state, record=None, verification_error=""):
    """Present supplied runtime state and verified evidence without opening files.

    The controller chooses which observed transitions deserve a notice. A final
    record must already be verified by the caller; completed worker status alone
    cannot establish saved evidence integrity or an engineering verdict.
    """
    status = state.get("status")
    if status not in _TERMINAL:
        return None
    id_ = state.get("id")
    if not isinstance(id_, str) or not id_.strip():
        raise ValueError("An outcome requires its recorded job ID.")
    metadata = read_run_metadata(record)
    name = _text(state.get("name")) or _text(metadata.get("name")) or "Coil design"
    channel = _text(state.get("channel_name")) or _text(metadata.get("channel_name"))
    if channel and channel != name:
        name = channel + " · " + name
    revision = state.get("project_revision", metadata.get("project_revision"))
    revision_text = (
        f"revision {revision}"
        if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0
        else "revision unrecorded"
    )
    context = name + " · " + revision_text
    if status == "failed":
        title, tone, action = "Execution failed", "error", "Inspect failure"
        caption = "Execution failed"
        detail = _text(state.get("error")) or _text(state.get("stage")) or "No worker failure detail was recorded."
    elif status in {"cancelled", "interrupted"}:
        title, tone, action = status.capitalize(), "warning", "View job"
        caption = title
        detail = "No completed result was produced."
        recorded = state.get("completed_stages", ())
        if isinstance(recorded, (list, tuple)):
            retained = tuple(dict.fromkeys(_STAGES[key] for key in recorded if isinstance(key, str) and key in _STAGES))
            if retained:
                detail += " Earlier recorded stages retained: " + ", ".join(retained) + "."
        detail += " Checkpoint integrity and compatibility must be verified before resuming."
        if _text(state.get("error")):
            detail += " " + _text(state["error"])
    elif verification_error:
        title, tone, action = "Execution finished · evidence verification failed", "error", "Inspect failure"
        caption = "Finished · verification failed"
        detail = _text(verification_error)
    elif record is None:
        title, tone, action = "Execution finished · evidence unverified", "warning", "View job"
        caption = "Finished · evidence unverified"
        detail = "Final saved evidence has not been verified. Completed execution does not establish field or conductor acceptance."
    else:
        verdicts = run_verdicts(metadata)
        constraints = metadata.get("constraints") or ()
        constraints = tuple(row for row in constraints if isinstance(row, Mapping))
        failed = sum(row.get("passed") is False for row in constraints)
        unverified = sum(not isinstance(row.get("passed"), bool) for row in constraints)
        summary = metadata.get("summary") or {}
        failure = bool(failed) or "Failed" in verdicts.values()
        failure |= summary.get("accepted") is False and summary.get("scope") in {
            "continuous_sheet",
            "fabrication_review",
        }
        detail = (
            f"Saved sheet checks: {verdicts['sheet']}; pipeline conductor geometry: {verdicts['sweeps']}; "
            f"fabrication review: {verdicts['fabrication']}."
        )
        if constraints:
            detail += f" Recorded constraints: {failed} failed, {unverified} unverified, {len(constraints)} total."
        boundary = metadata.get("pipeline_stage", state.get("pipeline_stage"))
        if failure:
            title, tone, action = "Execution finished · engineering checks failed", "error", "Review checks"
            caption = "Finished · checks failed"
        elif boundary in _STAGES and boundary != "evaluated":
            title, tone, action = f"Execution finished · {_STAGES[boundary]} study", "info", "View result"
            caption = f"Finished · {_STAGES[boundary]} study"
        elif unverified or any(value in {"Unverified", "Not run"} for value in verdicts.values()):
            title, tone, action = "Execution finished · engineering checks incomplete", "warning", "Review checks"
            caption = "Finished · checks incomplete"
        else:
            title, tone, action = "Execution finished · saved checks passed in scope", "info", "View result"
            caption = "Finished · checks passed in scope"
        detail += " Saved checks apply to their recorded scopes and do not establish construction release."
    return JobOutcome(id_, title + " · " + context, detail, tone, action, status, caption, name, revision_text)


class _NoticeLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.full_text = ""
        self._outcome = None
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_outcome(self, outcome):
        self._outcome = outcome
        self.full_text = outcome.title
        self.setAccessibleName(outcome.title)
        self._elide()

    def _elide(self):
        metrics, width = self.fontMetrics(), max(0, self.width())
        outcome = self._outcome
        if outcome is not None and outcome.source_name:
            caption = metrics.elidedText(outcome.caption or outcome.title, Qt.TextElideMode.ElideRight, width)
            suffix = " · " + outcome.revision_text
            name = metrics.elidedText(
                outcome.source_name,
                Qt.TextElideMode.ElideMiddle,
                max(0, width - metrics.horizontalAdvance(suffix)),
            )
            self.setText(caption + "\n" + name + suffix)
        else:
            self.setText(metrics.elidedText(self.full_text, Qt.TextElideMode.ElideRight, width))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


class JobOutcomeNotice(QWidget):
    """Keep unresolved failures visible while preserving deliberate notice selection."""

    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("job_outcome_notice")
        self._outcomes = {}
        self._dismissed = set()
        self._result_context = True
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        self.selector = QComboBox(self)
        self.selector.setObjectName("job_outcome_selector")
        self.selector.setAccessibleName("Unresolved computation outcomes")
        self.selector.setMinimumWidth(0)
        self.selector.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.selector.currentIndexChanged.connect(self._refresh_selected)
        layout.addWidget(self.selector, 1)
        self.label = _NoticeLabel(self)
        self.label.setObjectName("job_outcome_title")
        layout.addWidget(self.label, 3)
        self.action_button = QPushButton(self)
        self.action_button.setObjectName("job_outcome_action")
        self.action_button.clicked.connect(self._request_action)
        layout.addWidget(self.action_button)
        self.dismiss_button = QToolButton(self)
        self.dismiss_button.setObjectName("dismiss_job_outcome")
        self.dismiss_button.setIcon(icon("cancel"))
        self.dismiss_button.setAccessibleName("Dismiss selected outcome notice")
        self.dismiss_button.setToolTip("Dismiss this notice; the saved job and results remain available.")
        self.dismiss_button.clicked.connect(self.dismiss_selected)
        layout.addWidget(self.dismiss_button)
        self.hide()

    @property
    def outcomes(self):
        return tuple(self._outcomes.values())

    @property
    def selected(self):
        return self._outcomes.get(self.selector.currentData())

    def publish(self, outcome):
        if outcome is None or outcome.job_id in self._dismissed:
            return
        if self._outcomes.get(outcome.job_id) == outcome:
            return
        selected = self.selected
        selected_id = selected.job_id if selected is not None else None
        if selected is not None and _PRIORITY.get(outcome.tone, 4) < _PRIORITY.get(selected.tone, 4):
            selected_id = outcome.job_id
        self._outcomes[outcome.job_id] = outcome
        self._rebuild(selected_id)

    def dismiss_selected(self):
        outcome = self.selected
        if outcome is not None:
            self._dismissed.add(outcome.job_id)
            self._outcomes.pop(outcome.job_id, None)
            self._rebuild()

    def clear(self):
        self._outcomes.clear()
        self._dismissed.clear()
        self._rebuild()

    def set_result_context(self, available):
        """Adapt navigation when no project is open; saved verdicts stay intact."""
        self._result_context = bool(available)
        self._refresh_selected()

    def _rebuild(self, selected_id=None):
        ordered = sorted(self._outcomes.values(), key=lambda outcome: _PRIORITY.get(outcome.tone, 4))
        with QSignalBlocker(self.selector):
            self.selector.clear()
            for outcome in ordered:
                text = (
                    f"{outcome.source_name} · {outcome.revision_text} · {outcome.caption}"
                    if outcome.source_name
                    else outcome.title
                )
                self.selector.addItem(text, outcome.job_id)
                self.selector.setItemData(
                    self.selector.count() - 1, outcome.title + "\n" + outcome.detail, Qt.ItemDataRole.ToolTipRole
                )
            selected_index = self.selector.findData(selected_id)
            self.selector.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        self.selector.setVisible(len(ordered) > 1)
        self.selector.setToolTip(f"{len(ordered)} unresolved computation outcomes; select one to inspect or dismiss.")
        self._refresh_selected()
        self.setVisible(bool(ordered))

    def _refresh_selected(self, *_args):
        outcome = self.selected
        if outcome is None:
            self.label.clear()
            return
        self.label.set_outcome(outcome)
        self.label.setToolTip(outcome.title + "\n" + outcome.detail)
        self.label.setAccessibleDescription(outcome.detail)
        apply_tone(self.label, outcome.tone)
        job_only = not self._result_context and outcome.action_label in {"View result", "Review checks"}
        label = "View job" if job_only else outcome.action_label
        self.action_button.setText(label)
        self.action_button.setAccessibleName(label)
        self.action_button.setToolTip(
            outcome.title
            + "\n"
            + outcome.detail
            + ("\nOpen the owning project to review its saved evidence." if job_only else "")
        )

    def _request_action(self):
        outcome = self.selected
        if outcome is not None:
            self.action_requested.emit(outcome.job_id)
