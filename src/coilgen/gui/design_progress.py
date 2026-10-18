"""Selected-run computation status and persistent global activity.

The controller supplies completed stage boundaries recorded after checkpoint
storage, loaded stage snapshots or verified final results. Worker percentages
and free-text activity cannot certify stage completion or engineering
acceptance. This view does not start jobs or inspect artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QProgressBar, QSizePolicy, QToolButton, QWidget

from .colors import apply_tone
from .runtime_feedback import RuntimeFeedback, fraction_percent as _percent, runtime_feedback, work_detail


_STAGES = (
    ("prepared", "Prepare", "Prepare model and target samples", "mesh"),
    ("sheet", "Sheet", "Optimize the continuous current sheet", "stream-function"),
    ("contours", "Contours", "Generate contour winding paths", "contours"),
    ("routed", "Winding", "Connect and sweep conductors according to the chosen winding settings", "conductor-sweep"),
    (
        "evaluated",
        "Evaluate",
        "Evaluate the completed design and recorded engineering checks",
        "independent-validation",
    ),
)
_KEYS = tuple(item[0] for item in _STAGES)
_ACTIVE = {"queued", "running", "cancelling"}
_STATUS_TEXT = {
    "complete": "Recorded stage computation completed",
    "reusable": "Verified compatible checkpoint available for reuse",
    "current": "Computing this stage",
    "waiting": "Awaiting computation",
    "skipped": "Beyond the requested study boundary",
    "failed": "Execution stopped at this stage",
    "stopped": "This stage was not completed before execution stopped",
}


@dataclass(frozen=True, slots=True)
class StageProgress:
    key: str
    label: str
    description: str
    icon_name: str
    status: str
    viewed: bool = False
    inferred: bool = False

    @property
    def tooltip(self):
        text = f"{self.description}. {_STATUS_TEXT[self.status]}."
        if self.inferred:
            text += " Next requested stage without recorded completion; the worker activity is shown below."
        if self.viewed:
            text += " Currently shown in the plot."
        if self.status == "complete":
            text += (
                " Stage completion does not establish engineering acceptance."
                " Optional live display availability and final result integrity are assessed separately."
            )
        elif self.status == "reusable":
            text += (
                " Cached numerical evidence matches the applied inputs, engine and dependencies."
                " It is not a newly completed stage of this execution or an engineering acceptance certificate."
            )
        return text


@dataclass(frozen=True, slots=True)
class DesignProgressState:
    stages: tuple[StageProgress, ...]
    activity: str
    tooltip: str
    tone: str
    percent: int | None
    busy: bool
    follow_available: bool
    provenance: str
    status: str
    runtime: RuntimeFeedback = RuntimeFeedback()


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _work_detail(state):
    detail = work_detail(state)
    return " · " + detail if detail else ""


class ComputationIndicator(QWidget):
    """Persistent runtime activity, independent of the viewed design or result."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("computing_indicator")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self)
        self.label.setObjectName("computing_label")
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setAccessibleName("Current computation activity")
        font = self.label.font()
        font.setBold(True)
        self.label.setFont(font)
        layout.addWidget(self.label)
        self.progress = QProgressBar(self)
        self.progress.setFixedSize(36, 10)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.details_label = QLabel(self)
        self.details_label.setObjectName("computing_task_summary")
        self.details_label.setTextFormat(Qt.TextFormat.PlainText)
        self.details_label.setMinimumWidth(0)
        self.details_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.details_label.setAccessibleName("Current workers and operations")
        layout.addWidget(self.details_label, 2)
        self.runtime_label = QLabel(self)
        self.runtime_label.setObjectName("computing_runtime_observation")
        self.runtime_label.setTextFormat(Qt.TextFormat.PlainText)
        self.runtime_label.setMinimumWidth(0)
        self.runtime_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.runtime_label.setAccessibleName("Worker elapsed time and observed liveness")
        layout.addWidget(self.runtime_label, 1)
        self.tasks_button = QToolButton(self)
        self.tasks_button.setText("Tasks…")
        self.tasks_button.setAccessibleName("Show every current computation and its progress")
        self.tasks_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.tasks_menu = QMenu(self.tasks_button)
        self.tasks_menu.setToolTipsVisible(True)
        self.tasks_button.setMenu(self.tasks_menu)
        layout.addWidget(self.tasks_button)
        self.message = QLabel(self)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.message, 1)
        self.hide()

    def set_activity(self, states=(), *, working=(), unavailable=None):
        active = [state for state in states if state.get("status") in {"running", "cancelling"}]
        queued = [state for state in states if state.get("status") == "queued"]
        percent = None
        indeterminate = False
        busy = bool(active or working)
        details = []
        visible_tasks = []
        task_rows = []
        if unavailable:
            text = "Working… · job activity unavailable" if working else "Activity unavailable"
            busy = bool(working)
            details.append(unavailable)
        elif active:
            cancelling = all(state["status"] == "cancelling" for state in active)
            if working:
                text = f"Computing ({len(active) + len(working)} tasks)"
            elif len(active) > 1:
                text = f"{'Cancelling' if cancelling else 'Computing'} ({len(active)} jobs)"
            elif cancelling:
                text = "Cancelling…"
            else:
                runtime = runtime_feedback(active[0])
                percent = (
                    runtime.phase_percent
                    if runtime.phase_percent is not None
                    else (None if runtime.allocated else _percent(active[0].get("progress")))
                )
                indeterminate = bool(active[0].get("progress_indeterminate"))
                if indeterminate:
                    percent = None
                text = (
                    f"Operation {percent}%"
                    if runtime.phase_percent is not None
                    else f"Computing ({percent}%)"
                    if percent is not None
                    else "Computing…"
                )
            for state in active:
                value = _percent(state.get("progress"))
                details.append(
                    f"{state.get('name') or state.get('id') or 'Worker'}: {state['status']}"
                    + (f" ({value}%)" if value is not None else "")
                    + (f" · {state['stage']}" if state.get("stage") else "")
                    + _work_detail(state)
                )
            details.append(
                "Overall progress estimates allocated work, not elapsed or remaining time. Concurrent tasks have no combined percentage."
            )
        elif working:
            if len(working) == 1 and isinstance(working[0], Mapping):
                percent = _percent(working[0].get("progress"))
                indeterminate = bool(working[0].get("progress_indeterminate"))
                if indeterminate:
                    percent = None
                text = f"Computing ({percent}%)" if percent is not None else "Computing…"
            else:
                text = f"Working ({len(working)} tasks)" if len(working) > 1 else "Working…"
        elif queued:
            text = "Queued" if len(queued) == 1 else f"Queued ({len(queued)} jobs)"
            details.append("Waiting for an available worker; computation has not started.")
        else:
            self.label.clear()
            self.details_label.clear()
            self.runtime_label.clear()
            self.tasks_menu.close()
            self._set_task_rows(())
            self.hide()
            return
        for item in working:
            if isinstance(item, Mapping):
                value = _percent(item.get("progress"))
                details.append(
                    str(item.get("name", "Computing"))
                    + (f" ({value}%)" if value is not None else "")
                    + (f" · {item['stage']}" if item.get("stage") else "")
                    + _work_detail(item)
                )
            else:
                details.append(str(item))
        for state in () if unavailable else (*active, *queued):
            name = _text(state.get("name")) or _text(state.get("id")) or "Worker"
            status = state.get("status", "")
            value = _percent(state.get("progress")) if status != "queued" else None
            opaque = state.get("progress_indeterminate") or status == "cancelling"
            runtime = runtime_feedback(state)
            suffix = (
                f"operation {runtime.phase_percent}%"
                if runtime.phase_percent is not None and not opaque
                else f"allocated work {value}%"
                if value is not None and not opaque and runtime.allocated
                else f"{value}%"
                if value is not None and not opaque
                else "queued"
                if status == "queued"
                else status
            )
            operation = runtime.operation
            revision = state.get("project_revision")
            identity = f"revision {revision}" if isinstance(revision, int) and not isinstance(revision, bool) else ""
            if job_id := _text(state.get("id")):
                identity += (" · " if identity else "") + "job " + job_id
            visible_tasks.append(
                (operation + " · " if operation and len(active) == 1 and not working else "")
                + f"{name} {suffix}"
                + (" · no work total" if state.get("progress_indeterminate") else "")
            )
            task_rows.append(
                f"{name}: {status}"
                + (
                    f" (allocated work {value}%)"
                    if value is not None and runtime.allocated
                    else f" ({value}%)"
                    if value is not None
                    else ""
                )
                + (f"\n{identity}" if identity else "")
                + (f"\n{operation}" if operation else "")
                + _work_detail(state)
                + (f"\n{runtime.line}" if runtime.line else "")
                + (f"\n{runtime.tooltip}" if runtime.tooltip else "")
            )
        for item in working:
            state = item if isinstance(item, Mapping) else {"name": str(item)}
            name = _text(state.get("name")) or "Working"
            value = _percent(state.get("progress"))
            operation = _text(state.get("stage"))
            if state.get("progress_indeterminate"):
                value = None
            visible_tasks.append(
                f"{name}"
                + (f" {value}%" if value is not None else " · working")
                + (f" · {operation}" if not active and len(working) == 1 else "")
            )
            task_rows.append(
                name
                + (f" ({value}%)" if value is not None else " · working")
                + (f"\n{operation}" if operation else "")
                + _work_detail(state)
            )
        if unavailable:
            task_rows.append("Job runtime unavailable\n" + str(unavailable))
        runtime = (
            runtime_feedback(active[0])
            if len(active) == 1 and not unavailable
            else runtime_feedback(queued[0])
            if len(queued) == 1 and not active and not unavailable
            else RuntimeFeedback()
        )
        self.runtime_label.setText(runtime.line)
        self.runtime_label.setToolTip(runtime.line + ("\n" + runtime.tooltip if runtime.tooltip else ""))
        self.runtime_label.setAccessibleDescription(self.runtime_label.toolTip())
        self.runtime_label.setVisible(bool(runtime.line))
        if runtime.line:
            details.append(runtime.line + ". " + runtime.tooltip)
        summary = "  ·  ".join(visible_tasks)
        self.details_label.setText(summary)
        self.details_label.setToolTip(summary)
        self.details_label.setAccessibleDescription(summary)
        self.details_label.setVisible(bool(summary))
        self._set_task_rows(task_rows)
        self.label.setText(text)
        self.setToolTip("\n".join(details))
        self.label.setToolTip(self.toolTip())
        self.progress.setVisible(busy)
        self.progress.setRange(0, 100 if percent is not None and not indeterminate else 0)
        if percent is not None:
            self.progress.setValue(percent)
        apply_tone(
            self.label, "warning" if unavailable or any(state["status"] == "cancelling" for state in active) else "info"
        )
        self.show()

    def _set_task_rows(self, rows):
        actions = self.tasks_menu.actions()
        for index, row in enumerate(rows):
            action = actions[index] if index < len(actions) else self.tasks_menu.addAction("")
            action.setText(row.replace("&", "&&"))
            action.setToolTip(row)
        for action in actions[len(rows) :]:
            self.tasks_menu.removeAction(action)
            action.deleteLater()
        self.tasks_button.setVisible(bool(rows))


def build_design_progress(
    requested_stage,
    state=None,
    *,
    completed=(),
    reusable=(),
    preview_stage=None,
    following=True,
    pending=False,
    verification=None,
    active_stage=None,
    unavailable=None,
):
    """Build presentation without guessing completion from worker activity.

    ``completed`` must come from structured stage-boundary events emitted after
    checkpoint storage, verified live snapshots or verified final run records.
    ``reusable`` contains actual boundaries of a previously verified reuse plan;
    presentation never opens archives or infers cache compatibility itself.
    ``active_stage`` may identify the worker's current operation if structured;
    free-text ``state['stage']`` is never parsed for it. Without that identity, a
    running job highlights the first unrecorded stage as an explicitly inferred
    boundary. ``progress`` is a fraction in [0, 1]. ``unavailable`` reports a
    failed runtime refresh: retained activity is explicitly last known, without
    an active-stage highlight, progress claim or animation.
    """
    if requested_stage not in _KEYS:
        raise ValueError(f"Unknown design stage: {requested_stage!r}")
    if verification not in {None, "pending", "verified", "failed"}:
        raise ValueError(f"Unknown evidence verification state: {verification!r}")
    state = state if isinstance(state, Mapping) else {}
    status = _text(state.get("status")) or "idle"
    runtime_unavailable = unavailable is not None
    boundary = _KEYS.index(requested_stage)
    recorded = (
        frozenset(key for key in completed if isinstance(key, str) and key in _KEYS)
        if isinstance(completed, (tuple, list, set, frozenset))
        else frozenset()
    )
    recorded = recorded.intersection(_KEYS[: boundary + 1])
    cached = (
        frozenset(key for key in reusable if isinstance(key, str) and key in _KEYS)
        if isinstance(reusable, (tuple, list, set, frozenset))
        else frozenset()
    ).intersection(_KEYS[: boundary + 1])
    remaining = [key for key in _KEYS[: boundary + 1] if key not in recorded]
    explicit_active = active_stage in remaining
    current = active_stage if explicit_active else remaining[0] if remaining else None
    # Queued jobs have not entered the numerical pipeline. Completed workers
    # awaiting verification likewise cannot claim a current or finished stage.
    mark_current = not runtime_unavailable and status in {"running", "cancelling", "failed", "cancelled", "interrupted"}
    stages = []
    for index, (key, label, description, icon_name) in enumerate(_STAGES):
        stage_status = (
            "skipped"
            if index > boundary
            else "complete"
            if key in recorded
            else "failed"
            if mark_current and key == current and status == "failed"
            else "stopped"
            if mark_current and key == current and status in {"cancelled", "interrupted"}
            else "current"
            if mark_current and key == current
            else "reusable"
            if key in cached
            else "waiting"
        )
        stages.append(
            StageProgress(
                key,
                label,
                description,
                icon_name,
                stage_status,
                viewed=key == preview_stage and key in recorded,
                inferred=mark_current and key == current and not explicit_active,
            )
        )
    runtime = runtime_feedback(state)
    activity = runtime.operation
    name = _text(state.get("name"))
    revision = state.get("project_revision")
    provenance = name
    if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0:
        provenance += (" · " if provenance else "") + f"revision {revision}"
    tone = "info" if status in _ACTIVE else "neutral"
    if runtime_unavailable:
        description = "Runtime unavailable · last known: " + (status if status != "idle" else "status unrecorded")
        tone = "error" if verification == "failed" else "warning"
        if verification == "failed":
            description += " · saved evidence verification failed"
    elif verification == "failed":
        description, tone = "Evidence verification failed · see Messages", "error"
    elif verification == "pending":
        description = "Execution finished · verifying saved evidence…"
    elif status == "queued":
        description = "Queued · waiting for a worker"
    elif status == "running":
        description = "Computing · " + (activity or "worker activity unrecorded")
    elif status == "cancelling":
        description, tone = "Cancelling · waiting for the worker to stop", "warning"
        if activity:
            description += " · " + activity
    elif status == "failed":
        description, tone = "Execution failed · " + (_text(state.get("error")) or activity or "see Messages"), "error"
    elif status in {"cancelled", "interrupted"}:
        description, tone = status.capitalize() + " · recorded earlier stages retained", "warning"
    elif status == "completed":
        description = (
            "Stage evidence available · review engineering checks"
            if verification == "verified" and not remaining
            else "Execution finished · stage evidence unverified"
        )
        tone = "info" if verification == "verified" else "warning"
    elif status == "idle":
        description = "Applied model · changes pending" if pending else "Current model · stages await computation"
        if cached:
            description = (
                "Applied model · changes pending · verified checkpoints reusable"
                if pending
                else "Current model · verified checkpoints reusable"
            )
        tone = "warning" if pending else "neutral"
    else:
        description, tone = "Execution state unrecorded · review Jobs", "warning"
    if pending and status != "idle":
        description += " · model edits pending"
    # Keep the operation visible even when a long design name is elided.
    text = description + (" · " + provenance if provenance else "")
    tooltip = (
        text + ". Completed stages show recorded computation, not engineering acceptance."
        " Cached stages show verified boundaries available for reuse."
        " Optional live display availability and final result verification are separate checks."
    )
    if cached:
        tooltip += (
            " Cached stages can be reused after the worker rechecks compatibility; they are not newly computed stages."
        )
        if pending:
            tooltip += " Pending drafts are not included in that compatibility check and may invalidate the cache."
    if runtime_unavailable:
        tooltip += " Runtime refresh failed: " + (_text(unavailable) or "No current worker status is available.")
        if activity:
            tooltip += " Last known operation: " + activity + "."
        error = _text(state.get("error"))
        if error:
            tooltip += " Last known worker error: " + error + "."
        tooltip += " Follow run is unavailable until runtime status recovers; the selected plot is retained."
    elif following and status in _ACTIVE:
        tooltip += " The plot follows the latest loaded stage preview when available."
    elif status in _ACTIVE:
        tooltip += " Plot following is paused; the selected dataset is retained."
    percent = _percent(state.get("progress"))
    if runtime_unavailable:
        tooltip += (
            f" Last known worker progress: {percent}%; current progress is unavailable."
            if percent is not None
            else " Last known worker progress: unreported; current progress is unavailable."
        )
        percent = None
    elif percent is not None:
        tooltip += f" Worker-reported work progress: {percent}%; it does not verify stage completion."
        tooltip += " Overall progress estimates allocated work, not remaining time." + _work_detail(state)
    busy = not runtime_unavailable and (
        verification == "pending"
        or status == "cancelling"
        or (status == "running" and (percent is None or state.get("progress_indeterminate", False)))
    )
    return DesignProgressState(
        tuple(stages),
        text,
        tooltip,
        tone,
        percent,
        busy,
        not runtime_unavailable and status in _ACTIVE,
        provenance,
        "unavailable" if runtime_unavailable else status,
        RuntimeFeedback(operation=runtime.operation) if runtime_unavailable else runtime,
    )
