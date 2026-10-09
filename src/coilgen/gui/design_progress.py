"""Passive recorded computation progress beside the scientific plot.

The controller supplies completed stage boundaries recorded after checkpoint
storage, loaded stage snapshots or verified final results. Worker percentages
and free-text activity cannot certify stage completion or engineering
acceptance. This view does not start jobs or inspect artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

from PySide6.QtCore import QEvent, QRectF, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from .colors import apply_tone, tone_color
from .icons import icon


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


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _percent(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return min(99, round(value * 100)) if value < 1 else 100


def _work_detail(state):
    detail = ""
    if state.get("progress_indeterminate"):
        detail = " · current operation has no measurable work total"
    elif (value := _percent(state.get("phase_progress"))) is not None:
        detail = f" · current operation {value}%"
    completed, total = state.get("progress_completed"), state.get("progress_total")
    if isinstance(completed, int) and isinstance(total, int) and total > 0:
        detail += f" ({completed}/{total} work units)"
    return detail


class ComputationIndicator(QWidget):
    """Persistent runtime activity, independent of the viewed design or result."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("computing_indicator")
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
                percent = _percent(active[0].get("progress"))
                indeterminate = bool(active[0].get("progress_indeterminate"))
                text = f"Computing ({percent}%)" if percent is not None else "Computing…"
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
                text = f"Computing ({percent}%)" if percent is not None else "Computing…"
            else:
                text = f"Working ({len(working)} tasks)" if len(working) > 1 else "Working…"
        elif queued:
            text = "Queued" if len(queued) == 1 else f"Queued ({len(queued)} jobs)"
            details.append("Waiting for an available worker; computation has not started.")
        else:
            self.label.clear()
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
        self.label.setText(text)
        self.setToolTip("\n".join(details))
        self.label.setToolTip(self.toolTip())
        self.progress.setVisible(busy)
        self.progress.setRange(0, 100 if percent is not None and not indeterminate else 0)
        if percent is not None:
            self.progress.setValue(percent)
        apply_tone(self.label, "warning" if unavailable else "info")
        self.show()


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
    activity = _text(state.get("stage"))
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
        text + ". Solid blue stages show recorded computation, not engineering acceptance."
        " Dotted blue Cached stages show verified boundaries available for reuse."
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
        or (status in {"running", "cancelling"} and (percent is None or state.get("progress_indeterminate", False)))
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
    )


class _StageChip(QWidget):
    """Painted computation indicator with no action or keyboard focus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage = None
        self.setMinimumWidth(30)
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_stage(self, stage):
        self.stage = stage
        self.setObjectName(f"design_progress_{stage.key}")
        self.setProperty("stageStatus", stage.status)
        self.setProperty("plottedStage", stage.viewed)
        self.setAccessibleName(f"{stage.label}: {_STATUS_TEXT[stage.status]}")
        self.setAccessibleDescription(stage.tooltip)
        self.setToolTip(stage.tooltip)
        self.update()

    def sizeHint(self):
        label = self.stage.label if self.stage else "Prepare"
        return QSize(self.fontMetrics().horizontalAdvance(label) + 42, 34)

    def paintEvent(self, event):
        if self.stage is None:
            return
        stage = self.stage
        palette = self.palette()
        tone = (
            "error"
            if stage.status == "failed"
            else "warning"
            if stage.status in {"stopped", "current"}
            else "info"
            if stage.status in {"complete", "reusable"}
            else "neutral"
        )
        foreground = tone_color(tone, palette)
        muted = palette.color(palette.ColorGroup.Disabled, palette.ColorRole.WindowText)
        if stage.status in {"waiting", "skipped"}:
            foreground = muted
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(foreground, 1.6 if stage.status == "current" else 1)
        if stage.status == "skipped":
            pen.setStyle(Qt.PenStyle.DashLine)
        elif stage.status == "reusable":
            pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawRoundedRect(QRectF(1, 1, self.width() - 2, self.height() - 2), 5, 5)
        badge = {"complete": "evaluated", "current": "pending", "failed": "failed", "skipped": "skipped"}.get(
            stage.status
        )
        mode = QIcon.Mode.Disabled if stage.status in {"waiting", "skipped"} else QIcon.Mode.Normal
        painter.drawPixmap(6, 7, icon(stage.icon_name, badge=badge).pixmap(QSize(20, 20), mode))
        rect = self.rect().adjusted(31, 0, -5, 0)
        text = self.fontMetrics().elidedText(stage.label, Qt.TextElideMode.ElideRight, max(0, rect.width()))
        if stage.status == "reusable":
            text += "\n" + self.fontMetrics().elidedText("Cached", Qt.TextElideMode.ElideRight, max(0, rect.width()))
        painter.setPen(foreground)
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        if stage.viewed:
            painter.setPen(QPen(tone_color("info", palette), 3))
            painter.drawLine(6, self.height() - 4, self.width() - 6, self.height() - 4)


class _ActivityLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.full_text = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text):
        self.full_text = text
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self.full_text, Qt.TextElideMode.ElideRight, max(0, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


class DesignProgress(QWidget):
    """Compact progress summary; Follow run is its only interactive control."""

    follow_requested = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("design_progress")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.snapshot = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)
        chip_row = QHBoxLayout()
        chip_row.setSpacing(5)
        self.chips = {key: _StageChip(self) for key in _KEYS}
        for chip in self.chips.values():
            chip_row.addWidget(chip, 1)
        layout.addLayout(chip_row)
        activity_row = QHBoxLayout()
        activity_row.setSpacing(7)
        self.activity_label = _ActivityLabel(self)
        self.activity_label.setObjectName("design_progress_activity")
        activity_row.addWidget(self.activity_label, 1)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("design_progress_percent")
        self.progress_bar.setFixedWidth(95)
        self.progress_bar.setMaximumHeight(18)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setAccessibleName("Worker-reported work progress")
        self.progress_bar.setToolTip(
            "Reported execution progress. A percentage does not verify a completed design stage."
        )
        activity_row.addWidget(self.progress_bar)
        self.follow_button = QToolButton(self)
        self.follow_button.setObjectName("design_progress_follow")
        self.follow_button.setCheckable(True)
        self.follow_button.setText("Follow run")
        self.follow_button.setIcon(icon("visibility"))
        self.follow_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.follow_button.setAccessibleName("Follow completed stage previews in the plot")
        self.follow_button.setToolTip(
            "Show completed stage previews as the design runs. Turn off to keep inspecting the selected dataset."
        )
        self.follow_button.toggled.connect(self.follow_requested)
        activity_row.addWidget(self.follow_button)
        layout.addLayout(activity_row)
        self.set_state("evaluated")

    def set_state(self, requested_stage, state=None, **options):
        self.snapshot = build_design_progress(requested_stage, state, **options)
        snapshot = self.snapshot
        for stage in snapshot.stages:
            self.chips[stage.key].set_stage(stage)
        self.activity_label.set_full_text(snapshot.activity)
        self.activity_label.setToolTip(snapshot.tooltip)
        self.activity_label.setAccessibleDescription(snapshot.tooltip)
        apply_tone(self.activity_label, snapshot.tone)
        self.progress_bar.setVisible(snapshot.status not in {"idle", "unavailable"})
        self.progress_bar.setRange(0, 0 if snapshot.busy else 100)
        self.progress_bar.setValue(snapshot.percent if snapshot.percent is not None else 0)
        self.progress_bar.setFormat(
            "Queued"
            if snapshot.status == "queued"
            else f"{snapshot.percent}%"
            if snapshot.percent is not None
            else "Unreported"
        )
        self.progress_bar.setToolTip(snapshot.tooltip)
        last_status = _text(state.get("status")) if isinstance(state, Mapping) else ""
        self.follow_button.setVisible(
            snapshot.follow_available or snapshot.status == "unavailable" and last_status in _ACTIVE
        )
        self.follow_button.setEnabled(snapshot.follow_available)
        self.follow_button.setToolTip(
            "Runtime status is unavailable. Following can resume when status recovers; the selected plot is retained."
            if snapshot.status == "unavailable"
            else "Show completed stage previews as the design runs. Turn off to keep inspecting the selected dataset."
        )
        with QSignalBlocker(self.follow_button):
            self.follow_button.setChecked(bool(options.get("following", True)))
        self.updateGeometry()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.FontChange}:
            if self.snapshot is not None:
                apply_tone(self.activity_label, self.snapshot.tone)
                self.activity_label._elide()
                for chip in self.chips.values():
                    chip.update()
