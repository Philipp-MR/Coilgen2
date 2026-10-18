"""Persistent channel-by-channel computation plan beside the scientific view.

Rows report recorded evidence and actual worker activity. Selecting a design
step never starts computation or changes the plotted dataset.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QIcon, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .colors import apply_tone, tone_color
from .design_progress import build_design_progress
from .icons import icon


_STAGE_LABELS = {
    "complete": "Completed",
    "reusable": "Cached",
    "current": "Computing",
    "waiting": "Pending",
    "skipped": "Outside plan",
    "failed": "Failed",
    "stopped": "Stopped",
    "stale": "Inputs changed",
}
_ACTIVE = {"running", "cancelling", "queued"}
_CHANNEL_LABELS = {
    "idle": "Pending",
    "waiting": "Pending",
    "running": "Computing",
    "cancelling": "Cancelling",
    "queued": "Queued",
    "completed": "Completed",
    "complete": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "interrupted": "Interrupted",
    "excluded": "Excluded",
    "disabled": "Excluded",
    "stale": "Inputs changed",
    "unavailable": "Unavailable",
    "verifying": "Verifying",
    "unverified": "Unverified",
}
_IDENTITY_ROLE = Qt.ItemDataRole.UserRole


class DesignPlanPanel(QWidget):
    """A passive plan with explicit controls for inspection and computation."""

    follow_requested = Signal(bool)
    channel_requested = Signal(str)
    job_requested = Signal(str)
    compute_channels_requested = Signal()
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("design_plan_panel")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(280)
        self.plan = None
        self.snapshot = None
        self.channel_items = {}
        self.stage_items = {}
        self._worker_job_id = None
        self.worker_channel = None
        self._following = True
        self._following_job_id = None
        self._follow_unavailable = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)
        self.summary_label = self._label("No design open", "design_plan_summary")
        font = self.summary_label.font()
        font.setBold(True)
        self.summary_label.setFont(font)
        layout.addWidget(self.summary_label)
        self.editing_label = self._label("", "design_plan_editing")
        layout.addWidget(self.editing_label)
        self.scope_label = self._label(
            "Current model · completion is separate from engineering acceptance", "design_plan_scope"
        )
        self.scope_label.setToolTip(
            "The plan refers to applied model inputs. The plot may show a different channel, revision or saved run. "
            "Completed computation does not certify field accuracy or manufacturability."
        )
        layout.addWidget(self.scope_label)

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("design_plan_steps")
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(("Design step", "State"))
        self.tree.setAccessibleName("Computation plan by channel")
        self.tree.setAccessibleDescription(
            "Expand channels to inspect their five computation stages. Pending stages are grey; recorded completed "
            "stages use the normal text colour. Activate a channel header to edit that channel. Stage rows are passive."
        )
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.setIndentation(14)
        self.tree.setIconSize(QSize(18, 18))
        self.tree.setUniformRowHeights(True)
        self.tree.setAllColumnsShowFocus(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemActivated.connect(self._item_activated)
        layout.addWidget(self.tree, 1)

        self.activity_label = self._label("", "design_progress_activity")
        self.activity_label.setAccessibleName("Current worker activity")
        layout.addWidget(self.activity_label)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("design_progress_percent")
        self.progress_bar.setMaximumHeight(18)
        self.progress_bar.setAccessibleName("Active worker reported progress")
        self.progress_bar.setToolTip(
            "This worker's reported work progress, not a combined system percentage or engineering acceptance."
        )
        layout.addWidget(self.progress_bar)
        self.runtime_label = self._label("", "design_progress_runtime")
        self.runtime_label.setAccessibleName("Worker elapsed time and observed liveness")
        self.runtime_label.hide()
        layout.addWidget(self.runtime_label)
        runtime_row = QHBoxLayout()
        self.follow_button = QToolButton(self)
        self.follow_button.setObjectName("design_progress_follow")
        self.follow_button.setText("Follow run")
        self.follow_button.setIcon(icon("visibility"))
        self.follow_button.setCheckable(True)
        self.follow_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.follow_button.setAccessibleName("Follow completed previews in the plot")
        self.follow_button.toggled.connect(self.follow_requested)
        runtime_row.addWidget(self.follow_button)
        self.job_button = QToolButton(self)
        self.job_button.setText("Inspect worker")
        self.job_button.setAccessibleName("Inspect the reported worker")
        self.job_button.clicked.connect(self._inspect_worker)
        runtime_row.addWidget(self.job_button)
        runtime_row.addStretch(1)
        layout.addLayout(runtime_row)
        self.compute_channels_button = QPushButton("Compute enabled channels", self)
        self.compute_channels_button.setIcon(icon("compute-study"))
        self.compute_channels_button.setAccessibleName("Compute every enabled channel independently")
        self.compute_channels_button.setToolTip(
            "Run each enabled channel's configured study independently. Channel visibility in the plot does not affect this selection."
        )
        self.compute_channels_button.clicked.connect(self.compute_channels_requested)
        layout.addWidget(self.compute_channels_button)
        actions_row = QHBoxLayout()
        self.action_buttons = {}
        for action, text, tooltip in (
            ("plan", "Recompute plan…", "Inspect the study and checkpoint reuse plan before computing."),
            ("jobs", "Jobs…", "Inspect all worker jobs, including other channels and earlier revisions."),
        ):
            button = QToolButton(self)
            button.setText(text)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda _checked=False, key=action: self.action_requested.emit(key))
            self.action_buttons[action] = button
            actions_row.addWidget(button)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)
        self.set_state("evaluated")
        self.activity_label.clear()
        self.compute_channels_button.hide()
        self.editing_label.hide()

    def _label(self, text, object_name):
        label = QLabel(text, self)
        label.setObjectName(object_name)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return label

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(320)
        return size

    def set_plan(self, plan):
        """Update existing rows in place so a user's expansion and scroll persist."""
        self.plan = plan
        channels = plan.channels if plan is not None else ()
        wanted = {channel.channel_id for channel in channels}
        scroll = self.tree.verticalScrollBar().value()
        with QSignalBlocker(self.tree):
            for channel_id in tuple(self.channel_items):
                if channel_id not in wanted:
                    item = self.channel_items.pop(channel_id)
                    self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
                    for key in tuple(self.stage_items):
                        if key[0] == channel_id:
                            del self.stage_items[key]
            for index, channel in enumerate(channels):
                item = self.channel_items.get(channel.channel_id)
                if item is None:
                    item = QTreeWidgetItem()
                    item.setData(0, _IDENTITY_ROLE, ("channel", channel.channel_id))
                    self.channel_items[channel.channel_id] = item
                    self.tree.insertTopLevelItem(index, item)
                    item.setExpanded(True)
                elif self.tree.indexOfTopLevelItem(item) != index:
                    self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
                    self.tree.insertTopLevelItem(index, item)
                self._update_channel(item, channel)
        self.tree.verticalScrollBar().setValue(scroll)
        self.summary_label.setText(plan.summary if plan is not None else "No design open")
        self.summary_label.setToolTip(plan.tooltip if plan is not None else "")
        self.scope_label.setVisible(bool(channels))
        selected = next((channel for channel in channels if channel.selected), None)
        self.editing_label.setVisible(selected is not None)
        if selected is not None:
            self.editing_label.setText(f"Editing: {selected.name} · revision {selected.revision}")
            self.editing_label.setToolTip(selected.tooltip)
        self.compute_channels_button.setVisible(len(channels) > 1)
        self.compute_channels_button.setEnabled(
            bool(
                plan is not None
                and plan.enabled_channels
                and not any(channel.status == "unavailable" for channel in channels)
            )
        )
        self.action_buttons["plan"].setEnabled(bool(channels))
        self._render_activity()
        if plan is None:
            self.activity_label.clear()
            self.runtime_label.clear()
            self.runtime_label.hide()
            self.progress_bar.hide()
            self.follow_button.hide()
            self.job_button.hide()
            self.worker_channel = None

    def _update_channel(self, item, channel):
        item.setText(0, channel.name)
        display_status = (
            "failed"
            if channel.verification == "failed"
            else "verifying"
            if channel.verification == "pending"
            else channel.status
        )
        reporting_job = display_status in {*_ACTIVE, "failed", "verifying", "unavailable"}
        excluded = not channel.enabled and not reporting_job
        label = "Excluded" if excluded else _CHANNEL_LABELS.get(display_status, display_status.capitalize())
        if (
            channel.status == "running"
            and channel.percent is not None
            and not channel.busy
            and not channel.runtime.allocated
        ):
            label += f" {channel.percent}%"
        elif channel.enabled and channel.status in {"completed", "complete"} and channel.review_needed:
            label = "Review checks"
        item.setText(1, label)
        font = item.font(0)
        font.setBold(True)
        for column in (0, 1):
            item.setFont(column, font)
            item.setToolTip(column, channel.tooltip)
        if not channel.enabled and reporting_job:
            item.setToolTip(
                1,
                channel.tooltip
                + " This job retains its submitted inputs; exclusion applies to the next system compute.",
            )
        self._color_row(item, "waiting" if excluded else display_status, header=True)
        stage_keys = {stage.key for stage in channel.stages}
        for identity in tuple(self.stage_items):
            if identity[0] == channel.channel_id and identity[1] not in stage_keys:
                item.removeChild(self.stage_items.pop(identity))
        for index, stage in enumerate(channel.stages):
            identity = (channel.channel_id, stage.key)
            row = self.stage_items.get(identity)
            if row is None:
                row = QTreeWidgetItem()
                row.setData(0, _IDENTITY_ROLE, ("stage", *identity))
                self.stage_items[identity] = row
                item.insertChild(index, row)
            row.setText(0, f"{index + 1}. {stage.label}")
            row.setText(
                1,
                "Stopping"
                if channel.status == "cancelling" and stage.status == "current"
                else _STAGE_LABELS.get(stage.status, stage.status.capitalize()),
            )
            stage_icon = icon(stage.icon_name)
            if stage.status in {"waiting", "skipped", "stale", "reusable"}:
                stage_icon = QIcon(stage_icon.pixmap(self.tree.iconSize(), QIcon.Mode.Disabled))
            row.setIcon(0, stage_icon)
            for column in (0, 1):
                row.setToolTip(column, stage.tooltip)
            row.setData(0, Qt.ItemDataRole.AccessibleTextRole, f"{stage.label}: {row.text(1)}")
            row.setData(0, Qt.ItemDataRole.AccessibleDescriptionRole, stage.tooltip)
            self._color_row(row, stage.status)

    def _color_row(self, item, status, *, header=False):
        palette = self.tree.palette()
        muted = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
        tone = (
            "info"
            if status in {"current", "running", "queued"}
            else "error"
            if status == "failed"
            else "warning"
            if status in {"stopped", "cancelled", "interrupted", "cancelling", "unavailable", "unverified"}
            else "neutral"
        )
        foreground = tone_color(tone, palette)
        if status in {"waiting", "idle", "skipped", "stale", "reusable", "excluded", "disabled"}:
            foreground = muted
        for column in (0, 1):
            item.setForeground(column, QBrush(foreground))
            font = item.font(column)
            font.setBold(header or status == "current")
            item.setFont(column, font)
        if status in {"stale", "reusable"}:
            item.setForeground(1, QBrush(tone_color("warning" if status == "stale" else "info", palette)))

    def set_state(self, requested_stage, state=None, **options):
        """Keep the existing selected-run snapshot contract for the controller."""
        self.snapshot = build_design_progress(requested_stage, state, **options)
        self._following = bool(options.get("following", True))
        self._follow_unavailable = self.snapshot.status == "unavailable"
        self._render_activity()

    def set_following(self, following, job_id=None):
        """Reflect the plot's actual followed job without emitting user actions."""
        self._following = bool(following)
        self._following_job_id = job_id
        self._render_activity()

    def _render_activity(self):
        channels = self.plan.channels if self.plan is not None else ()
        worker = next(
            (channel for channel in channels if channel.selected and channel.status in {"running", "cancelling"}), None
        )
        worker = worker or next((channel for channel in channels if channel.status in {"running", "cancelling"}), None)
        worker = worker or next(
            (channel for channel in channels if channel.selected and channel.status == "queued"), None
        )
        worker = worker or next((channel for channel in channels if channel.status == "queued"), None)
        worker = worker or next((channel for channel in channels if channel.selected), None)
        self.worker_channel = worker
        if self.plan is not None and worker is not None:
            status, percent, busy = worker.status, worker.percent, worker.busy
            activity, tooltip = worker.activity, worker.tooltip
            runtime = worker.runtime
            self._worker_job_id = worker.job_id
            if status in _ACTIVE:
                activity = f"Worker: {worker.name} · revision {worker.revision}\n{activity}"
            visible = status in {"running", "cancelling", "queued", "verifying"} or worker.verification == "pending"
            unavailable = self._follow_unavailable or status == "unavailable"
            follow_available = status in _ACTIVE and not unavailable
            follow_visible = follow_available or unavailable and worker.job_id is not None
            tone = (
                "error"
                if status == "failed"
                else "warning"
                if status in {"stale", "unavailable", "cancelling"}
                else "info"
                if status in {"running", "queued"}
                else "neutral"
            )
        else:
            snapshot = self.snapshot
            if snapshot is None:
                return
            status, percent, busy = snapshot.status, snapshot.percent, snapshot.busy
            activity, tooltip = snapshot.activity, snapshot.tooltip
            runtime = snapshot.runtime
            visible = status in {"running", "cancelling", "queued"} or busy
            follow_available = snapshot.follow_available
            follow_visible = follow_available or self._follow_unavailable
            tone = snapshot.tone
            unavailable = self._follow_unavailable
            self._worker_job_id = None
        self.activity_label.setText(activity)
        self.activity_label.setToolTip(tooltip)
        self.activity_label.setAccessibleDescription(tooltip)
        apply_tone(self.activity_label, tone)
        self.runtime_label.setText(runtime.line)
        self.runtime_label.setToolTip(runtime.line + ("\n" + runtime.tooltip if runtime.tooltip else ""))
        self.runtime_label.setAccessibleDescription(self.runtime_label.toolTip())
        self.runtime_label.setVisible(bool(runtime.line))
        self.progress_bar.setVisible(visible)
        self.progress_bar.setRange(0, 0 if busy and status != "queued" else 100)
        displayed_percent = runtime.phase_percent if runtime.phase_percent is not None else percent
        self.progress_bar.setValue(displayed_percent if displayed_percent is not None else 0)
        self.progress_bar.setFormat(
            "Queued"
            if status == "queued"
            else "Cancelling"
            if status == "cancelling"
            else "Working"
            if busy
            else f"Operation {displayed_percent}%"
            if runtime.phase_percent is not None
            else f"Allocated work {percent}%"
            if percent is not None and runtime.allocated
            else f"{percent}%"
            if percent is not None
            else "Unreported"
        )
        self.progress_bar.setToolTip(tooltip)
        self.job_button.setVisible(self._worker_job_id is not None)
        self.job_button.setToolTip(f"Inspect worker {self._worker_job_id}" if self._worker_job_id else "")
        self.follow_button.setVisible(follow_visible)
        self.follow_button.setEnabled(follow_available)
        self.follow_button.setToolTip(
            "Runtime status is unavailable; the selected plot is retained."
            if unavailable
            else "Follow completed previews for the indicated worker. Following another channel first selects that channel. "
            "Turn off to keep the selected plot."
        )
        with QSignalBlocker(self.follow_button):
            self.follow_button.setChecked(
                self._following and (self._following_job_id is None or self._following_job_id == self._worker_job_id)
            )

    def _item_activated(self, item, _column):
        identity = item.data(0, _IDENTITY_ROLE)
        if identity and identity[0] == "channel":
            self.channel_requested.emit(identity[1])

    def _inspect_worker(self):
        if self._worker_job_id is not None:
            self.job_requested.emit(self._worker_job_id)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange}:
            if self.plan is not None:
                self.set_plan(self.plan)
