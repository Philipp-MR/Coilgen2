"""Compact staged evidence, saved margins, and linked diagnostic actions."""

from __future__ import annotations

import math
from numbers import Real

from PySide6.QtCore import QEvent, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .acceptance import acceptance_report
from .colors import apply_tone, item_foreground, tone_color
from .icons import icon
from .run_context import run_timestamp


_STAGE_TITLES = {"sheet": "Sheet", "contours": "Contours", "routed": "Connected", "physical": "Conductor"}
_STAGE_ICONS = {
    "sheet": "stream-function",
    "contours": "contours",
    "routed": "connected-winding",
    "physical": "conductor-sweep",
}


def _status_badge(status):
    if status.startswith("Passed"):
        return "passed"
    return {"Failed": "failed", "Evaluated": "evaluated", "Not run": "skipped"}.get(status, "unverified")


def _check_icon(check):
    suffix = check.key.rsplit(":", 1)[-1]
    return {
        "clearance": "wire-spacing",
        "bend": "bend-radius",
        "penetration": "collision",
        "frame": "surface-normal",
        "terminals": "terminals",
    }.get(suffix, _STAGE_ICONS[check.stage])


def _text(value, unit, *, signed=False):
    if value is None:
        return "—"
    factor, display_unit = {"T": (1e6, "µT"), "m": (1e3, "mm")}.get(unit, (1, unit))
    converted = value * factor
    if math.isfinite(converted):
        value = converted
    else:
        display_unit = unit
    return (f"{value:+.5g}" if signed else f"{value:.5g}") + (f" {display_unit}" if display_unit else "")


def _label(parent, name):
    label = QLabel(parent)
    label.setObjectName(name)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class AcceptancePanel(QWidget):
    """The owner supplies verified records and already resolved ProblemIssues."""

    action_requested = Signal(str)
    witness_requested = Signal(object)
    saved_settings_requested = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("acceptance_panel")
        self.report = acceptance_report()
        self.issues = ()
        self._record_key = None
        self._record = None
        self._project = None
        self._pending = False
        self._diagnostic = ""
        self._stage_filter = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        self.identity = _label(self, "acceptance_identity")
        layout.addWidget(self.identity)
        stage_layout = QHBoxLayout()
        stage_layout.setSpacing(5)
        self.stage_buttons = {}
        for key, title in _STAGE_TITLES.items():
            button = QToolButton(self)
            button.setObjectName(f"acceptance_stage_{key}")
            button.setText(title)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setIconSize(QSize(20, 20))
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, stage=key: self._select_stage(stage))
            stage_layout.addWidget(button, 1)
            self.stage_buttons[key] = button
        layout.addLayout(stage_layout)
        self.summary = _label(self, "acceptance_summary")
        layout.addWidget(self.summary)
        self.metrics = _label(self, "acceptance_metrics")
        layout.addWidget(self.metrics)
        filters = QHBoxLayout()
        self.filter_summary = _label(self, "acceptance_filter_summary")
        filters.addWidget(self.filter_summary, 1)
        self.show_all = QCheckBox("Show all checks", self)
        self.show_all.setObjectName("acceptance_show_all")
        self.show_all.setToolTip(
            "Include passed requirements and descriptive metrics. Stage buttons inspect that stage's checks."
        )
        self.show_all.toggled.connect(self._apply_filter)
        filters.addWidget(self.show_all)
        layout.addLayout(filters)
        self.table = QTableWidget(0, 6, self)
        self.table.setObjectName("acceptance_table")
        self.table.setHorizontalHeaderLabels(
            ["Stage / requirement", "Observed", "Saved limit", "Margin", "Status", "Evaluation scope"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(100)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(0, 340)
        for column in (1, 2, 3):
            self.table.setColumnWidth(column, 100)
        self.table.setColumnWidth(4, 105)
        layout.addWidget(self.table, 1)
        self.details = _label(self, "acceptance_details")
        self.details_scroll = QScrollArea(self)
        self.details_scroll.setObjectName("acceptance_details_scroll")
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.details_scroll.setMinimumHeight(24)
        self.details_scroll.setMaximumHeight(66)
        self.details_scroll.setWidget(self.details)
        layout.addWidget(self.details_scroll)
        action_layout = QHBoxLayout()
        self.locate = QPushButton("Locate saved evidence", self)
        self.locate.setObjectName("acceptance_locate")
        self.settings = QPushButton("Inspect saved settings", self)
        self.settings.setObjectName("acceptance_settings")
        self.edit_settings = QPushButton("Edit current channel settings", self)
        self.edit_settings.setObjectName("acceptance_edit_current_settings")
        self.problems = QPushButton("All recorded problems", self)
        self.problems.setObjectName("acceptance_problems")
        for button, name in (
            (self.locate, "locate"),
            (self.settings, "settings"),
            (self.edit_settings, "edit"),
            (self.problems, "warning"),
        ):
            button.setIcon(icon(name))
            button.setIconSize(QSize(18, 18))
            button.setAccessibleName(button.text())
        action_layout.addWidget(self.locate)
        action_layout.addWidget(self.problems)
        action_layout.addStretch()
        layout.addLayout(action_layout)
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(self.settings)
        settings_layout.addWidget(self.edit_settings)
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        self.table.itemSelectionChanged.connect(self._selected)
        self.locate.clicked.connect(self._locate)
        self.settings.clicked.connect(self._settings)
        self.edit_settings.clicked.connect(self._edit_settings)
        self.problems.clicked.connect(lambda: self.action_requested.emit("problems"))
        self.set_record(None)

    def set_record(self, record, *, project=None, issues=(), pending=False, diagnostic=""):
        """Refresh the evidence without rereading files or recomputing physics."""
        previous = self._check()
        metadata = record.metadata if record is not None else None
        self._record, self._project = record, project
        self._pending, self._diagnostic = bool(pending), str(diagnostic)
        record_key = (
            (
                str(getattr(record, "directory", id(record))),
                metadata.get("metadata_sha256"),
                metadata.get("request_sha256"),
            )
            if record is not None
            else None
        )
        same_record = record_key == self._record_key
        if not same_record:
            self._stage_filter = None
        self._record_key = record_key
        self.issues = tuple(issues)
        self.report = acceptance_report(metadata, project=project, pending=pending)
        identity = self.report.identity
        identity_tooltip = identity
        if record is not None:
            submitted = run_timestamp(record)
            identity += "\nSubmitted: " + (submitted.text if submitted.timestamp is not None else "date unrecorded")
            identity_tooltip += "\nRecorded job submission time. " + submitted.tooltip
        self.identity.setText(identity)
        self.identity.setToolTip(identity_tooltip)
        self.summary.setText(self.report.message)
        self._set_metrics(metadata or {})
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(len(self.report.checks))
        for row, check in enumerate(self.report.checks):
            limit = _text(check.effective_limit, check.unit)
            if check.effective_limit is not None:
                limit = ("≥ " if check.comparison == "lower" else "≤ ") + limit
            values = (
                f"{_STAGE_TITLES[check.stage]} · {check.requirement}",
                _text(check.observed, check.unit),
                limit,
                _text(check.margin, check.unit, signed=True),
                check.status,
                check.scope,
            )
            tooltip = (
                check.scope
                + "\nSaved requested limit: "
                + _text(check.limit, check.unit)
                + "\nEffective acceptance limit: "
                + _text(check.effective_limit, check.unit)
            )
            if check.margin is not None:
                tooltip += "\nRecorded margin: " + _text(check.margin, check.unit, signed=True)
                tooltip += (
                    "\nMargin = observed − minimum"
                    if check.comparison == "lower"
                    else "\nMargin = effective maximum − observed"
                )
            if check.limit is None:
                tooltip += "\nNo acceptance budget is attached to this descriptive metric."
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, check.key)
                if column == 0:
                    item.setIcon(icon(_check_icon(check), badge=_status_badge(check.status)))
                item.setToolTip(tooltip)
                self.table.setItem(row, column, item)
        for stage in self.report.stages:
            button = self.stage_buttons[stage.key]
            button.setText(f"{_STAGE_TITLES[stage.key]}: {stage.status}")
            badge = _status_badge(stage.status)
            button.setIcon(icon(_STAGE_ICONS[stage.key], badge=badge))
            button.setProperty("evidenceBadge", badge)
            button.setAccessibleName(button.text())
            button.setAccessibleDescription(stage.detail)
            button.setToolTip(stage.detail)
            button.setEnabled(any(check.stage == stage.key for check in self.report.checks))
        self.table.blockSignals(False)
        self._apply_filter(preserve_selection=False)
        keys = [check.key for check in self.report.checks]
        selection = (
            previous.key if same_record and previous is not None and previous.key in keys else self.report.focus_key
        )
        if selection in keys and not self.table.isRowHidden(keys.index(selection)):
            self.table.selectRow(keys.index(selection))
        elif any(not self.table.isRowHidden(row) for row in range(len(keys))):
            self.table.selectRow(next(row for row in range(len(keys)) if not self.table.isRowHidden(row)))
        else:
            self.table.clearSelection()
        self.problems.setEnabled(bool(self.issues))
        self._selected()
        self._refresh_colors()

    # Same verb as other saved-run panels for simple workbench integration.
    def set_run(self, record, **kwargs):
        self.set_record(record, **kwargs)

    def _check(self):
        row = self.table.currentRow() if hasattr(self, "table") else -1
        return self.report.checks[row] if 0 <= row < len(self.report.checks) else None

    def _issue(self, check):
        if self._diagnostic or check is None or check.status not in {"Failed", "Unverified", "Not designed"}:
            return None
        matching = [issue for issue in self.issues if check.issue_key and issue.key == check.issue_key]
        if not matching and check.stage == "physical":
            scope = (
                "pipeline_sweep_geometry"
                if check.key == "physical:sweep" or check.scope.startswith("Pipeline sweep")
                else "fabrication_review"
            )
            matching = [
                issue
                for issue in self.issues
                if issue.scope == scope and (not check.body_id or issue.body_id == check.body_id)
            ]
            reason = {
                "bend": "minimum_bend_radius_violation",
                "clearance": "sweep_clearance_violation",
                "penetration": "support_surface_penetration",
                "frame": "unresolved_frame_sampling",
            }.get(check.key.rsplit(":", 1)[-1])
            if reason:
                matching = [issue for issue in matching if issue.key.endswith(":" + reason)]
        if not matching and check.key == "sheet:acceptance":
            matching = [issue for issue in self.issues if issue.scope == "continuous_sheet"]
        errors = [issue for issue in matching if issue.severity == "error"]
        matching = errors or matching
        if self._record is not None:
            from pathlib import Path

            matching = [
                issue
                for issue in matching
                if not issue.run_directory
                or Path(issue.run_directory).resolve() == Path(self._record.directory).resolve()
            ]
        return next((issue for issue in matching if issue.locatable), matching[0] if matching else None)

    def _selected(self):
        check = self._check()
        issue = self._issue(check)
        self.locate.setEnabled(issue is not None and issue.locatable)
        self.locate.setToolTip(
            issue.location_description
            if issue is not None
            else "A recorded spatial witness is required; no location is inferred."
        )
        metadata = self._record.metadata if self._record is not None else {}
        request_saved = any(
            artifact.get("name") == "request.json"
            and isinstance(artifact.get("sha256"), str)
            and len(artifact["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in artifact["sha256"].lower())
            for artifact in metadata.get("artifacts") or ()
            if isinstance(artifact, dict)
        )
        self.settings.setEnabled(check is not None and request_saved and not self._diagnostic)
        self.settings.setToolTip(
            self._diagnostic
            or (
                "Open the verified immutable request recorded for this run."
                if request_saved
                else "A saved request.json with recorded integrity information is required."
            )
        )
        owner = metadata.get("project_id") or (metadata.get("job") or {}).get("project_id")
        matching_owner = self._project is not None and owner is not None and owner == self._project.id
        revision = getattr(self._project, "revision", "unrecorded")
        self.edit_settings.setText(
            f"Edit current channel settings · r{revision}" if matching_owner else "Edit current channel settings"
        )
        self.edit_settings.setAccessibleName(self.edit_settings.text())
        self.edit_settings.setEnabled(
            check is not None and bool(check.settings_node) and matching_owner and not self._diagnostic
        )
        self.edit_settings.setToolTip(
            self._diagnostic
            or (
                f"Edit applied revision {revision}; the saved run and its acceptance limits remain unchanged."
                if matching_owner
                else "Open the saved run's owning channel before editing its current settings."
            )
        )
        if check is None:
            self.details.setText("No saved evidence selected.")
        else:
            text = check.scope
            saved_revision = metadata.get(
                "project_revision", (metadata.get("job") or {}).get("project_revision", "unrecorded")
            )
            text += f"\nSaved source revision {saved_revision}; applied channel revision {revision}."
            if check.limit is not None:
                text += " Saved requested limit: " + _text(check.limit, check.unit) + "."
            if self._pending:
                text += " Current channel edits are pending."
            if check.margin is not None:
                text += "\nRecorded margin: " + _text(check.margin, check.unit, signed=True)
            if issue is not None:
                text += "\n" + issue.explanation + " " + issue.location_description
            elif check.status == "Evaluated":
                text += "\nDescriptive value: an acceptance claim requires its own saved contract and sampling scope."
            elif check.status == "Failed":
                text += (
                    "\nInspect the recorded failure and compare one controlled variant before changing the requirement."
                )
            self.details.setText(text)
        self.details.setToolTip(self.details.text())
        apply_tone(self.details, check.tone if check is not None else "neutral")

    def _select_stage(self, key):
        self._stage_filter = None if self._stage_filter == key else key
        self._apply_filter()

    def _apply_filter(self, *_args, preserve_selection=True):
        """Hide presentation rows only; saved checks and their row identities stay intact."""
        visible = []
        for row, check in enumerate(self.report.checks):
            stage_matches = self._stage_filter is None or check.stage == self._stage_filter
            needs_attention = check.status in {"Failed", "Unverified", "Not designed", "Not run"}
            show = stage_matches and (self.show_all.isChecked() or self._stage_filter is not None or needs_attention)
            self.table.setRowHidden(row, not show)
            if show:
                visible.append(row)
        for key, button in self.stage_buttons.items():
            button.setChecked(key == self._stage_filter)
        description = (
            f"{_STAGE_TITLES[self._stage_filter]} checks · click stage again to clear"
            if self._stage_filter is not None
            else "All saved checks"
            if self.show_all.isChecked()
            else "Failures and evidence gaps"
        )
        self.filter_summary.setText(f"{description} · {len(visible)} / {len(self.report.checks)}")
        if preserve_selection and self.table.currentRow() not in visible:
            if visible:
                row = next((row for row in visible if self.report.checks[row].status == "Failed"), visible[0])
                self.table.selectRow(row)
            else:
                self.table.clearSelection()
                self.table.setCurrentCell(-1, -1)
                self._selected()

    def _set_metrics(self, metadata):
        """Display only recorded finite quantities, with the recorded electrical/current scope."""

        def number(value):
            return (
                float(value)
                if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                else None
            )

        rows = []
        wires = metadata.get("wire_fields") or {}
        kind, evidence = next(
            ((kind, wires[kind]) for kind in ("routed", "contour") if (wires.get(kind) or {}).get("available") is True),
            (None, {}),
        )
        scope = "Connected conductor" if kind == "routed" else "Original contours"
        original = evidence.get("at_stream_current") or {}
        if original.get("available") is not True:
            original = evidence if evidence.get("current_refitted") is False else {}
        current = number(original.get("current_A"))
        if current is not None:
            rows.append(f"{scope}: original wire current {current:.5g} A")
        if evidence.get("current_refitted") is True and (current := number(evidence.get("current_A"))) is not None:
            rows.append(f"{scope}: saved refitted current {current:.5g} A")
        summary = metadata.get("summary") or {}
        if summary.get("scope") == "fabrication_review":
            circuit = []
            for label, key, factor, unit in (
                ("I", "current_a", 1, "A"),
                ("R", "resistance_ohm", 1, "Ω"),
                ("L", "inductance_h", 1e6, "µH"),
            ):
                value = number(summary.get(key))
                if value is not None and math.isfinite(value * factor):
                    circuit.append(f"{label} {value * factor:.5g} {unit}")
            if circuit:
                rows.append("Reviewed circuit: " + " · ".join(circuit))
            reviewed_power = number(summary.get("fabricated_power_w"))
            if reviewed_power is not None:
                rows.append(f"Reviewed conductor Joule power {reviewed_power:.5g} W")
        power = number(summary.get("power_w"))
        if power is not None:
            rows.append(f"Sheet power {power:.5g} W")
        self.metrics.setText(" · ".join(rows))
        self.metrics.setVisible(bool(rows))
        self.metrics.setToolTip(
            "Recorded quantities only. Sheet power precedes contouring and routing. Reviewed circuit estimates apply to the saved complete modeled circuit.\n"
            f"Wire current source: {evidence.get('current_source', 'unrecorded')}; refitted: {evidence.get('current_refitted', 'unrecorded')}."
        )

    def _locate(self):
        issue = self._issue(self._check())
        if issue is not None and issue.locatable and self.locate.isEnabled():
            self.witness_requested.emit(issue)

    def _settings(self):
        check = self._check()
        if check is not None and self.settings.isEnabled():
            self.saved_settings_requested.emit(self._record, check.settings_node)

    def _edit_settings(self):
        check = self._check()
        if check is not None and self.edit_settings.isEnabled():
            self.action_requested.emit(check.settings_node)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "table"):
            self._responsive_columns()
            QTimer.singleShot(0, self, self._reveal_selection)

    def _responsive_columns(self):
        """Keep evaluation scope visible at desktop and smaller window sizes."""
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        available = max(580, self.table.viewport().width())
        for column, share, minimum in (
            (0, 0.26, 150),
            (1, 0.105, 70),
            (2, 0.105, 70),
            (3, 0.105, 70),
            (4, 0.105, 75),
            (5, 0.32, 170),
        ):
            self.table.setColumnHidden(column, False)
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(column, max(minimum, int(available * share)))
        self.table.resizeRowsToContents()
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, min(110, self.table.rowHeight(row)))

    def _reveal_selection(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.scrollToItem(self.table.item(row, 0))

    def _refresh_colors(self):
        apply_tone(self.identity, self.report.identity_tone)
        apply_tone(
            self.summary,
            "error"
            if any(check.status == "Failed" for check in self.report.checks)
            else "warning"
            if self.report.focus_key
            else "neutral",
        )
        for row, check in enumerate(self.report.checks):
            for column, value in ((1, check.observed), (2, check.effective_limit), (3, check.margin)):
                item = self.table.item(row, column)
                if item is not None:
                    item.setForeground(
                        self.table.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
                        if value is None
                        else item_foreground(check.tone if column == 3 else "neutral", self.table.palette())
                    )
            for column in (3, 4):
                item = self.table.item(row, column)
                if item is not None and (column != 3 or check.margin is not None):
                    item.setForeground(item_foreground(check.tone, self.table.palette()))
        for stage in self.report.stages:
            tone = (
                "error"
                if stage.status == "Failed"
                else "success"
                if stage.status.startswith("Passed")
                else "info"
                if stage.status == "Evaluated"
                else "warning"
                if stage.status == "Unverified"
                else "neutral"
            )
            button = self.stage_buttons[stage.key]
            palette = QPalette(button.palette())
            contrast_palette = QPalette(palette)
            background = palette.color(QPalette.ColorRole.Button)
            contrast_palette.setColor(QPalette.ColorRole.Window, background)
            contrast_palette.setColor(QPalette.ColorRole.Base, background)
            palette.setColor(QPalette.ColorRole.ButtonText, tone_color(tone, contrast_palette))
            button.setPalette(palette)
            button.setProperty("semanticTone", tone)
        check = self._check()
        apply_tone(self.details, check.tone if check is not None else "neutral")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        ) and hasattr(self, "table"):
            QTimer.singleShot(0, self, self._refresh_colors)
