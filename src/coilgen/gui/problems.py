"""Object-linked engineering problems with honest spatial-witness availability."""

from PySide6.QtCore import QEvent, QSize, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..application.diagnostics import run_problems
from .colors import apply_tone, item_foreground
from .icons import icon


def _problem_icon(issue):
    """Describe only unambiguous recorded diagnostic categories."""
    reason = issue.key.rsplit(":", 1)[-1].lower()
    name = issue.title.lower().replace("_", " ")
    if reason in {"minimum_bend_radius_violation", "min_bend_radius_violation"} or "bend radius" in name:
        return "bend-radius"
    if reason in {"sweep_self_intersection", "support_surface_penetration"} or any(
        term in name for term in ("intersection", "penetration", "penetrates its support")
    ):
        return "collision"
    if reason in {"route_clearance_violation", "sweep_clearance_violation", "assembly_clearance_violation"} or any(
        term in name for term in ("clearance", "wire spacing", "wire separation")
    ):
        return "wire-spacing"
    if "field" in name and any(term in name for term in ("error", "residual", "rms", "standard deviation")):
        return "field-error"
    if "current density" in name:
        return "current-density"
    if "terminal" in name or "terminal" in reason:
        return "terminals"
    return "constraint"


class ProblemsPanel(QWidget):
    locate_requested = Signal(object)
    navigate_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.issues = ()
        self._diagnostic = ""
        self._saved_settings_reason = "Select a verified saved run first."
        layout = QVBoxLayout(self)
        self.summary = QLabel("Select a completed run to inspect its recorded engineering problems.")
        self.summary.setObjectName("problem_summary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        groups = QHBoxLayout()
        groups.addWidget(QLabel("Recorded scope / body"))
        self.group_filter = QComboBox(self)
        self.group_filter.setObjectName("problem_group_filter")
        self.group_filter.setToolTip(
            "Inspect one saved scope/body group without combining distinct engineering checks."
        )
        self.group_filter.addItem("All recorded groups", None)
        self.group_filter.currentIndexChanged.connect(self._apply_filter)
        groups.addWidget(self.group_filter, 1)
        layout.addLayout(groups)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("problems_table")
        self.table.setHorizontalHeaderLabels(["Problem", "Geometry / body", "Value / limit", "Location"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.details = QLabel("No run selected.")
        self.details.setObjectName("problem_details")
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.locate = QPushButton("Locate evidence")
        self.locate.setObjectName("problem_locate")
        self.navigate = QPushButton("Inspect saved settings")
        self.navigate.setObjectName("problem_settings")
        for button, name in ((self.locate, "locate"), (self.navigate, "settings")):
            button.setIcon(icon(name))
            button.setIconSize(QSize(18, 18))
            button.setAccessibleName(button.text())
        actions.addWidget(self.locate)
        actions.addWidget(self.navigate)
        actions.addStretch()
        layout.addLayout(actions)
        self.table.itemSelectionChanged.connect(self._selected)
        self.locate.clicked.connect(self._locate)
        self.navigate.clicked.connect(self._navigate)
        self._selected()

    def set_run(self, record, *, diagnostic=""):
        self._diagnostic = diagnostic
        has_request = record is not None and any(
            artifact.get("name") == "request.json" and artifact.get("sha256")
            for artifact in record.metadata.get("artifacts", ())
        )
        self._saved_settings_reason = diagnostic or (
            "" if has_request else "The original request.json was not saved with verified integrity."
        )
        self.issues = run_problems(record) if record is not None else ()
        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = sum(issue.severity == "warning" for issue in self.issues)
        self.summary.setText(
            f"{errors} errors · {warnings} warnings · Select a problem to locate saved evidence or inspect its saved settings."
            if self.issues
            else "No recorded problems. Checks that were not run remain unverified."
            if record is not None
            else "Select a completed run to inspect its recorded engineering problems."
        )
        self.table.setRowCount(len(self.issues))
        for row, issue in enumerate(self.issues):
            measured = "—" if issue.value is None else f"{issue.value:.6g}"
            budget = "—" if issue.limit is None else f"{issue.limit:.6g}"
            scope = {"pipeline_sweep_geometry": "Pipeline sweep", "fabrication_review": "Fabrication review"}.get(
                issue.scope, issue.scope.replace("_", " ")
            )
            values = (
                f"{issue.severity.capitalize()}: {issue.title}",
                " / ".join(value for value in (scope, issue.body_id) if value) or "Run",
                f"{measured} / {budget} {issue.unit}",
                "Recorded evidence" if issue.locatable else "Not recorded",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setIcon(
                        icon(_problem_icon(issue), badge="failed" if issue.severity == "error" else "unverified")
                    )
                item.setToolTip(issue.explanation)
                self.table.setItem(row, column, item)
        previous = self.group_filter.currentData()
        self.group_filter.blockSignals(True)
        self.group_filter.clear()
        self.group_filter.addItem(f"All recorded groups ({len(self.issues)})", None)
        groups = {}
        for issue in self.issues:
            key = (issue.scope, issue.body_id)
            groups[key] = groups.get(key, 0) + 1
        for key, count in groups.items():
            label = " / ".join(value.replace("_", " ") for value in key if value) or "Run"
            self.group_filter.addItem(f"{label} ({count})", key)
        self.group_filter.setCurrentIndex(max(0, self.group_filter.findData(previous)))
        self.group_filter.blockSignals(False)
        self._responsive_columns()
        if self.issues:
            self.table.selectRow(0)
        else:
            self.details.setText("No recorded problems." if record is not None else "No run selected.")
        self._selected()
        self._apply_filter()
        self._refresh_colors()

    def _apply_filter(self, *_args):
        if not hasattr(self, "table"):
            return
        selected = self.group_filter.currentData()
        visible = []
        for row, issue in enumerate(self.issues):
            show = selected is None or selected == (issue.scope, issue.body_id)
            self.table.setRowHidden(row, not show)
            if show:
                visible.append(row)
        if self.table.currentRow() not in visible:
            if visible:
                self.table.selectRow(visible[0])
            else:
                self.table.setCurrentCell(-1, -1)
                self._selected()

    def _responsive_columns(self):
        available = max(500, self.table.viewport().width())
        header = self.table.horizontalHeader()
        for column, share, minimum in ((0, 0.40, 180), (1, 0.29, 145), (2, 0.17, 90), (3, 0.14, 90)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(column, max(minimum, int(available * share)))
        self.table.resizeRowsToContents()
        for row in range(self.table.rowCount()):
            self.table.setRowHeight(row, min(100, self.table.rowHeight(row)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "table"):
            self._responsive_columns()

    def _issue(self):
        row = self.table.currentRow()
        return self.issues[row] if 0 <= row < len(self.issues) else None

    def _selected(self):
        issue = self._issue()
        self.locate.setEnabled(issue is not None and issue.locatable and not self._diagnostic)
        self.locate.setToolTip(
            self._diagnostic
            or (
                "Focus the 3D view on the recorded failure witness."
                if issue is not None and issue.locatable
                else "A spatial failure witness must be saved before this problem can be located."
            )
        )
        self.navigate.setEnabled(issue is not None and bool(issue.settings_node) and not self._saved_settings_reason)
        self.navigate.setToolTip(self._saved_settings_reason or "Inspect the verified request used by this saved run.")
        if issue is not None:
            self.details.setText(issue.explanation + "\n" + issue.location_description)
        apply_tone(self.details, issue.severity if issue is not None else "neutral")

    def _locate(self):
        issue = self._issue()
        if issue is not None and issue.locatable and not self._diagnostic:
            self.locate_requested.emit(issue)

    def _navigate(self):
        issue = self._issue()
        if issue is not None and not self._saved_settings_reason:
            self.navigate_requested.emit(issue.settings_node)

    def _refresh_colors(self):
        tone = (
            "error"
            if any(issue.severity == "error" for issue in self.issues)
            else "warning"
            if self.issues
            else "neutral"
        )
        apply_tone(self.summary, tone)
        for row, issue in enumerate(self.issues):
            item = self.table.item(row, 0)
            if item is not None:
                item.setForeground(item_foreground(issue.severity, self.table.palette()))
        issue = self._issue()
        apply_tone(self.details, issue.severity if issue is not None else "neutral")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "table"):
                # Qt propagates a parent's palette to its children after the
                # parent's changeEvent. Refresh against their settled palette.
                QTimer.singleShot(0, self, self._refresh_colors)
