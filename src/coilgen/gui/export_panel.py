"""Selection of immutable conductor solids; export never changes the design."""

from PySide6.QtCore import QEvent, QObject, QRunnable, QSignalBlocker, QSize, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .colors import apply_tone, item_foreground
from .icons import icon
from .run_context import run_compatibility


class GeometryExportPanel(QWidget):
    export_requested = Signal(object, str, bool)
    review_requested = Signal()
    problems_requested = Signal()

    def __init__(
        self, record=None, options=(), *, diagnostic="", parent=None, project=None, assembly=None, pending=False
    ):
        super().__init__(parent)
        self._has_record = record is not None
        self._record = record
        self._diagnostic = str(diagnostic)
        self._exporting = False
        self._review_allowed = False
        self.options = tuple(options)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scope = QLabel()
        self.scope.setWordWrap(True)
        self.scope.setTextFormat(Qt.TextFormat.PlainText)
        self.scope.setObjectName("geometry_export_scope")
        layout.addWidget(self.scope)
        metadata = record.metadata if record is not None else {}
        summary = metadata.get("summary") or {}
        review = metadata.get("fabrication") or {}
        revision = metadata.get("project_revision", (metadata.get("job") or {}).get("project_revision", "unverified"))
        if revision is None or (isinstance(revision, str) and not revision.strip()):
            revision = "unverified"
        if record is None:
            self.scope.setText("No saved run selected. Compute a design and review its conductor geometry first.")
        else:
            self.scope.setText(
                f"Saved run: {metadata.get('name', record.directory.name)} · revision {revision}. "
                "Exports use this run's exact solid geometry. Geometry availability, field acceptance and construction release are separate."
            )
        self.status = QLabel(diagnostic)
        self.status.setObjectName("geometry_export_status")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("geometry_export_bodies")
        self.table.setHorizontalHeaderLabels(["Conductor", "Units", "Checks", "Extent [mm]"])
        for column, tooltip in enumerate(
            (
                "Saved conductor body and file format. Hover over a body for its exact artifact name.",
                "STL uses the selected coordinate units; STEP retains its declared source units.",
                "Recorded geometry checks within the saved scope; this is not field acceptance or construction release.",
                "Physical x, y and z dimensions in millimetres, independent of the exported file units.",
            )
        ):
            self.table.horizontalHeaderItem(column).setToolTip(tooltip)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(160)
        self.units = QComboBox()
        self.units.setObjectName("geometry_export_units")
        self.units.setAccessibleName("STL coordinate units")
        self.units.addItem("STL coordinates: millimetres [mm]", "mm")
        self.units.addItem("STL coordinates: metres [m]", "m")
        self.units.setToolTip("STL has no unit declaration. Scaling is explicit and recorded in the CAD manifest.")
        self.unit_note = QLabel(
            "STL coordinates are scaled from the saved SI geometry to the selected units. "
            "STEP files retain their existing declared units. Extents below are physical dimensions in millimetres."
        )
        self.unit_note.setWordWrap(True)
        self.bounds = QLabel()
        self.bounds.setObjectName("geometry_export_bounds")
        self.bounds.setWordWrap(True)
        self.bounds.setTextFormat(Qt.TextFormat.PlainText)
        self.include_project = QCheckBox("Include project JSON and results")
        self.include_project.setIcon(icon("project-json"))
        self.include_project.setIconSize(QSize(18, 18))
        self.include_project.setAccessibleName(self.include_project.text())
        self.include_project.setChecked(True)
        self.include_project.setObjectName("geometry_export_include_project")
        self.include_project.setToolTip(
            "Save one self-contained project JSON containing model inputs and completed results with the CAD export."
        )
        self.companion_context = QLabel()
        self.companion_context.setObjectName("geometry_export_companion_context")
        self.companion_context.setTextFormat(Qt.TextFormat.PlainText)
        self.companion_context.setWordWrap(True)
        self.include_project.toggled.connect(
            lambda checked: self.companion_context.setVisible(checked and bool(self.companion_context.text()))
        )
        self.export_button = QPushButton("Export selected geometry…")
        self.export_button.setObjectName("export_selected_geometry")
        self.export_button.clicked.connect(self._request_export)
        self.review_button = QPushButton("Review current revision")
        self.review_button.setObjectName("export_run_review")
        self.review_button.clicked.connect(self.review_requested)
        self.review_scope = QLabel()
        self.review_scope.setObjectName("geometry_export_review_scope")
        self.review_scope.setTextFormat(Qt.TextFormat.PlainText)
        self.review_scope.setWordWrap(True)
        self.problems_button = QPushButton("Inspect geometry problems")
        self.problems_button.setObjectName("export_inspect_problems")
        self.problems_button.clicked.connect(self.problems_requested)
        for button, name in (
            (self.export_button, "export-conductor"),
            (self.review_button, "constraint"),
            (self.problems_button, "collision"),
        ):
            button.setIcon(icon(name))
            button.setIconSize(QSize(20, 20))
            button.setAccessibleName(button.text())
        if not self.options:
            if not diagnostic:
                self.status.setText(
                    "No solid STL or solid STEP was saved. "
                    + (
                        "Recorded fabrication checks failed; inspect the problems before recomputing."
                        if review.get("ready") is False
                        else "Review the current revision to produce reviewed conductor solids."
                    )
                )
            apply_tone(self.status, "error" if diagnostic or review.get("ready") is False else "warning")
        else:
            failed = (
                review.get("ready") is False
                or any(option.check_status == "failed" for option in self.options)
                or summary.get("sheet_accepted") is False
            )
            self.status.setText(
                f"{len(self.options)} saved solid files available. "
                + (
                    "Recorded checks failed; these files remain available as inspection geometry."
                    if failed
                    else "Some files are inspection geometry with unverified checks."
                    if any(option.inspection_only for option in self.options)
                    else "Inspect the separate field and circuit checks before using the geometry."
                )
            )
            apply_tone(
                self.status,
                "error"
                if failed
                else "warning"
                if any(option.inspection_only for option in self.options)
                else "success",
            )
        self.problems_button.setVisible(review.get("ready") is False or bool(review.get("failure_reasons")))
        self.review_button.setVisible(record is not None and not self.options and review.get("ready") is not False)
        self.review_scope.setVisible(not self.review_button.isHidden())
        if self._diagnostic:
            self.status.setText(self._diagnostic)
            apply_tone(self.status, "error")
        layout.addWidget(self.problems_button)
        layout.addWidget(self.review_button)
        layout.addWidget(self.review_scope)
        layout.addWidget(self.units)
        layout.addWidget(self.include_project)
        layout.addWidget(self.companion_context)
        layout.addWidget(self.export_button)
        layout.addWidget(self.table)
        layout.addWidget(self.bounds)
        layout.addWidget(self.unit_note)
        for control in (self.units, self.include_project, self.table, self.bounds, self.unit_note):
            control.setVisible(bool(self.options))
        release = QLabel(
            "Construction release: " + ("Released" if summary.get("construction_released") is True else "Not released")
        )
        release.setObjectName("geometry_export_release")
        layout.addWidget(release)
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(self.options))
            for row, option in enumerate(self.options):
                name = QTableWidgetItem(f"{option.body_id} · {option.format.upper()}")
                name.setIcon(icon("conductor-sweep"))
                name.setData(Qt.ItemDataRole.UserRole, option.name)
                name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                name.setCheckState(Qt.CheckState.Checked if option.format == "stl" else Qt.CheckState.Unchecked)
                name.setToolTip(option.name)
                self.table.setItem(row, 0, name)
                self.table.setItem(row, 1, QTableWidgetItem())
                status = QTableWidgetItem(
                    ("Inspection · " if option.inspection_only else "") + option.check_status.capitalize()
                )
                status.setForeground(
                    item_foreground(
                        "success"
                        if option.check_status == "passed"
                        else "error"
                        if option.check_status == "failed"
                        else "neutral",
                        self.palette(),
                    )
                )
                status.setToolTip(option.scope)
                status.setIcon(
                    icon(
                        "constraint",
                        badge={"passed": "passed", "failed": "failed"}.get(option.check_status, "unverified"),
                    )
                )
                self.table.setItem(row, 2, status)
                extent = [1000 * (hi - lo) for lo, hi in zip(*option.bounds_m)]
                self.table.setItem(row, 3, QTableWidgetItem(" × ".join(f"{value:.5g}" for value in extent)))
        self.table.itemChanged.connect(self._selection_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.currentCellChanged.connect(self._selection_changed)
        self.units.currentIndexChanged.connect(self._selection_changed)
        self._selection_changed()
        self.set_context(project, assembly=assembly, pending=pending)

    def set_context(self, project=None, *, assembly=None, pending=False):
        """Distinguish immutable CAD source from the current project companion."""
        metadata = self._record.metadata if self._record is not None else {}
        owner = metadata.get("project_id") or (metadata.get("job") or {}).get("project_id")
        matching_owner = project is not None and owner is not None and owner == project.id
        self._review_allowed = matching_owner and not self._diagnostic
        self.review_button.setEnabled(self._review_allowed and not self._exporting)
        review_text = (
            self._diagnostic
            if self._diagnostic
            else f"Review computes the current channel, applied revision {project.revision}. "
            "Pending edits are applied first; the selected saved run remains immutable."
            if matching_owner
            else "Open the saved run's owning channel to review its current revision."
        )
        self.review_scope.setText(review_text)
        self.review_button.setToolTip(review_text)
        if self._record is None or project is None and assembly is None:
            self.companion_context.clear()
            self.companion_context.hide()
            return
        current = (
            next((channel.project for channel in assembly.channels if channel.project.id == owner), None)
            if assembly is not None
            else project
        )
        current = current or project
        revision = metadata.get("project_revision", (metadata.get("job") or {}).get("project_revision", "unrecorded"))
        context, label = run_compatibility(metadata, current)
        current_revision = current.revision if current is not None else "unverified"
        companion = (
            f"Current assembly ({len(assembly.channels)} channels)" if assembly is not None else "Current project"
        )
        text = f"CAD source: saved revision {revision}. {companion} companion: applied revision {current_revision} · {label.lower()}."
        if pending:
            text += " Pending edits will be applied before saving the companion."
        if context == "earlier":
            text += " The project contains the current settings and saved earlier result; CAD remains the selected run's geometry."
        self.companion_context.setText(text)
        self.companion_context.setToolTip(text)
        apply_tone(self.companion_context, "warning" if context != "current" or pending else "info")
        self.companion_context.setVisible(self.include_project.isChecked())

    def selected_names(self):
        return tuple(
            self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            for row in range(self.table.rowCount())
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        )

    def _selection_changed(self, *_args):
        with QSignalBlocker(self.table):
            for row, option in enumerate(self.options):
                self.table.item(row, 1).setText(
                    f"STL · {self.units.currentData()}" if option.format == "stl" else f"STEP · {option.source_unit}"
                )
        names = self.selected_names()
        self.export_button.setEnabled(self._has_record and bool(names) and not self._exporting and not self._diagnostic)
        self.export_button.setToolTip(
            self._diagnostic or "Export the selected saved solids and their recorded provenance."
        )
        self.export_button.setText(
            "Exporting saved geometry…"
            if self._exporting
            else f"Export {len(names)} selected solid files…"
            if names
            else "Select a solid to export"
            if self.options
            else "No saved solids"
        )
        self.export_button.setAccessibleName(self.export_button.text())
        row = self.table.currentRow()
        selected = (
            self.options[row]
            if 0 <= row < len(self.options)
            else next((option for option in self.options if option.name in names), None)
        )
        if selected is None:
            self.bounds.setText("Select the saved solid bodies and formats to export.")
        else:
            self.bounds.setText(
                f"{selected.body_id} · world bounds [mm]: "
                + "; ".join(
                    f"{axis} {lo * 1000:.5g}…{hi * 1000:.5g}" for axis, lo, hi in zip("xyz", *selected.bounds_m)
                )
            )

    def _request_export(self):
        names = self.selected_names()
        if self._has_record and names and not self._exporting and not self._diagnostic:
            self.export_requested.emit(names, self.units.currentData(), self.include_project.isChecked())

    def set_exporting(self, exporting):
        self._exporting = bool(exporting)
        self.review_button.setEnabled(self._review_allowed and not self._exporting)
        self._selection_changed()

    def _refresh_colors(self):
        apply_tone(self.status, self.status.property("semanticTone") or "neutral")
        with QSignalBlocker(self.table):
            for row, option in enumerate(self.options):
                item = self.table.item(row, 2)
                if item is not None:
                    tone = {"passed": "success", "failed": "error"}.get(option.check_status, "neutral")
                    item.setForeground(item_foreground(tone, self.table.palette()))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "status") and hasattr(self, "table"):
                QTimer.singleShot(0, self, self._refresh_colors)


class _ExportSignals(QObject):
    finished = Signal(object, str)


class GeometryExportTask(QRunnable):
    """Write an already detached snapshot without calling any Qt-owned model."""

    def __init__(self, record, directory, names, unit, project=None):
        super().__init__()
        self.signals = _ExportSignals()
        self.record, self.directory, self.names, self.unit, self.project = record, directory, names, unit, project

    def run(self):
        from ..application.cad import export_geometry

        try:
            package = export_geometry(self.record, self.directory, self.names, unit=self.unit, project=self.project)
        except Exception as exc:
            self.signals.finished.emit(None, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(package, "")
