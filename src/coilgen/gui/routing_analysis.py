"""Prominent read-only inspection of one saved conductor support at a time."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..application.routing_analysis import draw_routing_analysis, draw_routing_closeup
from .field_analysis import _FigurePanel


class RoutingAnalysisDialog(QDialog):
    def __init__(self, analysis, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.setObjectName("routing_analysis_dialog")
        self.setWindowTitle("Interconnections · " + analysis.title)
        self.resize(1180, 890)
        layout = QVBoxLayout(self)
        self.identity = QLabel(
            f"{analysis.title} · fabrication review: {analysis.fabrication_status}\n{analysis.run_directory}"
        )
        self.identity.setTextFormat(Qt.TextFormat.PlainText)
        self.identity.setWordWrap(True)
        self.identity.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.identity)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Support"))
        self.support = QComboBox()
        self.support.setObjectName("routing_support")
        for part in sorted({source.part_index for source in analysis.sources}):
            self.support.addItem(f"Support {part + 1}", part)
        controls.addWidget(self.support)
        controls.addWidget(QLabel("Geometry"))
        self.source = QComboBox()
        self.source.setObjectName("routing_source")
        controls.addWidget(self.source, 1)
        self.envelopes = QCheckBox("Physical conductor envelope")
        self.envelopes.setToolTip("Exact saved profile rings; no width is invented when final frames are unavailable.")
        controls.addWidget(self.envelopes)
        self.export = QPushButton("Export figure…")
        controls.addWidget(self.export)
        layout.addLayout(controls)
        views = QHBoxLayout()
        views.addWidget(QLabel("View"))
        self.view = QComboBox()
        self.view.setObjectName("routing_view")
        self.view.addItems(("Whole support · all located checks", "Selected failure close-up"))
        views.addWidget(self.view)
        views.addWidget(QLabel("Close-up half-width [mm]"))
        self.radius = QDoubleSpinBox()
        self.radius.setObjectName("routing_closeup_radius_mm")
        self.radius.setRange(0.0, 10000.0)
        self.radius.setDecimals(2)
        self.radius.setSpecialValueText("Auto")
        self.radius.setToolTip(
            "Physical world-space half-width; changes only the camera, never geometry or acceptance limits."
        )
        views.addWidget(self.radius)
        views.addStretch(1)
        layout.addLayout(views)
        self.scope = QLabel()
        self.scope.setWordWrap(True)
        self.scope.setTextFormat(Qt.TextFormat.PlainText)
        self.scope.setObjectName("routing_scope")
        layout.addWidget(self.scope)
        self.plot = _FigurePanel(self)
        layout.addWidget(self.plot, 1)
        self.problems = QTableWidget(0, 3)
        self.problems.setObjectName("routing_witnesses")
        self.problems.setHorizontalHeaderLabels(("Recorded body check", "Value / limit", "Spatial evidence"))
        self.problems.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.problems.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.problems.setMaximumHeight(150)
        self.problems.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.problems)
        self.witness = QLabel("Select a recorded check to show its authenticated spatial witness.")
        self.witness.setTextFormat(Qt.TextFormat.PlainText)
        self.witness.setWordWrap(True)
        layout.addWidget(self.witness)
        self.support.currentIndexChanged.connect(self._support_changed)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.envelopes.toggled.connect(self._draw)
        self.problems.currentCellChanged.connect(self._problem_changed)
        self.view.currentIndexChanged.connect(self._draw)
        self.radius.valueChanged.connect(self._draw)
        self.export.clicked.connect(self._export)
        self._support_changed()

    @property
    def selected_source(self):
        key = self.source.currentData()
        return next((source for source in self.analysis.sources if source.key == key), None)

    def _support_changed(self, *_args):
        self.source.blockSignals(True)
        self.source.clear()
        candidates = [source for source in self.analysis.sources if source.part_index == self.support.currentData()]
        for source in candidates:
            self.source.addItem(source.label + " · " + (source.body_id or source.key), source.key)
        # Present the physical review first while leaving the exact raw route accessible.
        index = next((index for index, source in enumerate(candidates) if source.label.startswith("Strict")), 0)
        self.source.setCurrentIndex(index)
        self.source.blockSignals(False)
        self._source_changed()

    def _source_changed(self, *_args):
        source = self.selected_source
        if source is None:
            return
        current = (
            f"{source.current_A:+.5g} A"
            if source.current_A is not None
            else "current not recorded; arrows show path direction"
        )
        self.scope.setText(
            f"{source.status} · {current}\n{source.circuit_status}\n{source.support.scope} {source.scope}"
        )
        self.envelopes.setEnabled(source.rings_m is not None)
        self.envelopes.blockSignals(True)
        self.envelopes.setChecked(source.rings_m is not None)
        self.envelopes.blockSignals(False)
        self.problems.blockSignals(True)
        self.problems.setRowCount(len(source.issues))
        for index, issue in enumerate(source.issues):
            scale, unit = (1e3, "mm") if issue.unit == "m" else (1.0, issue.unit)
            value = "Not recorded" if issue.value is None else f"{issue.value * scale:.5g} {unit}"
            if issue.limit is not None:
                value += f" / {issue.limit * scale:.5g} {unit}"
            description = issue.location_description if issue.points_m else "No saved spatial witness"
            for column, text in enumerate((issue.title, value, description)):
                item = QTableWidgetItem(text)
                item.setToolTip(issue.explanation + "\n" + issue.scope + "\n" + issue.location_description)
                self.problems.setItem(index, column, item)
        self.problems.resizeColumnsToContents()
        self.problems.setCurrentCell(-1, -1)
        self.problems.blockSignals(False)
        self.view.setCurrentIndex(0)
        self._draw()

    def _problem_changed(self, *_args):
        source = self.selected_source
        index = self.problems.currentRow()
        located = source is not None and 0 <= index < len(source.issues) and bool(source.issues[index].points_m)
        self.view.setCurrentIndex(1 if located else 0)
        self._draw()

    def _draw(self, *_args):
        source = self.selected_source
        if source is None:
            return
        index = self.problems.currentRow()
        issue = source.issues[index] if 0 <= index < len(source.issues) else None
        closeup = self.view.currentIndex() == 1 and issue is not None and bool(issue.points_m)
        self.radius.setEnabled(closeup)
        draw = draw_routing_closeup if closeup else draw_routing_analysis
        options = {"radius_m": self.radius.value() / 1e3 or None} if closeup else {}
        envelope_error = ""
        try:
            draw(
                self.plot.figure,
                source,
                issue=issue if closeup else None,
                **options,
                envelopes=self.envelopes.isChecked() and source.rings_m is not None,
            )
        except ValueError as error:
            envelope_error = str(error)
            draw(self.plot.figure, source, issue=issue if closeup else None, **options, envelopes=False)
        self.witness.setText(
            issue.explanation + " " + issue.location_description
            if issue
            else "Select a recorded check to locate its saved witness. Projection crossings are not physical collision evidence; "
            "capsule overlaps remain conservative screens. External feeds are shown only when explicitly saved."
        )
        if envelope_error:
            self.witness.setText("Physical envelope display incomplete: " + envelope_error + " " + self.witness.text())
        self.plot.draw()

    def _export(self):
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Export interconnection figure", "interconnections.png", "PNG image (*.png)"
        )
        if path:
            self.plot.save_png(path)
