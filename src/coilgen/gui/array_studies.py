"""Array current fitting and training/held-out cohort studies."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..application.array_studies import (
    ArrayCase,
    ArrayChannel,
    ArrayStudySpec,
    CohortControls,
    evaluate_array,
    synthesize_cohort,
    synthesize_contour_cohort,
)
from ..application.gradient_studies import RigidTransform, _saved_winding
from ..application.service import DesignService
from ..persistence import atomic_json, json_value
from .gradient_tools import EngineeringDialog, _matrix, _number, _values


class ArrayCaseDialog(EngineeringDialog):
    def __init__(self, parent=None):
        super().__init__("Import registered array field case", parent)
        self.filename = QLineEdit()
        choose = QPushButton("Choose CSV…")
        choose.clicked.connect(self._choose)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.addWidget(self.filename)
        layout.addWidget(choose)
        self.form.addRow("Field map", row)
        self.name = QLineEdit()
        self.subject = QLineEdit("research")
        self.form.addRow("Case name", self.name)
        self.form.addRow("Subject / specimen", self.subject)
        self.units, self.fields = QComboBox(), QComboBox()
        self.units.addItems(["m", "mm", "cm"])
        self.fields.addItems(["T", "mT", "uT"])
        self.form.addRow("Coordinate unit", self.units)
        self.form.addRow("Field unit", self.fields)
        self.membership = QComboBox()
        self.membership.addItem("Training", "training")
        self.membership.addItem("Held out", "held_out")
        self.form.addRow("Membership", self.membership)
        self.disturbance = QCheckBox("Map is a disturbance to cancel (negate to obtain desired coil field)")
        self.disturbance.setChecked(True)
        self.form.addRow(self.disturbance)
        self.direction = QLineEdit("0 0 1")
        self.rotation = QPlainTextEdit("1 0 0\n0 1 0\n0 0 1")
        self.rotation.setMaximumHeight(80)
        self.translation = QLineEdit("0 0 0")
        self.form.addRow("Projection direction in source frame", self.direction)
        self.form.addRow("Source → world rotation", self.rotation)
        self.form.addRow("Source → world translation [m]", self.translation)
        self.rms, self.peak = _number(0), _number(0)
        self.form.addRow("RMS error limit [µT] · 0 unspecified", self.rms)
        self.form.addRow("Peak error limit [µT] · 0 unspecified", self.peak)
        self.note(
            "CSV: x,y,z,B or x,y,z,Bx,By,Bz. Coordinates and vector components rotate together. Named SI headers require metre/tesla units. Scalar maps retain their scalar values and rotate the declared projection direction."
        )
        self.resize(690, 690)

    def _choose(self):
        name, _ = QFileDialog.getOpenFileName(self, "Registered field map", "", "Field CSV (*.csv *.txt)")
        if name:
            self.filename.setText(name)
            self.name.setText(Path(name).stem)

    def value(self):
        direction = _matrix(self.direction.text(), 3, "Projection direction")
        translation = _matrix(self.translation.text(), 3, "Translation")
        if len(direction) != 1 or len(translation) != 1:
            raise ValueError("Enter one direction and one translation vector.")
        case = ArrayCase.from_csv(
            self.filename.text(),
            name=self.name.text(),
            subject=self.subject.text(),
            membership=self.membership.currentData(),
            length_unit=self.units.currentText(),
            field_unit=self.fields.currentText(),
            disturbance=self.disturbance.isChecked(),
            direction=direction[0],
            transform=RigidTransform(_matrix(self.rotation.toPlainText(), 3, "Rotation"), translation[0]),
        )
        return replace(
            case, rms_limit_T=_values(self.rms) * 1e-6 or None, peak_limit_T=_values(self.peak) * 1e-6 or None
        )


class _Signals(QObject):
    progress = Signal(str)
    finished = Signal(object, str)


class _ArrayTask(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.signals, self.operation, self.cancel = _Signals(), operation, Event()

    def report(self, text):
        from ..progress import DesignCancelled

        if self.cancel.is_set():
            raise DesignCancelled("Array study cancelled between numerical operations.")
        self.signals.progress.emit(text)

    def run(self):
        try:
            result = self.operation(self.report)
            self.report("Array study computation complete; publishing verified results.")
        except Exception as exc:
            self.signals.finished.emit(None, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(result, "")


class ArrayFieldCanvas(QWidget):
    """Physical sample scatter in an explicit world-plane projection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points, self.values, self.axes, self.quantity = None, None, (0, 1), "Residual"
        self.setMinimumHeight(180)

    def set_field(self, points, values, axes, quantity):
        self.points, self.values = np.asarray(points), np.asarray(values)
        self.axes, self.quantity = axes, quantity
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        if self.points is None:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a computed case to review its spatial field samples."
            )
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin, width, height = 50, max(self.width() - 100, 1), max(self.height() - 75, 1)
        selected = self.points[:, self.axes]
        lower, upper = np.min(selected, axis=0), np.max(selected, axis=0)
        span = upper - lower
        span = np.where(span > 0, span, 0.001)
        colorscale = max(float(np.max(np.abs(self.values))), 1e-30)
        # Preserve the limiting sample while bounding repaint work for dense datasets.
        indices = np.unique(
            np.r_[
                np.linspace(0, len(selected) - 1, min(5000, len(selected)), dtype=int), np.argmax(np.abs(self.values))
            ]
        )
        for index in indices:
            x, y = (selected[index] - lower) / span
            fraction = min(abs(float(self.values[index])) / colorscale, 1.0)
            color = QColor.fromRgbF(
                1.0 - fraction if self.values[index] >= 0 else 1.0,
                1.0 - fraction,
                1.0 if self.values[index] >= 0 else 1.0 - fraction,
            )
            painter.setPen(QPen(color if fraction > 0.1 else QColor("gray"), 1))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(margin + x * width, margin + (1.0 - y) * height), 3.0, 3.0)
        painter.setPen(self.palette().text().color())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(margin, margin, width, height)
        labels = "xyz"
        painter.drawText(margin, self.height() - 5, f"{labels[self.axes[0]]} [m]: {lower[0]:.4g} … {upper[0]:.4g}")
        painter.drawText(
            margin,
            14,
            f"World {labels[self.axes[0]]}{labels[self.axes[1]]} projection · {self.quantity} [µT] · blue +{colorscale * 1e6:.4g}, red −{colorscale * 1e6:.4g}",
        )
        painter.drawText(
            margin,
            32,
            f"{labels[self.axes[1]]} [m]: {lower[1]:.4g} … {upper[1]:.4g} · {len(self.points):,} physical samples; {len(indices):,} drawn",
        )


class ArrayStudyDialog(QDialog):
    studyChanged = Signal(dict)
    generatedProjects = Signal(object)
    busyChanged = Signal(bool)
    progressChanged = Signal(str)

    def __init__(self, assembly, parent=None, *, saved_spec=None, run_records=(), active_project=None):
        super().__init__(parent)
        self.setWindowTitle("Multichannel array and cohort studies")
        self.resize(1040, 720)
        self.assembly, self.active_project = assembly, active_project
        self.records = tuple(run_records)
        self.spec = ArrayStudySpec.from_dict(saved_spec) if saved_spec else ArrayStudySpec()
        cohort_controls = self.spec.cohort_controls or CohortControls()
        self._owned_dictionary = cohort_controls.contour_dictionary
        self.cases = list(self.spec.cases)
        self.channel_sources = list(self.spec.channels)
        self.result_value, self.results, self.worker = None, None, None
        self.cache = TemporaryDirectory(prefix="coilgen-array-archives-")
        self.layout_ = QVBoxLayout(self)
        note = QLabel(
            "Fit signed currents simultaneously using verified saved winding geometry. Each driver explicitly declares body current ratios. Array results assess the sampled scalar field and resistive power; coupled drive, feeds and manufacturing remain separate evidence."
        )
        note.setWordWrap(True)
        self.layout_.addWidget(note)
        self.tabs = QTabWidget()
        self.layout_.addWidget(self.tabs)
        inputs = QWidget()
        input_layout = QVBoxLayout(inputs)
        self.case_table = QTableWidget(0, 5)
        self.case_table.setHorizontalHeaderLabels(
            ["Case", "Membership", "Samples", "RMS limit [µT]", "Peak limit [µT]"]
        )
        input_layout.addWidget(self.case_table)
        actions = QHBoxLayout()
        for title, method in (("Import registered case…", self.import_case), ("Remove case", self.remove_case)):
            button = QPushButton(title)
            button.clicked.connect(method)
            actions.addWidget(button)
        input_layout.addLayout(actions)
        self.channels = QTableWidget(0, 6)
        self.channels.setHorizontalHeaderLabels(
            ["Channel / run", "Body current ratios", "Lower [A]", "Upper [A]", "R [Ω]", "Request identity"]
        )
        input_layout.addWidget(self.channels)
        actions = QHBoxLayout()
        for title, method in (("Add saved winding run…", self.add_channel), ("Remove channel", self.remove_channel)):
            button = QPushButton(title)
            button.clicked.connect(method)
            actions.addWidget(button)
        input_layout.addLayout(actions)
        self.power = _number(self.spec.total_power_limit_W or 0)
        self.center = QCheckBox("Fit a global frequency offset independently for each case")
        self.center.setChecked(self.spec.center_frequency)
        self.regularization = _number(self.spec.current_regularization)
        form = QFormLayout()
        form.addRow("Simultaneous resistive power limit [W] · 0 unspecified", self.power)
        form.addRow("Dimensionless current regularization", self.regularization)
        form.addRow(self.center)
        input_layout.addLayout(form)
        self.fit = QPushButton("Fit currents on saved winding geometry")
        self.fit.clicked.connect(self.run_fit)
        input_layout.addWidget(self.fit)
        self.tabs.addTab(inputs, "Cases and driver contracts")
        cohort = QWidget()
        cohort_layout = QVBoxLayout(cohort)
        cohort_note = QLabel(
            "Create a basis using training cases only, then fit held-out cases with that finished basis. POD and adaptive sectorization create portable channel projects on the active support. Contour-family assignment uses imported world/scanner contour dictionaries. Sheet and source-contour studies do not certify final routing."
        )
        cohort_note.setWordWrap(True)
        cohort_layout.addWidget(cohort_note)
        self.strategy = QComboBox()
        self.strategy.addItem("Resistance-weighted POD", "pod")
        self.strategy.addItem("Adaptive surface sectorization", "sectorization")
        self.strategy.addItem("Physical contour-family assignment", "contour_dictionary")
        self.strategy.setCurrentIndex(self.strategy.findData(cohort_controls.strategy))
        self.count = QSpinBox()
        self.count.setRange(1, 64)
        self.count.setValue(cohort_controls.channel_count)
        self.current_limit = _number(cohort_controls.current_limit_A, minimum=1e-9)
        self.cohort_resistances = QLineEdit(
            ""
            if cohort_controls.resistance_ohm is None
            else " ".join(str(value) for value in cohort_controls.resistance_ohm)
        )
        self.cohort_power = _number(cohort_controls.total_power_limit_W or 0)
        self.dictionary = QLineEdit()
        if self._owned_dictionary is not None:
            self.dictionary.setPlaceholderText("Owned contour dictionary is saved in this study")
        choose = QPushButton("Choose contour dictionary…")
        choose.clicked.connect(self.choose_dictionary)
        form = QFormLayout()
        form.addRow("Strategy", self.strategy)
        form.addRow("Channel count", self.count)
        form.addRow("Symmetric driver current limit [A]", self.current_limit)
        form.addRow("New channel resistances [Ω] · one value per channel", self.cohort_resistances)
        form.addRow("New array simultaneous power limit [W] · 0 unspecified", self.cohort_power)
        form.addRow("Registered source contour dictionary", self.dictionary)
        form.addRow(choose)
        cohort_layout.addLayout(form)
        self.synthesize = QPushButton("Synthesize from training cases and review held-out cases")
        self.synthesize.clicked.connect(self.run_synthesis)
        cohort_layout.addWidget(self.synthesize)
        cohort_layout.addStretch()
        self.tabs.addTab(cohort, "Cohort channel design")
        review = QWidget()
        review_layout = QVBoxLayout(review)
        self.scope = QLabel("No array study has been computed.")
        self.scope.setWordWrap(True)
        review_layout.addWidget(self.scope)
        self.result_table = QTableWidget(0, 9)
        self.result_table.setHorizontalHeaderLabels(
            [
                "Case",
                "Membership",
                "RMS [µT]",
                "Peak [µT]",
                "Currents [A]",
                "Power [W]",
                "Saturated",
                "Rank / condition",
                "Acceptance",
            ]
        )
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.itemSelectionChanged.connect(self._show_samples)
        review_layout.addWidget(self.result_table)
        views = QTabWidget()
        spatial = QWidget()
        spatial_layout = QVBoxLayout(spatial)
        selectors = QHBoxLayout()
        self.field_quantity, self.field_plane = QComboBox(), QComboBox()
        self.field_quantity.addItems(["Residual", "Combined", "Desired"])
        self.field_plane.addItem("World XY", (0, 1))
        self.field_plane.addItem("World XZ", (0, 2))
        self.field_plane.addItem("World YZ", (1, 2))
        self.field_quantity.currentIndexChanged.connect(self._show_field)
        self.field_plane.currentIndexChanged.connect(self._show_field)
        selectors.addWidget(self.field_quantity)
        selectors.addWidget(self.field_plane)
        spatial_layout.addLayout(selectors)
        self.field_canvas = ArrayFieldCanvas()
        spatial_layout.addWidget(self.field_canvas)
        views.addTab(spatial, "Spatial field samples")
        self.samples = QTableWidget(0, 5)
        self.samples.setHorizontalHeaderLabels(["x [m]", "y [m]", "z [m]", "Desired [µT]", "Residual [µT]"])
        self.samples.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        views.addTab(self.samples, "Largest residual samples")
        review_layout.addWidget(views)
        self.export = QPushButton("Export reproducible study and result data…")
        self.export.clicked.connect(self.export_results)
        self.export.setEnabled(False)
        review_layout.addWidget(self.export)
        self.add_generated = QPushButton("Add generated channel projects to this assembly")
        self.add_generated.clicked.connect(self._add_generated)
        self.add_generated.setEnabled(False)
        review_layout.addWidget(self.add_generated)
        self.tabs.addTab(review, "Array results")
        self.status = QLabel("Ready. No computation in progress.")
        self.status.setWordWrap(True)
        self.layout_.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.layout_.addWidget(self.progress)
        self.cancel = QPushButton("Cancel study after current operation")
        self.cancel.clicked.connect(lambda: self.worker and self.worker.cancel.set())
        self.cancel.hide()
        self.layout_.addWidget(self.cancel)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        self.layout_.addWidget(self.buttons)
        self._refresh_inputs()

    def _capture_edits(self):
        try:
            self.spec = self.value()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Correct array inputs", str(exc))
            return False
        self.cases, self.channel_sources = list(self.spec.cases), list(self.spec.channels)
        return True

    @staticmethod
    def _item(value, editable=True):
        item = QTableWidgetItem(str(value))
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def _refresh_inputs(self):
        self.case_table.setRowCount(len(self.cases))
        for row, case in enumerate(self.cases):
            for col, value in enumerate(
                (
                    case.name,
                    case.membership,
                    len(case.coordinates_m),
                    "" if case.rms_limit_T is None else case.rms_limit_T * 1e6,
                    "" if case.peak_limit_T is None else case.peak_limit_T * 1e6,
                )
            ):
                self.case_table.setItem(row, col, self._item("" if col == 1 else value, col in {3, 4}))
            membership = QComboBox()
            membership.addItem("Training", "training")
            membership.addItem("Held out", "held_out")
            membership.setCurrentIndex(membership.findData(case.membership))
            self.case_table.setCellWidget(row, 1, membership)
        self.channels.setRowCount(len(self.channel_sources))
        for row, channel in enumerate(self.channel_sources):
            for col, value in enumerate(
                (
                    channel.name,
                    " ".join(f"{body}={ratio:g}" for body, ratio in channel.body_current_ratios.items()),
                    channel.lower_A,
                    channel.upper_A,
                    channel.resistance_ohm or "",
                    channel.request_sha256[:12],
                )
            ):
                self.channels.setItem(row, col, self._item(value, col in {1, 2, 3, 4}))
        for table in (self.case_table, self.channels):
            table.resizeColumnsToContents()

    def value(self):
        cases, channels = [], []
        for row, case in enumerate(self.cases):
            limits = [
                float(self.case_table.item(row, col).text()) * 1e-6
                if self.case_table.item(row, col).text().strip()
                else None
                for col in (3, 4)
            ]
            cases.append(
                replace(
                    case,
                    membership=self.case_table.cellWidget(row, 1).currentData(),
                    rms_limit_T=limits[0],
                    peak_limit_T=limits[1],
                )
            )
        for row, channel in enumerate(self.channel_sources):
            ratios = {}
            for entry in self.channels.item(row, 1).text().replace(",", " ").split():
                body, ratio = entry.split("=", 1)
                if body in ratios:
                    raise ValueError("Duplicate body current ratio.")
                ratios[body] = float(ratio)
            resistance = self.channels.item(row, 4).text().strip()
            channels.append(
                replace(
                    channel,
                    body_current_ratios=ratios,
                    lower_A=float(self.channels.item(row, 2).text()),
                    upper_A=float(self.channels.item(row, 3).text()),
                    resistance_ohm=float(resistance) if resistance else None,
                )
            )
        return ArrayStudySpec(
            tuple(cases),
            tuple(channels),
            _values(self.power) or None,
            self.center.isChecked(),
            _values(self.regularization),
            self._cohort_controls(),
        )

    def _cohort_controls(self):
        dictionary = self._owned_dictionary
        if self.dictionary.text().strip():
            from ..cohort_contours import load_contour_dictionary

            dictionary = CohortControls.encode_dictionary(load_contour_dictionary(self.dictionary.text()))
        text = self.cohort_resistances.text().replace(",", " ").split()
        return CohortControls(
            self.strategy.currentData(),
            self.count.value(),
            _values(self.current_limit),
            tuple(float(value) for value in text) if text else None,
            _values(self.cohort_power) or None,
            dictionary,
        )

    def _apply(self):
        try:
            self.spec = self.value()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Correct array inputs", str(exc))
            return
        self.result_value = self.spec.to_dict()
        self.studyChanged.emit(self.result_value)
        self.accept()

    def import_case(self):
        if not self._capture_edits():
            return
        dialog = ArrayCaseDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            case = dialog.result_value
            if any(item.name == case.name for item in self.cases):
                QMessageBox.warning(
                    self, "Unique case name required", "Rename the imported case; that name already exists."
                )
                return
            self.cases.append(case)
            self._refresh_inputs()

    def remove_case(self):
        if not self._capture_edits():
            return
        row = self.case_table.currentRow()
        if row >= 0:
            self.cases.pop(row)
            self._refresh_inputs()

    def add_channel(self):
        if not self._capture_edits():
            return
        try:
            choices = [
                (
                    f"{record.metadata.get('name') or record.directory.name} · {record.metadata.get('request_sha256', '')[:12]}",
                    record,
                )
                for record in self.records
            ]
            if self.assembly is not None:
                for owned in self.assembly.channels:
                    for archive in owned.project.saved_runs:
                        choices.append((f"{owned.name} · {archive.alias} · owned archive", archive))
            if choices:
                labels = [label + f" [{index + 1}]" for index, (label, _) in enumerate(choices)] + [
                    "Locate another saved run directory…"
                ]
                selected, accepted = QInputDialog.getItem(
                    self,
                    "Select array winding source",
                    "Available completed run (verification follows selection)",
                    labels,
                    editable=False,
                )
                if not accepted:
                    return
                index = labels.index(selected)
            else:
                index = 0
            if index < len(choices):
                selected = choices[index][1]
                record = (
                    selected.materialize(Path(self.cache.name))
                    if hasattr(selected, "materialize")
                    else DesignService.load_run(selected.directory)
                )
            else:
                directory = QFileDialog.getExistingDirectory(
                    self, "Select completed saved winding run (result.json and manifest.json)"
                )
                if not directory:
                    return
                record = DesignService.load_run(directory)
            paths, currents, _, _ = _saved_winding(record, "original")
            scale = max(abs(value) for value in currents.values())
            ratios = {body: current / scale if scale else 1.0 for body, current in currents.items()}
            from uuid import uuid4

            channel = ArrayChannel.from_run(
                uuid4().hex, record.metadata.get("name") or record.directory.name, record, body_current_ratios=ratios
            )
            if any(
                item.request_sha256 == channel.request_sha256 and item.path_sha256 == channel.path_sha256
                for item in self.channel_sources
            ):
                raise ValueError(
                    "This geometry already drives an array channel. Duplicate sources would imply multiple currents in the same physical conductor."
                )
            self.channel_sources.append(channel)
            self._refresh_inputs()
            self.status.setText(
                "Added verified winding. Review the normalized recorded body current proportions and driver limits before fitting."
            )
        except (ValueError, TypeError, OSError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot use saved winding", str(exc))

    def remove_channel(self):
        if not self._capture_edits():
            return
        row = self.channels.currentRow()
        if row >= 0:
            self.channel_sources.pop(row)
            self._refresh_inputs()

    def choose_dictionary(self):
        path, _ = QFileDialog.getOpenFileName(self, "Registered contour dictionary", "", "Contour dictionary (*.npz)")
        if path:
            self.dictionary.setText(path)

    def _resolve(self, channel):
        for record in self.records:
            if record.metadata.get("request_sha256") == channel.request_sha256:
                return record
        if self.assembly is not None:
            for owned in self.assembly.channels:
                for archive in owned.project.saved_runs:
                    metadata = json.loads(dict(archive._files)["result.json"])
                    if metadata.get("request_sha256") == channel.request_sha256:
                        return archive.materialize(Path(self.cache.name))
        if Path(channel.run_directory).is_dir():
            return channel.run_directory
        raise ValueError(
            f"Locate saved winding run for {channel.name}; request {channel.request_sha256[:12]} is unavailable in this assembly."
        )

    def run_fit(self):
        try:
            captured = self.value()
            if not captured.channels or not captured.cases:
                raise ValueError("Import at least one case and add at least one saved winding channel.")
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Correct array inputs", str(exc))
            return
        self._start(lambda progress: evaluate_array(captured, run_resolver=self._resolve, progress=progress), captured)

    def run_synthesis(self):
        try:
            captured = self.value()
            if not captured.cases:
                raise ValueError("Import training and held-out cases first.")
            strategy, count, limit = self.strategy.currentData(), self.count.value(), _values(self.current_limit)
            if strategy == "contour_dictionary":
                dictionary = captured.cohort_controls.dictionary()

                def operation(progress):
                    return synthesize_contour_cohort(
                        captured, dictionary, channel_count=count, current_limit_A=limit, progress=progress
                    )
            else:
                if self.active_project is None:
                    raise ValueError("Select an active support project for common-topology cohort synthesis.")

                def operation(progress):
                    return synthesize_cohort(
                        self.active_project,
                        captured,
                        channel_count=count,
                        current_limit_A=limit,
                        strategy=strategy,
                        progress=progress,
                    )
        except (ValueError, TypeError, OSError) as exc:
            QMessageBox.warning(self, "Correct cohort inputs", str(exc))
            return
        self._start(operation, captured)

    def _start(self, operation, spec):
        if self.worker is not None:
            return
        self._pending_spec = spec
        self.worker = _ArrayTask(operation)
        self.worker.signals.progress.connect(self.status.setText)
        self.worker.signals.progress.connect(self.progressChanged.emit)
        self.worker.signals.finished.connect(self._finished)
        self.tabs.setEnabled(False)
        self.buttons.setEnabled(False)
        self.progress.show()
        self.cancel.show()
        self.status.setText("Computing array study…")
        self.progressChanged.emit("Computing array study…")
        self.busyChanged.emit(True)
        QThreadPool.globalInstance().start(self.worker)

    def _finished(self, result, error):
        self.worker = None
        self.tabs.setEnabled(True)
        self.buttons.setEnabled(True)
        self.progress.hide()
        self.cancel.hide()
        self.busyChanged.emit(False)
        if error:
            self.status.setText("Array study failed: " + error)
            return
        self.results = result
        self._computed_spec = self._pending_spec
        self.scope.setText(result["scope"])
        self.result_table.setRowCount(len(result["cases"]))
        for row, case in enumerate(result["cases"]):
            diagnostics = case["diagnostics"]
            condition = diagnostics["normalized_condition_number"]
            values = (
                case["name"],
                case["membership"],
                f"{case['rms_error_T'] * 1e6:.5g}",
                f"{case['peak_error_T'] * 1e6:.5g}",
                ", ".join(f"{value:.5g}" for value in case["currents_A"]),
                "Unavailable" if case["power_W"] is None else f"{case['power_W']:.5g}",
                ", ".join(str(index + 1) for index, value in enumerate(case["saturated_channels"]) if value) or "None",
                f"{diagnostics['rank']}/{diagnostics['channel_count']} / "
                + ("rank deficient" if condition is None else f"{condition:.4g}"),
                {True: "Pass", False: "Fail", None: "Unassessed"}[case.get("passed")],
            )
            for col, value in enumerate(values):
                self.result_table.setItem(row, col, self._item(value, False))
        self.result_table.resizeColumnsToContents()
        self.result_table.selectRow(0)
        self.export.setEnabled(True)
        self.tabs.setCurrentIndex(2)
        self.status.setText("Array study complete. Results refer to the captured case and driver contracts.")
        self.add_generated.setEnabled(bool(result.get("projects")))

    def _add_generated(self):
        if self.results and self.results.get("projects"):
            self.generatedProjects.emit(self.results["projects"])
            self.add_generated.setEnabled(False)
            self.status.setText(
                "Added generated channel projects. Compute and review their final winding geometry before treating the basis as a physical array."
            )

    def _show_samples(self):
        if self.results is None:
            return
        row = self.result_table.currentRow()
        if row < 0:
            return
        result = self.results["cases"][row]
        source = next(case for case in self._computed_spec.cases if case.name == result["name"])
        errors = np.asarray(result["residual_T"])
        # Largest errors first, including the limiting sample; never truncate to the first spatial rows.
        indices = np.argsort(np.abs(errors))[::-1][:1000]
        self.samples.setRowCount(len(indices))
        for row, index in enumerate(indices):
            for col, value in enumerate(
                (*source.coordinates_m[index], source.desired_field_T[index] * 1e6, errors[index] * 1e6)
            ):
                self.samples.setItem(row, col, self._item(f"{value:.7g}", False))
        self.samples.resizeColumnsToContents()
        self._show_field()

    def _show_field(self):
        if self.results is None:
            return
        row = self.result_table.currentRow()
        if row < 0:
            return
        result = self.results["cases"][row]
        source = next(case for case in self._computed_spec.cases if case.name == result["name"])
        quantity = self.field_quantity.currentText()
        if quantity == "Desired":
            field = source.desired_field_T
        elif quantity == "Combined":
            field = np.asarray(result["residual_T"]) + source.desired_field_T
        else:
            field = result["residual_T"]
        self.field_canvas.set_field(source.coordinates_m, field, self.field_plane.currentData(), quantity)

    def export_results(self):
        if self.results is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export array study", "array_study.json", "Portable array result (*.json)"
        )
        if not filename:
            return
        try:
            payload = dict(self.results)
            if payload.get("projects"):
                payload["projects"] = [project.to_dict() for project in payload["projects"]]
            payload["spec"] = self._computed_spec.to_dict()
            atomic_json(Path(filename), json_value(payload, nonfinite="raise"))
            self.status.setText(
                "Saved array study with source bindings, captured inputs, currents, errors and per-case diagnostics: "
                + filename
            )
        except (ValueError, TypeError, OSError) as exc:
            self.status.setText("Array study export failed: " + str(exc))

    def reject(self):
        if self.worker is not None:
            self.worker.cancel.set()
            self.status.setText("Cancellation requested. Waiting for the current numerical operation to finish.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
