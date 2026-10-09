"""Structured authorship of named target contracts and finite rigid scenarios."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from .gradient_tools import EngineeringDialog, _number, _values
from ..application.gradient_studies import RigidTransform
from ..application.research_requirements import (
    DesiredDataset,
    PhysicalScenario,
    RegionalRequirement,
    ResearchRequirements,
)
from ..target_regions import TargetRegion


def _triple(text, label):
    try:
        values = np.asarray(text.replace(",", " ").split(), dtype=float)
    except ValueError as exc:
        raise ValueError(f"{label}: enter three real numbers.") from exc
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: enter three finite Cartesian values.")
    return tuple(values)


class TransformEditor(QWidget):
    """Translation and axis-angle authoring; numerical storage is a proper rotation."""

    changed = Signal()

    def __init__(self, value=RigidTransform(), parent=None):
        super().__init__(parent)
        from scipy.spatial.transform import Rotation

        form = QFormLayout(self)
        self.translation = QLineEdit(" ".join(f"{value:.12g}" for value in value.translation_m))
        rotation = Rotation.from_matrix(value.rotation).as_rotvec()
        angle = np.linalg.norm(rotation)
        axis = rotation / angle if angle else np.array((0, 0, 1))
        self.axis = QLineEdit(" ".join(f"{value:.12g}" for value in axis))
        self.angle = _number(np.rad2deg(angle), minimum=-360, maximum=360)
        form.addRow("Translation x y z [m]", self.translation)
        form.addRow("Rotation axis x y z", self.axis)
        form.addRow("Rotation angle [deg]", self.angle)
        self.original = value
        self.dirty = False
        self.translation.textChanged.connect(self._changed)
        self.axis.textChanged.connect(self._changed)
        self.angle.valueChanged.connect(self._changed)

    def _changed(self):
        self.dirty = True
        self.changed.emit()

    def value(self):
        from scipy.spatial.transform import Rotation

        if not self.dirty:
            return self.original
        translation = _triple(self.translation.text(), "Translation")
        axis = np.asarray(_triple(self.axis.text(), "Rotation axis"))
        angle = np.deg2rad(_values(self.angle))
        norm = np.linalg.norm(axis)
        if angle and norm == 0:
            raise ValueError("Nonzero rotation requires a nonzero rotation axis.")
        matrix = Rotation.from_rotvec(axis / norm * angle).as_matrix() if angle else np.eye(3)
        return RigidTransform(tuple(map(tuple, matrix)), translation)


class RegionalRequirementDialog(EngineeringDialog):
    def __init__(self, value=None, parent=None):
        super().__init__("Named regional field requirement", parent)
        self.resize(720, 820)
        self.name = QLineEdit(value.name if value else "ROI")
        self.component = QComboBox()
        self.component.addItems(("Bx", "By", "Bz", "BdotB0", "vector"))
        self.component.setCurrentText(value.component if value else "Bz")
        self.protected = QCheckBox("Protected region: desired field is zero")
        self.protected.setChecked(bool(value and value.protected))
        self.rms_enabled = QCheckBox("Enforce RMS residual bound")
        self.rms_enabled.setChecked(bool(value and value.rms_limit_T is not None))
        self.rms = _number(value.rms_limit_T if value and value.rms_limit_T is not None else 1e-5)
        self.peak_enabled = QCheckBox("Enforce peak residual bound")
        self.peak_enabled.setChecked(value is None or value.peak_limit_T is not None)
        self.peak = _number(value.peak_limit_T if value and value.peak_limit_T is not None else 1e-4)
        self.weight = _number(value.weight if value else 1)
        for label, control in (
            ("Region name", self.name),
            ("World field component", self.component),
            ("Intent", self.protected),
            ("RMS acceptance", self.rms_enabled),
            ("RMS limit [T]", self.rms),
            ("Peak acceptance", self.peak_enabled),
            ("Peak limit [T]", self.peak),
            ("Fit-grid weight", self.weight),
        ):
            self.form.addRow(label, control)
        self.note(
            "Weights apply to the target grid and never relax physical bounds. Regional physical optimization minimizes sheet power unless an existing research objective is selected. Components and all maps use the world frame."
        )
        self.source = QComboBox()
        self.source.addItems(("Analytic sampled region", "Imported desired map"))
        recipe = value.design.recipe if value else None
        self.source.setCurrentIndex(0 if recipe or value is None else 1)
        self.form.addRow("Design samples", self.source)
        self.shape = QComboBox()
        self.shape.addItems(("sphere", "box", "cylinder", "ellipsoid"))
        self.center = QLineEdit("0 0 0")
        self.dimensions = QLineEdit("0.03")
        self.region_axis = QLineEdit("0 0 1")
        self.region_angle = _number(0, minimum=-360, maximum=360)
        self.sampling = QComboBox()
        self.sampling.addItems(("volume", "surface"))
        self.resolution = QSpinBox()
        self.resolution.setRange(2, 40)
        self.resolution.setValue(8)
        self.expressions = [QLineEdit(text) for text in ("0", "0", "0.04*x")]
        self.analytic_controls = QWidget()
        analytic = QFormLayout(self.analytic_controls)
        analytic.addRow("Shape", self.shape)
        analytic.addRow("Centre x y z [m]", self.center)
        analytic.addRow("Region rotation axis", self.region_axis)
        analytic.addRow("Region rotation [deg]", self.region_angle)
        analytic.addRow("Dimensions [m]", self.dimensions)
        analytic.addRow("Sampling domain", self.sampling)
        analytic.addRow("Sampling resolution", self.resolution)
        for label, control in zip(("Bx(x,y,z) [T]", "By(x,y,z) [T]", "Bz(x,y,z) [T]"), self.expressions):
            analytic.addRow(label, control)
        if recipe:
            geometry = TargetRegion.from_dict(recipe["region"])
            self.shape.setCurrentText(geometry.shape)
            self.center.setText(" ".join(str(value) for value in geometry.center_m))
            self.region_axis.setText(" ".join(str(value) for value in geometry.rotation_axis))
            self.region_angle.setValue(np.rad2deg(geometry.rotation_angle_rad))
            self.sampling.setCurrentText(geometry.sampling)
            dimensions = {
                "sphere": (geometry.radius_m,),
                "box": geometry.size_m,
                "cylinder": (geometry.radius_m, geometry.length_m),
                "ellipsoid": geometry.semi_axes_m,
            }[geometry.shape]
            self.dimensions.setText(" ".join(str(value) for value in dimensions))
            self.resolution.setValue(recipe["resolution"])
            for control, text in zip(self.expressions, recipe["expressions"]):
                control.setText(text)
        dims_note = QLabel(
            "Sphere: radius. Box: full x y z lengths. Cylinder: radius, length (axis z). Ellipsoid: three semi-axes. Expressions use world x,y,z in metres."
        )
        dims_note.setWordWrap(True)
        analytic.addRow(dims_note)
        self.form.addRow(self.analytic_controls)
        self.design = value.design if value else None
        self.validation = value.validation if value else None
        self.import_design = QPushButton("Import desired map…")
        self.import_design.clicked.connect(lambda: self._import(False))
        self.form.addRow(self.import_design)
        self.registration = TransformEditor(self.design.registration if self.design else RigidTransform())
        self.form.addRow("Design map registration", self.registration)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.form.addRow(self.preview)
        self.validation_mode = QComboBox()
        self.validation_mode.addItems(
            ("No independent desired dataset", "Import independent desired dataset", "Independent analytic sampling")
        )
        self.validation_mode.setCurrentIndex(1 if self.validation else 0)
        self.form.addRow("Validation samples", self.validation_mode)
        self.import_validation = QPushButton("Import independent desired map…")
        self.import_validation.clicked.connect(lambda: self._import(True))
        self.form.addRow(self.import_validation)
        self.validation_resolution = QSpinBox()
        self.validation_resolution.setRange(2, 40)
        self.validation_resolution.setValue(11)
        if recipe and self.validation and self.validation.recipe:
            self.validation_mode.setCurrentIndex(2)
            self.validation_resolution.setValue(self.validation.recipe["resolution"])
        self.form.addRow("Independent analytic resolution", self.validation_resolution)
        self.validation_registration = TransformEditor(
            self.validation.registration if self.validation else RigidTransform()
        )
        self.form.addRow("Validation map registration", self.validation_registration)
        self.validation_preview = QLabel()
        self.validation_preview.setWordWrap(True)
        self.form.addRow(self.validation_preview)
        self.note(
            "Native desired CSV/TSV/JSON/NPZ coordinates are metres and fields tesla; portable measured-map records carry explicit source units. Registration rotates vectors and points. Validation samples may not overlap any fitted region and are never used for optimization or current fitting."
        )
        self.source.currentIndexChanged.connect(self._update)
        self.validation_mode.currentIndexChanged.connect(self._update)
        self.registration.changed.connect(self._update)
        self.validation_registration.changed.connect(self._update)
        self.original = value
        self.analytic_dirty = False
        self.validation_dirty = False
        self.numeric_dirty = set()
        for key, control in (("rms_limit_T", self.rms), ("peak_limit_T", self.peak), ("weight", self.weight)):
            control.valueChanged.connect(lambda changed, key=key: self.numeric_dirty.add(key))
            control.lineEdit().textEdited.connect(lambda changed, key=key: self.numeric_dirty.add(key))
        for control in (self.center, self.dimensions, self.region_axis, *self.expressions):
            control.textChanged.connect(self._analytic_changed)
        for control in (self.shape, self.sampling):
            control.currentIndexChanged.connect(self._analytic_changed)
        for control in (self.region_angle, self.resolution):
            control.valueChanged.connect(self._analytic_changed)
        self.protected.toggled.connect(self._analytic_changed)
        self.validation_resolution.valueChanged.connect(self._validation_changed)
        self._update()

    def _analytic_changed(self):
        self.analytic_dirty = True

    def _validation_changed(self):
        self.validation_dirty = True

    def _update(self):
        analytic = self.source.currentIndex() == 0
        self.analytic_controls.setVisible(analytic)
        self.import_design.setVisible(not analytic)
        self.registration.setVisible(not analytic)
        self.preview.setVisible(not analytic)
        self.import_validation.setVisible(self.validation_mode.currentIndex() == 1)
        self.validation_resolution.setVisible(self.validation_mode.currentIndex() == 2)
        self.validation_registration.setVisible(self.validation_mode.currentIndex() == 1)
        self.validation_preview.setVisible(self.validation_mode.currentIndex() == 1)
        for label, dataset in ((self.preview, self.design), (self.validation_preview, self.validation)):
            if dataset:
                try:
                    transform = (
                        self.registration.value() if label is self.preview else self.validation_registration.value()
                    )
                    points, field = dataset.data.world_values(transform)
                except (TypeError, ValueError) as exc:
                    label.setText(str(exc))
                    continue
                label.setText(
                    f"{len(points):,} samples · {dataset.data.length_unit}, {dataset.data.field_unit} → world m,T\n"
                    f"Source: {dataset.data.source}\nWorld bounds: {points.min(axis=0)} to {points.max(axis=0)}\n"
                    f"Field range [T]: {field.min(axis=0)} to {field.max(axis=0)}"
                )
            else:
                label.setText("No desired map imported.")

    def _import(self, validation):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import desired target samples", "", "Field maps (*.csv *.tsv *.json *.npz)"
        )
        if not path:
            return
        try:
            dataset = DesiredDataset.from_file(path, independent=validation)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot import desired map", str(exc))
            return
        if validation:
            self.validation = dataset
        else:
            self.design = dataset
        self._update()

    def _region(self):
        shape = self.shape.currentText()
        text = self.dimensions.text().replace(",", " ").split()
        try:
            dimensions = tuple(float(value) for value in text)
        except ValueError as exc:
            raise ValueError("Dimensions require SI numeric values.") from exc
        keys = {
            "sphere": ("radius_m",),
            "box": ("size_m",),
            "cylinder": ("radius_m", "length_m"),
            "ellipsoid": ("semi_axes_m",),
        }[shape]
        if shape in {"box", "ellipsoid"}:
            values = {keys[0]: dimensions}
        else:
            if len(dimensions) != len(keys):
                raise ValueError(f"{shape}: expected {len(keys)} dimensions in metres.")
            values = dict(zip(keys, dimensions))
        return TargetRegion(
            shape,
            sampling=self.sampling.currentText(),
            center_m=_triple(self.center.text(), "Region centre"),
            rotation_axis=_triple(self.region_axis.text(), "Region rotation axis"),
            rotation_angle_rad=np.deg2rad(_values(self.region_angle)),
            **values,
        )

    def value(self):
        expressions = tuple("0" if self.protected.isChecked() else control.text() for control in self.expressions)
        if self.source.currentIndex() == 0:
            if self.original and self.original.design.recipe and not self.analytic_dirty:
                design = self.original.design
            else:
                design = DesiredDataset.analytic(self._region(), expressions, resolution=self.resolution.value())
        elif self.design is not None:
            design = replace(self.design, registration=self.registration.value())
        else:
            raise ValueError("Import design samples first.")
        validation = None
        if self.validation_mode.currentIndex() == 1:
            if self.validation is None:
                raise ValueError("Import independent desired validation samples first.")
            validation = replace(self.validation, registration=self.validation_registration.value())
        elif self.validation_mode.currentIndex() == 2:
            if self.source.currentIndex() != 0:
                raise ValueError("Independent analytic sampling requires an analytic design region.")
            if (
                self.original
                and self.original.validation
                and self.original.validation.recipe
                and not self.analytic_dirty
                and not self.validation_dirty
            ):
                validation = self.original.validation
            else:
                validation = DesiredDataset.analytic(
                    self._region(), expressions, resolution=self.validation_resolution.value(), independent=True
                )

        def number(control, name):
            value = _values(control)
            original = getattr(self.original, name, None)
            return original if original is not None and name not in self.numeric_dirty else value

        return RegionalRequirement(
            self.name.text(),
            design,
            self.component.currentText(),
            number(self.rms, "rms_limit_T") if self.rms_enabled.isChecked() else None,
            number(self.peak, "peak_limit_T") if self.peak_enabled.isChecked() else None,
            number(self.weight, "weight"),
            self.protected.isChecked(),
            validation,
        )


class PhysicalScenarioDialog(EngineeringDialog):
    def __init__(self, value=None, parent=None):
        super().__init__("Finite physical pose scenario", parent)
        self.name = QLineEdit(value.name if value else "Pose 1")
        self.membership = QComboBox()
        self.membership.addItem("Design: enforce during optimization", "design")
        self.membership.addItem("Held out: evaluate after optimization", "validation")
        self.membership.setCurrentIndex(1 if value and value.membership == "validation" else 0)
        self.form.addRow("Scenario name", self.name)
        self.form.addRow("Membership", self.membership)
        self.support = TransformEditor(value.support_transform if value else RigidTransform())
        self.samples = TransformEditor(value.sample_transform if value else RigidTransform())
        self.support_scale = QLineEdit(
            " ".join(str(number) for number in (value.support_scale_xyz if value else (1.0, 1.0, 1.0)))
        )
        self.scale_center = QLineEdit(
            " ".join(str(number) for number in (value.support_scale_center_m if value else (0.0, 0.0, 0.0)))
        )
        self.form.addRow("Support dimensional factors x y z", self.support_scale)
        self.form.addRow("Scale centre x y z [world m]", self.scale_center)
        self.form.addRow("Move all supports", self.support)
        self.form.addRow("Move samples and desired vectors", self.samples)
        self.note(
            "Positive dimensional factors act about the authored scale centre before rotation about the world origin and translation. Factors such as 1.001,1.001,1 model a finite radial dimension error. All support nodes and saved winding segments retain their correspondence. Sample motion rotates points and desired vectors. Each design case shares one sheet solution and original currents. Coverage is limited to these listed geometric perturbations; conductivity, conductor profile variation, power under perturbed resistance, arbitrary bowing and an uncertainty continuum remain unassessed."
        )

    def value(self):
        return PhysicalScenario(
            self.name.text(),
            self.membership.currentData(),
            self.support.value(),
            self.samples.value(),
            _triple(self.support_scale.text(), "Dimensional factors"),
            _triple(self.scale_center.text(), "Scale centre"),
        )


class ResearchRequirementsDialog(QDialog):
    def __init__(self, spec=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Regional requirements and finite physical scenarios")
        self.resize(770, 570)
        spec = ResearchRequirements.from_dict(spec) if spec is not None else None
        self.regions = list(spec.regions) if spec else []
        self.scenarios = list(spec.scenarios) if spec else []
        self.result_value = None
        layout = QVBoxLayout(self)
        note = QLabel(
            "Define each ROI or protected region with its own physical bound. Independent desired datasets and held-out poses are evaluated at saved currents. Nominal is always included in optimization."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        self.lists = {}
        for label, key in (("Named regions", "regions"), ("Finite physical poses", "scenarios")):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            items = QListWidget()
            self.lists[key] = items
            panel_layout.addWidget(items)
            row = QHBoxLayout()
            for text, operation in (("Add…", "add"), ("Edit…", "edit"), ("Remove", "remove")):
                button = QPushButton(text)
                button.clicked.connect(lambda checked=False, key=key, operation=operation: self._change(key, operation))
                row.addWidget(button)
            panel_layout.addLayout(row)
            items.itemDoubleClicked.connect(lambda item, key=key: self._change(key, "edit"))
            tabs.addTab(panel, label)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    def _refresh(self):
        self.lists["regions"].clear()
        for region in self.regions:
            limits = ", ".join(
                f"{label} ≤ {value:g} T"
                for label, value in (("RMS", region.rms_limit_T), ("peak", region.peak_limit_T))
                if value is not None
            )
            self.lists["regions"].addItem(
                f"{region.name} · {region.component} · {limits}\n"
                f"{len(region.design.data.coordinates):,} design samples · "
                + (
                    f"{len(region.validation.data.coordinates):,} independent samples"
                    if region.validation
                    else "independent sampling unavailable"
                )
            )
        self.lists["scenarios"].clear()
        for scenario in self.scenarios:
            self.lists["scenarios"].addItem(
                f"{scenario.name} · {scenario.membership}\n"
                f"Support Δ [m]: {scenario.support_transform.translation_m}; samples Δ [m]: {scenario.sample_transform.translation_m}"
                f"; dimensions: {scenario.support_scale_xyz}"
            )
        self.status.setText(
            f"{len(self.regions)} regions · nominal + {len(self.scenarios)} listed poses. "
            "Only design poses constrain optimization; held-out poses remain separate evidence."
        )

    def _change(self, key, operation):
        values = getattr(self, key)
        index = self.lists[key].currentRow()
        if operation != "add" and index < 0:
            return
        if operation == "remove":
            values.pop(index)
        else:
            editor = RegionalRequirementDialog if key == "regions" else PhysicalScenarioDialog
            dialog = editor(values[index] if operation == "edit" else None, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            if operation == "add":
                values.append(dialog.result_value)
            else:
                values[index] = dialog.result_value
        self._refresh()

    def value(self):
        return ResearchRequirements(tuple(self.regions), tuple(self.scenarios))

    def _accept(self):
        try:
            self.result_value = self.value()
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, "Correct research requirements", str(exc))
            return
        self.accept()


class ResearchRequirementsReviewDialog(QDialog):
    """Scoped regional evidence and its saved spatial failure witnesses."""

    def __init__(self, report, arrays=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Regional and finite-pose evidence")
        self.resize(1080, 800)
        self.report, self.arrays = report, arrays or {}
        self.rows = list(report.get("rows", ()))
        layout = QVBoxLayout(self)
        note = QLabel(
            "Evidence covers the listed poses and dimensional perturbations at unchanged saved currents. Design samples, independent samples and held-out cases have separate rows. A centroid result excludes external feeds unless the scope says complete electrical paths."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(len(self.rows), 10)
        self.table.setHorizontalHeaderLabels(
            (
                "Region",
                "Pose",
                "Membership",
                "Samples",
                "Physical scope",
                "Status",
                "RMS [T]",
                "RMS bound [T]",
                "Peak [T]",
                "Peak bound [T]",
            )
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for index, row in enumerate(self.rows):
            state = (
                "Unavailable"
                if not row.get("available")
                else "Pass"
                if row.get("passed") is True
                else "Fail"
                if row.get("passed") is False
                else "Unassessed"
            )
            values = (
                row.get("region"),
                row.get("scenario"),
                row.get("scenario_membership"),
                row.get("sampling"),
                row.get("scope"),
                state,
                row.get("rms_error_T"),
                row.get("rms_limit_T"),
                row.get("peak_error_T"),
                row.get("peak_limit_T"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(
                    f"{value:.5g}" if isinstance(value, float) else str(value) if value is not None else "—"
                )
                item.setToolTip(row.get("reason", ""))
                self.table.setItem(index, column, item)
        self.table.setMaximumHeight(280)
        layout.addWidget(self.table)
        self.witness = QLabel("Select a row to inspect its source and failure witness.")
        self.witness.setWordWrap(True)
        layout.addWidget(self.witness)
        self.figure = self.canvas = None
        if self.arrays:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            self.figure = Figure(figsize=(8, 4), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.canvas)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.itemSelectionChanged.connect(self._selected)
        if self.rows:
            self.table.selectRow(0)

    def _selected(self):
        index = self.table.currentRow()
        if index < 0:
            return
        row = self.rows[index]
        if not row.get("available"):
            self.witness.setText(row.get("reason", "Evidence unavailable."))
        else:
            point = row.get("peak_point_m", ())
            text = " ".join(f"{value:.8g}" for value in point)
            failed = row.get("failed_sample_indices", ())
            self.witness.setText(
                f"Peak witness x y z [m]: {text} · sample {row.get('peak_sample_index')} · "
                f"{len(failed)} samples exceed the peak bound.\n"
                f"Dataset SHA256: {row.get('dataset_sha256')} · "
                f"Current policy: {row.get('current_policy')}"
            )
            details = self.witness.text()
            effective = ", ".join(
                f"{name} {row[key]:.6g} T"
                for name, key in (("RMS", "effective_rms_limit_T"), ("peak", "effective_peak_limit_T"))
                if row.get(key) is not None
            )
            if effective:
                details += (
                    f"\nEffective bounds: {effective} · validation design margin {row.get('design_margin', 1):g}."
                )
            if row.get("reason"):
                details += "\n" + row["reason"]
            if row.get("protected"):
                details += (
                    f"\nSuppression {row.get('component')}: RMS {row.get('suppression_rms_T', 0):.5g} T, "
                    f"peak {row.get('suppression_peak_T', 0):.5g} T."
                )
            if row.get("regional_gradient_available"):
                details += (
                    f"\nDescriptive local gradient tensor [T/m]: "
                    f"{np.array2string(np.asarray(row['field_gradient_T_per_m']), precision=4)} · "
                    f"fit condition {row.get('regional_gradient_condition_number', 0):.5g}. No gradient acceptance claim."
                )
            elif row.get("regional_gradient_reason"):
                details += "\n" + row["regional_gradient_reason"]
            self.witness.setText(details)
        if self.figure is None:
            return
        self.figure.clear()
        prefix = row.get("array_prefix")
        keys = tuple(f"{prefix}_{name}" for name in ("points_m", "field_T", "target_T"))
        if not prefix or not all(key in self.arrays for key in keys):
            self.canvas.draw_idle()
            return
        from ..field_components import field_component_values

        points, field, target = (np.asarray(self.arrays[key]) for key in keys)
        component = row.get("component", "Bz")
        residual = field_component_values(
            field - target, component, b0_direction=self.report.get("b0_direction", (0, 0, 1))
        )
        error = np.abs(residual) if component != "vector" else np.linalg.norm(residual.reshape(-1, 3), axis=1)
        axes = self.figure.add_subplot(projection="3d")
        artist = axes.scatter(*points.T, c=error, cmap="magma", s=12)
        failed = np.asarray(row.get("failed_sample_indices", ()), dtype=int)
        if len(failed):
            axes.scatter(*points[failed].T, facecolors="none", edgecolors="cyan", s=35, label="Peak-bound failure")
            axes.legend()
        self.figure.colorbar(artist, ax=axes, pad=0.1, label="Residual magnitude [T]")
        axes.set(
            xlabel="x [m]",
            ylabel="y [m]",
            zlabel="z [m]",
            title=f"{row['region']} · {row['scenario']} · {row['scope']}",
        )
        self.canvas.draw_idle()
