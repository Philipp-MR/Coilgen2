"""Focused engineering editors over native settings and saved winding evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .controls import ScientificSpinBox
from ..persistence import json_value


def _number(value, *, minimum=0, maximum=1e12):
    control = ScientificSpinBox()
    control.setDecimals(17)
    control.setRange(minimum, maximum)
    control.setValue(float(value))
    return control


def _values(control):
    if not control.hasAcceptableInput():
        raise ValueError("Complete all numeric entries before applying.")
    control.interpretText()
    return control.value()


def _matrix(text, columns, label, *, empty=False):
    rows = [row.replace(",", " ").split() for row in text.splitlines() if row.strip()]
    if not rows and empty:
        return []
    try:
        values = np.asarray(rows, dtype=float)
    except ValueError as exc:
        raise ValueError(f"{label}: use numeric rows separated by spaces or commas.") from exc
    if values.ndim != 2 or values.shape[1] != columns or not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: expected finite rows with {columns} columns.")
    return values.tolist()


class EngineeringDialog(QDialog):
    """Validate the entire transaction before returning an accepted editor."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(590)
        self.layout_ = QVBoxLayout(self)
        self.content = QWidget()
        self.form = QFormLayout(self.content)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.content)
        self.layout_.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.layout_.addWidget(buttons)
        self.result_value = None

    def note(self, text):
        label = QLabel(text)
        label.setWordWrap(True)
        self.form.addRow(label)

    def _accept(self):
        try:
            self.result_value = self.value()
        except (ValueError, TypeError, OSError, KeyError) as exc:
            QMessageBox.warning(self, "Correct engineering inputs", str(exc))
            return
        self.accept()


class GradientRequirementsDialog(EngineeringDialog):
    def __init__(self, project, parent=None):
        super().__init__("Gradient requirements", parent)
        self.project = project
        self.controls = {}
        self.dirty = set()
        self.note(
            "Requested gradient and ROI are defined in Target field and regions. These tolerances use the native Bz gradient fit in world coordinates. A zero tolerance disables that check; it does not establish acceptance."
        )
        config = project.config
        for key, label in (
            ("gradient_strength_tolerance_percent", "Gradient strength tolerance [%]"),
            ("linearity_rms_tolerance_percent", "RMS linearity tolerance [%]"),
            ("linearity_peak_tolerance_percent", "Peak linearity tolerance [%]"),
            ("linearity_cross_gradient_tolerance_percent", "Cross-gradient tolerance [%]"),
            ("linearity_reference_radius_m", "Reference radius [m] · 0 uses ROI"),
        ):
            control = _number(getattr(config, key))
            control.textChanged.connect(lambda _text, field=key: self.dirty.add(field))
            self.controls[key] = control
            self.form.addRow(label, control)
        for key, label, choices in (
            (
                "linearity_region",
                "Fit region",
                [("Primary ROI", "dsv"), ("All samples", "all"), ("Suppression region", "suppression")],
            ),
            (
                "linearity_normalization",
                "Residual normalization",
                [
                    ("Requested gradient × radius", "requested_gradient_times_radius"),
                    ("Peak requested target", "target_peak"),
                ],
            ),
        ):
            control = QComboBox()
            for text, value in choices:
                control.addItem(text, value)
            index = control.findData(getattr(config, key))
            if index < 0:
                control.addItem(str(getattr(config, key)), getattr(config, key))
                index = control.count() - 1
            control.setCurrentIndex(index)
            control.currentIndexChanged.connect(lambda _index, field=key: self.dirty.add(field))
            self.controls[key] = control
            self.form.addRow(label, control)
        self.fit_offset = QCheckBox("Fit an offset field when reporting linearity")
        self.fit_offset.setChecked(config.linearity_fit_offset)
        self.fit_offset.toggled.connect(lambda _: self.dirty.add("linearity_fit_offset"))
        self.form.addRow(self.fit_offset)
        self.independent = QCheckBox("Review fabricated winding on independent samples at original current")
        existing = project.study.get("engineering")
        self.independent.setChecked(bool(existing and existing.get("independent_winding_validation", True)))
        self._independent_initial = self.independent.isChecked()
        self.independent.toggled.connect(lambda _: self.dirty.add("independent"))
        self.form.addRow(self.independent)
        self.note(
            "Independent winding review uses the saved electrical paths and thin-wire segment model. Enabling it requests Full design evaluation and Fabrication review; imported/custom target contracts can remain unverified."
        )
        self.resize(680, 660)

    def value(self):
        changes = {}
        for key, control in self.controls.items():
            value = control.currentData() if isinstance(control, QComboBox) else _values(control)
            if key in self.dirty:
                changes[key] = value
        if "linearity_fit_offset" in self.dirty:
            changes["linearity_fit_offset"] = self.fit_offset.isChecked()
        study = {}
        if "independent" in self.dirty and self.independent.isChecked() != self._independent_initial:
            from ..gradient_engineering import GradientEngineeringSpec

            engineering = json_value(self.project.study.get("engineering") or {})
            engineering["independent_winding_validation"] = self.independent.isChecked()
            study = {"engineering": GradientEngineeringSpec.from_dict(engineering).to_dict()}
            if self.independent.isChecked():
                study.update(pipeline_stage="evaluated", fabricate=True)
        return changes, study


class GradientGeometryDialog(EngineeringDialog):
    def __init__(self, project, parent=None):
        from ..application.gradient_geometry import CylindricalGeometry, CylinderSurface

        super().__init__("Primary, shield and external region", parent)
        self.original = CylindricalGeometry.from_project(project)
        self.note(
            "Apply creates owned cylindrical meshes in the shared world frame. Arbitrary imported surfaces are replaced only by applying this explicit new geometry. Target, material and solver requirements are retained."
        )
        self.surfaces = {}
        self.enabled = {}
        for role, fallback in (
            ("primary", CylinderSurface(0.25, 0.5)),
            ("shield", CylinderSurface(0.30, 0.55)),
            ("suppression", CylinderSurface(0.4, 0.7)),
        ):
            surface = getattr(self.original, role, None) if self.original else None
            surface = surface or fallback
            enabled = QCheckBox(role.title())
            enabled.setChecked(role == "primary" or bool(self.original and getattr(self.original, role)))
            enabled.setEnabled(role != "primary")
            self.enabled[role] = enabled
            self.form.addRow(enabled)
            radius, length = _number(surface.radius_m, minimum=1e-6), _number(surface.length_m, minimum=1e-6)
            angular, axial = QSpinBox(), QSpinBox()
            angular.setRange(4, 512)
            axial.setRange(1, 256)
            angular.setValue(surface.angular_divisions)
            axial.setValue(surface.axial_divisions)
            self.surfaces[role] = (radius, length, angular, axial)
            for label, control in zip(
                ("Radius [m]", "Length [m]", "Angular divisions", "Axial divisions"), self.surfaces[role], strict=True
            ):
                self.form.addRow(f"{role.title()} {label}", control)
                control.setEnabled(enabled.isChecked())
                enabled.toggled.connect(control.setEnabled)
        primary = self.original.primary if self.original else CylinderSurface(0.25, 0.5)
        self.center = QLineEdit(" ".join(f"{v:.17g}" for v in primary.center_m))
        self.axis = QLineEdit(" ".join(f"{v:.17g}" for v in primary.axis))
        self.form.addRow("Common center x y z [m]", self.center)
        self.form.addRow("Common axis x y z", self.axis)
        self.note(
            "The external cylinder is a closed suppression region. Set its RMS field bound in the optimization requirements; creating geometry does not invent an acceptance limit."
        )
        self.resize(700, 760)

    def value(self):
        from ..application.gradient_geometry import CylindricalGeometry, CylinderSurface

        centers = _matrix(self.center.text(), 3, "Center")
        axes = _matrix(self.axis.text(), 3, "Axis")
        if len(centers) != 1 or len(axes) != 1:
            raise ValueError("Enter exactly one common center and axis vector.")
        center, axis = tuple(centers[0]), tuple(axes[0])
        surfaces = {}
        for role, controls in self.surfaces.items():
            radius, length, angular, axial = controls
            original = getattr(self.original, role, None) if self.original else None
            surfaces[role] = (
                CylinderSurface(
                    _values(radius),
                    _values(length),
                    angular.value(),
                    axial.value(),
                    center,
                    axis,
                    azimuth_rad=original.azimuth_rad if original else 0.0,
                )
                if self.enabled[role].isChecked()
                else None
            )
        return CylindricalGeometry(**surfaces)


class ConvergenceDialog(EngineeringDialog):
    def __init__(self, project, parent=None):
        super().__init__("Convergence study", parent)
        self.project = project
        self.note(
            "Queue one-factor series against this immutable design. Mesh refinement, source quadrature and independent target sampling are separate axes. Candidate requests and owned inputs are saved with every run."
        )
        self.entries = {}
        for key, label, value in (
            ("mesh_refinements", "Native mesh refinement counts (0–3)", "0 1 2"),
            ("quadrature_orders", "Quadrature orders (1–16)", "2 4 6"),
            ("independent_resolutions", "Independent sample resolutions (2–128)", "12 18 24"),
        ):
            entry = QLineEdit(value)
            self.entries[key] = entry
            self.form.addRow(label, entry)
        self.relative = _number(0.01, maximum=1)
        self.absolute = _number(1e-7)
        self.form.addRow("Relative convergence tolerance [fraction]", self.relative)
        self.form.addRow("Absolute field tolerance [T]", self.absolute)
        self.note(
            "Convergence is assessed only from compatible verified evidence. Sample-count agreement alone cannot qualify quadrature, electrical paths or fabrication."
        )
        self.resize(680, 440)

    def value(self):
        from ..application.gradient_studies import _levels

        values = {}
        ranges = {"mesh_refinements": (0, 3), "quadrature_orders": (1, 16), "independent_resolutions": (2, 128)}
        for key, entry in self.entries.items():
            try:
                values[key] = tuple(int(v) for v in entry.text().replace(",", " ").split())
            except ValueError as exc:
                raise ValueError(f"{key}: supply whole numbers.") from exc
            values[key] = _levels(values[key], key, *ranges[key])
        existing = {
            "mesh_refinements": self.project.values["iteration_num_mesh_refinement"],
            "quadrature_orders": self.project.values["gauss_order"],
            "independent_resolutions": self.project.study["validation_resolution"],
        }
        candidate_count = sum(value != existing[key] for key, levels in values.items() for value in levels)
        if not 1 <= candidate_count <= 11:
            raise ValueError(
                "Choose 1–11 different numerical levels plus the baseline; candidate generation runs in the worker."
            )
        return values, {"relative_tolerance": _values(self.relative), "absolute_tolerance_T": _values(self.absolute)}


def _polyline_editor(form, label, points=()):
    editor = QPlainTextEdit("\n".join(" ".join(f"{v:.17g}" for v in row) for row in points))
    editor.setPlaceholderText("x_m y_m z_m · one vertex per row, in path order")
    editor.setMaximumHeight(100)
    button = QPushButton("Import points…")

    def load():
        filename, _ = QFileDialog.getOpenFileName(editor, label, "", "Coordinate text (*.csv *.txt);;All files (*)")
        if filename:
            try:
                editor.setPlainText(Path(filename).read_text(encoding="utf-8-sig"))
            except OSError as exc:
                QMessageBox.warning(editor, "Cannot import points", str(exc))

    button.clicked.connect(load)
    form.addRow(label, editor)
    form.addRow(button)
    return editor, button


class _EngineeringNumber(ScientificSpinBox):
    """Keep every stored double significant digit through Qt's interpretText."""

    def textFromValue(self, value):
        return self.locale().toString(value, "g", 17)


class CircuitDriveDialog(EngineeringDialog):
    def __init__(self, project, record, parent=None):
        from ..gradient_engineering import GradientEngineeringSpec, winding_path_sha256

        super().__init__("Terminals, feeds and drive waveform", parent)
        self.project, self.record = project, record
        self.original_value = json_value(project.study.get("engineering"))
        self.existing = GradientEngineeringSpec.from_dict(self.original_value or {})
        self.note(
            "Supply runs from the positive drive terminal to the retained winding start; return runs from its finish to the negative terminal. Distinct drive terminal positions require an explicit modeled closure. Coordinates use world-frame metres. Imported CSV/TXT polylines are headerless numeric x,y,z rows in path order."
        )
        arrays = record.select_arrays(copy=False)
        metadata = record.metadata
        plans = metadata.get("winding_steps") or []
        if not plans:
            raise ValueError("Select a completed Fabrication review run before defining its terminals and feeds.")
        recorded_bodies = {row["body_id"]: row for row in (metadata.get("fabrication") or {}).get("paths", ())}
        drive = self.existing.drive.to_dict() if self.existing.drive else {}
        drive_ids = list(drive.get("body_ids") or ())
        self._previous_drive_ids = drive_ids
        tabs = QTabWidget()
        self.body_controls = []
        self.body_points, self.body_closed = {}, {}
        self.current_digests, self.rebind_controls = {}, {}
        self.circuit_sources, self.circuit_imports, self.drive_members = {}, {}, {}
        existing = {item.body_id: item.to_dict() for item in self.existing.circuits}
        for index, plan in enumerate(plans):
            body_id = str(plan.get("body_id", f"body_{index}"))
            points = arrays.get(f"review_conductor_{index}_points_m")
            if points is None:
                raise ValueError(f"Missing recorded winding coordinates for {body_id}.")
            closed = recorded_bodies.get(body_id, {}).get("closed")
            if not isinstance(closed, bool):
                raise ValueError(f"Saved winding closure metadata is unavailable for {body_id}.")
            self.body_points[body_id], self.body_closed[body_id] = points, closed
            digest = winding_path_sha256(points)
            self.current_digests[body_id] = digest
            saved = existing.get(body_id, {})
            page = QWidget()
            form = QFormLayout(page)
            enabled = QCheckBox("Define this circuit")
            enabled.setChecked(bool(saved))
            form.addRow(enabled)
            binding = QLabel(
                f"Recorded winding: {len(points)} points; {'closed: an explicit feed break is required' if closed else 'open: no feed break is allowed'}. SHA-256: {digest}"
            )
            binding.setWordWrap(True)
            form.addRow(binding)
            rebind = QCheckBox("Bind edited circuit to this selected winding revision")
            rebind.setChecked(not saved)
            self.rebind_controls[body_id] = rebind
            form.addRow(rebind)
            if saved and saved["expected_path_sha256"] != digest:
                warning = QLabel(
                    "Stored circuit geometry is stale for this winding. Retain its reference or explicitly rebind and correct every endpoint before applying."
                )
                warning.setWordWrap(True)
                form.addRow(warning)
            polarity = QComboBox()
            polarity.addItem("Saved current direction", 1)
            polarity.addItem("Reverse saved current direction", -1)
            polarity.setCurrentIndex(1 if saved.get("polarity") == -1 else 0)
            cut = QSpinBox()
            original_cut = saved.get("feed_break_segment")
            cut.setRange(-1, max(len(points) - 2, original_cut or 0))
            cut.setSpecialValueText("No break · already open")
            cut.setValue(original_cut if original_cut is not None else -1)
            positive = QLineEdit(saved.get("positive_terminal", "+"))
            negative = QLineEdit(saved.get("negative_terminal", "−"))
            form.addRow("Current polarity", polarity)
            form.addRow("Feed break · saved interval index", cut)
            form.addRow("Positive terminal name", positive)
            form.addRow("Negative terminal name", negative)
            supply, supply_import = _polyline_editor(form, "Supply [m]", saved.get("supply_points_m", ()))
            return_, return_import = _polyline_editor(form, "Return [m]", saved.get("return_points_m", ()))
            closure, closure_import = _polyline_editor(
                form, "Explicit drive closure [m] · optional", saved.get("drive_closure_points_m") or ()
            )
            self.circuit_imports[body_id] = (supply_import, return_import, closure_import)
            source = QLineEdit(saved.get("source", f"User-authored feed against verified run {record.directory.name}"))
            self.circuit_sources[body_id] = source
            form.addRow("Circuit geometry source", source)
            member = QCheckBox("Include this circuit in the drive waveform / target")
            member.setChecked(body_id in drive_ids)
            self.drive_members[body_id] = member
            form.addRow(member)
            tabs.addTab(page, body_id)
            self.body_controls.append(
                (
                    body_id,
                    saved.get("expected_path_sha256", digest),
                    enabled,
                    polarity,
                    cut,
                    positive,
                    negative,
                    supply,
                    return_,
                    closure,
                )
            )
        self.form.addRow(tabs)
        self.unrepresented = [item for item in self.existing.circuits if item.body_id not in self.body_points]
        self.discard_missing = QCheckBox("Remove stored circuits whose body is absent from this selected run")
        if self.unrepresented:
            self.form.addRow(self.discard_missing)
            self.note("Stored circuits without matching body evidence remain preserved until explicitly removed.")
        self.drive_enabled = QCheckBox("Evaluate a prescribed drive")
        self.drive_enabled.setChecked(bool(drive))
        self.form.addRow(self.drive_enabled)
        self.drive_mode = QComboBox()
        if drive and not drive.get("time_s"):
            self.drive_mode.addItem("Preserve saved target gradient / slew and verified-reference inputs", "target")
        self.drive_mode.addItem("Explicit current waveform", "waveform")
        self.form.addRow("Drive mode", self.drive_mode)
        self.waveform_body_label = QLabel()
        self.waveform_body_label.setWordWrap(True)
        self.form.addRow(self.waveform_body_label)
        self.note(
            "Waveform rows contain time_s followed by signed current_A in the listed body order. Piecewise-linear currents define interval slopes and voltage extrema. Stored target references and mutual matrices remain unchanged unless explicitly replaced; geometry mismatch makes their estimates unavailable."
        )
        self.waveform = QPlainTextEdit()
        if drive.get("time_s"):
            self.waveform.setPlainText(
                "\n".join(
                    " ".join(f"{v:.17g}" for v in [time, *current])
                    for time, current in zip(drive["time_s"], drive["currents_A"], strict=True)
                )
            )
        self.waveform.setPlaceholderText("0 0\n0.001 100\n0.002 0")
        self.form.addRow("Time and current waveform [s, A]", self.waveform)
        self.limit_controls = []
        self.limit_by_body = {}
        for row in self.body_controls:
            body_id = row[0]
            index = drive_ids.index(body_id) if body_id in drive_ids else None
            current = self._number(drive["current_limits_A"][index] if index is not None else 1000)
            voltage = self._number(drive["voltage_limits_V"][index] if index is not None else 1000)
            self.limit_controls.append((current, voltage))
            self.limit_by_body[body_id] = (current, voltage)
            self.form.addRow(f"{body_id} amplifier current limit [A]", current)
            self.form.addRow(f"{body_id} amplifier voltage limit [V]", voltage)
        self.source = QLineEdit(drive.get("source", ""))
        self.form.addRow("Drive / amplifier source", self.source)
        self.retain_mutual = QCheckBox("Retain the stored mutual-inductance specification")
        self.retain_mutual.setChecked(drive.get("inductance_matrix") is not None)
        if drive.get("inductance_matrix") is not None:
            self.form.addRow(self.retain_mutual)
        thermal = drive.get("thermal") or {}
        self.thermal_enabled = QCheckBox("Estimate temperature with a measured shared lumped RC model")
        self.thermal_enabled.setChecked(bool(thermal))
        self.form.addRow(self.thermal_enabled)
        self.thermal_measured = QCheckBox("These lumped parameters were measured for this assembly")
        self.thermal_measured.setChecked(thermal.get("measured") is True)
        self.form.addRow(self.thermal_measured)
        self.thermal_controls = {}
        for key, label, fallback in (
            ("thermal_resistance_K_per_W", "Measured thermal resistance [K/W]", 1.0),
            ("heat_capacity_J_per_K", "Measured heat capacity [J/K]", 100.0),
            ("ambient_temperature_K", "Ambient temperature [K]", 293.15),
            ("initial_temperature_K", "Initial temperature [K]", 293.15),
            ("max_temperature_K", "Model temperature limit [K]", 333.15),
        ):
            control = self._number(thermal.get(key, fallback))
            self.thermal_controls[key] = control
            self.form.addRow(label, control)
        self.thermal_source = QLineEdit(thermal.get("source", ""))
        self.form.addRow("Measured thermal parameter source", self.thermal_source)
        self.note(
            "The measured shared RC model uses the explicit finite waveform, constant electrical resistance and aggregate Joule loss. It cannot establish cooling, hotspot, insulation or manufacturing qualification. Target gradient/slew alone does not define a duty cycle."
        )
        self.drive_enabled.toggled.connect(self._refresh_drive_controls)
        self.drive_mode.currentIndexChanged.connect(self._refresh_drive_controls)
        self.thermal_enabled.toggled.connect(self._refresh_drive_controls)
        for member in self.drive_members.values():
            member.toggled.connect(self._refresh_drive_controls)
        for body_id, _, enabled, *_ in self.body_controls:
            enabled.toggled.connect(lambda checked, body=body_id: self._circuit_toggled(body, checked))
        self._refresh_drive_controls()
        self.resize(800, 850)

    @staticmethod
    def _number(value):
        control = _EngineeringNumber()
        control.setDecimals(323)
        control.setRange(0, max(1e12, float(value)))
        control.setValue(float(value))
        return control

    def _drive_body_ids(self):
        selected = {
            row[0] for row in self.body_controls if row[2].isChecked() and self.drive_members[row[0]].isChecked()
        }
        old = [body for body in self._previous_drive_ids if body in selected]
        return old + [row[0] for row in self.body_controls if row[0] in selected and row[0] not in old]

    def _circuit_toggled(self, body, checked):
        if not checked or self.existing.drive is None:
            self.drive_members[body].setChecked(checked)
        self._refresh_drive_controls()

    def _refresh_drive_controls(self, *_args):
        active = self.drive_enabled.isChecked()
        for body, _, enabled, polarity, cut, positive, negative, supply, return_, closure in self.body_controls:
            defined = enabled.isChecked()
            for control in (
                polarity,
                positive,
                negative,
                supply,
                return_,
                closure,
                self.rebind_controls[body],
                self.circuit_sources[body],
                *self.circuit_imports[body],
            ):
                control.setEnabled(defined)
            cut.setEnabled(defined and self.body_closed[body])
            self.drive_members[body].setEnabled(defined and active)
        self.drive_mode.setEnabled(active)
        self.waveform.setEnabled(active and self.drive_mode.currentData() == "waveform")
        ids = self._drive_body_ids()
        self.waveform_body_label.setText(
            "Current columns after time_s: " + (", ".join(ids) if ids else "select the circuits to drive in their tabs")
        )
        for body, pair in self.limit_by_body.items():
            for control in pair:
                control.setEnabled(active and body in ids)
        self.source.setEnabled(active)
        self.retain_mutual.setEnabled(active)
        self.thermal_enabled.setEnabled(active)
        thermal = active and self.thermal_enabled.isChecked()
        for control in (self.thermal_measured, self.thermal_source, *self.thermal_controls.values()):
            control.setEnabled(thermal)

    def value(self):
        from ..gradient_engineering import CircuitSpec, GradientEngineeringSpec

        circuits = []
        if self.unrepresented and not self.discard_missing.isChecked():
            raise ValueError(
                "Stored circuits have no matching winding evidence. Explicitly remove them or select their recorded run."
            )
        existing = self.existing.circuit_map
        for body_id, digest, enabled, polarity, cut, positive, negative, supply, return_, closure in self.body_controls:
            if not enabled.isChecked():
                continue
            value = existing[body_id].to_dict() if body_id in existing else {}
            value.update(
                {
                    "body_id": body_id,
                    "expected_path_sha256": self.current_digests[body_id]
                    if self.rebind_controls[body_id].isChecked()
                    else digest,
                    "feed_break_segment": None if cut.value() == -1 else cut.value(),
                    "polarity": polarity.currentData(),
                    "positive_terminal": positive.text(),
                    "negative_terminal": negative.text(),
                    "supply_points_m": _matrix(supply.toPlainText(), 3, "Supply"),
                    "return_points_m": _matrix(return_.toPlainText(), 3, "Return"),
                    "drive_closure_points_m": _matrix(closure.toPlainText(), 3, "Closure", empty=True) or None,
                    "source": self.circuit_sources[body_id].text(),
                    "coordinate_frame": "world",
                    "length_unit": "m",
                }
            )
            circuit = CircuitSpec.from_dict(value)
            circuit.construct_path(self.body_points[body_id], closed=self.body_closed[body_id])
            circuits.append(circuit.to_dict())
        selected_circuits = {value["body_id"]: value for value in circuits}
        prior_order = [value.body_id for value in self.existing.circuits if value.body_id in selected_circuits]
        circuits = [selected_circuits[body] for body in prior_order] + [
            value for value in circuits if value["body_id"] not in prior_order
        ]
        drive = None
        if self.drive_enabled.isChecked():
            ids = self._drive_body_ids()
            previous = self.existing.drive.to_dict() if self.existing.drive else {}
            drive = dict(previous)
            drive.update(
                {
                    "body_ids": ids,
                    "current_limits_A": [_values(self.limit_by_body[body][0]) for body in ids],
                    "voltage_limits_V": [_values(self.limit_by_body[body][1]) for body in ids],
                    "source": self.source.text(),
                }
            )
            if self.drive_mode.currentData() == "waveform":
                waveform = np.asarray(_matrix(self.waveform.toPlainText(), len(ids) + 1, "Waveform"))
                for key in (
                    "target_gradient_T_per_m",
                    "target_slew_T_per_m_per_s",
                    "gradient_efficiencies_T_per_m_per_A",
                    "gradient_efficiency_path_sha256",
                    "gradient_efficiency_source_sha256",
                    "gradient_efficiency_source",
                ):
                    drive.pop(key, None)
                drive.update(time_s=waveform[:, 0].tolist(), currents_A=waveform[:, 1:].tolist())
            if not self.retain_mutual.isChecked():
                drive["inductance_matrix"] = None
            drive["thermal"] = None
            if self.thermal_enabled.isChecked():
                drive["thermal"] = {
                    **{key: _values(control) for key, control in self.thermal_controls.items()},
                    "source": self.thermal_source.text(),
                    "measured": self.thermal_measured.isChecked(),
                }
        if self.original_value is None and not circuits and drive is None:
            return None
        return GradientEngineeringSpec.from_dict(
            {
                "circuits": circuits,
                "drive": drive,
                "independent_winding_validation": self.existing.independent_winding_validation,
            }
        ).to_dict()


class MeasuredFieldDialog(EngineeringDialog):
    def __init__(self, parent=None, *, body_ids=()):
        super().__init__("Compare measured coil field", parent)
        self.note(
            "Import finite CSV rows: x y z Bx By Bz, with the units selected below. The full vector and coordinates are rotated into the world frame. Saved electrical geometry is evaluated directly at measured locations, without interpolation or a current fit."
        )
        self.filename = QLineEdit()
        button = QPushButton("Choose field CSV…")
        button.clicked.connect(self._choose)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.addWidget(self.filename)
        layout.addWidget(button)
        self.form.addRow("Measured map", row)
        self.length_unit, self.field_unit = QComboBox(), QComboBox()
        self.length_unit.addItems(["m", "mm"])
        self.field_unit.addItems(["T", "mT", "uT"])
        self.form.addRow("Coordinate unit", self.length_unit)
        self.form.addRow("Field unit", self.field_unit)
        self.rotation = QPlainTextEdit("1 0 0\n0 1 0\n0 0 1")
        self.rotation.setMaximumHeight(85)
        self.translation = QLineEdit("0 0 0")
        self.form.addRow("Measured → world rotation matrix", self.rotation)
        self.form.addRow("Measured → world translation [m]", self.translation)
        self.currents = QLineEdit()
        self.body_ids = tuple(body_ids)
        self.form.addRow("Calibration body_id=current_A pairs", self.currents)
        if self.body_ids:
            self.currents.setPlaceholderText(" ".join(f"{body}=I" for body in self.body_ids))
            self.note("Required saved winding bodies: " + ", ".join(self.body_ids))
        self.source = QLineEdit()
        self.form.addRow("Measurement / calibration source", self.source)
        self.independent = QCheckBox("These measurements were held out from coil optimization")
        self.form.addRow(self.independent)
        self.rms_limit, self.peak_limit = _number(0), _number(0)
        self.form.addRow("RMS vector error limit [µT] · 0 unspecified", self.rms_limit)
        self.form.addRow("Peak vector error limit [µT] · 0 unspecified", self.peak_limit)
        self.note(
            "Every modeled body's measured current must be declared explicitly. Calibration changes the comparison current only; it cannot overwrite the saved design's acceptance. Preserve the returned comparison archive with its source data and provenance."
        )
        self.resize(700, 590)

    def _choose(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Measured field", "", "Measured CSV (*.csv *.txt);;All files (*)"
        )
        if filename:
            self.filename.setText(filename)

    def value(self):
        from ..application.gradient_studies import RigidTransform

        filename = Path(self.filename.text()).expanduser().resolve()
        source_bytes = filename.read_bytes()
        raw = source_bytes.decode("utf-8-sig")
        if raw.splitlines() and raw.splitlines()[0].replace(" ", "") == "x_m,y_m,z_m,Bx_T,By_T,Bz_T":
            if self.length_unit.currentText() != "m" or self.field_unit.currentText() != "T":
                raise ValueError("The named SI CSV header requires meter and tesla units.")
            raw = "\n".join(raw.splitlines()[1:])
        values = _matrix(raw, 6, "Measured map")
        currents = {}
        for entry in self.currents.text().replace(",", " ").split():
            body, current = entry.split("=", 1)
            if not body or body in currents:
                raise ValueError("Repeated current calibration body.")
            currents[body] = float(current)
        if not self.independent.isChecked():
            raise ValueError("Declare that this measured map was held out from optimization.")
        if not currents or not all(np.isfinite(value) for value in currents.values()) or not self.source.text().strip():
            raise ValueError("Declare finite current calibration for every body and its source.")
        if self.body_ids and set(currents) != set(self.body_ids):
            raise ValueError("Declare exactly one calibrated signed current for every saved winding body.")
        translations = _matrix(self.translation.text(), 3, "Translation")
        if len(translations) != 1:
            raise ValueError("Enter exactly one measured-to-world translation vector.")
        transform = RigidTransform(_matrix(self.rotation.toPlainText(), 3, "Rotation"), translations[0])
        return {
            "values": values,
            "source_bytes": source_bytes,
            "source_file": str(filename),
            "length_unit": self.length_unit.currentText(),
            "field_unit": self.field_unit.currentText(),
            "rotation": transform.rotation,
            "translation_m": transform.translation_m,
            "currents_A": currents,
            "source": self.source.text(),
            "independent": True,
            "rms_limit_T": _values(self.rms_limit) * 1e-6 or None,
            "peak_limit_T": _values(self.peak_limit) * 1e-6 or None,
        }
