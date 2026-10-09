"""Focused circuit-reference import, external analysis and time/frequency review."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..application.dynamic_studies import (
    ANALYSIS_SCHEMA,
    REFERENCE_SCHEMA,
    DynamicStudySpec,
    import_circuit_reference,
    import_external_analysis,
    waveform_digest,
)
from ..gradient_engineering import GradientEngineeringSpec
from ..persistence import json_value
from .gradient_tools import EngineeringDialog, _number, _values


class TracePlot(QWidget):
    """Small dependency-free plot of explicit samples, including interval jumps."""

    def __init__(self, x, curves, *, x_label, y_label, limits=(), parent=None):
        super().__init__(parent)
        self.x = np.asarray(x, dtype=float)
        self.curves = [(name, np.asarray(values, dtype=float)) for name, values in curves]
        self.x_label, self.y_label, self.limits = x_label, y_label, tuple(limits)
        self.setMinimumSize(500, 220)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        foreground = self.palette().text().color()
        painter.setPen(foreground)
        painter.drawText(12, 20, self.y_label)
        if not len(self.x) or not self.curves:
            painter.drawText(50, 80, "No recorded samples")
            return
        all_y = np.concatenate([values for _, values in self.curves] + [np.asarray(self.limits)])
        if np.any(np.isinf(all_y)) or not np.all(np.isfinite(self.x)):
            painter.drawText(50, 80, "Invalid recorded samples")
            return
        all_y = all_y[np.isfinite(all_y)]
        if not len(all_y):
            painter.drawText(50, 80, "No finite phase samples: response magnitude is zero")
            return
        xmin, xmax, ymin, ymax = float(self.x.min()), float(self.x.max()), float(all_y.min()), float(all_y.max())
        if xmax == xmin:
            xmax = xmin + max(1, abs(xmin) * 0.1)
        if ymax == ymin:
            ymin, ymax = ymin - max(1e-12, abs(ymin) * 0.1), ymax + max(1e-12, abs(ymax) * 0.1)
        padding = (ymax - ymin) * 0.06
        ymin, ymax = ymin - padding, ymax + padding
        left, top, width, height = 84, 43, self.width() - 110, self.height() - 89

        def at(x, y):
            return QPointF(left + (x - xmin) / (xmax - xmin) * width, top + (ymax - y) / (ymax - ymin) * height)

        painter.drawLine(left, top, left, top + height)
        painter.drawLine(left, top + height, left + width, top + height)
        for fraction in (0, 0.5, 1):
            y = ymin + (ymax - ymin) * fraction
            painter.drawText(5, int(at(xmin, y).y()) + 4, f"{y:.4g}")
        painter.drawText(left, top + height + 18, f"{xmin:.4g}")
        painter.drawText(left + width - 65, top + height + 18, f"{xmax:.4g}")
        painter.drawText(left + width // 2 - 40, self.height() - 8, self.x_label)
        colors = ("#0067b1", "#cb5731", "#198447", "#8955ab", "#a07800", "#169092")
        legend_x = left
        for index, (name, values) in enumerate(self.curves):
            painter.setPen(QPen(QColor(colors[index % len(colors)]), 1.6))
            path = QPainterPath()
            started = False
            for point_index, (x, y) in enumerate(zip(self.x, values, strict=True)):
                if not np.isfinite(y):
                    started = False
                    continue
                point = at(x, y)
                path.lineTo(point) if started else path.moveTo(point)
                started = True
                if len(self.x) == 1:
                    painter.drawEllipse(point, 2.5, 2.5)
            painter.drawPath(path)
            painter.drawText(legend_x, 34, name)
            legend_x += max(90, len(name) * 7 + 15)
        painter.setPen(QPen(foreground, 1, Qt.PenStyle.DashLine))
        for limit in self.limits:
            painter.drawLine(at(xmin, limit), at(xmax, limit))


class DynamicStudyDialog(EngineeringDialog):
    """Apply references and external evidence as one validated project transaction."""

    def __init__(self, project, record=None, parent=None):
        super().__init__("Coupled circuits and external analysis evidence", parent)
        self.project, self.record = project, record
        self.original = json_value(project.study.get("dynamic_analysis"))
        self.spec = DynamicStudySpec.from_dict(self.original or {})
        self.engineering = GradientEngineeringSpec.from_dict(json_value(project.study.get("engineering")) or {})
        self.references = {"inductance": self.spec.inductance_reference, "efficiency": self.spec.efficiency_reference}
        self.analyses = list(self.spec.analyses)
        self.paths = self._current_paths()
        self.note(
            "Import body-ordered inductance and signed gradient efficiency references for the exact complete electrical paths. Numerical checks and source hashes verify the import/binding; external solver accuracy remains a separate engineering assessment."
        )
        identity = QPlainTextEdit()
        identity.setReadOnly(True)
        identity.setMaximumHeight(100)
        identity.setPlainText(
            "\n".join(f"{body}: {digest}" for body, digest in self.paths.items())
            or "Define complete circuits against a saved winding before importing circuit references."
        )
        self.form.addRow("Current electrical path SHA-256", identity)
        self.reference_labels = {}
        for kind, label in (
            ("inductance", "Coupled inductance [H]"),
            ("efficiency", "Signed gradient efficiency [T/m/A]"),
        ):
            row = QWidget()
            layout = QHBoxLayout(row)
            status = QLabel()
            status.setWordWrap(True)
            self.reference_labels[kind] = status
            layout.addWidget(status, 1)
            choose = QPushButton("Import…")
            choose.clicked.connect(lambda checked=False, name=kind: self._import_reference(name))
            clear = QPushButton("Remove")
            clear.clicked.connect(lambda checked=False, name=kind: self._remove_reference(name))
            layout.addWidget(choose)
            layout.addWidget(clear)
            self.form.addRow(label, row)
        template_row = QWidget()
        template_layout = QHBoxLayout(template_row)
        self.reference_schema_kind = QComboBox()
        self.reference_schema_kind.addItem("Inductance matrix", "inductance")
        self.reference_schema_kind.addItem("Signed gradient efficiencies", "efficiency")
        template = QPushButton("Copy circuit reference import schema")
        template.clicked.connect(self._copy_reference_schema)
        template_layout.addWidget(self.reference_schema_kind)
        template_layout.addWidget(template)
        self.form.addRow(template_row)
        drive = self.engineering.drive
        self.target_drive = QCheckBox("Use imported efficiency for a target gradient / slew drive")
        self.target_drive.setChecked(bool(drive and not drive.time_s))
        self.gradient = _number(drive.target_gradient_T_per_m if drive and not drive.time_s else 0.04, minimum=1e-15)
        self.slew = _number(drive.target_slew_T_per_m_per_s if drive and not drive.time_s else 100, minimum=1e-15)
        self.form.addRow(self.target_drive)
        self.form.addRow("Target gradient per driven body [T/m]", self.gradient)
        self.form.addRow("Target slew per driven body [T/m/s]", self.slew)
        self.note(
            "Target mode applies the existing per-body gradient/slew contract. A combined array field and signed-current fit is reviewed in the array study."
        )
        self.target_drive.toggled.connect(self._refresh)
        self.material = QLineEdit(self.spec.material_sha256 or "")
        material_row = QWidget()
        layout = QHBoxLayout(material_row)
        layout.addWidget(self.material)
        material_button = QPushButton("Hash material contract file…")
        material_button.clicked.connect(self._material_file)
        layout.addWidget(material_button)
        self.form.addRow("Assembly material contract SHA-256", material_row)
        self.environment = QLineEdit(self.spec.environment_sha256 or "")
        environment_row = QWidget()
        layout = QHBoxLayout(environment_row)
        layout.addWidget(self.environment)
        environment_button = QPushButton("Hash environment geometry file…")
        environment_button.clicked.connect(self._environment_file)
        layout.addWidget(environment_button)
        self.form.addRow("Surrounding geometry contract SHA-256", environment_row)
        self.note(
            "Use a reproducible material/source definition including conducting sleeves, conductivity, permeability and mechanical/thermal properties as applicable. A file digest binds external results to that definition; it does not infer properties from the support mesh."
        )
        self.form.addRow("Current waveform SHA-256", QLabel(waveform_digest(drive) or "No explicit current waveform"))
        self.analysis_list = QListWidget()
        self.analysis_list.setMinimumHeight(120)
        self.form.addRow("Imported external solver samples", self.analysis_list)
        row = QWidget()
        layout = QHBoxLayout(row)
        add, remove, schema = (
            QPushButton("Import analysis JSON…"),
            QPushButton("Remove selected"),
            QPushButton("Copy analysis schema"),
        )
        self.analysis_schema_kind = QComboBox()
        self.analysis_schema_kind.addItem("Frequency response", "frequency_response")
        self.analysis_schema_kind.addItem("Mechanical load / vibration", "mechanical")
        self.analysis_schema_kind.addItem("Spatial temperature", "spatial_thermal")
        add.clicked.connect(self._import_analysis)
        remove.clicked.connect(self._remove_analysis)
        schema.clicked.connect(self._copy_analysis_schema)
        for button in (add, remove, schema):
            layout.addWidget(button)
        self.form.addRow(row)
        self.form.addRow("Schema for external solver adapter", self.analysis_schema_kind)
        self.note(
            "Frequency responses preserve separate real/imaginary arrays and explicit Hz samples. Mechanical and spatial thermal results bind the signed waveform. Acceptance limits apply to supplied samples only; missing analyses remain unassessed. Apply, then recompute the fabrication review to check current geometry and display plots."
        )
        self._refresh()
        self.resize(920, 850)

    def _current_paths(self):
        if self.record is None:
            return {}
        plans = self.record.metadata.get("winding_steps", ())
        closed = {
            row["body_id"]: row.get("closed")
            for row in (self.record.metadata.get("fabrication") or {}).get("paths", ())
        }
        arrays = self.record.select_arrays(
            [f"review_conductor_{index}_points_m" for index in range(len(plans))], copy=False
        )
        paths = {}
        for index, plan in enumerate(plans):
            body = plan["body_id"]
            circuit = self.engineering.circuit_map.get(body)
            if circuit is not None and isinstance(closed.get(body), bool):
                try:
                    path = circuit.construct_path(arrays[f"review_conductor_{index}_points_m"], closed=closed[body])
                except (ValueError, KeyError):
                    continue
                paths[body] = path["electrical_path_sha256"]
        return paths

    def _refresh(self, *_):
        for kind, label in self.reference_labels.items():
            reference = self.references[kind]
            if reference is None:
                text = "No reference supplied"
            else:
                matched = tuple(self.paths.get(body) for body in reference.body_ids) == reference.path_sha256
                text = f"{reference.source}; {len(reference.body_ids)} bodies; {'geometry matches' if matched else 'geometry missing/stale'}"
                if reference.kind == "efficiency":
                    text += (
                        f"; {reference.field_component}, derivative along world {reference.gradient_direction_world}"
                    )
            label.setText(text)
        self.gradient.setEnabled(self.target_drive.isChecked())
        self.slew.setEnabled(self.target_drive.isChecked())
        self.analysis_list.clear()
        for analysis in self.analyses:
            self.analysis_list.addItem(
                f"{analysis.name} · {analysis.kind} · {len(analysis.axis)} {analysis.axis_unit} samples · {analysis.quantity} [{analysis.unit}]"
            )

    def _import_reference(self, kind):
        filename, _ = QFileDialog.getOpenFileName(self, f"Import {kind} reference", "", "Reference JSON (*.json)")
        if not filename:
            return
        try:
            reference = import_circuit_reference(filename)
            if reference.kind != kind:
                raise ValueError(f"Choose a {kind} reference file.")
        except (ValueError, TypeError, OSError) as exc:
            QMessageBox.warning(self, "Reference import failed", str(exc))
            return
        self.references[kind] = reference
        self._refresh()

    def _remove_reference(self, kind):
        self.references[kind] = None
        self._refresh()

    def _material_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select assembly material/source contract", "", "All files (*)")
        if filename:
            try:
                self.material.setText(sha256(Path(filename).read_bytes()).hexdigest())
            except OSError as exc:
                QMessageBox.warning(self, "Material source unavailable", str(exc))

    def _environment_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select conducting/mechanical environment geometry contract", "", "All files (*)"
        )
        if filename:
            try:
                self.environment.setText(sha256(Path(filename).read_bytes()).hexdigest())
            except OSError as exc:
                QMessageBox.warning(self, "Environment geometry source unavailable", str(exc))

    def _import_analysis(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Import external solver evidence", "", "Analysis JSON (*.json)")
        if not filename:
            return
        try:
            analysis = import_external_analysis(filename)
            if any(item.name == analysis.name for item in self.analyses):
                raise ValueError("Analysis names must be unique. Remove the existing record before replacing it.")
            self.analyses.append(analysis)
            DynamicStudySpec(
                analyses=self.analyses,
                material_sha256=self.material.text() or None,
                environment_sha256=self.environment.text() or None,
            )
        except (ValueError, TypeError, OSError) as exc:
            if "analysis" in locals() and self.analyses and self.analyses[-1] is analysis:
                self.analyses.pop()
            QMessageBox.warning(self, "Analysis import failed", str(exc))
            return
        self._refresh()

    def _remove_analysis(self):
        index = self.analysis_list.currentRow()
        if 0 <= index < len(self.analyses):
            del self.analyses[index]
            self._refresh()

    def _copy_reference_schema(self):
        ids = list(self.engineering.drive.body_ids) if self.engineering.drive else list(self.paths)
        kind = self.reference_schema_kind.currentData()
        value = {
            "schema": REFERENCE_SCHEMA,
            "kind": kind,
            "body_ids": ids,
            "path_sha256": [self.paths.get(body, "REPLACE_WITH_COMPLETE_PATH_SHA256") for body in ids],
            "values": np.eye(len(ids)).tolist() if kind == "inductance" else np.ones(len(ids)).tolist(),
            "unit": "uH" if kind == "inductance" else "mT/m/A",
            "source": "REPLACE_WITH_SOURCE",
            "method": "REPLACE_WITH_SOLVER_OR_MEASUREMENT_METHOD",
        }
        if kind == "efficiency":
            value.update(field_component="Bz", gradient_direction_world=[0, 0, 1])
        QApplication.clipboard().setText(json.dumps(value, indent=2))

    def _copy_analysis_schema(self):
        ids = list(self.paths)
        kind = self.analysis_schema_kind.currentData()
        quantity, unit = {
            "frequency_response": ("field_response", "T/A"),
            "mechanical": ("force", "N"),
            "spatial_thermal": ("temperature", "K"),
        }[kind]
        frequency = kind == "frequency_response"
        value = {
            "schema": ANALYSIS_SCHEMA,
            "name": "REPLACE_WITH_ANALYSIS_NAME",
            "kind": kind,
            "body_ids": ids,
            "path_sha256": [self.paths[body] for body in ids],
            "material_sha256": self.material.text() or "REPLACE_WITH_MATERIAL_SHA256",
            "environment_sha256": self.environment.text() or "REPLACE_WITH_ENVIRONMENT_GEOMETRY_SHA256",
            "excitation_body_id": (ids[0] if ids else "REPLACE_WITH_UNIT_CURRENT_DRIVE_BODY") if frequency else None,
            "source": "REPLACE_WITH_SOURCE",
            "solver": "REPLACE_WITH_SOLVER",
            "solver_version": "REPLACE_WITH_VERSION",
            "coordinate_frame": "world",
            "points_m": [[0, 0, 0]],
            "axis": [1000] if frequency else [0],
            "axis_unit": "Hz" if frequency else "s",
            "quantity": quantity,
            "unit": unit,
            "real": [[[293.15]]] if kind == "spatial_thermal" else [[[0, 0, 0]]],
            "imaginary": [[[0, 0, 0]]] if frequency else None,
            "waveform_sha256": None
            if frequency
            else waveform_digest(self.engineering.drive) or "REPLACE_WITH_WAVEFORM_SHA256",
            "limit": None,
        }
        QApplication.clipboard().setText(json.dumps(value, indent=2))

    def value(self):
        self.engineering_value()  # Validate the whole reference/drive transaction.
        if (
            self.original is None
            and not any(self.references.values())
            and not self.analyses
            and not self.material.text()
            and not self.environment.text()
        ):
            return None
        return DynamicStudySpec(
            self.references["inductance"],
            self.references["efficiency"],
            self.analyses,
            self.material.text() or None,
            self.environment.text() or None,
        ).to_dict()

    def engineering_value(self):
        if (
            not any(self.references.values())
            and not self.target_drive.isChecked()
            and self.spec.inductance_reference is None
        ):
            return self.engineering.to_dict() if self.project.study.get("engineering") is not None else None
        drive = self.engineering.drive
        if drive is None:
            raise ValueError("Define complete circuits, body current/voltage limits and a drive first.")
        value = drive.to_dict()
        if self.spec.inductance_reference is not None and self.references["inductance"] is None:
            value["inductance_matrix"] = None
        for reference in self.references.values():
            if reference is None:
                continue
            if reference.body_ids != drive.body_ids:
                raise ValueError("Imported reference body order must exactly match the authored drive order.")
            if tuple(self.paths.get(body) for body in reference.body_ids) != reference.path_sha256:
                raise ValueError(
                    "Imported reference does not match the exact complete electrical paths of this selected winding."
                )
            if reference.kind == "inductance":
                value["inductance_matrix"] = {
                    "body_ids": list(reference.body_ids),
                    "values_H": np.asarray(reference.values).tolist(),
                    "expected_electrical_path_sha256": list(reference.path_sha256),
                    "source": reference.source,
                    "verified": False,
                }
        if self.target_drive.isChecked():
            reference = self.references["efficiency"]
            if reference is None:
                raise ValueError(
                    "Target gradient/slew mode requires an imported geometry-bound signed efficiency reference."
                )
            value.update(
                time_s=[],
                currents_A=[],
                repetitions=1,
                rest_s=0,
                target_gradient_T_per_m=_values(self.gradient),
                target_slew_T_per_m_per_s=_values(self.slew),
                gradient_efficiencies_T_per_m_per_A=list(reference.values),
                gradient_efficiency_path_sha256=list(reference.path_sha256),
                gradient_efficiency_source_sha256=reference.source_sha256,
                gradient_efficiency_source=reference.source,
            )
        elif not drive.time_s:
            raise ValueError("Edit an explicit waveform in the circuit dialog before leaving target drive mode.")
        result = self.engineering.to_dict()
        result["drive"] = value
        return GradientEngineeringSpec.from_dict(result).to_dict()


class DynamicReviewDialog(QDialog):
    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drive and external coupled-physics review")
        self.resize(950, 790)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        engineering = report.get("engineering_review") or {}
        waveform = engineering.get("waveform")
        if waveform:
            page = QWidget()
            form = QVBoxLayout(page)
            label = QLabel(
                f"Finite drive: {waveform['repetitions']} repetitions; {waveform['duration_s']:.6g} s. "
                + waveform["scope"]
            )
            label.setWordWrap(True)
            form.addWidget(label)
            self.trace_body = QComboBox()
            self.trace_body.addItem("All driven bodies", None)
            for index, body in enumerate(waveform["body_ids"]):
                self.trace_body.addItem(body, index)
            form.addWidget(self.trace_body)
            time = np.asarray(waveform["time_s"])
            currents = np.asarray(waveform["currents_A"])
            body_rows = engineering.get("drive", {}).get("bodies", ())
            current_plot = TracePlot(
                time,
                [(body, currents[:, index]) for index, body in enumerate(waveform["body_ids"])],
                x_label="Time [s]",
                y_label="Signed current [A]",
                limits=[sign * row["current_limit_A"] for row in body_rows for sign in (-1, 1)],
            )
            form.addWidget(current_plot)
            interval_time = np.column_stack((time[:-1], time[1:])).ravel()
            voltage = np.stack(
                (waveform["interval_voltage_start_V"], waveform["interval_voltage_end_V"]), axis=1
            ).reshape(-1, len(waveform["body_ids"]))
            voltage_plot = TracePlot(
                interval_time,
                [(body, voltage[:, index]) for index, body in enumerate(waveform["body_ids"])],
                x_label="Time [s]",
                y_label="Circuit voltage [V]; slope jumps retained",
                limits=[sign * row["voltage_limit_V"] for row in body_rows for sign in (-1, 1)],
            )
            form.addWidget(voltage_plot)

            def select_body(*_):
                selected = self.trace_body.currentData()
                indices = range(len(waveform["body_ids"])) if selected is None else [selected]
                current_plot.curves = [(waveform["body_ids"][index], currents[:, index]) for index in indices]
                voltage_plot.curves = [(waveform["body_ids"][index], voltage[:, index]) for index in indices]
                selected_rows = [body_rows[index] for index in indices] if body_rows else []
                current_plot.limits = tuple(sign * row["current_limit_A"] for row in selected_rows for sign in (-1, 1))
                voltage_plot.limits = tuple(sign * row["voltage_limit_V"] for row in selected_rows for sign in (-1, 1))
                current_plot.update()
                voltage_plot.update()

            self.trace_body.currentIndexChanged.connect(select_body)
            thermal = engineering.get("thermal") or {}
            if thermal.get("available"):
                form.addWidget(
                    TracePlot(
                        time,
                        [("Shared lumped RC", thermal["sample_temperatures_K"])],
                        x_label="Time [s]",
                        y_label=f"Temperature [K]; exact interval peak {thermal['peak_temperature_K']:.6g} K",
                        limits=[thermal["temperature_limit_K"]],
                    )
                )
                form.addWidget(
                    QLabel(
                        "Temperature lines connect recorded waveform-knot values; interval extrema are evaluated analytically."
                    )
                )
            else:
                reason = QLabel(thermal.get("reason", "Thermal model unassessed"))
                reason.setWordWrap(True)
                form.addWidget(reason)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, "Current / voltage / temperature")
        for analysis in report.get("analyses", ()):
            page = QWidget()
            form = QVBoxLayout(page)
            state = (
                "PASS"
                if analysis["passed"] is True
                else "FAIL"
                if analysis["passed"] is False
                else "UNAVAILABLE"
                if not analysis["available"]
                else "UNASSESSED LIMIT"
            )
            text = QLabel(
                f"{analysis['name']}: {state}. Peak {analysis['peak']:.6g} {analysis['unit']}; "
                f"{analysis['solver']} {analysis['solver_version']}. "
                + analysis["scope"]
                + (" Reasons: " + "; ".join(analysis["reasons"]) if analysis["reasons"] else "")
            )
            text.setWordWrap(True)
            form.addWidget(text)
            response_plot = TracePlot(
                analysis["axis"],
                [("Sampled response magnitude", analysis["peak_by_axis"])],
                x_label=f"Sampling axis [{analysis['axis_unit']}]",
                y_label=f"{analysis['quantity']} [{analysis['unit']}]",
                limits=[analysis["limit"]] if analysis["limit"] is not None else (),
            )
            if analysis.get("witness_component_phase_deg") is not None:
                choice = QComboBox()
                choice.addItems(
                    [
                        "Peak magnitude across supplied samples",
                        "Component amplitudes at limiting location",
                        "Component phases at limiting location",
                    ]
                )
                form.addWidget(choice)

                def select_response(index, plot=response_plot, item=analysis):
                    plot.limits = ()
                    if index == 0:
                        plot.curves = [("Sampled response magnitude", np.asarray(item["peak_by_axis"]))]
                        plot.limits = (item["limit"],) if item["limit"] is not None else ()
                    else:
                        values = np.asarray(
                            item["witness_component_amplitudes"] if index == 1 else item["witness_component_phase_deg"],
                            dtype=float,
                        )
                        plot.curves = [
                            ("XYZ"[component] if values.shape[1] == 3 else item["quantity"], values[:, component])
                            for component in range(values.shape[1])
                        ]
                    plot.y_label = (
                        "Phase [deg]; zero-response phase omitted"
                        if index == 2
                        else f"{item['quantity']} [{item['unit']}]"
                    )
                    plot.update()

                choice.currentIndexChanged.connect(select_response)
            form.addWidget(response_plot)
            form.addWidget(
                QLabel(
                    f"Limiting sample: {analysis['limiting_axis_value']} {analysis['axis_unit']}, world {analysis['limiting_point_m']} m"
                )
            )
            provenance = QPlainTextEdit(
                json.dumps(
                    {
                        key: analysis[key]
                        for key in (
                            "source",
                            "source_sha256",
                            "solver",
                            "solver_version",
                            "complex_response",
                            "binding_verified",
                            "solver_accuracy_verified",
                        )
                    },
                    indent=2,
                )
            )
            provenance.setReadOnly(True)
            form.addWidget(provenance)
            self.tabs.addTab(page, analysis["name"])
        unassessed = report.get("unassessed_physics", ())
        label = QLabel(
            "Unassessed external physics: "
            + (", ".join(unassessed) if unassessed else "see each supplied evidence scope")
            + ". No construction qualification is implied."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        if not self.tabs.count():
            layout.insertWidget(
                0, QLabel("No waveform response or external analysis samples are available in this saved run.")
            )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
