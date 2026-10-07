"""Searchable engineering controls over one complete, uncommitted config draft."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import re

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import CoilGenConfig
from ..config_validation import validate_config
from ..research.convex import ResearchPhysicsSpec


@dataclass(frozen=True, slots=True)
class ExpertField:
    key: str
    label: str
    category: str
    help: str
    kind: str = "float"
    unit: str = ""
    display_scale: float = 1.0
    choices: tuple[str, ...] = ()
    optional: bool = False


FIELDS = (
    ExpertField(
        "routing_circuit_scope",
        "Connected circuit scope",
        "Routing",
        "Single routing scope connects across the support parts; per_part keeps connected circuits separate for each support. Independent-turn mode skips these interconnections. Feeds and drive terminals require separate definition.",
        "enum",
        choices=("single", "per_part"),
    ),
    ExpertField(
        "inter_group_connection_path_method",
        "Connection path",
        "Routing",
        "Direct 3-D paths can leave the support. Mesh geodesic paths follow its triangulation; route and clearance checks remain necessary.",
        "enum",
        choices=("direct_3d", "mesh_geodesic"),
    ),
    ExpertField(
        "inter_group_connection_candidate_count",
        "Connector candidates",
        "Routing",
        "Candidate count for inter-group connection search; more candidates increase computation.",
        "int",
    ),
    ExpertField(
        "inter_group_cut_alignment",
        "Axial / circumferential cut alignment",
        "Routing",
        "Auto uses the configured support topology. Explicit on/off changes cut selection; a cone does not become an analytic cylinder.",
        "enum",
        choices=("auto", "off", "on"),
    ),
    ExpertField(
        "joint_group_routing",
        "Joint group routing",
        "Routing",
        "Search joint group connections using configured geometry and field budgets.",
        "bool",
    ),
    ExpertField(
        "joint_group_routing_max_candidates",
        "Joint routing candidate budget",
        "Routing",
        "Total search budget, from 1 to 64. This controls search effort, not a convergence certificate.",
        "int",
    ),
    ExpertField(
        "min_inter_group_connector_clearance",
        "Inter-group connector clearance",
        "Routing",
        "Requested geometric separation between connectors; inspect actual route clearance after computation.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "min_turn_distance",
        "Minimum turn spacing",
        "Routing",
        "Requested winding separation; discretization and physical conductor dimensions also affect clearance.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "fail_on_route_quality_issues",
        "Reject route quality issues",
        "Routing",
        "Choose whether route-quality failures stop processing. Disabling this does not certify the route.",
        "bool",
    ),
    ExpertField(
        "conductor_cross_section_width",
        "Conductor width",
        "Conductor and support",
        "Physical swept-conductor width, distinct from the homogenized sheet thickness.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "conductor_cross_section_height",
        "Conductor height",
        "Conductor and support",
        "Physical conductor height used for realization and electrical estimates.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "conductor_mounting_mode",
        "Conductor placement",
        "Conductor and support",
        "Centered straddles the support. Surface outer/inner offsets the physical profile to the corresponding side, changing clearance and its field centroid.",
        "enum",
        choices=("centered", "surface_outer", "surface_inner"),
    ),
    ExpertField(
        "sweep_surface_realization",
        "Support realization",
        "Conductor and support",
        "Mesh follows the faceted support. Analytic realization is used only when the backend verifies a supported analytic surface; otherwise its existing fallback applies.",
        "enum",
        choices=("analytic", "mesh"),
    ),
    ExpertField(
        "sweep_insulation_thickness",
        "Insulation thickness",
        "Conductor and support",
        "Insulation enlarges the clearance envelope; it is not conducting cross-section area.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "sweep_manufacturing_tolerance",
        "Manufacturing tolerance",
        "Conductor and support",
        "Geometric manufacturing allowance applied by the existing sweep checks.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "normal_shift_length",
        "Normal shift distance",
        "Conductor and support",
        "Configured path shift along support normals, separate from profile mounting and insulation.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "skip_normal_shift",
        "Keep the unshifted winding",
        "Conductor and support",
        "Enabled preserves the path without applying the configured normal shift. Disable to apply the normal shift distance.",
        "bool",
    ),
    ExpertField(
        "sweep_bend_mode",
        "Corner realization",
        "Bends and sampling",
        "Miter and fillet retain the backend's existing corner construction. A requested radius can be infeasible on the support.",
        "enum",
        choices=("miter", "fillet"),
    ),
    ExpertField(
        "sweep_min_bend_radius",
        "Minimum bend radius",
        "Bends and sampling",
        "Requested physical bend radius. Zero retains the existing disabled-radius policy; inspect realized bends.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "sweep_bend_clearance_margin",
        "Bend clearance margin",
        "Bends and sampling",
        "Additional clearance requested around realized bends.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "sweep_bend_arc_angle_deg",
        "Bend arc angular step",
        "Bends and sampling",
        "Angular sampling of bend arcs; a smaller step increases geometric resolution and computation.",
        unit="°",
    ),
    ExpertField(
        "sweep_max_segment_length",
        "Maximum sweep segment length",
        "Bends and sampling",
        "Zero retains the existing disabled length limit. Positive values refine physical path sampling.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "sweep_max_chord_error",
        "Maximum chord error",
        "Bends and sampling",
        "Requested path approximation tolerance; this is not a B-field error budget.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "sweep_max_frame_rotation_deg",
        "Maximum frame rotation",
        "Bends and sampling",
        "Limits frame rotation per realized segment; maximum 180 degrees.",
        unit="°",
    ),
    ExpertField(
        "sweep_frame_method",
        "Conductor frame method",
        "Bends and sampling",
        "Surface-constrained transport guides the physical profile using support normals. Surface-normal framing follows the existing alternative policy.",
        "enum",
        choices=("surface_constrained_parallel_transport", "surface_normal"),
    ),
    ExpertField(
        "sweep_self_intersection_check",
        "Sweep self-intersection check",
        "Bends and sampling",
        "Capsule-fast checks the configured geometric envelope. Off leaves this check unverified.",
        "enum",
        choices=("capsule_fast", "off"),
    ),
    ExpertField(
        "surface_current_density_limit_A_per_m",
        "Maximum sheet current density",
        "Physical constraints",
        "Continuous-sheet current K [A/m], distinct from contour wire current [A]. Zero disables this configured bound.",
        unit="A/m",
    ),
    ExpertField(
        "research_physics.objective",
        "Physical objective",
        "Physical constraints",
        "An active research_physics specification invokes the physical convex solver. Mixture weights/scales remain available in JSON and are preserved here.",
        "enum",
        choices=("power", "magnetic_energy", "minimax", "mixture"),
    ),
    ExpertField(
        "research_physics.sheet_power_limit_W",
        "Sheet power bound",
        "Physical constraints",
        "Optional bound on homogenized continuous-sheet Joule power, before contouring and physical realization.",
        unit="W",
        optional=True,
    ),
    ExpertField(
        "research_physics.magnetic_energy_limit_J",
        "Magnetic energy bound",
        "Physical constraints",
        "Optional bound on the research model's magnetic energy. Energy quadrature remains unchanged.",
        unit="mJ",
        display_scale=1e3,
        optional=True,
    ),
    ExpertField(
        "research_physics.force_limit_N",
        "Net force bound",
        "Physical constraints",
        "Optional mechanical bound in the configured B₀/scenario model. Background fields and equalities remain preserved in JSON.",
        unit="N",
        optional=True,
    ),
    ExpertField(
        "research_physics.torque_limit_Nm",
        "Net torque bound",
        "Physical constraints",
        "Optional bound about the configured torque origin in the existing mechanical model.",
        unit="N·m",
        optional=True,
    ),
)


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _canonical(config):
    return json.loads(json.dumps(asdict(config), default=_json_value))


def _read(values, key, defaults):
    if "." not in key:
        return values[key]
    parent, child = key.split(".", 1)
    mapping = values.get(parent)
    return mapping.get(child, defaults[parent][child]) if isinstance(mapping, dict) else defaults[parent][child]


class _FieldEditor(QWidget):
    changed = Signal(str)
    reset_requested = Signal(str)

    def __init__(self, spec, default, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("expert_row_" + spec.key)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        label = QLabel(f"{spec.label} [{spec.unit}]" if spec.unit else spec.label)
        label.setWordWrap(True)
        default_text = "disabled" if default is None else repr(default)
        self.setToolTip(
            f"{spec.key}\n{spec.help}\nNative stored default: {default_text}. Reset edits this draft field only."
        )
        layout.addWidget(label)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.enabled_bound = QCheckBox("Use bound") if spec.optional else None
        if self.enabled_bound:
            row.addWidget(self.enabled_bound)
            self.enabled_bound.toggled.connect(self._changed)
        if spec.kind == "bool":
            self.control = QCheckBox("Enabled")
            signal = self.control.toggled
        elif spec.kind == "enum":
            self.control = QComboBox()
            self.control.addItems(spec.choices)
            self.control.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.control.setMinimumContentsLength(12)
            signal = self.control.currentTextChanged
        else:
            self.control = QLineEdit()
            self.control.setPlaceholderText(f"Value in {spec.unit}" if spec.unit else "Numeric value")
            signal = self.control.textChanged
        self.control.setObjectName(spec.key)
        label.setBuddy(self.control)
        self.control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self.control, 1)
        self.reset_button = QPushButton("Default")
        self.reset_button.setToolTip(self.toolTip())
        self.reset_button.clicked.connect(lambda: self.reset_requested.emit(spec.key))
        row.addWidget(self.reset_button)
        explanation = QLabel(spec.help)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.error = QLabel()
        self.error.setStyleSheet("color: #b02020")
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)
        signal.connect(self._changed)

    def _changed(self, *_args):
        if self.enabled_bound:
            self.control.setEnabled(self.enabled_bound.isChecked())
        self.changed.emit(self.spec.key)

    def set_value(self, value):
        with QSignalBlocker(self.control):
            if self.spec.kind == "bool":
                valid = isinstance(value, bool)
                self.control.setChecked(value if valid else False)
                self.control.setText(
                    "Enabled" if valid else f"Unsupported value retained: {value!r}; use JSON or Default"
                )
                self.control.setEnabled(valid)
            elif self.spec.kind == "enum":
                text = str(value)
                if self.control.findText(text) < 0:
                    self.control.addItem(text)
                self.control.setCurrentText(text)
            else:
                text = (
                    ""
                    if value is None
                    else format(value * self.spec.display_scale, ".12g")
                    if isinstance(value, (float, int))
                    else str(value)
                )
                self.control.setText(text)
        if self.enabled_bound:
            with QSignalBlocker(self.enabled_bound):
                self.enabled_bound.setChecked(value is not None)
            self.control.setEnabled(value is not None)
        self.set_error("")

    def read_value(self):
        if self.enabled_bound and not self.enabled_bound.isChecked():
            return None
        if self.spec.kind == "bool":
            return self.control.isChecked()
        if self.spec.kind == "enum":
            return self.control.currentText()
        text = self.control.text().strip()
        if self.spec.kind == "int":
            if not re.fullmatch(r"[+-]?\d+", text):
                raise ValueError("Enter an integer.")
            return int(text)
        try:
            value = float(text) / self.spec.display_scale
        except ValueError as exc:
            raise ValueError(f"Enter a numeric value{' in ' + self.spec.unit if self.spec.unit else ''}.") from exc
        if not math.isfinite(value):
            raise ValueError("Enter a finite numeric value.")
        return value

    def set_error(self, text):
        self.error.setText(text)
        self.error.setVisible(bool(text))


class ExpertSettingsDialog(QDialog):
    """Commit validated changes through the caller's existing undoable edit path.

    ``apply_callback(changes, label)`` must return True only after committing.
    Typed controls update selected keys; all other native/nested fields survive.
    JSON accepts native-unit overrides and provides access to every config field.
    """

    draft_changed = Signal(bool)

    def __init__(self, config, apply_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Expert CoilGen settings — engineering controls and native configuration JSON")
        self.resize(900, 780)
        self.apply_callback = apply_callback
        self._initial = _canonical(config)
        self._draft = deepcopy(self._initial)
        self._defaults = _canonical(CoilGenConfig())
        self._defaults["research_physics"] = ResearchPhysicsSpec().to_dict()
        self._typed_errors = {}
        self._json_dirty = False
        self._active_tab = 0
        self.fields = {}
        self.groups = {}
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Changes remain a draft until Apply. Display units are shown beside each parameter. JSON uses native parameter units, for example metres for conductor lengths and degrees for parameters ending in _deg. Target values follow target_field_units. Default resets only the selected field. Other constraints and advanced settings remain preserved."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setObjectName("expert_search")
        self.search.setPlaceholderText("Search label, native key, category or engineering description…")
        search_row.addWidget(self.search, 1)
        find = QPushButton("Find in JSON")
        find.clicked.connect(self.find_in_json)
        search_row.addWidget(find)
        layout.addLayout(search_row)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("expert_tabs")
        layout.addWidget(self.tabs, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        self.physics_note = QLabel()
        self.physics_note.setWordWrap(True)
        body_layout.addWidget(self.physics_note)
        self.configure_physics = QPushButton("Configure physical convex solver (power objective)")
        self.configure_physics.setObjectName("configure_research_physics")
        self.configure_physics.clicked.connect(self._enable_physics)
        body_layout.addWidget(self.configure_physics)
        for spec in FIELDS:
            if spec.category not in self.groups:
                group = QGroupBox(spec.category)
                group.setLayout(QVBoxLayout())
                self.groups[spec.category] = group
                body_layout.addWidget(group)
            editor = _FieldEditor(spec, _read(self._defaults, spec.key, self._defaults))
            editor.changed.connect(self._typed_changed)
            editor.reset_requested.connect(self.reset_field)
            self.fields[spec.key] = editor
            self.groups[spec.category].layout().addWidget(editor)
        self.no_match = QLabel("No typed parameter matches. Find in JSON to inspect all native settings.")
        self.no_match.setWordWrap(True)
        body_layout.addWidget(self.no_match)
        body_layout.addStretch()
        scroll.setWidget(body)
        self.tabs.addTab(scroll, "Engineering controls")
        self.json_editor = QPlainTextEdit()
        self.json_editor.setObjectName("native_settings_json")
        self.tabs.addTab(self.json_editor, "All native settings (JSON)")
        self.error_label = QLabel()
        self.error_label.setObjectName("expert_settings_error")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b02020")
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(buttons)
        self.search.textChanged.connect(self.filter_fields)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.json_editor.textChanged.connect(self._json_changed)
        self._update_fields()
        self._write_json()

    @property
    def pending_changes(self):
        """Include incomplete text that has not yet produced a parsed value."""
        return bool(self._json_dirty or self._typed_errors or self._draft != self._initial)

    def _notify_draft(self):
        self.draft_changed.emit(self.pending_changes)

    @property
    def draft(self):
        """Detached last parsed draft; invalid text is retained in its editor."""
        return deepcopy(self._draft)

    def filter_fields(self, text=None):
        terms = (self.search.text() if text is None else text).casefold().split()
        count = 0
        categories = set()
        for key, editor in self.fields.items():
            spec = editor.spec
            haystack = " ".join((key, spec.label, spec.category, spec.help)).casefold()
            visible = all(term in haystack for term in terms)
            editor.setVisible(visible)
            if visible:
                count += 1
                categories.add(spec.category)
        for category, group in self.groups.items():
            group.setVisible(category in categories)
        self.no_match.setVisible(not count)

    def _update_fields(self):
        active = isinstance(self._draft.get("research_physics"), dict)
        for key, editor in self.fields.items():
            editor.set_value(_read(self._draft, key, self._defaults))
            editor.setEnabled(active or not key.startswith("research_physics."))
        self.physics_note.setText(
            "Physical convex solver active. Typed bounds edit individual keys and preserve B₀, torque origin, equalities, scenarios, field constraints and acceptance policy. Disabling/replacing the complete specification requires an explicit JSON edit."
            if active
            else "Native solver active. Research bounds are inactive. Configuring the physical solver below is an explicit solver change to the power objective; the remaining native configuration stays preserved."
        )
        self.configure_physics.setVisible(not active)
        self.filter_fields()

    def _typed_changed(self, key):
        editor = self.fields[key]
        try:
            value = editor.read_value()
            self._write_field(key, value)
        except (ValueError, TypeError) as exc:
            self._typed_errors[key] = str(exc)
            editor.set_error(str(exc))
        else:
            self._typed_errors.pop(key, None)
            editor.set_error("")
            self._write_json()
        self.error_label.setText("\n".join(f"{name}: {message}" for name, message in self._typed_errors.items()))
        self._notify_draft()

    def _write_field(self, key, value):
        if "." in key:
            parent, child = key.split(".", 1)
            if not isinstance(self._draft.get(parent), dict):
                raise ValueError("Configure the physical solver explicitly before setting research bounds.")
            self._draft[parent][child] = value
        else:
            self._draft[key] = value

    def reset_field(self, key):
        """Explicitly reset exactly one visible/native field in the draft."""
        value = deepcopy(_read(self._defaults, key, self._defaults))
        try:
            self._write_field(key, value)
        except ValueError as exc:
            self.error_label.setText(str(exc))
            self._notify_draft()
            return False
        self._typed_errors.pop(key, None)
        self.fields[key].set_value(value)
        self._write_json()
        self.error_label.clear()
        self._notify_draft()
        return True

    def _enable_physics(self):
        if self._draft.get("research_physics") is None:
            self._draft["research_physics"] = {"objective": "power"}
            self._update_fields()
            self._write_json()
            self._notify_draft()

    def _json_changed(self):
        self._json_dirty = True
        self._notify_draft()

    def _write_json(self):
        with QSignalBlocker(self.json_editor):
            self.json_editor.setPlainText(json.dumps(self._draft, indent=2))
        self._json_dirty = False

    def _merge_json(self):
        if not self._json_dirty:
            return True
        try:
            values = json.loads(self.json_editor.toPlainText())
            if not isinstance(values, dict):
                raise ValueError("Native configuration must be a JSON object.")
            self._draft = {**self._draft, **values}
        except (ValueError, TypeError) as exc:
            self.error_label.setText(f"JSON draft: {exc}")
            self._notify_draft()
            return False
        self._json_dirty = False
        self._typed_errors.clear()
        self._update_fields()
        self.error_label.clear()
        self._notify_draft()
        return True

    def _tab_changed(self, index):
        valid = self._merge_json() if self._active_tab == 1 else not self._typed_errors
        if not valid:
            if self._typed_errors:
                self.error_label.setText(
                    "Correct invalid typed values before switching to JSON; the draft is retained."
                )
            with QSignalBlocker(self.tabs):
                self.tabs.setCurrentIndex(self._active_tab)
            return
        if index == 1:
            self._write_json()
        self._active_tab = index

    def find_in_json(self):
        self.tabs.setCurrentIndex(1)
        if self.tabs.currentIndex() != 1:
            return
        if not self.json_editor.find(self.search.text()):
            cursor = self.json_editor.textCursor()
            cursor.setPosition(0)
            self.json_editor.setTextCursor(cursor)
            self.json_editor.find(self.search.text())

    def apply_changes(self):
        if not self._merge_json() or self._typed_errors:
            if self._typed_errors:
                self.error_label.setText("Correct the highlighted typed values before applying. The draft is retained.")
            self._notify_draft()
            return False
        try:
            candidate = CoilGenConfig(**deepcopy(self._draft))
            report = validate_config(candidate)
            for editor in self.fields.values():
                editor.set_error("")
            for message in report.errors:
                for key, editor in self.fields.items():
                    if message.field == key or message.field == key.split(".", 1)[0]:
                        editor.set_error(message.message)
            report.raise_for_errors()
            changes = {key: deepcopy(value) for key, value in self._draft.items() if value != self._initial.get(key)}
            if changes and not self.apply_callback(changes, "Edit expert settings"):
                self.error_label.setText(
                    "The workbench rejected this configuration. The draft is retained; inspect its reported validation message."
                )
                self._notify_draft()
                return False
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, OSError, RuntimeError) as exc:
            self.error_label.setText(str(exc))
            self._notify_draft()
            return False
        self._initial = deepcopy(self._draft)
        self._write_json()
        self.error_label.setText(report.format(levels=("warning",)))
        self._notify_draft()
        return True

    def _accept(self):
        if self.apply_changes():
            self.accept()
