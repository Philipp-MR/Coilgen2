"""Physical design intent and configured requirements, without solving or editing.

The brief describes the editable model. It deliberately carries no verdicts:
computed evidence belongs to an immutable result and its evaluated scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..application.acceptance import field_acceptance_slack_T, independent_field_acceptance_slack_T
from ..config import CoilGenConfig
from ..engineering_limits import engineering_limit_diagnostics
from .colors import apply_tone
from .guidance import resolved_model_summary
from .icons import REGION_ICONS, icon
from .editors import SUPPORT_EDITORS
from .winding_intent import winding_intent


@dataclass(frozen=True, slots=True)
class BriefItem:
    key: str
    label: str
    value: str
    kind: str
    scope: str
    section: str
    tone: str = "neutral"
    icon_name: str = ""


@dataclass(frozen=True, slots=True)
class DesignBrief:
    name: str
    revision: int
    compact: str
    items: tuple[BriefItem, ...]
    pending: bool = False


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Physical settings must be finite.")
    return f"{value:g}"


def _vector(values, unit, factor=1.0):
    if len(values) != 3:
        raise ValueError("A world vector needs three components.")
    return f"({', '.join(_number(float(value) * factor) for value in values)}) {unit}"


def _circuit(config):
    if config.contour_only_sweep_mode:
        return "Independent closed turns; each turn requires its own drive/feed design."
    if config.skip_postprocessing:
        return "Unconnected contours; no connected circuit is requested."
    if config.routing_circuit_scope == "per_part":
        return "Separate connected circuit per support part; external inter-part feeds are separate."
    return "One connected series winding across support parts; terminal/feed realization is separate."


def _objective(config, physics):
    if physics is not None:
        objective = physics.get("objective", "power")
        return {
            "power": "Minimize continuous-sheet Joule loss [W]",
            "minimax": "Minimize peak surface current density [A/m]",
            "magnetic_energy": "Minimize continuous-sheet stored magnetic energy [J]",
            "mixture": "Minimize scaled mixture of configured physical objectives",
        }.get(objective, f"Physical objective: {objective}; inspect Advanced settings")
    return (
        f"Native {config.sf_opt_method} formulation; cost {config.stream_function_cost_function}; "
        f"quadratic mode {config.constrained_quadratic_objective_mode}"
    )


def _build_brief(project, pending):
    config = CoilGenConfig(**project.values)
    summaries = resolved_model_summary(project)
    physics = config.research_physics
    if physics is not None and not isinstance(physics, Mapping):
        raise ValueError("Physical objective and bounds must be a settings object.")
    study = project.study
    rows = []

    def add(key, label, value, kind, scope, section, tone="neutral", *, icon_name=""):
        rows.append(BriefItem(key, label, value, kind, scope, section, tone, icon_name))

    target_icon = (
        "target-map"
        if str(config.target_field_definition_file).lower() != "none"
        else "target-harmonic"
        if summaries["target"].startswith("Bz: real Y(")
        else "target-uniform"
        if summaries["target"].startswith("Uniform Bz")
        else "target-gradient"
        if summaries["target"].startswith(("Gx =", "Gy =", "Gz ="))
        else "field-component"
    )
    add(
        "target",
        "Target field",
        summaries["target"],
        "Intent",
        "Requested physical target",
        "target",
        icon_name=target_icon,
    )
    if config.target_field_definition_file != "none":
        region_shape = "imported_map"
    else:
        geometry = config.target_region_geometry if isinstance(config.target_region_geometry, dict) else {}
        region_shape = geometry.get("shape", "sphere")
        if config.target_region_geometry is None and config.target_mesh_file != "none":
            region_shape = "enclosing_mesh"
    add(
        "roi",
        "Field region / samples",
        summaries["roi"],
        "Requirement",
        "Field fitting region",
        "target",
        icon_name=REGION_ICONS[region_shape],
    )
    background = f"MRI reference B0 = {_number(config.b_0_strength_T)} T; configured world direction {_vector(config.b_0_direction, '')}"
    if physics is not None:
        background += f"; physical sheet-model background {_vector(physics.get('b0_T', (0, 0, 3)), 'T')}"
        if physics.get("background_B_per_face_T") is not None:
            background += "; spatial per-face background overrides the uniform mechanical-field assumption"
    add(
        "frame",
        "Frame and B0",
        f"World x, y, z in metres; target input units {config.target_field_units}. {background}.",
        "Assumption",
        "Imported asset coordinates remain authoritative; support rotations do not rotate the target",
        "target",
    )
    support_editor = next((entry for entry in SUPPORT_EDITORS if entry.source == config.coil_mesh_file), None)
    support_icon = (
        support_editor.icon_name
        if support_editor
        else "mesh"
        if str(config.coil_mesh_file).startswith("create ")
        else "support-import"
    )
    add(
        "support",
        "Mechanical support",
        summaries["support"],
        "Requirement",
        "Nominal support surface",
        "support",
        icon_name=support_icon,
    )
    add(
        "conductor",
        "Physical conductor",
        summaries["conductor"],
        "Assumption",
        "Wire profile is distinct from homogenized sheet thickness",
        "conductor",
    )
    add(
        "circuit",
        "Circuit intention",
        _circuit(config),
        "Intent",
        "Topology requested for contour realization",
        "winding",
        icon_name={
            "Connected circuit": "connected-winding",
            "Separate circuits per support": "separate-circuits",
            "Independent closed turns": "independent-turns",
            "Contours only": "contours",
        }[winding_intent(config)],
    )
    terminal = str(config.layout_terminal_strategy)
    add(
        "terminals",
        "Terminals / feed scope",
        f"Declared terminal strategy: {terminal}; pads {config.layout_terminal_pad_count}, exits {config.layout_terminal_exit_count}, keepouts {config.layout_keepout_region_count}.",
        "Scope",
        "Declarations do not establish a complete driven assembly or external return-field evaluation",
        "winding",
        "warning" if terminal == "undefined" else "neutral",
    )
    objective = _objective(config, physics)
    add(
        "objective",
        "Optimization preference",
        objective,
        "Preference",
        "Continuous current sheet; winding results are evaluated separately",
        "solve",
        icon_name={"power": "power-loss", "minimax": "current-density", "magnetic_energy": "inductance"}.get(
            physics.get("objective", "power") if physics is not None else None, "stream-function"
        ),
    )
    default_field = physics is None or physics.get("include_default_field_constraints", True)
    relative = _number(config.target_error_tolerance_percent)
    independent = _number(config.target_error_tolerance_percent / study["design_margin"])
    field = (
        f"Design relative budget {relative}%; metric {config.target_error_constraint_metric}. "
        f"Independent-sheet relative budget {independent}% (design / {_number(study['design_margin'])})."
        if default_field
        else f"Default target-field constraints are disabled in the physical specification; configured {relative}% budget is inactive in that default check."
    )
    extra = []
    for key, label in (
        ("target_error_tolerance_absolute_T", "absolute field"),
        ("target_error_std_limit_T", "residual standard deviation"),
        ("target_error_rms_limit_T", "residual RMS"),
    ):
        value = getattr(config, key)
        if value > 0:
            extra.append(f"{label} ≤ {_number(value * 1e6)} µT")
    if extra:
        field += " Configured absolute budgets: " + "; ".join(extra) + "."
    add(
        "field_budget",
        "Field requirements",
        field,
        "Requirement" if default_field else "Scope",
        "Configured budgets and validation sampling remain separate from realized-wire acceptance",
        "solve",
        "neutral" if default_field else "warning",
    )
    add(
        "acceptance",
        "Numerical acceptance policy",
        f"Solver field allowance {_number(field_acceptance_slack_T(config) * 1e9)} nT; independent-sheet allowance {_number(independent_field_acceptance_slack_T(config) * 1e9)} nT.",
        "Assumption",
        "Existing solver tolerances; these allowances do not replace field budgets",
        "solve",
    )
    hardware = engineering_limit_diagnostics(config)
    bounds = []
    if hardware["max_wire_current_A"] > 0:
        bounds.append(f"configured wire current ≤ {_number(hardware['max_wire_current_A'])} A")
    if hardware["min_wire_distance_m"] > 0:
        bounds.append(f"configured spacing ≥ {_number(hardware['min_wire_distance_m'] * 1e3)} mm")
    if hardware["engineering_surface_current_density_limit_A_per_m"] > 0:
        bounds.append(
            f"effective sheet K ≤ {_number(hardware['engineering_surface_current_density_limit_A_per_m'])} A/m ({hardware['engineering_surface_current_density_limit_source']})"
        )
    add(
        "hardware",
        "Current / spacing limits",
        "; ".join(bounds) + "." if bounds else "No current, spacing or surface-current limit is configured.",
        "Requirement" if bounds else "Scope",
        f"Contour mode: {hardware['contour_discretization_method']}; hardware infeasibility policy: {config.gradient_hardware_infeasible_policy}. A sheet K bound is not a conductor current-density or amplifier qualification.",
        "solve",
        "neutral" if bounds else "warning",
    )
    if physics is not None:
        physical = []
        for key, label, unit in (
            ("sheet_power_limit_W", "sheet Joule loss", "W"),
            ("magnetic_energy_limit_J", "sheet magnetic energy", "J"),
            ("force_limit_N", "net sheet force", "N"),
            ("torque_limit_Nm", "net sheet torque", "N·m"),
        ):
            value = physics.get(key)
            if value is not None:
                physical.append(f"{label} ≤ {_number(value)} {unit}")
        for key, label, unit in (
            ("force_equality_N", "force equality", "N"),
            ("torque_equality_Nm", "torque equality", "N·m"),
        ):
            if physics.get(key) is not None:
                physical.append(f"{label} {_vector(physics[key], unit)}")
        counts = (len(physics.get("field_constraints") or ()), len(physics.get("scenarios") or ()))
        if any(counts):
            physical.append(f"{counts[0]} additional field constraints; {counts[1]} finite field scenarios")
        add(
            "physical_bounds",
            "Other physical bounds",
            "; ".join(physical) + "."
            if physical
            else "No additional sheet power, energy, net force, torque or field-scenario bound is configured.",
            "Requirement" if physical else "Scope",
            "Continuous sheet in the configured background field; finite scenarios certify their sampled cases",
            "solve",
        )
    fabrication = (
        f"Mounting: {config.conductor_mounting_mode}; minimum bend radius {_number(config.sweep_min_bend_radius * 1e3)} mm; "
        f"manufacturing tolerance {_number(config.sweep_manufacturing_tolerance * 1e3)} mm. "
        f"Conductor solid generation {'requested' if not config.skip_sweep else 'disabled'}; strict fabrication review {'requested' if study['fabricate'] else 'not requested'}."
    )
    add(
        "manufacturing",
        "Manufacturing conditions",
        fabrication,
        "Requirement",
        "Requested geometric checks; completed evidence is attached to each saved run",
        "check",
    )
    wire = []
    for key, label in (
        ("layout_target_error_std_limit_T", "residual standard deviation"),
        ("layout_target_error_rms_limit_T", "residual RMS"),
    ):
        if getattr(config, key) > 0:
            wire.append(f"{label} ≤ {_number(getattr(config, key) * 1e6)} µT")
    add(
        "wire_budget",
        "Realized winding budgets",
        "; ".join(wire) + "."
        if wire
        else "No separate absolute winding RMS or standard-deviation limit is configured; inspect the saved winding field checks.",
        "Requirement" if wire else "Scope",
        f"Wire-current evaluation policy: {config.field_evaluation_current_source}; winding sampling and current must be read from saved evidence",
        "check",
    )
    add(
        "qualification",
        "Remaining engineering evidence",
        "Complete drive/feed geometry and external returns, amplifier voltage, cooling, temperature and duty cycle require separate qualification.",
        "Scope",
        "Geometric CAD export and sheet constraints do not establish construction release",
        "check",
        "warning",
    )
    compact = " · ".join(summaries[key].split(";")[0].rstrip(".") for key in ("target", "roi", "support"))
    compact += "\n" + summaries["conductor"].split(";")[0] + " · Circuit: " + winding_intent(config)
    return DesignBrief(project.name, project.revision, compact, tuple(rows), pending)


def design_brief(project, *, pending=False):
    """Read canonical project values only; never resolve assets or infer a verdict.

    Malformed scientific input must leave navigation available for correction.
    A pending badge refers to unapplied editor changes, not computed evidence.
    """
    try:
        return _build_brief(project, pending)
    except (ValueError, TypeError, IndexError, KeyError, OverflowError, ZeroDivisionError) as exc:
        item = BriefItem(
            "invalid",
            "Inspect physical settings",
            str(exc),
            "Scope",
            "Correct the model before computing",
            "solve",
            "warning",
        )
        return DesignBrief(
            project.name,
            project.revision,
            "Physical brief unavailable until invalid settings are corrected.",
            (item,),
            pending,
        )


def _brief_icon(item):
    if item.icon_name:
        return item.icon_name
    return {
        "target": "target-gradient",
        "roi": "roi",
        "frame": "field-component",
        "support": "support-cylinder",
        "conductor": "conductor-profile",
        "circuit": "connected-winding",
        "terminals": "terminals",
        "objective": "regularisation",
        "field_budget": "field-error",
        "acceptance": "constraint",
        "hardware": "current-density",
        "physical_bounds": "constraint",
        "manufacturing": "bend-radius",
        "wire_budget": "field-error",
        "qualification": "info",
    }.get(item.key, "settings")


class DesignBriefPanel(QWidget):
    """Persistent brief with optional detail and navigation to existing editors."""

    section_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("physical_design_brief")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMaximumHeight(120)
        self._brief = None
        self._restyling = False
        self.section_buttons = {}
        self.value_labels = {}
        self.kind_labels = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        header = QHBoxLayout()
        self.expand_button = QToolButton(self)
        self.expand_button.setObjectName("design_brief_expand")
        self.expand_button.setText("Physical design brief")
        self.expand_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.expand_button.setCheckable(True)
        self.expand_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.expand_button.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_button.toggled.connect(self._set_expanded)
        self.expand_button.setToolTip(
            "Inspect physical intent, configured requirements and model assumptions; click a row to edit it."
        )
        header.addWidget(self.expand_button)
        header.addStretch(1)
        self.revision_label = self._label("design_brief_revision", self)
        self.revision_label.setWordWrap(False)
        header.addWidget(self.revision_label)
        layout.addLayout(header)
        self.compact_label = self._label("design_brief_compact", self)
        layout.addWidget(self.compact_label)
        self.details = QScrollArea(self)
        self.details.setObjectName("design_brief_details")
        self.details.setFrameShape(QFrame.Shape.NoFrame)
        self.details.setWidgetResizable(True)
        self.details.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.details.setMaximumHeight(220)
        self.details.setMinimumHeight(50)
        self._rows = QWidget(self.details)
        self._grid = QGridLayout(self._rows)
        self._grid.setContentsMargins(3, 3, 8, 6)
        self._grid.setHorizontalSpacing(9)
        self._grid.setVerticalSpacing(9)
        self._grid.setColumnStretch(1, 1)
        self.details.setWidget(self._rows)
        self.details.hide()
        layout.addWidget(self.details)

    @staticmethod
    def _label(name, parent):
        label = QLabel(parent)
        label.setObjectName(name)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return label

    def set_project(self, project, *, pending=False):
        self.set_brief(design_brief(project, pending=pending))

    def set_brief(self, brief):
        if brief == self._brief:
            return
        self._brief = brief
        self.compact_label.setText(brief.compact)
        self.compact_label.setToolTip(
            f"{brief.name}: editable physical intent. Saved results retain their own immutable settings."
        )
        self.revision_label.setText(f"r{brief.revision} · edits pending" if brief.pending else f"r{brief.revision}")
        self.revision_label.setToolTip(
            "Unapplied editor values are not represented in this brief."
            if brief.pending
            else "Brief describes the current applied model revision."
        )
        for mapping in (self.section_buttons, self.value_labels, self.kind_labels):
            mapping.clear()
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for row, item in enumerate(brief.items):
            button = QToolButton(self._rows)
            button.setObjectName(f"brief_section_{item.key}")
            button.setText(item.label)
            button.setIcon(icon(_brief_icon(item)))
            button.setIconSize(QSize(18, 18))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setAccessibleName(f"Open {item.label}")
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            button.clicked.connect(lambda _checked=False, section=item.section: self.section_requested.emit(section))
            button.setToolTip(item.scope)
            button.setAccessibleDescription(item.scope)
            self._grid.addWidget(button, row, 0, Qt.AlignmentFlag.AlignTop)
            container = QWidget(self._rows)
            content = QVBoxLayout(container)
            content.setContentsMargins(0, 0, 0, 0)
            content.setSpacing(1)
            value = self._label(f"brief_value_{item.key}", container)
            value.setText(item.value)
            value.setToolTip(item.scope)
            kind = self._label(f"brief_kind_{item.key}", container)
            kind.setText(f"{item.kind} · {item.scope}")
            kind.setToolTip(item.scope)
            content.addWidget(value)
            content.addWidget(kind)
            self._grid.addWidget(container, row, 1)
            self.section_buttons[item.key] = button
            self.value_labels[item.key] = value
            self.kind_labels[item.key] = kind
        self._apply_colors()
        self.layout().invalidate()
        self.updateGeometry()

    def _set_expanded(self, expanded):
        self.details.setVisible(expanded)
        self.setMaximumHeight(300 if expanded else 120)
        self.expand_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.expand_button.setAccessibleName("Hide physical design brief" if expanded else "Show physical design brief")
        self.layout().invalidate()
        self.updateGeometry()

    def _preferred_height(self, width, *, minimum=False):
        """Hidden scroll contents must not retain the expanded size hint.

        Qt layouts can cache a previous word-wrapped label or scroll-area hint
        while dock minima change. Calculate the visible compact content at the
        actual width and retain explicit height budgets for both states.
        """
        margins = self.layout().contentsMargins()
        available = max(1, width - margins.left() - margins.right())
        summary_height = max(self.compact_label.heightForWidth(available), self.compact_label.fontMetrics().height())
        header_height = max(self.expand_button.sizeHint().height(), self.revision_label.sizeHint().height())
        height = margins.top() + margins.bottom() + header_height + self.layout().spacing() + summary_height
        if self.expand_button.isChecked():
            height += self.layout().spacing() + (
                self.details.minimumHeight() if minimum else self.details.maximumHeight()
            )
        return min(height, self.maximumHeight())

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(hint.width(), self._preferred_height(self.width()))

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QSize(hint.width(), self._preferred_height(self.width(), minimum=True))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.updateGeometry()

    def _apply_colors(self):
        if self._restyling or self._brief is None:
            return
        self._restyling = True
        try:
            apply_tone(self.revision_label, "warning" if self._brief.pending else "info")
            for item in self._brief.items:
                apply_tone(self.value_labels[item.key], item.tone)
                apply_tone(self.kind_labels[item.key], "neutral")
        finally:
            self._restyling = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange}:
            self._apply_colors()
