"""Searchable engineering controls over one complete, uncommitted config draft."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields as dataclass_fields, replace
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

from ..config import CoilGenConfig, OptimizationSettings
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
    json_shape: str = ""


FIELDS = (
    ExpertField(
        "contour_path_regularization_mode",
        "Unconnected contour processing",
        "Contour smoothing",
        "Optional contour processing before cutting/interconnection. Smoothing changes the physical paths; resampling only changes sampling and is deferred when routing requires intrinsic face cuts.",
        "enum",
        choices=("off", "parameterized_bspline", "resample"),
    ),
    ExpertField(
        "contour_smoothing_scale_mode",
        "Smoothing scale",
        "Contour smoothing",
        "Mesh-coupled mode controls amplitude at the local mesh wavelength. Legacy mode uses the saved dimensionless fit strength.",
        "enum",
        choices=("mesh_coupled", "legacy"),
    ),
    ExpertField(
        "contour_smoothing_mesh_gain",
        "Mesh-scale amplitude retained",
        "Contour smoothing",
        "Retained amplitude at the local mesh wavelength, strictly between zero and one. Smaller values smooth more strongly; topology and displacement guards may reduce smoothing.",
    ),
    ExpertField(
        "contour_smoothing_factor",
        "Legacy smoothing strength",
        "Contour smoothing",
        "Legacy dimensionless fit strength from zero to one; zero disables legacy smoothing. Does not control mesh-coupled mode.",
    ),
    ExpertField(
        "contour_smoothing_max_displacement",
        "Maximum contour displacement",
        "Contour smoothing",
        "Maximum permitted path displacement from the original contour. Zero disables this optional cap.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "contour_resample_target_segment_length",
        "Target contour segment length",
        "Contour smoothing",
        "Physical output sampling interval. Zero uses automatic sampling for smoothing and disables resampling-only mode. Smaller intervals increase geometry size and computation.",
        unit="mm",
        display_scale=1e3,
    ),
    ExpertField(
        "contour_smoothing_preserve_min_turn_distance",
        "Preserve contour centerline spacing",
        "Contour smoothing",
        "Apply the backend spacing guard to proposed smoothed contours. Physical conductor clearance and field accuracy remain separate evaluations.",
        "bool",
    ),
    ExpertField(
        "contour_smoothing_work_samples_per_mesh_length",
        "Fit samples per mesh length",
        "Contour smoothing",
        "Mesh-coupled B-spline work sampling density, at least four. Larger values increase fitting cost.",
        "int",
    ),
    ExpertField(
        "contour_smoothing_output_samples_per_mesh_length",
        "Output samples per mesh length",
        "Contour smoothing",
        "Mesh-coupled output sampling density, at least two. Larger values increase path size and downstream routing/sweep cost.",
        "int",
    ),
    ExpertField(
        "contour_smoothing_patch_rings",
        "Surface patch rings",
        "Contour smoothing",
        "Neighboring mesh face rings available for the surface chart, at least one. Larger patches can increase computation; invalid fits retain the original contour.",
        "int",
    ),
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
        "joint_group_routing_selection_policy",
        "Interconnection selection",
        "Routing",
        "Regular prioritizes fewer raised transfers, less added-route backtracking, excess travel and turning, then length. Physical retains historical jumps / fixed-current longitudinal field RMS / length ranking. All field, support and conductor checks remain required.",
        "enum",
        choices=("regular", "physical"),
    ),
    ExpertField(
        "joint_group_routing_port_search_count",
        "Attachment context budget",
        "Routing",
        "Maximum bounded attachment contexts to inspect. Search effort is not a proof of feasibility or optimality; forced cut sides remain binding.",
        "int",
    ),
    ExpertField(
        "joint_group_routing_cut_span_max_factor",
        "Maximum opening span / nominal span",
        "Routing",
        "Upper bound on searched cut-window span relative to the nominal opening. Wider windows can provide bend room; retained original contour geometry and all fixed-current field and body checks remain binding.",
    ),
    ExpertField(
        "joint_group_routing_seam_search_extent_rel",
        "Seam search / contour perimeter",
        "Routing",
        "Bounded attachment displacement as a fraction of the original contour perimeter. This controls port search extent, not conductor lift, bend radius or field acceptance.",
        unit="%",
        display_scale=100.0,
    ),
    ExpertField(
        "joint_group_routing_regular_field_tolerance_rel",
        "Regularity RMS allowance / peak target",
        "Routing",
        "Extra whole-coil fixed-current longitudinal RMS allowed above the original route when preferring regular connections, as percent of peak active DSV target. Each eligible part receives an equal share. Zero forbids RMS growth. This additional limit never relaxes target, suppression, gradient or routing-change gates.",
        unit="%",
        display_scale=100.0,
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


# These are native controls, including knobs not included in the public settings
# view. The remaining dataclass fields get typed native-unit editors below.
_OPTIMIZATION_EXTRAS = (
    "optimization_solver_residual_tolerance",
    "l_curve_curvature_smoothing_window",
    "l_curve_factor_count",
    "l_curve_factor_min_multiplier",
    "l_curve_factor_max_multiplier",
    "l_curve_regularization_factors",
    "preoptimized_stream_function_match_tolerance",
    "sf_source_file",
    "field_evaluation_current_source",
)
_OPTIMIZATION_KEYS = frozenset(field.name for field in dataclass_fields(OptimizationSettings)) | frozenset(
    _OPTIMIZATION_EXTRAS
)
_CHOICES = {
    "sf_opt_method": ("tikhonov", "target_error_constrained_efficiency", "constrained_quadratic"),
    "stream_function_cost_function": ("field_error_plus_resistance", "standard", "least_squares_plus_resistance"),
    "optimization_solver": ("auto", "solve", "pinv"),
    "constrained_optimization_solver": ("auto", "clarabel", "ecos", "scs"),
    "regularization_selection_method": ("l_curve", "auto", "fixed"),
    "l_curve_scan_method": ("normal_equations", "transformed_svd", "auto"),
    "constrained_quadratic_objective_mode": (
        "balanced",
        "resistance",
        "resistance_plus_penalties",
        "smoothness_priority",
        "contourability_priority",
        "minimax_current_density",
    ),
    "constrained_quadratic_fallback_policy": ("target_error_scan", "diagnostic", "strict"),
    "target_error_constraint_mode": ("separate", "combined"),
    "target_error_constraint_metric": ("norm", "rms", "std"),
    "mri_shim_metric_mode": ("off", "report", "optimize"),
    "stream_function_basis": ("vertex", "cotangent_spectral_lowpass"),
    "stream_function_spectral_cutoff_policy": ("hard", "minimum_modes"),
    "field_evaluation_current_source": ("target_fit", "stream_function"),
    "research_physics.objective": ("power", "magnetic_energy", "minimax", "mixture"),
}
_HELP = {
    "sf_opt_method": "Tikhonov balances weighted field residual and regularization. Fixed-error efficiency searches regularization; constrained_quadratic imposes direct convex constraints. Active research_physics takes precedence.",
    "stream_function_cost_function": "Native least-squares field residual plus resistance regularization. This setting does not replace the active research physical objective or direct constrained objective mode.",
    "tikonov_reg_factor": "Native Tikhonov factor; used directly with fixed selection, or as the scan reference. Its effect depends on the weighted field operator and resistance normalization.",
    "regularization_selection_method": "Fixed uses the supplied factor; l_curve/auto use the existing regularization selection. Numerical conditioning and candidate feasibility still need inspection.",
    "l_curve_scan_method": "Normal equations retain the native scan. Transformed SVD changes the linear-algebra route for compatible objectives; unsupported cases follow the backend's explicit policy.",
    "optimization_solver": "Native linear solve policy: auto chooses the existing backend route; solve and pinv request direct or pseudoinverse handling. Residual checks remain active.",
    "optimization_solver_residual_tolerance": "Relative linear-system residual acceptance for the native solver; this is distinct from target field RMS and engineering acceptance.",
    "constrained_optimization_solver": "Installed conic backend selection. Auto uses the backend's supported solver ordering; selecting a solver does not install it or relax physical bounds.",
    "constrained_optimization_tolerance": "Numerical conic solver tolerance. Independently measured physical residual acceptance remains separate.",
    "constrained_optimization_threads": "Native solver thread request; zero keeps the existing automatic/default policy.",
    "constrained_optimization_max_iterations": "Conic solver iteration limit. Reaching this limit does not establish convergence or feasibility.",
    "constrained_quadratic_objective_mode": "Select the existing direct constrained objective: resistance/penalties, smoothness or minimax sheet current. Feasible field and physical bounds remain separate requirements.",
    "constrained_quadratic_fallback_policy": "Strict rejects an unmet constrained contract; diagnostic retains an explicitly failed diagnostic result; target_error_scan permits the existing regularization-search fallback.",
    "target_error_constraint_metric": "Norm is the residual L2 contract; RMS normalizes by sample count and STD removes the mean. These metrics are not interchangeable acceptance budgets.",
    "target_error_constraint_mode": "Separate keeps target and suppression budgets distinct; combined uses the existing weighted joint residual contract.",
    "stream_function_basis": "Full vertex basis or cotangent Laplace–Beltrami lowpass basis. Lowpass truncation changes the admissible winding spectrum and can reduce field feasibility.",
    "stream_function_spectral_cutoff_length_m": "Physical spectral cutoff length in native metres; zero retains the existing disabled cutoff. Inspect retained modes and field feasibility.",
    "stream_function_smoothness_factor": "Additional native stream-function smoothness penalty; changing it changes the optimized winding tradeoff.",
    "research_physics": "Complete optional physical convex specification in native JSON. Null keeps it inactive; explicit configuration changes solver precedence. Individual research fields preserve every other nested key.",
    "research_physics.objective": "Power [W], magnetic energy [J], minimax sheet current [A/m], or dimensionless weighted mixture. An active physical specification takes precedence over the native method.",
    "research_physics.objective_weights": "Nonnegative weights keyed by power, magnetic_energy and minimax. Mixture requires at least one positive weight; objectives are divided by their reference scales.",
    "research_physics.objective_scales": "Positive reference scales in SI: power [W], magnetic_energy [J], minimax [A/m]. Missing scales use the existing feasible reference solution policy.",
    "research_physics.field_constraints": "Native JSON list of LinearFieldConstraint objects: name, matrix, target, tolerance, norm, unit and optional group_size. Matrix columns are full nodal stream functions [A].",
    "research_physics.scenarios": "Native JSON list of finite FieldScenario objects: name, matrix, target and optional rms_limit_T/peak_limit_T/group_size. Only recorded finite scenarios are covered.",
    "research_physics.energy": "EnergyQuadrature native JSON settings. Quadrature order, face limits and convergence checks determine availability and accuracy; changing them does not certify convergence.",
    "research_physics.b0_T": "Three Cartesian background B0 components [T] in the model coordinate frame; used for force and torque operators.",
    "research_physics.background_B_per_face_T": "Optional native JSON array of Cartesian background fields per surface face [T]; null uses uniform b0_T.",
    "research_physics.torque_origin_m": "Three Cartesian coordinates [m] of the torque origin in the model coordinate frame.",
    "research_physics.force_equality_N": "Optional three-component exact net-force equality [N]. Disabled/null adds no equality; zero vector explicitly requests zero force.",
    "research_physics.torque_equality_Nm": "Optional three-component exact net-torque equality [N·m]. Disabled/null adds no equality; zero vector explicitly requests zero torque.",
    "research_physics.acceptance_absolute_by_unit": "Native JSON map of absolute residual acceptance by physical unit. These numerical allowances are distinct from the specified field/mechanical budgets.",
}


def _category(key):
    if key.startswith("research_physics."):
        return "Research physics"
    if key.startswith(("regularization", "l_curve")) or key == "tikonov_reg_factor":
        return "Optimization · regularization"
    if key.startswith("stream_function_") and key != "stream_function_cost_function":
        return "Optimization · basis and smoothness"
    if key.startswith("mri_shim_"):
        return "Optimization · MRI shim objective"
    if key.startswith(("target_error", "suppression", "shield_gradient", "layout_target")):
        return "Optimization · field constraints"
    if key.startswith(("linearity", "gradient_slice", "gradient_strength")):
        return "Optimization · gradient evaluation"
    if key.startswith(("contour", "turn_spacing", "zero_crossing")):
        return "Optimization · winding realization"
    return "Optimization · solver and objective"


def _native_field(key, annotation):
    optional = "None" in str(annotation)
    label = key.rsplit(".", 1)[-1].replace("_", " ").capitalize()
    category = _category(key)
    help_text = _HELP.get(
        key, f"Native {key} setting. Values retain native units and are validated together when Apply is pressed."
    )
    if key in _CHOICES:
        return ExpertField(key, label, category, help_text, "enum", choices=_CHOICES[key], optional=optional)
    annotation = str(annotation)
    if any(name in annotation for name in ("Sequence", "tuple", "Mapping", "dict", "Any", "EnergyQuadrature")):
        shape = (
            "mapping"
            if any(name in annotation for name in ("Mapping", "dict", "EnergyQuadrature"))
            else (
                "int_or_sequence"
                if "int | Sequence" in annotation
                else "sequence"
                if "Sequence" in annotation or "tuple" in annotation
                else ""
            )
        )
        return ExpertField(key, label, category, help_text, "json", optional=optional, json_shape=shape)
    kind = (
        "bool"
        if "bool" in annotation
        else "int"
        if annotation.removesuffix(" | None") == "int"
        else ("string" if annotation.removesuffix(" | None") == "str" else "float")
    )
    unit = next(
        (
            unit
            for suffix, unit in (
                ("_T", "T"),
                ("_m", "m"),
                ("_A", "A"),
                ("_W", "W"),
                ("_J", "J"),
                ("_N", "N"),
                ("_Nm", "N·m"),
                ("_hz", "Hz"),
                ("_percent", "%"),
            )
            if key.endswith(suffix)
        ),
        "",
    )
    return ExpertField(key, label, category, help_text, kind, unit=unit, optional=optional)


def expert_fields():
    """Cover native optimization/physics dataclasses without shadow schemas."""
    specs = {spec.key: spec for spec in FIELDS}
    annotations = {field.name: field.type for field in dataclass_fields(CoilGenConfig)}
    for key in (field.name for field in dataclass_fields(OptimizationSettings)):
        if key not in specs:
            specs[key] = _native_field(key, annotations[key])
    for key in _OPTIMIZATION_EXTRAS:
        specs[key] = _native_field(key, annotations[key])
    for field in dataclass_fields(ResearchPhysicsSpec):
        key = "research_physics." + field.name
        if key not in specs:
            specs[key] = _native_field(key, field.type)
        elif key == "research_physics.objective":
            specs[key] = replace(specs[key], help=_HELP[key])
    return tuple(specs.values())


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _check_finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    if isinstance(value, (list, dict)):
        for item in value.values() if isinstance(value, dict) else value:
            _check_finite_json(item)


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
        self.enabled_bound = QCheckBox("Use value" if spec.kind == "json" else "Use bound") if spec.optional else None
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
        elif spec.kind == "json":
            self.control = QPlainTextEdit()
            self.control.setMaximumHeight(110)
            self.control.setPlaceholderText("Native JSON object / array; retain SI units")
            signal = self.control.textChanged
        else:
            self.control = QLineEdit()
            self.control.setPlaceholderText(
                f"Value in {spec.unit}" if spec.unit else "Text value" if spec.kind == "string" else "Numeric value"
            )
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
        self._loaded_value = deepcopy(value)
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
            elif self.spec.kind == "json":
                self.control.setPlainText("" if value is None else json.dumps(value, indent=2))
            else:
                text = (
                    ""
                    if value is None
                    else repr(value if self.spec.display_scale == 1 else value * self.spec.display_scale)
                    if self.spec.kind in {"float", "int"}
                    and isinstance(value, (float, int))
                    and not isinstance(value, bool)
                    else str(value)
                )
                self.control.setText(text)
        self._loaded_text = (
            self.control.toPlainText()
            if self.spec.kind == "json"
            else (self.control.currentText() if self.spec.kind == "enum" else self.control.text())
        )
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
        text = (
            self.control.toPlainText()
            if self.spec.kind == "json"
            else (self.control.currentText() if self.spec.kind == "enum" else self.control.text())
        )
        if text == self._loaded_text and self._loaded_value is not None:
            return deepcopy(self._loaded_value)
        if self.spec.kind == "enum":
            return self.control.currentText()
        if self.spec.kind == "string":
            return text
        text = text.strip()
        if self.spec.kind == "json":
            try:
                value = json.loads(text, parse_constant=_reject_nonfinite)
                _check_finite_json(value)
            except ValueError as exc:
                raise ValueError("Enter valid finite native JSON: " + str(exc)) from exc
            shape = self.spec.json_shape
            if value is None and self.spec.optional:
                return None
            if shape == "mapping" and not isinstance(value, dict):
                raise ValueError("Enter a JSON object.")
            if shape == "sequence" and not isinstance(value, list):
                raise ValueError("Enter a JSON array.")
            if shape == "int_or_sequence" and not (
                isinstance(value, list) or isinstance(value, int) and not isinstance(value, bool)
            ):
                raise ValueError("Enter an integer or a JSON array.")
            return value
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
        self.optimization_only = QCheckBox("Optimization and physics only")
        self.optimization_only.setObjectName("expert_optimization_only")
        self.optimization_only.toggled.connect(self._filter_mode_changed)
        search_row.addWidget(self.optimization_only)
        find = QPushButton("Find in JSON")
        find.clicked.connect(self.find_in_json)
        search_row.addWidget(find)
        layout.addLayout(search_row)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("expert_tabs")
        layout.addWidget(self.tabs, 1)
        scroll = self.typed_scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = self._body_layout = QVBoxLayout(body)
        self.physics_note = QLabel()
        self.physics_note.setWordWrap(True)
        body_layout.addWidget(self.physics_note)
        self.configure_physics = QPushButton("Configure physical convex solver (power objective)")
        self.configure_physics.setObjectName("configure_research_physics")
        self.configure_physics.clicked.connect(self._enable_physics)
        body_layout.addWidget(self.configure_physics)
        for spec in expert_fields():
            if spec.category not in self.groups:
                group = QGroupBox(spec.category)
                group.setLayout(QVBoxLayout())
                self.groups[spec.category] = group
                body_layout.addWidget(group)
            editor = _FieldEditor(spec, self._default_field(spec.key))
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

    def _default_field(self, key):
        return None if key == "research_physics" else _read(self._defaults, key, self._defaults)

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
        terms = (text if isinstance(text, str) else self.search.text()).casefold().split()
        count = 0
        categories = set()
        for key, editor in self.fields.items():
            spec = editor.spec
            haystack = " ".join((key, spec.label, spec.category, spec.help)).casefold()
            relevant = (
                not self.optimization_only.isChecked()
                or key in _OPTIMIZATION_KEYS
                or key.startswith("research_physics.")
            )
            visible = relevant and all(term in haystack for term in terms)
            editor.setVisible(visible)
            if visible:
                count += 1
                categories.add(spec.category)
        for category, group in self.groups.items():
            group.setVisible(category in categories)
        self.no_match.setVisible(not count)

    def _filter_mode_changed(self, active):
        categories = list(self.groups)
        if active:
            categories.sort(key=lambda category: not category.startswith("Optimization ·"))
        for index, category in enumerate(categories, start=2):
            self._body_layout.removeWidget(self.groups[category])
            self._body_layout.insertWidget(index, self.groups[category])
        self.filter_fields()

    def focus_optimization(self):
        """Expose solver/objective/basis/constraint controls without changing data."""
        self.tabs.setCurrentIndex(0)
        if self.tabs.currentIndex() != 0:
            return False
        self.optimization_only.setChecked(True)
        self.search.clear()
        self.typed_scroll.verticalScrollBar().setValue(0)
        self.search.setFocus()
        return True

    def focus_native_settings(self):
        """Open the full native-unit JSON draft; invalid typed text stays intact."""
        self.tabs.setCurrentIndex(1)
        if self.tabs.currentIndex() != 1:
            return False
        self.json_editor.setFocus()
        return True

    def _update_fields(self):
        active = isinstance(self._draft.get("research_physics"), dict)
        for key, editor in self.fields.items():
            if key not in self._typed_errors:
                editor.set_value(_read(self._draft, key, self._defaults))
            editor.setEnabled(active or not key.startswith("research_physics."))
        self.physics_note.setText(
            "Physical convex solver active. Objectives, bounds, equalities, energy, scenarios and acceptance policy are editable below. Each field preserves the other nested values. The complete optional JSON control can explicitly replace or disable the specification."
            if active
            else "The imported physical specification is not a JSON object and is retained. Use native JSON, Default, or explicit Configure to replace it."
            if self._draft.get("research_physics") is not None
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
            if key == "research_physics":
                self._update_fields()
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
        value = deepcopy(self._default_field(key))
        try:
            self._write_field(key, value)
        except ValueError as exc:
            self.error_label.setText(str(exc))
            self._notify_draft()
            return False
        self._typed_errors.pop(key, None)
        self.fields[key].set_value(value)
        if key == "research_physics":
            self._update_fields()
        self._write_json()
        self.error_label.clear()
        self._notify_draft()
        return True

    def _enable_physics(self):
        if not isinstance(self._draft.get("research_physics"), dict):
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
