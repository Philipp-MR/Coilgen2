"""Shared form metadata, units and conversion semantics for all clients."""

from __future__ import annotations
from dataclasses import dataclass, replace
import math
from typing import Any, Callable
from ..field_components import TARGET_COMPONENTS, normalized_b0_direction
from ..target import _target_gradient_normalization_method
from ..target_regions import TargetRegion
from . import cylinder_pair as _pair
from .support_parameters import SUPPORT_EDITORS
from .targets import ORIGIN_KEYS, TARGET_NORMALIZATION_MODES, recognize_solid_harmonic_target


@dataclass(frozen=True)
class SettingField:
    """A form field over one or more configuration (or study) entries."""

    node: str
    key: str
    label: str
    kind: str  # number, integer, choice, text, bool, vector
    get: Callable[[Any, dict], Any]
    set: Callable[[Any, Any, dict], dict]
    choices: tuple = ()
    unit: str = ""
    help: str = ""
    study: bool = False


def _scaled(name, scale):
    return (
        lambda config, study: float(getattr(config, name)) * scale,
        lambda value, config, study: {name: float(value) / scale},
    )


def _plain(name, cast=None):
    return (
        lambda config, study: getattr(config, name),
        lambda value, config, study: {name: cast(value) if cast else value},
    )


def _inverted(name):
    return (lambda config, study: not bool(getattr(config, name)), lambda value, config, study: {name: not bool(value)})


def _vector(name):
    def parse(value, config, study):
        values = value if isinstance(value, (list, tuple)) else str(value).replace(";", ",").split(",")
        numbers = tuple(float(item) for item in values)
        if len(numbers) != 3:
            raise ValueError(f"{name} needs three comma-separated numbers.")
        return {name: numbers}

    return (lambda config, study: ", ".join(str(float(v)).removesuffix(".0") for v in getattr(config, name)), parse)


_TARGET_CENTER_FIELDS = tuple(f"target_region_center_{axis}_mm" for axis in "xyz")
_TARGET_RADIUS_FIELD = "target_region_radius_mm"
_TARGET_REGION_FIELDS = frozenset((*_TARGET_CENTER_FIELDS, _TARGET_RADIUS_FIELD))


def target_scaling_mode(config, values):
    """Resolve the draft scaling choice using the same policy as target evaluation."""
    candidate = replace(
        config,
        target_gradient_normalization=values.get("target_gradient_normalization", config.target_gradient_normalization),
        design_context=values.get("design_context", config.design_context),
    )
    source = "file" if str(config.target_field_definition_file).strip().lower() not in {"none", ""} else "expression"
    return _target_gradient_normalization_method(candidate, source=source)


def _target_region(config):
    if config.target_region_geometry is None:
        return None
    return TargetRegion.from_dict(config.target_region_geometry)


def _origin_parameter(index):
    def read(config, study):
        return recognize_solid_harmonic_target(config.field_shape_function).center_m[index] * 1000

    return read, lambda value, config, study: {}


def _target_region_values(config, region):
    center = region.center_m if region else (0.0, 0.0, 0.0)
    radius = region.radius_m if region and region.radius_m is not None else config.target_region_radius
    return {
        **{key: value * 1000 for key, value in zip(_TARGET_CENTER_FIELDS, center)},
        _TARGET_RADIUS_FIELD: radius * 1000,
    }


def _target_region_edits(config, values):
    """Commit radius and world placement together while preserving the legacy grid until needed."""
    if not values:
        return {}
    region = _target_region(config)
    initial = _target_region_values(config, region)
    try:
        edits = {key: float(value) for key, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter finite ROI radius and world center coordinates.") from exc
    edits = {key: value for key, value in edits.items() if value != initial[key]}
    if not edits:
        return {}
    if not all(math.isfinite(value) for value in edits.values()):
        raise ValueError("Enter finite ROI radius and world center coordinates.")
    radius = edits.get(_TARGET_RADIUS_FIELD)
    changes = {"target_region_radius": radius / 1000} if radius is not None else {}
    moved = any(key in edits for key in _TARGET_CENTER_FIELDS)
    if moved or (region is not None and radius is not None):
        definition = region.to_dict() if region else {"shape": "sphere", "radius_m": config.target_region_radius}
        if moved:
            original_center = region.center_m if region else (0.0, 0.0, 0.0)
            definition["center_m"] = [
                edits[key] / 1000 if key in edits else previous
                for key, previous in zip(_TARGET_CENTER_FIELDS, original_center)
            ]
        if radius is not None:
            definition["radius_m"] = radius / 1000
        changes.update(target_region_geometry=definition, set_roi_into_mesh_center=False)
    return changes


def _target_region_parameter(key):
    return (
        lambda config, study: _target_region_values(config, _target_region(config))[key],
        lambda value, config, study: _target_region_edits(config, {key: value}),
    )


def _contour_treatment(config, study):
    from ..contours import _normalise_contour_path_regularization_mode
    from ..contour_smoothing import _normalise_scale_mode

    mode = _normalise_contour_path_regularization_mode(config.contour_path_regularization_mode)
    if mode == "parameterized_bspline":
        return _normalise_scale_mode(config.contour_smoothing_scale_mode)
    return mode


def _set_contour_treatment(value, config, study):
    if value == _contour_treatment(config, study):
        return {}
    if value in {"off", "resample"}:
        return {"contour_path_regularization_mode": value}
    if value in {"mesh_coupled", "legacy"}:
        return {"contour_path_regularization_mode": "parameterized_bspline", "contour_smoothing_scale_mode": value}
    raise ValueError("Choose one of the available contour path treatments.")


_CYLINDER_KEYS = {
    "length_mm": "cylinder_length_mm",
    "radius_mm": "cylinder_radius_mm",
    "angular_divisions": "cylinder_angular",
    "length_divisions": "cylinder_axial",
    "rotation_angle_deg": "cylinder_rotation_deg",
    **{f"rotation_axis_{axis}": f"cylinder_rotation_axis_{axis}" for axis in "xyz"},
}
_CYLINDER_LABELS = {
    "cylinder_length_mm": "Cylinder length",
    "cylinder_radius_mm": "Cylinder radius",
    "cylinder_angular": "Azimuthal divisions",
    "cylinder_rotation_deg": "Cylinder rotation angle",
}
_SUPPORT_PREFIXES = {
    "create cylinder mesh": "cylinder",
    "create bi-planary mesh": "biplanar",
    "create planary mesh": "planar",
    "create conical frustum mesh": "conical",
}
_SUPPORT_FIELDS = {
    (_CYLINDER_KEYS[field.key] if prefix == "cylinder" else f"{prefix}_{field.key}"): (editor, field)
    for editor in SUPPORT_EDITORS
    if (prefix := _SUPPORT_PREFIXES.get(editor.source))
    for field in editor.fields
    if prefix != "cylinder" or field.key in _CYLINDER_KEYS
}


def _support_edits(config, values, source):
    edits = {}
    for key, value in values.items():
        editor, field = _SUPPORT_FIELDS[key]
        parameters = getattr(config, editor.parameter_field)
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field.label} must be finite.")
        if field.integer and not number.is_integer():
            raise ValueError(f"{field.label} must be a whole number.")
        if number != editor.display_value(field, parameters):
            if editor.source != source:
                raise ValueError("Apply or discard the dimension edits before changing support surfaces.")
            edits[field.key] = int(number) if field.integer else number
    if not edits:
        return {}
    editor = next(editor for editor in SUPPORT_EDITORS if editor.source == source)
    return {editor.parameter_field: editor.apply_edits(getattr(config, editor.parameter_field), edits)}


def _support_parameter(key):
    editor, field = _SUPPORT_FIELDS[key]
    return (
        lambda config, study: editor.display_value(field, getattr(config, editor.parameter_field)),
        lambda value, config, study: _support_edits(config, {key: value}, config.coil_mesh_file),
    )


def _linked_background_change(config, changes):
    """Track an MRI-field edit only when the uniform mechanical background was linked."""
    physics = config.research_physics
    if physics is None or physics.get("background_B_per_face_T") is not None:
        return {}
    direction = changes.get("b_0_direction", config.b_0_direction)
    strength = changes.get("b_0_strength_T", config.b_0_strength_T)
    if strength == config.b_0_strength_T and tuple(direction) == tuple(config.b_0_direction):
        return {}

    from ..research.convex import ResearchPhysicsSpec

    background = ResearchPhysicsSpec.from_dict(physics).b0_T
    previous = normalized_b0_direction(config.b_0_direction) * config.b_0_strength_T
    if background is None or not all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(background, previous, strict=True)
    ):
        return {}
    return {"research_physics": {**physics, "b0_T": (normalized_b0_direction(direction) * strength).tolist()}}


def _study(name, cast):
    return (lambda config, study: study.get(name), lambda value, config, study: {name: cast(value)})


def _field(node, key, label, kind, accessors, **kwargs) -> SettingField:
    get, set_ = accessors
    return SettingField(node, key, label, kind, get, set_, **kwargs)


def _pair_fields(specs):
    """Shielded-pair fields read and write owned meshes: the session fills and applies them."""

    def unavailable(value, config, study):
        raise ValueError("Apply cylinder pair dimensions through the settings form.")

    return tuple(
        _field(node, key, label, kind, (lambda config, study: None, unavailable), unit=unit, help=help_)
        for key, node, label, kind, unit, help_ in specs
    )


SETTINGS_NODES = (
    ("geometry", "Geometry and mesh"),
    ("target", "Target field and B0"),
    ("conductor", "Conductor and material"),
    ("optimization", "Stream function optimization"),
    ("winding", "Winding and routing"),
    ("print", "3D-print former"),
    ("validation", "Validation settings"),
)
MESH_SOURCES = tuple(_SUPPORT_PREFIXES)
# layout_terminal_strategy values, in menu order; labels live in submitted_design.
TERMINAL_STRATEGIES = ("undefined", "pads", "wire_stubs", "waived")

SETTINGS = (
    _field(
        "geometry",
        "coil_mesh_file",
        "Support surface",
        "choice",
        _plain("coil_mesh_file"),
        choices=(*MESH_SOURCES, _pair.SHIELDED_PAIR_SOURCE),
    ),
    *_pair_fields(_pair.PAIR_FIELDS),
    *(
        _field(
            "geometry",
            key,
            _CYLINDER_LABELS.get(key, field.label),
            "integer" if field.integer else "number",
            _support_parameter(key),
            unit=field.suffix.strip(),
        )
        for key, (_editor, field) in _SUPPORT_FIELDS.items()
    ),
    _field(
        "geometry",
        "imported_mesh_length_scale_to_m",
        "Imported mesh unit scale",
        "number",
        _plain("imported_mesh_length_scale_to_m", float),
        help="0.001 for STL files in millimetres.",
    ),
    _field(
        "target",
        "field_shape_function",
        "Target expression (x, y, z in metres)",
        "text",
        _plain("field_shape_function", str),
        help="The expression uses world coordinates. For an absolute Bz target in tesla, choose Absolute field expression.",
    ),
    *(
        _field(
            "target",
            key,
            f"Field-pattern origin {axis} (world)",
            "number",
            _origin_parameter(index),
            unit="mm",
            help="Moves the harmonic field independently of the sampling region.",
        )
        for index, (key, axis) in enumerate(zip(ORIGIN_KEYS, "xyz"))
    ),
    _field(
        "target",
        "target_field_component",
        "Target field component",
        "choice",
        _plain("target_field_component"),
        choices=TARGET_COMPONENTS,
    ),
    _field(
        "target",
        "target_gradient_strength",
        "Gradient amplitude",
        "number",
        _scaled("target_gradient_strength", 1000.0),
        unit="mT/m",
    ),
    _field(
        "target", "target_amplitude_native", "Target amplitude", "number", _plain("target_gradient_strength", float)
    ),
    _field(
        "target",
        "target_gradient_normalization",
        "Target scaling",
        "choice",
        _plain("target_gradient_normalization"),
        choices=tuple(names[0] for names, _label in TARGET_NORMALIZATION_MODES),
        help="Absolute field expression uses the entered values directly. Automatic fits a slope for MRI gradient expressions and uses a multiplier in other contexts.",
    ),
    _field(
        "target",
        "target_field_units",
        "Field expression units",
        "choice",
        _plain("target_field_units"),
        choices=("T", "Hz"),
    ),
    _field(
        "target",
        "target_gradient_axis",
        "Gradient axis",
        "choice",
        _plain("target_gradient_axis"),
        choices=("x", "y", "z", "magnitude", "auto"),
        help="Use magnitude (|G|) for combined targets such as x+y. Current and error always use the full expression.",
    ),
    _field(
        "target",
        _TARGET_RADIUS_FIELD,
        "ROI radius",
        "number",
        _target_region_parameter(_TARGET_RADIUS_FIELD),
        unit="mm",
    ),
    *(
        _field(
            "target",
            key,
            f"ROI world center {axis}",
            "number",
            _target_region_parameter(key),
            unit="mm",
            help="Moves the sampling region in world coordinates. The field expression keeps the same coordinates."
            if axis == "X"
            else "",
        )
        for key, axis in zip(_TARGET_CENTER_FIELDS, "XYZ")
    ),
    _field(
        "target",
        "target_region_resolution",
        "ROI resolution",
        "integer",
        _plain("target_region_resolution", int),
        help="For an explicit volume, this is the number of bins per local axis. The target plot reports the actual sample count.",
    ),
    _field(
        "target",
        "design_context",
        "Design context",
        "choice",
        _plain("design_context"),
        choices=("mr_gradient", "mr_shim", "free", "mr_rf"),
    ),
    _field("target", "b_0_direction", "B0 direction (x, y, z)", "vector", _vector("b_0_direction")),
    _field("target", "b_0_strength_T", "B0 strength", "number", _plain("b_0_strength_T", float), unit="T"),
    *_pair_fields(_pair.SUPPRESSION_SHAPE_FIELDS),
    _field(
        "target",
        "suppression_rms_limit_uT",
        "External field RMS limit",
        "number",
        _scaled("suppression_field_rms_limit_T", 1e6),
        unit="µT",
        help="RMS field allowed on the suppression surface. Zero sets no limit.",
    ),
    _field(
        "target",
        "suppression_field_component",
        "Limited field component",
        "choice",
        _plain("suppression_field_component"),
        choices=_pair.SUPPRESSION_COMPONENTS,
    ),
    _field(
        "target",
        "suppression_surface_weighting",
        "Sample weighting",
        "choice",
        _plain("suppression_surface_weighting"),
        choices=_pair.SUPPRESSION_WEIGHTINGS,
        help="By area weights each sample by its share of the surface, so dense mesh regions do not dominate.",
    ),
    _field(
        "conductor",
        "conductor_width_mm",
        "Conductor width",
        "number",
        _scaled("conductor_cross_section_width", 1000.0),
        unit="mm",
    ),
    _field(
        "conductor",
        "conductor_height_mm",
        "Conductor height",
        "number",
        _scaled("conductor_cross_section_height", 1000.0),
        unit="mm",
    ),
    _field(
        "conductor",
        "insulation_mm",
        "Insulation thickness",
        "number",
        _scaled("sweep_insulation_thickness", 1000.0),
        unit="mm",
    ),
    _field(
        "conductor",
        "conductor_mounting_mode",
        "Conductor side of support",
        "choice",
        _plain("conductor_mounting_mode"),
        choices=("centered", "surface_outer", "surface_inner"),
    ),
    _field(
        "conductor",
        "conductor_path_reference",
        "Sweep reference point",
        "choice",
        _plain("conductor_path_reference"),
        choices=("centroid", "profile_origin"),
        help="Profile origin (0, 0) sweeps about the profile's (0, 0) point, e.g. a litz-wire centre in an omega "
        "holder. Profile centroid follows the cross-section's area centroid.",
    ),
    _field("conductor", "return_path_lift", "Lift crossing return paths", "bool", _inverted("skip_normal_shift")),
    _field(
        "conductor", "normal_shift_mm", "Return-path lift", "number", _scaled("normal_shift_length", 1000.0), unit="mm"
    ),
    _field(
        "print",
        "support_body_mode",
        "Print a former with wire grooves",
        "choice",
        _plain("support_body_mode"),
        choices=("off", "inside", "outside"),
        help="Optional. Choose whether the solid wall grows inside or outside the winding; the conductor "
        "paths are cut into it as grooves when the design is computed, and the former is exported as its own STL.",
    ),
    _field(
        "print",
        "support_body_thickness_mm",
        "Wall thickness",
        "number",
        _scaled("support_body_thickness", 1000.0),
        unit="mm",
    ),
    _field(
        "print",
        "support_body_groove_clearance_mm",
        "Groove clearance",
        "number",
        _scaled("support_body_groove_clearance", 1000.0),
        unit="mm",
        help="Gap around the insulated conductor in the groove, e.g. your printer's fit tolerance.",
    ),
    _field(
        "print",
        "support_body_groove_extra_depth_mm",
        "Extra groove depth (advanced)",
        "number",
        _scaled("support_body_groove_extra_depth", 1000.0),
        unit="mm",
        help="Optional: cut the groove deeper than the conductor.",
    ),
    _field(
        "optimization",
        "sf_opt_method",
        "Stream-function method",
        "choice",
        _plain("sf_opt_method"),
        choices=("tikhonov", "target_error_constrained_efficiency", "constrained_quadratic"),
    ),
    _field(
        "optimization", "tikonov_reg_factor", "Regularization factor", "number", _plain("tikonov_reg_factor", float)
    ),
    _field(
        "optimization",
        "target_error_tolerance_percent",
        "Target error tolerance",
        "number",
        _plain("target_error_tolerance_percent", float),
        unit="%",
    ),
    _field("winding", "levels", "Contour levels", "integer", _plain("levels", int)),
    _field(
        "winding",
        "contour_treatment",
        "Contour path treatment",
        "choice",
        (_contour_treatment, _set_contour_treatment),
        choices=("off", "mesh_coupled", "legacy", "resample"),
        help="Smooth closed contours on the support before routing. If no fit passes the limits, keep the extracted contour.",
    ),
    _field(
        "winding",
        "contour_smoothing_mesh_gain",
        "Gain at mesh wavelength",
        "number",
        _plain("contour_smoothing_mesh_gain", float),
        help="Between 0 and 1. Lower values suppress mesh-scale kinks more strongly.",
    ),
    _field(
        "winding",
        "contour_smoothing_factor",
        "Legacy smoothing strength",
        "number",
        _plain("contour_smoothing_factor", float),
        help="Zero disables the legacy fit. Increase toward 1 for stronger smoothing.",
    ),
    _field(
        "winding",
        "contour_displacement_mm",
        "Maximum contour displacement",
        "number",
        _scaled("contour_smoothing_max_displacement", 1000),
        unit="mm",
        help="Maximum allowed departure from the extracted contour. Zero removes the displacement cap.",
    ),
    _field(
        "winding",
        "contour_smoothing_preserve_min_turn_distance",
        "Guard inter-turn spacing during smoothing",
        "bool",
        _plain("contour_smoothing_preserve_min_turn_distance", bool),
        help="Reject fits that violate the requested spacing. With zero spacing, or when the original contours already miss the limit, preserve their original minimum separation.",
    ),
    _field(
        "winding",
        "contour_smoothing_local_fallback",
        "Try local corner blends if a fit is rejected",
        "bool",
        _plain("contour_smoothing_local_fallback", bool),
        help="Round individual corners on the support when a whole-contour fit fails. The displacement cap and spacing guard must both be active.",
    ),
    _field(
        "winding",
        "contour_segment_mm",
        "Target contour segment length",
        "number",
        _scaled("contour_resample_target_segment_length", 1000),
        unit="mm",
        help="Zero uses automatic B-spline sampling and disables resampling-only mode.",
    ),
    _field(
        "winding",
        "min_turn_distance_mm",
        "Minimum centerline spacing",
        "number",
        _scaled("min_turn_distance", 1000.0),
        unit="mm",
        help="Distance between centerlines before routing; the final copper-and-insulation clearance is checked separately.",
    ),
    _field(
        "winding", "min_bend_mm", "Minimum bend radius", "number", _scaled("sweep_min_bend_radius", 1000.0), unit="mm"
    ),
    _field(
        "winding",
        "interconnection_axis",
        "Interconnection direction",
        "choice",
        _plain("interconnection_axis"),
        choices=("b0", "support_axis"),
        help="Along the support bore routes cuts and returns along the bore; try it when the bore is "
        "perpendicular to B0, as in transverse-field or Halbach magnets.",
    ),
    _field("winding", "sweep", "Generate conductor solids", "bool", _inverted("skip_sweep")),
    _field(
        "winding",
        "layout_terminal_strategy",
        "Terminal handling",
        "choice",
        _plain("layout_terminal_strategy"),
        choices=TERMINAL_STRATEGIES,
        help="How the coil's feed ends are connected. CoilGen records this choice for review; it does not "
        "build pads or stubs.",
    ),
    _field(
        "validation",
        "validation_resolution",
        "Independent grid resolution",
        "integer",
        _study("validation_resolution", int),
        study=True,
    ),
    _field("validation", "design_margin", "Design budget ratio", "number", _study("design_margin", float), study=True),
    _field("validation", "fabricate", "Fabrication review", "bool", _study("fabricate", bool), study=True),
)
SETTINGS_BY_KEY = {item.key: item for item in SETTINGS}
