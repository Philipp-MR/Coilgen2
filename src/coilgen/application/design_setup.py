"""One retained setup draft for both interfaces, with no numerical work on edit."""

from __future__ import annotations

from dataclasses import replace
import math

from .conductor_profiles import is_rectangular_profile, round_wire_options, round_wire_profile
from .session import ProjectSettings
from .settings import SettingField, _SUPPORT_FIELDS, _field, _plain, _scaled, target_scaling_mode
from .submitted_design import choice_title
from .targets import ORIGIN_KEYS, target_normalization_choices
from ..config_semantics import normalize_field_budget_metric
from ..target_regions import target_region_summary


SECTIONS = (
    ("geometry", "Support geometry", "Choose the surface that carries the winding."),
    ("target", "Target field", "Choose the field and the region where it should match."),
    ("conductor", "Conductor", "Set the physical wire or track dimensions."),
    ("optimization", "Optimization", "Choose how to fit the target field."),
    ("winding", "Winding", "Set the turns, spacing and path treatment."),
    ("print", "Printed former", "Optional support with grooves for the wire."),
    ("validation", "Validation", "Choose the checks for the computed design."),
)
COMMON = {
    "geometry": {"coil_mesh_file"},
    "target": {"setup_field", "target_amplitude_native", "target_region_radius_mm"},
    "conductor": {"setup_profile", "conductor_width_mm", "conductor_height_mm"},
    "optimization": {"sf_opt_method", "target_error_tolerance_percent"},
    "winding": {"levels", "min_turn_distance_mm", "contour_treatment"},
    "print": {"support_body_mode", "support_body_thickness_mm"},
    "validation": {"fabricate"},
}
FIELD_CHOICES = {
    "x": "X gradient",
    "y": "Y gradient",
    "z": "Z gradient",
    "1": "Uniform field",
    "custom": "Custom expression",
}
PROFILE_CHOICES = {"round": "Round wire", "rectangle": "Rectangular wire / track", "custom": "Saved custom profile"}

# Additional controls use the same configuration accessors as the existing editors.
EXTRA_FIELDS = (
    _field(
        "optimization",
        "target_error_tolerance_absolute_T",
        "Additional absolute field error budget",
        "number",
        _plain("target_error_tolerance_absolute_T", float),
        unit="T",
        help="Added to the relative budget unless an explicit RMS or standard deviation limit is active.",
    ),
    _field(
        "optimization",
        "target_error_constraint_metric",
        "Field error metric",
        "choice",
        _plain("target_error_constraint_metric"),
        choices=("norm", "rms", "std"),
    ),
    _field(
        "optimization",
        "target_error_rms_limit_T",
        "Absolute RMS error limit",
        "number",
        _plain("target_error_rms_limit_T", float),
        unit="T",
    ),
    _field(
        "optimization",
        "target_error_std_limit_T",
        "Absolute standard deviation limit",
        "number",
        _plain("target_error_std_limit_T", float),
        unit="T",
    ),
    _field(
        "optimization",
        "constrained_optimization_tolerance",
        "Solver tolerance",
        "number",
        _plain("constrained_optimization_tolerance", float),
    ),
    _field(
        "optimization",
        "constrained_optimization_max_iterations",
        "Maximum solver iterations",
        "integer",
        _plain("constrained_optimization_max_iterations", int),
    ),
    _field("optimization", "max_current_A", "Current limit", "number", _plain("max_current_A", float), unit="A"),
    _field(
        "winding",
        "contour_discretization_method",
        "Choose contour spacing by",
        "choice",
        _plain("contour_discretization_method"),
        choices=("level_count", "engineering", "auto"),
    ),
    _field(
        "winding",
        "max_wire_current_A",
        "Wire current limit",
        "number",
        _plain("max_wire_current_A", float),
        unit="A",
        help="Zero uses the general current limit. Used by current-and-spacing contouring.",
    ),
    _field(
        "winding",
        "min_wire_distance_mm",
        "Engineering centerline spacing",
        "number",
        _scaled("min_wire_distance_m", 1000),
        unit="mm",
        help="Zero uses the minimum turn spacing. This is centerline spacing, not copper edge clearance.",
    ),
    _field(
        "conductor",
        "specific_conductivity_conductor",
        "Conductor resistivity",
        "number",
        _plain("specific_conductivity_conductor", float),
        unit="Ω m",
    ),
)
EXTRA_BY_KEY = {item.key: item for item in EXTRA_FIELDS}


class SetupInputError(ValueError):
    def __init__(self, message, *, key="", section=""):
        super().__init__(message)
        self.key, self.section = key, section


def _same(value, saved):
    if isinstance(saved, (int, float)) and not isinstance(saved, bool):
        try:
            return float(value) == saved
        except (TypeError, ValueError):
            return False
    return value == saved


class DesignSetupDraft:
    """Preserve untouched native precision and commit all sections atomically."""

    def __init__(self, project):
        self.project = project
        self.model = ProjectSettings(project)
        self.values = self.model.settings_values()
        config = project.config
        self.values.update({item.key: item.get(config, project.study) for item in EXTRA_FIELDS})
        expression = config.field_shape_function.strip()
        self.values["setup_field"] = expression if expression in FIELD_CHOICES else "custom"
        width, height = config.conductor_cross_section_width, config.conductor_cross_section_height
        self.round_options = round_wire_options(config.cross_sectional_points, width) if width == height else None
        self.values["setup_profile"] = (
            "round"
            if self.round_options
            else "rectangle"
            if is_rectangular_profile(config.cross_sectional_points)
            else "custom"
        )
        self.amplitude_scale = self._amplitude_scale()
        if "target_amplitude_native" in self.values:
            self.values["target_amplitude_native"] *= self.amplitude_scale
        self.initial = dict(self.values)

    def _amplitude_scale(self):
        return (
            1000
            if target_scaling_mode(self.project.config, self.values) == "linear_fit"
            and self.values.get("target_field_units") == "T"
            else 1
        )

    @property
    def dirty(self):
        return any(not _same(value, self.initial.get(key)) for key, value in self.values.items())

    def update(self, values):
        previous = self.values.get("setup_field")
        self.values.update(values)
        field = self.values["setup_field"]
        if field != previous and field != "custom":
            self.values["field_shape_function"] = field
            self.values["target_gradient_normalization"] = "linear_fit" if field in "xyz" else "scale"
            if field in "xyz":
                self.values["target_gradient_axis"] = field
        scale = self._amplitude_scale()
        if scale != self.amplitude_scale:
            try:
                self.values["target_amplitude_native"] = (
                    float(self.values["target_amplitude_native"]) * scale / self.amplitude_scale
                )
                self.initial["target_amplitude_native"] *= scale / self.amplitude_scale
            except (KeyError, TypeError, ValueError):
                pass  # Keep incomplete text; commit will identify the field.
            self.amplitude_scale = scale
        # A newly selected support uses that support's saved parameter block.
        for item in self.fields():
            if item.key not in self.values:
                value = item.get(self.project.config, self.project.study)
                self.values[item.key] = value
                self.initial[item.key] = value

    def fields(self, node=None):
        fields = self.model.settings_fields(
            node,
            support_source=self.values.get("coil_mesh_file"),
            former_mode=self.values.get("support_body_mode"),
        )
        fields = (*fields, *(item for item in EXTRA_FIELDS if node is None or item.node == node))
        result = []
        profile = self.values.get("setup_profile")
        for item in fields:
            if (
                item.key in ORIGIN_KEYS
                and self.values.get("field_shape_function") != self.project.config.field_shape_function
            ):
                continue
            if (
                item.key.startswith("target_error_")
                and (self.project.config.research_physics or {}).get("include_default_field_constraints") is False
            ):
                continue
            # These controls belong to Winding; never render a duplicate input.
            if item.key in {"return_path_lift", "normal_shift_mm"}:
                if node == "conductor":
                    continue
                item = replace(item, node="winding")
            if item.key == "field_shape_function":
                result.append(
                    SettingField(
                        "target",
                        "setup_field",
                        "Field type",
                        "choice",
                        lambda *_: "custom",
                        lambda *_: {},
                        choices=tuple(FIELD_CHOICES),
                    )
                )
            if item.key == "conductor_width_mm":
                result.append(
                    SettingField(
                        "conductor",
                        "setup_profile",
                        "Profile",
                        "choice",
                        lambda *_: "custom",
                        lambda *_: {},
                        choices=tuple(PROFILE_CHOICES),
                    )
                )
                if profile == "round":
                    item = replace(item, label="Wire diameter")
            if item.key == "conductor_height_mm" and profile == "round":
                continue
            if item.key == "target_amplitude_native":
                mode = target_scaling_mode(self.project.config, self.values)
                if mode == "none":
                    continue
                unit = self.values.get("target_field_units", "T")
                item = replace(
                    item,
                    label="Field multiplier" if mode == "scale" else "Gradient amplitude",
                    unit="" if mode == "scale" else "mT/m" if unit == "T" else unit + "/m",
                )
            if item.key == "tikonov_reg_factor" and (
                self.project.config.research_physics is not None
                or self.values.get("sf_opt_method") == "constrained_quadratic"
            ):
                continue
            if item.key == "levels" and self.values.get("contour_discretization_method") == "engineering":
                continue
            if (
                item.key == "target_gradient_axis"
                and target_scaling_mode(self.project.config, self.values) != "linear_fit"
            ):
                continue
            treatment = self.values.get("contour_treatment", "off")
            if item.key == "contour_smoothing_mesh_gain" and treatment != "mesh_coupled":
                continue
            if item.key == "contour_smoothing_factor" and treatment != "legacy":
                continue
            if item.key in {
                "contour_displacement_mm",
                "contour_smoothing_preserve_min_turn_distance",
                "contour_smoothing_local_fallback",
            } and treatment not in {"legacy", "mesh_coupled"}:
                continue
            if item.key == "contour_segment_mm" and treatment == "off":
                continue
            if item.key == "contour_smoothing_local_fallback" and not self.values.get(
                "contour_smoothing_preserve_min_turn_distance"
            ):
                continue
            result.append(item)
        return tuple(result)

    def common(self, field):
        if field.key == "tikonov_reg_factor":
            return self.values.get("sf_opt_method") == "tikhonov"
        metric = normalize_field_budget_metric(self.values.get("target_error_constraint_metric", "norm"))
        explicit_key = "target_error_" + metric + "_limit_T"
        try:
            explicit = float(self.values.get(explicit_key, 0)) > 0
        except (TypeError, ValueError):
            explicit = True  # Keep an incomplete active budget visible.
        if field.key == "target_error_tolerance_percent":
            return not explicit and self.values.get("sf_opt_method") != "tikhonov"
        if field.key == explicit_key and explicit:
            return True
        if field.key == "field_shape_function":
            return self.values.get("setup_field") == "custom"
        if field.key in _SUPPORT_FIELDS:
            _, dimension = _SUPPORT_FIELDS[field.key]
            return not any(token in dimension.key for token in ("rotation", "center", "normal", "division", "edge"))
        if field.key.startswith("pair_"):
            return field.key in {"pair_primary_radius_mm", "pair_shield_radius_mm", "pair_length_mm"}
        if field.key == "contour_discretization_method":
            return True
        if field.key == "max_wire_current_A":
            return self.values.get("contour_discretization_method") == "engineering"
        return field.key in COMMON[field.node]

    def entries(self, node):
        entries = []
        for item in self.fields(node):
            value = self.values.get(item.key, item.get(self.project.config, self.project.study))
            choices = list(item.choices)
            if item.kind == "choice":
                if value not in choices:
                    choices.insert(0, value)
                choices = [{"value": v, "title": self.choice_label(item.key, v)} for v in choices]
            entries.append(
                {
                    "key": item.key,
                    "label": item.label + (f" [{item.unit}]" if item.unit else ""),
                    "kind": item.kind,
                    "choices": choices,
                    "help": item.help,
                    "common": self.common(item),
                }
            )
        return entries

    @staticmethod
    def choice_label(key, value):
        if key == "setup_field":
            return FIELD_CHOICES.get(value, str(value))
        if key == "setup_profile":
            return PROFILE_CHOICES.get(value, str(value))
        if key == "coil_mesh_file":
            if str(value).lower().endswith(".npz"):
                return "Project-owned surface"
            if str(value).lower().endswith(".stl"):
                return "Imported STL surface"
            return str(value).removeprefix("create ").removesuffix(" mesh").replace("_", " ").capitalize()
        if key == "target_gradient_normalization":
            return dict(target_normalization_choices(value)).get(value, str(value))
        if key == "contour_discretization_method":
            return {
                "level_count": "Number of contour levels",
                "engineering": "Current and spacing limits",
                "auto": "Automatic",
            }.get(value, str(value))
        if key == "contour_treatment":
            return {
                "off": "Original contours",
                "mesh_coupled": "Smooth on surface",
                "legacy": "Legacy smoothing",
                "resample": "Resample only",
            }.get(value, str(value))
        return choice_title(key, value)

    def summary(self, node):
        config = self.project.config
        if node == "target":
            try:
                return "Applied region: " + target_region_summary(config)
            except (TypeError, ValueError):
                return "Check the target region settings."
        if node == "conductor" and self.values.get("setup_profile") == "custom":
            return "Saved custom cross-section retained; dimensions describe its nominal size."
        if node == "optimization":
            physics = config.research_physics
            objective = (
                "Saved physical objective: "
                + str(physics.get("objective", "power"))
                + ". Existing constraints are retained."
                if physics
                else "Saved objective: " + str(config.stream_function_cost_function).replace("_", " ") + "."
            )
            if physics and physics.get("include_default_field_constraints") is False:
                return objective + " Custom field constraints are edited under More tools → Advanced parameters."
            return (
                objective
                + " A positive explicit RMS or standard deviation limit overrides the relative and additional absolute budgets."
            )
        if node == "validation":
            return "Input readiness does not mean the resulting coil has passed its engineering checks."
        return ""

    def sections(self):
        return [
            {"key": key, "title": title, "help": help_, "summary": self.summary(key), "fields": self.entries(key)}
            for key, title, help_ in SECTIONS
        ]

    def changes(self):
        changes = {}
        for item in self.fields():
            key = item.key
            if key.startswith("setup_") or key not in self.values:
                continue
            value = self.values[key]
            if _same(value, self.initial.get(key)):
                continue
            if item.kind in {"number", "integer"}:
                try:
                    number = float(value)
                    if not math.isfinite(number) or (item.kind == "integer" and not number.is_integer()):
                        raise ValueError
                    value = int(number) if item.kind == "integer" else number
                except (TypeError, ValueError, OverflowError) as exc:
                    raise SetupInputError(
                        f"{item.label}: enter a finite {'whole number' if item.kind == 'integer' else 'number'}.",
                        key=key,
                        section=item.node,
                    ) from exc
            if item.kind == "bool" and not isinstance(value, bool):
                raise SetupInputError(f"{item.label}: choose on or off.", key=key, section=item.node)
            changes[key] = value
        return changes

    def build(self, *, extra_changes=None):
        """Return one validated project revision; failed edits leave the source untouched."""
        changes = self.changes()
        extras = dict(extra_changes or {})
        if "target_amplitude_native" in changes:
            changes["target_amplitude_native"] /= self.amplitude_scale
        for key in tuple(changes):
            if key in EXTRA_BY_KEY:
                extras.update(EXTRA_BY_KEY[key].set(changes.pop(key), self.project.config, self.project.study))
        if (
            self.project.config.research_physics is not None
            and changes.get("sf_opt_method", "constrained_quadratic") != "constrained_quadratic"
        ):
            raise SetupInputError(
                "This design has a physical objective and constraints. Edit those in Advanced parameters before selecting another solver method.",
                key="sf_opt_method",
                section="optimization",
            )
        profile = self.values["setup_profile"]
        profile_changed = profile != self.initial["setup_profile"]
        if profile == "round" and (profile_changed or "conductor_width_mm" in changes):
            diameter = float(self.values["conductor_width_mm"]) / 1000
            segments, equal_area = self.round_options or (16, False)
            extras.update(
                conductor_cross_section_width=diameter,
                conductor_cross_section_height=diameter,
                cross_sectional_points=round_wire_profile(diameter, segments, equal_area=equal_area),
            )
        elif profile_changed and profile == "rectangle":
            extras["cross_sectional_points"] = ((0.0, 0.0),)
        elif profile_changed and profile == "custom":
            raise SetupInputError(
                "Use the profile editor to define a custom cross-section.", key="setup_profile", section="conductor"
            )
        if not changes and not extras:
            return self.project
        # A fresh adapter prevents a failed second attempt from using a prior candidate.
        model = ProjectSettings(self.project)
        try:
            return model.apply_settings(changes, extras)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            message = str(exc)
            field = next((item for item in self.fields() if item.key in message), None)
            raise SetupInputError(message, key=field.key if field else "", section=field.node if field else "") from exc
