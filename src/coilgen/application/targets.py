"""Explicit analytic target recipes shared by scripts and the workbench."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import math
from numbers import Real

from ..target import _solid_harmonic_parameters


TARGET_NORMALIZATION_MODES = (
    (("auto", ""), "Automatic by design context"),
    (("off", "none", "disabled"), "Absolute field expression"),
    (("linear_fit", "fit", "gradient_fit", "linear"), "Fit a gradient amplitude"),
    (("scale", "multiplier"), "Multiply the field expression"),
    (("legacy", "legacy_extrema", "extrema", "max_min"), "Legacy extrema scaling"),
)


def target_normalization_choices(saved):
    """One choice per behavior, retaining the original spelling of the saved mode."""
    key = str(saved).strip().lower().replace("-", "_")
    choices = tuple((saved if key in names else names[0], label) for names, label in TARGET_NORMALIZATION_MODES)
    if saved not in {value for value, _label in choices}:
        return ((saved, str(saved)), *choices)
    return choices


@dataclass(frozen=True, slots=True)
class SolidHarmonicTarget:
    """One Bz mode in the normalized real solid-Ylm convention.

    ``coefficient_T`` multiplies (r/reference_radius_m)**degree * real_Ylm,
    including the existing Condon--Shortley phase. It is not a pole amplitude.
    Center coordinates use the world frame. The reference radius is explicit
    and does not change when the optimization or validation grid changes.
    """

    degree: int
    order: int
    coefficient_T: float
    reference_radius_m: float
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        degree, order, radius, center = _solid_harmonic_parameters(
            self.degree, self.order, self.reference_radius_m, self.center_m
        )
        if (
            isinstance(self.coefficient_T, bool)
            or not isinstance(self.coefficient_T, Real)
            or not math.isfinite(self.coefficient_T)
        ):
            raise ValueError("Solid harmonic coefficient must be finite [T].")
        for key, value in (
            ("degree", degree),
            ("order", order),
            ("reference_radius_m", radius),
            ("center_m", center),
            ("coefficient_T", float(self.coefficient_T)),
        ):
            object.__setattr__(self, key, value)

    @property
    def expression(self) -> str:
        geometry = ",".join(repr(value) for value in (self.reference_radius_m, *self.center_m))
        return f"{self.coefficient_T!r}*solid_harmonic({self.degree},{self.order},x,y,z,{geometry})"

    def to_config_overrides(self) -> dict:
        """Use absolute tesla without the gradient/shim amplitude rescaling.

        Clear a previously imported target map, which otherwise takes priority
        over expressions. Existing ROI geometry and suppression regions remain
        explicit caller choices.
        """
        return {
            "field_shape_function": self.expression,
            "target_field_definition_file": "none",
            "target_gradient_normalization": "off",
            "target_field_units": "T",
            "target_field_component": "Bz",
            "design_context": "mr_shim",
        }


def solid_harmonic_target(
    degree: int,
    order: int,
    coefficient_T: float,
    reference_radius_m: float,
    center_m=(0.0, 0.0, 0.0),
) -> SolidHarmonicTarget:
    return SolidHarmonicTarget(degree, order, coefficient_T, reference_radius_m, center_m)


ORIGIN_KEYS = tuple(f"field_origin_{axis}_mm" for axis in "xyz")


def translated_expression(original, values):
    """Merge world-mm controls with an editable expression in one transaction."""
    expression = values.get("field_shape_function", original)
    previous = recognize_solid_harmonic_target(original)
    if previous is None:
        return expression
    center = list(previous.center_m)
    changed = {}
    for index, key in enumerate(ORIGIN_KEYS):
        if key in values and float(values[key]) != center[index] * 1000:
            changed[index] = float(values[key]) / 1000
    if not changed:
        return expression
    target = recognize_solid_harmonic_target(expression)
    if target is None:
        raise ValueError("Apply the new expression separately from the field-pattern origin.")
    center = list(target.center_m)
    for index, value in changed.items():
        center[index] = value
    return replace(target, center_m=tuple(center)).expression


def target_probe_text(config):
    """Describe world probes for a recognized absolute expression; ROI placement is independent."""
    import numpy as np
    from ..target import evaluate_field_expression, _target_gradient_normalization_method

    target = recognize_solid_harmonic_target(config.field_shape_function)
    if target is None or str(config.target_field_definition_file).lower() not in {"", "none"}:
        return ""

    def world(xyz):
        return "(" + ", ".join(f"{1000 * v:g}" for v in xyz) + ") mm"

    region = config.target_region_geometry or {}
    if region.get("shape") == "enclosing_mesh" or (
        not region and str(config.target_mesh_file).strip().lower() not in {"", "none"}
    ):
        roi = "defined by the imported mesh in world coordinates"
    elif not region and config.set_roi_into_mesh_center:
        roi = "support mesh center"
    else:
        roi = world(region.get("center_m", (0, 0, 0)))
    line = f"World coordinates · Field-pattern origin {world(target.center_m)} · ROI center {roi}."
    if (
        config.target_field_units == "T"
        and config.target_field_component == "Bz"
        and _target_gradient_normalization_method(config, source="expression") == "none"
    ):
        radius = target.reference_radius_m
        points = np.array([[-radius, 0, 0], [0, 0, 0], [radius, 0, 0], [0, 0, radius]])
        values = evaluate_field_expression(target.expression, points) * 1e6
        line += " Bz [µT] at " + " · ".join(f"{world(p)} = {v:.4g}" for p, v in zip(points, values))
    return line


def field_budget_text(config, margin):
    from ..optimization import _normalise_target_error_constraint_metric

    ratio = float(margin)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        return "Enter a design / validation ratio greater than zero and at most one."
    metric = _normalise_target_error_constraint_metric(config)
    if (config.research_physics or {}).get("include_default_field_constraints") is False:
        return "Custom field constraints · Independent verification requires a matching validation operator or dataset."
    if metric == "std":
        design = (
            f"{config.target_error_std_limit_T * 1e6:g} µT"
            if config.target_error_std_limit_T > 0
            else f"{config.target_error_tolerance_percent:g}% of the weighted target standard deviation"
            + (
                f" + {config.target_error_tolerance_absolute_T * 1e6:g} µT"
                if config.target_error_tolerance_absolute_T
                else ""
            )
        )
        return f"Design residual standard deviation limit · {design}. Independent standard deviation check unavailable."
    if metric == "rms" and config.target_error_rms_limit_T > 0:
        limit = config.target_error_rms_limit_T * 1e6
        return f"Absolute RMS budget per component · Design {limit:g} µT · Independent {limit / ratio:g} µT."
    limit = config.target_error_tolerance_percent
    text = (
        f"Relative RMS budget · Design {limit:g}% · Independent {limit / ratio:g}% · "
        "100 × weighted RMS(field − target) / weighted RMS(target)."
    )
    if config.target_error_tolerance_absolute_T:
        text += (
            f" An additional {config.target_error_tolerance_absolute_T:g} T absolute {metric} term "
            "enters the design limit before the validation ratio is applied."
        )
    return text


def result_field_scope(meta):
    """Report counts and limits from the saved record, including historical partial results."""
    validation = meta.get("validation") or {}
    wires = meta.get("wire_fields") or {}
    wire = next((wires[k] for k in ("routed", "contour") if (wires.get(k) or {}).get("available")), {})
    fixed = wire.get("at_stream_current") or {}
    if fixed.get("available"):
        wire = fixed
    parts = []
    relative = validation.get("relative_limit")
    margin = validation.get("design_margin")
    if validation.get("contract_kind") == "relative_rms" and relative is not None and margin:
        parts.append(
            f"Stream function RMS limits · Design {relative * margin * 100:g}% · Independent {relative * 100:g}%"
        )
    elif validation.get("limit") is not None:
        parts.append(
            f"Independent {validation.get('contract_kind', 'field')} limit {validation['limit']:.4g} {validation.get('unit', '')}"
        )
    if wire.get("sample_count"):
        parts.append(f"Design grid {wire['sample_count']} samples")
    if validation.get("point_count"):
        parts.append(f"Independent grid {validation['point_count']} samples")
    if wire.get("current_A") is not None and wire.get("current_refitted") is False:
        parts.append(f"Wire fields at original current {wire['current_A']:.5g} A · best-fit current is diagnostic")
    return " · ".join(parts)


def recognize_solid_harmonic_target(expression: str) -> SolidHarmonicTarget | None:
    """Recognize a literal recipe without evaluating arbitrary Python.

    Only coefficient*solid_harmonic(l,m,x,y,z,R,cx,cy,cz) is recognized.
    Other analytic expressions remain ordinary editable expression targets.
    """
    if not isinstance(expression, str) or len(expression) > 4096:
        return None
    try:
        node = ast.parse(expression.strip(), mode="eval").body
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            return None
        call = node.right
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "solid_harmonic"
            or call.keywords
            or len(call.args) != 9
            or any(not isinstance(value, ast.Name) or value.id != name for value, name in zip(call.args[2:5], "xyz"))
        ):
            return None
        coefficient = ast.literal_eval(node.left)
        degree, order = (ast.literal_eval(value) for value in call.args[:2])
        radius, *center = (ast.literal_eval(value) for value in call.args[5:])
        return SolidHarmonicTarget(degree, order, coefficient, radius, tuple(center))
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
