"""Explicit analytic target recipes shared by scripts and the workbench."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from numbers import Real

from ..target import _solid_harmonic_parameters


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
