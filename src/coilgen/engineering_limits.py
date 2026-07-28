from __future__ import annotations

from typing import Any

import numpy as np


LEVEL_COUNT_METHODS = {"level_count", "levels", "contour_levels", "legacy", "offset_levels"}
ENGINEERING_METHODS = {
    "engineering",
    "current_spacing",
    "current_limited",
    "wire_current_spacing",
    "wire_current",
    "physical",
}


def contour_discretization_method(config: Any) -> str:
    raw = str(getattr(config, "contour_discretization_method", "level_count")).strip().lower().replace("-", "_")
    if raw in {"", "auto"}:
        return "engineering" if engineering_specs_present(config) else "level_count"
    if raw in ENGINEERING_METHODS:
        return "engineering"
    return "level_count"


def engineering_specs_present(config: Any) -> bool:
    return max_wire_current_A(config) > 0.0 and min_wire_distance_m(config) > 0.0


def engineering_contouring_enabled(config: Any) -> bool:
    return contour_discretization_method(config) == "engineering"


def max_wire_current_A(config: Any) -> float:
    value = _positive_float(getattr(config, "max_wire_current_A", 0.0))
    if value > 0.0:
        return value
    return _positive_float(getattr(config, "max_current_A", 0.0))


def min_wire_distance_m(config: Any) -> float:
    value = _positive_float(getattr(config, "min_wire_distance_m", 0.0))
    if value > 0.0:
        return value
    return _positive_float(getattr(config, "min_turn_distance", 0.0))


def engineering_contour_step_safety_factor(config: Any) -> float:
    value = _positive_float(getattr(config, "engineering_contour_step_safety_factor", 1.0), default=1.0)
    return min(max(value, np.finfo(float).eps), 1.0)


def engineering_max_contour_step_A(config: Any) -> float:
    return max_wire_current_A(config) * engineering_contour_step_safety_factor(config)


def explicit_surface_current_density_limit_A_per_m(config: Any) -> float:
    return _positive_float(getattr(config, "surface_current_density_limit_A_per_m", 0.0))


def engineering_surface_current_density_limit_A_per_m(config: Any) -> float:
    explicit = explicit_surface_current_density_limit_A_per_m(config)
    current = engineering_max_contour_step_A(config)
    distance = min_wire_distance_m(config)
    derived = current / distance if current > 0.0 and distance > 0.0 else 0.0
    positive = [value for value in (explicit, derived) if value > 0.0 and np.isfinite(value)]
    return min(positive) if positive else 0.0


def engineering_contour_max_levels(config: Any) -> int:
    try:
        return max(int(getattr(config, "engineering_contour_max_levels", 0)), 0)
    except (TypeError, ValueError):
        return 0


def engineering_limit_diagnostics(config: Any) -> dict[str, float | int | str | bool]:
    method = contour_discretization_method(config)
    current = max_wire_current_A(config)
    step = engineering_max_contour_step_A(config)
    distance = min_wire_distance_m(config)
    explicit_density = explicit_surface_current_density_limit_A_per_m(config)
    derived_density = step / distance if step > 0.0 and distance > 0.0 else 0.0
    density_limit = engineering_surface_current_density_limit_A_per_m(config)
    if explicit_density > 0.0 and derived_density > 0.0:
        density_source = "explicit" if explicit_density <= derived_density else "wire_current_and_spacing"
    elif explicit_density > 0.0:
        density_source = "explicit"
    elif derived_density > 0.0:
        density_source = "wire_current_and_spacing"
    else:
        density_source = "disabled"
    return {
        "contour_discretization_method": method,
        "engineering_contouring_enabled": bool(method == "engineering"),
        "max_wire_current_A": float(current),
        "engineering_contour_step_safety_factor": float(engineering_contour_step_safety_factor(config)),
        "engineering_max_contour_step_A": float(step),
        "min_wire_distance_m": float(distance),
        "surface_current_density_limit_A_per_m": float(explicit_density),
        "derived_surface_current_density_limit_A_per_m": float(derived_density),
        "engineering_surface_current_density_limit_A_per_m": float(density_limit),
        "engineering_surface_current_density_limit_source": density_source,
        "engineering_contour_max_levels": int(engineering_contour_max_levels(config)),
    }


def _positive_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number) or number <= 0.0:
        return float(default)
    return float(number)
