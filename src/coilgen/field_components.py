from __future__ import annotations

from typing import Any

import numpy as np


SCALAR_TARGET_COMPONENTS = ("Bx", "By", "Bz", "BdotB0")
TARGET_COMPONENTS = (*SCALAR_TARGET_COMPONENTS, "vector")
EVALUATION_COMPONENTS = (*TARGET_COMPONENTS, "Bmag")


def normalize_field_component(component: Any, *, allow_bmag: bool = False) -> str:
    key = str(component or "Bz").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "x": "Bx",
        "bx": "Bx",
        "b_x": "Bx",
        "y": "By",
        "by": "By",
        "b_y": "By",
        "z": "Bz",
        "bz": "Bz",
        "b_z": "Bz",
        "b0": "BdotB0",
        "b_dot_b0": "BdotB0",
        "bdotb0": "BdotB0",
        "b_parallel": "BdotB0",
        "parallel_b0": "BdotB0",
        "vector": "vector",
        "full_vector": "vector",
        "bvector": "vector",
        "bmag": "Bmag",
        "b_mag": "Bmag",
        "magnitude": "Bmag",
    }
    resolved = aliases.get(key)
    allowed = EVALUATION_COMPONENTS if allow_bmag else TARGET_COMPONENTS
    if resolved not in allowed:
        known = ", ".join(allowed)
        raise ValueError(f"Unknown field component {component!r}. Known components: {known}.")
    return resolved


def field_component_label(component: Any) -> str:
    resolved = normalize_field_component(component, allow_bmag=True)
    if resolved == "BdotB0":
        return "B dot B0"
    if resolved == "vector":
        return "full vector B"
    if resolved == "Bmag":
        return "|B|"
    return resolved


def field_component_values(field: Any, component: Any, *, b0_direction: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Return scalar/vector target rows selected from an ``(n, 3)`` field array."""

    resolved = normalize_field_component(component, allow_bmag=True)
    values = np.asarray(field, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("field must have shape (n, 3).")
    if resolved == "Bx":
        return values[:, 0]
    if resolved == "By":
        return values[:, 1]
    if resolved == "Bz":
        return values[:, 2]
    if resolved == "BdotB0":
        direction = normalized_b0_direction(b0_direction)
        return values @ direction
    if resolved == "Bmag":
        return np.linalg.norm(values, axis=1)
    if resolved == "vector":
        return values.reshape(-1)
    raise AssertionError(resolved)


def sensitivity_component_matrix(
    sensitivity_matrix: np.ndarray,
    component: Any,
    *,
    b0_direction: Any = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return target rows selected from a ``(3, n_points, n_unknowns)`` sensitivity matrix."""

    resolved = normalize_field_component(component)
    sensitivity = np.asarray(sensitivity_matrix, dtype=float)
    if sensitivity.ndim != 3 or sensitivity.shape[0] != 3:
        raise ValueError("sensitivity_matrix must have shape (3, n_points, n_unknowns).")
    if resolved == "Bx":
        return sensitivity[0]
    if resolved == "By":
        return sensitivity[1]
    if resolved == "Bz":
        return sensitivity[2]
    if resolved == "BdotB0":
        direction = normalized_b0_direction(b0_direction)
        return direction[0] * sensitivity[0] + direction[1] * sensitivity[1] + direction[2] * sensitivity[2]
    if resolved == "vector":
        return np.moveaxis(sensitivity, 0, 1).reshape(3 * sensitivity.shape[1], sensitivity.shape[2])
    raise AssertionError(resolved)


def component_sample_values(values: Any, component: Any) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    resolved = normalize_field_component(component, allow_bmag=True)
    if resolved == "vector":
        return np.repeat(array, 3)
    return array


def normalized_b0_direction(value: Any) -> np.ndarray:
    direction = np.asarray(value, dtype=float).reshape(-1)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("b0_direction must contain three finite values.")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-15:
        raise ValueError("b0_direction must be nonzero.")
    return direction / norm
