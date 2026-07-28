from __future__ import annotations

from typing import Any

import numpy as np


PROTON_GYROMAGNETIC_RATIO_HZ_PER_T = 42.57747892e6


def normalize_field_units(units: Any) -> str:
    """Return the canonical magnetic-field unit name used by CoilGen."""

    text = str(units).strip().lower().replace(" ", "").replace("_", "")
    if text in {"t", "tesla"}:
        return "T"
    if text in {"hz", "hertz"}:
        return "Hz"
    raise ValueError("Magnetic field units must be 'T' or 'Hz'.")


def field_values_to_tesla(values: Any, units: Any) -> np.ndarray:
    """Convert scalar or vector B-field values from user units to tesla."""

    array = np.asarray(values, dtype=float)
    unit = normalize_field_units(units)
    if unit == "T":
        return array
    if unit == "Hz":
        return array / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
    raise AssertionError(unit)


def field_values_from_tesla(values: Any, units: Any) -> np.ndarray:
    """Convert scalar or vector B-field values from tesla to user units."""

    array = np.asarray(values, dtype=float)
    unit = normalize_field_units(units)
    if unit == "T":
        return array
    if unit == "Hz":
        return array * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
    raise AssertionError(unit)


def field_unit_label(units: Any, *, per_amp: bool = False, gradient: bool = False) -> str:
    unit = normalize_field_units(units)
    if gradient and per_amp:
        return f"{unit}/(A m)"
    if gradient:
        return f"{unit}/m"
    if per_amp:
        return f"{unit}/A"
    return unit


def config_target_field_units(config: Any) -> str:
    return normalize_field_units(getattr(config, "target_field_units", "T"))


def config_field_display_units(config: Any) -> str:
    return normalize_field_units(getattr(config, "field_display_units", "T"))
