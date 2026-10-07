"""Small deterministic proposal grids; physical sampling controls stay fixed."""

from __future__ import annotations


def validate_search_level(level):
    if level is not None and (isinstance(level, bool) or not isinstance(level, int) or level not in (0, 1)):
        raise ValueError("Curve search level must be 0, 1 or None for thorough search.")
    return level


def representative_values(values, level, *, initial=2, broader=5):
    """Choose spread-out proposal parameters before any curve is discretized.

    These are search proposals, never a bound on feasibility or field quality.
    Both extremes remain represented; the broader stage fills interior values.
    """
    values = tuple(values)
    count = len(values) if level is None else min(len(values), initial if level == 0 else broader)
    if count <= 1:
        return values[:count]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return tuple(values[index] for index in dict.fromkeys(indices))


def cylinder_trial_count(level):
    # Cubic profiles plus at most two compact directions times radius trials.
    return 96 if level is None else 8 if level == 0 else 28


def transition_trial_count(level):
    # Direct cubics, three-span B0 profiles, planar radii, cylinder base grid.
    return 216 if level is None else 28 if level == 0 else 80
