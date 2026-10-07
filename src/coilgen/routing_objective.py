"""Physical-length routing preferences; feasibility is checked independently."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingSearchCost:
    """Additive jump count and geometric guide, never a field certificate.

    Graph transitions attach to protected surface intervals, so their excursions
    are additive. The fabricated closed circuit is measured again before adoption.
    """

    jumps: int
    surrogate_m: float

    def __post_init__(self):
        if isinstance(self.jumps, bool) or int(self.jumps) != self.jumps or self.jumps < 0:
            raise ValueError("A routing jump count must be a nonnegative integer.")
        if np.isnan(self.surrogate_m) or self.surrogate_m < 0:
            raise ValueError("A routing guide must be nonnegative.")

    def __float__(self):
        return float(self.surrogate_m)

    def __add__(self, other):
        if isinstance(other, RoutingSearchCost):
            return RoutingSearchCost(self.jumps + other.jumps, self.surrogate_m + other.surrogate_m)
        return RoutingSearchCost(self.jumps, self.surrogate_m + float(other))

    __radd__ = __add__


def search_cost_key(cost):
    """Lexicographic ordering; scalar callers retain their length-only scope."""
    return getattr(cost, "jumps", 0), round(float(cost), 12)


def route_order_key(metrics):
    """One transitive key for all candidates, at 1 fT and 0.1 nm resolution.

    Pairwise epsilon comparisons are not transitive and can select different
    winners as enumeration order changes. Fixed quantization defines ties once.
    """
    jumps = metrics["jump_count"]
    error, length = float(metrics["longitudinal_rms_T"]), float(metrics["length_m"])
    if isinstance(jumps, bool) or int(jumps) != jumps or jumps < 0:
        raise ValueError("A measured nonnegative integer jump count is required.")
    if not np.isfinite(error) or not np.isfinite(length) or min(error, length) < 0:
        raise ValueError("Realized routing field error and length must be finite and nonnegative.")
    return int(jumps), round(error, 15), round(length, 10)


def route_is_better(candidate, reference):
    """Compare realized routes: jumps, fixed-current longitudinal RMS, length."""
    return route_order_key(candidate) < route_order_key(reference)


def routing_metrics(points, b0_direction, weight=1.0, *, mutable_segments=None):
    """Measure added routing without rewarding changes to field-producing turns.

    The transverse length is integral(1 - (t dot b0)**2) ds. It is invariant
    under current reversal, B0 reversal, rigid rotations and linear resampling.
    ``mutable_segments`` selects added connections for the directional penalty;
    total length always includes the complete path. Values have units of metres.
    This preference is never a substitute for field or finite-body validation.
    """
    points = np.asarray(points, dtype=float)
    direction = np.asarray(b0_direction, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Routing points must be finite with shape (n, 3).")
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("B0 direction must be a finite three-vector.")
    scale = float(np.max(np.abs(direction)))
    if scale == 0:
        raise ValueError("B0 direction must be nonzero.")
    direction = direction / scale
    direction /= np.linalg.norm(direction)
    if isinstance(weight, (bool, np.bool_)) or not np.isscalar(weight):
        raise ValueError("The B0 routing weight must be finite and nonnegative.")
    weight = float(weight)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("The B0 routing weight must be finite and nonnegative.")
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    if not np.all(np.isfinite(length)):
        raise ValueError("Routing interval lengths must be finite.")
    cosine = np.divide(delta @ direction, length, out=np.zeros_like(length), where=length > 0)
    transverse = length * np.clip(1.0 - cosine**2, 0.0, 1.0)
    if mutable_segments is not None:
        mutable = np.asarray(mutable_segments, dtype=bool)
        if mutable.shape != length.shape:
            raise ValueError("The mutable routing mask must match the path intervals.")
        transverse = transverse[mutable]
    total = float(length.sum())
    transverse_total = float(transverse.sum())
    return {
        "length_m": total,
        "transverse_length_m": transverse_total,
        "cost_m": total + weight * transverse_total,
    }


def routing_cost(points, b0_direction, weight=1.0):
    """Return length plus weighted transverse length for a mutable connection."""
    return routing_metrics(points, b0_direction, weight)["cost_m"]
