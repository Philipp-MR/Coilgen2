"""Realized routing preferences; feasibility is checked independently."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


REGULARITY_NAMES = (
    "added_backtracking_m",
    "added_attachment_turning_rad",
    "added_bending_energy",
    "added_join_bend_consistency",
    "added_excess_length_m",
    "added_turning_rad",
)


def routing_selection_policy(config):
    """Read the same canonical enum accepted by configuration validation.

    Objects predating the explicit setting retain the historical physical mode.
    Invalid values remain invalid; this getter never silently selects a policy.
    """
    return str(getattr(config, "joint_group_routing_selection_policy", "physical")).strip().lower().replace("-", "_")


@dataclass(frozen=True)
class RoutingSearchCost:
    """Additive jump count and geometric guide, never a field certificate.

    Graph transitions attach to protected surface intervals, so their excursions
    are additive. The fabricated closed circuit is measured again before adoption.
    """

    jumps: int
    surrogate_m: float
    regularity: tuple[float, ...] = ()

    def __post_init__(self):
        if isinstance(self.jumps, bool) or int(self.jumps) != self.jumps or self.jumps < 0:
            raise ValueError("A routing jump count must be a nonnegative integer.")
        if np.isnan(self.surrogate_m) or self.surrogate_m < 0:
            raise ValueError("A routing guide must be nonnegative.")
        if any(not np.isfinite(value) or value < 0 for value in self.regularity):
            raise ValueError("Routing regularity guides must be finite and nonnegative.")

    def __float__(self):
        return float(self.surrogate_m)

    def __add__(self, other):
        if isinstance(other, RoutingSearchCost):
            size = max(len(self.regularity), len(other.regularity))
            if self.regularity and other.regularity and len(self.regularity) != len(other.regularity):
                raise ValueError("Routing regularity guides must share one metric contract.")
            first = self.regularity or (0.0,) * size
            second = other.regularity or (0.0,) * size
            return RoutingSearchCost(
                self.jumps + other.jumps,
                self.surrogate_m + other.surrogate_m,
                tuple(a + b for a, b in zip(first, second)),
            )
        return RoutingSearchCost(self.jumps, self.surrogate_m + float(other), self.regularity)

    __radd__ = __add__


def search_cost_key(cost):
    """Lexicographic ordering; scalar callers retain their length-only scope."""
    return (
        getattr(cost, "jumps", 0),
        *(round(value, 10) for value in getattr(cost, "regularity", ())),
        round(float(cost), 12),
    )


def regularity_order_key(metrics):
    """A transitive geometric guide; neither field nor feasibility is implied.

    Missing attachment/energy values permit reading historical diagnostic
    records. Every new measured route supplies the complete metric contract.
    """
    values = tuple(float(metrics.get(name, 0.0)) for name in REGULARITY_NAMES)
    if not np.all(np.isfinite(values)) or min(values) < 0:
        raise ValueError("Measured added-route regularity must be finite and nonnegative.")
    return tuple(round(value, 10) for value in values)


def route_order_key(metrics, *, policy="physical"):
    """One transitive key per policy, at 1 fT / 0.1 nm / 1e-10 rad resolution.

    Pairwise epsilon comparisons are not transitive and can select different
    winners as enumeration order changes. Fixed quantization defines ties once.
    """
    jumps = metrics["jump_count"]
    error, length = float(metrics["longitudinal_rms_T"]), float(metrics["length_m"])
    if isinstance(jumps, bool) or int(jumps) != jumps or jumps < 0:
        raise ValueError("A measured nonnegative integer jump count is required.")
    if not np.isfinite(error) or not np.isfinite(length) or min(error, length) < 0:
        raise ValueError("Realized routing field error and length must be finite and nonnegative.")
    if policy == "physical":
        return int(jumps), round(error, 15), round(length, 10)
    if policy != "regular":
        raise ValueError("Routing selection policy must be 'physical' or 'regular'.")
    regular = regularity_order_key(metrics)
    if metrics.get("regularity_is_lower_bound", False):
        raise ValueError("Regular selection requires explicit added-route provenance.")
    return int(jumps), *regular, round(length, 10), round(error, 15)


def _turn_angle(first, second):
    return float(np.arctan2(np.linalg.norm(np.cross(first, second)), np.dot(first, second)))


def _direction_runs(delta, lengths):
    """Collapse only collinear subdivisions for a sampling-neutral bend guide."""
    directions, totals = [], []
    for value, length in zip(delta, lengths):
        direction = value / length
        if directions and _turn_angle(directions[-1], direction) <= 1e-12:
            totals[-1] += float(length)
        else:
            directions.append(direction)
            totals.append(float(length))
    return directions, totals


def join_bend_consistency(energies, groups):
    """Soft normalized-bend variance of joins belonging to the same group.

    Different group geometry, return routes and single joins are not coupled.
    A repeated family at another physical scale has the same descriptor. This
    preference neither assumes physical symmetry nor establishes feasibility.
    """
    if len(energies) != len(groups) or any(not np.isfinite(value) or value < 0 for value in energies):
        raise ValueError("Join bend descriptors require complete groups and finite nonnegative energy.")
    cohorts = {}
    for energy, group in zip(energies, groups):
        if group is not None:
            cohorts.setdefault(group, []).append(float(energy))
    score = 0.0
    for values in cohorts.values():
        if len(values) > 1:
            scale = max(1.0, max(values))
            normalized = np.asarray(values) / scale
            score += float(np.var(normalized) / ((1.0 / scale) ** 2 + np.mean(normalized) ** 2))
    return score


def transition_regularity(points, tangent_before=None, tangent_after=None):
    """Measure one added transition and its optional directed attachments."""
    points = np.asarray(points, dtype=float)
    sections, mask = [points], [True] * max(0, len(points) - 1)
    for tangent, before in ((tangent_before, True), (tangent_after, False)):
        if tangent is None:
            continue
        tangent = np.asarray(tangent, dtype=float)
        if tangent.shape != (3,) or not np.isfinite(tangent).all() or np.linalg.norm(tangent) == 0:
            raise ValueError("Transition attachments require finite nonzero directed tangents.")
        if before:
            sections.insert(0, (points[0] - tangent)[None])
            mask.insert(0, False)
        else:
            sections.append((points[-1] + tangent)[None])
            mask.append(False)
    return added_route_regularity(np.vstack(sections), mask)


def route_is_better(candidate, reference, *, policy="physical"):
    """Compare validated routes under one explicit, transitive policy.

    Regular selection requires a separately enforced, baseline-anchored field
    tolerance. This function does not grant field or manufacturing acceptance.
    """
    return route_order_key(candidate, policy=policy) < route_order_key(reference, policy=policy)


def added_route_regularity(points, mutable_segments, *, span_ids=None, closed=False):
    """Measure only contiguous added spans, without charging retained turns.

    Backtracking is half of total absolute chord-axis travel minus net travel;
    excess length is arc length minus chord. Total turning is the sum of angles
    between nonzero intervals. These measures are invariant to rigid motion,
    current reversal and subdivision of straight intervals. A closed added span
    has no chord direction: its backtracking is zero and its full length is
    counted as excess. Attachment angles measure added/protected and declared
    span interfaces once. Bending energy is L*sum(theta**2/dual_length) within
    each added span, an approximation to L*integral(kappa**2 ds). It measures
    concentration without changing units under geometric scale changes; exact
    collinear subdivisions are coalesced only for this diagnostic calculation.
    Unknown contour ownership must be excluded by the caller
    and reported as incomplete provenance, rather than scored as a connection.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Routing points must be finite with shape (n, 3).")
    mutable = np.asarray(mutable_segments, dtype=bool)
    if mutable.shape != (max(0, len(points) - 1),):
        raise ValueError("The mutable routing mask must match the path intervals.")
    identities = (None,) * len(mutable) if span_ids is None else tuple(span_ids)
    if len(identities) != len(mutable):
        raise ValueError("Added-span identities must match the path intervals.")
    spans = []
    for index, added in enumerate(mutable):
        if not added:
            continue
        if spans and spans[-1][-1] == index - 1 and identities[index] == identities[index - 1]:
            spans[-1].append(index)
        else:
            spans.append([index])
    # Closed circuits have no privileged storage seam. Join a split added span
    # across it before measuring the chord, so rotating the point record is neutral.
    if closed and len(spans) > 1 and mutable[0] and mutable[-1] and identities[0] == identities[-1]:
        spans[0] = spans.pop() + spans[0]
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    if not np.all(np.isfinite(length)):
        raise ValueError("Routing interval lengths must be finite.")
    backtracking = excess = turning = added_length = bending = attachment = maximum_attachment = 0.0
    join_energies, join_groups = [], []
    for indices in spans:
        ds = delta[indices]
        lengths = length[indices]
        ds = ds[lengths > 0]
        lengths = lengths[lengths > 0]
        if not len(ds):
            continue
        arc = float(lengths.sum())
        chord = ds.sum(axis=0)
        chord_length = float(np.linalg.norm(chord))
        added_length += arc
        excess += max(0.0, arc - chord_length)
        if chord_length > np.finfo(float).eps * arc * 16:
            axis = chord / chord_length
            backtracking += max(0.0, 0.5 * (float(np.abs(ds @ axis).sum()) - chord_length))
        tangent = ds / lengths[:, None]
        if len(tangent) > 1:
            # atan2 avoids acos amplifying roundoff at collinear subdivisions.
            turning += float(
                np.arctan2(
                    np.linalg.norm(np.cross(tangent[:-1], tangent[1:]), axis=1),
                    np.einsum("ij,ij->i", tangent[:-1], tangent[1:]),
                ).sum()
            )
        if closed and len(indices) == len(mutable) and len(tangent) > 1:
            turning += float(
                np.arctan2(np.linalg.norm(np.cross(tangent[-1], tangent[0])), np.dot(tangent[-1], tangent[0]))
            )
        run_tangents, run_lengths = _direction_runs(ds, lengths)
        cyclic = closed and len(indices) == len(mutable)
        if cyclic and len(run_tangents) > 1 and _turn_angle(run_tangents[-1], run_tangents[0]) <= 1e-12:
            run_lengths[0] += run_lengths.pop()
            run_tangents.pop()
        bend_pairs = list(zip(range(len(run_tangents) - 1), range(1, len(run_tangents))))
        if cyclic and len(run_tangents) > 1:
            bend_pairs.append((len(run_tangents) - 1, 0))
        span_energy = arc * sum(
            _turn_angle(run_tangents[a], run_tangents[b]) ** 2 / (0.5 * (run_lengths[a] + run_lengths[b]))
            for a, b in bend_pairs
        )
        bending += span_energy
        owner = identities[indices[0]]
        if isinstance(owner, tuple) and len(owner) == 6 and owner[-1] == "joint_spiral_join":
            join_energies.append(span_energy)
            join_groups.append(owner[1])
    # Measure each attachment boundary once, including the cyclic storage seam.
    nonzero = np.flatnonzero(length > 0)
    pairs = list(zip(nonzero[:-1], nonzero[1:]))
    if closed and len(nonzero) > 1:
        pairs.append((nonzero[-1], nonzero[0]))
    for before, after in pairs:
        if (mutable[before] or mutable[after]) and (
            mutable[before] != mutable[after] or identities[before] != identities[after]
        ):
            angle = _turn_angle(delta[before] / length[before], delta[after] / length[after])
            attachment += angle
            maximum_attachment = max(maximum_attachment, angle)
    return {
        "added_span_count": len(spans),
        "added_length_m": added_length,
        "added_backtracking_m": backtracking,
        "added_excess_length_m": excess,
        "added_turning_rad": turning,
        "added_attachment_turning_rad": attachment,
        "added_maximum_attachment_angle_rad": maximum_attachment,
        "added_bending_energy": bending,
        "added_join_bend_consistency": join_bend_consistency(join_energies, join_groups),
    }


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
