"""Bounded alternative cut contexts, leaving the manufactured baseline intact."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import pickle

import numpy as np

from .layout import _coherent_axial_ring_group_indices, _force_cut_selection_for_group, interconnect_within_groups
from .models import CoilPart


@dataclass
class JointCutVariants:
    variants: list[CoilPart] = field(default_factory=list)
    reasons: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)


def _isolated_context(part):
    """Copy object state while sharing large numerical buffers read-only."""
    memo, visited = {}, set()

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, np.ndarray):
            view = value.view()
            view.flags.writeable = False
            memo[id(value)] = view
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                visit(item)

    visit(part)
    return deepcopy(part, memo)


def _full_contour_signature(part):
    digest = hashlib.sha256()
    for loops in (part.contour_lines, *(group.loops for group in part.groups)):
        digest.update(str(len(loops)).encode())
        for loop in loops:
            # Serialize each contour independently: copies may intentionally
            # break aliases between contour_lines and a group's source loops.
            for name, value in vars(loop).items():
                digest.update(name.encode())
                if isinstance(value, np.ndarray):
                    digest.update(str((value.dtype.str, value.shape)).encode())
                    digest.update(value.tobytes())
                else:
                    digest.update(pickle.dumps(value, protocol=5))
    return digest.digest()


def _cut_signature(part):
    return tuple(tuple((loop.v.shape, loop.v.tobytes()) for loop in group.opened_loops) for group in part.groups)


def _baseline_signature(part):
    return hashlib.sha256(
        pickle.dumps(
            (
                part.mesh.vertices,
                part.mesh.faces,
                part.wire_path,
                part.fabrication_path,
                part.swept_body,
                part.swept_bodies,
                part.manufacturing_diagnostics,
                part.normal_shift_provenance,
                part.normal_shift_diagnostics,
                part.shift_array,
                part.points_to_shift,
                part.field_by_layout,
            ),
            protocol=5,
        )
    ).digest()


def _opened_contour_follows_source(source, opened):
    """Verify one forward contiguous arc, including interpolated cut endpoints."""
    if (source.potential, source.current_orientation, source.turn_multiplicity) != (
        opened.potential,
        opened.current_orientation,
        opened.turn_multiplicity,
    ):
        return False
    full, points = np.asarray(source.v), np.asarray(opened.v)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
        return False
    if full.ndim != 2 or full.shape[1] != 3 or len(full) < 4 or not np.all(np.isfinite(full)):
        return False
    if not np.array_equal(full[0], full[-1]):
        full = np.vstack((full, full[:1]))  # Full contours may store the seam implicitly.
    lengths = np.linalg.norm(np.diff(full, axis=0), axis=1)
    usable = lengths > 1e-14
    starts, directions, lengths = full[:-1][usable], np.diff(full, axis=0)[usable], lengths[usable]
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    perimeter = cumulative[-1]
    if not np.isfinite(perimeter) or perimeter <= 1e-12:
        return False
    tolerance = max(1e-12, perimeter * 1e-10)
    exact = {tuple(point): distance for point, distance in zip(starts, cumulative[:-1])}
    if len(exact) != len(starts):
        return False  # Ambiguous repeated vertices do not define a unique arc.
    positions = []
    for point in points:
        position = exact.get(tuple(point))
        if position is None:
            fraction = np.clip(np.einsum("ij,ij->i", point - starts, directions) / lengths**2, 0, 1)
            errors = np.linalg.norm(starts + fraction[:, None] * directions - point, axis=1)
            nearest = int(np.argmin(errors))
            if errors[nearest] > tolerance:
                return False
            position = cumulative[nearest] + fraction[nearest] * lengths[nearest]
        positions.append(position)
    arc = np.mod(np.diff(positions), perimeter)
    actual = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return bool(
        np.all(actual > 1e-14)
        and np.all(np.abs(arc - actual) <= tolerance)
        and float(arc.sum()) < perimeter - tolerance
    )


def _context_reasons(original, context, changed):
    if _full_contour_signature(context) != _full_contour_signature(original):
        return ("original_full_contours_changed",)
    if len(context.connected_groups) != len(context.groups):
        return ("incomplete_within_group_reconstruction",)
    if any(
        (before.loop_indices, before.outer_loop_indices) != (after.loop_indices, after.outer_loop_indices)
        for before, after in zip(original.groups, context.groups)
    ):
        return ("source_group_membership_changed",)
    for index in changed:
        group = context.groups[index]
        if len(group.opened_loops) != len(group.loops):
            return ("incomplete_opened_contours",)
        if any(
            not _opened_contour_follows_source(full, opened) for full, opened in zip(group.loops, group.opened_loops)
        ):
            return ("opened_contour_direction_or_geometry_changed",)
        connection = context.connected_groups[index]
        if any(
            len(points) < 2 or not np.all(np.isfinite(points))
            for points in (connection.v, connection.spiral_in_v, connection.spiral_out_v, connection.return_path_v)
        ):
            return ("invalid_within_group_connection",)
    return ()


def prepare_joint_group_context(part, config):
    """Expose directed ports on singleton groups in an isolated cut context.

    A sole closed contour already is a complete circuit and is left intact.
    In a multi-group circuit, a singleton needs one declared opening just as
    each turn in a larger group does. Only that forward contour arc becomes
    protected; the complete source loop and manufactured baseline stay intact.
    Callers must use the returned context throughout proposal and validation.
    """
    if len(part.groups) <= 1:
        return part
    if len(part.connected_groups) != len(part.groups):
        raise ValueError("missing_group_context")
    changed = [
        index
        for index, (group, connected) in enumerate(zip(part.groups, part.connected_groups))
        if len(group.loops) == 1 and len(connected.return_path_v) < 2
    ]
    if not changed:
        return part
    if any(part.groups[index].loops[0].turn_multiplicity != 1 for index in changed):
        raise ValueError("invalid_or_unexpanded_full_contours")
    context = _isolated_context(part)
    baseline_signature = _baseline_signature(context)
    original_groups, original_connections = list(context.groups), list(context.connected_groups)
    context.groups = deepcopy(context.groups)
    interconnect_within_groups([context], replace(config, progress_flag=False), open_singletons=True)
    for index in range(len(context.groups)):
        if index not in changed:
            context.groups[index] = original_groups[index]
            context.connected_groups[index] = original_connections[index]
    reasons = _context_reasons(part, context, changed)
    if _baseline_signature(context) != baseline_signature:
        reasons = (*reasons, "baseline_geometry_or_state_changed")
    if reasons:
        raise ValueError("; ".join(reasons))
    context.topology_diagnostics["joint_singleton_cut_group_indices"] = changed
    return context


def propose_joint_cut_variants(part, config, *, maximum_candidates=2):
    """Propose at most two alternate high/low rail contexts without adoption.

    First flip every unforced group opening, then flip the worst-score
    free group. Axial ring groups retain their common automatic side; a forced
    member pins that side, otherwise the cohort flips together. Unchanged
    groups, forced cuts, full directed contours, support,
    baseline wire and physical body are retained. Only production within-group
    cut/rail generation runs; there is no shift, sweep or field recomputation.
    Every returned context still requires complete routing and validation.
    """
    if isinstance(maximum_candidates, bool) or not isinstance(maximum_candidates, int) or maximum_candidates < 0:
        raise ValueError("maximum_candidates must be a nonnegative integer.")
    result = JointCutVariants(
        diagnostics={
            "bounded_high_low_search": True,
            "global_search_exhaustive": False,
            "manufacturing_feasibility_proven": False,
            "maximum_candidates": min(maximum_candidates, 2),
            "attempts": [],
            "variants": [],
        }
    )
    if maximum_candidates == 0:
        result.reasons = ("cut_candidate_budget_zero",)
        return result
    if not getattr(config, "joint_group_routing_search_cuts", True):
        result.reasons = ("cut_search_disabled",)
        return result
    count = len(part.groups)
    if not count or count != len(part.connected_groups):
        result.reasons = ("missing_group_context",)
        return result
    try:
        part = prepare_joint_group_context(part, config)
    except (ValueError, FloatingPointError) as error:
        result.reasons = ("singleton_cut_context_unavailable",)
        result.diagnostics["detail"] = str(error)
        return result
    singleton_cuts = count > 1 and any(len(group.loops) == 1 for group in part.groups)
    reconstruction_options = {"open_singletons": True} if singleton_cuts else {}
    forced = [_force_cut_selection_for_group(config, index) for index in range(count)]
    free = [
        index
        for index, group in enumerate(part.groups)
        if forced[index] is None and (len(group.loops) > 1 or singleton_cuts and len(group.loops) == 1)
    ]
    coupled = set(_coherent_axial_ring_group_indices(part, config))
    pinned = any(forced[index] is not None for index in coupled)
    if pinned:
        free = [index for index in free if index not in coupled]
    result.diagnostics["forced_group_indices"] = [index for index, side in enumerate(forced) if side is not None]
    result.diagnostics["coupled_axial_group_indices"] = sorted(coupled)
    result.diagnostics["coupled_axial_side_pinned"] = pinned
    result.diagnostics["free_group_indices"] = free
    if not free:
        result.reasons = ("no_unforced_multi_loop_groups",)
        return result
    if any(not group.loops or any(loop.turn_multiplicity != 1 for loop in group.loops) for group in part.groups):
        result.reasons = ("invalid_or_unexpanded_full_contours",)
        return result
    sides = [group.within_group_cut_side.lower() for group in part.groups]
    if any(sides[index] not in {"high", "low"} for index in free):
        canonical = _isolated_context(part)
        try:
            interconnect_within_groups([canonical], replace(config, progress_flag=False), **reconstruction_options)
        except (ValueError, FloatingPointError) as error:
            result.reasons = ("baseline_cut_side_unavailable",)
            result.diagnostics["detail"] = str(error)
            return result
        for index in free:
            if sides[index] not in {"high", "low"}:
                sides[index] = canonical.groups[index].within_group_cut_side.lower()
        result.diagnostics["canonical_side_inference"] = True
    free = [index for index in free if sides[index] in {"high", "low"}]
    if not coupled.issubset(free):
        free = [index for index in free if index not in coupled]
    if not free:
        result.reasons = ("baseline_cut_side_unavailable",)
        return result
    if coupled.issubset(free) and len({sides[index] for index in coupled}) > 1:
        result.reasons = ("incoherent_axial_cut_baseline",)
        return result

    def priority(index):
        score = float(part.groups[index].within_group_cut_score)
        return (score if np.isfinite(score) else np.inf, -index)

    worst = max(free, key=priority)
    plans = [tuple(free)]
    if len(free) > 1:
        worst_plan = tuple(sorted(coupled)) if worst in coupled else (worst,)
        if worst_plan not in plans:
            plans.append(worst_plan)
    seen = {_cut_signature(part)}
    rejected = set()
    for changed in plans[:maximum_candidates]:
        choices = [
            forced[index] or (sides[index] if sides[index] in {"high", "low"} else "auto") for index in range(count)
        ]
        for index in changed:
            choices[index] = "low" if sides[index] == "high" else "high"
        context = _isolated_context(part)
        baseline_signature = _baseline_signature(context)
        original_groups, original_connections = list(context.groups), list(context.connected_groups)
        # Production mutates each group's opening metadata, so isolate those
        # groups again before restoring every untouched/forced baseline group.
        context.groups = deepcopy(context.groups)
        attempt = {"changed_group_indices": list(changed), "requested_sides": choices}
        try:
            interconnect_within_groups(
                [context],
                replace(config, force_cut_selection=tuple(choices), progress_flag=False),
                **reconstruction_options,
            )
            for index in range(count):
                if index not in changed:
                    context.groups[index] = original_groups[index]
                    context.connected_groups[index] = original_connections[index]
            reasons = _context_reasons(part, context, changed)
            if _baseline_signature(context) != baseline_signature:
                reasons = (*reasons, "baseline_geometry_or_state_changed")
            if any(context.groups[index].within_group_cut_side.lower() != choices[index] for index in changed):
                reasons = (*reasons, "requested_cut_side_unavailable")
        except (ValueError, FloatingPointError) as error:
            reasons = ("within_group_reconstruction_failed",)
            attempt["detail"] = str(error)
        if not reasons and _cut_signature(context) in seen:
            reasons = ("duplicate_cut_geometry",)
        attempt["reasons"] = list(reasons)
        result.diagnostics["attempts"].append(attempt)
        rejected.update(reasons)
        if reasons:
            continue
        seen.add(_cut_signature(context))
        result.variants.append(context)
        result.diagnostics["variants"].append(
            {
                "name": "opposite_free_rails" if len(changed) > 1 else f"opposite_group_{changed[0]}",
                "changed_group_indices": list(changed),
                "selected_sides": [group.within_group_cut_side for group in context.groups],
            }
        )
    result.diagnostics["candidate_count"] = len(result.variants)
    result.reasons = () if result.variants else tuple(sorted(rejected or {"no_alternative_cut_context"}))
    return result
