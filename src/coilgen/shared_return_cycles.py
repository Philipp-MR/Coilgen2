"""Directed two-group cycle seeds with separated, trimmed return ports.

These are proposals only.  Original winding intervals remain exact and the
caller must still shape, field-screen and physically certify the entire cycle.
"""

from __future__ import annotations

import numpy as np

from .group_routing import GroupRouteCandidate, GroupRouteProposals, _edge_key, _protected_group_segments
from .models import InterGroupCut, WirePath, WirePathGeometry, WireSegmentOwner


def _clip_return(points, first, last):
    """Keep a directed arclength interval, retaining every original interior vertex."""
    points = np.asarray(points, dtype=float)
    if not 0 <= first < last <= 1 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("Invalid directed return interval.")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    if cumulative[-1] <= 1e-12:
        raise ValueError("Degenerate return rail.")
    low, high = np.asarray((first, last)) * cumulative[-1]

    def point_at(position):
        if position == 0:
            return points[0].copy()
        if position == cumulative[-1]:
            return points[-1].copy()
        index = min(int(np.searchsorted(cumulative, position, side="right") - 1), len(lengths) - 1)
        return points[index] + (position - cumulative[index]) / lengths[index] * (points[index + 1] - points[index])

    interior = points[(cumulative > low) & (cumulative < high)]
    return np.vstack((point_at(low), interior, point_at(high)))


def _bridge_pair_distance(first, second):
    """Exact finite-segment distance; no inference from a support development."""
    from .geometry_kernels import segment_distances

    first, second = np.asarray(first), np.asarray(second)
    if min(len(first), len(second)) < 2:
        return float("inf")
    minimum = float("inf")
    for a, b in zip(first[:-1], first[1:]):
        minimum = min(
            minimum,
            float(
                np.min(
                    segment_distances(
                        np.broadcast_to(a, second[:-1].shape),
                        np.broadcast_to(b - a, second[:-1].shape),
                        second[:-1],
                        np.diff(second, axis=0),
                    )
                )
            ),
        )
    return minimum


def propose_shared_return_cycles(part, config, *, maximum_candidates=24):
    """Open both old return rails before fusing two directed winding groups.

    The historical full-return construction revisits a spiral terminal: a
    bridge enters that terminal and a later return leaves the identical point.
    Here one rail retains a prefix and the other a suffix, so both new bridges
    have separate ports.  Neither contour direction nor a protected interval is
    reversed.  This two-group family complements the general cyclic search.
    """
    from .layout import _group_route_variants, _inter_group_connector_result, _strict_geodesic_inter_group_paths
    from .layout import _use_geodesic_inter_group_paths
    from .surface_distance import SurfaceGeodesicSolver

    result = GroupRouteProposals(
        diagnostics={
            "family": "separated_trimmed_return_ports",
            "two_group_family": True,
            "manufacturing_feasibility_proven": False,
            "physical_feasibility_proven": False,
            "attempted_count": 0,
            "bridge_pair_intersection_rejected_count": 0,
        }
    )
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or not 1 <= maximum_candidates <= 64
    ):
        raise ValueError("Shared-return candidate count must be an integer in [1, 64].")
    if len(part.groups) != 2 or len(part.connected_groups) != 2:
        result.reason = "shared_return_requires_two_groups"
        return result
    try:
        protected, lookup = _protected_group_segments(part)
    except ValueError as error:
        result.reason = "shared_return_protected_geometry_unavailable"
        result.diagnostics["detail"] = str(error)
        return result
    solver = SurfaceGeodesicSolver(part.mesh, refine_routes=True) if _use_geodesic_inter_group_paths(config) else None
    allow_direct = not _strict_geodesic_inter_group_paths(config)
    variants = [_group_route_variants(group) for group in part.connected_groups]
    ranked, seen = [], set()
    for first_index, second_index in ((0, 1), (1, 0)):
        for first in variants[first_index]:
            for second in variants[second_index]:
                first_spiral, second_spiral = np.asarray(first.spiral_v), np.asarray(second.spiral_v)
                first_ids = tuple(lookup.get(_edge_key(a, b)) for a, b in zip(first_spiral[:-1], first_spiral[1:]))
                second_ids = tuple(lookup.get(_edge_key(a, b)) for a, b in zip(second_spiral[:-1], second_spiral[1:]))
                actual = [identity for identity in (*first_ids, *second_ids) if identity is not None]
                if len(actual) != len(protected) or set(actual) != set(protected):
                    continue
                for first_fraction, second_fraction in (
                    (0.25, 0.25),
                    (0.5, 0.5),
                    (0.75, 0.75),
                    (0.25, 0.75),
                    (0.75, 0.25),
                ):
                    result.diagnostics["attempted_count"] += 1
                    try:
                        first_return = _clip_return(first.return_path_v, first_fraction, 1.0)
                        second_return = _clip_return(second.return_path_v, 0.0, second_fraction)
                    except ValueError:
                        continue
                    sections, identities, owners = [], [], []

                    def append(points, owner, ids=None):
                        points = np.asarray(points, dtype=float)
                        if len(points) < 2:
                            return
                        if sections and np.linalg.norm(sections[-1][-1] - points[0]) > 1e-10:
                            raise ValueError("Discontinuous shared-return section.")
                        sections.append(points)
                        labels = (None,) * (len(points) - 1) if ids is None else ids
                        identities.extend(labels)
                        owners.extend(
                            WireSegmentOwner("contour_loop", group_index=identity[0], loop_index=identity[1])
                            if identity is not None
                            else owner
                            for identity in labels
                        )

                    def bridge(a, b, role):
                        if np.linalg.norm(a - b) <= 1e-12:
                            return np.asarray([a])
                        route = _inter_group_connector_result(a, b, solver, allow_direct_fallback=allow_direct)
                        if len(route.path_v) < 2:
                            raise ValueError("Unsupported shared-return connector.")
                        if role == "primary":
                            owner = WireSegmentOwner(
                                "inter_group_connector",
                                connector_index=0,
                                group1_index=first_index,
                                group2_index=second_index,
                                connector_role="joint_return",
                            )
                        elif role == "closing":
                            owner = WireSegmentOwner(
                                "inter_group_connector",
                                connector_index=1,
                                group1_index=second_index,
                                group2_index=first_index,
                                connector_role="joint_return",
                            )
                        else:
                            group_index = second_index if role == "second_return_approach" else first_index
                            owner = WireSegmentOwner("within_group_return", group_index=group_index)
                        append(route.path_v, owner)
                        return route.path_v

                    try:
                        append(
                            first_spiral,
                            WireSegmentOwner(
                                "contour_group", group_index=first_index, connector_role="joint_spiral_join"
                            ),
                            first_ids,
                        )
                        spiral_bridge = bridge(first_spiral[-1], second_spiral[0], "primary")
                        append(
                            second_spiral,
                            WireSegmentOwner(
                                "contour_group", group_index=second_index, connector_role="joint_spiral_join"
                            ),
                            second_ids,
                        )
                        bridge(second_spiral[-1], second_return[0], "second_return_approach")
                        append(second_return, WireSegmentOwner("within_group_return", group_index=second_index))
                        return_bridge = bridge(second_return[-1], first_return[0], "closing")
                        append(first_return, WireSegmentOwner("within_group_return", group_index=first_index))
                        bridge(first_return[-1], first_spiral[0], "first_return_approach")
                    except ValueError:
                        continue
                    pair_distance = _bridge_pair_distance(spiral_bridge, return_bridge)
                    if pair_distance <= 1e-9:
                        result.diagnostics["bridge_pair_intersection_rejected_count"] += 1
                        continue
                    points = np.vstack((sections[0], *(section[1:] for section in sections[1:])))
                    if np.linalg.norm(points[0] - points[-1]) > 1e-10:
                        continue
                    key = points.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                    cuts = [
                        InterGroupCut(
                            cut1_v=first_spiral[-2:].copy(),
                            cut2_v=second_spiral[:2].copy(),
                            connection_path_v=spiral_bridge.copy(),
                            group1_index=first_index,
                            group2_index=second_index,
                            is_return_connection=True,
                            selected_point1_v=spiral_bridge[:1].copy(),
                            selected_point2_v=spiral_bridge[-1:].copy(),
                        ),
                        InterGroupCut(
                            cut1_v=second_return[-2:].copy(),
                            cut2_v=first_return[:2].copy(),
                            connection_path_v=return_bridge.copy(),
                            group1_index=second_index,
                            group2_index=first_index,
                            is_return_connection=True,
                            selected_point1_v=return_bridge[:1].copy(),
                            selected_point2_v=return_bridge[-1:].copy(),
                        ),
                    ]
                    candidate = GroupRouteCandidate(
                        WirePath(points, owners, [WirePathGeometry.SURFACE] * len(owners)),
                        tuple(identities),
                        protected,
                        (first_index, second_index),
                        (
                            f"{first.name}:trim_return_{first_fraction:g}",
                            f"{second.name}:trim_return_{second_fraction:g}",
                        ),
                        cuts,
                        length,
                        float(
                            np.linalg.norm(np.diff(first_return, axis=0), axis=1).sum()
                            + np.linalg.norm(np.diff(second_return, axis=0), axis=1).sum()
                        ),
                    )
                    candidate.shared_return_seed_diagnostics = {
                        "bridge_pair_distance_m": pair_distance,
                        "return_fractions": [first_fraction, second_fraction],
                    }
                    ranked.append(((-min(pair_distance, 0.01), length, key), candidate))
    ranked.sort(key=lambda item: item[0])
    result.candidates = [candidate for _, candidate in ranked[:maximum_candidates]]
    result.reason = "candidates_available" if result.candidates else "no_separated_shared_return_cycle"
    result.diagnostics.update(
        candidate_count=len(result.candidates), shortlist_truncated=len(ranked) > maximum_candidates
    )
    return result
