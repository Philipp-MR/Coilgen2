"""Shape complete return-to-group transitions without moving winding turns."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, replace
import hashlib
from time import perf_counter

import numpy as np

from .connector_bridge_validation import _validate_support
from .connector_bridges import BridgeGeneration
from .faceted_support import faceted_cylinder_support
from .joint_curve_search import transition_trial_count
from .group_routing import GroupRouteProposals, RoutingEvaluationBudgetExceeded, _length, _protected_group_segments
from .geometry_collision import _segment_bvh_candidate_pairs
from .joint_transition_curves import (
    _DistanceBudgetExceeded,
    _SegmentDistanceIndex,
    AXIAL_RETURN_SPAN_TRIALS,
    generate_axial_return_curves,
    generate_joint_corridor_curves,
    generate_joint_transition_curves,
)
from .joint_surface_curves import PLANAR_RETURN_TRIALS, generate_planar_return_curves
from .joint_cylinder_curves import CYLINDER_TRANSITION_TRIALS, _cylinder_chart, generate_cylindrical_transition_curves
from .manufacturing import _SurfaceProjector, _cross_section_envelope_radius, conductor_cross_section_points
from .models import WirePath, WirePathGeometry
from .routing_objective import RoutingSearchCost, routing_cost
from .routing_jumps import raised_span_metrics


_TRANSITION_TRIALS = 64 + AXIAL_RETURN_SPAN_TRIALS + PLANAR_RETURN_TRIALS + CYLINDER_TRANSITION_TRIALS
_AXIAL_RETURN_FAMILIES = frozenset(
    {
        "b0_straight_return",
        "planar_biarc_return",
        "planar_biarc_lifted_return",
        "cylinder_compact_return",
        "cylinder_compact_lifted_return",
    }
)
_RETURN_FAMILY_GROUPS = {
    "intrinsic_compact": frozenset({"cylinder_compact_return", "cylinder_compact_lifted_return"}),
    "b0_straight": frozenset({"b0_straight_return"}),
    "planar": frozenset({"planar_biarc_return", "planar_biarc_lifted_return"}),
}


def _return_family_counts(choices, selected, return_blocks=None):
    families = [
        getattr(choices[index][option], "profile_family", "")
        for index, option in enumerate(selected)
        if return_blocks is None or return_blocks[index]
    ]
    return {name: sum(family in members for family in families) for name, members in _RETURN_FAMILY_GROUPS.items()}


def _detour_curve_key(curve):
    if getattr(curve, "profile_family", "") not in {"cylinder_detour_return", "cylinder_detour_lifted_return"}:
        return None
    controls = curve.control_points_m
    winding = float(controls[-1, 0] - controls[0, 0])
    extent = max(0.0, float(np.ptp(controls[:, 1]) - abs(controls[-1, 1] - controls[0, 1])))
    return winding, extent


def _detour_return_key(metrics):
    metrics = [value for value in metrics if value is not None]
    if not metrics:
        return (), 0.0, 0.0, 0.0
    winding = tuple(int(np.sign(value[0])) for value in metrics)
    return (
        winding,
        round(abs(sum(value[0] for value in metrics)), 12),
        min(value[1] for value in metrics),
        sum(value[1] for value in metrics),
    )


def _detour_rail_representatives(ordered, key, *, complete=True):
    """Keep wide counter-directed cylinder pairs for independent field checks.

    Saved GY probes show opposite circumferential return currents on distant
    rails reduce longitudinal perturbation. This is bounded proposal diversity,
    not a field or physical-clearance certificate. Partial prefixes retain both
    winding classes so the eventual counter-directed pair remains reachable.
    """
    records = [(item, key(item)) for item in ordered]
    count = max((len(value[0]) for _item, value in records), default=0)
    if count == 0 or (complete and count < 2):
        return []
    records = [(item, value) for item, value in records if len(value[0]) == count]
    if complete:
        balance = min(value[1] for _item, value in records)
        records = [(item, value) for item, value in records if value[1] == balance]
    representatives = {}
    for item, value in records:
        prior = representatives.get(value[0])
        if prior is None or (round(value[2], 12), round(value[3], 12)) > (
            round(prior[1][2], 12),
            round(prior[1][3], 12),
        ):
            representatives[value[0]] = item, value
    return [item for item, _value in representatives.values()]


def _protected_clearance_key(protected, clearance, continuation_length):
    """Identify the exact segment multiset and continuation exclusion contract."""
    segments = np.asarray(protected, dtype=float).reshape(-1, 2, 3)
    digest = hashlib.sha256(b"".join(sorted(segment.tobytes() for segment in segments))).digest()
    return digest, float(clearance), float(continuation_length)


def _curve_geometry(curve):
    explicit = getattr(curve, "segment_geometry", ())
    if explicit:
        if len(explicit) != len(curve.points) - 1:
            raise ValueError("Curve geometry must match its sampled intervals.")
        return list(explicit)
    # This family follows a nominal facet and passes the same independent
    # support screen. Keep that contract through analytic surface realization;
    # a FREE_3D label would hold its chord below the realized cylinder while
    # neighboring retained contours move onto the analytic support.
    nominal = getattr(curve, "profile_family", "") in {"planar_biarc_return", "cylinder_surface"}
    return [WirePathGeometry.SURFACE if nominal else WirePathGeometry.FREE_3D] * (len(curve.points) - 1)


@dataclass(frozen=True)
class JointShapeLimits:
    """Deterministic work limits; each cut context has two spatial work phases.

    Support generation and mutable-transition coupling each receive their
    declared pair limit. Neither phase resets the other's consumed work.
    """

    maximum_shape_trials: int = 32768
    maximum_curve_samples: int = 2097152
    maximum_interaction_scores: int = 32768
    maximum_spatial_pairs: int = 8000000  # support generation
    maximum_coupling_spatial_pairs: int = 8000000
    beam_width: int = 48


class _ShapeWork:
    def __init__(self, limits):
        self.limits = limits
        self.spatial_pairs = 0
        self.interaction_scores = 0

    def consume_pairs(self, count):
        self.spatial_pairs += count
        if self.spatial_pairs > self.limits.maximum_spatial_pairs:
            raise _DistanceBudgetExceeded("joint_spatial_pair_budget_exceeded")

    def score(self, index, geometry, clearance, excluded=()):
        self.interaction_scores += 1
        if self.interaction_scores > self.limits.maximum_interaction_scores:
            raise _DistanceBudgetExceeded("joint_interaction_score_budget_exceeded")
        return index.gap(geometry, clearance, excluded=excluded, consume=self.consume_pairs)


class JointTransitionEvaluator:
    """Reusable whole-transition geometry for state search and final shaping.

    Costs describe supported, tangent-matched candidate polylines, not a finite
    body or field certificate. Protected-contour clearance is screened with the
    same conservative envelope as coupled shaping; connected endpoint
    continuations remain the final sweep's responsibility. Mutable transitions
    are checked against one another only after selecting a complete circuit.

    The instance snapshots its physical inputs. Reuse is scoped to this cut
    context; ``check_context`` rejects changed support, constraints or winding
    geometry. Geometry-only external cache entries are support-checked afresh.
    Explicit work exhaustion raises rather than masquerading as infeasibility.
    """

    def __init__(
        self,
        part,
        config,
        *,
        projector=None,
        transition_cache=None,
        limits=None,
        _protected_segments=None,
        progress_callback=None,
        search_mode="thorough",
        search_level=0,
    ):
        if search_mode not in ("progressive", "thorough"):
            raise ValueError("Joint curve search mode must be progressive or thorough.")
        if isinstance(search_level, bool) or not isinstance(search_level, int) or search_level not in (0, 1, 2):
            raise ValueError("Joint curve search level must be 0, 1 or 2.")
        self.search_mode = search_mode
        self.search_level = search_level
        self.progress_callback = progress_callback
        self.limits = limits or JointShapeLimits()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in vars(self.limits).values()
        ):
            raise ValueError("Joint shaping work limits must be positive integers.")
        self.config_key = self._config_key(config)
        self.support_key = self._support_key(part)
        self.protected = _protected_segments if _protected_segments is not None else _protected_group_segments(part)[0]
        self.protected = {key: np.asarray(value, dtype=float).copy() for key, value in self.protected.items()}
        if len(self.protected) > 100000:
            raise RoutingEvaluationBudgetExceeded("curve_input_budget_exceeded")
        self.obstacles = np.asarray(list(self.protected.values()), dtype=float).reshape(-1, 2, 3)
        if not np.all(np.isfinite(self.obstacles)):
            raise ValueError("Protected geometry must be finite.")
        self.obstacle_key = hashlib.sha256(self.obstacles.tobytes()).digest()
        self.obstacle_index = _SegmentDistanceIndex(self.obstacles)
        self.projector = copy(projector or _SurfaceProjector(part.mesh))
        # Projector spatial trees are read-only, but NumPy inputs normally alias
        # the live mesh. Snapshot those arrays so later mutation cannot change
        # an in-flight graph evaluation before its final context check.
        for name in ("vertices", "faces", "triangles", "face_normals", "vertex_normals"):
            if hasattr(self.projector, name):
                setattr(self.projector, name, getattr(self.projector, name).copy())
        # cKDTree may alias its construction array unless copy_data is set.
        for name in ("vertex_tree", "centroid_tree"):
            tree = getattr(self.projector, name, None)
            if tree is not None:
                setattr(self.projector, name, type(tree)(tree.data, copy_data=True))
        self.transition_cache = transition_cache
        self.work = _ShapeWork(self.limits)
        self.coupling_work = _ShapeWork(
            replace(self.limits, maximum_spatial_pairs=self.limits.maximum_coupling_spatial_pairs)
        )
        self.b0_direction = tuple(config.b_0_direction)
        self.b0_weight = float(config.joint_group_routing_b0_weight)
        self.jump_tolerance = float(config.routing_jump_height_tolerance_m)
        self.jump_spacing = float(config.routing_jump_sample_spacing_m)
        self._jump_counts = {}
        self._jump_geometry_counts = {}
        routing_cost(np.zeros((2, 3)), self.b0_direction, self.b0_weight)
        self.allowed_shift = 0.0 if config.skip_normal_shift else float(config.normal_shift_length)
        envelope = _cross_section_envelope_radius(conductor_cross_section_points(config))
        self.minimum_radius = max(float(config.sweep_min_bend_radius), 2.0 * envelope)
        self.clearance = max(
            2 * envelope
            + 2 * float(config.sweep_insulation_thickness)
            + float(config.sweep_manufacturing_tolerance)
            + float(config.sweep_bend_clearance_margin),
            float(config.min_wire_distance_m),
        )
        self._protected_clearance_context = _protected_clearance_key(
            self.obstacles, self.clearance, 3 * max(self.clearance, self.minimum_radius)
        )
        self._protected_clearance_certificates = {}
        # These certificates belong to this immutable support/cut snapshot.
        # External geometry caches deliberately do not carry them between
        # evaluators. Widening a grid often regenerates identical polylines.
        self._curve_screen_results = {}
        self._coupling_geometries = {}
        self._coupling_geometry_samples = 0
        self._coupling_pair_scores = {}
        self.curve_limits = dict(
            maximum_height_m=abs(self.allowed_shift),
            minimum_radius_m=self.minimum_radius,
            maximum_segment_length_m=float(config.sweep_max_segment_length) or max(envelope, self.minimum_radius / 3),
            chord_tolerance_m=min(float(config.sweep_max_chord_error) or envelope / 40, envelope / 40),
            maximum_tangent_angle_deg=min(float(config.sweep_max_frame_rotation_deg) / 2.5, 2.0),
            # Physical jump classification precedes local shortlisting. A
            # scalar-length shortlist must not hide a lower-jump direct curve.
            maximum_candidates=64,
            b0_direction=self.b0_direction,
            b0_weight=self.b0_weight,
            clearance_score_cap_m=self.clearance,
            search_level=None if search_mode == "thorough" or search_level == 2 else search_level,
        )
        self.incident = {}
        for index, segment in enumerate(self.obstacles):
            for endpoint in segment:
                self.incident.setdefault(tuple(endpoint), []).append(index)
        self._evaluated = {}
        self._prior_evaluated = {}
        self._corridor_keys = set()
        self.last_rejected_reasons = ()
        self.diagnostics = dict(
            generated_transition_count=0,
            generated_curve_count=0,
            generated_curve_family_counts={},
            protected_clearance_rejections_before_support=0,
            search_mode=search_mode,
            search_level=search_level if search_mode == "progressive" else "thorough",
            transition_cache_hits=0,
            curve_sample_count=0,
            shape_trial_upper_bound=0,
            generation_seconds=0.0,
            support_checks=0,
            support_screen_cache_hits=0,
            jump_geometry_cache_hits=0,
            coupling_geometry_cache_hits=0,
            coupling_pair_cache_hits=0,
            cost_is_finite_body_certificate=False,
            curve_families=[
                "direct_arch",
                "b0_straight_return",
                "planar_biarc_return",
                "cylinder_surface",
                "cylinder_compact_return",
                "cylinder_compact_lifted_return",
                "cylinder_detour_return",
                "cylinder_detour_lifted_return",
                "support_waypoints",
                "b0_corridor",
            ],
            unsupported_transition_count=0,
            unsupported_transition_reasons={},
            failed_transition_witnesses=[],
            corridor_expansion_count=0,
        )

    def __getstate__(self):
        # Saved routing trials retain geometry/work caches, never a live UI
        # callback (which may close over the current execution context).
        state = vars(self).copy()
        state["progress_callback"] = None
        return state

    def set_search_level(self, level):
        """Widen proposals without resetting cumulative physical work budgets.

        Previously screened curves remain available. The support snapshot and
        its chart are immutable; external geometry caches include the grid
        level and cannot replace wider generation with an earlier shortlist.
        """
        if isinstance(level, bool) or not isinstance(level, int) or level not in (0, 1, 2):
            raise ValueError("Joint curve search level must be 0, 1 or 2.")
        if level < self.search_level:
            raise ValueError("Joint curve search can only widen its proposal grid.")
        if self.search_mode != "progressive" or level == self.search_level:
            return
        self.search_level = level
        self.curve_limits["search_level"] = None if level == 2 else level
        self._prior_evaluated.update(self._evaluated)
        self._evaluated.clear()
        self._corridor_keys.clear()
        self.diagnostics["search_level"] = level

    @staticmethod
    def _config_key(config):
        profile = conductor_cross_section_points(config)
        names = (
            "b_0_direction",
            "joint_group_routing_b0_weight",
            "routing_jump_height_tolerance_m",
            "routing_jump_sample_spacing_m",
            "skip_normal_shift",
            "normal_shift_length",
            "sweep_min_bend_radius",
            "sweep_max_segment_length",
            "sweep_max_chord_error",
            "sweep_max_frame_rotation_deg",
            "sweep_insulation_thickness",
            "sweep_manufacturing_tolerance",
            "sweep_bend_clearance_margin",
            "min_wire_distance_m",
        )
        return tuple(repr(getattr(config, name)) for name in names), np.asarray(profile).tobytes()

    @staticmethod
    def _support_key(part):
        return tuple(
            hashlib.sha256(np.ascontiguousarray(values).tobytes()).digest() if values is not None else None
            for values in (part.mesh.vertices, part.mesh.faces, part.mesh.face_normals, part.mesh.vertex_normals)
        )

    def check_context(self, part, config, protected_segments):
        if self.config_key != self._config_key(config) or self.support_key != self._support_key(part):
            raise ValueError("Transition evaluator physical context changed.")
        if self.protected.keys() != protected_segments.keys() or any(
            not np.array_equal(value, protected_segments[key]) for key, value in self.protected.items()
        ):
            raise ValueError("Transition evaluator protected contour context changed.")

    def _consume_trials(self, count):
        if self.diagnostics["shape_trial_upper_bound"] + count > self.limits.maximum_shape_trials:
            self.diagnostics["budget_exhaustion_reason"] = "joint_shape_trial_budget_exceeded"
            self.diagnostics["maximum_shape_trials"] = self.limits.maximum_shape_trials
            raise RoutingEvaluationBudgetExceeded("joint_shape_trial_budget_exceeded")
        self.diagnostics["shape_trial_upper_bound"] += count

    def _record_unsupported(self, source, before, after):
        self.diagnostics["unsupported_transition_count"] += 1
        reasons = self.diagnostics["unsupported_transition_reasons"]
        for reason in self.last_rejected_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
        witnesses = self.diagnostics["failed_transition_witnesses"]
        if len(witnesses) < 8:
            witnesses.append(
                dict(
                    endpoints_m=source[[0, -1]].tolist(),
                    tangent_before=before.tolist(),
                    tangent_after=after.tolist(),
                    source_point_count=len(source),
                    reasons=list(self.last_rejected_reasons),
                )
            )

    def _supported(self, generation, source, normal):
        if "budget_exceeded" in generation.reason:
            raise RoutingEvaluationBudgetExceeded(generation.reason)
        count = sum(len(curve.points) for curve in generation.candidates)
        if self.diagnostics["curve_sample_count"] + count > self.limits.maximum_curve_samples:
            raise RoutingEvaluationBudgetExceeded("joint_curve_sample_budget_exceeded")
        self.diagnostics["curve_sample_count"] += count
        self.diagnostics["generated_curve_count"] = self.diagnostics.get("generated_curve_count", 0) + len(
            generation.candidates
        )
        families = self.diagnostics.setdefault("generated_curve_family_counts", {})
        for curve in generation.candidates:
            family = getattr(curve, "profile_family", "direct_arch")
            families[family] = families.get(family, 0) + 1
        supported, rejected = [], set(generation.rejected_reasons)
        excluded = _continuation_segments(
            self.obstacles, source[[0, -1]], 3 * max(self.clearance, self.minimum_radius), self.incident
        )
        for curve in generation.candidates:
            geometry = np.stack((curve.points[:-1], curve.points[1:]), axis=1)
            reasons = self._screen_curve(geometry, source, normal, excluded)
            rejected.update(reasons)
            if not reasons:
                self.jump_count(curve)
                if np.array_equal(curve.points[[0, -1]], source[[0, -1]]):
                    self._protected_clearance_certificates[id(curve)] = (
                        curve,
                        hashlib.sha256(np.ascontiguousarray(curve.points).tobytes()).digest(),
                        source[[0, -1]].tobytes(),
                    )
                supported.append(curve)
        if not generation.candidates:
            rejected.add(generation.reason)
        return supported, rejected

    def _screen_curve(self, geometry, source, normal, excluded):
        """Reuse completed exact screens, never failures caused by a work cap.

        Source ports determine the planar origin and connected-continuation
        exclusions. The normal and lift sign determine the support side.
        Geometry labels affect jump classification, not this chord screen.
        Retain the complete bytes so different sampled chords cannot alias.
        """
        key = (
            geometry.shape,
            geometry.tobytes(),
            source[[0, -1]].tobytes(),
            np.asarray(normal, dtype=float).tobytes(),
            self.allowed_shift,
            self.clearance,
            tuple(excluded),
            getattr(self, "search_mode", "thorough"),
        )
        # Compatibility with saved evaluators created before screen reuse.
        cache = self.__dict__.setdefault("_curve_screen_results", {})
        if key in cache:
            self.diagnostics["support_screen_cache_hits"] = self.diagnostics.get("support_screen_cache_hits", 0) + 1
            progress_callback = getattr(self, "progress_callback", None)
            if progress_callback is not None:
                progress_callback(self.diagnostics)
            return cache[key]
        gap = None
        try:
            if getattr(self, "search_mode", "thorough") == "progressive":
                gap = self.work.score(self.obstacle_index, geometry, self.clearance, excluded)
                if gap + 1e-12 < self.clearance:
                    self.diagnostics["protected_clearance_rejections_before_support"] = (
                        self.diagnostics.get("protected_clearance_rejections_before_support", 0) + 1
                    )
                    cache[key] = ("protected_corridor_clearance",)
                    return cache[key]
            self.diagnostics["support_checks"] += 1
            progress_callback = getattr(self, "progress_callback", None)
            if progress_callback is not None:
                progress_callback(self.diagnostics)
            reasons = tuple(
                _validate_support(self.projector, source[0], geometry, normal, self.allowed_shift, 1e-8, {})
            )
            if any("budget_exceeded" in reason for reason in reasons):
                raise RoutingEvaluationBudgetExceeded(next(reason for reason in reasons if "budget_exceeded" in reason))
            if not reasons:
                if gap is None:
                    gap = self.work.score(self.obstacle_index, geometry, self.clearance, excluded)
                if gap + 1e-12 < self.clearance:
                    reasons = ("protected_corridor_clearance",)
            cache[key] = reasons
            return reasons
        except _DistanceBudgetExceeded as error:
            raise RoutingEvaluationBudgetExceeded(str(error)) from error

    def protected_clearance_certificate(self, curve, context, endpoints):
        """Reuse only unchanged positive screens under the identical contract."""
        certificate = getattr(self, "_protected_clearance_certificates", {}).get(id(curve))
        if certificate is None or context != self._protected_clearance_context:
            return False
        retained, points_digest, ports = certificate
        return (
            retained is curve
            and ports == np.asarray(endpoints, dtype=float).tobytes()
            and points_digest == hashlib.sha256(np.ascontiguousarray(curve.points).tobytes()).digest()
        )

    def work_diagnostics(self):
        """Report separately bounded support and coupling work per cut context."""
        coupling = getattr(self, "coupling_work", None)
        support_pairs = self.work.spatial_pairs
        coupling_pairs = coupling.spatial_pairs if coupling is not None else 0
        support_limit = self.limits.maximum_spatial_pairs
        coupling_limit = self.limits.maximum_coupling_spatial_pairs
        return dict(
            support_spatial_candidate_pairs=support_pairs,
            coupling_spatial_candidate_pairs=coupling_pairs,
            total_spatial_candidate_pairs=support_pairs + coupling_pairs,
            maximum_support_spatial_pairs=support_limit,
            maximum_coupling_spatial_pairs=coupling_limit,
            maximum_total_spatial_pairs=support_limit + coupling_limit,
            spatial_pair_budget_scope="per_cut_context",
        )

    def coupling_geometry(self, curve):
        """Content-keyed immutable segment/index records for coupled ranking."""
        points = np.asarray(curve.points, dtype=float)
        key = points.shape, points.tobytes()
        cache = self.__dict__.setdefault("_coupling_geometries", {})
        if key in cache:
            self.diagnostics["coupling_geometry_cache_hits"] = (
                self.diagnostics.get("coupling_geometry_cache_hits", 0) + 1
            )
            return cache[key]
        segments = np.stack((points[:-1], points[1:]), axis=1)
        segments.flags.writeable = False
        record = segments, _SegmentDistanceIndex(segments), key
        samples = getattr(self, "_coupling_geometry_samples", 0)
        if samples + len(points) <= self.limits.maximum_curve_samples:
            cache[key] = record
            self._coupling_geometry_samples = samples + len(points)
        return record

    def coupling_gap(self, first, second, cap, work):
        """Preserve query direction and cap while reusing completed pair scores.

        The indexed first curve and queried second curve stay ordered: reversing
        them could change floating-point traversal, pruning, or work consumption.
        Neither mutable geometry nor interrupted/exhausted queries are cached.
        """
        key = first[2], second[2], float(cap)
        cache = self.__dict__.setdefault("_coupling_pair_scores", {})
        progress_callback = getattr(self, "progress_callback", None)
        if progress_callback is not None:
            # Coupling can otherwise spend seconds between phase boundaries.
            self.diagnostics.update(self.work_diagnostics())
            progress_callback(self.diagnostics)
        if key in cache:
            self.diagnostics["coupling_pair_cache_hits"] = self.diagnostics.get("coupling_pair_cache_hits", 0) + 1
            return cache[key]
        gap = work.score(first[1], second[0], cap)
        geometries = self.__dict__.get("_coupling_geometries", {})
        if (
            len(cache) < self.limits.maximum_interaction_scores
            and geometries.get(first[2]) is first
            and geometries.get(second[2]) is second
        ):
            # Pair keys retain only interned geometry bytes, so the sample cap
            # bounds their memory as well as the segment/index memo's memory.
            cache[key] = gap
        return gap

    def jump_count(self, curve):
        key = id(curve)
        geometry = _curve_geometry(curve)
        geometry_key = (
            np.asarray(curve.points, dtype=float).shape,
            np.asarray(curve.points, dtype=float).tobytes(),
            tuple(WirePathGeometry(value) for value in geometry),
            self.jump_tolerance,
            self.jump_spacing,
        )
        previous = self._jump_counts.get(key)
        if previous is not None and len(previous) == 3 and previous[0] is curve and previous[2] == geometry_key:
            return previous[1]
        cache = self.__dict__.setdefault("_jump_geometry_counts", {})
        if geometry_key in cache:
            count = cache[geometry_key]
            self.diagnostics["jump_geometry_cache_hits"] = self.diagnostics.get("jump_geometry_cache_hits", 0) + 1
        else:
            if not hasattr(self, "_jump_projector"):
                # This facade measures distances to actual triangle-covered
                # facets. Original normals remain authoritative for mounting.
                self._jump_projector = faceted_cylinder_support(self.projector) or self.projector
            try:
                metrics = raised_span_metrics(
                    curve.points,
                    self._jump_projector,
                    segment_geometry=geometry,
                    tolerance_m=self.jump_tolerance,
                    maximum_sample_spacing_m=self.jump_spacing,
                )
            except ValueError as error:
                if "budget_exceeded" in str(error):
                    raise RoutingEvaluationBudgetExceeded(str(error)) from error
                raise
            count = cache[geometry_key] = metrics["jump_count"]
        # Keep objects alive so IDs cannot be recycled; also check bytes because
        # a caller may mutate a generated curve after its initial screen.
        self._jump_counts[key] = curve, count, geometry_key
        return count

    def _curve_key(self, curve):
        return self.jump_count(curve), round(routing_cost(curve.points, self.b0_direction, self.b0_weight), 12)

    def _shortlist(self, curves, maximum=16):
        """Keep jump/family alternatives at their supported height extremes.

        Exact electromagnetic ranking occurs after constructing a complete
        circuit. Keep distinct families so a geometric guide alone cannot erase
        all straight-B0 returns before that evaluation. Equal-jump lifted
        returns at full height can clear another mutable return where a
        cheaper half-height sibling cannot; retain both before scalar fill.
        """
        ordered = sorted(curves, key=self._curve_key)
        progressive = getattr(self, "search_mode", "thorough") == "progressive"
        if progressive:
            maximum = max(maximum, 32)
        grouped = {}
        for curve in ordered:
            family = getattr(curve, "profile_family", "direct_arch")
            signature = self.jump_count(curve), family
            if progressive and family.startswith("cylinder"):
                controls = curve.control_points_m.reshape(-1, 3)
                signature += (int(np.sign(controls[-1, 0] - controls[0, 0])),)
                if family.startswith("cylinder_detour"):
                    signature += (int(np.sign(controls[2, 1] - controls[1, 1])),)
            grouped.setdefault(signature, []).append(curve)
        representatives = []
        for family in grouped.values():
            heights = [float(getattr(curve, "height_m", 0.0)) for curve in family]
            for height in dict.fromkeys((min(heights), max(heights))):
                # The family is already ordered by jumps and geometric guide,
                # so the first curve at an extreme is its cheapest sibling.
                options = [curve for curve, value in zip(family, heights) if value == height]
                representatives.append(options[0])
                if progressive and getattr(options[0], "profile_family", "").startswith("cylinder_detour"):

                    def extent(curve):
                        return abs(curve.control_points_m[2, 1] - curve.control_points_m[1, 1])

                    representatives.extend((min(options, key=extent), max(options, key=extent)))
        selected = []
        for curve in sorted(representatives, key=self._curve_key):
            if any(curve is prior for prior in selected):
                continue
            selected.append(curve)
            if len(selected) >= maximum:
                break
        for curve in ordered:
            if len(selected) >= maximum:
                break
            if all(curve is not prior for prior in selected):
                selected.append(curve)
        return tuple(sorted(selected, key=self._curve_key))

    def supported_curves(self, source_points, tangent_before, tangent_after):
        source = np.asarray(source_points, dtype=float)
        before, after = np.asarray(tangent_before, dtype=float), np.asarray(tangent_after, dtype=float)
        if (
            source.ndim != 2
            or source.shape[1:] != (3,)
            or len(source) < 2
            or any(
                value.shape != (3,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) <= 1e-15
                for value in (before, after)
            )
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("Whole transitions need finite points and nonzero directed tangents.")
        if len(source) > 4096:
            raise RoutingEvaluationBudgetExceeded("curve_input_budget_exceeded")
        key = source.tobytes(), before.tobytes(), after.tobytes()
        if key in self._evaluated:
            self.diagnostics["transition_cache_hits"] += 1
            choices, self.last_rejected_reasons = self._evaluated[key]
            return choices
        # Normal projection and every curve family are counted only once per
        # unique transition. No unsupported/budget-limited result is cached.
        self._consume_trials(transition_trial_count(self.curve_limits.get("search_level")))
        self.diagnostics["generated_transition_count"] += 1
        # Older saved evaluators predate the optional live progress callback.
        progress_callback = getattr(self, "progress_callback", None)
        if progress_callback is not None:
            progress_callback(self.diagnostics)
        endpoint_normals = self.projector.project_with_face_normals(source[[0, -1]])[2]
        normal = endpoint_normals.sum(axis=0)
        intrinsic_only = np.linalg.norm(normal) <= 1e-12
        if intrinsic_only:
            # Opposite cylinder facets have no common Cartesian arch frame.
            # Their developed surface still admits a directed connection.
            # Spatial support certification uses each chord's nearest facet;
            # the first port's frame only seeds its planar fast path.
            normal = endpoint_normals[0].copy()
        normal /= np.linalg.norm(normal)
        signed_normal = normal * (-1 if self.allowed_shift < 0 else 1)
        cache_key = (
            self.obstacle_key,
            source.tobytes(),
            before.tobytes(),
            after.tobytes(),
            signed_normal.tobytes(),
            tuple(self.curve_limits.items()),
        )
        generation = self.transition_cache.get(cache_key) if self.transition_cache is not None else None
        started = perf_counter()
        if generation is None:
            generation = (
                BridgeGeneration(reason="opposed_support_normals", rejected_reasons=("opposed_support_normals",))
                if intrinsic_only
                else generate_joint_transition_curves(
                    source,
                    before,
                    after,
                    signed_normal,
                    self.obstacles,
                    **self.curve_limits,
                    obstacle_index=self.obstacle_index,
                    distance_budget=self.work.consume_pairs,
                )
            )
            if "budget_exceeded" not in generation.reason:
                axial = (
                    BridgeGeneration()
                    if intrinsic_only
                    else generate_axial_return_curves(
                        source,
                        before,
                        after,
                        signed_normal,
                        self.obstacles,
                        **self.curve_limits,
                        obstacle_index=self.obstacle_index,
                        distance_budget=self.work.consume_pairs,
                    )
                )
                if "budget_exceeded" in axial.reason:
                    generation = axial
                else:
                    planar = (
                        BridgeGeneration()
                        if intrinsic_only
                        else generate_planar_return_curves(
                            self.projector,
                            source,
                            before,
                            after,
                            signed_normal,
                            self.obstacles,
                            **self.curve_limits,
                            obstacle_index=self.obstacle_index,
                            distance_budget=self.work.consume_pairs,
                            shape_trial_budget=self._consume_trials,
                        )
                    )
                    if "budget_exceeded" in planar.reason:
                        raise RoutingEvaluationBudgetExceeded(planar.reason)
                    if not hasattr(self, "_cylinder_chart_snapshot"):
                        try:
                            self._cylinder_chart_snapshot = _cylinder_chart(self.projector)
                            self._cylinder_chart_error = None
                        except (AttributeError, TypeError, ValueError) as error:
                            self._cylinder_chart_snapshot = None
                            self._cylinder_chart_error = str(error)
                    cylindrical = generate_cylindrical_transition_curves(
                        source,
                        before,
                        after,
                        self.projector,
                        **self.curve_limits,
                        shape_trial_budget=self._consume_trials,
                        shape_trials_reserved=True,
                        lift_sign=-1 if self.allowed_shift < 0 else 1,
                        cylinder_chart=self._cylinder_chart_snapshot,
                        cylinder_chart_error=self._cylinder_chart_error,
                    )
                    if "budget_exceeded" in cylindrical.reason:
                        raise RoutingEvaluationBudgetExceeded(cylindrical.reason)
                    curves = (*generation.candidates, *axial.candidates, *planar.candidates, *cylindrical.candidates)
                    generation = BridgeGeneration(
                        curves,
                        "candidates_available" if curves else "no_feasible_joint_transition",
                        tuple(
                            sorted(
                                set(generation.rejected_reasons)
                                | set(axial.rejected_reasons)
                                | set(planar.rejected_reasons)
                                | set(cylindrical.rejected_reasons)
                            )
                        ),
                    )
            if self.transition_cache is not None and "budget_exceeded" not in generation.reason:
                # Bound an externally reused cache as well as this instance.
                if len(self.transition_cache) < self.limits.maximum_shape_trials // _TRANSITION_TRIALS:
                    self.transition_cache[cache_key] = generation
        choices, rejected = self._supported(generation, source, normal)
        prior = getattr(self, "_prior_evaluated", {}).get(key)
        if prior is not None:
            choices.extend(prior[0])
            rejected.update(prior[1])
        if not choices and not intrinsic_only:
            extra, failures = self._generate_corridor(source, before, after, normal)
            self._corridor_keys.add(key)
            choices.extend(extra)
            rejected.update(failures)
        self.diagnostics["generation_seconds"] += perf_counter() - started
        self.diagnostics["spatial_candidate_pairs"] = self.work.spatial_pairs
        self.diagnostics.update(self.work_diagnostics())
        self.last_rejected_reasons = tuple(sorted(rejected))
        result = self._shortlist(choices)
        self._evaluated[key] = result, self.last_rejected_reasons
        if not result:
            self._record_unsupported(source, before, after)
        if progress_callback is not None:
            progress_callback(self.diagnostics)
        return result

    def _generate_corridor(self, source, before, after, normal):
        # At most three corridors, two heights, three handles and nine cubic
        # spans each. This work is separate from the direct arch family.
        level = self.curve_limits.get("search_level")
        self._consume_trials(54 if level == 0 else 108 if level == 1 else 162)
        self.diagnostics["corridor_expansion_count"] += 1
        corridor = generate_joint_corridor_curves(
            source,
            before,
            after,
            normal * (-1 if self.allowed_shift < 0 else 1),
            self.obstacles,
            **self.curve_limits,
            obstacle_index=self.obstacle_index,
            distance_budget=self.work.consume_pairs,
        )
        return self._supported(corridor, source, normal)

    def expand_corridors(self, source_points, tangent_before, tangent_after):
        """Lazily add alternatives for a witnessed conflict in a full circuit."""
        choices = self.supported_curves(source_points, tangent_before, tangent_after)
        source, before, after = (
            np.asarray(value, dtype=float) for value in (source_points, tangent_before, tangent_after)
        )
        key = source.tobytes(), before.tobytes(), after.tobytes()
        if key in self._corridor_keys or not choices:
            return choices
        normal = self.projector.project_with_face_normals(source[[0, -1]])[2].sum(axis=0)
        if np.linalg.norm(normal) <= 1e-12:
            return choices  # An intrinsic route has no common Cartesian corridor frame.
        normal /= np.linalg.norm(normal)
        started = perf_counter()
        extra, rejected = self._generate_corridor(source, before, after, normal)
        self._corridor_keys.add(key)
        reasons = tuple(sorted(set(self._evaluated[key][1]) | rejected))
        result = self._shortlist((*choices, *extra))
        self._evaluated[key] = result, reasons
        self.diagnostics["generation_seconds"] += perf_counter() - started
        self.diagnostics["spatial_candidate_pairs"] = self.work.spatial_pairs
        self.diagnostics.update(self.work_diagnostics())
        return result

    def __call__(self, points, tangent_before, tangent_after):
        curves = self.supported_curves(points, tangent_before, tangent_after)
        return min((routing_cost(curve.points, self.b0_direction, self.b0_weight) for curve in curves), default=np.inf)

    def search_cost(self, points, tangent_before, tangent_after):
        curves = self.supported_curves(points, tangent_before, tangent_after)
        if not curves:
            return RoutingSearchCost(0, np.inf)
        curve = min(curves, key=self._curve_key)
        return RoutingSearchCost(self.jump_count(curve), routing_cost(curve.points, self.b0_direction, self.b0_weight))


def _continuation_segments(segments, attachments, arclength, incident=None):
    """Remove only connected endpoint continuations from the ranking score.

    This is not a manufacturing exclusion: the final body still checks every
    conductor. It prevents fixed neighboring samples dominating a remote-gap
    score for every tangent-continuous proposal.
    """
    if incident is None:
        incident = {}
        for index, segment in enumerate(segments):
            for endpoint in segment:
                incident.setdefault(tuple(endpoint), []).append(index)
    removed = set()
    for attachment in attachments:
        node, previous, travelled = tuple(attachment), -1, 0.0
        while travelled < arclength:
            available = [index for index in incident.get(node, ()) if index != previous]
            if len(available) != 1:
                break
            index = available[0]
            if index in removed:
                break
            removed.add(index)
            segment = segments[index]
            following = segment[1] if tuple(segment[0]) == node else segment[0]
            travelled += float(np.linalg.norm(following - node))
            node, previous = tuple(following), index
    return removed


def _remote_protected_segments(segments, attachments, arclength):
    removed = _continuation_segments(segments, attachments, arclength)
    return segments[[index for index in range(len(segments)) if index not in removed]]


def _capped_curve_gap(first, second, cap):
    """Only cross-set pairs are queried; budget exhaustion remains explicit."""
    return _SegmentDistanceIndex(second).gap(first, cap)


def _rank_curve_combinations(
    choices,
    protected,
    clearance,
    continuation_length,
    count,
    *,
    b0_direction=(0, 0, 1),
    b0_weight=0.0,
    work=None,
    diagnostics=None,
    protected_index=None,
    jump_counts=None,
    return_blocks=None,
    protected_clearance_certificate=None,
    coupling_evaluator=None,
):
    """Beam search on a conservative spatial interaction graph.

    A missing graph edge means every alternative pair is separated by the
    capped scoring distance, as certified by their enclosing boxes. It is not
    a physical conductor clearance certificate. Dense interactions exhaust an
    explicit score budget instead of silently being omitted.
    """
    work = work or _ShapeWork(JointShapeLimits())
    diagnostics = diagnostics if diagnostics is not None else {}
    reused_pairs_before = (
        coupling_evaluator.diagnostics.get("coupling_pair_cache_hits", 0) if coupling_evaluator is not None else 0
    )
    if coupling_evaluator is None:
        segments = [
            [np.stack((curve.points[:-1], curve.points[1:]), axis=1) for curve in available] for available in choices
        ]
        indexes = [[_SegmentDistanceIndex(curve) for curve in available] for available in segments]
        coupling_records = None
    else:
        coupling_records = [[coupling_evaluator.coupling_geometry(curve) for curve in domain] for domain in choices]
        segments = [[record[0] for record in domain] for domain in coupling_records]
        indexes = [[record[1] for record in domain] for domain in coupling_records]
    low = np.array([np.min([index.minimum.min(axis=0) for index in available], axis=0) for available in indexes])
    high = np.array([np.max([index.maximum.max(axis=0) for index in available], axis=0) for available in indexes])
    pairs = _segment_bvh_candidate_pairs(
        low, high, clearance + 1e-12, leaf_size=4, max_pairs=work.limits.maximum_interaction_scores
    )
    if pairs is None:
        raise _DistanceBudgetExceeded("joint_interaction_graph_budget_exceeded")
    if len(pairs):
        delta = np.maximum(0, np.maximum(low[pairs[:, 0]] - high[pairs[:, 1]], low[pairs[:, 1]] - high[pairs[:, 0]]))
        pairs = pairs[np.einsum("ij,ij->i", delta, delta) <= (clearance + 1e-12) ** 2]
    neighbors = [[] for _ in choices]
    for first, second in pairs:
        neighbors[second].append(int(first))
    diagnostics.update(
        interacting_transition_pairs=len(pairs), possible_transition_pairs=len(choices) * (len(choices) - 1) // 2
    )
    protected_index = protected_index if protected_index is not None else _SegmentDistanceIndex(protected)
    incident = {}
    for index, segment in enumerate(protected):
        for endpoint in segment:
            incident.setdefault(tuple(endpoint), []).append(index)
    fixed_scores = []
    certificate_context = (
        _protected_clearance_key(protected, clearance, continuation_length)
        if protected_clearance_certificate is not None
        else None
    )
    reused_certificates = 0
    for available, geometry in zip(choices, segments):
        endpoints = available[0].points[[0, -1]]
        excluded = _continuation_segments(protected, endpoints, continuation_length, incident)
        scores = []
        for curve, curve_geometry in zip(available, geometry):
            if protected_clearance_certificate is not None and protected_clearance_certificate(
                curve, certificate_context, endpoints
            ):
                scores.append(clearance)
                reused_certificates += 1
            else:
                scores.append(work.score(protected_index, curve_geometry, clearance, excluded))
        fixed_scores.append(scores)
    diagnostics["reused_protected_clearance_certificates"] = reused_certificates
    costs = [[routing_cost(curve.points, b0_direction, b0_weight) for curve in available] for available in choices]
    jumps = jump_counts if jump_counts is not None else [[0] * len(available) for available in choices]
    detour_metrics = [
        [_detour_curve_key(curve) if return_blocks is None or return_blocks[index] else None for curve in available]
        for index, available in enumerate(choices)
    ]
    paired_detour_domains = sum(any(value is not None for value in domain) for domain in detour_metrics) >= 2
    pair_scores, future_cache = {}, {}
    future_checks, future_failures = 0, 0
    fixed_domains = [
        [option for option, gap in enumerate(scores) if gap + 1e-12 >= clearance] for scores in fixed_scores
    ]

    def pair_gap(first, first_option, second, second_option):
        key = first, first_option, second, second_option
        if key not in pair_scores:
            if coupling_evaluator is None:
                pair_scores[key] = work.score(indexes[first][first_option], segments[second][second_option], clearance)
            else:
                pair_scores[key] = coupling_evaluator.coupling_gap(
                    coupling_records[first][first_option], coupling_records[second][second_option], clearance, work
                )
        return pair_scores[key]

    def future_jump_bound(selected):
        """Necessary completion domains and an admissible remaining-jump bound.

        Future/future conflicts are deliberately unresolved here. Prefixes
        with no nominal completion remain as fallbacks for actual realization.
        """
        nonlocal future_checks, future_failures
        if selected in future_cache:
            return future_cache[selected]
        future_checks += 1
        minimum_jumps = 0
        for future in range(len(selected), len(choices)):
            assigned_neighbors = [prior for prior in neighbors[future] if prior < len(selected)]
            compatible = [
                option
                for option in fixed_domains[future]
                if all(
                    pair_gap(prior, selected[prior], future, option) + 1e-12 >= clearance
                    for prior in assigned_neighbors
                )
            ]
            if not compatible:
                future_failures += 1
                future_cache[selected] = None
                return None
            minimum_jumps += min(jumps[future][option] for option in compatible)
        future_cache[selected] = minimum_jumps
        return minimum_jumps

    def state_key(state):
        selected, gap, cost = state
        future_bound = future_jump_bound(selected) if gap + 1e-12 >= clearance else None
        feasible = future_bound is not None
        return (
            not feasible,
            0 if feasible else -round(gap, 12),
            sum(jumps[index][option] for index, option in enumerate(selected)) + (future_bound or 0),
            round(cost, 12),
            selected,
        )

    def retain(states, limit):
        ordered = sorted(states, key=state_key)
        if not ordered:
            return []
        selected = ordered[:1]
        # Equal-jump routes with real axial spans reach full-circuit field
        # evaluation even when their geometric length guide is less favorable.
        # Equal-jump representatives remain useful when the nominal gap is
        # infeasible: analytic realization can change it, and final finite-body
        # checks still reject insufficient clearance. Do not require identical
        # rounded gap scores before examining a distinct complete family.
        best_key = state_key(ordered[0])
        eligible = [
            state for state in ordered if (state_key(state)[0], state_key(state)[2]) == (best_key[0], best_key[2])
        ]
        counts = {state[0]: _return_family_counts(choices, state[0], return_blocks) for state in eligible}
        if paired_detour_domains:
            representatives = _detour_rail_representatives(
                eligible,
                lambda state: _detour_return_key(
                    [detour_metrics[index][option] for index, option in enumerate(state[0])]
                ),
                complete=len(eligible[0][0]) == len(choices),
            )
            for representative in representatives:
                if representative not in selected and len(selected) < limit:
                    selected.append(representative)
        for family in _RETURN_FAMILY_GROUPS:
            maximum = max(counts[state[0]][family] for state in eligible)
            if maximum == 0:
                continue
            representative = next(state for state in eligible if counts[state[0]][family] == maximum)
            if representative not in selected and len(selected) < limit:
                selected.append(representative)

        def axial_count(state):
            return sum(
                (return_blocks is None or return_blocks[index])
                and getattr(choices[index][option], "profile_family", "") in _AXIAL_RETURN_FAMILIES
                for index, option in enumerate(state[0])
            )

        for axial in sorted({axial_count(state) for state in eligible}, reverse=True):
            if len(selected) >= limit:
                break
            representative = next(state for state in eligible if axial_count(state) == axial)
            if representative not in selected and len(selected) < limit:
                selected.append(representative)
        for state in ordered:
            if len(selected) >= limit:
                break
            if state not in selected:
                selected.append(state)
        return sorted(selected, key=state_key)

    # Each state records selected options, worst remote gap and routing cost.
    states = [((), clearance, 0.0)]
    for block, available in enumerate(choices):
        future_cache.clear()
        expanded = []
        for selected, worst, length in states:
            for option, curve in enumerate(available):
                gap = min(worst, fixed_scores[block][option])
                for prior_block in neighbors[block]:
                    prior_option = selected[prior_block]
                    gap = min(gap, pair_gap(prior_block, prior_option, block, option))
                expanded.append(((*selected, option), gap, length + costs[block][option]))
        states = retain(expanded, work.limits.beam_width)
    diagnostics.update(
        scored_transition_option_pairs=len(pair_scores),
        interaction_scores=work.interaction_scores,
        spatial_candidate_pairs=work.spatial_pairs,
        forward_domain_checks=future_checks,
        forward_domain_failed_prefixes=future_failures,
        forward_domain_scope="necessary_nominal_completion_conditions_only",
    )
    reused_pairs = (
        coupling_evaluator.diagnostics.get("coupling_pair_cache_hits", 0) - reused_pairs_before
        if coupling_evaluator is not None
        else 0
    )
    diagnostics.update(
        reused_transition_option_pair_scores=reused_pairs,
        new_transition_option_pair_scores=len(pair_scores) - reused_pairs,
    )
    if states and states[0][1] + 1e-12 < clearance:
        selected = states[0][0]
        conflicts = [
            (first, second)
            for (first, option1, second, option2), gap in pair_scores.items()
            if gap + 1e-12 < clearance and selected[first] == option1 and selected[second] == option2
        ]
        diagnostics["conflicting_transition_pairs"] = [list(pair) for pair in sorted(set(conflicts))[:64]]
    else:
        diagnostics["conflicting_transition_pairs"] = []
    return retain(states, count)


def shape_joint_group_route(
    part,
    candidate,
    config,
    *,
    maximum_candidates=6,
    maximum_transitions=None,
    projector=None,
    transition_cache=None,
    limits=None,
    transition_evaluator=None,
    progress_callback=None,
):
    """Replace whole mutable return corridors with tangent-matched lifted curves.

    Intra-group spiral joins also receive tangent-matched curves. Every
    original directed opened-contour interval remains exact.
    Support checks concern the proposed polyline; whole-body certification is
    deliberately left to the independent manufacturing validator.

    Work is bounded by analytic shape trials, stored curve samples and spatial
    interaction queries, rather than an arbitrary winding-transition count.
    ``maximum_transitions`` remains an optional explicit caller admission limit.
    """
    result = GroupRouteProposals(reason="no_supported_transition")
    started = perf_counter()
    if transition_evaluator is not None:
        if limits is not None and limits != transition_evaluator.limits:
            result.reason = "transition_evaluator_limits_mismatch"
            return result
        limits = transition_evaluator.limits
    else:
        limits = limits or JointShapeLimits()
    if (
        (maximum_transitions is not None and (not isinstance(maximum_transitions, int) or maximum_transitions < 1))
        or not 1 <= maximum_candidates <= 64
        or any(not isinstance(value, int) or value < 1 for value in vars(limits).values())
    ):
        result.reason = "invalid_shape_budget"
        return result
    ids = candidate.segment_ids
    transition_count = sum(
        value is None and (index == 0 or ids[index - 1] is not None) for index, value in enumerate(ids)
    )
    if maximum_transitions is not None and transition_count > maximum_transitions:
        result.reason = "joint_transition_budget_exceeded"
        result.diagnostics.update(transition_count=transition_count, maximum_transitions=maximum_transitions)
        return result
    # A reused evaluator already paid for its cached curves during graph search.
    # Its actual new work remains bounded by _consume_trials; a blanket estimate
    # must not reject a complete proposal whose transitions need no generation.
    if transition_evaluator is None and _TRANSITION_TRIALS * transition_count > limits.maximum_shape_trials:
        result.reason = "joint_shape_trial_budget_exceeded"
        result.diagnostics.update(
            transition_count=transition_count,
            shape_trial_upper_bound=_TRANSITION_TRIALS * transition_count,
            maximum_shape_trials=limits.maximum_shape_trials,
        )
        return result
    result.diagnostics.update(
        transition_count=transition_count,
        shape_trial_upper_bound=_TRANSITION_TRIALS * transition_count,
        work_limits=vars(limits).copy(),
    )
    if len(candidate.protected_segments) > 100000:
        result.reason = "curve_input_budget_exceeded"
        result.diagnostics["protected_segment_count"] = len(candidate.protected_segments)
        return result
    b0_direction = tuple(config.b_0_direction)
    b0_weight = float(config.joint_group_routing_b0_weight)
    try:
        routing_cost(candidate.path.v, b0_direction, b0_weight)
    except (TypeError, ValueError):
        result.reason = "invalid_b0_objective"
        return result
    try:
        evaluator = transition_evaluator or JointTransitionEvaluator(
            part,
            config,
            projector=projector,
            transition_cache=transition_cache,
            limits=limits,
            _protected_segments=candidate.protected_segments,
        )
        evaluator.check_context(part, config, candidate.protected_segments)
    except RoutingEvaluationBudgetExceeded as error:
        result.reason = str(error)
        return result
    except ValueError as error:
        result.reason = "invalid_transition_evaluator_context"
        result.diagnostics["context_error"] = str(error)
        return result
    points, ids = candidate.path.v, candidate.segment_ids
    obstacles, obstacle_index = evaluator.obstacles, evaluator.obstacle_index
    clearance, minimum_radius = evaluator.clearance, evaluator.minimum_radius
    if not hasattr(evaluator, "coupling_work"):
        evaluator.coupling_work = _ShapeWork(
            replace(evaluator.limits, maximum_spatial_pairs=evaluator.limits.maximum_coupling_spatial_pairs)
        )
    work = evaluator.coupling_work
    blocks, choices = [], []
    if progress_callback is not None:
        progress_callback({"phase": "curve transitions", "completed": 0, "total": transition_count})
    index = 0
    while index < len(ids):
        if ids[index] is not None:
            index += 1
            continue
        end = index + 1
        while end < len(ids) and ids[end] is None:
            end += 1
        owners = candidate.path.segment_owners[index:end]
        connectors = [owner for owner in owners if owner.kind == "inter_group_connector"]
        if index == 0 or ids[index - 1] is None or (end < len(ids) and ids[end] is None):
            result.reason = "missing_protected_transition_tangents"
            if progress_callback is not None:
                progress_callback({"phase": "curve transitions", "status": "stopped", "message": result.reason})
            return result
        after = points[end + 1] - points[end] if end < len(ids) else points[1] - points[0]
        source, before = points[index : end + 1], points[index] - points[index - 1]
        try:
            supported = evaluator.supported_curves(source, before, after)
        except RoutingEvaluationBudgetExceeded as error:
            if progress_callback is not None:
                progress_callback({"phase": "curve transitions", "status": "stopped", "message": str(error)})
            result.reason = str(error)
            result.diagnostics.update(evaluator.diagnostics)
            result.diagnostics.update(
                elapsed_s=perf_counter() - started,
                failed_transition_index=len(blocks),
                source_segment_span=[index, end],
            )
            return result
        except ValueError as error:
            if progress_callback is not None:
                progress_callback({"phase": "curve transitions", "status": "stopped", "message": str(error)})
            result.reason = "invalid_transition_geometry"
            result.diagnostics["transition_error"] = str(error)
            return result
        if progress_callback is not None:
            progress_callback({"phase": "curve transitions", "completed": len(blocks) + 1, "total": transition_count})
        if not supported:
            if progress_callback is not None:
                progress_callback(
                    {"phase": "curve transitions", "status": "stopped", "message": "unsupported transition"}
                )
            result.diagnostics.update(evaluator.diagnostics)
            result.diagnostics.update(
                rejected_reasons=list(evaluator.last_rejected_reasons),
                elapsed_s=perf_counter() - started,
                failed_transition_index=len(blocks),
                source_segment_span=[index, end],
            )
            return result
        blocks.append((index, end, connectors[0] if connectors else owners[0]))
        choices.append(supported)
        index = end
    if not blocks:
        result.reason = "no_joint_transition"
        if progress_callback is not None:
            progress_callback({"phase": "curve transitions", "status": "stopped", "message": result.reason})
        return result
    if progress_callback is not None:
        progress_callback({"phase": "curve transitions", "status": "finished"})
        progress_callback({"phase": "curve coupling", "completed": 0, "total": None})
    # Coupled ranking is necessary: separately safe lifted transitions can
    # intersect each other at equal height. The final body remains authoritative.
    return_blocks = tuple(
        owner.kind == "inter_group_connector" or getattr(owner, "connector_role", "") == "joint_return"
        for _start, _end, owner in blocks
    )
    try:
        combinations = _rank_curve_combinations(
            choices,
            obstacles,
            clearance,
            3 * max(clearance, minimum_radius),
            maximum_candidates,
            b0_direction=b0_direction,
            b0_weight=b0_weight,
            work=work,
            diagnostics=result.diagnostics,
            protected_index=obstacle_index,
            jump_counts=[[evaluator.jump_count(curve) for curve in available] for available in choices],
            return_blocks=return_blocks,
            protected_clearance_certificate=evaluator.protected_clearance_certificate,
            coupling_evaluator=evaluator,
        )
        conflicts = result.diagnostics.get("conflicting_transition_pairs", [])
        if conflicts:
            # Local feasibility does not reserve space against other selected
            # connections. Expand only witnessed conflicts, then rerank once.
            implicated = sorted({index for pair in conflicts for index in pair})
            result.diagnostics["corridor_conflict_expansion_indices"] = implicated
            for block_index in implicated:
                start, end, _owner = blocks[block_index]
                after = points[end + 1] - points[end] if end < len(ids) else points[1] - points[0]
                choices[block_index] = evaluator.expand_corridors(
                    points[start : end + 1], points[start] - points[start - 1], after
                )
            combinations = _rank_curve_combinations(
                choices,
                obstacles,
                clearance,
                3 * max(clearance, minimum_radius),
                maximum_candidates,
                b0_direction=b0_direction,
                b0_weight=b0_weight,
                work=work,
                diagnostics=result.diagnostics,
                protected_index=obstacle_index,
                jump_counts=[[evaluator.jump_count(curve) for curve in available] for available in choices],
                return_blocks=return_blocks,
                protected_clearance_certificate=evaluator.protected_clearance_certificate,
                coupling_evaluator=evaluator,
            )
    except (_DistanceBudgetExceeded, RoutingEvaluationBudgetExceeded) as error:
        if progress_callback is not None:
            progress_callback({"phase": "curve coupling", "status": "stopped", "message": str(error)})
        result.reason = str(error)
        result.diagnostics.update(
            interaction_scores=work.interaction_scores,
            spatial_candidate_pairs=work.spatial_pairs,
            elapsed_s=perf_counter() - started,
            **evaluator.work_diagnostics(),
        )
        return result
    if progress_callback is not None:
        progress_callback({"phase": "curve coupling", "status": "finished"})
    for selected, _gap, _length_score in combinations:
        sections, owners, geometry, identities = [points[:1]], [], [], []
        cursor = 0
        cuts = deepcopy(candidate.cuts)
        for (start, end, owner), available, choice in zip(blocks, choices, selected):
            sections.append(points[cursor + 1 : start + 1])
            owners.extend(candidate.path.segment_owners[cursor:start])
            geometry.extend(candidate.path.segment_geometry[cursor:start])
            identities.extend(ids[cursor:start])
            curve = available[choice]
            sections.append(curve.points[1:])
            count = len(curve.points) - 1
            owners.extend([owner] * count)
            geometry.extend(_curve_geometry(curve))
            identities.extend([None] * count)
            if owner.kind == "inter_group_connector":
                cut = cuts[owner.connector_index]
                cut.connection_path_v = curve.points.copy()
                cut.selected_point1_v, cut.selected_point2_v = curve.points[:1].copy(), curve.points[-1:].copy()
                cut.cut1_v = points[start - 1 : start + 1].copy()
                following = points[1] if end == len(ids) else points[end + 1]
                cut.cut2_v = np.asarray([points[end], following])
                cut.connector_distance = _length(curve.points)
                cut.direct_distance = float(np.linalg.norm(curve.points[-1] - curve.points[0]))
                cut.geodesic_distance = float("inf")
                cut.used_direct_fallback = False
                cut.used_clearance_fallback = False
                cut.normal_shift_paths_v = []
                cut.connection_method = "joint_smooth_return"
            cursor = end
        sections.append(points[cursor + 1 :])
        owners.extend(candidate.path.segment_owners[cursor:])
        geometry.extend(candidate.path.segment_geometry[cursor:])
        identities.extend(ids[cursor:])
        path = WirePath(np.vstack(sections), owners, geometry)
        result.candidates.append(
            replace(
                candidate,
                path=path,
                cuts=cuts,
                segment_ids=tuple(identities),
                length_m=_length(path.v),
                retained_return_length_m=0.0,
            )
        )
        result.candidates[-1].joint_return_family_counts = _return_family_counts(choices, selected, return_blocks)
        detour_key = _detour_return_key(
            [
                _detour_curve_key(choices[index][option]) if return_blocks[index] else None
                for index, option in enumerate(selected)
            ]
        )
        if len(detour_key[0]) >= 2:
            result.candidates[-1].joint_detour_rail_key = detour_key
    result.reason = "shaped_proposals_require_validation"
    evaluator.diagnostics.update(evaluator.work_diagnostics())
    result.diagnostics.update(
        transition_count=len(blocks),
        choices_per_transition=[len(values) for values in choices],
        candidate_count=len(result.candidates),
        candidate_return_family_counts=[candidate.joint_return_family_counts for candidate in result.candidates],
        proposed_remote_gap_scores_m=[state[1] for state in combinations],
        clearance_score_is_finite_body_certificate=False,
        routing_objective="jump_count_then_length_plus_b0_transverse_guide",
        proposed_jump_counts=[
            sum(evaluator.jump_count(choices[index][option]) for index, option in enumerate(state[0]))
            for state in combinations
        ],
        b0_weight=b0_weight,
        proposed_mutable_routing_costs_m=[state[2] for state in combinations],
        generated_transition_count=evaluator.diagnostics["generated_transition_count"],
        curve_sample_count=evaluator.diagnostics["curve_sample_count"],
        generation_seconds=evaluator.diagnostics["generation_seconds"],
        transition_evaluator=dict(evaluator.diagnostics),
        elapsed_s=perf_counter() - started,
        **evaluator.work_diagnostics(),
    )
    return result
