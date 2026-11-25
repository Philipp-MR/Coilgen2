"""Atomic, budgeted adoption of independently validated joint group routes."""

from __future__ import annotations

import time
import hashlib
import pickle
from copy import copy
from dataclasses import replace

from .config import CoilGenConfig
from .models import CoilPart, TargetField

import numpy as np

from .group_route_validation import (
    GroupRouteValidationLimits,
    _isolated_part,
    _PHYSICAL_STATE,
    prepare_group_route_baseline,
    validate_group_route_candidate,
)
from .group_routing import (
    GroupRouteProposals,
    GroupRoutingLimits,
    RoutingEvaluationBudgetExceeded,
    _length,
    propose_joint_group_topologies,
    propose_joint_group_routes,
)
from .group_route_support import check_group_route_support_domain, certify_group_route_support
from .joint_route_shapes import JointTransitionEvaluator, _detour_rail_representatives, shape_joint_group_route
from .joint_cut_variants import propose_joint_cut_variants, prepare_joint_group_context
from .layout import interconnection_axis
from .manufacturing import (
    _SurfaceProjector,
    _required_centerline_clearance,
    conductor_cross_section_points,
    create_swept_coil_meshes,
)
from .progress import ProgressReporter
from .routing_progress import RoutingProgress
from .physics import biot_savart_wire_path
from .routing_objective import (
    added_route_regularity,
    routing_metrics,
    route_is_better,
    routing_selection_policy,
    regularity_order_key,
)
from .routing_jumps import fabricated_jump_metrics, raised_span_metrics
from .routing_field_objective import longitudinal_field_distortion
from .routing_baseline import assess_routing_baseline
from .contours import contour_evaluation_points
from .group_route_clearance import audit_interpart_clearance
from .routing_symmetry import audit_routing_symmetry
from .shared_return_cycles import propose_shared_return_cycles


def _route_metrics(part, config, *, complete_provenance=False):
    """Measure fabricated jumps and retain the geometric guide for diagnostics.

    Joint proposals explicitly label each protected contour interval. Legacy
    group owners do not distinguish ordinary joins from turns; omitting their
    directional penalty underestimates the diagnostic guide. This guide is not
    used for the historical jumps/field/length comparison. Regularity metrics
    include only explicit joins/returns, never ambiguous legacy contour groups.
    """
    physical = part.fabrication_path
    if physical is None:
        raise ValueError("Missing fabricated conductor geometry.")
    points = physical.wire_points_m
    owners = getattr(physical, "segment_owners", ())
    complete = complete_provenance or part.joint_route_diagnostics.get("status") == "accepted"
    known = {"contour_loop", "within_group_return", "inter_group_connector"}
    if complete:
        if len(owners) != len(points) - 1:
            raise ValueError("Missing explicit fabricated-interval provenance.")
        mutable = np.array([owner.kind != "contour_loop" for owner in owners], dtype=bool)
        lower_bound = False
    elif len(owners) == len(points) - 1:
        mutable = np.array(
            [
                owner.kind in known - {"contour_loop"}
                or (owner.kind == "contour_group" and getattr(owner, "connector_role", "") == "joint_spiral_join")
                for owner in owners
            ],
            dtype=bool,
        )
        # Explicit join ownership survives later retention and serialization;
        # the latest adoption status must not erase known directional costs.
        lower_bound = any(owner.kind not in known and not is_mutable for owner, is_mutable in zip(owners, mutable))
    else:
        mutable = np.zeros(len(points) - 1, dtype=bool)
        lower_bound = True
    values = routing_metrics(
        points,
        interconnection_axis(config, part.mesh),
        config.joint_group_routing_b0_weight,
        mutable_segments=mutable,
    )
    explicit_added = (
        np.array(
            [
                owner.kind in {"within_group_return", "inter_group_connector"}
                or (owner.kind == "contour_group" and getattr(owner, "connector_role", "") == "joint_spiral_join")
                for owner in owners
            ],
            dtype=bool,
        )
        if len(owners) == len(points) - 1
        else np.zeros(len(points) - 1, dtype=bool)
    )
    regularity_unknown = len(owners) != len(points) - 1 or any(
        owner.kind != "contour_loop" and not added for owner, added in zip(owners, explicit_added)
    )
    span_ids = (
        tuple(
            (
                owner.kind,
                owner.group_index,
                owner.connector_index,
                owner.group1_index,
                owner.group2_index,
                owner.connector_role,
            )
            for owner in owners
        )
        if len(owners) == len(points) - 1
        else None
    )
    values.update(
        added_route_regularity(
            points,
            explicit_added,
            span_ids=span_ids,
            closed=getattr(physical, "closed", False),
        )
    )
    values["regularity_is_lower_bound"] = regularity_unknown
    values.update(cost_is_lower_bound=lower_bound, geometry="fabricated_centroid")
    values.update(fabricated_jump_metrics(part, config))
    return values


def _retained_diagnostics(part, reason, **details):
    """Retaining an existing object is not a certificate of manufacturing success."""
    valid = part.manufacturing_diagnostics.get("manufacturing_ok")
    return {
        "status": "baseline_retained",
        "reason": reason,
        "selected_route_status": (
            "manufacturing_checks_passed"
            if isinstance(valid, (bool, np.bool_)) and valid
            else "invalid"
            if isinstance(valid, (bool, np.bool_))
            else "unverified"
        ),
        "selected_manufacturing_failure_reasons": part.manufacturing_diagnostics.get(
            "manufacturing_failure_reasons", ""
        ),
        "validation_basis": "existing_production_diagnostics",
        **details,
    }


def refresh_retained_route_diagnostics(parts):
    """Attach final manufacturing results to routes retained without body checks."""
    for part in parts:
        diagnostics = part.joint_route_diagnostics
        if diagnostics.get("status") != "baseline_retained":
            continue
        diagnostics["selected_manufacturing_failure_reasons"] = part.manufacturing_diagnostics.get(
            "manufacturing_failure_reasons", ""
        )
        if diagnostics.get("validation_basis") == "existing_production_diagnostics":
            refreshed = _retained_diagnostics(part, diagnostics["reason"])
            diagnostics.update(refreshed)


def _order_complete_return_families(ordered_candidates, *, selection_policy="physical"):
    """Reserve complete-family representatives within each measured jump class."""
    result = []
    for jumps in sorted({item[0] for item in ordered_candidates}):
        group = [item for item in ordered_candidates if item[0] == jumps]
        selected = group[:1]
        if selection_policy == "regular":
            physical = min(group, key=lambda item: round(item[1], 12))
            if physical is not selected[0]:
                selected.append(physical)
        for representative in _detour_rail_representatives(
            group, lambda item: getattr(item[4], "joint_detour_rail_key", ((), 0.0, 0.0, 0.0))
        ):
            if all(representative is not prior for prior in selected):
                selected.append(representative)
        for family in ("intrinsic_compact", "b0_straight", "planar"):
            counts = [getattr(item[4], "joint_return_family_counts", {}).get(family, 0) for item in group]
            maximum = max(counts)
            if maximum:
                representative = next(item for item, count in zip(group, counts) if count == maximum)
                if all(representative is not prior for prior in selected):
                    selected.append(representative)
        selected.extend(item for item in group if all(item is not prior for prior in selected))
        result.extend(selected)
    return result


def _proposal_regularity(candidate):
    """Exact protected identities delimit the mutable proposal spans."""
    points = np.asarray(candidate.path.v, dtype=float)
    identities = tuple(candidate.segment_ids)
    if len(identities) != len(points) - 1:
        raise ValueError("Proposal regularity requires complete protected interval identities.")
    return added_route_regularity(
        points,
        [identity is None for identity in identities],
        span_ids=[
            (
                owner.kind,
                owner.group_index,
                owner.connector_index,
                owner.group1_index,
                owner.group2_index,
                owner.connector_role,
            )
            for owner in candidate.path.segment_owners
        ],
        closed=True,
    )


def _cut_context_fingerprint(context):
    """Keep different directed return seeds even when the opened ports match."""
    digest = hashlib.sha256()

    def add_points(points):
        points = np.ascontiguousarray(points, dtype=float)
        digest.update(repr(points.shape).encode())
        digest.update(points.tobytes())

    for group in context.groups:
        for loop in group.opened_loops:
            add_points(loop.v)
    for group in context.connected_groups:
        for attribute in ("spiral_in_v", "spiral_out_v", "return_path_v"):
            add_points(getattr(group, attribute))
    return digest.digest()


def _topology_search_order(candidates, *, attempted_indices=(), failure_witnesses=()):
    """Softly prefer another cycle order and avoid witnessed directed returns.

    Every pending proposal remains available. A failed group pair is a search
    preference, never a claim that another port or winding state is infeasible.
    """
    attempted = set(attempted_indices)
    seen_orders = {candidates[index].group_sequence for index in attempted}
    failed_pairs = set()
    for witness in failure_witnesses:
        owner = witness.get("owner", {})
        if owner.get("kind") == "inter_group_connector" and owner.get("connector_role") == "joint_return":
            first, second = owner.get("group1_index", -1), owner.get("group2_index", -1)
            if first >= 0 and second >= 0 and first != second:
                failed_pairs.add((first, second))

    def key(index):
        order = candidates[index].group_sequence
        witnessed_returns = sum(pair in failed_pairs for pair in zip(order, order[1:] + order[:1]))
        return witnessed_returns, order in seen_orders, index

    return sorted((index for index in range(len(candidates)) if index not in attempted), key=key)


def _work_budget_reason(reason):
    """Distinguish computational work from the independent physical field budget."""
    reason = str(reason)
    if reason == "field_screen_candidate_budget_exhausted":
        return True
    if "work_limit" in reason or "work_budget" in reason:
        return True
    return "field" not in reason and (
        reason.endswith("_budget_exceeded")
        or reason in {"expanded_state_budget", "connector_evaluation_budget", "search_budget_exhausted"}
    )


_BEND_RADIUS_TRANSITION_REASONS = frozenset({"insufficient_bend_radius", "insufficient_lifted_bend_radius"})


def _bend_radius_blocked_first_stage(diagnostics):
    """Return fail-fast evidence when the cheap first shaping stage is bend-radius blocked.

    Progressive search only. The first stage must have shaped no candidate, and
    every attempt must have stopped at an unsupported transition whose rejected
    curves include a bend-radius violation. In that state the conductor cannot
    turn between neighbouring windings at the configured spacing, and the
    later port relocation / wider curve stage has not produced an adoptable
    route in the benchmark templates. This is a work heuristic, not a proof of
    infeasibility; ``joint_group_routing_search_mode="thorough"`` keeps the
    complete bounded search.
    """
    stages = diagnostics.get("search_stages") or ()
    if not stages or stages[0].get("level") != 0 or stages[0].get("shaped_candidates", 0):
        return None
    attempts = diagnostics.get("attempts") or ()
    if not attempts:
        return None
    for attempt in attempts:
        if list(attempt.get("reasons") or ()) != ["no_supported_transition"]:
            return None
        unsupported = (attempt.get("shaping") or {}).get("unsupported_transition_reasons") or ()
        if not _BEND_RADIUS_TRANSITION_REASONS.intersection(unsupported):
            return None
    return {
        "applied": True,
        "rule": "first_stage_all_transitions_bend_radius_blocked",
        "first_stage_attempt_count": len(attempts),
        "skipped": "failure_directed_port_search_and_wider_curve_stage",
        "full_search_option": "joint_group_routing_search_mode='thorough'",
        "infeasibility_proven": False,
    }


def _candidate_fingerprint(context_name, candidate):
    """Identify identical validation inputs within an unchanged cut context."""
    return hashlib.sha256(
        pickle.dumps(
            (
                context_name,
                candidate.path,
                candidate.segment_ids,
                candidate.protected_segments,
                candidate.group_sequence,
                candidate.cuts,
            ),
            protocol=5,
        )
    ).digest()


def improve_group_routes(parts, config, target_field):
    """Return selected parts; a rejected attempt preserves its original objects.

    Call after cleanup and fabrication-centerline realization, before final
    sweeping, PCB, inductance and field evaluation. Full body previews are
    generated lazily for finalists on isolated parts. Target-grid fields are cached;
    all candidates use the same original current. Per-part budgets sum to at
    most the configured whole-coil vector-change allowance. This is a bounded
    improvement search over directed ports and bounded cut alternatives, not
    a global routing optimizer.
    """
    return _JointRoutingSearch(parts, config, target_field).run()


class _JointRoutingSearch:
    """One bounded routing transaction with explicit preparation and adoption phases.

    State is private to this invocation. Candidates share the original current
    and reference fields; only adopt() replaces selected geometry.
    """

    def __init__(self, parts: list[CoilPart], config: CoilGenConfig, target_field: TargetField | None):
        self.parts = parts
        self.config = config
        self.target_field = target_field

    def run(self):
        if not self.prepare():
            return self.parts
        self.progress = ProgressReporter.from_config(self.config, label="Joint routing")
        # Part counts are meaningful, but they do not estimate the adaptive
        # transition search or finalist validation remaining within each part.
        self.task = self.progress.task("completed parts", len(self.eligible), indeterminate=True)
        for self.completed, self.index in enumerate(self.eligible):
            if not self.prepare_part():
                continue
            self.validate_candidates()
            self.adopt()
            self.report()
        self.task.done()
        return self.selected

    def prepare(self):
        if (
            not self.config.joint_group_routing
            or self.config.skip_sweep
            or self.config.skip_postprocessing
            or self.config.contour_only_sweep_mode
        ):
            return False
        self.production_config = self.config
        self.selection_policy = routing_selection_policy(self.config)
        regular_raw = getattr(self.config, "joint_group_routing_regular_field_tolerance_rel", 0.001)
        if self.selection_policy not in {"regular", "physical"}:
            for self.part in self.parts:
                self.part.joint_route_diagnostics = _retained_diagnostics(self.part, "invalid_routing_selection_policy")
            return False
        try:
            self.regular_relative = float(regular_raw)
            regular_valid = (
                not isinstance(regular_raw, (bool, np.bool_))
                and np.isscalar(regular_raw)
                and np.isfinite(self.regular_relative)
                and self.regular_relative >= 0
            )
        except (TypeError, ValueError):
            regular_valid = False
        if not regular_valid:
            for self.part in self.parts:
                self.part.joint_route_diagnostics = _retained_diagnostics(
                    self.part, "invalid_regularity_field_tolerance"
                )
            return False
        self.search_mode = getattr(self.config, "joint_group_routing_search_mode", "progressive")
        if self.search_mode not in {"progressive", "thorough"}:
            for self.part in self.parts:
                self.part.joint_route_diagnostics = _retained_diagnostics(
                    self.part, "invalid_joint_routing_search_mode"
                )
            return False
        self.progressive = self.search_mode == "progressive"
        self.source_contour_clearance = max(
            float(self.config.min_wire_distance_m), float(self.config.min_turn_distance)
        )
        self.routing_clearance_override = getattr(self.config, "joint_group_routing_clearance_m", None)
        if self.routing_clearance_override is not None:
            try:
                override = float(self.routing_clearance_override)
                valid = (
                    not isinstance(self.routing_clearance_override, (bool, np.bool_))
                    and np.isfinite(override)
                    and override > 0
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                for self.part in self.parts:
                    self.part.joint_route_diagnostics = _retained_diagnostics(
                        self.part, "invalid_joint_routing_clearance"
                    )
                return False
            requested = override
            existing_physical = _required_centerline_clearance(conductor_cross_section_points(self.config), self.config)
            # Only the isolated routing preview receives this extra margin. It
            # makes the explicit centerline target hold after analytic realization,
            # while retaining conductor, insulation and manufacturing floors. Both
            # contour-spacing aliases remain unchanged on the source config.
            margin = float(self.config.sweep_bend_clearance_margin) + max(0.0, requested - existing_physical)
            self.config = replace(
                self.config,
                min_wire_distance_m=requested,
                min_turn_distance=requested,
                sweep_bend_clearance_margin=margin,
            )
        self.physical_clearance = _required_centerline_clearance(
            conductor_cross_section_points(self.config), self.config
        )
        self.routing_clearance = max(
            float(self.config.min_wire_distance_m), float(self.config.min_turn_distance), self.physical_clearance
        )
        self.selected = list(self.parts)
        self.routing_limits = GroupRoutingLimits(
            return_fractions=(0.0,), candidate_count=max(6, self.config.joint_group_routing_max_candidates)
        )
        for self.part in self.parts:
            if self.part.wire_path is not None and len(self.part.groups) > self.routing_limits.maximum_groups:
                self.part.joint_route_diagnostics = _retained_diagnostics(self.part, "group_count_budget_exceeded")
        self.eligible = [
            i
            for i, part in enumerate(self.parts)
            if part.wire_path is not None
            and 1 <= len(part.groups) <= self.routing_limits.maximum_groups
            and (len(part.groups) > 1 or len(part.groups[0].loops) > 1)
        ]
        if not self.eligible:
            return False
        if self.config.make_cylndrical_pcb:
            # PCB layer routing has its own spiral-in/out topology; a 3-D lifted
            # circuit cannot silently stand in for a validated two-layer layout.
            for self.index in self.eligible:
                self.parts[self.index].joint_route_diagnostics = _retained_diagnostics(
                    self.parts[self.index], "pcb_layer_routing"
                )
            return False
        admitted = []
        for self.index in self.eligible:
            domain = check_group_route_support_domain(self.parts[self.index].mesh, self.config)
            if domain.accepted:
                admitted.append(self.index)
            else:
                self.parts[self.index].joint_route_diagnostics = _retained_diagnostics(
                    self.parts[self.index],
                    "support_domain_not_certified",
                    support_reasons=domain.reasons,
                    support=domain.diagnostics,
                )
        self.eligible = admitted
        if not self.eligible:
            return False
        self.limits = GroupRouteValidationLimits()
        self.current = self.parts[0].contour_step
        reason = ""
        if self.current is None or not np.isfinite(self.current) or self.current == 0:
            reason = "invalid_fixed_current"
        elif any(
            part.contour_step is None
            or not np.isfinite(part.contour_step)
            or not np.isclose(part.contour_step, self.current, rtol=1e-12, atol=0.0)
            for part in self.parts
        ):
            reason = "inconsistent_series_current"
        elif self.target_field is None or not 0 < len(self.target_field.coords) <= self.limits.maximum_target_points:
            reason = "target_context_missing_or_over_budget"
        elif (
            np.shape(self.target_field.coords) != (len(self.target_field.coords), 3)
            or np.shape(self.target_field.b) != np.shape(self.target_field.coords)
            or np.shape(self.target_field.weights) != (len(self.target_field.coords),)
            or np.shape(self.target_field.suppressed) != (len(self.target_field.coords),)
            or not np.all(np.isfinite(self.target_field.coords))
        ):
            reason = "invalid_target_context"
        elif any(part.wire_path is None or part.fabrication_path is None for part in self.parts):
            reason = "missing_baseline_physical_geometry"
        elif any(not part.contour_lines for part in self.parts):
            reason = "missing_original_contour_reference"
        if reason:
            for self.index in self.eligible:
                self.parts[self.index].joint_route_diagnostics = _retained_diagnostics(self.parts[self.index], reason)
            return False
        active_dsv = (
            np.isfinite(self.target_field.weights) & (self.target_field.weights > 0) & ~self.target_field.suppressed
        )
        target_scale = float(np.max(np.linalg.norm(self.target_field.b[active_dsv], axis=1), initial=0.0))
        relative = float(self.config.joint_group_routing_max_field_change_rel)
        if not np.isfinite(target_scale) or not np.isfinite(relative) or relative < 0 or not np.any(active_dsv):
            for self.index in self.eligible:
                self.parts[self.index].joint_route_diagnostics = _retained_diagnostics(
                    self.parts[self.index], "invalid_field_budget_context"
                )
            return False
        self.field_budget = relative * target_scale / abs(self.current) / len(self.eligible)
        self.regular_rms_tolerance = float(self.regular_relative) * target_scale / len(self.eligible)
        longitudinal_relative = getattr(self.config, "joint_group_routing_max_longitudinal_field_change_rel", None)
        self.longitudinal_kwargs = {}
        if longitudinal_relative is not None:
            if (
                isinstance(longitudinal_relative, (bool, np.bool_))
                or not np.isfinite(longitudinal_relative)
                or longitudinal_relative <= 0
            ):
                for self.index in self.eligible:
                    self.parts[self.index].joint_route_diagnostics = _retained_diagnostics(
                        self.parts[self.index], "invalid_longitudinal_field_budget_context"
                    )
                return False
            self.longitudinal_kwargs["maximum_longitudinal_field_delta_t_per_a"] = (
                longitudinal_relative * target_scale / abs(self.current) / len(self.eligible)
            )
        # Impossible geometry and exhausted transition budgets need no Biot–Savart
        # work. Initialize the whole-coil fields only for the first shaped proposal.
        self.route_fields = self.physical_fields = self.reference_field = None
        # Preserve originals on rejected attempts. Body previews are needed only
        # after a complete candidate survives transition shaping, including all
        # background circuits needed for field and interpart-clearance acceptance.
        self.physical_selected = list(self.selected)
        return True

    def prepare_part(self):
        self.part = self.parts[self.index]
        if self.production_config.routing_group_plan is not None:
            from .group_circuits import circuit_config

            self.config = circuit_config(self.part, self.config)
        self.task.update(
            self.completed, message=f"part {self.index + 1}/{len(self.parts)}: searching group routes", force=True
        )
        self.start_time = time.perf_counter()
        try:
            self.baseline_metrics = _route_metrics(self.part, self.config)
        except (ValueError, TypeError, AttributeError):
            self.part.joint_route_diagnostics = _retained_diagnostics(
                self.part, "baseline_objective_geometry_unavailable"
            )
            return False
        self.diagnostics = {
            **_retained_diagnostics(self.part, "search_in_progress"),
            "scope": "directed_group_cycle_with_bounded_cut_alternatives",
            "fixed_current_a": float(self.current),
            "maximum_field_delta_t_per_a": self.field_budget,
            "baseline_length_m": self.baseline_metrics["length_m"],
            "baseline_raw_route_length_m": _length(self.part.wire_path.v),
            "baseline_routing_metrics": self.baseline_metrics,
            "b0_direction": np.asarray(self.config.b_0_direction, dtype=float).tolist(),
            "b0_weight": float(self.config.joint_group_routing_b0_weight),
            "objective_scope": (
                "cyclic_jumps_then_added_backtracking_attachment_bending_join_consistency_excess_turning_then_length_then_longitudinal_rms"
                if self.selection_policy == "regular"
                else "cyclic_jumps_then_fixed_current_longitudinal_rms_then_length"
            ),
            "selection_policy": self.selection_policy,
            "regularity_metric_scope": (
                "explicit_added_fabricated_spans_with_directed_interfaces_and_within_group_join_consistency"
            ),
            "regularity_field_tolerance_rel": float(self.regular_relative),
            "regularity_part_rms_tolerance_T": self.regular_rms_tolerance,
            "regularity_field_tolerance_basis": "baseline_anchored_equal_share_of_whole_coil_peak_target_allowance",
            "field_reference": "original_oriented_contours",
            "field_acceptance_policy": getattr(self.config, "joint_group_routing_field_acceptance", "absolute"),
            "source_contour_clearance_m": self.source_contour_clearance,
            "routing_clearance_override_m": self.routing_clearance_override,
            "routing_clearance_target_m": self.routing_clearance,
            "physical_clearance_required_m": self.physical_clearance,
            "jump_optimality_proven": False,
            "search_mode": self.search_mode,
            "search_coverage": "bounded_progressive" if self.progressive else "bounded_broad_transition_search",
            "maximum_cut_contexts": 32,
            "attempts": [],
        }
        if self.longitudinal_kwargs:
            self.diagnostics.update(self.longitudinal_kwargs)
        self.task.update(
            self.completed,
            message=f"part {self.index + 1}/{len(self.parts)}: checking mesh, SF and contour symmetry",
            force=True,
        )
        self.symmetry = audit_routing_symmetry(self.part)
        self.diagnostics["symmetry"] = self.symmetry
        self.projector = _SurfaceProjector(self.part.mesh)
        self.transition_cache = {}
        self.work_progress = RoutingProgress(self.progress, prefix=f"part {self.index + 1}/{len(self.parts)}")
        self.active_work_progress = self.work_progress

        try:
            initial_context = prepare_joint_group_context(self.part, self.config)
        except ValueError as error:
            self.part.joint_route_diagnostics = _retained_diagnostics(
                self.part, "singleton_cut_context_unavailable", detail=str(error)
            )
            return False
        self.contexts = [self.context_proposal("existing_cuts", initial_context)]
        self.proposals = self.contexts[0][2]
        self.diagnostics["search"] = self.proposals.diagnostics
        if self.config.joint_group_routing_search_cuts:
            cut_variants = propose_joint_cut_variants(self.part, self.config)
            self.diagnostics["cut_search"] = {**cut_variants.diagnostics, "reasons": cut_variants.reasons}
            for variant_index, context in enumerate(cut_variants.variants):
                self.contexts.append(self.context_proposal(f"alternate_cuts_{variant_index + 1}", context))
            self.diagnostics["alternate_topology_searches"] = [
                {"context": name, "reason": alternatives.reason, **alternatives.diagnostics}
                for name, _context, alternatives, _evaluator in self.contexts[1:]
            ]
        self.baselines = {}
        self.baseline_feasibility = None
        self.certified_baseline = None
        self.baseline_certification_failure = ()
        self.diagnostics["baseline_certification_deferred"] = True
        self.candidate_count = 0
        self.field_candidate_count = 0
        self.validated_context_counts = {}
        self.validated_candidates = set()
        self.best = None
        self.work_progress.prefix = f"part {self.index + 1}/{len(self.parts)}"
        self.validation_total = self.validation_stage_start = 0
        self.audit_progress = RoutingProgress(self.progress, prefix=f"part {self.index + 1}/{len(self.parts)} finalist")
        self.active_audit_phase = None
        return True

    def field(self, points):
        return biot_savart_wire_path(points, self.target_field.coords, worker_count=1, max_chunk_memory_mb=32.0)

    def materialize_baselines(self, work_progress):
        unswept_count = sum(other.swept_body is None for other in self.selected)
        work_progress.start("baseline body previews", unswept_count)
        preview_count = 0
        for other_index, other in enumerate(self.selected):
            if other.swept_body is None:
                preview = _isolated_part(other)
                create_swept_coil_meshes([preview], self.production_config)
                self.physical_selected[other_index] = preview
                preview_count += 1
                work_progress.update(preview_count, force=True)
            else:
                self.physical_selected[other_index] = other
        work_progress.finish()

    def context_proposal(self, name, context):
        # A context without singleton cuts can be the original part. Keep
        # physical-state rebinding local even when its geometry is shared.
        if context is self.part and self.part.swept_body is None:
            context = copy(context)
        self.task.update(
            self.completed, message=f"part {self.index + 1}/{len(self.parts)}: {name} transition search", force=True
        )
        self.work_progress.prefix = f"part {self.index + 1}/{len(self.parts)}: {name}"
        self.work_progress.start("transition search", None)
        active_search_phase = None

        def search_progress(values):
            nonlocal active_search_phase
            if values.get("status") in {"finished", "stopped"}:
                self.work_progress.finish(message=values.get("message") or values["status"])
                return
            if values["phase"] != active_search_phase:
                active_search_phase = values["phase"]
                self.work_progress.start(
                    active_search_phase, values["total"], upper_bound=values.get("upper_bound", False)
                )
            self.work_progress.update(values["completed"], message=values.get("message", ""))

        def transition_progress(values):
            self.active_work_progress.activity(
                message=(
                    f"{name}; "
                    f"transitions={values['generated_transition_count']}; "
                    f"curve samples={values['curve_sample_count']}; "
                    f"support checks={values['support_checks']}"
                ),
            )

        try:
            evaluator = JointTransitionEvaluator(
                context,
                self.config,
                projector=self.projector,
                transition_cache=self.transition_cache,
                progress_callback=transition_progress,
                **({"search_mode": self.search_mode, "search_level": 0} if self.progressive else {}),
            )
        except (ValueError, RoutingEvaluationBudgetExceeded) as error:
            return (
                name,
                context,
                GroupRouteProposals(
                    reason="transition_context_unavailable",
                    diagnostics={
                        "detail": str(error),
                        "budget_exhausted": isinstance(error, RoutingEvaluationBudgetExceeded),
                    },
                ),
                None,
            )
        # Progressive ranking spends no fine curve work on unused edges.
        # Physical shaping and fixed-current validation remain authoritative.
        planner = propose_joint_group_topologies if self.progressive else propose_joint_group_routes
        proposals = planner(
            context,
            self.config,
            limits=self.routing_limits,
            b0_direction=self.config.b_0_direction,
            b0_weight=self.config.joint_group_routing_b0_weight,
            progress_callback=search_progress,
            **({"symmetry": self.symmetry} if self.progressive else {"transition_evaluator": evaluator}),
        )
        if self.selection_policy == "regular":
            shared = propose_shared_return_cycles(context, self.config)
            proposals.diagnostics["shared_return_cycles"] = {"reason": shared.reason, **shared.diagnostics}
            if shared.candidates:
                # Reserve the next winding trial for a genuinely different
                # retained-return topology. It still goes through the same
                # shaping, original-reference fields and body validator.
                ordinary = proposals.candidates
                proposals.candidates = [
                    family[rank]
                    for rank in range(max(len(ordinary), len(shared.candidates)))
                    for family in (ordinary, shared.candidates)
                    if rank < len(family)
                ]
                proposals.reason = "candidates_available"
        self.work_progress.finish(message=proposals.reason)
        self.task.update(
            self.completed,
            message=f"part {self.index + 1}/{len(self.parts)}: {name} search completed; {proposals.reason}",
            force=True,
        )
        return name, context, proposals, evaluator

    def validation_progress(self, event):
        if event["phase"] != self.active_audit_phase:
            self.active_audit_phase = event["phase"]
            self.audit_progress.start(self.active_audit_phase, event["total"])
        self.audit_progress.update(event["completed"], message=event.get("message", ""))
        if event.get("status") in {"finished", "stopped"}:
            self.audit_progress.finish(event.get("message") or event["status"], status=event["status"])
            self.active_audit_phase = None

    def shaped_candidates(self, maximum_shape_attempts, shape_total):
        alternatives = []
        shape_count = 0
        attempted_topologies = {}
        # Each cut context receives its best topology before further orders
        # consume the preview budget. Shape alternatives are a second pass.
        for rank in range(max(len(values.candidates) for _name, _context, values, _evaluator in self.contexts)):
            for name, context, values, evaluator in self.contexts:
                if rank >= len(values.candidates):
                    continue
                if shape_count >= maximum_shape_attempts:
                    self.diagnostics["shape_search_budget_exhausted"] = True
                    break
                candidate_index = rank
                if self.selection_policy == "regular":
                    tried = attempted_topologies.setdefault(name, set())
                    witnesses = [
                        attempt["shaping"]["failed_transition_witness"]
                        for attempt in self.diagnostics["attempts"]
                        if attempt.get("shaping", {}).get("failed_transition_witness")
                    ]
                    candidate_index = _topology_search_order(
                        values.candidates, attempted_indices=tried, failure_witnesses=witnesses
                    )[0]
                    tried.add(candidate_index)
                topology = values.candidates[candidate_index]
                shape_count += 1
                self.diagnostics["shape_attempt_count"] = self.diagnostics.get("shape_attempt_count", 0) + 1
                self.work_progress.activity(
                    message=f"{name}: shaping topology {shape_count}/{shape_total}",
                    force=True,
                )
                curve_progress = RoutingProgress(
                    self.progress,
                    prefix=f"part {self.index + 1}/{len(self.parts)} {name} topology {shape_count}/{shape_total}",
                )
                active_curve_phase = None

                def shape_progress(event):
                    nonlocal active_curve_phase
                    if event.get("status") in {"finished", "stopped"}:
                        curve_progress.finish(event.get("message") or event["status"])
                    elif event["phase"] != active_curve_phase:
                        active_curve_phase = event["phase"]
                        curve_progress.start(active_curve_phase, event["total"])
                        curve_progress.update(event["completed"])
                    else:
                        curve_progress.update(event["completed"], message=event.get("message", ""))

                self.active_work_progress = curve_progress
                try:
                    shapes = shape_joint_group_route(
                        context,
                        topology,
                        self.config,
                        maximum_candidates=3,
                        projector=self.projector,
                        transition_cache=self.transition_cache,
                        transition_evaluator=evaluator,
                        progress_callback=shape_progress,
                    )
                finally:
                    self.active_work_progress = self.work_progress
                self.work_progress.update(shape_count, message=shapes.reason, force=True)
                if not shapes.candidates:
                    self.diagnostics["attempts"].append(
                        {
                            "context": name,
                            "topology_index": candidate_index,
                            "group_sequence": list(topology.group_sequence),
                            "reasons": [shapes.reason],
                            "shaping": shapes.diagnostics,
                            "eligible_for_adoption": False,
                            "selected": False,
                        }
                    )
                    continue
                yield name, context, shapes.candidates[0]
                alternatives.append((name, context, shapes.candidates[1:]))
            if shape_count >= maximum_shape_attempts:
                break
        for rank in range(2):
            for name, context, values in alternatives:
                if rank < len(values):
                    yield name, context, values[rank]
        self.diagnostics["shape_search_budget_exhausted"] = self.diagnostics.get(
            "shape_search_budget_exhausted", False
        ) or (shape_count < sum(len(values.candidates) for _name, _context, values, _evaluator in self.contexts))

    def ordered_proposals(self):
        stages = (0, 1) if self.progressive or self.selection_policy == "regular" else (0,)
        self.diagnostics["search_stages"] = []
        for level in stages:
            if level and (
                self.best is not None
                or self.candidate_count >= self.config.joint_group_routing_max_candidates
                or self.field_candidate_count >= self.limits.maximum_field_candidates
            ):
                break
            if level and self.progressive:
                fail_fast = _bend_radius_blocked_first_stage(self.diagnostics)
                if fail_fast is not None:
                    self.diagnostics["fail_fast"] = fail_fast
                    break
            if level:
                # A local failed transition identifies a useful port search
                # direction. Keep incumbent geometry immutable and propose
                # additional cut contexts once, before widening curve grids.
                witnesses = [
                    attempt["shaping"]["failed_transition_witness"]
                    for attempt in self.diagnostics["attempts"]
                    if attempt.get("shaping", {}).get("failed_transition_witness")
                ]
                if self.selection_policy == "regular" and self.config.joint_group_routing_search_cuts and witnesses:
                    requested = int(getattr(self.config, "joint_group_routing_port_search_count", 8))
                    context_limit = min(32, max(len(self.contexts), requested))
                    additions = (
                        propose_joint_cut_variants(
                            self.part,
                            self.config,
                            maximum_candidates=max(0, context_limit - 1),
                            failure_witnesses=witnesses,
                        )
                        if requested > 0
                        else GroupRouteProposals(reason="port_context_search_disabled")
                    )
                    self.diagnostics["failure_directed_cut_search"] = {
                        **additions.diagnostics,
                        "reasons": getattr(
                            additions, "reasons", (additions.reason,) if hasattr(additions, "reason") else ()
                        ),
                        "witness_count": len(witnesses),
                        "maximum_total_contexts": context_limit,
                    }
                    seen = {_cut_context_fingerprint(context) for _name, context, _values, _evaluator in self.contexts}
                    for context in getattr(additions, "variants", ()):
                        key = _cut_context_fingerprint(context)
                        if key in seen or len(self.contexts) >= context_limit:
                            continue
                        seen.add(key)
                        self.contexts.append(
                            self.context_proposal(f"failure_directed_ports_{len(self.contexts)}", context)
                        )
                    self.diagnostics["failure_directed_cut_search"]["context_count_after_search"] = len(self.contexts)
                for _name, _context, _values, evaluator in self.contexts:
                    if evaluator is not None:
                        evaluator.set_search_level(level)
            maximum_shape_attempts = (
                min(3, self.config.joint_group_routing_max_candidates)
                if self.progressive and level == 0
                else max(6, self.config.joint_group_routing_max_candidates)
                if self.progressive
                else max(6, 3 * self.config.joint_group_routing_max_candidates)
            )
            if level and self.selection_policy == "regular":
                # The physical-preview budget stays unchanged. Separate
                # bounded cheap shaping work lets every new port context
                # reach its two directed winding leaders before adoption.
                maximum_shape_attempts = max(maximum_shape_attempts, 2 * len(self.contexts))
            shape_total = min(
                maximum_shape_attempts, sum(len(values.candidates) for _name, _context, values, _e in self.contexts)
            )
            self.work_progress.start(f"topology shaping (stage {level + 1})", shape_total)
            ordered = []
            for context_name, context, candidate in self.shaped_candidates(maximum_shape_attempts, shape_total):
                try:
                    jumps = raised_span_metrics(
                        candidate.path.v,
                        self.projector,
                        closed=True,
                        segment_geometry=candidate.path.segment_geometry,
                        tolerance_m=self.config.routing_jump_height_tolerance_m,
                        maximum_sample_spacing_m=self.config.routing_jump_sample_spacing_m,
                    )
                    guide = routing_metrics(
                        candidate.path.v,
                        interconnection_axis(self.config, self.part.mesh),
                        self.config.joint_group_routing_b0_weight,
                    )
                except (ValueError, AttributeError) as error:
                    self.diagnostics["attempts"].append(
                        {"context": context_name, "reasons": ["jump_measurement_unavailable"], "detail": str(error)}
                    )
                    continue
                if self.selection_policy == "regular":
                    candidate.regularity_guide = _proposal_regularity(candidate)
                ordered.append((jumps["jump_count"], guide["cost_m"], context_name, context, candidate))
            self.work_progress.finish()
            ordered.sort(
                key=lambda item: (
                    item[0],
                    self.validated_context_counts.get(item[2], 0) if level else 0,
                    *(regularity_order_key(item[4].regularity_guide) if self.selection_policy == "regular" else ()),
                    round(item[1], 12),
                )
            )
            ordered = _order_complete_return_families(ordered, selection_policy=self.selection_policy)
            shaped_count = len(ordered)
            if self.progressive or self.selection_policy == "regular":
                unique, seen = [], set(self.validated_candidates)
                for item in ordered:
                    key = _candidate_fingerprint(item[2], item[4])
                    if key in seen:
                        self.diagnostics["duplicate_candidate_count"] = (
                            self.diagnostics.get("duplicate_candidate_count", 0) + 1
                        )
                    else:
                        seen.add(key)
                        unique.append(item)
                ordered = unique
            remaining = self.config.joint_group_routing_max_candidates - self.candidate_count
            # Reserve physical checks for broader curves when compact routes
            # fail. Acceptance thresholds and total body-preview budget hold.
            stage_budget = (
                min(remaining, max(1, (remaining + 1) // 2)) if self.progressive and level == 0 else remaining
            )
            self.validation_total = min(len(ordered), stage_budget)
            self.validation_stage_start = self.candidate_count
            self.diagnostics["search_stages"].append(
                {
                    "level": level,
                    "shaped_candidates": shaped_count,
                    "distinct_candidates": len(ordered),
                    "physical_candidate_budget": self.validation_total,
                    "field_candidate_budget": min(
                        len(ordered), self.limits.maximum_field_candidates - self.field_candidate_count
                    ),
                }
            )
            self.diagnostics["proposed_jump_budgets"] = sorted(
                set(self.diagnostics.get("proposed_jump_budgets", [])) | {item[0] for item in ordered}
            )
            if self.validation_total:
                self.work_progress.start(f"finalist validation (stage {level + 1})", self.validation_total)
                for item in ordered:
                    if self.candidate_count - self.validation_stage_start >= stage_budget:
                        self.diagnostics["physical_candidate_budget_exhausted"] = True
                        break
                    if self.field_candidate_count >= self.limits.maximum_field_candidates:
                        self.diagnostics["field_screen_budget_exhausted"] = True
                        break
                    yield item
                self.work_progress.finish()

    def prepare_baseline_fields(self):
        self.task.update(self.completed, message="preparing baseline bodies for finalist validation", force=True)
        self.materialize_baselines(self.work_progress)
        field_total = 2 * len(self.selected) + sum(len(original.contour_lines) for original in self.parts)
        self.work_progress.start("baseline fields", field_total)
        field_count = 0
        self.route_fields, self.physical_fields = [], []
        for original in self.selected:
            self.route_fields.append(self.field(original.wire_path.v))
            field_count += 1
            self.work_progress.update(field_count)
        for original in self.physical_selected:
            self.physical_fields.append(self.field(original.fabrication_path.wire_points_m))
            field_count += 1
            self.work_progress.update(field_count)
        self.reference_field = np.zeros_like(self.target_field.coords, dtype=float)
        for original in self.parts:
            for contour in original.contour_lines:
                points = contour_evaluation_points(original.mesh, contour, self.config)
                if len(points) < 3 or not np.all(np.isfinite(points)) or np.linalg.norm(points[0] - points[-1]) > 1e-8:
                    raise ValueError("Original contour field reference requires closed winding contours.")
                self.reference_field += contour.turn_multiplicity * self.field(points)
                field_count += 1
                self.work_progress.update(field_count)
        self.work_progress.finish()

    def validate_candidates(self):
        for proposed_jumps, _guide, context_name, context, candidate in self.ordered_proposals():
            self.field_candidate_count += 1
            if self.progressive or self.selection_policy == "regular":
                self.validated_candidates.add(_candidate_fingerprint(context_name, candidate))
            self.validated_context_counts[context_name] = self.validated_context_counts.get(context_name, 0) + 1
            if self.route_fields is None:
                self.prepare_baseline_fields()
            baseline_part = self.physical_selected[self.index]
            if "longitudinal_rms_T" not in self.baseline_metrics:
                self.baseline_metrics = _route_metrics(baseline_part, self.config)
                self.diagnostics.update(
                    baseline_length_m=self.baseline_metrics["length_m"],
                    baseline_routing_metrics=self.baseline_metrics,
                )
                baseline_field_metrics = longitudinal_field_distortion(
                    self.reference_field,
                    sum(self.physical_fields, np.zeros_like(self.target_field.coords)),
                    self.target_field,
                    self.config.b_0_direction,
                    self.current,
                )
                self.baseline_metrics.update(
                    longitudinal_rms_T=baseline_field_metrics["rms_T"],
                    longitudinal_peak_T=baseline_field_metrics["peak_T"],
                )
                self.diagnostics["regularity_maximum_longitudinal_rms_T"] = (
                    self.baseline_metrics["longitudinal_rms_T"] + self.regular_rms_tolerance
                )
                self.diagnostics["baseline_comparison_policy"] = (
                    "physical"
                    if self.selection_policy == "regular" and self.baseline_metrics["regularity_is_lower_bound"]
                    else self.selection_policy
                )
            if context_name not in self.baselines:
                # Cut contexts were prepared before dense geometry existed.
                # Bind only physical state, preserving their directed cuts and
                # groups. The validator copies these buffers before auditing.
                for name in _PHYSICAL_STATE:
                    setattr(context, name, getattr(baseline_part, name))
                self.baselines[context_name] = prepare_group_route_baseline(
                    context,
                    self.config,
                    self.target_field.coords,
                    self.limits,
                    reuse_existing_sweep=True,
                    audit_geometry=False,
                )
            if self.field_candidate_count == 1:
                self.work_progress.start("finalist validation (stage 1)", self.validation_total)
            self.work_progress.activity(
                message=(
                    f"{context_name}: field screen {self.field_candidate_count}/{self.limits.maximum_field_candidates}; "
                    f"body previews {self.candidate_count}/{self.config.joint_group_routing_max_candidates}"
                ),
                force=True,
            )
            result = validate_group_route_candidate(
                context,
                candidate.path,
                self.config,
                protected_segments=candidate.protected_segments,
                candidate_segment_ids=candidate.segment_ids,
                group_sequence=candidate.group_sequence,
                required_groups=tuple(range(len(self.part.groups))),
                candidate_cuts=candidate.cuts,
                target_points_m=self.target_field.coords,
                maximum_field_delta_t_per_a=self.field_budget,
                **self.longitudinal_kwargs,
                target_field=self.target_field,
                current_a=self.current,
                other_field_per_amp=sum(
                    self.route_fields[: self.index] + self.route_fields[self.index + 1 :],
                    np.zeros_like(self.target_field.coords),
                ),
                other_physical_field_per_amp=sum(
                    self.physical_fields[: self.index] + self.physical_fields[self.index + 1 :],
                    np.zeros_like(self.target_field.coords),
                ),
                apply_normal_shift=False,
                baseline=self.baselines[context_name],
                limits=self.limits,
                progress_callback=self.validation_progress,
            )
            # Production validators record whether expensive body work began.
            # Unmarked validators retain their historical whole-candidate
            # accounting (including third-party validators/test doubles).
            if result.diagnostics.get("body_preview_started", True):
                self.candidate_count += 1
            reasons = list(result.reasons)
            candidate_metrics = None
            if result.accepted:
                try:
                    candidate_metrics = _route_metrics(result.prepared_part, self.config, complete_provenance=True)
                    candidate_physical_field = getattr(result, "physical_field_per_amp", None)
                    if candidate_physical_field is None:
                        # Compatibility with older validators and test doubles.
                        candidate_physical_field = self.field(result.prepared_part.fabrication_path.wire_points_m)
                    candidate_physical_field = np.asarray(candidate_physical_field, dtype=float)
                    candidate_field_metrics = longitudinal_field_distortion(
                        self.reference_field,
                        candidate_physical_field
                        + sum(
                            self.physical_fields[: self.index] + self.physical_fields[self.index + 1 :],
                            np.zeros_like(self.target_field.coords),
                        ),
                        self.target_field,
                        self.config.b_0_direction,
                        self.current,
                    )
                    candidate_metrics.update(
                        longitudinal_rms_T=candidate_field_metrics["rms_T"],
                        longitudinal_peak_T=candidate_field_metrics["peak_T"],
                    )
                    if self.selection_policy == "regular":
                        if candidate_metrics["regularity_is_lower_bound"]:
                            reasons.append("missing_candidate_regularity_provenance")
                        if (
                            candidate_metrics["longitudinal_rms_T"]
                            > self.diagnostics["regularity_maximum_longitudinal_rms_T"]
                        ):
                            reasons.append("regularity_field_rms_tolerance_exceeded")
                except ValueError:
                    reasons.append("missing_candidate_objective_provenance")
                support = certify_group_route_support(result.prepared_part, self.config)
                result.diagnostics["support"] = support.diagnostics
                reasons.extend(support.reasons)
                if not reasons and any(
                    j != self.index and not other.manufacturing_diagnostics.get("manufacturing_ok", False)
                    for j, other in enumerate(self.physical_selected)
                ):
                    # Containment assumes independently valid conductor solids.
                    # A per-part proposal cannot certify or repair another body.
                    reasons.append("other_part_manufacturing_not_certified")
                if not reasons and len(self.parts) > 1:
                    bodies = [
                        result.prepared_part.swept_body if j == self.index else other.swept_body
                        for j, other in enumerate(self.physical_selected)
                    ]
                    gap = (
                        2 * float(self.config.sweep_insulation_thickness)
                        + float(self.config.sweep_manufacturing_tolerance)
                        + float(self.config.sweep_bend_clearance_margin)
                    )
                    audit, collision_reason = audit_interpart_clearance(bodies, gap, self.limits)
                    result.diagnostics["whole_coil_clearance"] = audit
                    if collision_reason:
                        reasons.append(collision_reason)
                if not reasons:
                    if self.baseline_certification_failure:
                        reasons.extend(self.baseline_certification_failure)
                    elif self.baseline_feasibility is None:
                        # A field/body/support-rejected candidate cannot be
                        # adopted, so its incumbent needs no independent audit.
                        # Certify the unchanged incumbent before granting a
                        # repair exception or comparing objective values.
                        self.audit_progress.prefix = f"part {self.index + 1}/{len(self.parts)} baseline"
                        self.work_progress.activity(message="certifying incumbent before route adoption", force=True)
                        reference = self.baselines[context_name]
                        certified = prepare_group_route_baseline(
                            context,
                            self.config,
                            self.target_field.coords,
                            self.limits,
                            reuse_existing_sweep=True,
                            progress_callback=self.validation_progress,
                        )
                        if certified.fingerprint != reference.fingerprint:
                            self.baseline_certification_failure = ("stale_baseline",)
                        elif certified.physical_field_per_amp is None:
                            self.baseline_certification_failure = ("baseline_physical_field_unavailable",)
                        elif not np.array_equal(
                            certified.physical_field_per_amp, reference.physical_field_per_amp, equal_nan=True
                        ):
                            self.baseline_certification_failure = ("baseline_physical_field_changed",)
                        else:
                            self.certified_baseline = certified
                            self.baseline_feasibility = assess_routing_baseline(
                                baseline_part,
                                self.physical_selected,
                                self.index,
                                self.config,
                                self.target_field,
                                self.current,
                                self.physical_fields,
                                self.route_fields,
                                certified,
                                self.limits,
                            )
                            self.diagnostics["baseline_feasibility"] = self.baseline_feasibility
                            self.diagnostics["baseline_certification_deferred"] = False
                            self.diagnostics["selected_route_status"] = (
                                "valid" if self.baseline_feasibility["accepted"] else "invalid"
                            )
                            self.diagnostics["validation_basis"] = (
                                "independent_baseline_fixed_current_and_finite_body_checks"
                            )
                        reasons.extend(self.baseline_certification_failure)
                        self.audit_progress.prefix = f"part {self.index + 1}/{len(self.parts)} finalist"
                    if self.certified_baseline is not None:
                        result.diagnostics["baseline_manufacturing_reasons"] = self.certified_baseline.reasons
                        result.diagnostics["baseline_geometry_audit_deferred"] = False
                if (
                    not reasons
                    and self.baseline_feasibility["accepted"]
                    and not route_is_better(
                        candidate_metrics, self.baseline_metrics, policy=self.diagnostics["baseline_comparison_policy"]
                    )
                ):
                    reasons.append("no_routing_objective_or_manufacturing_improvement")
            self.diagnostics["attempts"].append(
                {
                    "context": context_name,
                    "group_sequence": candidate.group_sequence,
                    "variants": candidate.variant_names,
                    "length_m": candidate.length_m,
                    "routing_metrics": candidate_metrics,
                    "proposed_jump_count": proposed_jumps,
                    "proposed_return_family_counts": getattr(candidate, "joint_return_family_counts", {}),
                    "proposed_join_family_consistency": getattr(candidate, "joint_join_family_diagnostics", {}),
                    "reasons": reasons,
                    "validation": result.diagnostics,
                    "eligible_for_adoption": not reasons and result.accepted,
                    "selected": False,
                }
            )
            if not reasons and result.accepted:
                if self.best is None or route_is_better(candidate_metrics, self.best[1], policy=self.selection_policy):
                    self.best = (
                        result.prepared_part,
                        candidate_metrics,
                        context_name,
                        len(self.diagnostics["attempts"]) - 1,
                        candidate_physical_field,
                    )
            self.work_progress.update(
                self.candidate_count - self.validation_stage_start,
                message="accepted" if not reasons and result.accepted else "rejected",
                force=True,
            )
            if self.candidate_count >= self.config.joint_group_routing_max_candidates:
                self.diagnostics["physical_candidate_budget_exhausted"] = True
                break
        if self.validation_total:
            self.work_progress.finish()

    def adopt(self):
        # All alternatives for one part were compared against the same
        # baseline and other-part backgrounds. Commit only the winner once.
        if self.best is not None:
            self.selected[self.index], metrics, context_name, attempt_index, accepted_physical_field = self.best
            self.selected[self.index].mesh = self.part.mesh
            self.physical_selected[self.index] = self.selected[self.index]
            self.selected[self.index].pcb_layout = None
            self.route_fields[self.index] = self.field(self.selected[self.index].wire_path.v)
            self.physical_fields[self.index] = accepted_physical_field
            self.diagnostics["attempts"][attempt_index]["selected"] = True
            field_validation = self.diagnostics["attempts"][attempt_index]["validation"]
            absolute_status = [
                field_validation.get(f"{stage}_absolute_field_contract_met") for stage in ("route", "physical")
            ]
            self.diagnostics.update(
                status="accepted",
                reason="validated_routing_improvement",
                selection_basis="objective_improvement" if self.baseline_feasibility["accepted"] else "baseline_repair",
                selected_route_status="valid",
                selected_cut_context=context_name,
                selected_attempt_index=attempt_index,
                validation_basis="independent_fixed_current_and_finite_body_checks",
                selected_manufacturing_failure_reasons="",
                selected_length_m=metrics["length_m"],
                selected_routing_metrics=metrics,
                selected_join_family_consistency=self.diagnostics["attempts"][attempt_index].get(
                    "proposed_join_family_consistency", {}
                ),
                selected_physical_minimum_centerline_distance_m=(
                    value
                    if (
                        value := self.selected[self.index].manufacturing_diagnostics.get(
                            "sweep_capsule_min_centerline_distance_m"
                        )
                    )
                    and np.isfinite(value)
                    and value > 0
                    else None
                ),
                selected_physical_clearance_lower_bound_m=self.physical_clearance,
                # Routing acceptance under explicit nonregression is distinct
                # from overall field qualification, which final reports retain.
                selected_absolute_field_contract_met=(
                    all(absolute_status)
                    if all(isinstance(value, (bool, np.bool_)) for value in absolute_status)
                    else None
                ),
                selected_absolute_field_contract_reasons={
                    stage: field_validation.get(f"{stage}_absolute_field_contract_reasons", "")
                    for stage in ("route", "physical")
                },
                selected_inherited_field_contract_violations={
                    stage: field_validation.get(f"{stage}_inherited_field_contract_violations", "")
                    for stage in ("route", "physical")
                },
            )
            if self.diagnostics["field_acceptance_policy"] == "budget_only":
                for key in (
                    "budget_waived_field_contract_violations",
                    "newly_failed_field_contract_violations",
                    "regressed_inherited_field_contract_violations",
                ):
                    self.diagnostics[f"selected_{key}"] = {
                        stage: field_validation.get(f"{stage}_{key}", "") for stage in ("route", "physical")
                    }

    def report(self):
        self.diagnostics.update(
            candidate_count=self.candidate_count,
            body_preview_candidate_count=self.candidate_count,
            field_candidate_count=self.field_candidate_count,
            maximum_field_candidates=self.limits.maximum_field_candidates,
            elapsed_s=time.perf_counter() - self.start_time,
        )
        self.diagnostics["transition_searches"] = [
            {"context": name, **getattr(evaluator, "diagnostics", {})}
            for name, _context, _values, evaluator in self.contexts
            if evaluator is not None
        ]
        if not any(values.candidates for _name, _context, values, _evaluator in self.contexts):
            self.diagnostics["reason"] = self.proposals.reason
        elif self.diagnostics["status"] == "baseline_retained":
            self.diagnostics["reason"] = "no_validated_improvement"
        self.diagnostics["accepted_improvement_found_within_budget"] = self.best is not None
        self.diagnostics["infeasibility_proven"] = False
        work_reasons = sorted(
            {
                reason
                for attempt in self.diagnostics["attempts"]
                for reason in attempt.get("reasons", [])
                if _work_budget_reason(reason)
            }
        )
        if self.diagnostics.get("field_screen_budget_exhausted"):
            work_reasons.append("field_screen_candidate_budget_exhausted")
        for _name, _context, values, evaluator in self.contexts:
            records = (values.diagnostics, getattr(evaluator, "diagnostics", {}))
            for record in records:
                reason = record.get("budget_exhaustion_reason")
                if reason and reason not in work_reasons:
                    work_reasons.append(reason)
            if _work_budget_reason(values.reason) and values.reason not in work_reasons:
                work_reasons.append(values.reason)
        work_reasons.sort()
        self.diagnostics["work_budget_exhausted"] = bool(work_reasons)
        self.diagnostics["work_budget_exhaustion_reasons"] = work_reasons
        self.diagnostics["search_failure_scope"] = (
            "work_budget"
            if work_reasons
            else "first_stage_bend_radius_fail_fast"
            if self.diagnostics.get("fail_fast")
            else "bounded_proposal_coverage"
            if self.diagnostics.get("shape_search_budget_exhausted")
            else "physical_candidate_budget"
            if self.diagnostics.get("physical_candidate_budget_exhausted")
            else "bounded_port_and_curve_families"
        )
        self.selected[self.index].joint_route_diagnostics = self.diagnostics
        self.task.advance(
            message=f"part {self.index + 1}/{len(self.parts)}: {self.diagnostics['status']}; {self.diagnostics['reason']}",
            force=True,
        )
