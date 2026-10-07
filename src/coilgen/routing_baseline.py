"""Certify whether an existing route is a feasible optimization incumbent."""

from __future__ import annotations

import numpy as np

from .group_route_clearance import audit_interpart_clearance
from .group_route_fields import assess_group_route_field
from .group_route_support import certify_group_route_support


def assess_routing_baseline(
    part,
    parts,
    index,
    config,
    target_field,
    current_a,
    physical_fields,
    route_fields,
    prepared_baseline,
    limits,
):
    """Apply baseline feasibility gates before demanding objective improvement.

    This is not a change budget against source contours. The existing whole-coil
    fields are compared with themselves at the unchanged current to enforce only
    configured target/suppression/linearity contracts. A failed or unchecked
    incumbent cannot veto an independently certified repair on objective value.
    No geometry, field cache, or source diagnostics are mutated.
    """
    diagnostics = {"accepted": False, "reasons": []}
    reasons = diagnostics["reasons"]
    try:
        if not 0 <= index < len(parts) or len(physical_fields) != len(parts) or len(route_fields) != len(parts):
            raise ValueError("Baseline field caches must cover every part and identify the selected part.")
        reasons.extend(prepared_baseline.reasons)
        diagnostics["manufacturing"] = dict(prepared_baseline.diagnostics)
        if not prepared_baseline.diagnostics.get("manufacturing_ok", False):
            reasons.append("baseline_manufacturing_not_certified")
        if not prepared_baseline.diagnostics.get("exact_collision_completed", False):
            reasons.append("baseline_exact_collision_unchecked")
        support = certify_group_route_support(part, config)
        diagnostics["support"] = support.diagnostics
        if not support.accepted:
            reasons.extend(support.reasons or ("baseline_support_not_certified",))

        for stage, own, fields in (
            ("route", prepared_baseline.route_field_per_amp, route_fields),
            ("physical", prepared_baseline.physical_field_per_amp, physical_fields),
        ):
            arrays = [own if j == index else values for j, values in enumerate(fields)]
            if any(values is None or np.shape(values) != np.shape(target_field.coords) for values in arrays):
                reasons.append(f"baseline_{stage}_field_unavailable")
                continue
            # The field gate owns the active sample mask. Nonfinite arithmetic
            # at explicitly inactive samples must not invalidate an incumbent.
            with np.errstate(over="ignore", invalid="ignore"):
                whole_field = np.sum(np.asarray(arrays, dtype=float), axis=0)
            assessment = assess_group_route_field(whole_field, whole_field, target_field, config, current_a, 0.0)
            diagnostics[f"{stage}_field"] = assessment.metrics
            if not assessment.accepted:
                reasons.extend(f"{stage}_{reason}" for reason in assessment.reasons)
            elif not assessment.metrics.get("checked", False):
                reasons.append(f"baseline_{stage}_field_unchecked")

        if len(parts) > 1:
            other_bodies_valid = all(
                j == index or other.manufacturing_diagnostics.get("manufacturing_ok", False)
                for j, other in enumerate(parts)
            )
            if not other_bodies_valid:
                reasons.append("other_part_manufacturing_not_certified")
            else:
                gap = (
                    2 * float(config.sweep_insulation_thickness)
                    + float(config.sweep_manufacturing_tolerance)
                    + float(config.sweep_bend_clearance_margin)
                )
                bodies = [part.swept_body if j == index else other.swept_body for j, other in enumerate(parts)]
                audit, collision_reason = audit_interpart_clearance(bodies, gap, limits)
                diagnostics["whole_coil_clearance"] = audit
                if collision_reason:
                    reasons.append(collision_reason)
                elif not audit.get("interpart_clearance_completed", False):
                    reasons.append("baseline_interpart_clearance_unchecked")
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
        FloatingPointError,
        OverflowError,
    ) as error:
        reasons.append("baseline_certification_incomplete")
        diagnostics["detail"] = str(error)
    diagnostics["reasons"] = list(dict.fromkeys(reasons))
    diagnostics["accepted"] = not diagnostics["reasons"]
    return diagnostics
