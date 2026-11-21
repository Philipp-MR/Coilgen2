"""Independent engineering checks for manually revised winding circuits."""

from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class EditorValidation:
    identity: str
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)
    parts: tuple = ()


def validate_winding_revision(identity, sources, plan, config, target):
    """Use production swept bodies, exact collisions and same-current field gates.

    The aggregate field is formed in tesla using each recorded contour current.
    The existing field assessor receives that field at unit scaling: no fitted
    current or offset is introduced. A failed baseline never waives a new limit.
    """
    from .group_circuits import review_circuit_clearance
    from .group_route_fields import assess_group_route_field
    from .group_route_support import certify_group_route_support
    from .group_route_validation import GroupRouteValidationLimits, _sweep_preview
    from .interconnection_editor import apply_editor_plan
    from .physics import biot_savart_wire_path
    from .progress import DesignCancelled
    from .routing_field_objective import longitudinal_field_distortion

    reasons, diagnostics, parts = [], {"current_refitted": False, "circuits": []}, []
    try:
        parts = apply_editor_plan(sources, plan)
        if target is None or len(target.coords) > GroupRouteValidationLimits().maximum_target_points:
            raise ValueError("Target field is missing or exceeds the validation point budget.")
        targets = np.asarray(target.coords)
        if targets.ndim != 2 or targets.shape[1] != 3 or not len(targets) or not np.all(np.isfinite(targets)):
            raise ValueError("The validation target needs finite physical coordinates.")
        active = (
            (np.asarray(target.weights) > 0) & np.isfinite(target.weights) & ~np.asarray(target.suppressed, dtype=bool)
        )
        if not active.any():
            raise ValueError("No active target samples are available for the routing field budget.")
        scale = float(np.max(np.linalg.norm(target.b[active], axis=1)))
        if not np.isfinite(scale):
            raise ValueError("Target normalization is nonfinite.")
        preview_config = replace(config, skip_sweep=False, contour_only_sweep_mode=False, progress_flag=False)
        before_route, after_route, before_physical, after_physical = [
            np.zeros_like(targets, dtype=float) for _ in range(4)
        ]
        physical_complete = True
        for index, (source, part) in enumerate(zip(sources, parts), 1):
            current = source.contour_step
            if current is None or not np.isfinite(current) or current == 0:
                raise ValueError(f"Circuit {index} has no finite nonzero recorded winding current.")

            def field(points):
                return current * biot_savart_wire_path(points, targets, worker_count=1, max_chunk_memory_mb=32.0)

            before_route += field(source.wire_path.v)
            after_route += field(part.wire_path.v)
            metrics, failures = _sweep_preview(part, preview_config, GroupRouteValidationLimits())
            reasons.extend(f"Circuit {index}: {reason}" for reason in failures)
            support = certify_group_route_support(part, preview_config)
            metrics["independent_support"] = support.diagnostics
            reasons.extend(f"Circuit {index}: {reason}" for reason in support.reasons)
            diagnostics["circuits"].append({"index": index, "current_A": current, "manufacturing": metrics})
            # Always regenerate the reference from its exact automatic route;
            # reopening a manual run must not compare against stale physical data.
            reference = deepcopy(source)
            _sweep_preview(reference, preview_config, GroupRouteValidationLimits())
            if reference.fabrication_path is None or part.fabrication_path is None:
                physical_complete = False
            else:
                before_physical += field(reference.fabrication_path.wire_points_m)
                after_physical += field(part.fabrication_path.wire_points_m)
        budget = float(config.joint_group_routing_max_field_change_rel) * scale
        longitudinal = config.joint_group_routing_max_longitudinal_field_change_rel
        options = (
            {} if longitudinal is None else {"maximum_longitudinal_field_delta_t_per_a": float(longitudinal) * scale}
        )
        stages = [("route", before_route, after_route)]
        if physical_complete:
            stages.append(("physical", before_physical, after_physical))
        else:
            reasons.append("Physical fixed-current field comparison is incomplete")
        for stage, before, after in stages:
            assessment = assess_group_route_field(before, after, target, config, 1.0, budget, **options)
            diagnostics[stage] = {
                **assessment.metrics,
                "vector_delta_budget_T": budget,
                "longitudinal_change": longitudinal_field_distortion(before, after, target, config.b_0_direction, 1.0),
            }
            # Display witness only: preserve the assessor's exact acceptance logic.
            delta = np.linalg.norm(after - before, axis=1)
            budget_samples = np.isfinite(target.weights) & (np.asarray(target.weights) > 0)
            worst = int(np.argmax(np.where(budget_samples, delta, -np.inf)))
            diagnostics[stage]["maximum_delta_witness"] = dict(
                point_m=targets[worst].tolist(), delta_T=float(delta[worst]), budget_T=budget
            )
            reasons.extend(f"{stage}: {reason}" for reason in assessment.reasons)
        assembly = review_circuit_clearance(parts, preview_config)
        diagnostics["assembly"] = assembly
        if not assembly.get("checked") or not assembly.get("passed"):
            reasons.append("Assembly clearance did not pass")
    except DesignCancelled:
        raise
    except (ValueError, TypeError, RuntimeError, FloatingPointError, IndexError) as error:
        reasons.append(str(error))
    return EditorValidation(identity, not reasons, tuple(dict.fromkeys(reasons)), diagnostics, tuple(parts))
