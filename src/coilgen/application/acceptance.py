"""Active solver allowance controls without changing acceptance mathematics."""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

from ..config import CoilGenConfig


def field_acceptance_slack_T(config: CoilGenConfig) -> float:
    """Return the absolute field allowance used by the active optimization solver.

    Independent research validation currently always uses ResearchPhysicsSpec,
    including its defaults for native solver studies. Therefore, for native
    studies this solver allowance differs from that independent-check policy.
    The helper exposes the existing behavior and does not alter that policy.
    """
    if config.research_physics is None:
        return float(config.target_error_constraint_acceptance_absolute_T)
    from ..research.convex import ResearchPhysicsSpec

    return float(ResearchPhysicsSpec.from_dict(config.research_physics).acceptance_absolute_by_unit.get("T", 0.0))


def field_acceptance_slack_overrides(config: CoilGenConfig, allowance_T: float) -> dict:
    """Edit the active allowance, retaining all other physical spec values."""
    if (
        isinstance(allowance_T, bool)
        or not isinstance(allowance_T, Real)
        or not math.isfinite(allowance_T)
        or allowance_T < 0
    ):
        raise ValueError("Absolute field acceptance slack must be finite and nonnegative [T].")
    allowance_T = float(allowance_T)
    if config.research_physics is None:
        return {"target_error_constraint_acceptance_absolute_T": allowance_T}
    from ..research.convex import ResearchPhysicsSpec

    spec = ResearchPhysicsSpec.from_dict(config.research_physics)
    physics = (
        spec.to_dict()
        if isinstance(config.research_physics, ResearchPhysicsSpec)
        else deepcopy(dict(config.research_physics))
    )
    physics["acceptance_absolute_by_unit"] = {**spec.acceptance_absolute_by_unit, "T": allowance_T}
    ResearchPhysicsSpec.from_dict(physics)
    return {"research_physics": physics}


def independent_field_acceptance_slack_T(config: CoilGenConfig) -> float:
    """Report the existing independent-check allowance for either solver mode.

    This intentionally follows research.validation.field_budget_check, which
    currently resolves a default physical spec even for native studies.
    """
    from ..research.convex import ResearchPhysicsSpec

    return float(ResearchPhysicsSpec.from_dict(config.research_physics or {}).acceptance_absolute_by_unit.get("T", 0.0))
