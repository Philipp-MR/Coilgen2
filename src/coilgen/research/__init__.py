"""Headless physical optimization, reproducible studies and fabrication checks.

Use ``CoilGenConfig.research_physics`` to opt into physical convex objectives.
The existing ``coilgen`` API keeps its default numerical behavior. Optional
solver and acceleration dependencies are loaded when those paths execute.
"""

from .convex import (
    FieldScenario,
    LinearFieldConstraint,
    ParameterizedResearchProblem,
    ResearchPhysicsSpec,
    preflight_research_physics,
)
from .physics import EnergyQuadrature
from .materials import WireSheetMaterial
from .requests import ResearchRequest
from .application import run_design
from .jobs import JobManager, RuntimePolicy
from .studies import ResearchStudy, ToleranceStudy, pareto_front

__all__ = [
    "EnergyQuadrature",
    "FieldScenario",
    "LinearFieldConstraint",
    "ParameterizedResearchProblem",
    "ResearchPhysicsSpec",
    "preflight_research_physics",
    "WireSheetMaterial",
    "ResearchRequest",
    "run_design",
    "JobManager",
    "RuntimePolicy",
    "ResearchStudy",
    "ToleranceStudy",
    "pareto_front",
]
