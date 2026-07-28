from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .conversion import coil_design_problem_to_config
from .problem import CoilDesignProblem


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Generic result envelope for high-level solver implementations."""

    status: str
    solution: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status.lower() in {"ok", "success", "converged"}


@runtime_checkable
class CoilDesignSolver(Protocol):
    """Protocol for solvers that consume a ``CoilDesignProblem``."""

    name: str

    def solve(self, problem: CoilDesignProblem) -> SolverResult:
        """Solve a high-level coil-design problem."""


@dataclass(slots=True)
class PipelineCoilDesignSolver:
    """Adapter that runs a ``CoilDesignProblem`` through the current pipeline."""

    name = "coilgen_pipeline"
    project_root: str | Path | None = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    def solve(self, problem: CoilDesignProblem) -> SolverResult:
        problem.validate()
        config = coil_design_problem_to_config(problem, config_overrides=self.config_overrides)
        if self.dry_run:
            return SolverResult(
                status="ok", solution=config, diagnostics=problem.summary(), message="Converted to CoilGenConfig."
            )
        from .pipeline import run_coilgen

        result = run_coilgen(config, project_root=Path(self.project_root) if self.project_root is not None else None)
        return SolverResult(
            status="ok", solution=result, diagnostics=problem.summary(), message="Pipeline run completed."
        )
