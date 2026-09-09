"""Supported backends for CoilGen's general conic stream-function formulation.

This is deliberately not a catalog of every CVXPY solver. Quadratic-only
backends cannot handle the vector current-density and residual-norm constraints
used by this formulation. Backend installation does not certify a successful
solve; solver status and physical residuals must still be checked afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Iterable


_SOLVERS = (
    ("CLARABEL", "interior_point"),
    ("ECOS", "interior_point"),
    ("SCS", "operator_splitting"),
)


@dataclass(frozen=True, slots=True)
class ConicSolverCapability:
    """One supported SOC backend and its current CVXPY installation status.

    ``auto_priority`` is one-based and preserves CoilGen's fallback order;
    it is not a performance ranking for every problem. ``installed`` means
    CVXPY detects the backend, without running an optimization problem.
    """

    name: str
    method: str
    installed: bool
    auto_priority: int


def supported_conic_solvers() -> tuple[str, ...]:
    """Return supported names in automatic order, without importing CVXPY."""

    return tuple(name for name, _method in _SOLVERS)


def conic_solver_capabilities(*, installed_solvers: Iterable[str] | None = None) -> tuple[ConicSolverCapability, ...]:
    """Describe supported solvers, discovering optional CVXPY only on demand.

    Supplying installed names avoids imports and lets an existing CVXPY caller
    reuse its own discovery. Missing CVXPY produces supported-but-uninstalled
    entries; other discovery failures propagate instead of hiding a broken
    environment. Unlisted backends are not supported by this adapter.
    """

    if installed_solvers is None:
        try:
            cvxpy = import_module("cvxpy")
        except ModuleNotFoundError as exc:
            if exc.name != "cvxpy":
                raise
            installed_solvers = ()
        else:
            installed_solvers = cvxpy.installed_solvers()
    installed = {str(name).strip().upper() for name in installed_solvers}
    return tuple(
        ConicSolverCapability(name, method, name in installed, priority)
        for priority, (name, method) in enumerate(_SOLVERS, start=1)
    )


def conic_solver_order(requested: str = "auto", *, installed_solvers: Iterable[str] | None = None) -> tuple[str, ...]:
    """Select supported installed backends without changing the formulation.

    Automatic selection can return an empty tuple. An explicit unsupported or
    unavailable backend fails rather than silently selecting another solver.
    """

    name = str(requested).strip().upper()
    if name not in {"", "AUTO", *supported_conic_solvers()}:
        detail = " OSQP does not support second-order cone constraints." if name == "OSQP" else ""
        raise ValueError(
            f"Solver '{name}' is not supported for CoilGen's general conic stream-function formulation."
            f"{detail} Choose auto, CLARABEL, ECOS, or SCS."
        )
    capabilities = conic_solver_capabilities(installed_solvers=installed_solvers)
    available = tuple(capability.name for capability in capabilities if capability.installed)
    if name in {"", "AUTO"}:
        return available
    if name not in available:
        raise RuntimeError(f"Requested CVXPY solver '{name}' is not installed.")
    return (name,)
