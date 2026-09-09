from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamFunctionOptimizationScheme:
    """User-readable description of an implemented optimization scheme."""

    name: str
    cost_function: str
    formula: str
    description: str


STREAM_FUNCTION_OPTIMIZATION_SCHEMES: tuple[StreamFunctionOptimizationScheme, ...] = (
    StreamFunctionOptimizationScheme(
        name="tikhonov",
        cost_function="field_error_plus_resistance",
        formula=("min_x ||A x - b||_2^2 + lambda^2 x^T R x + eta x^T C M^-1 C x + mu x^T S x"),
        description=(
            "Default linear stream-function solve. The optional smoothness and "
            "turn-spacing quadratic terms are active only when their factors are positive."
        ),
    ),
    StreamFunctionOptimizationScheme(
        name="target_error_constrained_efficiency",
        cost_function="minimum_resistance_subject_to_field_error",
        formula="search the Tikhonov family for low x^T R x within the configured field-error budgets",
        description=(
            "Restricted scalar regularization search. For one matching L2 residual bound and "
            "no extra objective penalties it parameterizes the constrained minimum-sheet-power "
            "solution. Separate constraints or different residual metrics require the direct "
            "conic method for a general constrained optimum."
        ),
    ),
    StreamFunctionOptimizationScheme(
        name="constrained_quadratic",
        cost_function="minimum_resistance_subject_to_field_error_socp",
        formula="min_x x^T Q x, or min_{x,t} t with ||K_face x||_2 <= t; subject to physical field and engineering bounds",
        description=(
            "Direct fixed-error constrained solve using CVXPY and an installed conic solver "
            "(CLARABEL, ECOS, or SCS). Objectives: sheet power plus optional quadratic penalties, "
            "smoothness priority, or pure minimax vector current density. A scalar-search fallback "
            "is diagnostic and cannot certify the full conic contract."
        ),
    ),
)


def available_stream_function_optimization_schemes() -> tuple[StreamFunctionOptimizationScheme, ...]:
    return STREAM_FUNCTION_OPTIMIZATION_SCHEMES


def stream_function_optimization_scheme(name: str) -> StreamFunctionOptimizationScheme | None:
    normalised = str(name).strip().lower().replace("-", "_")
    for scheme in STREAM_FUNCTION_OPTIMIZATION_SCHEMES:
        if scheme.name == normalised:
            return scheme
    return None
