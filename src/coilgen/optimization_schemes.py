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
        formula=("min_x ||A x - b||_2^2 + ||lambda R x||_2^2 + eta x^T C M^-1 C x + mu x^T S x"),
        description=(
            "Default linear stream-function solve. The optional smoothness and "
            "turn-spacing quadratic terms are active only when their factors are positive."
        ),
    ),
    StreamFunctionOptimizationScheme(
        name="target_error_constrained_efficiency",
        cost_function="minimum_resistance_subject_to_field_error",
        formula="min_x ||R x||_2 subject to ||A x - b||_2 <= epsilon ||b||_2",
        description=(
            "Selects the highest-efficiency stream function that still satisfies a user-specified "
            "weighted target-field error budget. This is useful for geometry studies where target "
            "error should be held fixed while comparing efficiency."
        ),
    ),
    StreamFunctionOptimizationScheme(
        name="constrained_quadratic",
        cost_function="minimum_resistance_subject_to_field_error_socp",
        formula="min_x ||R x||_2^2 subject to ||A x - b||_2 <= epsilon ||b||_2",
        description=(
            "Direct fixed-error constrained solve using CVXPY and an installed conic solver "
            "(CLARABEL, ECOS, or SCS). If CVXPY is unavailable, CoilGen falls back to the "
            "Tikhonov multiplier search and records that fallback in the diagnostics."
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
