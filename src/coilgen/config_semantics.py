"""Shared spelling and unit semantics; no solvers or UI dependencies.

Normalization preserves unknown draft values. Validators and solver entry
points remain responsible for rejecting unsupported values. Historical input
spellings are retained separately so validation acceptance does not broaden.
"""

from types import MappingProxyType

OPTIMIZATION_METHOD_ALIASES = {
    **dict.fromkeys(("tikkonov", "tikhonov", "tikhonov_regularized_least_squares"), "tikhonov"),
    **dict.fromkeys(
        (
            "target_error_constrained_efficiency",
            "efficiency_with_target_error_constraint",
            "fixed_error_efficiency",
            "minimum_resistance_subject_to_field_error",
        ),
        "target_error_constrained_efficiency",
    ),
    **dict.fromkeys(
        (
            "constrained_quadratic",
            "cvxpy_constrained",
            "minimum_resistance_constrained",
            "minimum_resistance_subject_to_field_error_socp",
            "direct_fixed_error",
            "socp",
            "second_order_cone",
        ),
        "constrained_quadratic",
    ),
}
OBJECTIVE_MODE_ALIASES = {
    **dict.fromkeys(("balanced", "resistance", "resistance_plus_penalties"), "balanced"),
    **dict.fromkeys(("smoothness_priority", "smoothest_feasible", "contourability_priority"), "smoothness_priority"),
    "minimax_current_density": "minimax_current_density",
}

FIELD_BUDGET_METRIC_ALIASES = {
    "norm": "norm",
    "l2": "norm",
    "l2_norm": "norm",
    "residual_norm": "norm",
    "rms": "rms",
    "root_mean_square": "rms",
    "std": "std",
    "standard_deviation": "std",
    "homogeneity": "std",
}
OPTIMIZATION_METHOD_ALIASES = MappingProxyType(OPTIMIZATION_METHOD_ALIASES)
OBJECTIVE_MODE_ALIASES = MappingProxyType(OBJECTIVE_MODE_ALIASES)
FIELD_BUDGET_METRIC_ALIASES = MappingProxyType(FIELD_BUDGET_METRIC_ALIASES)
OPTIMIZATION_METHOD_INPUTS = frozenset(OPTIMIZATION_METHOD_ALIASES) | {"target-error-constrained-efficiency"}
OBJECTIVE_MODE_INPUTS = frozenset(OBJECTIVE_MODE_ALIASES) | {"smoothness-priority", "minimax-current-density"}
FIELD_BUDGET_METRIC_INPUTS = frozenset(FIELD_BUDGET_METRIC_ALIASES) | {"standard-deviation"}
LENGTH_UNITS_TO_M = MappingProxyType({"m": 1.0, "mm": 1e-3, "cm": 1e-2})
VECTOR_FIELD_UNITS_TO_T = MappingProxyType({"T": 1.0, "mT": 1e-3, "uT": 1e-6, "µT": 1e-6})


def normalized_key(value):
    return str(value).strip().lower().replace("-", "_")


def normalize_optimization_method(value):
    key = normalized_key(value)
    return OPTIMIZATION_METHOD_ALIASES.get(key, key)


def normalize_field_budget_metric(value):
    key = normalized_key(value)
    return FIELD_BUDGET_METRIC_ALIASES.get(key, key)


def normalize_objective_mode(value):
    key = normalized_key(value)
    return OBJECTIVE_MODE_ALIASES.get(key, key)
