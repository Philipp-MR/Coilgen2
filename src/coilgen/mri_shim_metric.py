"""Acquisition-agnostic MRI B0 shim quality metric.

The metric is independent of the correction hardware and of sequence parameters
such as TE, TR, echo spacing, phase-encode bandwidth, voxel size, or encoding
direction.  It operates on the combined scalar B0 residual in Hz and combines:

* weighted mean-free residual RMS (bulk homogeneity),
* weighted CVaR of absolute residual (artifact-prone tail), and
* an isotropic spatial-gradient RMS scaled by the ROI size (local roughness).

The last term is rotationally invariant and has units of Hz after multiplication
by the geometry-derived ROI length.  Sequence-specific EPI displacement and
signal-loss predictions can be evaluated separately when an acquisition is
known; they are deliberately not assumptions of this optimization objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

try:
    from scipy import sparse
    from scipy.spatial import cKDTree
except ModuleNotFoundError as exc:  # SciPy is an optional spatial dependency.
    sparse = None  # type: ignore[assignment]
    cKDTree = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _SCIPY_IMPORT_ERROR = None

from .field_units import PROTON_GYROMAGNETIC_RATIO_HZ_PER_T


@dataclass(frozen=True, slots=True)
class MRIShimMetricOptions:
    """Settings for the protocol-independent residual-risk score."""

    mode: str = "off"
    cvar_alpha: float = 0.95
    residual_rms_weight: float = 1.0
    residual_cvar_weight: float = 0.25
    spatial_gradient_weight: float = 0.10
    spatial_scale_m: float = 0.0
    reference_frequency_hz: float = 50.0
    derivative_neighbor_count: int = 18

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower().replace("-", "_")
        if mode not in {"off", "report", "optimize"}:
            raise ValueError("MRI shim metric mode must be 'off', 'report', or 'optimize'.")
        object.__setattr__(self, "mode", mode)
        alpha = float(self.cvar_alpha)
        if not np.isfinite(alpha) or not 0.5 <= alpha < 1.0:
            raise ValueError("mri_shim_cvar_alpha must be finite and in [0.5, 1).")
        for name in (
            "residual_rms_weight",
            "residual_cvar_weight",
            "spatial_gradient_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if mode == "optimize" and self.objective_weight_sum <= 0.0:
            raise ValueError("At least one MRI shim objective weight must be positive in optimize mode.")
        spatial_scale = float(self.spatial_scale_m)
        if not np.isfinite(spatial_scale) or spatial_scale < 0.0:
            raise ValueError("mri_shim_spatial_scale_m must be finite and non-negative (zero selects auto).")
        reference = float(self.reference_frequency_hz)
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("mri_shim_reference_frequency_hz must be finite and positive.")
        if int(self.derivative_neighbor_count) < 4:
            raise ValueError("mri_shim_derivative_neighbor_count must be at least 4.")
        object.__setattr__(self, "derivative_neighbor_count", int(self.derivative_neighbor_count))

    @property
    def report_enabled(self) -> bool:
        return self.mode in {"report", "optimize"}

    @property
    def optimize_enabled(self) -> bool:
        return self.mode == "optimize"

    @property
    def objective_weight_sum(self) -> float:
        return float(self.residual_rms_weight + self.residual_cvar_weight + self.spatial_gradient_weight)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, config: Any) -> "MRIShimMetricOptions":
        return cls(
            mode=getattr(config, "mri_shim_metric_mode", "off"),
            cvar_alpha=getattr(config, "mri_shim_cvar_alpha", 0.95),
            residual_rms_weight=getattr(config, "mri_shim_residual_rms_weight", 1.0),
            residual_cvar_weight=getattr(config, "mri_shim_residual_cvar_weight", 0.25),
            spatial_gradient_weight=getattr(config, "mri_shim_spatial_gradient_weight", 0.10),
            spatial_scale_m=getattr(config, "mri_shim_spatial_scale_m", 0.0),
            reference_frequency_hz=getattr(config, "mri_shim_reference_frequency_hz", 50.0),
            derivative_neighbor_count=getattr(config, "mri_shim_derivative_neighbor_count", 18),
        )


@dataclass(frozen=True, slots=True)
class MRIShimMetricModel:
    """Prepared metric operators for one weighted target point cloud."""

    options: MRIShimMetricOptions
    coords_m: np.ndarray
    normalized_weights: np.ndarray
    gradient_operators_per_m: tuple[Any, Any, Any]
    spatial_scale_m: float

    @classmethod
    def from_samples(
        cls,
        coords_m: np.ndarray,
        weights: np.ndarray | None,
        options: MRIShimMetricOptions,
    ) -> "MRIShimMetricModel":
        _require_scipy()
        if not options.report_enabled:
            raise ValueError("An MRI shim metric model requires report or optimize mode.")
        coords = np.asarray(coords_m, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 4:
            raise ValueError("MRI metric coordinates must have finite shape (samples, 3), with at least 4 samples.")
        if not np.all(np.isfinite(coords)):
            raise ValueError("MRI metric coordinates must contain only finite values.")
        sample_weights = (
            np.ones(len(coords), dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        )
        if len(sample_weights) != len(coords):
            raise ValueError("MRI metric weights must match the coordinate count.")
        if np.any(~np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
            raise ValueError("MRI metric weights must be finite and non-negative.")
        weight_sum = float(np.sum(sample_weights))
        if weight_sum <= 0.0:
            raise ValueError("MRI metric requires at least one positive sample weight.")
        normalized_weights = sample_weights / weight_sum

        spatial_scale = float(options.spatial_scale_m)
        if spatial_scale == 0.0:
            centroid = np.sum(normalized_weights[:, None] * coords, axis=0)
            spatial_scale = float(np.sqrt(np.dot(normalized_weights, np.sum(np.square(coords - centroid), axis=1))))
        if not np.isfinite(spatial_scale) or spatial_scale <= 0.0:
            raise ValueError("Could not derive a positive characteristic length from the target geometry.")

        gradient_operators = _local_gradient_matrices(
            coords,
            options.derivative_neighbor_count,
        )
        coords = coords.copy()
        normalized_weights = normalized_weights.copy()
        coords.setflags(write=False)
        normalized_weights.setflags(write=False)
        return cls(
            options=options,
            coords_m=coords,
            normalized_weights=normalized_weights,
            gradient_operators_per_m=gradient_operators,
            spatial_scale_m=spatial_scale,
        )

    def evaluate(self, residual_field_T: np.ndarray) -> dict[str, float | str | int]:
        residual = np.asarray(residual_field_T, dtype=float).reshape(-1)
        if len(residual) != len(self.coords_m) or not np.all(np.isfinite(residual)):
            raise ValueError("MRI residual must be a finite scalar field matching the target samples.")
        frequency_hz = PROTON_GYROMAGNETIC_RATIO_HZ_PER_T * residual
        weights = self.normalized_weights
        frequency_adjustment_hz = float(np.dot(weights, frequency_hz))
        centered_hz = frequency_hz - frequency_adjustment_hz
        abs_hz = np.abs(centered_hz)
        residual_rms_hz = _weighted_rms(centered_hz, weights)
        residual_cvar_hz = _weighted_cvar(abs_hz, weights, float(self.options.cvar_alpha))
        gradient_components = [
            np.asarray(operator @ centered_hz, dtype=float).reshape(-1) for operator in self.gradient_operators_per_m
        ]
        gradient_magnitude_hz_per_m = np.sqrt(sum(np.square(component) for component in gradient_components))
        gradient_rms_hz_per_m = _weighted_rms(gradient_magnitude_hz_per_m, weights)
        scaled_gradient_rms_hz = self.spatial_scale_m * gradient_rms_hz_per_m
        composite_score_hz = float(
            float(self.options.residual_rms_weight) * residual_rms_hz
            + float(self.options.residual_cvar_weight) * residual_cvar_hz
            + float(self.options.spatial_gradient_weight) * scaled_gradient_rms_hz
        )
        reference = float(self.options.reference_frequency_hz)
        return {
            "mri_shim_metric_mode": self.options.mode,
            "mri_shim_metric_definition": "rms_plus_cvar_tail_plus_isotropic_spatial_gradient",
            "mri_shim_derivative_estimator": "local_weighted_least_squares_point_cloud",
            "mri_shim_sample_count": int(len(centered_hz)),
            "mri_shim_frequency_adjustment_Hz": frequency_adjustment_hz,
            "mri_shim_residual_rms_Hz": residual_rms_hz,
            "mri_shim_residual_std_Hz": residual_rms_hz,
            "mri_shim_residual_p95_abs_Hz": _weighted_quantile(abs_hz, weights, 0.95),
            "mri_shim_residual_cvar_abs_Hz": residual_cvar_hz,
            "mri_shim_residual_max_abs_Hz": float(np.max(abs_hz)),
            "mri_shim_residual_fraction_over_reference": float(np.dot(weights, abs_hz > reference)),
            "mri_shim_spatial_scale_m": float(self.spatial_scale_m),
            "mri_shim_gradient_rms_Hz_per_m": gradient_rms_hz_per_m,
            "mri_shim_scaled_gradient_rms_Hz": scaled_gradient_rms_hz,
            "mri_shim_composite_score_Hz": composite_score_hz,
            "mri_shim_composite_score_per_reference": composite_score_hz / reference,
        }


def build_cvxpy_mri_shim_objective(
    cp: Any,
    model: MRIShimMetricModel,
    residual_field_T: Any,
    *,
    name_prefix: str = "mri_shim",
) -> tuple[Any, list[Any], Mapping[str, Any]]:
    """Return the convex MRI score in Hz, its constraints, and components."""

    _require_scipy()
    weights = np.asarray(model.normalized_weights, dtype=float)
    sqrt_weights = np.sqrt(weights)
    frequency_hz = PROTON_GYROMAGNETIC_RATIO_HZ_PER_T * residual_field_T
    centered_hz = frequency_hz - weights @ frequency_hz
    residual_rms_hz = cp.norm(cp.multiply(sqrt_weights, centered_hz), 2)

    cvar_threshold = cp.Variable(nonneg=True, name=f"{name_prefix}_cvar_threshold_Hz")
    cvar_excess = cp.Variable(len(weights), nonneg=True, name=f"{name_prefix}_cvar_excess_Hz")
    constraints = [cvar_excess >= cp.abs(centered_hz) - cvar_threshold]
    residual_cvar_hz = cvar_threshold + (weights @ cvar_excess) / (1.0 - float(model.options.cvar_alpha))

    gradient_rows = sparse.vstack(model.gradient_operators_per_m, format="csr")
    tiled_sqrt_weights = np.tile(sqrt_weights, 3)
    gradient_rms_hz_per_m = cp.norm(
        cp.multiply(tiled_sqrt_weights, gradient_rows @ centered_hz),
        2,
    )
    scaled_gradient_rms_hz = float(model.spatial_scale_m) * gradient_rms_hz_per_m
    score_hz = (
        float(model.options.residual_rms_weight) * residual_rms_hz
        + float(model.options.residual_cvar_weight) * residual_cvar_hz
        + float(model.options.spatial_gradient_weight) * scaled_gradient_rms_hz
    )
    return (
        score_hz,
        constraints,
        {
            "centered_frequency_Hz": centered_hz,
            "residual_rms_Hz": residual_rms_hz,
            "residual_cvar_abs_Hz": residual_cvar_hz,
            "gradient_rms_Hz_per_m": gradient_rms_hz_per_m,
            "scaled_gradient_rms_Hz": scaled_gradient_rms_hz,
        },
    )


def evaluate_mri_shim_residual(
    coords_m: np.ndarray,
    residual_field_T: np.ndarray,
    *,
    weights: np.ndarray | None,
    options: MRIShimMetricOptions,
) -> dict[str, float | str | int]:
    """Evaluate the protocol-independent metric on any shim residual."""

    return MRIShimMetricModel.from_samples(coords_m, weights, options).evaluate(residual_field_T)


def _local_gradient_matrices(
    coords_m: np.ndarray,
    neighbor_count: int,
) -> tuple[Any, Any, Any]:
    """Return x/y/z local-affine derivative matrices on an irregular cloud."""

    _require_scipy()
    coords = np.asarray(coords_m, dtype=float)
    sample_count = len(coords)
    query_count = min(sample_count, int(neighbor_count) + 1)
    distances, indices = cKDTree(coords).query(coords, k=query_count)
    if query_count == 1:
        distances = np.asarray(distances, dtype=float)[:, None]
        indices = np.asarray(indices, dtype=np.int64)[:, None]
    rows_by_axis: list[list[int]] = [[], [], []]
    columns_by_axis: list[list[int]] = [[], [], []]
    values_by_axis: list[list[float]] = [[], [], []]

    for row in range(sample_count):
        neighbors = np.asarray(indices[row], dtype=np.int64).reshape(-1)
        neighbor_distances = np.asarray(distances[row], dtype=float).reshape(-1)
        keep = (neighbors != row) & np.isfinite(neighbor_distances) & (neighbor_distances > 0.0)
        neighbors = neighbors[keep]
        neighbor_distances = neighbor_distances[keep]
        if len(neighbors) < 3:
            raise ValueError("Target samples do not provide enough distinct neighbors for MRI gradients.")
        offsets = coords[neighbors] - coords[row]
        local_scale = float(np.median(neighbor_distances))
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            raise ValueError("Target sample spacing is degenerate in the MRI gradient estimator.")
        scaled_offsets = offsets / local_scale
        kernel = np.exp(-np.square(neighbor_distances / local_scale))
        weighted_offsets = scaled_offsets * np.sqrt(kernel)[:, None]
        inverse = np.linalg.pinv(weighted_offsets.T @ weighted_offsets, rcond=1.0e-10)
        coefficient_matrix = inverse @ scaled_offsets.T @ np.diag(kernel) / local_scale
        for axis in range(3):
            coefficients = np.asarray(coefficient_matrix[axis], dtype=float).reshape(-1)
            rows_by_axis[axis].extend([row] * (len(neighbors) + 1))
            columns_by_axis[axis].extend(int(value) for value in neighbors)
            columns_by_axis[axis].append(row)
            values_by_axis[axis].extend(float(value) for value in coefficients)
            values_by_axis[axis].append(float(-np.sum(coefficients)))

    matrices = []
    for axis in range(3):
        matrix = sparse.csr_matrix(
            (values_by_axis[axis], (rows_by_axis[axis], columns_by_axis[axis])),
            shape=(sample_count, sample_count),
            dtype=float,
        )
        matrix.eliminate_zeros()
        matrices.append(matrix)
    return tuple(matrices)  # type: ignore[return-value]


def _require_scipy() -> None:
    if sparse is None or cKDTree is None:
        raise ModuleNotFoundError(
            "The acquisition-agnostic MRI shim metric requires SciPy. "
            "Install CoilGen with the spatial or full extra: "
            "python -m pip install -e '.[spatial]'"
        ) from _SCIPY_IMPORT_ERROR


def _weighted_rms(values: np.ndarray, normalized_weights: np.ndarray) -> float:
    return float(np.sqrt(np.dot(normalized_weights, np.square(np.asarray(values, dtype=float)))))


def _weighted_quantile(values: np.ndarray, normalized_weights: np.ndarray, quantile: float) -> float:
    data = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(normalized_weights, dtype=float).reshape(-1)
    order = np.argsort(data, kind="mergesort")
    sorted_data = data[order]
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, float(quantile), side="left")), len(data) - 1)
    return float(sorted_data[index])


def _weighted_cvar(
    nonnegative_losses: np.ndarray,
    normalized_weights: np.ndarray,
    alpha: float,
) -> float:
    threshold = _weighted_quantile(nonnegative_losses, normalized_weights, alpha)
    excess = np.maximum(np.asarray(nonnegative_losses, dtype=float) - threshold, 0.0)
    return float(threshold + np.dot(normalized_weights, excess) / (1.0 - alpha))
