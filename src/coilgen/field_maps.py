from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .diagnostic_cache import diagnostic_fingerprint
from .field_units import config_field_display_units, field_unit_label, field_values_from_tesla
from .metrics import dsv_target_mask, suppression_target_mask
from .models import CoilGenResult
from .spherical_harmonics import (
    SphericalHarmonicFit,
    compare_spherical_harmonic_fits,
    fit_spherical_harmonics_many,
)


AXIS_NAMES = ("x", "y", "z")
FIELD_UNIT = "T/A"
GRADIENT_UNIT = "T/(A m)"
RELATIVE_ERROR_UNIT = "%"
REFERENCE_SCALE_FLOOR = 1e-30


@dataclass(frozen=True)
class ScalarMap:
    key: str
    label: str
    values: np.ndarray
    unit: str = ""
    current_convention: str = ""
    normalization: str = ""

    @property
    def label_with_unit(self) -> str:
        return format_label_with_unit(self.label, self.unit)


@dataclass(frozen=True)
class SliceSelection:
    axis: int
    value: float
    mask: np.ndarray
    plot_axes: tuple[int, int]


@dataclass(frozen=True)
class InterpolatedSlice:
    axis: int
    value: float
    mask: np.ndarray
    plot_axes: tuple[int, int]
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    support_mask: np.ndarray


@dataclass(frozen=True)
class _LocalGradientBasis:
    neighbor_indices: np.ndarray
    operators: np.ndarray
    coords: np.ndarray
    identifiable: np.ndarray


@dataclass(frozen=True)
class _InterpolationPlan:
    neighbor_indices: np.ndarray
    weights: np.ndarray
    exact_indices: np.ndarray
    support_mask: np.ndarray


def field_scalar_maps(result: CoilGenResult) -> dict[str, ScalarMap]:
    """Return per-target-point Bz maps for target, layout, stream, and layout error."""

    target = display_field_values_1a(result, target_bz_1a(result))
    layout = display_field_values_1a(result, layout_bz_1a(result))
    stream = display_field_values_1a(result, stream_bz_1a(result))
    unit = display_field_unit_1a(result)
    labels = _field_role_labels(result)
    return {
        "target": ScalarMap("target", labels["target"], target, unit),
        "layout": ScalarMap("layout", labels["layout"], layout, unit),
        "stream": ScalarMap("stream", labels["stream"], stream, unit),
        "error": ScalarMap("error", labels["layout_target_error"], layout - target, unit),
    }


def shim_field_scalar_maps(result: CoilGenResult) -> dict[str, ScalarMap]:
    """Return final-current Bz maps useful for shim-coil inspection.

    Unlike :func:`field_scalar_maps`, these values are not normalized per
    ampere. For shim workflows the target field is usually the cancellation
    field, so the unshimmed inhomogeneity is ``-target`` and the remaining
    inhomogeneity after shimming is ``coil - target``.
    """

    target = display_field_values(result, np.asarray(result.target_field.b, dtype=float)[:, 2])
    layout = display_field_values(result, np.asarray(result.field_by_layout, dtype=float)[:, 2])
    stream = display_field_values(result, np.asarray(result.b_field_opt_sf, dtype=float)[:, 2])
    unit = display_field_unit(result)
    labels = _field_role_labels(result)
    return {
        "target": ScalarMap("target", labels["target_absolute"], target, unit),
        "layout": ScalarMap("layout", labels["layout"], layout, unit),
        "stream": ScalarMap("stream", labels["stream"], stream, unit),
        "before_inhomogeneity": ScalarMap(
            "before_inhomogeneity",
            labels["before_inhomogeneity"],
            -target,
            unit,
        ),
        "layout_remaining_inhomogeneity": ScalarMap(
            "layout_remaining_inhomogeneity",
            labels["layout_remaining_inhomogeneity"],
            layout - target,
            unit,
        ),
        "stream_remaining_inhomogeneity": ScalarMap(
            "stream_remaining_inhomogeneity",
            labels["stream_remaining_inhomogeneity"],
            stream - target,
            unit,
        ),
        "layout_residual": ScalarMap(
            "layout_residual",
            labels["target_layout_residual"],
            target - layout,
            unit,
        ),
        "stream_residual": ScalarMap(
            "stream_residual",
            labels["target_stream_residual"],
            target - stream,
            unit,
        ),
        "layout_improvement_before_minus_after": ScalarMap(
            "layout_improvement_before_minus_after",
            labels["layout_improvement"],
            -target - (layout - target),
            unit,
        ),
        "stream_improvement_before_minus_after": ScalarMap(
            "stream_improvement_before_minus_after",
            labels["stream_improvement"],
            -target - (stream - target),
            unit,
        ),
    }


def shim_homogeneity_stats(
    result: CoilGenResult,
    *,
    source: str = "after",
    domain: str = "dsv",
) -> dict[str, float | str]:
    """Return signed shim-field map statistics in display units."""

    source_key = _shim_map_key(source)
    residual = np.asarray(shim_field_scalar_maps(result)[source_key].values, dtype=float)
    mask = target_domain_mask(result, domain)
    finite = mask & np.isfinite(residual)
    unit = display_field_unit(result)
    if not np.any(finite):
        return {
            "unit": unit,
            "sample_count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
            "peak_to_peak": float("nan"),
        }
    values = residual[finite]
    return {
        "unit": unit,
        "sample_count": float(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95_abs": float(np.percentile(np.abs(values), 95.0)),
        "max_abs": float(np.max(np.abs(values))),
        "peak_to_peak": float(np.max(values) - np.min(values)),
    }


def shim_spherical_harmonic_fits(
    result: CoilGenResult,
    *,
    source: str = "layout",
    domain: str = "dsv",
    max_order: int = 4,
    center: np.ndarray | None = None,
    radius: float | None = None,
    use_target_weights: bool = True,
) -> dict[str, SphericalHarmonicFit]:
    """Fit before/after shim inhomogeneity maps with solid SH basis terms."""

    coords = np.asarray(result.target_field.coords, dtype=float)
    maps = shim_field_scalar_maps(result)
    before_values = np.asarray(maps["before_inhomogeneity"].values, dtype=float).reshape(-1)
    after_key = _shim_map_key(_shim_after_source(source))
    after_values = np.asarray(maps[after_key].values, dtype=float).reshape(-1)
    if len(before_values) != len(coords) or len(after_values) != len(coords):
        raise ValueError("shim field maps must match target coordinate count.")
    finite = (
        target_domain_mask(result, domain)
        & np.all(np.isfinite(coords), axis=1)
        & np.isfinite(before_values)
        & np.isfinite(after_values)
    )
    if not np.any(finite):
        raise ValueError("shim_spherical_harmonic_fits has no finite target samples.")
    weights = None
    if use_target_weights and getattr(result.target_field, "weights", None) is not None:
        candidate = np.asarray(result.target_field.weights, dtype=float).reshape(-1)
        if len(candidate) == len(coords):
            weights = candidate[finite]
    return fit_spherical_harmonics_many(
        coords[finite],
        {"before": before_values[finite], "after": after_values[finite]},
        max_order=max_order,
        weights=weights,
        center=center,
        radius=radius,
        unit=maps["before_inhomogeneity"].unit,
    )


def shim_spherical_harmonic_comparison(
    result: CoilGenResult,
    *,
    source: str = "layout",
    domain: str = "dsv",
    max_order: int = 4,
    min_contribution_fraction: float = 0.01,
    max_terms: int | None = 18,
    center: np.ndarray | None = None,
    radius: float | None = None,
    use_target_weights: bool = True,
    fits: Mapping[str, SphericalHarmonicFit] | None = None,
) -> list[dict]:
    """Return significant before/after SH terms for a shim result."""

    fits = (
        fits
        if fits is not None
        else shim_spherical_harmonic_fits(
            result,
            source=source,
            domain=domain,
            max_order=max_order,
            center=center,
            radius=radius,
            use_target_weights=use_target_weights,
        )
    )
    return compare_spherical_harmonic_fits(
        fits["before"],
        fits["after"],
        min_contribution_fraction=min_contribution_fraction,
        max_terms=max_terms,
    )


def coil_field_spherical_harmonic_maps(result: CoilGenResult) -> dict[str, ScalarMap]:
    """Return final-current scalar Bz maps used for general coil SH decomposition."""

    target = display_field_values(result, np.asarray(result.target_field.b, dtype=float)[:, 2])
    layout = display_field_values(result, np.asarray(result.field_by_layout, dtype=float)[:, 2])
    stream = display_field_values(result, np.asarray(result.b_field_opt_sf, dtype=float)[:, 2])
    unit = display_field_unit(result)
    labels = _field_role_labels(result)
    return {
        "target": ScalarMap("target", labels["target_absolute"], target, unit),
        "layout": ScalarMap("layout", labels["layout"], layout, unit),
        "stream": ScalarMap("stream", labels["stream"], stream, unit),
        "layout_error": ScalarMap("layout_error", labels["layout_target_error"], layout - target, unit),
        "stream_error": ScalarMap("stream_error", labels["stream_target_error"], stream - target, unit),
    }


def coil_spherical_harmonic_fits(
    result: CoilGenResult,
    *,
    domain: str = "dsv",
    max_order: int = 4,
    sources: Sequence[str] = ("target", "layout", "stream", "layout_error"),
    center: np.ndarray | None = None,
    radius: float | None = None,
    use_target_weights: bool = True,
) -> dict[str, SphericalHarmonicFit]:
    """Fit target, layout, stream, and residual coil fields with solid SH terms.

    The fitted scalar is the final-current Bz field over the selected target
    domain. Coordinates are normalized by the fitted DSV radius, so coefficients
    have the same units as the scalar field and can be compared across channels
    that share the same target volume.
    """

    coords = np.asarray(result.target_field.coords, dtype=float)
    maps = coil_field_spherical_harmonic_maps(result)
    source_keys = tuple(_normalise_coil_sh_source(source) for source in sources)
    unknown = [source for source in source_keys if source not in maps]
    if unknown:
        raise ValueError(f"Unknown coil spherical-harmonic source(s): {', '.join(unknown)}.")
    values_by_source = {source: np.asarray(maps[source].values, dtype=float).reshape(-1) for source in source_keys}
    finite = target_domain_mask(result, domain) & np.all(np.isfinite(coords), axis=1)
    for values in values_by_source.values():
        if len(values) != len(coords):
            raise ValueError("coil SH field maps must match target coordinate count.")
        finite &= np.isfinite(values)
    if not np.any(finite):
        raise ValueError("coil_spherical_harmonic_fits has no finite target samples.")
    weights = None
    if use_target_weights and getattr(result.target_field, "weights", None) is not None:
        candidate = np.asarray(result.target_field.weights, dtype=float).reshape(-1)
        if len(candidate) == len(coords):
            weights = candidate[finite]

    if not source_keys:
        return {}
    return fit_spherical_harmonics_many(
        coords[finite],
        {source: values_by_source[source][finite] for source in source_keys},
        max_order=max_order,
        weights=weights,
        center=center,
        radius=radius,
        unit=maps[source_keys[0]].unit,
    )


def coil_spherical_harmonic_coefficient_rows(
    result: CoilGenResult,
    *,
    domain: str = "dsv",
    max_order: int = 4,
    sources: Sequence[str] = ("target", "layout", "stream", "layout_error"),
    min_contribution_fraction: float = 0.01,
    max_terms: int | None = 18,
    center: np.ndarray | None = None,
    radius: float | None = None,
    use_target_weights: bool = True,
    fits: Mapping[str, SphericalHarmonicFit] | None = None,
) -> list[dict]:
    """Return SH coefficient rows for general coil field diagnostics."""

    fits = (
        fits
        if fits is not None
        else coil_spherical_harmonic_fits(
            result,
            domain=domain,
            max_order=max_order,
            sources=sources,
            center=center,
            radius=radius,
            use_target_weights=use_target_weights,
        )
    )
    source_keys = tuple(fits)
    reference = fits[source_keys[0]]
    rows: list[dict[str, Any]] = []
    for index, term in enumerate(reference.terms):
        row: dict[str, Any] = {
            "term": term,
            "term_label": term.display_label,
            "degree": int(term.degree),
            "order": int(term.order),
            "index": int(index),
            "unit": reference.unit,
            "sources": source_keys,
        }
        max_contribution = 0.0
        max_fraction = 0.0
        for source, fit in fits.items():
            coefficient = float(np.asarray(fit.coefficients, dtype=float).reshape(-1)[index])
            basis_rms = float(_fit_basis_rms(fit)[index])
            contribution = abs(coefficient) * basis_rms
            fraction = contribution / max(float(fit.field_rms), REFERENCE_SCALE_FLOOR)
            row[f"{source}_coefficient"] = coefficient
            row[f"{source}_contribution_rms"] = float(contribution)
            row[f"{source}_contribution_fraction"] = float(fraction)
            max_contribution = max(max_contribution, float(contribution))
            max_fraction = max(max_fraction, float(fraction))
        row["max_contribution_rms"] = float(max_contribution)
        row["max_contribution_fraction"] = float(max_fraction)
        rows.append(row)
    rows.sort(key=lambda row: float(row["max_contribution_rms"]), reverse=True)
    threshold = max(float(min_contribution_fraction), 0.0)
    significant = [row for row in rows if float(row["max_contribution_fraction"]) >= threshold]
    if not significant:
        significant = rows
    if max_terms is not None:
        significant = significant[: max(1, int(max_terms))]
    return significant


def target_domain_mask(result: CoilGenResult, domain: str = "dsv") -> np.ndarray:
    """Return a target-point mask for DSV, suppression, or all samples."""

    key = str(domain).strip().lower().replace("-", "_")
    if key in {"dsv", "target", "primary", "internal"}:
        return dsv_target_mask(result.target_field)
    if key in {"suppression", "secondary", "fringe", "shield"}:
        return suppression_target_mask(result.target_field)
    if key in {"all", "any", "full"}:
        return np.ones(len(result.target_field.coords), dtype=bool)
    raise ValueError("domain must be 'dsv', 'suppression', or 'all'.")


def slice_mosaic_selections(
    coords: np.ndarray,
    *,
    axis: int = 2,
    slice_count: int = 12,
    sample_mask: np.ndarray | None = None,
    min_points: int = 8,
) -> tuple[SliceSelection, ...]:
    """Choose multiple representative slices through a target point cloud."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        return tuple()
    axis = int(axis)
    sample_mask = (
        np.ones(len(coords), dtype=bool) if sample_mask is None else np.asarray(sample_mask, dtype=bool).reshape(-1)
    )
    if len(sample_mask) != len(coords):
        raise ValueError("sample_mask must match coordinate count.")
    finite = sample_mask & np.all(np.isfinite(coords), axis=1)
    if not np.any(finite):
        return tuple()
    axis_values = coords[:, axis]
    domain_values = axis_values[finite]
    count = max(1, min(int(slice_count), len(domain_values)))
    rounded = np.round(axis_values, decimals=12)
    unique_values = np.unique(np.round(domain_values, decimals=12))
    if len(unique_values) == 0:
        return tuple()
    if len(unique_values) <= count:
        selected_values = unique_values
        allow_slab_fallback = False
    else:
        target_positions = np.linspace(float(np.min(domain_values)), float(np.max(domain_values)), count)
        selected_values = np.asarray(
            [unique_values[int(np.argmin(np.abs(unique_values - position)))] for position in target_positions],
            dtype=float,
        )
        selected_values = np.unique(selected_values)
        allow_slab_fallback = len(unique_values) > max(4 * count, count + 2)

    selections: list[SliceSelection] = []
    seen: set[tuple[float, tuple[int, ...]]] = set()
    plot_axes = _slice_plot_axes(axis)
    for value in selected_values:
        mask = finite & np.isclose(rounded, value, rtol=0.0, atol=1e-12)
        if allow_slab_fallback and np.count_nonzero(mask) < min(int(min_points), int(np.count_nonzero(finite))):
            nearest_count = min(max(int(min_points), 1), int(np.count_nonzero(finite)))
            finite_indices = np.flatnonzero(finite)
            order = finite_indices[np.argsort(np.abs(axis_values[finite] - float(value)))[:nearest_count]]
            mask = np.zeros(len(coords), dtype=bool)
            mask[order] = True
        indices = tuple(np.flatnonzero(mask).tolist())
        if not indices:
            continue
        actual_value = float(np.median(axis_values[mask]))
        dedupe_key = (round(actual_value, 12), indices)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        selections.append(SliceSelection(axis=axis, value=actual_value, mask=mask, plot_axes=plot_axes))
    return tuple(selections)


def interpolated_slice_mosaic(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    axis: int = 2,
    slice_count: int = 12,
    sample_mask: np.ndarray | None = None,
    grid_size: int | tuple[int, int] = 96,
    min_points: int = 24,
    slab_thickness: float | None = None,
    idw_neighbors: int = 12,
    idw_power: float = 2.0,
    plan_cache: dict[str, Any] | None = None,
) -> tuple[InterpolatedSlice, ...]:
    """Interpolate point-cloud values into filled 2-D slabs for a slice mosaic."""

    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")
    if len(values) != len(coords):
        raise ValueError("values must match coordinate count.")
    axis = int(axis)
    sample_mask = (
        np.ones(len(coords), dtype=bool) if sample_mask is None else np.asarray(sample_mask, dtype=bool).reshape(-1)
    )
    if len(sample_mask) != len(coords):
        raise ValueError("sample_mask must match coordinate count.")
    finite = sample_mask & np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    if not np.any(finite):
        return tuple()

    grid_x_count, grid_y_count = _normalise_grid_size(grid_size)
    plot_axes = _slice_plot_axes(axis)
    x_range = _expanded_range(coords[finite, plot_axes[0]])
    y_range = _expanded_range(coords[finite, plot_axes[1]])
    grid_x = np.linspace(x_range[0], x_range[1], grid_x_count)
    grid_y = np.linspace(y_range[0], y_range[1], grid_y_count)

    axis_values = coords[:, axis]
    finite_axis_values = axis_values[finite]
    z_min = float(np.min(finite_axis_values))
    z_max = float(np.max(finite_axis_values))
    count = max(1, int(slice_count))
    if z_max <= z_min:
        slice_values = np.asarray([float(np.median(finite_axis_values))], dtype=float)
    else:
        count = min(count, max(len(np.unique(np.round(finite_axis_values, decimals=12))), 1))
        slice_values = np.linspace(z_min, z_max, count)
    half_thickness = _slice_half_thickness(finite_axis_values, len(slice_values), slab_thickness)

    slices: list[InterpolatedSlice] = []
    finite_indices = np.flatnonzero(finite)
    minimum = min(max(int(min_points), 3), len(finite_indices))
    for slice_value in slice_values:
        mask = finite & (np.abs(axis_values - float(slice_value)) <= half_thickness)
        if np.count_nonzero(mask) < minimum:
            nearest = finite_indices[np.argsort(np.abs(axis_values[finite] - float(slice_value)))[:minimum]]
            mask = np.zeros(len(coords), dtype=bool)
            mask[nearest] = True
        if np.count_nonzero(mask) < 3:
            continue
        points_2d = coords[mask][:, plot_axes]
        point_values = values[mask]
        grid_values, support_mask = interpolate_points_to_grid(
            points_2d,
            point_values,
            grid_x,
            grid_y,
            neighbors=idw_neighbors,
            power=idw_power,
            plan_cache=plan_cache,
        )
        actual_value = float(np.median(axis_values[mask]))
        slices.append(
            InterpolatedSlice(
                axis=axis,
                value=actual_value,
                mask=mask,
                plot_axes=plot_axes,
                x=grid_x,
                y=grid_y,
                values=grid_values,
                support_mask=support_mask,
            )
        )
    return tuple(slices)


def interpolate_points_to_grid(
    points_2d: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    neighbors: int = 12,
    power: float = 2.0,
    plan_cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate scattered 2-D values onto a regular grid using IDW."""

    points_2d = np.asarray(points_2d, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    grid_x = np.asarray(grid_x, dtype=float).reshape(-1)
    grid_y = np.asarray(grid_y, dtype=float).reshape(-1)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must have shape (n, 2).")
    finite = np.all(np.isfinite(points_2d), axis=1) & np.isfinite(values)
    points_2d = points_2d[finite]
    values = values[finite]
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    grid_points = np.column_stack((grid_xx.ravel(), grid_yy.ravel()))
    shape = (len(grid_y), len(grid_x))
    if len(points_2d) == 0 or len(grid_points) == 0:
        return np.full(shape, np.nan, dtype=float), np.zeros(shape, dtype=bool)

    if plan_cache is None:
        support_mask = _grid_support_mask(points_2d, grid_points).reshape(shape)
        interpolated = _idw_interpolate(points_2d, values, grid_points, neighbors=neighbors, power=power).reshape(shape)
    else:
        # Include the finite selection as well as projected coordinates: two
        # slabs can project to identical points but refer to different samples.
        key = diagnostic_fingerprint(points_2d, finite, grid_x, grid_y, int(neighbors), float(power))
        plan = plan_cache.get(key)
        if not isinstance(plan, _InterpolationPlan):
            plan = _build_interpolation_plan(points_2d, grid_points, neighbors=neighbors, power=power)
            plan_cache[key] = plan
        support_mask = plan.support_mask.reshape(shape)
        interpolated = _apply_interpolation_plan(values, plan).reshape(shape)
    interpolated[~support_mask] = np.nan
    return interpolated, support_mask


def field_error_scalar_maps(result: CoilGenResult) -> dict[str, ScalarMap]:
    """Return absolute Bz error maps over internal DSV target points."""

    if _error_maps_use_fitted_current(result):
        target = display_field_values(result, _target_bz(result))
        layout = display_field_values(result, _layout_bz(result))
        stream = display_field_values(result, _stream_bz(result))
        unit = display_field_unit(result)
    else:
        target = display_field_values_1a(result, target_bz_1a(result))
        layout = display_field_values_1a(result, layout_bz_1a(result))
        stream = display_field_values_1a(result, stream_bz_1a(result))
        unit = display_field_unit_1a(result)
    dsv_mask = dsv_target_mask(result.target_field)
    labels = _field_role_labels(result)
    convention = _field_error_current_convention(result)
    return {
        "layout_vs_target": ScalarMap(
            "layout_vs_target",
            f"DSV {labels['layout_target_error']} Error",
            _values_on_domain(layout - target, dsv_mask),
            unit,
            current_convention=convention,
        ),
        "stream_vs_target": ScalarMap(
            "stream_vs_target",
            f"DSV {labels['stream_target_error']} Error",
            _values_on_domain(stream - target, dsv_mask),
            unit,
            current_convention=convention,
        ),
        "layout_vs_stream": ScalarMap(
            "layout_vs_stream",
            f"DSV {labels['layout']} - {labels['stream']} Error",
            _values_on_domain(layout - stream, dsv_mask),
            unit,
            current_convention=convention,
        ),
    }


def field_relative_error_scalar_maps(result: CoilGenResult) -> dict[str, ScalarMap]:
    """Return DSV Bz error maps normalized by the internal target-field scale."""

    if _error_maps_use_fitted_current(result):
        target = _target_bz(result)
        layout = _layout_bz(result)
        stream = _stream_bz(result)
    else:
        target = target_bz_1a(result)
        layout = layout_bz_1a(result)
        stream = stream_bz_1a(result)
    dsv_mask = dsv_target_mask(result.target_field)
    labels = _field_role_labels(result)
    convention = _field_error_current_convention(result)
    target_scale = _reference_scale(target[dsv_mask])
    stream_scale = _reference_scale(stream[dsv_mask])
    return {
        "layout_vs_target": ScalarMap(
            "layout_vs_target",
            f"DSV {labels['layout_target_error']} Relative Error",
            _values_on_domain(_relative_percent(layout - target, target_scale), dsv_mask),
            RELATIVE_ERROR_UNIT,
            current_convention=convention,
            normalization="Target comparisons: 100 * signed difference / peak |target Bz| on DSV",
        ),
        "stream_vs_target": ScalarMap(
            "stream_vs_target",
            f"DSV {labels['stream_target_error']} Relative Error",
            _values_on_domain(_relative_percent(stream - target, target_scale), dsv_mask),
            RELATIVE_ERROR_UNIT,
            current_convention=convention,
            normalization="Target comparisons: 100 * signed difference / peak |target Bz| on DSV",
        ),
        "layout_vs_stream": ScalarMap(
            "layout_vs_stream",
            f"DSV {labels['layout']} - {labels['stream']} Relative Error",
            _values_on_domain(_relative_percent(layout - stream, stream_scale), dsv_mask),
            RELATIVE_ERROR_UNIT,
            current_convention=convention,
            normalization="Stream comparison: 100 * signed difference / peak |stream Bz| on DSV",
        ),
    }


def gradient_vector_maps(result: CoilGenResult, *, neighbor_count: int = 12) -> dict[str, ScalarMap]:
    """Return local DSV gradients of Bz for target, layout, stream, and layout error."""

    coords = np.asarray(result.target_field.coords, dtype=float)
    dsv_mask = dsv_target_mask(result.target_field)
    dsv_coords = coords[dsv_mask]
    neighbor_count = _normalized_neighbor_count(dsv_coords, neighbor_count)
    cache_key = f"gradient_vector_maps:dsv:{neighbor_count}"
    signature = diagnostic_fingerprint(
        coords,
        dsv_mask,
        target_bz_1a(result),
        layout_bz_1a(result),
        stream_bz_1a(result),
        config_field_display_units(result.input_data),
        _field_role_labels(result),
    )
    cached = result.diagnostic_cache.get(cache_key)
    if isinstance(cached, tuple) and cached[0] == signature:
        return cached[1]

    fields = field_scalar_maps(result)
    basis = _cached_local_gradient_basis(
        result,
        dsv_coords,
        neighbor_count,
        cache_key=f"local_gradient_basis:dsv:{neighbor_count}",
    )
    maps: dict[str, ScalarMap] = {}
    for key, field_map in fields.items():
        gradients = _apply_local_gradient_basis(np.asarray(field_map.values, dtype=float)[dsv_mask], basis)
        full_gradients = np.full((len(coords), 3), np.nan, dtype=float)
        full_gradients[dsv_mask] = gradients
        maps[key] = ScalarMap(
            key,
            f"DSV {field_map.label}",
            full_gradients,
            display_gradient_unit_1a(result),
            current_convention="Per-ampere local least-squares gradients; unidentifiable components are masked",
        )
    result.diagnostic_cache[cache_key] = (signature, maps)
    return maps


def gradient_component_scalar_maps(
    result: CoilGenResult,
    *,
    component: str = "magnitude",
    neighbor_count: int = 12,
) -> dict[str, ScalarMap]:
    """Return one scalar gradient component for each field source."""

    component_key = str(component).lower()
    gradients = gradient_vector_maps(result, neighbor_count=neighbor_count)
    output: dict[str, ScalarMap] = {}
    for key, gradient_map in gradients.items():
        vectors = np.asarray(gradient_map.values, dtype=float)
        values, label_suffix = _gradient_component(vectors, component_key)
        output[key] = ScalarMap(
            key,
            f"{gradient_map.label} {label_suffix}",
            values,
            gradient_map.unit,
            current_convention=gradient_map.current_convention,
        )
    return output


def gradient_relative_error_scalar_maps(
    result: CoilGenResult,
    *,
    component: str = "magnitude",
    neighbor_count: int = 12,
) -> dict[str, ScalarMap]:
    """Return local Bz-gradient error maps normalized by the global reference-gradient scale."""

    component_key = str(component).lower()
    gradients = gradient_vector_maps(result, neighbor_count=neighbor_count)
    target_vectors = np.asarray(gradients["target"].values, dtype=float)
    layout_vectors = np.asarray(gradients["layout"].values, dtype=float)
    stream_vectors = np.asarray(gradients["stream"].values, dtype=float)
    target_values, label_suffix = _gradient_component(target_vectors, component_key)
    layout_values, _ = _gradient_component(layout_vectors, component_key)
    stream_values, _ = _gradient_component(stream_vectors, component_key)
    target_scale = _gradient_reference_scale(target_vectors)
    stream_scale = _gradient_reference_scale(stream_vectors)
    labels = _field_role_labels(result)
    return {
        "layout_vs_target": ScalarMap(
            "layout_vs_target",
            f"DSV {labels['layout_target_error']} Gradient {label_suffix} Relative Error",
            _relative_percent(layout_values - target_values, target_scale),
            RELATIVE_ERROR_UNIT,
            current_convention=gradients["target"].current_convention,
            normalization="Target comparisons: 100 * signed difference / peak |grad target Bz| on DSV",
        ),
        "stream_vs_target": ScalarMap(
            "stream_vs_target",
            f"DSV {labels['stream_target_error']} Gradient {label_suffix} Relative Error",
            _relative_percent(stream_values - target_values, target_scale),
            RELATIVE_ERROR_UNIT,
            current_convention=gradients["target"].current_convention,
            normalization="Target comparisons: 100 * signed difference / peak |grad target Bz| on DSV",
        ),
        "layout_vs_stream": ScalarMap(
            "layout_vs_stream",
            f"DSV {labels['layout']} - {labels['stream']} Gradient {label_suffix} Relative Error",
            _relative_percent(layout_values - stream_values, stream_scale),
            RELATIVE_ERROR_UNIT,
            current_convention=gradients["target"].current_convention,
            normalization="Stream comparison: 100 * signed difference / peak |grad stream Bz| on DSV",
        ),
    }


def local_scalar_gradients(coords: np.ndarray, values: np.ndarray, *, neighbor_count: int = 12) -> np.ndarray:
    """Estimate local scalar gradients by nearest-neighbor least squares."""

    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")
    if len(coords) != len(values):
        raise ValueError("values must match coordinate count.")
    basis = _build_local_gradient_basis(coords, _normalized_neighbor_count(coords, neighbor_count))
    return _apply_local_gradient_basis(values, basis)


def central_slice_selection(
    coords: np.ndarray,
    axis: int,
    *,
    min_points: int = 8,
) -> SliceSelection:
    """Select a central cross-section through a target-point cloud."""

    coords = np.asarray(coords, dtype=float)
    axis = int(axis)
    axis_values = coords[:, axis]
    if len(axis_values) == 0:
        return SliceSelection(axis=axis, value=0.0, mask=np.zeros(0, dtype=bool), plot_axes=_slice_plot_axes(axis))

    unique_values = np.unique(np.round(axis_values, decimals=12))
    center = float(np.median(axis_values))
    slice_value = float(unique_values[int(np.argmin(np.abs(unique_values - center)))])
    mask = np.isclose(axis_values, slice_value, rtol=0.0, atol=1e-10)
    if np.count_nonzero(mask) >= min(min_points, len(axis_values)):
        return SliceSelection(axis=axis, value=slice_value, mask=mask, plot_axes=_slice_plot_axes(axis))

    count = min(max(int(min_points), 1), len(axis_values))
    nearest = np.argsort(np.abs(axis_values - slice_value))[:count]
    mask = np.zeros(len(axis_values), dtype=bool)
    mask[nearest] = True
    return SliceSelection(axis=axis, value=slice_value, mask=mask, plot_axes=_slice_plot_axes(axis))


def target_bz_1a(result: CoilGenResult) -> np.ndarray:
    target = result.target_field_1A if result.target_field_1A is not None else result.target_field
    return np.asarray(target.b[:, 2], dtype=float)


def layout_bz_1a(result: CoilGenResult) -> np.ndarray:
    field = result.field_layout_per1Amp if result.field_layout_per1Amp is not None else result.field_by_layout
    return np.asarray(field[:, 2], dtype=float)


def suppression_field_scalar_maps(result: CoilGenResult) -> dict[str, ScalarMap]:
    """Return Bz and |B| maps only on secondary suppression/fringe-field points."""

    mask = suppression_target_mask(result.target_field)
    layout = _field_1a(
        result.field_layout_per1Amp if result.field_layout_per1Amp is not None else result.field_by_layout
    )
    stream = _field_1a(result.b_field_opt_sf_1A if result.b_field_opt_sf_1A is not None else result.b_field_opt_sf)
    layout_bz = display_field_values_1a(result, layout[:, 2])
    layout_bmag = display_field_values_1a(result, np.linalg.norm(layout, axis=1))
    stream_bz = display_field_values_1a(result, stream[:, 2])
    stream_bmag = display_field_values_1a(result, np.linalg.norm(stream, axis=1))
    unit = display_field_unit_1a(result)
    labels = _field_role_labels(result)
    return {
        "layout_bz": ScalarMap(
            "layout_bz",
            f"Suppression {labels['layout']}",
            _values_on_domain(layout_bz, mask),
            unit,
        ),
        "layout_bmag": ScalarMap(
            "layout_bmag",
            f"Suppression {labels['layout_magnitude']}",
            _values_on_domain(layout_bmag, mask),
            unit,
        ),
        "stream_bz": ScalarMap(
            "stream_bz",
            f"Suppression {labels['stream']}",
            _values_on_domain(stream_bz, mask),
            unit,
        ),
        "stream_bmag": ScalarMap(
            "stream_bmag", "Suppression Stream Function |B|", _values_on_domain(stream_bmag, mask), unit
        ),
    }


def stream_bz_1a(result: CoilGenResult) -> np.ndarray:
    field = result.b_field_opt_sf_1A if result.b_field_opt_sf_1A is not None else result.b_field_opt_sf
    return np.asarray(field[:, 2], dtype=float)


def _target_bz(result: CoilGenResult) -> np.ndarray:
    return np.asarray(result.target_field.b[:, 2], dtype=float)


def _layout_bz(result: CoilGenResult) -> np.ndarray:
    return np.asarray(result.field_by_layout[:, 2], dtype=float)


def _stream_bz(result: CoilGenResult) -> np.ndarray:
    return np.asarray(result.b_field_opt_sf[:, 2], dtype=float)


def _error_maps_use_fitted_current(result: CoilGenResult) -> bool:
    config = getattr(result, "input_data", None)
    source = str(getattr(config, "field_evaluation_current_source", "target_fit")).strip().lower().replace("-", "_")
    return source in {
        "target_projection",
        "projection",
        "least_squares",
        "least_squares_projection",
        "target_std_projection",
        "std_projection",
        "homogeneity_projection",
        "mean_free_projection",
    }


def _field_error_current_convention(result: CoilGenResult) -> str:
    if _error_maps_use_fitted_current(result):
        return "Final-current comparison: layout at evaluated current; stream at contour-step current"
    return "Per-ampere comparison: target and stream normalized by contour-step current"


def display_field_values_1a(result: CoilGenResult, values_t_per_amp: np.ndarray) -> np.ndarray:
    return field_values_from_tesla(values_t_per_amp, config_field_display_units(result.input_data))


def display_field_values(result: CoilGenResult, values_t: np.ndarray) -> np.ndarray:
    return field_values_from_tesla(values_t, config_field_display_units(result.input_data))


def display_field_unit(result: CoilGenResult) -> str:
    return field_unit_label(config_field_display_units(result.input_data))


def display_field_unit_1a(result: CoilGenResult) -> str:
    return field_unit_label(config_field_display_units(result.input_data), per_amp=True)


def display_gradient_unit_1a(result: CoilGenResult) -> str:
    return field_unit_label(config_field_display_units(result.input_data), per_amp=True, gradient=True)


def _field_role_labels(result: CoilGenResult) -> dict[str, str]:
    """Describe whether ``result.target_field`` is a system or actuator subtarget."""

    sparsified = result_uses_sparsified_contour_layout(result)
    if sparsified:
        return {
            "target": "Target Bz (requested)",
            "target_absolute": "Requested Target Bz",
            "layout": "Final Sparsified Weighted-Contour Bz",
            "layout_magnitude": "Final Sparsified Weighted-Contour |B|",
            "stream": "Stream-Function Bz (continuous reference)",
            "layout_target_error": "Final Sparsified Weighted Contours - Target Bz",
            "stream_target_error": "Stream Function - Target Bz",
            "before_inhomogeneity": "Before Shimming B0 Inhomogeneity",
            "layout_remaining_inhomogeneity": "After Shimming with Final Sparsified Contours",
            "stream_remaining_inhomogeneity": "After Shimming Remaining B0 Inhomogeneity (Stream Function)",
            "target_layout_residual": "Target - Final Sparsified Weighted-Contour Bz",
            "target_stream_residual": "Target - Stream-Function Bz",
            "layout_improvement": "Sparsified-Contour Shim Improvement Before - After",
            "stream_improvement": "Shim Improvement Before - After (Stream Function)",
        }
    return {
        "target": "Target Bz (requested)",
        "target_absolute": "Requested Target Bz",
        "layout": "Connected Wire Bz",
        "layout_magnitude": "Connected Wire |B|",
        "stream": "Stream-Function Bz",
        "layout_target_error": "Connected Wire - Target Bz",
        "stream_target_error": "Stream Function - Target Bz",
        "before_inhomogeneity": "Before Shimming B0 Inhomogeneity",
        "layout_remaining_inhomogeneity": "After Shimming Remaining B0 Inhomogeneity",
        "stream_remaining_inhomogeneity": "After Shimming Remaining B0 Inhomogeneity (Stream Function)",
        "target_layout_residual": "Target - Connected Wire Bz",
        "target_stream_residual": "Target - Stream-Function Bz",
        "layout_improvement": "Shim Improvement Before - After",
        "stream_improvement": "Shim Improvement Before - After (Stream Function)",
    }


def result_uses_sparsified_contour_layout(result: CoilGenResult) -> bool:
    """Return whether weighted sparsified contours are the realized final layout."""

    return any(getattr(part, "contour_sparsification", None) is not None for part in result.coil_parts)


def _normalise_coil_sh_source(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "connected": "layout",
        "connected_wire": "layout",
        "wire": "layout",
        "wire_path": "layout",
        "sf": "stream",
        "stream_function": "stream",
        "error": "layout_error",
        "residual": "layout_error",
        "layout_residual": "layout_error",
        "connected_error": "layout_error",
        "stream_residual": "stream_error",
    }
    return aliases.get(key, key)


def _fit_basis_rms(fit: SphericalHarmonicFit) -> np.ndarray:
    if fit.basis_rms is not None:
        return np.asarray(fit.basis_rms, dtype=float).reshape(-1)
    return np.ones(len(fit.coefficients), dtype=float)


def _gradient_component(vectors: np.ndarray, component: str) -> tuple[np.ndarray, str]:
    if component in {"magnitude", "mag", "norm"}:
        return np.linalg.norm(vectors, axis=1), "|grad Bz|"
    component_by_name = {
        "x": 0,
        "dx": 0,
        "dbz_dx": 0,
        "y": 1,
        "dy": 1,
        "dbz_dy": 1,
        "z": 2,
        "dz": 2,
        "dbz_dz": 2,
    }
    if component not in component_by_name:
        raise ValueError("Gradient component must be magnitude, x, y, z, dbz_dx, dbz_dy, or dbz_dz.")
    axis = component_by_name[component]
    return vectors[:, axis], f"dBz/d{AXIS_NAMES[axis]}"


def _shim_map_key(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    aliases = {
        "before": "before_inhomogeneity",
        "pre": "before_inhomogeneity",
        "pre_shim": "before_inhomogeneity",
        "pre_shimming": "before_inhomogeneity",
        "unshimmed": "before_inhomogeneity",
        "fieldmap": "before_inhomogeneity",
        "inhomogeneity": "before_inhomogeneity",
        "before_inhomogeneity": "before_inhomogeneity",
        "after": "layout_remaining_inhomogeneity",
        "post": "layout_remaining_inhomogeneity",
        "post_shim": "layout_remaining_inhomogeneity",
        "post_shimming": "layout_remaining_inhomogeneity",
        "remaining": "layout_remaining_inhomogeneity",
        "remaining_inhomogeneity": "layout_remaining_inhomogeneity",
        "layout_remaining": "layout_remaining_inhomogeneity",
        "layout_remaining_inhomogeneity": "layout_remaining_inhomogeneity",
        "after_stream": "stream_remaining_inhomogeneity",
        "stream_remaining": "stream_remaining_inhomogeneity",
        "stream_remaining_inhomogeneity": "stream_remaining_inhomogeneity",
        "layout": "layout_remaining_inhomogeneity",
        "connected": "layout_remaining_inhomogeneity",
        "connected_wire": "layout_remaining_inhomogeneity",
        "wire": "layout_remaining_inhomogeneity",
        "wire_layout": "layout_remaining_inhomogeneity",
        "coil": "layout_remaining_inhomogeneity",
        "coilfield": "layout_remaining_inhomogeneity",
        "stream": "stream_remaining_inhomogeneity",
        "stream_function": "stream_remaining_inhomogeneity",
        "sf": "stream_remaining_inhomogeneity",
        "layout_residual": "layout_residual",
        "stream_residual": "stream_residual",
        "target_minus_layout": "layout_residual",
        "target_minus_stream": "stream_residual",
        "improvement": "layout_improvement_before_minus_after",
        "delta": "layout_improvement_before_minus_after",
        "delta_improvement": "layout_improvement_before_minus_after",
        "before_minus_after": "layout_improvement_before_minus_after",
        "layout_improvement": "layout_improvement_before_minus_after",
        "wire_improvement": "layout_improvement_before_minus_after",
        "connected_improvement": "layout_improvement_before_minus_after",
        "layout_improvement_before_minus_after": "layout_improvement_before_minus_after",
        "stream_improvement": "stream_improvement_before_minus_after",
        "before_minus_after_stream": "stream_improvement_before_minus_after",
        "stream_improvement_before_minus_after": "stream_improvement_before_minus_after",
    }
    if key not in aliases:
        raise ValueError("source must be 'before', 'after', 'layout'/'coil', 'stream', or 'improvement'.")
    return aliases[key]


def _shim_after_source(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    stream_aliases = {
        "after_stream",
        "stream",
        "stream_function",
        "sf",
        "stream_remaining",
        "stream_remaining_inhomogeneity",
        "stream_residual",
        "target_minus_stream",
        "stream_improvement",
        "before_minus_after_stream",
        "stream_improvement_before_minus_after",
    }
    return "after_stream" if key in stream_aliases else "after"


def format_label_with_unit(label: str, unit: str | None) -> str:
    unit_text = "" if unit is None else str(unit).strip()
    if not unit_text:
        return str(label)
    return f"{label} [{unit_text}]"


def _reference_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return REFERENCE_SCALE_FLOOR
    return max(float(np.max(np.abs(finite))), REFERENCE_SCALE_FLOOR)


def _gradient_reference_scale(vectors: np.ndarray) -> float:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        return REFERENCE_SCALE_FLOOR
    magnitudes = np.linalg.norm(vectors, axis=1)
    if not np.any(np.isfinite(magnitudes)):
        return float("nan")
    return _reference_scale(magnitudes)


def _relative_percent(delta: np.ndarray, reference_scale: float) -> np.ndarray:
    scale = max(float(reference_scale), REFERENCE_SCALE_FLOOR)
    return np.asarray(delta, dtype=float) / scale * 100.0


def _cached_local_gradient_basis(
    result: CoilGenResult,
    coords: np.ndarray,
    neighbor_count: int,
    *,
    cache_key: str | None = None,
) -> _LocalGradientBasis:
    cache_key = cache_key or f"local_gradient_basis:{neighbor_count}"
    signature = diagnostic_fingerprint(coords, neighbor_count)
    cached = result.diagnostic_cache.get(cache_key)
    if isinstance(cached, tuple) and cached[0] == signature:
        return cached[1]
    basis = _build_local_gradient_basis(coords, neighbor_count)
    result.diagnostic_cache[cache_key] = (signature, basis)
    return basis


def _field_1a(field: np.ndarray | None) -> np.ndarray:
    if field is None:
        return np.zeros((0, 3), dtype=float)
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or field.shape[1] != 3:
        return np.zeros((0, 3), dtype=float)
    return field


def _values_on_domain(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    output = np.full(len(mask), np.nan, dtype=float)
    if len(mask) == len(values):
        output[mask] = values[mask]
    elif len(values) == int(np.count_nonzero(mask)):
        output[mask] = values
    return output


def _normalise_grid_size(grid_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(grid_size, tuple):
        if len(grid_size) != 2:
            raise ValueError("grid_size tuple must contain (x_count, y_count).")
        x_count, y_count = int(grid_size[0]), int(grid_size[1])
    else:
        x_count = y_count = int(grid_size)
    return max(8, x_count), max(8, y_count)


def _expanded_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return -1.0, 1.0
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if maximum <= minimum:
        pad = max(abs(minimum), 1.0) * 0.05
        return minimum - pad, maximum + pad
    pad = 0.02 * (maximum - minimum)
    return minimum - pad, maximum + pad


def _slice_half_thickness(axis_values: np.ndarray, slice_count: int, slab_thickness: float | None) -> float:
    axis_values = np.asarray(axis_values, dtype=float).reshape(-1)
    axis_values = axis_values[np.isfinite(axis_values)]
    if len(axis_values) == 0:
        return 0.0
    if slab_thickness is not None:
        thickness = abs(float(slab_thickness))
        if thickness <= 0.0:
            raise ValueError("slab_thickness must be positive when provided.")
        return thickness / 2.0
    span = float(np.max(axis_values) - np.min(axis_values))
    if span <= 0.0:
        return max(_typical_axis_spacing(axis_values), 1.0e-12)
    if slice_count <= 1:
        return span / 2.0
    return max(span / max(slice_count - 1, 1) * 0.55, _typical_axis_spacing(axis_values) * 0.75, 1.0e-12)


def _typical_axis_spacing(axis_values: np.ndarray) -> float:
    unique = np.unique(np.round(np.asarray(axis_values, dtype=float), decimals=12))
    if len(unique) < 2:
        return 0.0
    diffs = np.diff(unique)
    positive = diffs[diffs > 0.0]
    if len(positive) == 0:
        return 0.0
    return float(np.median(positive))


def _grid_support_mask(points_2d: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    hull = _convex_hull_2d(points_2d)
    if len(hull) >= 3:
        return _points_in_convex_polygon(grid_points, hull)
    nearest_distance = _nearest_distance_2d(points_2d, grid_points)
    limit = max(_typical_point_spacing_2d(points_2d) * 2.5, 1.0e-12)
    return nearest_distance <= limit


def _idw_interpolate(
    points_2d: np.ndarray,
    values: np.ndarray,
    grid_points: np.ndarray,
    *,
    neighbors: int,
    power: float,
) -> np.ndarray:
    point_count = len(points_2d)
    if point_count == 1:
        return np.full(len(grid_points), float(values[0]), dtype=float)
    neighbor_count = max(1, min(int(neighbors), point_count))
    output = np.empty(len(grid_points), dtype=float)
    chunk_size = _interpolation_chunk_size(point_count)
    exponent = max(float(power), 1.0)
    eps = 1.0e-30
    for start in range(0, len(grid_points), chunk_size):
        stop = min(start + chunk_size, len(grid_points))
        block = grid_points[start:stop]
        delta = block[:, None, :] - points_2d[None, :, :]
        distances2 = np.einsum("ijk,ijk->ij", delta, delta)
        if neighbor_count < point_count:
            indices = np.argpartition(distances2, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
            distances2 = np.take_along_axis(distances2, indices, axis=1)
            neighbor_values = values[indices]
        else:
            neighbor_values = np.broadcast_to(values, distances2.shape)
        nearest = np.min(distances2, axis=1)
        exact = nearest <= eps
        block_output = np.empty(stop - start, dtype=float)
        if np.any(exact):
            exact_indices = np.argmin(distances2[exact], axis=1)
            block_output[exact] = neighbor_values[exact, exact_indices]
        if np.any(~exact):
            d2 = distances2[~exact]
            weights = 1.0 / np.power(np.maximum(d2, eps), exponent / 2.0)
            block_output[~exact] = np.sum(weights * neighbor_values[~exact], axis=1) / np.sum(weights, axis=1)
        output[start:stop] = block_output
    return output


def _interpolation_chunk_size(point_count: int) -> int:
    # Per pair: two-coordinate deltas, squared distance, partition index;
    # leave room for selected distances and weights as well.
    return max(1, (16 * 1024 * 1024) // max(48 * int(point_count), 1))


def _build_interpolation_plan(
    points_2d: np.ndarray,
    grid_points: np.ndarray,
    *,
    neighbors: int,
    power: float,
) -> _InterpolationPlan:
    point_count = len(points_2d)
    count = max(1, min(int(neighbors), point_count))
    indices = np.empty((len(grid_points), count), dtype=np.intp)
    weights = np.zeros((len(grid_points), count), dtype=float)
    exact_indices = np.full(len(grid_points), -1, dtype=np.intp)
    exponent = max(float(power), 1.0)
    chunk_size = _interpolation_chunk_size(point_count)
    for start in range(0, len(grid_points), chunk_size):
        stop = min(start + chunk_size, len(grid_points))
        delta = grid_points[start:stop, None, :] - points_2d[None, :, :]
        distances2 = np.einsum("ijk,ijk->ij", delta, delta)
        if count < point_count:
            selected = np.argpartition(distances2, kth=count - 1, axis=1)[:, :count]
            distances2 = np.take_along_axis(distances2, selected, axis=1)
        else:
            selected = np.broadcast_to(np.arange(point_count), distances2.shape)
        indices[start:stop] = selected
        exact = np.min(distances2, axis=1) <= 1.0e-30
        if np.any(exact):
            nearest = np.argmin(distances2[exact], axis=1)
            exact_indices[start:stop][exact] = selected[exact, nearest]
        if np.any(~exact):
            weights[start:stop][~exact] = 1.0 / np.power(np.maximum(distances2[~exact], 1.0e-30), exponent / 2.0)
    return _InterpolationPlan(indices, weights, exact_indices, _grid_support_mask(points_2d, grid_points))


def _apply_interpolation_plan(values: np.ndarray, plan: _InterpolationPlan) -> np.ndarray:
    output = np.empty(len(plan.neighbor_indices), dtype=float)
    exact = plan.exact_indices >= 0
    output[exact] = values[plan.exact_indices[exact]]
    if np.any(~exact):
        weights = plan.weights[~exact]
        output[~exact] = np.sum(weights * values[plan.neighbor_indices[~exact]], axis=1) / np.sum(weights, axis=1)
    return output


def _nearest_distance_2d(points_2d: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    if len(points_2d) == 0:
        return np.full(len(grid_points), np.inf, dtype=float)
    output = np.empty(len(grid_points), dtype=float)
    chunk_size = _interpolation_chunk_size(len(points_2d))
    for start in range(0, len(grid_points), chunk_size):
        stop = min(start + chunk_size, len(grid_points))
        delta = grid_points[start:stop, None, :] - points_2d[None, :, :]
        distances2 = np.einsum("ijk,ijk->ij", delta, delta)
        output[start:stop] = np.sqrt(np.min(distances2, axis=1))
    return output


def _typical_point_spacing_2d(points_2d: np.ndarray) -> float:
    points_2d = np.asarray(points_2d, dtype=float)
    if len(points_2d) < 2:
        return 0.0
    sample = points_2d[: min(len(points_2d), 512)]
    delta = sample[:, None, :] - sample[None, :, :]
    distances2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(distances2, np.inf)
    nearest = np.sqrt(np.min(distances2, axis=1))
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if len(nearest) == 0:
        return 0.0
    return float(np.median(nearest))


def _convex_hull_2d(points_2d: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points_2d, dtype=float), axis=0)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) <= 1:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(hull, dtype=float)


def _points_in_convex_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    next_polygon = np.roll(polygon, -1, axis=0)
    edges = next_polygon - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    tolerance = max(float(np.max(np.ptp(polygon, axis=0))), 1.0) * 1.0e-12
    return np.all(cross >= -tolerance, axis=1) | np.all(cross <= tolerance, axis=1)


def _build_local_gradient_basis(coords: np.ndarray, neighbor_count: int) -> _LocalGradientBasis:
    coords = np.asarray(coords, dtype=float)
    n_points = len(coords)
    finite_indices = np.flatnonzero(np.all(np.isfinite(coords), axis=1))
    count = min(max(int(neighbor_count), 0), max(len(finite_indices) - 1, 0))
    if count == 0:
        return _LocalGradientBasis(
            neighbor_indices=np.zeros((n_points, 0), dtype=np.int64),
            operators=np.zeros((n_points, 3, 0), dtype=float),
            coords=coords.copy(),
            identifiable=np.zeros((n_points, 3), dtype=bool),
        )
    neighbor_indices = np.zeros((n_points, count), dtype=np.int64)
    neighbor_indices[finite_indices] = finite_indices[_nearest_neighbor_indices(coords[finite_indices], count)]
    operators = np.zeros((n_points, 3, count), dtype=float)
    identifiable = np.zeros((n_points, 3), dtype=bool)
    for point_ind in finite_indices:
        neighbors = neighbor_indices[point_ind]
        design = coords[neighbors] - coords[point_ind]
        operators[point_ind] = np.linalg.pinv(design)
        identifiable[point_ind] = _identifiable_gradient_components(operators[point_ind], design)
    return _LocalGradientBasis(neighbor_indices, operators, coords.copy(), identifiable)


def _identifiable_gradient_components(operator: np.ndarray, design: np.ndarray) -> np.ndarray:
    # A component is determined only when its coordinate vector belongs to the
    # stencil's row space. Planar data can determine tangential derivatives,
    # but a minimum-norm zero normal derivative is not a measurement.
    return np.max(np.abs(operator @ design - np.eye(3)), axis=1) <= 1.0e-10


def _apply_local_gradient_basis(values: np.ndarray, basis: _LocalGradientBasis) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(basis.neighbor_indices):
        raise ValueError("values must match gradient-basis point count.")
    if basis.neighbor_indices.shape[1] == 0:
        return np.full((len(values), 3), np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        rhs = values[basis.neighbor_indices] - values[:, None]
    finite = np.isfinite(rhs)
    gradients = np.einsum("nck,nk->nc", basis.operators, np.where(finite, rhs, 0.0))
    gradients[~basis.identifiable] = np.nan
    valid_centers = np.isfinite(values) & np.all(np.isfinite(basis.coords), axis=1)
    gradients[~valid_centers] = np.nan
    for point_ind in np.flatnonzero(valid_centers & ~np.all(finite, axis=1)):
        selected = finite[point_ind]
        gradients[point_ind] = np.nan
        if not np.any(selected):
            continue
        neighbors = basis.neighbor_indices[point_ind, selected]
        design = basis.coords[neighbors] - basis.coords[point_ind]
        operator = np.linalg.pinv(design)
        valid_components = _identifiable_gradient_components(operator, design)
        estimate = operator @ rhs[point_ind, selected]
        gradients[point_ind, valid_components] = estimate[valid_components]
    return gradients


def _nearest_neighbor_indices(coords: np.ndarray, neighbor_count: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n_points = len(coords)
    neighbor_count = _normalized_neighbor_count(coords, neighbor_count)
    if n_points < 2 or neighbor_count <= 0:
        return np.zeros((n_points, 0), dtype=np.int64)
    norms = np.einsum("ij,ij->i", coords, coords)
    max_block_bytes = 64 * 1024 * 1024
    block_size = max(1, int(max_block_bytes // max(8 * n_points, 1)))
    output = np.empty((n_points, neighbor_count), dtype=np.int64)
    for start in range(0, n_points, block_size):
        stop = min(start + block_size, n_points)
        block = coords[start:stop]
        distances = norms[start:stop, None] + norms[None, :] - 2.0 * (block @ coords.T)
        np.maximum(distances, 0.0, out=distances)
        rows = np.arange(stop - start)
        distances[rows, start + rows] = np.inf
        candidates = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
        candidate_distances = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(candidate_distances, axis=1)
        output[start:stop] = np.take_along_axis(candidates, order, axis=1)
    return output


def _normalized_neighbor_count(coords: np.ndarray, neighbor_count: int) -> int:
    n_points = len(coords)
    if n_points < 2:
        return 0
    return max(1, min(int(neighbor_count), n_points - 1))


def _slice_plot_axes(axis: int) -> tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(int(axis))
    return int(axes[0]), int(axes[1])
