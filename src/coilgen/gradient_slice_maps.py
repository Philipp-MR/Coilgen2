from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .engineering_metrics import ConcomitantFieldReport
from .metrics import dsv_target_mask, suppression_target_mask
from .parallel import chunk_ranges, resolve_worker_count
from .physics import biot_savart_wire_path
from .progress import ProgressReporter


AXIS_NAMES = ("x", "y", "z")
FIELD_COMPONENTS = {"bx": 0, "by": 1, "bz": 2}
GRADIENT_SLICE_MAP_CACHE_KEY = "gradient_slice_maps"
BIOT_SAVART_BYTES_PER_TARGET_SEGMENT_PAIR = 160


@dataclass(frozen=True, slots=True)
class GradientSliceMapSpec:
    """Pixel-grid diagnostic definition for MR-gradient slice maps."""

    enabled: bool = False
    axis: str = "z"
    slice_count: int = 9
    grid_size: int | tuple[int, int] = 96
    domain: str = "dsv"
    source: str = "layout"
    current: str = "target"
    field_components: tuple[str, ...] = ("Bx", "By", "Bz")
    gradient_components: tuple[str, ...] = ("dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag")
    concomitant_components: tuple[str, ...] = ("concomitant_Bz", "concomitant_grad_mag")
    gradient_method: str = "finite_difference"
    gradient_step_m: float | None = None
    overlay: tuple[str, ...] = ("wire_path", "contours")
    overlay_mode: str = "slice_slab"
    overlay_slab_thickness_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", _axis_name(self.axis))
        object.__setattr__(self, "slice_count", max(1, int(self.slice_count)))
        object.__setattr__(self, "grid_size", _normalise_grid_size(self.grid_size))
        object.__setattr__(self, "domain", _normalise_domain(self.domain))
        object.__setattr__(self, "source", _normalise_source(self.source))
        object.__setattr__(self, "current", _normalise_current(self.current))
        object.__setattr__(
            self, "field_components", tuple(str(item).strip() for item in self.field_components if str(item).strip())
        )
        object.__setattr__(
            self,
            "gradient_components",
            tuple(str(item).strip() for item in self.gradient_components if str(item).strip()),
        )
        object.__setattr__(
            self,
            "concomitant_components",
            tuple(str(item).strip() for item in self.concomitant_components if str(item).strip()),
        )
        object.__setattr__(self, "gradient_method", _normalise_gradient_method(self.gradient_method))
        if self.gradient_step_m is not None:
            object.__setattr__(self, "gradient_step_m", float(self.gradient_step_m))
        object.__setattr__(
            self, "overlay", tuple(_normalise_overlay_item(item) for item in self.overlay if str(item).strip())
        )
        object.__setattr__(self, "overlay_mode", _normalise_overlay_mode(self.overlay_mode))
        if self.overlay_slab_thickness_m is not None:
            object.__setattr__(self, "overlay_slab_thickness_m", float(self.overlay_slab_thickness_m))

    def to_config_overrides(self) -> dict[str, Any]:
        return {
            "gradient_slice_maps_enabled": bool(self.enabled),
            "gradient_slice_map_axis": self.axis,
            "gradient_slice_map_slice_count": int(self.slice_count),
            "gradient_slice_map_grid_size": self.grid_size,
            "gradient_slice_map_domain": self.domain,
            "gradient_slice_map_source": self.source,
            "gradient_slice_map_current": self.current,
            "gradient_slice_map_field_components": self.field_components,
            "gradient_slice_map_gradient_components": self.gradient_components,
            "gradient_slice_map_concomitant_components": self.concomitant_components,
            "gradient_slice_map_gradient_method": self.gradient_method,
            "gradient_slice_map_gradient_step_m": 0.0 if self.gradient_step_m is None else float(self.gradient_step_m),
            "gradient_slice_map_overlay": self.overlay,
            "gradient_slice_map_overlay_mode": self.overlay_mode,
            "gradient_slice_map_overlay_slab_thickness_m": (
                0.0 if self.overlay_slab_thickness_m is None else float(self.overlay_slab_thickness_m)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GradientSliceMapSlice:
    axis: int
    value: float
    plot_axes: tuple[int, int]
    x: np.ndarray
    y: np.ndarray
    coords: np.ndarray
    mask: np.ndarray
    maps: Mapping[str, np.ndarray] = field(default_factory=dict)
    dsv_outline_center_m: tuple[float, float] | None = None
    dsv_outline_radius_m: float | None = None


@dataclass(frozen=True, slots=True)
class GradientSliceMapResult:
    spec: GradientSliceMapSpec
    axis: int
    axis_name: str
    plot_axes: tuple[int, int]
    center_m: np.ndarray
    reference_radius_m: float
    current_scale_A: float
    gradient_step_m: float
    slices: tuple[GradientSliceMapSlice, ...]
    component_units: Mapping[str, str] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)

    def available_components(self, requested: Sequence[str]) -> tuple[str, ...]:
        keys = set(self.component_units)
        return tuple(component for component in requested if component in keys)


def gradient_slice_map_spec_from_value(value: Any) -> GradientSliceMapSpec:
    if isinstance(value, GradientSliceMapSpec):
        return value
    if isinstance(value, Mapping):
        return GradientSliceMapSpec(**dict(value))
    raise TypeError("slice_maps must be a GradientSliceMapSpec or mapping.")


def gradient_slice_map_spec_from_config(config: Any) -> GradientSliceMapSpec:
    return GradientSliceMapSpec(
        enabled=bool(getattr(config, "gradient_slice_maps_enabled", False)),
        axis=getattr(config, "gradient_slice_map_axis", "z"),
        slice_count=int(getattr(config, "gradient_slice_map_slice_count", 9)),
        grid_size=getattr(config, "gradient_slice_map_grid_size", 96),
        domain=getattr(config, "gradient_slice_map_domain", "dsv"),
        source=getattr(config, "gradient_slice_map_source", "layout"),
        current=getattr(config, "gradient_slice_map_current", "target"),
        field_components=tuple(getattr(config, "gradient_slice_map_field_components", ("Bx", "By", "Bz"))),
        gradient_components=tuple(
            getattr(config, "gradient_slice_map_gradient_components", ("dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag"))
        ),
        concomitant_components=tuple(
            getattr(config, "gradient_slice_map_concomitant_components", ("concomitant_Bz", "concomitant_grad_mag"))
        ),
        gradient_method=getattr(config, "gradient_slice_map_gradient_method", "finite_difference"),
        gradient_step_m=_none_if_not_positive(getattr(config, "gradient_slice_map_gradient_step_m", 0.0)),
        overlay=tuple(getattr(config, "gradient_slice_map_overlay", ("wire_path", "contours"))),
        overlay_mode=getattr(config, "gradient_slice_map_overlay_mode", "slice_slab"),
        overlay_slab_thickness_m=_none_if_not_positive(
            getattr(config, "gradient_slice_map_overlay_slab_thickness_m", 0.0)
        ),
    )


def gradient_slice_maps_enabled(config: Any) -> bool:
    return bool(getattr(config, "gradient_slice_maps_enabled", False))


def gradient_slice_map_result(result: Any, spec: GradientSliceMapSpec | None = None) -> GradientSliceMapResult:
    cache = getattr(result, "diagnostic_cache", None)
    if spec is None and isinstance(cache, dict) and GRADIENT_SLICE_MAP_CACHE_KEY in cache:
        cached = cache[GRADIENT_SLICE_MAP_CACHE_KEY]
        if isinstance(cached, GradientSliceMapResult):
            return cached
    evaluated = evaluate_gradient_slice_maps(result, spec=spec)
    if isinstance(cache, dict):
        cache[GRADIENT_SLICE_MAP_CACHE_KEY] = evaluated
    return evaluated


def evaluate_gradient_slice_maps(result: Any, spec: GradientSliceMapSpec | None = None) -> GradientSliceMapResult:
    config = getattr(result, "input_data", None)
    resolved_spec = spec or gradient_slice_map_spec_from_config(config)
    if not resolved_spec.enabled:
        raise ValueError("Gradient slice maps are disabled.")
    if resolved_spec.gradient_method != "finite_difference":
        raise ValueError("Only gradient_method='finite_difference' is implemented.")

    axis = _axis_index(resolved_spec.axis)
    plot_axes = _slice_plot_axes(axis)
    region = _slice_region(result, resolved_spec.domain, axis=axis, plot_axes=plot_axes)
    grid_x, grid_y = _regular_grid(region.x_range, region.y_range, resolved_spec.grid_size)
    slice_values = _slice_values(region, resolved_spec.slice_count)
    gradient_step = _gradient_step_m(resolved_spec, grid_x, grid_y, slice_values, region.reference_radius_m)
    source = _normalise_source(resolved_spec.source)
    current_scale = _current_scale_A(result, resolved_spec.current)
    polylines = _source_polylines(result, source) if source != "target" else ()
    reporter = ProgressReporter.from_config(config, label="Gradient slice maps")
    task = reporter.task("slice-grid field evaluation", len(slice_values)) if reporter.enabled else None

    raw_slices: list[dict[str, Any]] = []
    component_units: dict[str, str] = {}
    fit_coords: list[np.ndarray] = []
    fit_bz: list[np.ndarray] = []
    for value in slice_values:
        coords_grid, mask = _slice_grid_coords(grid_x, grid_y, value, axis=axis, plot_axes=plot_axes, region=region)
        points = coords_grid[mask]
        raw_slices.append(
            {
                "value": float(value),
                "coords": coords_grid,
                "mask": mask,
                "maps": {},
                "points": points,
            }
        )

    all_points = _stack_slice_points(raw_slices)
    base_all = _evaluate_source_field_batched(
        result,
        source=source,
        points=all_points,
        current=resolved_spec.current,
        current_scale_A=current_scale,
        polylines=polylines,
        config=config,
    )
    derivative_all = (
        _finite_difference_field_gradient_batched(
            result,
            source=source,
            points=all_points,
            step_m=gradient_step,
            current=resolved_spec.current,
            current_scale_A=current_scale,
            polylines=polylines,
            config=config,
        )
        if resolved_spec.gradient_components
        else None
    )

    point_offset = 0
    for slice_ind, item in enumerate(raw_slices, start=1):
        points = item["points"]
        mask = item["mask"]
        point_count = len(points)
        next_offset = point_offset + point_count
        maps = item["maps"]
        base_field = base_all[point_offset:next_offset]
        item["base_field"] = base_field
        _store_field_components(maps, component_units, resolved_spec.field_components, base_field, mask)

        derivative = None
        if derivative_all is not None:
            derivative = derivative_all[point_offset:next_offset]
            item["derivative"] = derivative
            _store_gradient_components(maps, component_units, resolved_spec.gradient_components, derivative, mask)
        else:
            item["derivative"] = None

        if len(points):
            fit_coords.append(points)
            fit_bz.append(base_field[:, 2])
        if task is not None:
            task.update(slice_ind)
        point_offset = next_offset
    if task is not None:
        task.done()

    coeff = _fit_affine_bz(
        np.vstack(fit_coords) if fit_coords else np.zeros((0, 3), dtype=float),
        np.concatenate(fit_bz) if fit_bz else np.zeros(0, dtype=float),
        fit_offset=bool(getattr(config, "linearity_fit_offset", True)),
    )
    gradient = coeff[1:4]
    reference_scale = _linearity_reference_scale_T(result, region.reference_radius_m)
    residual_values: list[np.ndarray] = []
    concomitant_values: list[np.ndarray] = []
    slices: list[GradientSliceMapSlice] = []
    for item in raw_slices:
        points = item["points"]
        mask = item["mask"]
        maps = item["maps"]
        base_field = item["base_field"]
        residual = base_field[:, 2] - _affine_bz(points, coeff)
        residual_values.append(residual)
        if "linearity_residual" in resolved_spec.gradient_components:
            maps["linearity_residual"] = _grid_from_points(residual, mask)
            component_units["linearity_residual"] = "T"
        if resolved_spec.concomitant_components:
            concomitant = _concomitant_bz(points, gradient, config)
            concomitant_values.append(concomitant)
            _store_concomitant_components(
                maps,
                component_units,
                resolved_spec.concomitant_components,
                points,
                concomitant,
                gradient,
                gradient_step,
                mask,
                config,
            )
        slices.append(
            GradientSliceMapSlice(
                axis=axis,
                value=float(item["value"]),
                plot_axes=plot_axes,
                x=grid_x,
                y=grid_y,
                coords=item["coords"],
                mask=mask,
                maps=dict(maps),
                dsv_outline_center_m=_slice_dsv_outline_center(region, plot_axes),
                dsv_outline_radius_m=_slice_dsv_outline_radius(region, axis, float(item["value"])),
            )
        )

    residual_all = np.concatenate(residual_values) if residual_values else np.zeros(0, dtype=float)
    concomitant_all = np.concatenate(concomitant_values) if concomitant_values else np.zeros(0, dtype=float)
    summary = _summary_dict(
        resolved_spec,
        axis=axis,
        sample_count=int(sum(np.count_nonzero(item["mask"]) for item in raw_slices)),
        region=region,
        current_scale=current_scale,
        gradient_step=gradient_step,
        coeff=coeff,
        residual=residual_all,
        reference_scale=reference_scale,
        concomitant=concomitant_all,
    )
    return GradientSliceMapResult(
        spec=resolved_spec,
        axis=axis,
        axis_name=AXIS_NAMES[axis],
        plot_axes=plot_axes,
        center_m=region.center,
        reference_radius_m=region.reference_radius_m,
        current_scale_A=float(current_scale),
        gradient_step_m=float(gradient_step),
        slices=tuple(slices),
        component_units=component_units,
        summary=summary,
    )


def gradient_slice_map_summary_dict(result: Any) -> dict[str, Any]:
    config = getattr(result, "input_data", None)
    if not gradient_slice_maps_enabled(config):
        return {
            "gradient_slice_maps_enabled": False,
            "gradient_slice_maps_status": "not_enabled",
        }
    try:
        evaluated = gradient_slice_map_result(result)
    except ValueError as exc:
        return {
            "gradient_slice_maps_enabled": True,
            "gradient_slice_maps_status": "failed",
            "gradient_slice_maps_failure": str(exc),
        }
    return dict(evaluated.summary)


def gradient_slice_overlay_segments(
    result: Any,
    slice_data: GradientSliceMapSlice,
    *,
    spec: GradientSliceMapSpec | None = None,
) -> dict[str, list[np.ndarray]]:
    resolved_spec = spec or gradient_slice_map_spec_from_config(getattr(result, "input_data", None))
    if resolved_spec.overlay_mode == "none" or not resolved_spec.overlay:
        return {}
    mode = resolved_spec.overlay_mode
    slab = _overlay_slab_thickness(slice_data, resolved_spec)
    half = slab / 2.0
    output: dict[str, list[np.ndarray]] = {}
    for source in resolved_spec.overlay:
        lines = _overlay_polylines(result, source)
        segments: list[np.ndarray] = []
        for line in lines:
            points = np.asarray(line, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
                continue
            starts = points[:-1]
            ends = points[1:]
            if mode == "project_all":
                selected = np.ones(len(starts), dtype=bool)
            else:
                mid = (starts[:, slice_data.axis] + ends[:, slice_data.axis]) / 2.0
                near_mid = np.abs(mid - slice_data.value) <= half
                crosses = (starts[:, slice_data.axis] - slice_data.value) * (
                    ends[:, slice_data.axis] - slice_data.value
                ) <= 0.0
                near_endpoint = (np.abs(starts[:, slice_data.axis] - slice_data.value) <= half) | (
                    np.abs(ends[:, slice_data.axis] - slice_data.value) <= half
                )
                selected = near_mid | (crosses & near_endpoint)
            for start, end in zip(starts[selected], ends[selected]):
                segments.append(
                    np.asarray(
                        [
                            [start[slice_data.plot_axes[0]], start[slice_data.plot_axes[1]]],
                            [end[slice_data.plot_axes[0]], end[slice_data.plot_axes[1]]],
                        ],
                        dtype=float,
                    )
                )
        if segments:
            output[source] = segments
    return output


@dataclass(frozen=True, slots=True)
class _SliceRegion:
    center: np.ndarray
    reference_radius_m: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    spherical: bool


def _slice_region(result: Any, domain: str, *, axis: int, plot_axes: tuple[int, int]) -> _SliceRegion:
    target = getattr(result, "target_field", None)
    coords = np.asarray(getattr(target, "coords", np.zeros((0, 3))), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError("Gradient slice maps require target-field coordinates.")
    mask = _domain_mask(target, domain)
    finite = mask & np.all(np.isfinite(coords), axis=1)
    if not np.any(finite):
        raise ValueError(f"Gradient slice maps have no finite points in domain {domain!r}.")
    selected = coords[finite]
    center = np.mean(selected, axis=0)
    config = getattr(result, "input_data", None)
    configured_radius = _positive_float(getattr(config, "linearity_reference_radius_m", 0.0))
    if configured_radius <= 0.0:
        configured_radius = _positive_float(getattr(config, "target_region_radius", 0.0))
    spherical = _normalise_domain(domain) == "dsv"
    radius = (
        configured_radius
        if configured_radius > 0.0 and spherical
        else float(np.max(np.linalg.norm(selected - center, axis=1)))
    )
    radius = max(radius, 1.0e-12)
    if spherical:
        x_range = (float(center[plot_axes[0]] - radius), float(center[plot_axes[0]] + radius))
        y_range = (float(center[plot_axes[1]] - radius), float(center[plot_axes[1]] + radius))
        z_range = (float(center[axis] - radius), float(center[axis] + radius))
    else:
        mins = np.min(selected, axis=0)
        maxs = np.max(selected, axis=0)
        x_range = _expanded_range((mins[plot_axes[0]], maxs[plot_axes[0]]))
        y_range = _expanded_range((mins[plot_axes[1]], maxs[plot_axes[1]]))
        z_range = _expanded_range((mins[axis], maxs[axis]))
    return _SliceRegion(
        center=np.asarray(center, dtype=float),
        reference_radius_m=float(radius),
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        spherical=spherical,
    )


def _domain_mask(target_field: Any, domain: str) -> np.ndarray:
    key = _normalise_domain(domain)
    if key == "dsv":
        return dsv_target_mask(target_field)
    if key == "suppression":
        return suppression_target_mask(target_field)
    if key == "all":
        return np.ones(len(getattr(target_field, "coords", ())), dtype=bool)
    raise ValueError(f"Unsupported gradient slice-map domain: {domain!r}")


def _regular_grid(
    x_range: tuple[float, float], y_range: tuple[float, float], grid_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    x_count, y_count = grid_size
    return np.linspace(x_range[0], x_range[1], x_count), np.linspace(y_range[0], y_range[1], y_count)


def _slice_values(region: _SliceRegion, slice_count: int) -> np.ndarray:
    if slice_count <= 1 or region.z_range[1] <= region.z_range[0]:
        return np.asarray([(region.z_range[0] + region.z_range[1]) / 2.0], dtype=float)
    if region.spherical:
        lower, upper = region.z_range
        count = int(slice_count)
        step = (upper - lower) / float(count)
        return lower + (np.arange(count, dtype=float) + 0.5) * step
    return np.linspace(region.z_range[0], region.z_range[1], int(slice_count))


def _slice_dsv_outline_center(region: _SliceRegion, plot_axes: tuple[int, int]) -> tuple[float, float] | None:
    if not region.spherical:
        return None
    return (float(region.center[plot_axes[0]]), float(region.center[plot_axes[1]]))


def _slice_dsv_outline_radius(region: _SliceRegion, axis: int, value: float) -> float | None:
    if not region.spherical:
        return None
    dz = float(value) - float(region.center[axis])
    radius_squared = float(region.reference_radius_m) ** 2 - dz * dz
    if radius_squared <= 0.0:
        return None
    return float(np.sqrt(radius_squared))


def _slice_grid_coords(
    x: np.ndarray,
    y: np.ndarray,
    z_value: float,
    *,
    axis: int,
    plot_axes: tuple[int, int],
    region: _SliceRegion,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.zeros((len(y), len(x), 3), dtype=float)
    coords[:, :, plot_axes[0]] = xx
    coords[:, :, plot_axes[1]] = yy
    coords[:, :, axis] = float(z_value)
    if region.spherical:
        centered = coords - region.center.reshape(1, 1, 3)
        mask = np.linalg.norm(centered, axis=2) <= region.reference_radius_m * (1.0 + 1.0e-12)
    else:
        mask = np.ones((len(y), len(x)), dtype=bool)
    return coords, mask


def _stack_slice_points(raw_slices: Sequence[Mapping[str, Any]]) -> np.ndarray:
    point_arrays = [np.asarray(item["points"], dtype=float) for item in raw_slices if len(item["points"])]
    return np.vstack(point_arrays) if point_arrays else np.zeros((0, 3), dtype=float)


def _evaluate_source_field_batched(
    result: Any,
    *,
    source: str,
    points: np.ndarray,
    current: str,
    current_scale_A: float,
    polylines: Sequence[np.ndarray],
    config: Any,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    if source == "target":
        return _evaluate_source_field(
            result,
            source=source,
            points=points,
            current=current,
            current_scale_A=current_scale_A,
            polylines=polylines,
            config=config,
        )
    batch_size = _field_sample_batch_size(config)
    if len(points) <= batch_size:
        return _evaluate_source_field(
            result,
            source=source,
            points=points,
            current=current,
            current_scale_A=current_scale_A,
            polylines=polylines,
            config=config,
        )
    field = np.zeros((len(points), 3), dtype=float)
    for start, stop in chunk_ranges(len(points), batch_size):
        field[start:stop] = _evaluate_source_field(
            result,
            source=source,
            points=points[start:stop],
            current=current,
            current_scale_A=current_scale_A,
            polylines=polylines,
            config=config,
        )
    return field


def _evaluate_source_field(
    result: Any,
    *,
    source: str,
    points: np.ndarray,
    current: str,
    current_scale_A: float,
    polylines: Sequence[np.ndarray],
    config: Any,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    if source == "target":
        return _target_field_at_points(result, points, current=current)
    if not polylines:
        raise ValueError(f"Gradient slice-map source {source!r} has no evaluable wire or contour paths.")
    worker_count = resolve_worker_count(config)
    field = np.zeros((len(points), 3), dtype=float)
    for line in polylines:
        field += biot_savart_wire_path(
            line,
            points,
            chunk_size=int(getattr(config, "biot_savart_chunk_size", 512)),
            worker_count=worker_count,
            min_work_size=int(getattr(config, "parallel_min_work_size", 50000)),
            max_chunk_memory_mb=float(getattr(config, "parallel_max_chunk_memory_mb", 64.0)),
        )
    return field * float(current_scale_A)


def _target_field_at_points(result: Any, points: np.ndarray, *, current: str) -> np.ndarray:
    coeff = _target_affine_coeff(result)
    scale = _target_source_current_scale(result, current)
    field = np.zeros((len(points), 3), dtype=float)
    field[:, 2] = _affine_bz(points, coeff) * scale
    return field


def _target_source_current_scale(result: Any, current: str) -> float:
    key = _normalise_current(current)
    if key == "target":
        return 1.0
    target_current = _target_current_A(result)
    if not np.isfinite(target_current) or abs(target_current) <= 1.0e-30:
        return 1.0
    if key == "per_amp":
        return float(1.0 / target_current)
    if key == "stream_function":
        return float(_current_scale_A(result, "stream_function") / target_current)
    return 1.0


def _target_affine_coeff(result: Any) -> np.ndarray:
    target = getattr(result, "target_field", None)
    coords = np.asarray(getattr(target, "coords", np.zeros((0, 3))), dtype=float)
    field = np.asarray(getattr(target, "b", np.zeros((0, 3))), dtype=float)
    if (
        coords.ndim == 2
        and coords.shape[1] == 3
        and field.ndim == 2
        and field.shape[1] >= 3
        and len(coords) == len(field)
    ):
        mask = _domain_mask(target, "dsv") & np.all(np.isfinite(coords), axis=1) & np.isfinite(field[:, 2])
        coeff = _fit_affine_bz(coords[mask], field[mask, 2], fit_offset=True)
        if np.all(np.isfinite(coeff)):
            return coeff
    config = getattr(result, "input_data", None)
    axis = _axis_index(getattr(config, "target_gradient_axis", getattr(config, "field_shape_function", "z")))
    gradient = np.zeros(3, dtype=float)
    gradient[axis] = _positive_float(getattr(config, "target_gradient_strength", 0.0))
    return np.concatenate(([0.0], gradient))


def _finite_difference_field_gradient_batched(
    result: Any,
    *,
    source: str,
    points: np.ndarray,
    step_m: float,
    current: str,
    current_scale_A: float,
    polylines: Sequence[np.ndarray],
    config: Any,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros((0, 3, 3), dtype=float)
    derivative = np.zeros((len(points), 3, 3), dtype=float)
    sample_batch_size = max(_field_sample_batch_size(config) // 2, 1)
    for start, stop in chunk_ranges(len(points), sample_batch_size):
        chunk = points[start:stop]
        for dim in range(3):
            offset = np.zeros(3, dtype=float)
            offset[dim] = float(step_m)
            paired_points = np.vstack((chunk + offset, chunk - offset))
            paired_field = _evaluate_source_field_batched(
                result,
                source=source,
                points=paired_points,
                current=current,
                current_scale_A=current_scale_A,
                polylines=polylines,
                config=config,
            )
            midpoint = len(chunk)
            derivative[start:stop, :, dim] = (paired_field[:midpoint] - paired_field[midpoint:]) / (2.0 * float(step_m))
    return derivative


def _finite_difference_field_gradient(
    result: Any,
    *,
    source: str,
    points: np.ndarray,
    step_m: float,
    current: str,
    current_scale_A: float,
    polylines: Sequence[np.ndarray],
    config: Any,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3, 3), dtype=float)
    derivative = np.zeros((len(points), 3, 3), dtype=float)
    for dim in range(3):
        offset = np.zeros(3, dtype=float)
        offset[dim] = float(step_m)
        plus = _evaluate_source_field(
            result,
            source=source,
            points=points + offset,
            current=current,
            current_scale_A=current_scale_A,
            polylines=polylines,
            config=config,
        )
        minus = _evaluate_source_field(
            result,
            source=source,
            points=points - offset,
            current=current,
            current_scale_A=current_scale_A,
            polylines=polylines,
            config=config,
        )
        derivative[:, :, dim] = (plus - minus) / (2.0 * float(step_m))
    return derivative


def _field_sample_batch_size(config: Any) -> int:
    try:
        max_chunk_memory_mb = float(getattr(config, "parallel_max_chunk_memory_mb", 64.0))
    except (TypeError, ValueError):
        max_chunk_memory_mb = 64.0
    if not np.isfinite(max_chunk_memory_mb) or max_chunk_memory_mb <= 0.0:
        max_chunk_memory_mb = 64.0
    max_bytes = max(max_chunk_memory_mb, 1.0) * 1024.0 * 1024.0
    return max(int(max_bytes // BIOT_SAVART_BYTES_PER_TARGET_SEGMENT_PAIR), 1)


def _store_field_components(
    maps: dict[str, np.ndarray],
    component_units: dict[str, str],
    components: Sequence[str],
    field: np.ndarray,
    mask: np.ndarray,
) -> None:
    for component in components:
        key = _normalise_component_key(component)
        if key not in FIELD_COMPONENTS:
            continue
        public_key = _field_component_public_key(key)
        maps[public_key] = _grid_from_points(field[:, FIELD_COMPONENTS[key]], mask)
        component_units[public_key] = "T"


def _store_gradient_components(
    maps: dict[str, np.ndarray],
    component_units: dict[str, str],
    components: Sequence[str],
    derivative: np.ndarray,
    mask: np.ndarray,
) -> None:
    for component in components:
        key = _normalise_component_key(component)
        if key == "gradbz_mag":
            values = np.linalg.norm(derivative[:, 2, :], axis=1)
            maps["gradBz_mag"] = _grid_from_points(values, mask)
            component_units["gradBz_mag"] = "T/m"
            continue
        parsed = _parse_gradient_component(key)
        if parsed is None:
            continue
        field_ind, axis_ind, public_key = parsed
        maps[public_key] = _grid_from_points(derivative[:, field_ind, axis_ind], mask)
        component_units[public_key] = "T/m"


def _store_concomitant_components(
    maps: dict[str, np.ndarray],
    component_units: dict[str, str],
    components: Sequence[str],
    points: np.ndarray,
    concomitant: np.ndarray,
    gradient: np.ndarray,
    step_m: float,
    mask: np.ndarray,
    config: Any,
) -> None:
    for component in components:
        key = _normalise_component_key(component)
        if key in {"concomitant_bz", "concomitant"}:
            maps["concomitant_Bz"] = _grid_from_points(concomitant, mask)
            component_units["concomitant_Bz"] = "T"
        elif key in {"concomitant_percent", "concomitant_percent_of_target_peak", "concomitant_bz_percent"}:
            target_peak = _positive_float(
                abs(float(getattr(config, "target_gradient_strength", 0.0)))
                * float(getattr(config, "target_region_radius", 0.0))
            )
            values = np.full(len(concomitant), np.nan, dtype=float)
            if target_peak > 0.0:
                values = 100.0 * concomitant / target_peak
            maps["concomitant_percent_of_target_peak"] = _grid_from_points(values, mask)
            component_units["concomitant_percent_of_target_peak"] = "%"
        elif key == "concomitant_grad_mag":
            values = _concomitant_gradient_magnitude(points, gradient, step_m, config)
            maps["concomitant_grad_mag"] = _grid_from_points(values, mask)
            component_units["concomitant_grad_mag"] = "T/m"


def _concomitant_bz(points: np.ndarray, gradient: np.ndarray, config: Any) -> np.ndarray:
    return ConcomitantFieldReport.field_values(
        points,
        gradient,
        main_field_T=float(getattr(config, "b_0_strength_T", 1.0)),
        b0_direction=getattr(config, "b_0_direction", (0.0, 0.0, 1.0)),
    )


def _concomitant_gradient_magnitude(points: np.ndarray, gradient: np.ndarray, step_m: float, config: Any) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    components = []
    for dim in range(3):
        offset = np.zeros(3, dtype=float)
        offset[dim] = float(step_m)
        plus = _concomitant_bz(points + offset, gradient, config)
        minus = _concomitant_bz(points - offset, gradient, config)
        components.append((plus - minus) / (2.0 * float(step_m)))
    return np.linalg.norm(np.column_stack(components), axis=1)


def _grid_from_points(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grid = np.full(mask.shape, np.nan, dtype=float)
    if np.count_nonzero(mask) == len(values):
        grid[mask] = np.asarray(values, dtype=float).reshape(-1)
    return grid


def _source_polylines(
    result: Any,
    source: str,
    *,
    expand_turn_multiplicity: bool = True,
) -> tuple[np.ndarray, ...]:
    lines: list[np.ndarray] = []
    for part in getattr(result, "coil_parts", ()) or ():
        if source == "layout":
            wire = getattr(part, "wire_path", None)
            points = np.asarray(getattr(wire, "v", np.zeros((0, 3))), dtype=float)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2:
                lines.append(points)
        elif source in {"contours", "stream"}:
            for contour in getattr(part, "contour_lines", ()) or ():
                points = np.asarray(getattr(contour, "v", np.zeros((0, 3))), dtype=float)
                if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2:
                    repetitions = int(getattr(contour, "turn_multiplicity", 1)) if expand_turn_multiplicity else 1
                    lines.extend([points] * max(repetitions, 1))
    return tuple(lines)


def _overlay_polylines(result: Any, source: str) -> tuple[np.ndarray, ...]:
    key = _normalise_overlay_item(source)
    if key == "wire_path":
        return _source_polylines(result, "layout")
    if key in {"contours", "raw_contours"}:
        return _source_polylines(result, "contours", expand_turn_multiplicity=False)
    return tuple()


def _current_scale_A(result: Any, current: str) -> float:
    key = _normalise_current(current)
    if key == "per_amp":
        return 1.0
    if key == "stream_function":
        value = _positive_float(getattr(result, "potential_step", np.nan))
        return value if value > 0.0 else 1.0
    value = _target_current_A(result)
    return value if np.isfinite(value) and abs(value) > 1.0e-30 else 1.0


def _target_current_A(result: Any) -> float:
    layout_gradient = getattr(result, "layout_gradient", None)
    config = getattr(result, "input_data", None)
    if isinstance(layout_gradient, Mapping):
        achieved = _finite_float(layout_gradient.get("achieved_gradient", np.nan))
        requested = _finite_float(getattr(config, "target_gradient_strength", np.nan))
        if np.isfinite(achieved) and abs(achieved) > 1.0e-30 and np.isfinite(requested):
            return float(requested / achieved)
    return _finite_float(getattr(result, "needed_current_layout", np.nan))


def _fit_affine_bz(coords: np.ndarray, bz: np.ndarray, *, fit_offset: bool) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    bz = np.asarray(bz, dtype=float).reshape(-1)
    finite = (
        np.all(np.isfinite(coords), axis=1) & np.isfinite(bz)
        if coords.ndim == 2 and coords.shape[1] == 3
        else np.zeros(0, dtype=bool)
    )
    if np.count_nonzero(finite) < (4 if fit_offset else 3):
        return np.full(4, np.nan, dtype=float)
    selected = coords[finite]
    values = bz[finite]
    if fit_offset:
        matrix = np.column_stack([np.ones(len(selected)), selected])
        coeff, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        return np.asarray(coeff, dtype=float)
    gradient, *_ = np.linalg.lstsq(selected, values, rcond=None)
    return np.concatenate(([0.0], np.asarray(gradient, dtype=float).reshape(3)))


def _affine_bz(points: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=float)
    if coeff.shape != (4,) or not np.all(np.isfinite(coeff)):
        return np.full(len(points), np.nan, dtype=float)
    return coeff[0] + np.asarray(points, dtype=float) @ coeff[1:4]


def _linearity_reference_scale_T(result: Any, radius: float) -> float:
    config = getattr(result, "input_data", None)
    requested = _positive_float(getattr(config, "target_gradient_strength", 0.0))
    if requested > 0.0 and radius > 0.0:
        return float(abs(requested) * radius)
    return float("nan")


def _summary_dict(
    spec: GradientSliceMapSpec,
    *,
    axis: int,
    sample_count: int,
    region: _SliceRegion,
    current_scale: float,
    gradient_step: float,
    coeff: np.ndarray,
    residual: np.ndarray,
    reference_scale: float,
    concomitant: np.ndarray,
) -> dict[str, Any]:
    finite_residual = residual[np.isfinite(residual)]
    finite_concomitant = concomitant[np.isfinite(concomitant)]
    rms_residual = float(np.sqrt(np.mean(finite_residual * finite_residual))) if len(finite_residual) else float("nan")
    peak_residual = float(np.max(np.abs(finite_residual))) if len(finite_residual) else float("nan")
    return {
        "gradient_slice_maps_enabled": bool(spec.enabled),
        "gradient_slice_maps_status": "evaluated",
        "gradient_slice_map_axis": AXIS_NAMES[axis],
        "gradient_slice_map_domain": spec.domain,
        "gradient_slice_map_source": spec.source,
        "gradient_slice_map_current": spec.current,
        "gradient_slice_map_slice_count": int(spec.slice_count),
        "gradient_slice_map_grid_x_count": int(spec.grid_size[0]),
        "gradient_slice_map_grid_y_count": int(spec.grid_size[1]),
        "gradient_slice_map_sample_count": int(sample_count),
        "gradient_slice_map_current_A": float(current_scale),
        "gradient_slice_map_gradient_step_m": float(gradient_step),
        "gradient_slice_map_reference_radius_m": float(region.reference_radius_m),
        "gradient_slice_map_reference_scale_T": float(reference_scale),
        "gradient_slice_map_fit_offset_T": float(coeff[0]) if coeff.shape == (4,) else float("nan"),
        "gradient_slice_map_fit_dbz_dx_T_per_m": float(coeff[1]) if coeff.shape == (4,) else float("nan"),
        "gradient_slice_map_fit_dbz_dy_T_per_m": float(coeff[2]) if coeff.shape == (4,) else float("nan"),
        "gradient_slice_map_fit_dbz_dz_T_per_m": float(coeff[3]) if coeff.shape == (4,) else float("nan"),
        "gradient_slice_map_linearity_rms_residual_T": rms_residual,
        "gradient_slice_map_linearity_peak_abs_residual_T": peak_residual,
        "gradient_slice_map_linearity_rms_percent": _percent(rms_residual, reference_scale),
        "gradient_slice_map_linearity_peak_percent": _percent(peak_residual, reference_scale),
        "gradient_slice_map_concomitant_rms_T": (
            float(np.sqrt(np.mean(finite_concomitant * finite_concomitant)))
            if len(finite_concomitant)
            else float("nan")
        ),
        "gradient_slice_map_concomitant_peak_abs_T": (
            float(np.max(np.abs(finite_concomitant))) if len(finite_concomitant) else float("nan")
        ),
    }


def _gradient_step_m(
    spec: GradientSliceMapSpec,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    slice_values: np.ndarray,
    radius: float,
) -> float:
    configured = _positive_float(spec.gradient_step_m)
    if configured > 0.0:
        return configured
    spacings = []
    for values in (grid_x, grid_y, slice_values):
        diffs = np.diff(np.asarray(values, dtype=float))
        positive = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0.0)])
        if len(positive):
            spacings.append(float(np.min(positive)))
    typical = min(spacings) if spacings else max(float(radius) * 0.02, 1.0e-4)
    return max(typical * 0.25, float(radius) * 1.0e-4, 1.0e-6)


def _overlay_slab_thickness(slice_data: GradientSliceMapSlice, spec: GradientSliceMapSpec) -> float:
    configured = _positive_float(spec.overlay_slab_thickness_m)
    if configured > 0.0:
        return configured
    dx = _typical_spacing(slice_data.x)
    dy = _typical_spacing(slice_data.y)
    return max(min(value for value in (dx, dy) if value > 0.0), 1.0e-6) if dx > 0.0 or dy > 0.0 else 1.0e-3


def _normalise_grid_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("gradient slice-map grid_size must be an int or (x_count, y_count).")
        return max(8, int(value[0])), max(8, int(value[1]))
    count = max(8, int(value))
    return count, count


def _normalise_domain(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "dsv",
        "fov": "dsv",
        "roi": "dsv",
        "target": "dsv",
        "primary": "dsv",
        "dsv": "dsv",
        "suppression": "suppression",
        "secondary": "suppression",
        "shield": "suppression",
        "all": "all",
        "any": "all",
        "full": "all",
    }
    return aliases.get(key, key)


def _normalise_source(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "layout",
        "layout": "layout",
        "wire": "layout",
        "wire_path": "layout",
        "connected": "layout",
        "connected_wire": "layout",
        "contour": "contours",
        "contours": "contours",
        "loops": "contours",
        "unconnected": "contours",
        "stream": "stream",
        "stream_function": "stream",
        "sf": "stream",
        "target": "target",
        "ideal": "target",
    }
    return aliases.get(key, key)


def _normalise_current(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "target",
        "target": "target",
        "requested": "target",
        "operating": "target",
        "fitted": "target",
        "per_amp": "per_amp",
        "per_a": "per_amp",
        "1a": "per_amp",
        "unit": "per_amp",
        "stream": "stream_function",
        "stream_function": "stream_function",
        "contour_step": "stream_function",
        "sf": "stream_function",
    }
    return aliases.get(key, key)


def _normalise_gradient_method(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {"": "finite_difference", "finite": "finite_difference", "fd": "finite_difference"}
    return aliases.get(key, key)


def _normalise_overlay_item(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "wire": "wire_path",
        "wire_path": "wire_path",
        "layout": "wire_path",
        "connected_wire": "wire_path",
        "contour": "contours",
        "contours": "contours",
        "raw_contours": "contours",
        "loops": "contours",
    }
    return aliases.get(key, key)


def _normalise_overlay_mode(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "slice_slab",
        "slab": "slice_slab",
        "slice": "slice_slab",
        "slice_slab": "slice_slab",
        "project_all": "project_all",
        "all": "project_all",
        "none": "none",
        "off": "none",
    }
    return aliases.get(key, key)


def _axis_index(value: Any) -> int:
    key = _axis_name(value)
    return {"x": 0, "y": 1, "z": 2}[key]


def _axis_name(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("g") and len(text) == 2:
        text = text[1]
    if text in {"0", "x"}:
        return "x"
    if text in {"1", "y"}:
        return "y"
    if text in {"2", "z"}:
        return "z"
    raise ValueError("gradient slice-map axis must be 'x', 'y', or 'z'.")


def _slice_plot_axes(axis: int) -> tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(int(axis))
    return int(axes[0]), int(axes[1])


def _normalise_component_key(value: Any) -> str:
    return str(value).strip().replace("/", "_").replace(" ", "_").lower()


def _field_component_public_key(key: str) -> str:
    return {"bx": "Bx", "by": "By", "bz": "Bz"}[key]


def _parse_gradient_component(key: str) -> tuple[int, int, str] | None:
    aliases = {
        "dbx_dx": (0, 0, "dBx_dx"),
        "dbx_dy": (0, 1, "dBx_dy"),
        "dbx_dz": (0, 2, "dBx_dz"),
        "dby_dx": (1, 0, "dBy_dx"),
        "dby_dy": (1, 1, "dBy_dy"),
        "dby_dz": (1, 2, "dBy_dz"),
        "dbz_dx": (2, 0, "dBz_dx"),
        "dbz_dy": (2, 1, "dBz_dy"),
        "dbz_dz": (2, 2, "dBz_dz"),
    }
    return aliases.get(key)


def _positive_float(value: Any) -> float:
    number = _finite_float(value)
    return float(number) if np.isfinite(number) and number > 0.0 else 0.0


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _none_if_not_positive(value: Any) -> float | None:
    number = _positive_float(value)
    return number if number > 0.0 else None


def _expanded_range(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return -1.0, 1.0
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    if maximum <= minimum:
        pad = max(abs(minimum), 1.0) * 0.05
        return minimum - pad, maximum + pad
    pad = 0.02 * (maximum - minimum)
    return minimum - pad, maximum + pad


def _typical_spacing(values: np.ndarray) -> float:
    diffs = np.diff(np.asarray(values, dtype=float))
    positive = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0.0)])
    return float(np.median(positive)) if len(positive) else 0.0


def _percent(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) <= 1.0e-30:
        return float("nan")
    return float(100.0 * abs(value) / abs(reference))
