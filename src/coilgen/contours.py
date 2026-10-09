from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .basis import current_density_from_basis
from .config import CoilGenConfig
from .contour_smoothing import MeshContourBSplineSmoother
from .engineering_limits import (
    engineering_contour_max_levels,
    engineering_contouring_enabled,
    engineering_limit_diagnostics,
    engineering_max_contour_step_A,
    max_wire_current_A,
    min_wire_distance_m,
)
from .field_components import component_sample_values, field_component_values
from .metrics import dsv_target_mask
from .models import CoilPart, ContourLine, Mesh, TargetField, WirePath
from .parallel import parallel_map, resolve_worker_count, should_parallelize
from .physics import biot_savart_wire_path
from .progress import ProgressReporter, progress_scope
from .symmetry import (
    absolute_tolerance as _symmetry_absolute_tolerance,
    generated_candidate_symmetric_part_pairs as _symmetry_candidate_part_pairs,
    geometry_scale as _symmetry_geometry_scale,
    nearest_indices as _symmetry_nearest_indices,
    normalise_symmetry_mode,
    reflect_points as _symmetry_reflect_points,
)


@dataclass(frozen=True)
class SurfaceContourSegment:
    points: np.ndarray
    barycentric: np.ndarray
    face_index: int
    endpoint_keys: tuple[tuple[int, int, int], tuple[int, int, int]]


def calc_potential_levels(
    parts: list[CoilPart], combined_mesh: Mesh, config: CoilGenConfig
) -> tuple[list[CoilPart], int]:
    if engineering_contouring_enabled(config):
        return _calc_engineering_potential_levels(parts, combined_mesh, config)

    num_levels = int(config.levels)
    level_offset = float(config.pot_offset_factor)
    if config.level_set_method == "primary":
        level_references = [_contour_level_reference_values(part.stream_function, config) for part in parts]
        ranges = np.asarray([reference["range"] for reference in level_references], dtype=float)
        primary_surface_ind = int(np.argmax(ranges))
        primary_level_offset = _part_contour_phase(config, primary_surface_ind)
        num_levels, spacing_diagnostics = _spacing_limited_level_count_and_diagnostics(
            parts,
            ranges[primary_surface_ind],
            num_levels,
            primary_level_offset,
            config,
        )
        spacing_limited_primary_level_count = num_levels
        contour_step = _requested_level_count_contour_step(
            ranges[primary_surface_ind],
            num_levels,
            primary_level_offset,
            config,
        )
        feasible_counts = [
            _fixed_step_level_count(
                float(sf_range),
                contour_step,
                _part_contour_phase(config, part_ind),
                config,
            )
            for part_ind, sf_range in enumerate(ranges)
        ]
        final_counts = list(feasible_counts)
        final_counts[primary_surface_ind] = min(num_levels, feasible_counts[primary_surface_ind])
        pair_labels = [""] * len(parts)
        pair_reduced = [False] * len(parts)
        for pair in _generated_symmetric_part_pairs(parts, config):
            if not pair:
                continue
            pair_count = min(final_counts[part_ind] for part_ind in pair)
            label = ",".join(str(part_ind + 1) for part_ind in pair)
            for part_ind in pair:
                pair_labels[part_ind] = label
                pair_reduced[part_ind] = final_counts[part_ind] != pair_count
                final_counts[part_ind] = pair_count
                if part_ind == primary_surface_ind:
                    num_levels = pair_count
        for part_ind, part in enumerate(parts):
            reference = level_references[part_ind]
            reference_values = np.asarray(reference["values"], dtype=float)
            part_level_offset = _part_contour_phase(config, part_ind)
            part.contour_step = float(contour_step)
            part.potential_level_list = _fixed_step_contour_levels(
                reference_values,
                contour_step,
                final_counts[part_ind],
                zero_crossing_strategy=str(getattr(config, "zero_crossing_level_strategy", "paired")),
                phase_offset=part_level_offset,
                grid_mode=_contour_level_grid_mode(config),
            )
            zero_crossing_strategy = _normalise_zero_crossing_level_strategy(config)
            requested_count = int(final_counts[part_ind])
            selected_count = int(len(part.potential_level_list))
            part.contour_level_diagnostics = {
                "method": "primary_shared_current_step",
                "primary_surface": bool(part_ind == primary_surface_ind),
                "stream_function_min": float(np.min(part.stream_function)),
                "stream_function_max": float(np.max(part.stream_function)),
                "stream_function_range": float(np.ptp(part.stream_function)),
                "contour_level_range_percentile": float(reference["percentile"]),
                "contour_level_reference_min": float(reference["min"]),
                "contour_level_reference_max": float(reference["max"]),
                "contour_level_reference_range": float(reference["range"]),
                "contour_step_A": float(contour_step),
                "contour_phase_base": float(level_offset),
                "contour_phase_delta": float(_part_contour_phase_delta(config, part_ind)),
                "contour_phase_effective": float(part_level_offset),
                "requested_level_count": int(config.levels),
                "spacing_limited_primary_level_count": int(spacing_limited_primary_level_count),
                "feasible_level_count": int(feasible_counts[part_ind]),
                "final_level_count": selected_count,
                "zero_crossing_level_strategy": zero_crossing_strategy,
                "zero_crossing_paired_levels": bool(zero_crossing_strategy == "paired"),
                "zero_crossing_even_level_count_enforced": bool(
                    zero_crossing_strategy == "paired" and selected_count != requested_count
                ),
                "contour_levels_gauge_invariant": bool(zero_crossing_strategy == "paired"),
                "contour_level_origin_A": float(_contour_level_origin(reference_values)),
                "paired_level_count_even_enforced": bool(
                    zero_crossing_strategy == "paired" and selected_count != requested_count
                ),
                "symmetry_pair": pair_labels[part_ind],
                "symmetry_pair_locked": bool(pair_labels[part_ind]),
                "symmetry_pair_reduced_level_count": bool(pair_reduced[part_ind]),
            }
            part.contour_level_diagnostics.update(spacing_diagnostics)
        _record_symmetric_level_diagnostics(parts, config)
        return parts, primary_surface_ind
    if config.level_set_method == "combined":
        sf = getattr(combined_mesh, "stream_function")
        reference = _contour_level_reference_values(sf, config)
        reference_values = np.asarray(reference["values"], dtype=float)
        num_levels = _spacing_limited_level_count(parts, float(reference["range"]), num_levels, level_offset, config)
        step = _requested_level_count_contour_step(float(reference["range"]), num_levels, level_offset, config)
        for part_ind, part in enumerate(parts):
            part_level_offset = _part_contour_phase(config, part_ind)
            part.contour_step = float(step)
            part.potential_level_list = _fixed_step_contour_levels(
                reference_values,
                float(step),
                int(num_levels),
                zero_crossing_strategy=str(getattr(config, "zero_crossing_level_strategy", "paired")),
                phase_offset=part_level_offset,
                grid_mode=_contour_level_grid_mode(config),
            )
        return parts, 0
    if config.level_set_method == "independent":
        for part_ind, part in enumerate(parts):
            part_level_offset = _part_contour_phase(config, part_ind)
            reference = _contour_level_reference_values(part.stream_function, config)
            reference_values = np.asarray(reference["values"], dtype=float)
            part_levels = _spacing_limited_level_count(
                [part],
                float(reference["range"]),
                num_levels,
                part_level_offset,
                config,
            )
            step = _requested_level_count_contour_step(
                float(reference["range"]), part_levels, part_level_offset, config
            )
            part.contour_step = float(step)
            part.potential_level_list = _fixed_step_contour_levels(
                reference_values,
                float(step),
                int(part_levels),
                zero_crossing_strategy=str(getattr(config, "zero_crossing_level_strategy", "paired")),
                phase_offset=part_level_offset,
                grid_mode=_contour_level_grid_mode(config),
            )
        return parts, 0
    raise ValueError(f"Unknown level_set_method: {config.level_set_method!r}.")


def _calc_engineering_potential_levels(
    parts: list[CoilPart],
    combined_mesh: Mesh,
    config: CoilGenConfig,
) -> tuple[list[CoilPart], int]:
    level_offset = float(config.pot_offset_factor)
    base_step, base_diagnostics = _engineering_base_contour_step_and_diagnostics(parts, config)
    if config.level_set_method == "primary":
        level_references = [_contour_level_reference_values(part.stream_function, config) for part in parts]
        ranges = np.asarray([reference["range"] for reference in level_references], dtype=float)
        primary_surface_ind = int(np.argmax(ranges))
        primary_level_offset = _part_contour_phase(config, primary_surface_ind)
        contour_step, cap_diagnostics = _engineering_apply_level_cap(
            base_step,
            float(ranges[primary_surface_ind]),
            primary_level_offset,
            config,
        )
        final_counts = [
            _engineering_fixed_step_level_count(
                float(sf_range),
                contour_step,
                _part_contour_phase(config, part_ind),
                config,
            )
            for part_ind, sf_range in enumerate(ranges)
        ]
        pair_labels = [""] * len(parts)
        pair_reduced = [False] * len(parts)
        for pair in _generated_symmetric_part_pairs(parts, config):
            if not pair:
                continue
            pair_count = min(final_counts[part_ind] for part_ind in pair)
            label = ",".join(str(part_ind + 1) for part_ind in pair)
            for part_ind in pair:
                pair_labels[part_ind] = label
                pair_reduced[part_ind] = final_counts[part_ind] != pair_count
                final_counts[part_ind] = pair_count
        zero_crossing_strategy = _normalise_zero_crossing_level_strategy(config)
        for part_ind, part in enumerate(parts):
            reference = level_references[part_ind]
            reference_values = np.asarray(reference["values"], dtype=float)
            part_level_offset = _part_contour_phase(config, part_ind)
            part.contour_step = float(contour_step)
            part.potential_level_list = _fixed_step_contour_levels(
                reference_values,
                contour_step,
                final_counts[part_ind],
                zero_crossing_strategy=zero_crossing_strategy,
                phase_offset=part_level_offset,
                grid_mode=_contour_level_grid_mode(config),
            )
            requested_count = int(final_counts[part_ind])
            selected_count = int(len(part.potential_level_list))
            part.contour_level_diagnostics = {
                **base_diagnostics,
                **cap_diagnostics,
                "method": "engineering_current_spacing",
                "primary_surface": bool(part_ind == primary_surface_ind),
                "stream_function_min": float(np.min(part.stream_function)),
                "stream_function_max": float(np.max(part.stream_function)),
                "stream_function_range": float(np.ptp(part.stream_function)),
                "contour_level_range_percentile": float(reference["percentile"]),
                "contour_level_reference_min": float(reference["min"]),
                "contour_level_reference_max": float(reference["max"]),
                "contour_level_reference_range": float(reference["range"]),
                "contour_step_A": float(contour_step),
                "contour_phase_base": float(level_offset),
                "contour_phase_delta": float(_part_contour_phase_delta(config, part_ind)),
                "contour_phase_effective": float(part_level_offset),
                "requested_level_count": 0,
                "feasible_level_count": int(final_counts[part_ind]),
                "final_level_count": selected_count,
                "zero_crossing_level_strategy": zero_crossing_strategy,
                "zero_crossing_paired_levels": bool(zero_crossing_strategy == "paired"),
                "zero_crossing_even_level_count_enforced": bool(
                    zero_crossing_strategy == "paired" and selected_count != requested_count
                ),
                "contour_levels_gauge_invariant": bool(zero_crossing_strategy == "paired"),
                "contour_level_origin_A": float(_contour_level_origin(reference_values)),
                "paired_level_count_even_enforced": bool(
                    zero_crossing_strategy == "paired" and selected_count != requested_count
                ),
                "symmetry_pair": pair_labels[part_ind],
                "symmetry_pair_locked": bool(pair_labels[part_ind]),
                "symmetry_pair_reduced_level_count": bool(pair_reduced[part_ind]),
            }
        _record_symmetric_level_diagnostics(parts, config)
        return parts, primary_surface_ind

    if config.level_set_method == "combined":
        sf = getattr(combined_mesh, "stream_function")
        reference = _contour_level_reference_values(sf, config)
        reference_values = np.asarray(reference["values"], dtype=float)
        contour_step, cap_diagnostics = _engineering_apply_level_cap(
            base_step,
            float(reference["range"]),
            level_offset,
            config,
        )
        level_count = _engineering_fixed_step_level_count(float(reference["range"]), contour_step, level_offset, config)
        for part_ind, part in enumerate(parts):
            part_level_offset = _part_contour_phase(config, part_ind)
            part.contour_step = float(contour_step)
            part.potential_level_list = _fixed_step_contour_levels(
                reference_values,
                float(contour_step),
                int(level_count),
                zero_crossing_strategy=str(getattr(config, "zero_crossing_level_strategy", "paired")),
                phase_offset=part_level_offset,
                grid_mode=_contour_level_grid_mode(config),
            )
            part.contour_level_diagnostics = {
                **base_diagnostics,
                **cap_diagnostics,
                "method": "engineering_current_spacing_combined",
                "contour_step_A": float(contour_step),
                "contour_phase_base": float(level_offset),
                "contour_phase_delta": float(_part_contour_phase_delta(config, part_ind)),
                "contour_phase_effective": float(part_level_offset),
                "final_level_count": int(len(part.potential_level_list)),
            }
        return parts, 0

    if config.level_set_method == "independent":
        for part_ind, part in enumerate(parts):
            part_level_offset = _part_contour_phase(config, part_ind)
            reference = _contour_level_reference_values(part.stream_function, config)
            reference_values = np.asarray(reference["values"], dtype=float)
            contour_step, cap_diagnostics = _engineering_apply_level_cap(
                base_step,
                float(reference["range"]),
                part_level_offset,
                config,
            )
            level_count = _engineering_fixed_step_level_count(
                float(reference["range"]), contour_step, part_level_offset, config
            )
            part.contour_step = float(contour_step)
            part.potential_level_list = _fixed_step_contour_levels(
                reference_values,
                float(contour_step),
                int(level_count),
                zero_crossing_strategy=str(getattr(config, "zero_crossing_level_strategy", "paired")),
                phase_offset=part_level_offset,
                grid_mode=_contour_level_grid_mode(config),
            )
            part.contour_level_diagnostics = {
                **base_diagnostics,
                **cap_diagnostics,
                "method": "engineering_current_spacing_independent",
                "contour_step_A": float(contour_step),
                "contour_phase_base": float(level_offset),
                "contour_phase_delta": float(_part_contour_phase_delta(config, part_ind)),
                "contour_phase_effective": float(part_level_offset),
                "final_level_count": int(len(part.potential_level_list)),
            }
        return parts, 0
    raise ValueError(f"Unknown level_set_method: {config.level_set_method!r}.")


def _engineering_base_contour_step_and_diagnostics(
    parts: list[CoilPart],
    config: CoilGenConfig,
) -> tuple[float, dict[str, float | int | bool | str]]:
    current_limit = engineering_max_contour_step_A(config)
    distance = min_wire_distance_m(config)
    density_percentile = _turn_spacing_density_percentile(config)
    required_step = 0.0
    reference_density = 0.0
    max_density = 0.0
    for part in parts:
        current_density = _part_current_density(part)
        if current_density is None:
            continue
        density_norm = np.linalg.norm(current_density, axis=1)
        finite = density_norm[np.isfinite(density_norm) & (density_norm > 0.0)]
        if finite.size:
            part_reference = _current_density_reference_value(finite, density_percentile)
            part_max = float(np.max(finite))
            reference_density = max(reference_density, part_reference)
            max_density = max(max_density, part_max)
            if distance > 0.0:
                required_step = max(required_step, distance * part_reference)

    if required_step > 0.0:
        selected_step = required_step
        reason = "min_wire_distance_times_current_density"
    else:
        selected_step = current_limit
        reason = "max_current_no_density_reference"
    feasible = True
    if current_limit > 0.0 and selected_step > current_limit:
        selected_step = current_limit
        feasible = False
        reason = "max_current_cap_spacing_bound_infeasible"
    if selected_step <= 0.0 or not np.isfinite(selected_step):
        raise ValueError(
            "Engineering contouring could not determine a positive contour step. "
            "Set max_current_A/max_wire_current_A and min_wire_distance_m."
        )
    diagnostics: dict[str, float | int | bool | str] = {
        **engineering_limit_diagnostics(config),
        "engineering_requested_max_wire_current_A": float(max_wire_current_A(config)),
        "engineering_required_contour_step_A": float(required_step),
        "engineering_reference_current_density_A_per_m": float(reference_density),
        "engineering_max_current_density_A_per_m": float(max_density),
        "engineering_selected_contour_step_A": float(selected_step),
        "engineering_selected_contour_current_A": float(selected_step),
        "engineering_current_limit_feasible": bool(feasible),
        "engineering_contour_step_selection_reason": reason,
        "turn_spacing_density_percentile": float(density_percentile),
    }
    return float(selected_step), diagnostics


def _engineering_apply_level_cap(
    base_step: float,
    stream_function_range: float,
    level_offset: float,
    config: CoilGenConfig,
) -> tuple[float, dict[str, float | int | bool | str]]:
    selected_step = float(base_step)
    max_levels = engineering_contour_max_levels(config)
    current_limit = engineering_max_contour_step_A(config)
    cap_step = 0.0
    cap_applied = False
    cap_feasible = True
    if max_levels > 0 and np.isfinite(stream_function_range) and stream_function_range > 0.0:
        if _contour_level_grid_mode(config) == "fixed_step_phase":
            denominator = max(float(max_levels), np.finfo(float).eps)
        else:
            denominator = max(float(max_levels - 1) + 2.0 * float(level_offset), np.finfo(float).eps)
        cap_step = float(stream_function_range) / denominator
        if cap_step > selected_step:
            if current_limit > 0.0 and cap_step > current_limit:
                selected_step = current_limit
                cap_feasible = False
            else:
                selected_step = cap_step
                cap_applied = True
    level_count = _engineering_fixed_step_level_count(stream_function_range, selected_step, level_offset, config)
    diagnostics: dict[str, float | int | bool | str] = {
        "engineering_level_cap_requested": bool(max_levels > 0),
        "engineering_level_cap_max_levels": int(max_levels),
        "engineering_level_cap_step_A": float(cap_step),
        "engineering_level_cap_applied": bool(cap_applied),
        "engineering_level_cap_feasible_with_current_limit": bool(cap_feasible),
        "engineering_effective_level_count": int(level_count),
    }
    return float(selected_step), diagnostics


def _engineering_fixed_step_level_count(
    stream_function_range: float,
    contour_step: float,
    level_offset: float,
    config: CoilGenConfig,
) -> int:
    count = _fixed_step_level_count(stream_function_range, contour_step, level_offset, config)
    if count <= 0 and np.isfinite(stream_function_range) and stream_function_range > 0.0:
        return 1
    return int(count)


def _contour_level_reference_values(
    stream_function: np.ndarray, config: CoilGenConfig
) -> dict[str, float | np.ndarray]:
    values = np.asarray(stream_function, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "values": values,
            "percentile": 100.0,
            "min": float("nan"),
            "max": float("nan"),
            "range": 0.0,
        }
    percentile = _contour_level_range_percentile(config)
    if percentile >= 100.0:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    else:
        tail = 0.5 * (100.0 - percentile)
        lower, upper = (float(value) for value in np.percentile(finite, [tail, 100.0 - tail]))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower = float(np.min(finite))
            upper = float(np.max(finite))
            percentile = 100.0
    reference = np.clip(values, lower, upper)
    return {
        "values": reference,
        "percentile": float(percentile),
        "min": lower,
        "max": upper,
        "range": float(max(upper - lower, 0.0)),
    }


def _contour_level_range_percentile(config: CoilGenConfig) -> float:
    try:
        value = float(getattr(config, "contour_level_range_percentile", 100.0))
    except (TypeError, ValueError):
        return 100.0
    if not np.isfinite(value):
        return 100.0
    return min(max(value, np.finfo(float).eps), 100.0)


def _spacing_limited_level_count(
    parts: list[CoilPart],
    stream_function_range: float,
    requested_levels: int,
    level_offset: float,
    config: CoilGenConfig,
) -> int:
    count, _ = _spacing_limited_level_count_and_diagnostics(
        parts,
        stream_function_range,
        requested_levels,
        level_offset,
        config,
    )
    return count


def _spacing_limited_level_count_and_diagnostics(
    parts: list[CoilPart],
    stream_function_range: float,
    requested_levels: int,
    level_offset: float,
    config: CoilGenConfig,
) -> tuple[int, dict[str, float | int | bool]]:
    requested_count = max(int(requested_levels), 1)
    threshold = float(getattr(config, "min_turn_distance", 0.0))
    density_percentile = _turn_spacing_density_percentile(config)
    diagnostics: dict[str, float | int | bool] = {
        "turn_spacing_level_limiter_enabled": bool(np.isfinite(threshold) and threshold > 0.0),
        "turn_spacing_density_percentile": float(density_percentile),
        "turn_spacing_requested_min_distance_m": float(threshold),
        "turn_spacing_requested_level_count": int(requested_count),
        "turn_spacing_stream_function_range_A": float(stream_function_range),
        "turn_spacing_required_contour_step_A": 0.0,
        "turn_spacing_reference_current_density_A_per_m": 0.0,
        "turn_spacing_max_current_density_A_per_m": 0.0,
        "turn_spacing_unlimited_level_count": int(requested_count),
        "turn_spacing_limited_level_count": int(requested_count),
    }
    if not np.isfinite(threshold) or threshold <= 0.0:
        return requested_count, diagnostics
    required_step = 0.0
    reference_density = 0.0
    max_density = 0.0
    for part in parts:
        current_density = _part_current_density(part)
        if current_density is None:
            continue
        density_norm = np.linalg.norm(current_density, axis=1)
        finite = density_norm[np.isfinite(density_norm) & (density_norm > 0.0)]
        if finite.size:
            part_reference = _current_density_reference_value(finite, density_percentile)
            part_max = float(np.max(finite))
            reference_density = max(reference_density, part_reference)
            max_density = max(max_density, part_max)
            required_step = max(required_step, threshold * part_reference)
    diagnostics["turn_spacing_required_contour_step_A"] = float(required_step)
    diagnostics["turn_spacing_reference_current_density_A_per_m"] = float(reference_density)
    diagnostics["turn_spacing_max_current_density_A_per_m"] = float(max_density)
    if required_step <= 0.0 or not np.isfinite(stream_function_range) or stream_function_range <= 0.0:
        return requested_count, diagnostics
    if _contour_level_grid_mode(config) == "fixed_step_phase":
        max_levels = int(np.floor(stream_function_range / required_step))
    else:
        max_levels = int(np.floor(stream_function_range / required_step - 2.0 * level_offset + 1.0))
    limited_count = max(1, min(requested_count, max_levels))
    diagnostics["turn_spacing_limited_level_count"] = int(limited_count)
    return limited_count, diagnostics


def _turn_spacing_density_percentile(config: CoilGenConfig) -> float:
    try:
        value = float(getattr(config, "turn_spacing_density_percentile", 100.0))
    except (TypeError, ValueError):
        return 100.0
    if not np.isfinite(value):
        return 100.0
    return min(max(value, np.finfo(float).eps), 100.0)


def _current_density_reference_value(finite_density: np.ndarray, percentile: float) -> float:
    finite = np.asarray(finite_density, dtype=float)
    if finite.size == 0:
        return 0.0
    if percentile >= 100.0:
        return float(np.max(finite))
    return float(np.percentile(finite, percentile))


def _fixed_step_level_count(
    stream_function_range: float,
    contour_step: float,
    level_offset: float,
    config: CoilGenConfig | None = None,
) -> int:
    if (
        not np.isfinite(stream_function_range)
        or not np.isfinite(contour_step)
        or contour_step <= 0.0
        or stream_function_range <= 0.0
    ):
        return 0
    if config is not None and _contour_level_grid_mode(config) == "fixed_step_phase":
        phase = _safe_contour_phase(level_offset)
        raw_count = np.ceil(stream_function_range / contour_step - phase)
    else:
        raw_count = stream_function_range / contour_step - 2.0 * float(level_offset) + 1.0
    tolerance = 32.0 * np.finfo(float).eps * max(abs(raw_count), 1.0)
    return max(int(np.floor(raw_count + tolerance)), 0)


def _fixed_step_contour_levels(
    stream_function: np.ndarray,
    contour_step: float,
    level_count: int,
    *,
    zero_crossing_strategy: str = "paired",
    phase_offset: float = 0.5,
    grid_mode: str = "centered_margin",
) -> np.ndarray:
    if _normalise_contour_level_grid_mode(grid_mode) == "fixed_step_phase":
        if _normalise_zero_crossing_level_strategy(zero_crossing_strategy) == "paired":
            return _paired_centered_fixed_step_levels(stream_function, contour_step, level_count)
        return _phase_shifted_fixed_step_levels(stream_function, contour_step, level_count, phase_offset)
    if _normalise_zero_crossing_level_strategy(zero_crossing_strategy) == "paired":
        return _paired_centered_fixed_step_levels(stream_function, contour_step, level_count)
    return _centered_fixed_step_levels(stream_function, contour_step, level_count)


def _phase_shifted_fixed_step_levels(
    stream_function: np.ndarray,
    contour_step: float,
    level_count: int,
    phase_offset: float,
) -> np.ndarray:
    count = max(int(level_count), 0)
    if count <= 0 or not np.isfinite(contour_step) or contour_step <= 0.0:
        return np.zeros(0, dtype=float)
    values = np.asarray(stream_function, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(0, dtype=float)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    phase = _safe_contour_phase(phase_offset)
    levels = lower + (np.arange(count, dtype=float) + phase) * float(contour_step)
    tolerance = 64.0 * np.finfo(float).eps * max(abs(lower), abs(upper), 1.0)
    return levels[(levels > lower + tolerance) & (levels < upper - tolerance)]


def _requested_level_count_contour_step(
    stream_function_range: float,
    level_count: int,
    level_offset: float,
    config: CoilGenConfig,
) -> float:
    if _contour_level_grid_mode(config) == "fixed_step_phase":
        denominator = max(float(max(int(level_count), 1)), np.finfo(float).eps)
    else:
        denominator = max(float(int(level_count) - 1) + 2.0 * float(level_offset), np.finfo(float).eps)
    return float(stream_function_range) / denominator


def _contour_level_grid_mode(config: CoilGenConfig) -> str:
    return _normalise_contour_level_grid_mode(getattr(config, "contour_level_grid_mode", "centered_margin"))


def _normalise_contour_level_grid_mode(value: object) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"fixed_step_phase", "fixed_phase", "quantizer_phase"}:
        return "fixed_step_phase"
    return "centered_margin"


def _safe_contour_phase(value: object) -> float:
    try:
        phase = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not np.isfinite(phase):
        return 0.5
    return float(phase % 1.0)


def _part_contour_phase_delta(config: CoilGenConfig, part_index: int) -> float:
    deltas = tuple(getattr(config, "contour_part_phase_offset_deltas", ()) or ())
    if part_index < 0 or part_index >= len(deltas):
        return 0.0
    return _safe_contour_phase(deltas[part_index])


def _part_contour_phase(config: CoilGenConfig, part_index: int) -> float:
    """Return the effective quantizer phase for one disconnected support part."""

    return _safe_contour_phase(float(config.pot_offset_factor) + _part_contour_phase_delta(config, part_index))


def _normalise_zero_crossing_level_strategy(config_or_value: CoilGenConfig | str) -> str:
    value = (
        getattr(config_or_value, "zero_crossing_level_strategy", "paired")
        if not isinstance(config_or_value, str)
        else config_or_value
    )
    key = str(value).strip().lower().replace("-", "_")
    if key in {"paired", "symmetric"}:
        return "paired"
    if key in {"full_range", "centered"}:
        return "full_range"
    return "paired"


def _contour_level_origin(stream_function: np.ndarray) -> float:
    values = np.asarray(stream_function, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return 0.5 * (float(np.min(finite)) + float(np.max(finite)))


def _paired_centered_fixed_step_levels(
    stream_function: np.ndarray, contour_step: float, level_count: int
) -> np.ndarray:
    """Return gauge-invariant paired contour levels around the stream midrange.

    The absolute value of the stream function is arbitrary: adding a constant
    leaves the surface current density and B-field unchanged.  Pairing levels
    around the midrange therefore preserves the old symmetric level intent
    without making the contour topology depend on whether zero is inside the
    current gauge.
    """

    count = int(level_count)
    if count % 2:
        count -= 1
    if count < 2 or not np.isfinite(contour_step) or contour_step <= 0.0:
        return np.zeros(0, dtype=float)
    values = np.asarray(stream_function, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(0, dtype=float)
    origin = _contour_level_origin(finite)
    if not np.isfinite(origin):
        return np.zeros(0, dtype=float)
    positive_span = float(np.max(finite)) - origin
    negative_span = origin - float(np.min(finite))
    limiting_span = min(positive_span, negative_span)
    if not np.isfinite(limiting_span) or limiting_span <= 0.0:
        return np.zeros(0, dtype=float)
    max_half_count = int(np.floor(limiting_span / float(contour_step) + 0.5 - 32.0 * np.finfo(float).eps))
    half_count = min(count // 2, max_half_count)
    if half_count <= 0:
        return np.zeros(0, dtype=float)
    offsets = (np.arange(half_count, dtype=float) + 0.5) * float(contour_step)
    return np.concatenate((origin - offsets[::-1], origin + offsets))


def _centered_fixed_step_levels(stream_function: np.ndarray, contour_step: float, level_count: int) -> np.ndarray:
    count = int(level_count)
    if count <= 0 or not np.isfinite(contour_step) or contour_step <= 0.0:
        return np.zeros(0, dtype=float)
    sf_min = float(np.min(stream_function))
    sf_range = float(np.ptp(stream_function))
    span = float(max(count - 1, 0)) * float(contour_step)
    margin = 0.5 * (sf_range - span)
    if not np.isfinite(margin):
        return np.zeros(0, dtype=float)
    margin = max(margin, 0.0)
    return np.arange(count, dtype=float) * float(contour_step) + (sf_min + margin)


def _generated_symmetric_part_pairs(parts: list[CoilPart], config: CoilGenConfig) -> list[tuple[int, ...]]:
    return [
        pair
        for pair in _generated_candidate_symmetric_part_pairs(parts, config)
        if _parts_have_matching_mesh_size(parts, pair)
    ]


def _generated_candidate_symmetric_part_pairs(parts: list[CoilPart], config: CoilGenConfig) -> list[tuple[int, ...]]:
    return _symmetry_candidate_part_pairs(str(getattr(config, "coil_mesh_file", "")), len(parts))


def _parts_have_matching_mesh_size(parts: list[CoilPart], pair: tuple[int, ...]) -> bool:
    if not pair:
        return False
    first = parts[pair[0]].mesh
    return all(
        parts[part_ind].mesh.n_vertices == first.n_vertices and parts[part_ind].mesh.n_faces == first.n_faces
        for part_ind in pair[1:]
    )


def _record_symmetric_level_diagnostics(parts: list[CoilPart], config: CoilGenConfig) -> None:
    mode = _contour_symmetry_validation_mode(config)
    if mode == "off":
        return
    reports = []
    for pair in _generated_candidate_symmetric_part_pairs(parts, config):
        report, match_indices = _symmetric_geometry_report(parts, pair, config)
        report.update(_symmetric_stream_function_report(parts, pair, match_indices, config))
        report.update(_symmetric_level_report(parts, pair))
        _write_pair_symmetry_diagnostics(parts, pair, report, mode)
        reports.append(report)
    if _single_mesh_symmetry_validation_requested(parts, config):
        report = _single_mesh_level_symmetry_report(parts[0], config)
        parts[0].contour_level_diagnostics["symmetry_validation_mode"] = mode
        parts[0].contour_level_diagnostics.update(report)
        reports.append(report)
    _raise_for_strict_symmetry_failures(reports, mode, stage="level selection")


def _contour_symmetry_validation_mode(config: CoilGenConfig) -> str:
    return normalise_symmetry_mode(getattr(config, "contour_symmetry_validation", "auto"))


def _symmetric_geometry_report(
    parts: list[CoilPart],
    pair: tuple[int, ...],
    config: CoilGenConfig,
) -> tuple[dict[str, object], np.ndarray | None]:
    report: dict[str, object] = {
        "symmetry_pair": ",".join(str(part_ind + 1) for part_ind in pair),
        "symmetry_pair_indices": [int(part_ind) for part_ind in pair],
        "symmetry_geometry_checked": False,
        "symmetry_geometry_ok": False,
        "symmetry_warnings": [],
    }
    warnings: list[str] = report["symmetry_warnings"]  # type: ignore[assignment]
    if len(pair) != 2:
        warnings.append("Only two-part mirror symmetry pairs are supported.")
        return report, None
    part_a, part_b = parts[pair[0]], parts[pair[1]]
    if not _parts_have_matching_mesh_size(parts, pair):
        warnings.append("Symmetry pair parts have different mesh sizes.")
        report.update(
            {
                "symmetry_geometry_checked": True,
                "symmetry_geometry_vertex_count_a": int(part_a.mesh.n_vertices),
                "symmetry_geometry_vertex_count_b": int(part_b.mesh.n_vertices),
                "symmetry_geometry_face_count_a": int(part_a.mesh.n_faces),
                "symmetry_geometry_face_count_b": int(part_b.mesh.n_faces),
            }
        )
        return report, None

    vertices_a = np.asarray(part_a.mesh.vertices, dtype=float)
    vertices_b = np.asarray(part_b.mesh.vertices, dtype=float)
    center_a = np.mean(vertices_a, axis=0)
    center_b = np.mean(vertices_b, axis=0)
    normal = center_b - center_a
    centroid_distance = float(np.linalg.norm(normal))
    scale = _pair_geometry_scale(vertices_a, vertices_b)
    if centroid_distance <= 1e-14 * max(scale, 1.0):
        warnings.append("Symmetry pair centroids are coincident; mirror plane cannot be inferred.")
        report.update(
            {
                "symmetry_geometry_checked": True,
                "symmetry_geometry_scale_m": float(scale),
                "symmetry_geometry_centroid_distance_m": centroid_distance,
            }
        )
        return report, None

    normal = normal / centroid_distance
    plane_point = 0.5 * (center_a + center_b)
    reflected_a = _reflect_points(vertices_a, plane_point, normal)
    direct_distances = np.linalg.norm(reflected_a - vertices_b, axis=1)
    direct_max = float(np.max(direct_distances)) if direct_distances.size else 0.0
    tolerance = _geometry_abs_tolerance(config, scale)
    match_indices: np.ndarray | None = np.arange(len(vertices_a), dtype=np.int64)
    distances = direct_distances
    if direct_max > tolerance:
        nearest_indices = _nearest_vertex_indices(reflected_a, vertices_b)
        if nearest_indices is not None:
            nearest_distances = np.linalg.norm(reflected_a - vertices_b[nearest_indices], axis=1)
            if float(np.max(nearest_distances)) < direct_max:
                match_indices = nearest_indices
                distances = nearest_distances
        else:
            match_indices = None

    max_distance = float(np.max(distances)) if distances.size else float("inf")
    rms_distance = float(np.sqrt(np.mean(distances**2))) if distances.size else float("inf")
    geometry_ok = bool(np.isfinite(max_distance) and max_distance <= tolerance)
    if not geometry_ok:
        warnings.append("Reflected vertex cloud does not match its symmetry partner within tolerance.")
    report.update(
        {
            "symmetry_geometry_checked": True,
            "symmetry_geometry_ok": geometry_ok,
            "symmetry_geometry_scale_m": float(scale),
            "symmetry_geometry_tolerance_m": float(tolerance),
            "symmetry_geometry_centroid_distance_m": centroid_distance,
            "symmetry_geometry_rms_distance_m": rms_distance,
            "symmetry_geometry_max_distance_m": max_distance,
            "symmetry_mirror_plane_point_m": [float(value) for value in plane_point],
            "symmetry_mirror_plane_normal": [float(value) for value in normal],
        }
    )
    return report, match_indices if geometry_ok else None


def _pair_geometry_scale(vertices_a: np.ndarray, vertices_b: np.ndarray) -> float:
    return _symmetry_geometry_scale(np.vstack((vertices_a, vertices_b)))


def _geometry_abs_tolerance(config: CoilGenConfig, scale: float) -> float:
    relative_tolerance = float(getattr(config, "contour_symmetry_geometry_tolerance", 1e-6))
    return _symmetry_absolute_tolerance(relative_tolerance, scale)


def _reflect_points(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return _symmetry_reflect_points(points, plane_point, normal)


def _nearest_vertex_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    return _symmetry_nearest_indices(source, target)


def _symmetric_stream_function_report(
    parts: list[CoilPart],
    pair: tuple[int, ...],
    match_indices: np.ndarray | None,
    config: CoilGenConfig,
) -> dict[str, object]:
    report: dict[str, object] = {
        "symmetry_stream_function_checked": False,
        "symmetry_stream_function_ok": False,
    }
    if len(pair) != 2 or match_indices is None:
        return report
    part_a, part_b = parts[pair[0]], parts[pair[1]]
    if part_a.stream_function is None or part_b.stream_function is None:
        return report
    psi_a = np.asarray(part_a.stream_function, dtype=float)
    psi_b = np.asarray(part_b.stream_function, dtype=float)
    if len(psi_a) != len(match_indices) or len(psi_b) <= int(np.max(match_indices, initial=-1)):
        return report
    matched_b = psi_b[match_indices]
    plus = _stream_symmetry_residual(psi_a, matched_b, sign=1.0)
    minus = _stream_symmetry_residual(psi_a, matched_b, sign=-1.0)
    selected = plus if plus["rms"] <= minus["rms"] else minus
    scale = max(float(np.ptp(psi_a)), float(np.ptp(matched_b)), float(np.max(np.abs(psi_a))), 1e-30)
    normalized_rms = float(selected["rms"] / scale)
    normalized_max = float(selected["max"] / scale)
    tolerance = float(getattr(config, "contour_symmetry_stream_function_tolerance", 1e-5))
    return {
        "symmetry_stream_function_checked": True,
        "symmetry_stream_function_ok": bool(normalized_rms <= tolerance),
        "symmetry_stream_function_sign": int(selected["sign"]),
        "symmetry_stream_function_offset_A": float(selected["offset"]),
        "symmetry_stream_function_rms_A": float(selected["rms"]),
        "symmetry_stream_function_max_A": float(selected["max"]),
        "symmetry_stream_function_scale_A": float(scale),
        "symmetry_stream_function_normalized_rms": normalized_rms,
        "symmetry_stream_function_normalized_max": normalized_max,
        "symmetry_stream_function_tolerance": tolerance,
    }


def _stream_symmetry_residual(psi_a: np.ndarray, psi_b: np.ndarray, *, sign: float) -> dict[str, float]:
    offset = float(np.mean(psi_b - sign * psi_a))
    residual = psi_b - (sign * psi_a + offset)
    return {
        "sign": float(sign),
        "offset": offset,
        "rms": float(np.sqrt(np.mean(residual * residual))),
        "max": float(np.max(np.abs(residual))),
    }


def _symmetric_level_report(parts: list[CoilPart], pair: tuple[int, ...]) -> dict[str, object]:
    counts = [
        0 if parts[part_ind].potential_level_list is None else int(len(parts[part_ind].potential_level_list))
        for part_ind in pair
    ]
    steps = [
        float("nan") if parts[part_ind].contour_step is None else float(parts[part_ind].contour_step)
        for part_ind in pair
    ]
    finite_steps = [step for step in steps if np.isfinite(step)]
    step_delta = max(finite_steps) - min(finite_steps) if finite_steps else float("nan")
    step_scale = max(max((abs(step) for step in finite_steps), default=0.0), 1.0)
    relative_step_delta = float(step_delta / step_scale) if np.isfinite(step_delta) else float("nan")
    centering_errors = [_level_centering_error(parts[part_ind]) for part_ind in pair]
    level_count_delta = max(counts) - min(counts) if counts else 0
    step_ok = bool(len(finite_steps) == len(pair) and np.isfinite(relative_step_delta) and relative_step_delta <= 1e-12)
    level_ok = bool(level_count_delta == 0 and step_ok)
    return {
        "symmetry_level_checked": True,
        "symmetry_level_ok": level_ok,
        "symmetry_level_counts": counts,
        "symmetry_level_count_delta": int(level_count_delta),
        "symmetry_level_contour_steps_A": steps,
        "symmetry_level_step_relative_delta": relative_step_delta,
        "symmetry_level_centering_error_A": centering_errors,
    }


def _level_centering_error(part: CoilPart) -> float:
    levels = part.potential_level_list
    if levels is None or len(levels) == 0 or part.stream_function is None:
        return float("nan")
    sf_min = float(np.min(part.stream_function))
    sf_max = float(np.max(part.stream_function))
    lower = float(levels[0] - sf_min)
    upper = float(sf_max - levels[-1])
    return abs(lower - upper)


def _write_pair_symmetry_diagnostics(
    parts: list[CoilPart],
    pair: tuple[int, ...],
    report: dict[str, object],
    mode: str,
) -> None:
    for part_ind in pair:
        if not 0 <= part_ind < len(parts):
            continue
        parts[part_ind].contour_level_diagnostics["symmetry_validation_mode"] = mode
        parts[part_ind].contour_level_diagnostics.update(report)


def _raise_for_strict_symmetry_failures(reports: list[dict[str, object]], mode: str, *, stage: str) -> None:
    if mode != "strict":
        return
    messages: list[str] = []
    for report in reports:
        label = str(report.get("symmetry_pair", "?"))
        if not bool(report.get("symmetry_geometry_ok", False)):
            messages.append(f"pair {label}: geometric mirror check failed")
        if report.get("symmetry_stream_function_checked") and not bool(
            report.get("symmetry_stream_function_ok", False)
        ):
            messages.append(f"pair {label}: stream-function mirror check failed")
        if report.get("symmetry_level_checked") and not bool(report.get("symmetry_level_ok", False)):
            messages.append(f"pair {label}: contour-level symmetry check failed")
        if report.get("symmetry_extracted_contour_checked") and not bool(
            report.get("symmetry_extracted_contour_ok", False)
        ):
            messages.append(f"pair {label}: extracted-contour symmetry check failed")
    if messages:
        raise ValueError(f"Contour symmetry validation failed after {stage}: " + "; ".join(messages))


def extract_contours(parts: list[CoilPart], config: CoilGenConfig) -> list[CoilPart]:
    reporter = ProgressReporter.from_config(config, label="Contours")
    for part in parts:
        part.potential_level_list = regularize_contour_levels(
            part.mesh, part.stream_function, part.potential_level_list
        )
    task = reporter.task(
        "support levels, cleanup and symmetry review", sum(len(part.potential_level_list) + 1 for part in parts) + 1
    )
    for part_ind, part in enumerate(parts, start=1):
        mesh = part.mesh
        contours: list[ContourLine] = []
        stitch_diagnostics = _empty_stitch_diagnostics()
        regularized_levels = part.potential_level_list
        levels = list(regularized_levels)
        for level in levels:
            segments = _segments_for_level(mesh, part.stream_function, float(level))
            stitched_contours, level_stitch_diagnostics = _stitch_segments_with_diagnostics(segments, float(level))
            contours.extend(stitched_contours)
            _accumulate_stitch_diagnostics(stitch_diagnostics, level_stitch_diagnostics)
            task.advance()
        reporter.report(None, f"cleaning contours on support {part_ind}/{len(parts)}")
        closed_contour_count_before_cleaning = len(contours)
        cleaned_contours = []
        for contour in contours:
            cleaned = clean_contour_line(contour, config)
            if cleaned is not None and len(cleaned.v) >= config.min_point_loop_number:
                cleaned_contours.append(cleaned)
        contours = validate_and_correct_contour_currents(part, cleaned_contours)
        part.contour_lines = contours
        part.combined_loop_length = float(sum(contour.effective_turn_length for contour in contours))
        part.contour_level_diagnostics["regularized_level_count"] = int(len(regularized_levels))
        part.contour_level_diagnostics["extracted_contour_count"] = int(len(contours))
        part.contour_level_diagnostics["contour_cleaning_rejected_count"] = int(
            max(0, closed_contour_count_before_cleaning - len(cleaned_contours))
        )
        part.contour_level_diagnostics["contour_cleaning_rejected_fraction"] = (
            float(max(0, closed_contour_count_before_cleaning - len(cleaned_contours)))
            / float(closed_contour_count_before_cleaning)
            if closed_contour_count_before_cleaning > 0
            else 0.0
        )
        part.contour_level_diagnostics.update(_finalize_stitch_diagnostics(stitch_diagnostics))
        task.advance()
    _enforce_symmetric_extracted_contour_counts(parts, config)
    _record_symmetric_extraction_diagnostics(parts, config)
    task.done()
    return parts


def _enforce_symmetric_extracted_contour_counts(parts: list[CoilPart], config: CoilGenConfig) -> None:
    mode = _contour_symmetry_validation_mode(config)
    if mode == "off":
        return
    for pair in _generated_candidate_symmetric_part_pairs(parts, config):
        if len(pair) != 2 or not _parts_have_matching_mesh_size(parts, pair):
            continue
        part_a, part_b = parts[pair[0]], parts[pair[1]]
        if not part_a.contour_lines and not part_b.contour_lines:
            continue
        before_counts = [len(part_a.contour_lines), len(part_b.contour_lines)]
        sign = int(part_a.contour_level_diagnostics.get("symmetry_stream_function_sign", 1))
        keep_a, keep_b = _symmetric_contour_keep_indices(part_a, part_b, sign=sign)
        if keep_a is None or keep_b is None:
            continue
        removed_a = len(part_a.contour_lines) - len(keep_a)
        removed_b = len(part_b.contour_lines) - len(keep_b)
        if removed_a <= 0 and removed_b <= 0:
            continue
        part_a.contour_lines = [part_a.contour_lines[ind] for ind in sorted(keep_a)]
        part_b.contour_lines = [part_b.contour_lines[ind] for ind in sorted(keep_b)]
        for part in (part_a, part_b):
            part.combined_loop_length = float(sum(contour.effective_turn_length for contour in part.contour_lines))
            part.contour_level_diagnostics["extracted_contour_count"] = int(len(part.contour_lines))
        report = {
            "symmetry_pair_extracted_contour_counts_before_pruning": before_counts,
            "symmetry_pair_extracted_contours_pruned": [int(removed_a), int(removed_b)],
            "symmetry_pair_extracted_contour_pruning_applied": True,
        }
        part_a.contour_level_diagnostics.update(report)
        part_b.contour_level_diagnostics.update(report)


def _symmetric_contour_keep_indices(
    part_a: CoilPart,
    part_b: CoilPart,
    *,
    sign: int,
) -> tuple[set[int], set[int]] | tuple[None, None]:
    if part_a.potential_level_list is None or part_b.potential_level_list is None:
        return None, None
    levels_a = np.asarray(part_a.potential_level_list, dtype=float)
    levels_b = np.asarray(part_b.potential_level_list, dtype=float)
    if levels_a.size == 0 or levels_b.size == 0:
        return None, None
    by_level_a = _contours_by_nearest_level(part_a)
    by_level_b = _contours_by_nearest_level(part_b)
    if not by_level_a and not by_level_b:
        return None, None

    keep_a: set[int] = set()
    keep_b: set[int] = set()
    paired_level_count = min(len(levels_a), len(levels_b))
    for level_a in range(paired_level_count):
        level_b = level_a if sign >= 0 else len(levels_b) - 1 - level_a
        contours_a = by_level_a.get(level_a, [])
        contours_b = by_level_b.get(level_b, [])
        keep_count = min(len(contours_a), len(contours_b))
        keep_a.update(_longest_contour_indices(part_a.contour_lines, contours_a, keep_count))
        keep_b.update(_longest_contour_indices(part_b.contour_lines, contours_b, keep_count))
    return keep_a, keep_b


def _contours_by_nearest_level(part: CoilPart) -> dict[int, list[int]]:
    levels = np.asarray(part.potential_level_list, dtype=float)
    grouped: dict[int, list[int]] = defaultdict(list)
    if levels.size == 0:
        return grouped
    finite_levels = np.isfinite(levels)
    if not np.any(finite_levels):
        return grouped
    finite_indices = np.flatnonzero(finite_levels)
    finite_values = levels[finite_indices]
    for contour_ind, contour in enumerate(part.contour_lines):
        potential = float(contour.potential)
        if not np.isfinite(potential):
            continue
        nearest = int(finite_indices[int(np.argmin(np.abs(finite_values - potential)))])
        grouped[nearest].append(contour_ind)
    return grouped


def _longest_contour_indices(contours: list[ContourLine], indices: list[int], keep_count: int) -> list[int]:
    if keep_count <= 0 or not indices:
        return []
    ordered = sorted(indices, key=lambda ind: contours[ind].length, reverse=True)
    return ordered[:keep_count]


def _record_symmetric_extraction_diagnostics(parts: list[CoilPart], config: CoilGenConfig) -> None:
    mode = _contour_symmetry_validation_mode(config)
    if mode == "off":
        return
    reports = []
    for pair in _generated_candidate_symmetric_part_pairs(parts, config):
        counts = [len(parts[part_ind].contour_lines) for part_ind in pair]
        if not counts:
            continue
        delta = max(counts) - min(counts)
        report = {
            "symmetry_pair": ",".join(str(part_ind + 1) for part_ind in pair),
            "symmetry_extracted_contour_checked": True,
            "symmetry_extracted_contour_ok": bool(delta == 0),
            "symmetry_pair_extracted_contour_counts": list(counts),
            "symmetry_pair_extracted_contour_count_delta": int(delta),
        }
        for part_ind in pair:
            parts[part_ind].contour_level_diagnostics.update(report)
        reports.append(report)
    if _single_mesh_symmetry_validation_requested(parts, config):
        report = _single_mesh_extraction_symmetry_report(parts[0], config)
        parts[0].contour_level_diagnostics["symmetry_validation_mode"] = mode
        parts[0].contour_level_diagnostics.update(report)
        reports.append(report)
    _raise_for_strict_symmetry_failures(reports, mode, stage="contour extraction")


def _single_mesh_symmetry_validation_requested(parts: list[CoilPart], config: CoilGenConfig) -> bool:
    source = str(getattr(config, "symmetry_reduction_source", "")).strip().lower().replace("-", "_")
    return len(parts) == 1 and source in {"single_mesh", "single", "mesh"}


def _configured_mirror_plane(config: CoilGenConfig) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(getattr(config, "symmetry_reduction_mirror_plane_point", (0.0, 0.0, 0.0)), dtype=float).reshape(
        3
    )
    normal = np.asarray(
        getattr(config, "symmetry_reduction_mirror_plane_normal", (1.0, 0.0, 0.0)), dtype=float
    ).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("symmetry_reduction_mirror_plane_normal must be finite and non-zero.")
    return point, normal / norm


def _single_mesh_level_symmetry_report(part: CoilPart, config: CoilGenConfig) -> dict[str, object]:
    report, match_indices, _point, _normal = _single_mesh_geometry_report(part, config)
    report.update(_single_mesh_stream_function_report(part, match_indices, config))
    report.update(_single_mesh_level_report(part, report, config))
    return report


def _single_mesh_extraction_symmetry_report(part: CoilPart, config: CoilGenConfig) -> dict[str, object]:
    report, match_indices, point, normal = _single_mesh_geometry_report(part, config)
    stream_report = _single_mesh_stream_function_report(part, match_indices, config)
    report.update(stream_report)
    report.update(_single_mesh_level_report(part, report, config))
    report.update(_single_mesh_contour_report(part, report, point, normal, config))
    return report


def _single_mesh_geometry_report(
    part: CoilPart,
    config: CoilGenConfig,
) -> tuple[dict[str, object], np.ndarray | None, np.ndarray, np.ndarray]:
    point, normal = _configured_mirror_plane(config)
    report: dict[str, object] = {
        "symmetry_pair": "single_mesh",
        "symmetry_pair_indices": [0],
        "symmetry_reduction_source": "single_mesh",
        "symmetry_geometry_checked": True,
        "symmetry_geometry_ok": False,
        "symmetry_warnings": [],
    }
    vertices = np.asarray(part.mesh.vertices, dtype=float)
    if len(vertices) == 0:
        report["symmetry_warnings"] = ["Single mesh has no vertices."]
        return report, None, point, normal
    reflected = _reflect_points(vertices, point, normal)
    match_indices = _nearest_vertex_indices(reflected, vertices)
    if match_indices is None:
        report["symmetry_warnings"] = ["Single-mesh vertex matching would be too expensive without scipy."]
        return report, None, point, normal
    match_indices = np.asarray(match_indices, dtype=np.int64)
    distances = np.linalg.norm(reflected - vertices[match_indices], axis=1)
    scale = _symmetry_geometry_scale(vertices)
    tolerance = _geometry_abs_tolerance(config, scale)
    max_distance = float(np.max(distances)) if distances.size else float("inf")
    rms_distance = float(np.sqrt(np.mean(distances * distances))) if distances.size else float("inf")
    unique_mapping = len(np.unique(match_indices)) == len(match_indices)
    involution_mapping = bool(np.all(match_indices[match_indices] == np.arange(len(match_indices), dtype=np.int64)))
    ok = bool(unique_mapping and involution_mapping and np.isfinite(max_distance) and max_distance <= tolerance)
    warnings: list[str] = []
    if not ok:
        warnings.append("Reflected single-mesh vertex cloud does not match itself within tolerance.")
    report.update(
        {
            "symmetry_geometry_ok": ok,
            "symmetry_geometry_scale_m": float(scale),
            "symmetry_geometry_tolerance_m": float(tolerance),
            "symmetry_geometry_rms_distance_m": rms_distance,
            "symmetry_geometry_max_distance_m": max_distance,
            "symmetry_geometry_unique_mapping": bool(unique_mapping),
            "symmetry_geometry_involution_mapping": bool(involution_mapping),
            "symmetry_mirror_plane_point_m": [float(value) for value in point],
            "symmetry_mirror_plane_normal": [float(value) for value in normal],
            "symmetry_warnings": warnings,
        }
    )
    return report, match_indices if ok else None, point, normal


def _single_mesh_stream_function_report(
    part: CoilPart,
    match_indices: np.ndarray | None,
    config: CoilGenConfig,
) -> dict[str, object]:
    report: dict[str, object] = {
        "symmetry_stream_function_checked": False,
        "symmetry_stream_function_ok": False,
    }
    if match_indices is None or part.stream_function is None:
        return report
    psi = np.asarray(part.stream_function, dtype=float)
    if len(psi) != len(match_indices):
        return report
    matched = psi[np.asarray(match_indices, dtype=np.int64)]
    plus = _stream_symmetry_residual(psi, matched, sign=1.0)
    minus = _stream_symmetry_residual(psi, matched, sign=-1.0)
    selected = plus if plus["rms"] <= minus["rms"] else minus
    scale = max(float(np.ptp(psi)), float(np.max(np.abs(psi))), 1e-30)
    normalized_rms = float(selected["rms"] / scale)
    normalized_max = float(selected["max"] / scale)
    tolerance = float(getattr(config, "contour_symmetry_stream_function_tolerance", 1e-5))
    return {
        "symmetry_stream_function_checked": True,
        "symmetry_stream_function_ok": bool(normalized_rms <= tolerance),
        "symmetry_stream_function_sign": int(selected["sign"]),
        "symmetry_stream_function_offset_A": float(selected["offset"]),
        "symmetry_stream_function_rms_A": float(selected["rms"]),
        "symmetry_stream_function_max_A": float(selected["max"]),
        "symmetry_stream_function_scale_A": float(scale),
        "symmetry_stream_function_normalized_rms": normalized_rms,
        "symmetry_stream_function_normalized_max": normalized_max,
        "symmetry_stream_function_tolerance": tolerance,
        "symmetry_stream_function_plus_normalized_rms": float(plus["rms"] / scale),
        "symmetry_stream_function_minus_normalized_rms": float(minus["rms"] / scale),
    }


def _single_mesh_level_report(part: CoilPart, report: dict[str, object], config: CoilGenConfig) -> dict[str, object]:
    levels = part.potential_level_list
    if levels is None:
        return {"symmetry_level_checked": False, "symmetry_level_ok": False}
    values = np.asarray(levels, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"symmetry_level_checked": True, "symmetry_level_ok": True, "symmetry_level_count": 0}
    sign = float(report.get("symmetry_stream_function_sign", 1))
    offset = float(report.get("symmetry_stream_function_offset_A", 0.0))
    expected = sign * finite + offset
    errors = np.asarray([float(np.min(np.abs(finite - value))) for value in expected], dtype=float)
    scale = max(float(np.ptp(finite)), float(np.max(np.abs(finite))), abs(float(part.contour_step or 0.0)), 1e-30)
    tolerance = _single_mesh_potential_tolerance(part, scale, config)
    max_error = float(np.max(errors)) if errors.size else 0.0
    return {
        "symmetry_level_checked": True,
        "symmetry_level_ok": bool(max_error <= tolerance),
        "symmetry_level_count": int(len(finite)),
        "symmetry_level_potential_tolerance_A": float(tolerance),
        "symmetry_level_potential_max_error_A": max_error,
        "symmetry_level_potential_rms_error_A": float(np.sqrt(np.mean(errors * errors))) if errors.size else 0.0,
    }


def _single_mesh_potential_tolerance(part: CoilPart, scale: float, config: CoilGenConfig) -> float:
    tolerance = float(getattr(config, "contour_symmetry_stream_function_tolerance", 1e-5)) * max(float(scale), 1.0)
    step = float(part.contour_step or 0.0)
    if np.isfinite(step) and step > 0.0:
        tolerance = max(tolerance, 0.25 * step)
    return max(tolerance, 1e-12)


def _single_mesh_contour_report(
    part: CoilPart,
    report: dict[str, object],
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, object]:
    contours = list(part.contour_lines or [])
    if not contours:
        return {
            "symmetry_extracted_contour_checked": True,
            "symmetry_extracted_contour_ok": True,
            "symmetry_single_mesh_contour_count": 0,
        }
    potentials = np.asarray([float(contour.potential) for contour in contours], dtype=float)
    multiplicities = np.asarray([int(contour.turn_multiplicity) for contour in contours], dtype=np.int64)
    centroids = np.asarray([np.mean(np.asarray(contour.v, dtype=float), axis=0) for contour in contours], dtype=float)
    contour_points = [np.asarray(contour.v, dtype=float) for contour in contours if len(contour.v)]
    scale = (
        _symmetry_geometry_scale(np.vstack(contour_points))
        if contour_points
        else _symmetry_geometry_scale(part.mesh.vertices)
    )
    geometry_tolerance = _geometry_abs_tolerance(config, scale)
    potential_scale = max(
        float(np.ptp(potentials)), float(np.max(np.abs(potentials))), abs(float(part.contour_step or 0.0)), 1e-30
    )
    potential_tolerance = _single_mesh_potential_tolerance(part, potential_scale, config)
    sign = float(report.get("symmetry_stream_function_sign", 1))
    offset = float(report.get("symmetry_stream_function_offset_A", 0.0))

    unmatched = 0
    potential_errors: list[float] = []
    geometry_max_distances: list[float] = []
    geometry_rms_distances: list[float] = []
    matched_indices: list[int] = []
    reflected_centroids = _reflect_points(centroids, plane_point, plane_normal)
    assignment_cost = np.full((len(contours), len(contours)), np.inf, dtype=float)
    for contour_ind, contour in enumerate(contours):
        expected_potential = sign * float(contour.potential) + offset
        candidate_errors = np.abs(potentials - expected_potential)
        eligible = np.flatnonzero(
            (candidate_errors <= potential_tolerance) & (multiplicities == int(contour.turn_multiplicity))
        )
        if eligible.size:
            centroid_distances = np.linalg.norm(centroids[eligible] - reflected_centroids[contour_ind][None, :], axis=1)
            assignment_cost[contour_ind, eligible] = candidate_errors[eligible] / max(
                potential_tolerance, 1.0e-30
            ) + centroid_distances / max(geometry_tolerance, 1.0e-30)
    assignments = _minimum_cost_unique_assignment(assignment_cost)
    for contour_ind, contour in enumerate(contours):
        expected_potential = sign * float(contour.potential) + offset
        candidate_errors = np.abs(potentials - expected_potential)
        matched_ind = assignments.get(contour_ind)
        if matched_ind is None:
            unmatched += 1
            potential_errors.append(float(np.min(candidate_errors)) if candidate_errors.size else float("inf"))
            continue
        matched_indices.append(matched_ind)
        potential_errors.append(float(candidate_errors[matched_ind]))
        reflected_points = _reflect_points(np.asarray(contour.v, dtype=float), plane_point, plane_normal)
        distance_stats = _bidirectional_polyline_distance_stats(
            reflected_points, np.asarray(contours[matched_ind].v, dtype=float)
        )
        geometry_max_distances.append(float(distance_stats["max"]))
        geometry_rms_distances.append(float(distance_stats["rms"]))

    potential_max_error = float(np.max(potential_errors)) if potential_errors else 0.0
    geometry_max = float(np.max(geometry_max_distances)) if geometry_max_distances else float("inf")
    geometry_rms = (
        float(np.sqrt(np.mean(np.square(geometry_rms_distances)))) if geometry_rms_distances else float("inf")
    )
    unique_matched_count = int(len(set(matched_indices)))
    matched_count = int(len(contours) - unmatched)
    ok = bool(
        unmatched == 0
        and unique_matched_count == matched_count
        and potential_max_error <= potential_tolerance
        and geometry_max <= geometry_tolerance
    )
    return {
        "symmetry_extracted_contour_checked": True,
        "symmetry_extracted_contour_ok": ok,
        "symmetry_single_mesh_contour_count": int(len(contours)),
        "symmetry_single_mesh_matched_contour_count": matched_count,
        "symmetry_single_mesh_unmatched_contour_count": int(unmatched),
        "symmetry_single_mesh_unique_matched_contour_count": unique_matched_count,
        "symmetry_single_mesh_turn_multiplicity_checked": True,
        "symmetry_single_mesh_contour_geometry_tolerance_m": float(geometry_tolerance),
        "symmetry_single_mesh_contour_geometry_max_distance_m": geometry_max,
        "symmetry_single_mesh_contour_geometry_rms_distance_m": geometry_rms,
        "symmetry_single_mesh_contour_potential_tolerance_A": float(potential_tolerance),
        "symmetry_single_mesh_contour_potential_max_error_A": potential_max_error,
        "symmetry_single_mesh_contour_potential_rms_error_A": float(np.sqrt(np.mean(np.square(potential_errors))))
        if potential_errors
        else 0.0,
    }


def _minimum_cost_unique_assignment(cost: np.ndarray) -> dict[int, int]:
    """Return a one-to-one finite assignment, using Hungarian matching when available."""

    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.size == 0:
        return {}
    finite = np.isfinite(cost)
    if not np.any(finite):
        return {}
    finite_max = float(np.max(cost[finite]))
    invalid_cost = max(1.0e6 * max(finite_max, 1.0), 1.0e12)
    working = np.where(finite, cost, invalid_cost)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, columns = linear_sum_assignment(working)
        return {int(row): int(column) for row, column in zip(rows, columns) if finite[int(row), int(column)]}
    except Exception:
        candidates = np.argwhere(finite)
        order = np.argsort(cost[finite])
        assigned_rows: set[int] = set()
        assigned_columns: set[int] = set()
        assignments: dict[int, int] = {}
        for candidate_ind in order:
            row, column = (int(value) for value in candidates[int(candidate_ind)])
            if row in assigned_rows or column in assigned_columns:
                continue
            assignments[row] = column
            assigned_rows.add(row)
            assigned_columns.add(column)
        return assignments


def _bidirectional_polyline_distance_stats(points_a: np.ndarray, points_b: np.ndarray) -> dict[str, float]:
    distances_ab = _points_to_polyline_distances(points_a, points_b)
    distances_ba = _points_to_polyline_distances(points_b, points_a)
    distances = np.concatenate((distances_ab, distances_ba))
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {"max": float("inf"), "rms": float("inf")}
    return {
        "max": float(np.max(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
    }


def _points_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    if len(polyline) < 2:
        return np.full(len(points), float("inf"), dtype=float)
    starts = polyline[:-1]
    ends = polyline[1:]
    segments = ends - starts
    denom = np.sum(segments * segments, axis=1)
    valid = denom > 0.0
    if not np.any(valid):
        return np.full(len(points), float("inf"), dtype=float)
    starts = starts[valid]
    segments = segments[valid]
    denom = denom[valid]
    out = np.empty(len(points), dtype=float)
    batch_size = 128
    for start_ind in range(0, len(points), batch_size):
        batch = points[start_ind : start_ind + batch_size]
        rel = batch[:, None, :] - starts[None, :, :]
        t = np.sum(rel * segments[None, :, :], axis=2) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
        distances = np.linalg.norm(batch[:, None, :] - closest, axis=2)
        out[start_ind : start_ind + len(batch)] = np.min(distances, axis=1)
    return out


def regularize_contour_levels(mesh: Mesh, stream_function: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Move contour levels away from mesh-vertex stream-function values.

    Intrinsic topology needs regular level sets.  A contour crossing exactly
    through a triangle vertex makes the virtual cut ambiguous, and a near-vertex
    crossing can create numerically degenerate contour segments.  The adjustment
    is intentionally tiny and bounded between neighbouring requested levels.
    """

    requested = np.asarray(levels, dtype=float).copy()
    if requested.size == 0:
        return requested
    sf = np.asarray(stream_function, dtype=float)
    if sf.ndim != 1 or len(sf) != mesh.n_vertices:
        return requested
    finite_vertices = np.isfinite(sf)
    if not np.any(finite_vertices):
        return requested
    finite_sf = sf[finite_vertices]
    sf_min = float(np.min(finite_sf))
    sf_max = float(np.max(finite_sf))
    if not sf_max > sf_min:
        return requested

    margins = _vertex_level_safety_margins(mesh, sf)
    sorted_vertex_values = np.sort(finite_sf)
    lower_bounds, upper_bounds = _requested_level_bounds(requested, sf_min, sf_max)
    adjusted = requested.copy()
    for level_ind, level in enumerate(requested):
        if not np.isfinite(level):
            continue
        lower = float(lower_bounds[level_ind])
        upper = float(upper_bounds[level_ind])
        if not lower < upper:
            continue
        candidate = float(np.clip(level, np.nextafter(lower, upper), np.nextafter(upper, lower)))
        if _is_regular_contour_level(candidate, sf, margins):
            adjusted[level_ind] = candidate
            continue
        replacement = _nearest_regular_level_candidate(
            candidate,
            lower,
            upper,
            sf,
            margins,
            sorted_vertex_values,
        )
        if replacement is not None:
            adjusted[level_ind] = replacement
    return adjusted


def _vertex_level_safety_margins(mesh: Mesh, stream_function: np.ndarray) -> np.ndarray:
    finite = np.isfinite(stream_function)
    finite_values = stream_function[finite]
    if len(finite_values):
        scale = max(float(np.ptp(finite_values)), float(np.max(np.abs(finite_values))), 1.0)
    else:
        scale = 1.0
    floor = 100.0 * np.finfo(float).eps * scale
    margins = np.full(mesh.n_vertices, floor, dtype=float)
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        return margins
    for face in mesh.faces:
        for node_a, node_b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            node_a = int(node_a)
            node_b = int(node_b)
            if (
                node_a < 0
                or node_b < 0
                or node_a >= len(stream_function)
                or node_b >= len(stream_function)
                or not np.isfinite(stream_function[node_a])
                or not np.isfinite(stream_function[node_b])
            ):
                continue
            span = abs(float(stream_function[node_b] - stream_function[node_a]))
            margin = max(2.5e-8 * span, floor)
            margins[node_a] = max(margins[node_a], margin)
            margins[node_b] = max(margins[node_b], margin)
    return margins


def _requested_level_bounds(levels: np.ndarray, sf_min: float, sf_max: float) -> tuple[np.ndarray, np.ndarray]:
    levels = np.asarray(levels, dtype=float)
    lower = np.full(len(levels), sf_min, dtype=float)
    upper = np.full(len(levels), sf_max, dtype=float)
    finite = np.isfinite(levels)
    if len(levels) > 1 and np.all(np.diff(levels[finite]) >= 0.0):
        for ind in range(len(levels)):
            if ind > 0 and np.isfinite(levels[ind - 1]) and np.isfinite(levels[ind]):
                lower[ind] = max(lower[ind], 0.5 * float(levels[ind - 1] + levels[ind]))
            if ind + 1 < len(levels) and np.isfinite(levels[ind]) and np.isfinite(levels[ind + 1]):
                upper[ind] = min(upper[ind], 0.5 * float(levels[ind] + levels[ind + 1]))
    return lower, upper


def _is_regular_contour_level(level: float, stream_function: np.ndarray, margins: np.ndarray) -> bool:
    finite = np.isfinite(stream_function)
    if not np.any(finite):
        return True
    return bool(not np.any(np.abs(stream_function[finite] - level) <= margins[finite]))


def _nearest_regular_level_candidate(
    level: float,
    lower: float,
    upper: float,
    stream_function: np.ndarray,
    margins: np.ndarray,
    sorted_vertex_values: np.ndarray,
) -> float | None:
    finite = np.isfinite(stream_function)
    scale = max(float(np.ptp(stream_function[finite])), float(np.max(np.abs(stream_function[finite]))), 1.0)
    near = finite & (np.abs(stream_function - level) <= margins)
    base_shift = max(float(np.max(margins[near])) if np.any(near) else 0.0, 100.0 * np.finfo(float).eps * scale)
    candidates = []
    for factor in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        shift = factor * base_shift
        candidates.extend((level - shift, level + shift))
    candidates = [
        float(candidate)
        for candidate in candidates
        if lower < candidate < upper and _is_regular_contour_level(float(candidate), stream_function, margins)
    ]
    if candidates:
        return min(candidates, key=lambda candidate: abs(candidate - level))

    inside = sorted_vertex_values[(sorted_vertex_values > lower) & (sorted_vertex_values < upper)]
    breaks = np.unique(np.concatenate((np.asarray([lower], dtype=float), inside, np.asarray([upper], dtype=float))))
    if len(breaks) < 2:
        return None
    interval_midpoints = 0.5 * (breaks[:-1] + breaks[1:])
    interval_midpoints = [
        float(candidate)
        for candidate in interval_midpoints
        if lower < candidate < upper and _is_regular_contour_level(float(candidate), stream_function, margins)
    ]
    if not interval_midpoints:
        return None
    return min(interval_midpoints, key=lambda candidate: abs(candidate - level))


def validate_and_correct_contour_currents(
    part: CoilPart,
    contours: list[ContourLine] | None = None,
    *,
    relative_tolerance: float = 1e-10,
) -> list[ContourLine]:
    """Orient contours so their tangent follows the stream-function surface current."""

    input_contours = part.contour_lines if contours is None else contours
    corrected: list[ContourLine] = []
    for contour in input_contours:
        alignment, scale = contour_current_alignment(part, contour)
        if scale > 0.0 and alignment < -relative_tolerance * scale:
            corrected.append(reverse_contour_line(contour))
        elif abs(float(contour.current_orientation) - 1.0) > 1e-15:
            corrected.append(_copy_contour_with_orientation(contour, 1.0))
        else:
            corrected.append(contour)
    if contours is None:
        part.contour_lines = corrected
    return corrected


def contour_current_alignment(part: CoilPart, contour: ContourLine) -> tuple[float, float]:
    """Return signed and absolute tangent-current alignment for one contour."""

    if contour.segment_faces is None or len(contour.v) < 2:
        return 0.0, 0.0
    current_density = _part_current_density(part)
    if current_density is None:
        return 0.0, 0.0
    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    segment_vectors = np.asarray(contour.v[1:] - contour.v[:-1], dtype=float)
    if len(segment_faces) != len(segment_vectors):
        return 0.0, 0.0
    valid = (segment_faces >= 0) & (segment_faces < len(current_density))
    if not np.any(valid):
        return 0.0, 0.0
    segment_vectors = segment_vectors[valid]
    currents = current_density[segment_faces[valid]]
    signed = float(np.sum(segment_vectors * currents))
    scale = float(np.sum(np.linalg.norm(segment_vectors, axis=1) * np.linalg.norm(currents, axis=1)))
    return signed, scale


def reverse_contour_line(contour: ContourLine) -> ContourLine:
    """Reverse a contour while keeping point and segment provenance consistent."""

    segment_barycentric = None
    if contour.segment_barycentric is not None:
        segment_barycentric = contour.segment_barycentric[::-1, ::-1, :].copy()
    return ContourLine(
        v=contour.v[::-1].copy(),
        potential=contour.potential,
        current_orientation=1.0,
        surface_faces=None if contour.surface_faces is None else contour.surface_faces[::-1].copy(),
        surface_barycentric=None if contour.surface_barycentric is None else contour.surface_barycentric[::-1].copy(),
        surface_edge_vertices=None
        if contour.surface_edge_vertices is None
        else contour.surface_edge_vertices[::-1].copy(),
        surface_edge_t=None if contour.surface_edge_t is None else contour.surface_edge_t[::-1].copy(),
        segment_faces=None if contour.segment_faces is None else contour.segment_faces[::-1].copy(),
        segment_barycentric=segment_barycentric,
        turn_multiplicity=contour.turn_multiplicity,
    )


def filter_loop_significance(parts: list[CoilPart], target_field: TargetField, config: CoilGenConfig) -> list[CoilPart]:
    pruning_mode = _normalise_contour_loop_pruning_mode(getattr(config, "contour_loop_pruning_mode", "off"))
    if config.min_loop_signifcance <= 0 and pruning_mode == "off":
        return parts
    worker_count = resolve_worker_count(config)
    reporter = ProgressReporter.from_config(config, label="Loop significance")
    task = reporter.task("contour fields and pruning review", sum(len(part.contour_lines) for part in parts) + 1)
    field_by_key: dict[tuple[int, int], np.ndarray] = {}
    significance_by_key: dict[tuple[int, int], float] = {}
    for part_ind, part in enumerate(parts, start=1):
        if not part.contour_lines:
            continue
        work_size = sum(max(len(contour.v) - 1, 0) for contour in part.contour_lines) * len(target_field.coords)
        if should_parallelize(work_size, config, worker_count=worker_count) and len(part.contour_lines) > 1:
            fields = parallel_map(
                lambda contour: (
                    biot_savart_wire_path(
                        contour.v,
                        target_field.coords,
                        chunk_size=int(config.biot_savart_chunk_size),
                        worker_count=1,
                        min_work_size=int(config.parallel_min_work_size),
                        max_chunk_memory_mb=float(config.parallel_max_chunk_memory_mb),
                    )
                    * float(contour.turn_multiplicity)
                ),
                part.contour_lines,
                worker_count=worker_count,
                progress_callback=lambda done, total: task.advance(),
            )
        else:
            fields = []
            for contour in part.contour_lines:
                with progress_scope(
                    100 * task.completed / task.total,
                    100 * (task.completed + 1) / task.total,
                    "Evaluating contour significance",
                    label=reporter.label,
                ):
                    fields.append(
                        float(contour.turn_multiplicity)
                        * biot_savart_wire_path(
                            contour.v,
                            target_field.coords,
                            chunk_size=int(config.biot_savart_chunk_size),
                            worker_count=worker_count,
                            min_work_size=int(config.parallel_min_work_size),
                            max_chunk_memory_mb=float(config.parallel_max_chunk_memory_mb),
                            progress=reporter,
                        )
                    )
                task.advance()
        stacked = np.stack(fields, axis=2)
        combined = stacked.sum(axis=2)
        scale = max(float(np.mean(np.abs(combined[:, 2]))), 1e-30)
        significance = np.asarray(
            [np.max(np.abs(stacked[:, 2, ind])) / scale * 100.0 for ind in range(stacked.shape[2])]
        )
        for contour_ind, field in enumerate(fields):
            key = (part_ind - 1, contour_ind)
            field_by_key[key] = np.asarray(field, dtype=float)
            significance_by_key[key] = float(significance[contour_ind])

    all_keys = [
        (part_ind, contour_ind) for part_ind, part in enumerate(parts) for contour_ind in range(len(part.contour_lines))
    ]
    if not all_keys:
        task.done()
        return parts
    reporter.report(None, "reviewing contour significance and pruning groups")
    groups, protected_keys, symmetry_complete = _contour_loop_pruning_groups(parts, config)
    keep_keys = set(all_keys)
    significance_removed: set[tuple[int, int]] = set()
    threshold = float(config.min_loop_signifcance)
    if threshold > 0.0:
        if bool(getattr(config, "contour_loop_pruning_preserve_symmetry", True)):
            for group in groups:
                group_keys = set(group)
                if group_keys & protected_keys:
                    continue
                group_significance = max(
                    (significance_by_key.get(key, float("inf")) for key in group),
                    default=float("inf"),
                )
                if group_significance < threshold:
                    significance_removed.update(group_keys)
        else:
            significance_removed.update(
                key for key in all_keys if significance_by_key.get(key, float("inf")) < threshold
            )
        keep_keys.difference_update(significance_removed)

    pruning_diagnostics: dict[str, Any] = {
        "contour_loop_pruning_mode": pruning_mode,
        "contour_loop_pruning_symmetry_preserved": bool(
            getattr(config, "contour_loop_pruning_preserve_symmetry", True)
        ),
        "contour_loop_pruning_symmetry_grouping_complete": bool(symmetry_complete),
        "contour_loop_pruning_raw_loop_count": int(len(all_keys)),
        "contour_loop_pruning_significance_removed_loop_count": int(len(significance_removed)),
        "contour_loop_pruning_significance_threshold_percent": threshold,
    }
    if pruning_mode == "residual_budget" and keep_keys:
        keep_keys, residual_diagnostics = _prune_contour_loops_to_residual_budget(
            parts,
            target_field,
            config,
            field_by_key,
            groups,
            protected_keys,
            keep_keys,
        )
        pruning_diagnostics.update(residual_diagnostics)

    for part_ind, part in enumerate(parts):
        original_contours = list(part.contour_lines)
        retained_indices = [
            contour_ind for contour_ind in range(len(original_contours)) if (part_ind, contour_ind) in keep_keys
        ]
        part.contour_lines = [original_contours[contour_ind] for contour_ind in retained_indices]
        part.loop_significance = np.asarray(
            [significance_by_key[(part_ind, contour_ind)] for contour_ind in retained_indices],
            dtype=float,
        )
        part.combined_loop_length = float(sum(contour.effective_turn_length for contour in part.contour_lines))
        part.contour_level_diagnostics.update(pruning_diagnostics)
        part.contour_level_diagnostics["contour_loop_pruning_retained_loop_count"] = int(len(part.contour_lines))
    if bool(getattr(config, "contour_loop_pruning_preserve_symmetry", True)):
        _record_symmetric_extraction_diagnostics(parts, config)
    task.done()
    return parts


def _normalise_contour_loop_pruning_mode(value: Any) -> str:
    key = str(value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"residual_budget", "field_residual_budget", "backward_elimination"}:
        return "residual_budget"
    return "off"


def _contour_loop_pruning_groups(
    parts: list[CoilPart],
    config: CoilGenConfig,
    *,
    preserve_symmetry: bool | None = None,
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]], bool]:
    """Return mirror-orbit groups so pruning never deletes only one partner."""

    all_keys = [
        (part_ind, contour_ind) for part_ind, part in enumerate(parts) for contour_ind in range(len(part.contour_lines))
    ]
    preserve = (
        bool(getattr(config, "contour_loop_pruning_preserve_symmetry", True))
        if preserve_symmetry is None
        else bool(preserve_symmetry)
    )
    if not preserve:
        return [[key] for key in all_keys], set(), True
    symmetry_mode = _contour_symmetry_validation_mode(config)
    if symmetry_mode == "off":
        return [[key] for key in all_keys], set(), True

    source = str(getattr(config, "symmetry_reduction_source", "")).strip().lower().replace("-", "_")
    if len(parts) == 1 and source in {"single_mesh", "single", "mesh"}:
        if symmetry_mode == "auto" and not bool(
            parts[0].contour_level_diagnostics.get("symmetry_extracted_contour_ok", False)
        ):
            return [[key] for key in all_keys], set(), True
        return _single_mesh_contour_pruning_groups(parts[0], config)

    groups: list[list[tuple[int, int]]] = []
    assigned: set[tuple[int, int]] = set()
    paired_keys: set[tuple[int, int]] = set()
    symmetric_pairs = _generated_symmetric_part_pairs(parts, config)
    if not symmetric_pairs:
        return [[key] for key in all_keys], set(), True
    for pair in symmetric_pairs:
        if len(pair) != 2:
            continue
        part_a, part_b = parts[pair[0]], parts[pair[1]]
        paired_keys.update(
            (part_ind, contour_ind) for part_ind in pair for contour_ind in range(len(parts[part_ind].contour_lines))
        )
        sign = int(part_a.contour_level_diagnostics.get("symmetry_stream_function_sign", 1))
        by_level_a = _contours_by_nearest_level(part_a)
        by_level_b = _contours_by_nearest_level(part_b)
        level_count = min(
            0 if part_a.potential_level_list is None else len(part_a.potential_level_list),
            0 if part_b.potential_level_list is None else len(part_b.potential_level_list),
        )
        for level_a in range(level_count):
            level_b = level_a if sign >= 0 else level_count - 1 - level_a
            indices_a = sorted(by_level_a.get(level_a, ()), key=lambda ind: part_a.contour_lines[ind].length)
            indices_b = sorted(by_level_b.get(level_b, ()), key=lambda ind: part_b.contour_lines[ind].length)
            for contour_a, contour_b in zip(indices_a, indices_b):
                if (
                    part_a.contour_lines[contour_a].turn_multiplicity
                    != part_b.contour_lines[contour_b].turn_multiplicity
                ):
                    continue
                group = [(pair[0], contour_a), (pair[1], contour_b)]
                groups.append(group)
                assigned.update(group)
    protected = paired_keys - assigned
    groups.extend([[key] for key in sorted(set(all_keys) - assigned)])
    return groups, protected, not protected


def _single_mesh_contour_pruning_groups(
    part: CoilPart,
    config: CoilGenConfig,
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]], bool]:
    contours = list(part.contour_lines)
    keys = [(0, contour_ind) for contour_ind in range(len(contours))]
    if not contours:
        return [], set(), True
    point, normal = _configured_mirror_plane(config)
    potentials = np.asarray([float(contour.potential) for contour in contours], dtype=float)
    multiplicities = np.asarray([int(contour.turn_multiplicity) for contour in contours], dtype=np.int64)
    centroids = np.asarray([np.mean(np.asarray(contour.v, dtype=float), axis=0) for contour in contours])
    reflected_centroids = _reflect_points(centroids, point, normal)
    contour_points = [np.asarray(contour.v, dtype=float) for contour in contours if len(contour.v)]
    scale = _symmetry_geometry_scale(np.vstack(contour_points)) if contour_points else 1.0
    geometry_tolerance = _geometry_abs_tolerance(config, scale)
    potential_scale = max(
        float(np.ptp(potentials)),
        float(np.max(np.abs(potentials))),
        abs(float(part.contour_step or 0.0)),
        1.0e-30,
    )
    potential_tolerance = _single_mesh_potential_tolerance(part, potential_scale, config)
    sign = float(part.contour_level_diagnostics.get("symmetry_stream_function_sign", 1))
    offset = float(part.contour_level_diagnostics.get("symmetry_stream_function_offset_A", 0.0))
    cost = np.full((len(contours), len(contours)), np.inf, dtype=float)
    for contour_ind, contour in enumerate(contours):
        expected = sign * float(contour.potential) + offset
        potential_errors = np.abs(potentials - expected)
        eligible = np.flatnonzero(
            (potential_errors <= potential_tolerance) & (multiplicities == int(contour.turn_multiplicity))
        )
        if eligible.size:
            centroid_distances = np.linalg.norm(
                centroids[eligible] - reflected_centroids[contour_ind][None, :],
                axis=1,
            )
            cost[contour_ind, eligible] = potential_errors[eligible] / max(
                potential_tolerance, 1.0e-30
            ) + centroid_distances / max(geometry_tolerance, 1.0e-30)
    assignments = _minimum_cost_unique_assignment(cost)
    valid_assignments: dict[int, int] = {}
    for contour_ind, matched_ind in assignments.items():
        reflected = _reflect_points(np.asarray(contours[contour_ind].v, dtype=float), point, normal)
        distance = _bidirectional_polyline_distance_stats(
            reflected,
            np.asarray(contours[matched_ind].v, dtype=float),
        )["max"]
        if np.isfinite(distance) and float(distance) <= geometry_tolerance:
            valid_assignments[int(contour_ind)] = int(matched_ind)

    parent = list(range(len(contours)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a: int, index_b: int) -> None:
        root_a, root_b = find(index_a), find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for contour_ind, matched_ind in valid_assignments.items():
        if valid_assignments.get(matched_ind) == contour_ind:
            union(contour_ind, matched_ind)

    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    protected: set[tuple[int, int]] = set()
    for contour_ind, key in enumerate(keys):
        grouped[find(contour_ind)].append(key)
        matched_ind = valid_assignments.get(contour_ind)
        if matched_ind is None or valid_assignments.get(matched_ind) != contour_ind:
            protected.add(key)
    groups = [grouped[root] for root in sorted(grouped)]
    return groups, protected, not protected


def _prune_contour_loops_to_residual_budget(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
    field_by_key: dict[tuple[int, int], np.ndarray],
    groups: list[list[tuple[int, int]]],
    protected_keys: set[tuple[int, int]],
    initial_keep_keys: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    active_groups = [
        [key for key in group if key in initial_keep_keys]
        for group in groups
        if any(key in initial_keep_keys for key in group)
    ]
    active_groups = [group for group in active_groups if group]
    keep_keys = set(initial_keep_keys)
    combined_field = sum(
        (field_by_key[key] for key in keep_keys),
        start=np.zeros((len(target_field.coords), 3), dtype=float),
    )
    metric_context = _contour_loop_metric_context(parts, target_field, config)
    baseline_metric, baseline_current = _contour_loop_residual_metric(
        combined_field,
        metric_context,
    )
    relative_tolerance = max(
        float(getattr(config, "contour_loop_pruning_residual_relative_tolerance", 0.0)),
        0.0,
    )
    absolute_tolerance = max(
        float(getattr(config, "contour_loop_pruning_residual_absolute_tolerance_T", 0.0)),
        0.0,
    )
    allowed_metric = baseline_metric * (1.0 + relative_tolerance) + absolute_tolerance
    maximum_removed = max(int(getattr(config, "contour_loop_pruning_max_removed_loops", 0)), 0)
    removed_keys: set[tuple[int, int]] = set()
    selected_metric = baseline_metric
    selected_current = baseline_current
    progress = ProgressReporter.from_config(config, label="Loop significance")

    while len(active_groups) > 1:
        progress.report(None, f"testing residual-budget removals for {len(active_groups)} active loop groups")
        candidates: list[tuple[float, int, float, np.ndarray]] = []
        for group_ind, group in enumerate(active_groups):
            if set(group) & protected_keys:
                continue
            if maximum_removed and len(removed_keys) + len(group) > maximum_removed:
                continue
            group_field = sum(
                (field_by_key[key] for key in group),
                start=np.zeros_like(combined_field),
            )
            candidate_field = combined_field - group_field
            candidate_metric, candidate_current = _contour_loop_residual_metric(
                candidate_field,
                metric_context,
            )
            candidates.append((candidate_metric, group_ind, candidate_current, candidate_field))
        if not candidates:
            break
        candidate_metric, group_ind, candidate_current, candidate_field = min(
            candidates,
            key=lambda item: (item[0], -len(active_groups[item[1]]), item[1]),
        )
        numerical_tolerance = 1.0e-12 * max(abs(allowed_metric), abs(baseline_metric), 1.0e-30)
        if candidate_metric > allowed_metric + numerical_tolerance:
            break
        group = active_groups.pop(group_ind)
        removed_keys.update(group)
        keep_keys.difference_update(group)
        combined_field = candidate_field
        selected_metric = candidate_metric
        selected_current = candidate_current

    return keep_keys, {
        "contour_loop_pruning_metric": metric_context["metric"],
        "contour_loop_pruning_baseline_metric_T": float(baseline_metric),
        "contour_loop_pruning_allowed_metric_T": float(allowed_metric),
        "contour_loop_pruning_selected_metric_T": float(selected_metric),
        "contour_loop_pruning_metric_change_T": float(selected_metric - baseline_metric),
        "contour_loop_pruning_relative_tolerance": float(relative_tolerance),
        "contour_loop_pruning_absolute_tolerance_T": float(absolute_tolerance),
        "contour_loop_pruning_baseline_current_A": float(baseline_current),
        "contour_loop_pruning_selected_current_A": float(selected_current),
        "contour_loop_pruning_removed_loop_count": int(len(removed_keys)),
        "contour_loop_pruning_retained_total_loop_count": int(len(keep_keys)),
        "contour_loop_pruning_removed_loop_indices": [
            [int(part_ind), int(contour_ind)] for part_ind, contour_ind in sorted(removed_keys)
        ],
    }


def _contour_loop_metric_context(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
) -> dict[str, Any]:
    component = str(getattr(config, "target_field_component", "Bz"))
    b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    target_values = field_component_values(
        target_field.b,
        component,
        b0_direction=b0_direction,
    ).reshape(-1)
    weights = component_sample_values(np.asarray(target_field.weights, dtype=float), component).reshape(-1)
    mask = component_sample_values(dsv_target_mask(target_field), component).astype(bool).reshape(-1)
    mask &= np.isfinite(target_values) & np.isfinite(weights) & (weights > 0.0)
    metric = str(getattr(config, "target_error_constraint_metric", "rms")).strip().lower().replace("-", "_")
    if metric in {"std", "standard_deviation", "homogeneity"}:
        metric = "std"
    elif metric in {"norm", "l2", "weighted_norm"}:
        metric = "norm"
    else:
        metric = "rms"
    return {
        "target_values": target_values,
        "weights": weights,
        "mask": mask,
        "metric": metric,
        "component": component,
        "b0_direction": b0_direction,
        "source": str(getattr(config, "field_evaluation_current_source", "stream_function")),
        "contour_step": float(parts[0].contour_step or 1.0) if parts else 1.0,
        "target_scale": float(getattr(config, "target_gradient_strength", 1.0)),
    }


def _contour_loop_residual_metric(
    field_per_amp: np.ndarray,
    context: dict[str, Any],
) -> tuple[float, float]:
    from .evaluation import _field_evaluation_current

    current = _field_evaluation_current(
        field_per_amp,
        context["target_values"],
        context["target_scale"],
        context["mask"],
        contour_step=context["contour_step"],
        source=context["source"],
        component=context["component"],
        b0_direction=context["b0_direction"],
        sample_weights=context["weights"],
    )
    predicted = field_component_values(
        field_per_amp,
        context["component"],
        b0_direction=context["b0_direction"],
    ).reshape(-1) * float(current)
    residual = predicted - context["target_values"]
    mask = context["mask"] & np.isfinite(residual)
    if not np.any(mask):
        return float("inf"), float(current)
    values = residual[mask]
    weights = context["weights"][mask]
    weight_sum = float(np.sum(weights))
    if context["metric"] == "std":
        mean = float(np.sum(weights * values) / weight_sum)
        metric = float(np.sqrt(np.sum(weights * np.square(values - mean)) / weight_sum))
    elif context["metric"] == "norm":
        metric = float(np.sqrt(np.sum(weights * np.square(values))))
    else:
        metric = float(np.sqrt(np.sum(weights * np.square(values)) / weight_sum))
    return metric, float(current)


def regularize_contour_paths(parts: list[CoilPart], config: CoilGenConfig) -> list[CoilPart]:
    """Apply optional parameterized B-spline smoothing to unconnected contours.

    Circular cylinders use their exact developed 2D coordinates; arbitrary
    surfaces use a harmonic 2D chart of the local mesh patch.  The periodic
    cubic B-spline is sampled at regular physical arc length and traced back
    through the chart triangles, keeping every output segment exactly on the
    original current-carrying surface.
    """

    mode = _normalise_contour_path_regularization_mode(getattr(config, "contour_path_regularization_mode", "off"))
    target_length = _safe_nonnegative_float(getattr(config, "contour_resample_target_segment_length", 0.0))
    smoothing_factor = float(getattr(config, "contour_smoothing_factor", 0.0))
    smoothing_factor = min(max(smoothing_factor, 0.0), 1.0)
    smoothing_scale_mode = str(getattr(config, "contour_smoothing_scale_mode", "legacy")).strip().lower()
    smoothing_mesh_gain = float(getattr(config, "contour_smoothing_mesh_gain", 0.1))
    smoothing_work_samples = max(
        int(getattr(config, "contour_smoothing_work_samples_per_mesh_length", 6)),
        4,
    )
    smoothing_output_samples = max(
        int(getattr(config, "contour_smoothing_output_samples_per_mesh_length", 8)),
        2,
    )
    max_displacement = _safe_nonnegative_float(getattr(config, "contour_smoothing_max_displacement", 0.0))
    patch_rings = max(int(getattr(config, "contour_smoothing_patch_rings", 4)), 1)
    mesh_coupled = smoothing_scale_mode.replace("-", "_") in {"mesh", "mesh_coupled", "mesh_scaled"}
    do_smoothing = mode == "parameterized_bspline" and (
        (mesh_coupled and 0.0 < smoothing_mesh_gain < 1.0) or (not mesh_coupled and smoothing_factor > 0.0)
    )
    do_resampling = mode == "resample" and target_length > 0.0
    resampling_deferred = False
    routing_active = not bool(getattr(config, "skip_postprocessing", False)) and not bool(
        getattr(config, "contour_only_sweep_mode", False)
    )
    if do_resampling and routing_active:
        # Intrinsic topology requires each contour to cut a face only once.
        # Resampling inside mesh faces is exact-on-surface but creates repeated
        # face cuts, so it must happen after routing/manufacturing, not here.
        do_resampling = False
        resampling_deferred = True
    if not do_smoothing and not do_resampling and not resampling_deferred:
        return parts

    reporter = ProgressReporter.from_config(config, label="Contour regularization")
    task = reporter.task(
        "contour processing steps",
        sum(
            len(part.contour_lines) * (int(do_smoothing) + int(do_resampling) + 1) + 1
            for part in parts
            if part.contour_lines
        ),
    )
    for part in parts:
        if not part.contour_lines:
            continue
        reporter.report(None, "checking original contour spacing")
        original_min_distance = _part_min_centerline_distance(part.contour_lines)
        candidate_contours = list(part.contour_lines)
        smoothing_applied = False
        smoothing_rejected = False
        smoothing_contour_count = 0
        smoothing_failed_count = 0
        smoothing_spacing_rejected_count = 0
        smoothing_displacements: list[float] = []
        smoothing_patch_face_counts: list[int] = []
        smoothing_segment_cv_before: list[float] = []
        smoothing_segment_cv_after: list[float] = []
        smoothing_parameterizations: list[str] = []
        smoothing_mesh_scales: list[float] = []
        smoothing_mesh_gains: list[float] = []
        smoothing_work_sample_counts: list[int] = []
        smoothing_output_target_lengths: list[float] = []
        smoothing_failure_reasons: dict[str, int] = {}
        proposed_min_distance = original_min_distance
        if do_smoothing:
            smoother = MeshContourBSplineSmoother(part.mesh, patch_rings=patch_rings)
            smoothing_results = []
            for contour in candidate_contours:
                smoothing_results.append(
                    smoother.smooth(
                        contour,
                        smoothing_factor=smoothing_factor,
                        target_segment_length_m=target_length,
                        max_displacement_m=max_displacement,
                        scale_mode=smoothing_scale_mode,
                        mesh_gain=smoothing_mesh_gain,
                        work_samples_per_mesh_length=smoothing_work_samples,
                        output_samples_per_mesh_length=smoothing_output_samples,
                    )
                )
                task.advance()
            smoothed = [result.contour for result in smoothing_results]
            reporter.report(None, "checking proposed contour spacing")
            proposed_min_distance = _part_min_centerline_distance(smoothed)
            proposed = [result.applied for result in smoothing_results]
            accepted = list(proposed)
            if any(proposed) and bool(getattr(config, "contour_smoothing_preserve_min_turn_distance", True)):
                smoothed, accepted = _accept_spacing_safe_smoothed_contours(
                    candidate_contours,
                    smoothed,
                    proposed,
                    [result.segment_length_cv_before - result.segment_length_cv_after for result in smoothing_results],
                    original_min_distance,
                    config,
                )
                smoothing_spacing_rejected_count = sum(
                    proposed_flag and not accepted_flag for proposed_flag, accepted_flag in zip(proposed, accepted)
                )
                smoothing_rejected = smoothing_spacing_rejected_count > 0
            smoothing_contour_count = sum(accepted)
            smoothing_failed_count = sum(not result.applied for result in smoothing_results)
            for result in smoothing_results:
                if not result.applied:
                    smoothing_failure_reasons[result.reason] = smoothing_failure_reasons.get(result.reason, 0) + 1
            smoothing_displacements = [
                result.max_displacement_m for result, accepted_flag in zip(smoothing_results, accepted) if accepted_flag
            ]
            smoothing_patch_face_counts = [
                result.patch_face_count for result, accepted_flag in zip(smoothing_results, accepted) if accepted_flag
            ]
            smoothing_segment_cv_before = [
                result.segment_length_cv_before
                for result, accepted_flag in zip(smoothing_results, accepted)
                if accepted_flag
            ]
            smoothing_segment_cv_after = [
                result.segment_length_cv_after
                for result, accepted_flag in zip(smoothing_results, accepted)
                if accepted_flag
            ]
            smoothing_parameterizations = [
                result.parameterization for result, accepted_flag in zip(smoothing_results, accepted) if accepted_flag
            ]
            smoothing_mesh_scales = [
                result.mesh_scale_m
                for result, accepted_flag in zip(smoothing_results, accepted)
                if accepted_flag and result.mesh_scale_m > 0.0
            ]
            smoothing_mesh_gains = [
                result.mesh_gain_applied for result, accepted_flag in zip(smoothing_results, accepted) if accepted_flag
            ]
            smoothing_work_sample_counts = [
                result.work_sample_count for result, accepted_flag in zip(smoothing_results, accepted) if accepted_flag
            ]
            smoothing_output_target_lengths = [
                result.output_target_segment_length_m
                for result, accepted_flag in zip(smoothing_results, accepted)
                if accepted_flag
            ]
            if smoothing_contour_count:
                candidate_contours = smoothed
                smoothing_applied = True
        if do_resampling:
            resampled_contours = []
            for contour in candidate_contours:
                resampled_contours.append(_resample_contour_on_surface(part.mesh, contour, target_length))
                task.advance()
            candidate_contours = resampled_contours
        cleaned_contours = []
        for contour in candidate_contours:
            cleaned = clean_contour_line(contour, config)
            if cleaned is not None and len(cleaned.v) >= config.min_point_loop_number:
                cleaned_contours.append(cleaned)
            task.advance()
        part.contour_lines = cleaned_contours
        part.combined_loop_length = float(sum(contour.length for contour in part.contour_lines))
        part.contour_level_diagnostics.update(
            {
                "contour_path_regularization_mode": mode,
                "contour_path_resampling_applied": bool(do_resampling),
                "contour_path_resampling_deferred_until_after_routing": bool(resampling_deferred),
                "contour_path_resample_target_segment_length_m": float(target_length),
                "contour_path_smoothing_applied": bool(smoothing_applied),
                "contour_path_smoothing_rejected": bool(smoothing_rejected),
                "contour_path_smoothing_scope": "unconnected_contours",
                "contour_path_smoothing_method": _contour_smoothing_method(
                    do_smoothing,
                    smoothing_parameterizations,
                ),
                "contour_path_smoothing_parameterizations": tuple(sorted(set(smoothing_parameterizations))),
                "contour_path_smoothing_factor": float(smoothing_factor),
                "contour_path_smoothing_scale_mode": ("mesh_coupled" if mesh_coupled else "legacy"),
                "contour_path_smoothing_mesh_gain_requested": float(smoothing_mesh_gain),
                "contour_path_smoothing_mesh_scale_m": float(
                    np.mean(smoothing_mesh_scales) if smoothing_mesh_scales else 0.0
                ),
                "contour_path_smoothing_mesh_gain_applied": float(
                    np.mean(smoothing_mesh_gains) if smoothing_mesh_gains else 1.0
                ),
                "contour_path_smoothing_work_sample_count_mean": float(
                    np.mean(smoothing_work_sample_counts) if smoothing_work_sample_counts else 0.0
                ),
                "contour_path_smoothing_output_target_segment_length_m": float(
                    np.mean(smoothing_output_target_lengths) if smoothing_output_target_lengths else target_length
                ),
                "contour_path_smoothing_patch_rings": int(patch_rings),
                "contour_path_smoothing_max_displacement_m": float(max_displacement),
                "contour_path_smoothing_displacement_cap_enabled": bool(max_displacement > 0.0),
                "contour_path_smoothing_contour_count": int(smoothing_contour_count if smoothing_applied else 0),
                "contour_path_smoothing_failed_count": int(smoothing_failed_count),
                "contour_path_smoothing_failure_reasons": dict(sorted(smoothing_failure_reasons.items())),
                "contour_path_smoothing_spacing_rejected_count": int(smoothing_spacing_rejected_count),
                "contour_path_smoothing_min_distance_before_m": float(original_min_distance),
                "contour_path_smoothing_min_distance_proposed_m": float(proposed_min_distance),
                "contour_path_smoothing_min_distance_after_m": float(_part_min_centerline_distance(candidate_contours)),
                "contour_path_smoothing_max_applied_displacement_m": float(max(smoothing_displacements, default=0.0)),
                "contour_path_smoothing_mean_patch_face_count": float(
                    np.mean(smoothing_patch_face_counts) if smoothing_patch_face_counts else 0.0
                ),
                "contour_path_smoothing_segment_length_cv_before": float(
                    np.mean(smoothing_segment_cv_before) if smoothing_segment_cv_before else 0.0
                ),
                "contour_path_smoothing_segment_length_cv_after": float(
                    np.mean(smoothing_segment_cv_after) if smoothing_segment_cv_after else 0.0
                ),
                "contour_path_count_after_regularization": int(len(part.contour_lines)),
            }
        )
        task.advance()
    task.done()
    return parts


def _contour_smoothing_method(enabled: bool, parameterizations: list[str]) -> str:
    if not enabled:
        return "off"
    selected = set(parameterizations)
    if selected == {"developed_circular_cylinder"}:
        return "developed_cylinder_periodic_bspline"
    if selected <= {"harmonic_local_patch"}:
        return "harmonic_parameterized_periodic_bspline"
    return "mixed_surface_parameterized_periodic_bspline"


def _normalise_contour_path_regularization_mode(value: object) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"none", "false", "0"}:
        return "off"
    if key in {"resample", "sample", "sampling"}:
        return "resample"
    if key in {
        "parameterized_bspline",
        "parameterised_bspline",
        "bspline",
        "smooth",
        "smoothing",
        "resample_smooth",
        "smooth_resample",
        "resample_and_smooth",
        "smooth_and_resample",
    }:
        return "parameterized_bspline"
    return "off"


def _safe_nonnegative_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number) or number < 0.0:
        return 0.0
    return number


def _resample_contour_on_surface(mesh: Mesh, contour: ContourLine, target_length: float) -> ContourLine:
    if target_length <= 0.0 or len(contour.v) < 2:
        return contour
    if contour.segment_faces is None or contour.segment_barycentric is None:
        return contour
    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    segment_barycentric = np.asarray(contour.segment_barycentric, dtype=float)
    if len(segment_faces) != len(contour.v) - 1 or segment_barycentric.shape != (len(contour.v) - 1, 2, 3):
        return contour

    vertices: list[np.ndarray] = [np.asarray(contour.v[0], dtype=float).copy()]
    surface_faces: list[int] = [int(segment_faces[0])]
    surface_barycentric: list[np.ndarray] = [segment_barycentric[0, 0].copy()]
    new_segment_faces: list[int] = []
    new_segment_barycentric: list[np.ndarray] = []

    for segment_ind, face_ind_raw in enumerate(segment_faces):
        face_ind = int(face_ind_raw)
        start_bary = segment_barycentric[segment_ind, 0]
        end_bary = segment_barycentric[segment_ind, 1]
        start = np.asarray(contour.v[segment_ind], dtype=float)
        end = np.asarray(contour.v[segment_ind + 1], dtype=float)
        length = float(np.linalg.norm(end - start))
        split_count = max(int(np.ceil(length / target_length)), 1)
        previous_bary = start_bary.copy()
        for split_ind in range(1, split_count + 1):
            t = split_ind / split_count
            bary = (1.0 - t) * start_bary + t * end_bary
            point = _point_from_face_barycentric(mesh, face_ind, bary)
            new_segment_faces.append(face_ind)
            new_segment_barycentric.append(np.vstack((previous_bary, bary)))
            vertices.append(point)
            surface_faces.append(face_ind)
            surface_barycentric.append(bary.copy())
            previous_bary = bary.copy()

    return ContourLine(
        v=np.asarray(vertices, dtype=float),
        potential=contour.potential,
        current_orientation=contour.current_orientation,
        surface_faces=np.asarray(surface_faces, dtype=np.int64),
        surface_barycentric=np.asarray(surface_barycentric, dtype=float),
        surface_edge_vertices=None,
        surface_edge_t=None,
        segment_faces=np.asarray(new_segment_faces, dtype=np.int64),
        segment_barycentric=np.asarray(new_segment_barycentric, dtype=float),
        turn_multiplicity=contour.turn_multiplicity,
    )


def contour_evaluation_points(mesh: Mesh, contour: ContourLine, config: CoilGenConfig) -> np.ndarray:
    """Return the stored contour polyline used for field and routing evaluation."""

    return np.asarray(contour.v, dtype=float)


def _accept_spacing_safe_smoothed_contours(
    original_contours: list[ContourLine],
    proposed_contours: list[ContourLine],
    proposed: list[bool],
    regularity_improvements: list[float],
    original_min_distance: float,
    config: CoilGenConfig,
) -> tuple[list[ContourLine], list[bool]]:
    """Accept useful contours independently instead of rejecting the full set."""

    accepted = list(proposed)
    accepted_contours = [
        proposed if is_accepted else original
        for original, proposed, is_accepted in zip(original_contours, proposed_contours, accepted)
    ]
    threshold = float(getattr(config, "min_turn_distance", 0.0))
    required_distance = (
        threshold
        if threshold > 0.0 and np.isfinite(original_min_distance) and original_min_distance >= threshold
        else original_min_distance
    )
    if not np.isfinite(required_distance):
        return accepted_contours, accepted

    distance_tolerance = max(1.0e-9, 1.0e-4 * required_distance)
    while True:
        revert: set[int] = set()
        for first in range(len(accepted_contours)):
            for second in range(first + 1, len(accepted_contours)):
                if (
                    _contour_pair_min_distance(accepted_contours[first].v, accepted_contours[second].v)
                    + distance_tolerance
                    >= required_distance
                ):
                    continue
                if accepted[first] and accepted[second]:
                    revert.add(first if regularity_improvements[first] <= regularity_improvements[second] else second)
                elif accepted[first]:
                    revert.add(first)
                elif accepted[second]:
                    revert.add(second)
        if not revert:
            break
        for index in revert:
            accepted[index] = False
            accepted_contours[index] = original_contours[index]
    return accepted_contours, accepted


def _point_from_face_barycentric(mesh: Mesh, face_ind: int, barycentric: np.ndarray) -> np.ndarray:
    face = mesh.faces[int(face_ind)]
    return np.asarray(barycentric, dtype=float) @ mesh.vertices[face]


def _part_min_centerline_distance(contours: list[ContourLine]) -> float:
    if len(contours) < 2:
        return float("inf")
    minimum = float("inf")
    for ind_a in range(len(contours)):
        for ind_b in range(ind_a + 1, len(contours)):
            distance = _contour_pair_min_distance(contours[ind_a].v, contours[ind_b].v)
            minimum = min(minimum, distance)
    return minimum


def _contour_pair_min_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    pts_a = np.asarray(points_a, dtype=float)
    pts_b = np.asarray(points_b, dtype=float)
    if len(pts_a) < 2 or len(pts_b) < 2:
        return float("inf")
    starts_a = pts_a[:-1]
    directions_a = np.diff(pts_a, axis=0)
    starts_b = pts_b[:-1]
    directions_b = np.diff(pts_b, axis=0)
    if len(starts_a) == 0 or len(starts_b) == 0:
        return float("inf")

    minimum = float("inf")
    chunk_size = max(1, int(250_000 // len(starts_b)))
    c = np.einsum("ij,ij->i", directions_b, directions_b)[None, :]
    eps = 1e-30
    c_safe = np.where(c > eps, c, 1.0)
    for start in range(0, len(starts_a), chunk_size):
        p0 = starts_a[start : start + chunk_size]
        u = directions_a[start : start + chunk_size]
        w = p0[:, None, :] - starts_b[None, :, :]
        a = np.einsum("ij,ij->i", u, u)[:, None]
        a_safe = np.where(a > eps, a, 1.0)
        b = np.einsum("ik,jk->ij", u, directions_b)
        d = np.einsum("ik,ijk->ij", u, w)
        e = np.einsum("jk,ijk->ij", directions_b, w)
        denom = a_safe * c_safe - b * b
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(denom) > eps, (b * e - c_safe * d) / denom, 0.0)
        s = np.clip(s, 0.0, 1.0)
        t = (b * s + e) / c_safe
        t_negative = t < 0.0
        t = np.where(t_negative, 0.0, t)
        s = np.where(t_negative, np.clip(-d / a_safe, 0.0, 1.0), s)
        t_positive = t > 1.0
        t = np.where(t_positive, 1.0, t)
        s = np.where(t_positive, np.clip((b - d) / a_safe, 0.0, 1.0), s)
        closest_delta = w + s[:, :, None] * u[:, None, :] - t[:, :, None] * directions_b[None, :, :]
        distances = np.linalg.norm(closest_delta, axis=2)

        point_point = (a <= eps) & (c <= eps)
        if np.any(point_point):
            distances = np.where(point_point, np.linalg.norm(w, axis=2), distances)
        point_segment_a = (a <= eps) & (c > eps)
        if np.any(point_segment_a):
            point_t = np.clip(e / c_safe, 0.0, 1.0)
            point_delta = w - point_t[:, :, None] * directions_b[None, :, :]
            distances = np.where(point_segment_a, np.linalg.norm(point_delta, axis=2), distances)
        point_segment_b = (a > eps) & (c <= eps)
        if np.any(point_segment_b):
            point_s = np.clip(-d / a_safe, 0.0, 1.0)
            point_delta = w + point_s[:, :, None] * u[:, None, :]
            distances = np.where(point_segment_b, np.linalg.norm(point_delta, axis=2), distances)
        minimum = min(minimum, float(np.min(distances)))
    return minimum


def _segments_for_level(mesh: Mesh, stream_function: np.ndarray, level: float) -> list[SurfaceContourSegment]:
    segments: list[SurfaceContourSegment] = []
    for face_ind, face in enumerate(mesh.faces):
        values = stream_function[face]
        if not (np.min(values) < level < np.max(values)):
            continue
        xyz_points: list[np.ndarray] = []
        bary_points: list[np.ndarray] = []
        endpoint_keys: list[tuple[int, int, int]] = []
        for edge_a, edge_b in ((0, 1), (1, 2), (2, 0)):
            node_a = face[edge_a]
            node_b = face[edge_b]
            value_a = stream_function[node_a]
            value_b = stream_function[node_b]
            span = value_b - value_a
            if abs(span) < 1e-15:
                continue
            t = (level - value_a) / span
            if 0.0 < t < 1.0:
                xyz_points.append(mesh.vertices[node_a] + t * (mesh.vertices[node_b] - mesh.vertices[node_a]))
                bary = np.zeros(3, dtype=float)
                bary[edge_a] = 1.0 - t
                bary[edge_b] = t
                bary_points.append(bary)
                endpoint_keys.append(_edge_endpoint_key(int(node_a), int(node_b), float(t)))
        if len(xyz_points) == 2:
            segments.append(
                SurfaceContourSegment(
                    points=np.asarray(xyz_points, dtype=float),
                    barycentric=np.asarray(bary_points, dtype=float),
                    face_index=int(face_ind),
                    endpoint_keys=(endpoint_keys[0], endpoint_keys[1]),
                )
            )
    return segments


def _edge_endpoint_key(node_a: int, node_b: int, t_from_a_to_b: float) -> tuple[int, int, int]:
    low = min(node_a, node_b)
    high = max(node_a, node_b)
    t_sorted = t_from_a_to_b if node_a == low else 1.0 - t_from_a_to_b
    return low, high, int(round(t_sorted / 1e-10))


def _empty_stitch_diagnostics() -> dict[str, float | int]:
    return {
        "contour_segment_count": 0,
        "contour_stitch_endpoint_count": 0,
        "contour_stitch_degree1_endpoint_count": 0,
        "contour_stitch_degree_gt2_endpoint_count": 0,
        "contour_stitch_max_endpoint_degree": 0,
        "contour_closed_path_count_before_cleaning": 0,
        "contour_closed_segment_count_before_cleaning": 0,
        "contour_closed_path_total_length_m_before_cleaning": 0.0,
        "contour_open_path_count_discarded": 0,
        "contour_open_segment_count_discarded": 0,
        "contour_open_path_total_length_m_discarded": 0.0,
        "contour_open_path_max_length_m_discarded": 0.0,
        "contour_open_endpoint_gap_max_m": 0.0,
        "_contour_open_endpoint_gap_square_sum_m2": 0.0,
    }


def _accumulate_stitch_diagnostics(
    total: dict[str, float | int],
    update: dict[str, float | int],
) -> None:
    sum_keys = (
        "contour_segment_count",
        "contour_stitch_endpoint_count",
        "contour_stitch_degree1_endpoint_count",
        "contour_stitch_degree_gt2_endpoint_count",
        "contour_closed_path_count_before_cleaning",
        "contour_closed_segment_count_before_cleaning",
        "contour_closed_path_total_length_m_before_cleaning",
        "contour_open_path_count_discarded",
        "contour_open_segment_count_discarded",
        "contour_open_path_total_length_m_discarded",
        "_contour_open_endpoint_gap_square_sum_m2",
    )
    max_keys = (
        "contour_stitch_max_endpoint_degree",
        "contour_open_path_max_length_m_discarded",
        "contour_open_endpoint_gap_max_m",
    )
    for key in sum_keys:
        total[key] = total.get(key, 0) + update.get(key, 0)
    for key in max_keys:
        total[key] = max(float(total.get(key, 0.0)), float(update.get(key, 0.0)))


def _finalize_stitch_diagnostics(diagnostics: dict[str, float | int]) -> dict[str, float | int]:
    report = dict(diagnostics)
    open_path_count = int(report.get("contour_open_path_count_discarded", 0) or 0)
    open_gap_square_sum = float(report.pop("_contour_open_endpoint_gap_square_sum_m2", 0.0) or 0.0)
    open_length = float(report.get("contour_open_path_total_length_m_discarded", 0.0) or 0.0)
    closed_length = float(report.get("contour_closed_path_total_length_m_before_cleaning", 0.0) or 0.0)
    report["contour_open_endpoint_gap_rms_m"] = (
        float(np.sqrt(open_gap_square_sum / open_path_count)) if open_path_count > 0 else 0.0
    )
    report["contour_discarded_open_path_length_fraction"] = (
        float(open_length / (open_length + closed_length)) if (open_length + closed_length) > 0.0 else 0.0
    )
    return report


def _polyline_length(points: list[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _stitch_segments_with_diagnostics(
    segments: list[SurfaceContourSegment],
    level: float,
) -> tuple[list[ContourLine], dict[str, float | int]]:
    diagnostics = _empty_stitch_diagnostics()
    diagnostics["contour_segment_count"] = int(len(segments))
    if not segments:
        return [], diagnostics

    endpoint_to_segments: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for seg_ind, segment in enumerate(segments):
        endpoint_to_segments[segment.endpoint_keys[0]].append(seg_ind)
        endpoint_to_segments[segment.endpoint_keys[1]].append(seg_ind)

    endpoint_degrees = [len(indices) for indices in endpoint_to_segments.values()]
    diagnostics["contour_stitch_endpoint_count"] = int(len(endpoint_degrees))
    diagnostics["contour_stitch_degree1_endpoint_count"] = int(sum(1 for degree in endpoint_degrees if degree == 1))
    diagnostics["contour_stitch_degree_gt2_endpoint_count"] = int(sum(1 for degree in endpoint_degrees if degree > 2))
    diagnostics["contour_stitch_max_endpoint_degree"] = int(max(endpoint_degrees, default=0))

    used = np.zeros(len(segments), dtype=bool)
    contours: list[ContourLine] = []
    for start_ind in range(len(segments)):
        if used[start_ind]:
            continue
        used[start_ind] = True
        start_segment = segments[start_ind]
        xyz_path = [start_segment.points[0], start_segment.points[1]]
        bary_path = [start_segment.barycentric[0], start_segment.barycentric[1]]
        point_faces = [start_segment.face_index, start_segment.face_index]
        endpoint_keys = [start_segment.endpoint_keys[0], start_segment.endpoint_keys[1]]
        edge_vertices = [
            _edge_vertices_from_key(start_segment.endpoint_keys[0]),
            _edge_vertices_from_key(start_segment.endpoint_keys[1]),
        ]
        edge_t = [_edge_t_from_key(start_segment.endpoint_keys[0]), _edge_t_from_key(start_segment.endpoint_keys[1])]
        segment_faces = [start_segment.face_index]
        segment_barycentric = [start_segment.barycentric.copy()]

        _extend_path(
            xyz_path,
            bary_path,
            point_faces,
            endpoint_keys,
            edge_vertices,
            edge_t,
            segment_faces,
            segment_barycentric,
            used,
            segments,
            endpoint_to_segments,
            forward=True,
        )
        _extend_path(
            xyz_path,
            bary_path,
            point_faces,
            endpoint_keys,
            edge_vertices,
            edge_t,
            segment_faces,
            segment_barycentric,
            used,
            segments,
            endpoint_to_segments,
            forward=False,
        )
        path_segment_count = int(len(segment_faces))
        is_closed = len(xyz_path) >= 4 and endpoint_keys[0] == endpoint_keys[-1]
        if is_closed:
            xyz_path[-1] = xyz_path[0]
            bary_path[-1] = bary_path[0]
            point_faces[-1] = point_faces[0]
            edge_vertices[-1] = edge_vertices[0]
            edge_t[-1] = edge_t[0]
            diagnostics["contour_closed_path_count_before_cleaning"] = (
                int(diagnostics["contour_closed_path_count_before_cleaning"]) + 1
            )
            diagnostics["contour_closed_segment_count_before_cleaning"] = (
                int(diagnostics["contour_closed_segment_count_before_cleaning"]) + path_segment_count
            )
            diagnostics["contour_closed_path_total_length_m_before_cleaning"] = float(
                diagnostics["contour_closed_path_total_length_m_before_cleaning"]
            ) + _polyline_length(xyz_path)
            contours.append(
                ContourLine(
                    v=np.asarray(xyz_path, dtype=float),
                    potential=level,
                    current_orientation=1.0,
                    surface_faces=np.asarray(point_faces, dtype=np.int64),
                    surface_barycentric=np.asarray(bary_path, dtype=float),
                    surface_edge_vertices=np.asarray(edge_vertices, dtype=np.int64),
                    surface_edge_t=np.asarray(edge_t, dtype=float),
                    segment_faces=np.asarray(segment_faces, dtype=np.int64),
                    segment_barycentric=np.asarray(segment_barycentric, dtype=float),
                )
            )
        else:
            path_length = _polyline_length(xyz_path)
            endpoint_gap = (
                float(np.linalg.norm(np.asarray(xyz_path[0]) - np.asarray(xyz_path[-1]))) if xyz_path else 0.0
            )
            diagnostics["contour_open_path_count_discarded"] = int(diagnostics["contour_open_path_count_discarded"]) + 1
            diagnostics["contour_open_segment_count_discarded"] = (
                int(diagnostics["contour_open_segment_count_discarded"]) + path_segment_count
            )
            diagnostics["contour_open_path_total_length_m_discarded"] = (
                float(diagnostics["contour_open_path_total_length_m_discarded"]) + path_length
            )
            diagnostics["contour_open_path_max_length_m_discarded"] = max(
                float(diagnostics["contour_open_path_max_length_m_discarded"]),
                path_length,
            )
            diagnostics["contour_open_endpoint_gap_max_m"] = max(
                float(diagnostics["contour_open_endpoint_gap_max_m"]),
                endpoint_gap,
            )
            diagnostics["_contour_open_endpoint_gap_square_sum_m2"] = (
                float(diagnostics["_contour_open_endpoint_gap_square_sum_m2"]) + endpoint_gap * endpoint_gap
            )
    return contours, diagnostics


def _extend_path(
    xyz_path: list[np.ndarray],
    bary_path: list[np.ndarray],
    point_faces: list[int],
    endpoint_keys: list[tuple[int, int, int]],
    edge_vertices: list[tuple[int, int]],
    edge_t: list[float],
    segment_faces: list[int],
    segment_barycentric: list[np.ndarray],
    used: np.ndarray,
    segments: list[SurfaceContourSegment],
    endpoint_to_segments: dict[tuple[int, int, int], list[int]],
    forward: bool,
) -> None:
    while True:
        endpoint_key = endpoint_keys[-1] if forward else endpoint_keys[0]
        candidates = endpoint_to_segments.get(endpoint_key, [])
        next_ind = next((candidate for candidate in candidates if not used[candidate]), None)
        if next_ind is None:
            return
        used[next_ind] = True
        segment = segments[next_ind]
        if segment.endpoint_keys[0] == endpoint_key:
            shared_point_ind = 0
            next_point_ind = 1
        elif segment.endpoint_keys[1] == endpoint_key:
            shared_point_ind = 1
            next_point_ind = 0
        else:
            continue
        if forward:
            xyz_path.append(segment.points[next_point_ind])
            bary_path.append(segment.barycentric[next_point_ind])
            point_faces.append(segment.face_index)
            endpoint_keys.append(segment.endpoint_keys[next_point_ind])
            edge_vertices.append(_edge_vertices_from_key(segment.endpoint_keys[next_point_ind]))
            edge_t.append(_edge_t_from_key(segment.endpoint_keys[next_point_ind]))
            segment_faces.append(segment.face_index)
            segment_barycentric.append(segment.barycentric[[shared_point_ind, next_point_ind]].copy())
        else:
            xyz_path.insert(0, segment.points[next_point_ind])
            bary_path.insert(0, segment.barycentric[next_point_ind])
            point_faces.insert(0, segment.face_index)
            endpoint_keys.insert(0, segment.endpoint_keys[next_point_ind])
            edge_vertices.insert(0, _edge_vertices_from_key(segment.endpoint_keys[next_point_ind]))
            edge_t.insert(0, _edge_t_from_key(segment.endpoint_keys[next_point_ind]))
            segment_faces.insert(0, segment.face_index)
            segment_barycentric.insert(0, segment.barycentric[[next_point_ind, shared_point_ind]].copy())


def _edge_vertices_from_key(key: tuple[int, int, int]) -> tuple[int, int]:
    return int(key[0]), int(key[1])


def _edge_t_from_key(key: tuple[int, int, int]) -> float:
    return float(key[2]) * 1e-10


def clean_contour_line(contour: ContourLine, config: CoilGenConfig) -> ContourLine | None:
    """Clean duplicate, tiny, and extreme-angle vertices from a contour."""

    if contour.segment_faces is not None:
        return _clean_surface_contour_line(contour, config)
    v = np.asarray(contour.v, dtype=float)
    finite = np.all(np.isfinite(v), axis=1)
    v = v[finite]
    if len(v) < 4:
        return None
    if np.linalg.norm(v[0] - v[-1]) > 1e-10:
        v = np.vstack((v, v[0]))
    return ContourLine(
        v=v,
        potential=contour.potential,
        current_orientation=contour.current_orientation,
        turn_multiplicity=contour.turn_multiplicity,
    )


def _clean_surface_contour_line(contour: ContourLine, config: CoilGenConfig) -> ContourLine | None:
    v = np.asarray(contour.v, dtype=float)
    surface_faces = np.asarray(contour.surface_faces, dtype=np.int64) if contour.surface_faces is not None else None
    barycentric = (
        np.asarray(contour.surface_barycentric, dtype=float) if contour.surface_barycentric is not None else None
    )
    edge_vertices = (
        np.asarray(contour.surface_edge_vertices, dtype=np.int64) if contour.surface_edge_vertices is not None else None
    )
    edge_t = np.asarray(contour.surface_edge_t, dtype=float) if contour.surface_edge_t is not None else None
    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    segment_barycentric = (
        np.asarray(contour.segment_barycentric, dtype=float) if contour.segment_barycentric is not None else None
    )

    finite = np.all(np.isfinite(v), axis=1)
    if barycentric is not None:
        finite &= np.all(np.isfinite(barycentric), axis=1)
    if edge_t is not None:
        finite &= np.isfinite(edge_t)
    if not np.all(finite):
        # Dropping arbitrary interior points breaks contour-to-face provenance.
        return None
    if segment_barycentric is not None and not np.all(np.isfinite(segment_barycentric)):
        return None
    if len(v) < 4 or len(segment_faces) != len(v) - 1:
        return None
    if segment_barycentric is not None and segment_barycentric.shape != (len(v) - 1, 2, 3):
        return None
    if np.linalg.norm(v[0] - v[-1]) > 1e-8:
        return None

    v = v.copy()
    v[-1] = v[0]
    if barycentric is not None:
        barycentric = barycentric.copy()
        barycentric[-1] = barycentric[0]
    if surface_faces is not None:
        surface_faces = surface_faces.copy()
        surface_faces[-1] = surface_faces[0]
    if edge_vertices is not None:
        edge_vertices = edge_vertices.copy()
        edge_vertices[-1] = edge_vertices[0]
    if edge_t is not None:
        edge_t = edge_t.copy()
        edge_t[-1] = edge_t[0]

    cleaned = _remove_degenerate_surface_segments(
        v,
        surface_faces,
        barycentric,
        edge_vertices,
        edge_t,
        segment_faces,
        segment_barycentric,
    )
    if cleaned is None:
        return None
    (
        v,
        surface_faces,
        barycentric,
        edge_vertices,
        edge_t,
        segment_faces,
        segment_barycentric,
    ) = cleaned

    if len(v) < int(config.min_point_loop_number):
        return None
    return ContourLine(
        v=v,
        potential=contour.potential,
        current_orientation=contour.current_orientation,
        surface_faces=surface_faces,
        surface_barycentric=barycentric,
        surface_edge_vertices=edge_vertices,
        surface_edge_t=edge_t,
        segment_faces=segment_faces.copy(),
        segment_barycentric=None if segment_barycentric is None else segment_barycentric.copy(),
        turn_multiplicity=contour.turn_multiplicity,
    )


def _remove_degenerate_surface_segments(
    v: np.ndarray,
    surface_faces: np.ndarray | None,
    barycentric: np.ndarray | None,
    edge_vertices: np.ndarray | None,
    edge_t: np.ndarray | None,
    segment_faces: np.ndarray,
    segment_barycentric: np.ndarray | None,
) -> (
    tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        np.ndarray | None,
    ]
    | None
):
    v = np.asarray(v, dtype=float).copy()
    surface_faces = None if surface_faces is None else np.asarray(surface_faces, dtype=np.int64).copy()
    barycentric = None if barycentric is None else np.asarray(barycentric, dtype=float).copy()
    edge_vertices = None if edge_vertices is None else np.asarray(edge_vertices, dtype=np.int64).copy()
    edge_t = None if edge_t is None else np.asarray(edge_t, dtype=float).copy()
    segment_faces = np.asarray(segment_faces, dtype=np.int64).copy()
    segment_barycentric = None if segment_barycentric is None else np.asarray(segment_barycentric, dtype=float).copy()

    while len(v) >= 4:
        lengths = np.linalg.norm(np.diff(v, axis=0), axis=1)
        degenerate = np.where(lengths <= 1e-12)[0]
        if len(degenerate) == 0:
            break
        segment_ind = int(degenerate[0])
        point_ind = segment_ind + 1
        if point_ind == len(v) - 1:
            point_ind = len(v) - 2
        if point_ind <= 0:
            point_ind = 1

        keep_points = np.ones(len(v), dtype=bool)
        keep_points[point_ind] = False
        keep_segments = np.ones(len(v) - 1, dtype=bool)
        keep_segments[segment_ind] = False

        v = v[keep_points]
        surface_faces = None if surface_faces is None else surface_faces[keep_points]
        barycentric = None if barycentric is None else barycentric[keep_points]
        edge_vertices = None if edge_vertices is None else edge_vertices[keep_points]
        edge_t = None if edge_t is None else edge_t[keep_points]
        segment_faces = segment_faces[keep_segments]
        segment_barycentric = None if segment_barycentric is None else segment_barycentric[keep_segments]

        if len(v) < 4 or len(segment_faces) != len(v) - 1:
            return None
        v[-1] = v[0]
        if surface_faces is not None:
            surface_faces[-1] = surface_faces[0]
        if barycentric is not None:
            barycentric[-1] = barycentric[0]
        if edge_vertices is not None:
            edge_vertices[-1] = edge_vertices[0]
        if edge_t is not None:
            edge_t[-1] = edge_t[0]

    if len(v) < 4 or len(segment_faces) != len(v) - 1:
        return None
    if segment_barycentric is not None and segment_barycentric.shape != (len(v) - 1, 2, 3):
        return None
    return (
        v,
        surface_faces,
        barycentric,
        edge_vertices,
        edge_t,
        segment_faces,
        segment_barycentric,
    )


def build_simple_wire_paths(parts: list[CoilPart]) -> list[CoilPart]:
    for part in parts:
        if not part.contour_lines:
            part.wire_path = WirePath(v=np.zeros((0, 3)))
            continue
        remaining = list(part.contour_lines)
        first = remaining.pop(0)
        xyz_path = [*first.v]
        while remaining:
            current = xyz_path[-1]
            best_ind = 0
            best_point_ind = 0
            best_distance = float("inf")
            for idx, contour in enumerate(remaining):
                search_points = _unique_closed_points(contour.v)
                distances = np.linalg.norm(search_points - current, axis=1)
                point_ind = int(np.argmin(distances))
                distance = float(distances[point_ind])
                if distance < best_distance:
                    best_ind = idx
                    best_point_ind = point_ind
                    best_distance = distance
            contour = remaining.pop(best_ind)
            xyz = _rotate_closed_path_preserving_orientation(contour.v, best_point_ind)
            xyz_path.extend(xyz)
        part.wire_path = WirePath(v=np.asarray(xyz_path, dtype=float))
    return parts


def _part_current_density(part: CoilPart) -> np.ndarray | None:
    if part.current_density is not None:
        return np.asarray(part.current_density, dtype=float)
    if part.current_density_mat is None or part.stream_function is None:
        return current_density_from_basis(part)
    return np.tensordot(part.stream_function, part.current_density_mat, axes=(0, 0))


def _copy_contour_with_orientation(contour: ContourLine, current_orientation: float) -> ContourLine:
    return ContourLine(
        v=contour.v.copy(),
        potential=contour.potential,
        current_orientation=float(current_orientation),
        surface_faces=None if contour.surface_faces is None else contour.surface_faces.copy(),
        surface_barycentric=None if contour.surface_barycentric is None else contour.surface_barycentric.copy(),
        surface_edge_vertices=None if contour.surface_edge_vertices is None else contour.surface_edge_vertices.copy(),
        surface_edge_t=None if contour.surface_edge_t is None else contour.surface_edge_t.copy(),
        segment_faces=None if contour.segment_faces is None else contour.segment_faces.copy(),
        segment_barycentric=None if contour.segment_barycentric is None else contour.segment_barycentric.copy(),
        turn_multiplicity=contour.turn_multiplicity,
    )


def _unique_closed_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-10:
        return points[:-1]
    return points


def _rotate_closed_path_preserving_orientation(points: np.ndarray, point_ind: int) -> np.ndarray:
    unique = _unique_closed_points(points)
    if len(unique) == 0:
        return unique.copy()
    point_ind = int(point_ind) % len(unique)
    return np.vstack((unique[point_ind:], unique[: point_ind + 1]))
