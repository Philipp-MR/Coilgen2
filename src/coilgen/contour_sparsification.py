from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CoilGenConfig
from .contours import (
    _contour_loop_pruning_groups,
    _segments_for_level,
    _stitch_segments_with_diagnostics,
    clean_contour_line,
    contour_evaluation_points,
    regularize_contour_levels,
    validate_and_correct_contour_currents,
)
from .field_components import component_sample_values, field_component_values
from .layout import topological_loop_grouping
from .metrics import dsv_target_mask
from .models import (
    CoilPart,
    ContourLine,
    ContourMergeRecord,
    ContourSparsificationResult,
    Mesh,
    TargetField,
)
from .physics import biot_savart_wire_path
from .progress import ProgressReporter


@dataclass
class _MergeProposal:
    part_index: int
    pass_index: int
    source_indices: tuple[int, int]
    source_original_indices: tuple[int, ...]
    source_turn_multiplicities: tuple[int, int]
    candidate: ContourLine
    source_field: np.ndarray
    candidate_field: np.ndarray
    pair_rms_error: float
    pair_peak_error: float
    positioning_method: str
    accepted: bool = False
    rejection_reason: str = "field_error_budget"
    representative_contour_index: int = -1

    @property
    def field_delta(self) -> np.ndarray:
        return self.candidate_field - self.source_field


def sparsify_contours(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
    *,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Optionally replace adjacent contours by weighted representatives.

    The active ``part.contour_lines`` remain nominal electromagnetic contours.
    Iterative passes may create representatives with multiplicity greater than
    two; each contributes its integer multiplicity times its unit-current
    Biot-Savart field. Every accepted pass is checked against the original
    unsparsified contour field, so error budgets do not reset between passes.
    This stage deliberately does not invent separated physical tracks.
    """

    mode = _normalise_sparsification_mode(getattr(config, "contour_sparsification_mode", "off"))
    if mode == "off" or not parts:
        return parts

    source_snapshots = [deepcopy(part.contour_lines) for part in parts]
    source_members = [[(index,) for index in range(len(contours))] for contours in source_snapshots]
    original_level_indices = [
        _nearest_level_indices(part, contours) for part, contours in zip(parts, source_snapshots, strict=True)
    ]
    baseline_cache = _source_field_cache(parts, target_field, config)
    baseline_field = _sum_fields(baseline_cache.values(), len(target_field.coords))
    accepted_field = baseline_field.copy()
    all_proposals: list[_MergeProposal] = []
    max_merges = max(int(getattr(config, "contour_sparsification_max_merges", 0)), 0)
    max_passes = max(int(getattr(config, "contour_sparsification_max_passes", 1)), 1)
    accepted_count = 0
    accepted_merges_by_pass: list[int] = []
    completed_passes = 0
    for pass_index in range(1, max_passes + 1):
        if max_merges > 0 and accepted_count >= max_merges:
            break
        for part in parts:
            _clear_topology_state(part)
        topological_loop_grouping(parts, config=config, progress=progress)
        field_cache = _source_field_cache(parts, target_field, config)
        proposals = _build_merge_proposals(
            parts,
            target_field,
            config,
            field_cache,
            source_members,
            source_snapshots,
            original_level_indices,
            pass_index=pass_index,
        )
        if not proposals:
            break
        completed_passes = pass_index
        accepted_this_pass = 0
        for bundle, complete in _proposal_bundles(parts, proposals, config):
            if not complete:
                for proposal in bundle:
                    proposal.rejection_reason = "incomplete_symmetry_orbit"
                continue
            if max_merges > 0 and accepted_count + len(bundle) > max_merges:
                for proposal in bundle:
                    proposal.rejection_reason = "maximum_merge_count"
                continue
            candidate_field = accepted_field + _sum_fields(
                (proposal.field_delta for proposal in bundle),
                len(target_field.coords),
            )
            rms_percent, peak_percent = _relative_field_error_percent(
                candidate_field,
                baseline_field,
                target_field,
                config,
            )
            if not _within_error_budget(rms_percent, peak_percent, config):
                continue
            for proposal in bundle:
                proposal.accepted = True
                proposal.rejection_reason = ""
            accepted_field = candidate_field
            accepted_count += len(bundle)
            accepted_this_pass += len(bundle)
        all_proposals.extend(proposals)
        accepted_merges_by_pass.append(accepted_this_pass)
        if accepted_this_pass == 0:
            break
        source_members = _apply_pass_proposals(parts, source_members, proposals)

    final_rms_percent, final_peak_percent = _relative_field_error_percent(
        accepted_field,
        baseline_field,
        target_field,
        config,
    )
    _finalize_sparsification(
        parts,
        source_snapshots,
        source_members,
        all_proposals,
        mode=mode,
        final_rms_percent=final_rms_percent,
        final_peak_percent=final_peak_percent,
        requested_passes=max_passes,
        completed_passes=completed_passes,
        accepted_merges_by_pass=accepted_merges_by_pass,
        max_turn_multiplicity=int(getattr(config, "contour_sparsification_max_turn_multiplicity", 2)),
    )
    return parts


def _normalise_sparsification_mode(value: object) -> str:
    key = str(value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    return "pairwise" if key in {"pairwise", "pairs", "pair"} else "off"


def _build_merge_proposals(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
    field_cache: dict[tuple[int, int], np.ndarray],
    source_members: list[list[tuple[int, ...]]],
    source_snapshots: list[list[ContourLine]],
    original_level_indices: list[dict[int, int]],
    *,
    pass_index: int,
) -> list[_MergeProposal]:
    proposals: list[_MergeProposal] = []
    for part_index, part in enumerate(parts):
        used: set[int] = set()
        for group in part.groups:
            ordered = sorted(
                (int(index) for index in group.loop_indices),
                key=lambda index: (part.contour_lines[index].potential, index),
            )
            position = 0
            while position + 1 < len(ordered):
                first, second = ordered[position], ordered[position + 1]
                if (
                    first in used
                    or second in used
                    or not _eligible_pair(
                        part,
                        first,
                        second,
                        source_members[part_index],
                        original_level_indices[part_index],
                        config,
                    )
                ):
                    position += 1
                    continue
                proposal = _proposal_for_pair(
                    part,
                    part_index,
                    first,
                    second,
                    target_field,
                    config,
                    field_cache,
                    source_members[part_index],
                    source_snapshots[part_index],
                    pass_index=pass_index,
                )
                if proposal is not None:
                    proposals.append(proposal)
                    used.update((first, second))
                    position += 2
                else:
                    position += 1
    return proposals


def _nearest_level_indices(part: CoilPart, contours: list[ContourLine]) -> dict[int, int]:
    if part.potential_level_list is None:
        return {}
    levels = np.asarray(part.potential_level_list, dtype=float).reshape(-1)
    if len(levels) == 0:
        return {}
    return {
        contour_index: int(np.argmin(np.abs(levels - float(contour.potential))))
        for contour_index, contour in enumerate(contours)
    }


def _eligible_pair(
    part: CoilPart,
    first: int,
    second: int,
    source_members: list[tuple[int, ...]],
    original_level_indices: dict[int, int],
    config: CoilGenConfig,
) -> bool:
    contour_a = part.contour_lines[first]
    contour_b = part.contour_lines[second]
    max_multiplicity = max(int(getattr(config, "contour_sparsification_max_turn_multiplicity", 2)), 2)
    if contour_a.turn_multiplicity + contour_b.turn_multiplicity > max_multiplicity:
        return False
    levels_a = sorted(
        {original_level_indices[index] for index in source_members[first] if index in original_level_indices}
    )
    levels_b = sorted(
        {original_level_indices[index] for index in source_members[second] if index in original_level_indices}
    )
    if not levels_a or not levels_b:
        return False
    adjacent = levels_a[-1] + 1 == levels_b[0] or levels_b[-1] + 1 == levels_a[0]
    if not adjacent:
        return False
    if not np.isfinite(contour_a.potential) or not np.isfinite(contour_b.potential):
        return False
    return abs(float(contour_a.potential) - float(contour_b.potential)) > 0.0


def _proposal_for_pair(
    part: CoilPart,
    part_index: int,
    first: int,
    second: int,
    target_field: TargetField,
    config: CoilGenConfig,
    field_cache: dict[tuple[int, int], np.ndarray],
    source_members: list[tuple[int, ...]],
    source_snapshot: list[ContourLine],
    *,
    pass_index: int,
) -> _MergeProposal | None:
    contour_a = part.contour_lines[first]
    contour_b = part.contour_lines[second]
    source_original_indices = tuple(sorted((*source_members[first], *source_members[second])))
    source_turn_multiplicities = (int(contour_a.turn_multiplicity), int(contour_b.turn_multiplicity))
    total_multiplicity = sum(source_turn_multiplicities)
    source_field = field_cache[(part_index, first)] + field_cache[(part_index, second)]
    method = str(getattr(config, "contour_sparsification_positioning", "midpoint_level")).strip().lower()
    weighted_level = _source_weighted_potential(source_snapshot, source_original_indices)
    requested_levels = _candidate_levels(contour_a, contour_b, weighted_level, method, config)
    candidates: list[tuple[float, float, ContourLine, np.ndarray]] = []
    for requested_level in requested_levels:
        candidate = _matching_contour_at_level(
            part,
            requested_level,
            contour_a,
            contour_b,
            source_turn_multiplicities,
            config,
        )
        if candidate is None:
            continue
        candidate.turn_multiplicity = total_multiplicity
        candidate_field = float(total_multiplicity) * _unit_contour_field(
            part.mesh, candidate, target_field, config
        )
        residual = candidate_field - source_field
        rms_error, peak_error = _field_error_values(residual, target_field, config)
        candidates.append((rms_error, peak_error, candidate, candidate_field))
    if not candidates:
        return None
    rms_error, peak_error, candidate, candidate_field = min(
        candidates,
        key=lambda item: (item[0], item[1], abs(item[2].potential - weighted_level)),
    )
    return _MergeProposal(
        part_index=part_index,
        pass_index=pass_index,
        source_indices=(first, second),
        source_original_indices=source_original_indices,
        source_turn_multiplicities=source_turn_multiplicities,
        candidate=candidate,
        source_field=source_field,
        candidate_field=candidate_field,
        pair_rms_error=rms_error,
        pair_peak_error=peak_error,
        positioning_method=method,
    )


def _source_weighted_potential(source_snapshot: list[ContourLine], source_indices: tuple[int, ...]) -> float:
    weights = np.asarray([source_snapshot[index].turn_multiplicity for index in source_indices], dtype=float)
    potentials = np.asarray([source_snapshot[index].potential for index in source_indices], dtype=float)
    return float(np.dot(weights, potentials) / np.sum(weights))


def _candidate_levels(
    contour_a: ContourLine,
    contour_b: ContourLine,
    weighted_level: float,
    method: str,
    config: CoilGenConfig,
) -> tuple[float, ...]:
    if method == "midpoint_level":
        return (weighted_level,)
    candidate_count = max(int(getattr(config, "contour_sparsification_field_fit_candidate_count", 5)), 1)
    if candidate_count == 1:
        offsets = np.asarray([0.0])
    else:
        offsets = np.linspace(-0.25, 0.25, candidate_count)
    total = float(contour_a.turn_multiplicity + contour_b.turn_multiplicity)
    weighted_fraction = float(contour_b.turn_multiplicity) / total
    fractions = np.clip(weighted_fraction + offsets, 0.05, 0.95)
    levels = [weighted_level]
    levels.extend(
        (1.0 - float(fraction)) * float(contour_a.potential) + float(fraction) * float(contour_b.potential)
        for fraction in fractions
    )
    unique: list[float] = []
    for level in levels:
        if not any(np.isclose(level, existing, rtol=1.0e-12, atol=1.0e-15) for existing in unique):
            unique.append(float(level))
    return tuple(unique)


def _matching_contour_at_level(
    part: CoilPart,
    requested_level: float,
    contour_a: ContourLine,
    contour_b: ContourLine,
    source_turn_multiplicities: tuple[int, int],
    config: CoilGenConfig,
) -> ContourLine | None:
    regularized = regularize_contour_levels(
        part.mesh,
        np.asarray(part.stream_function, dtype=float),
        np.asarray([requested_level], dtype=float),
    )
    if len(regularized) != 1:
        return None
    level = float(regularized[0])
    segments = _segments_for_level(part.mesh, np.asarray(part.stream_function, dtype=float), level)
    contours, _diagnostics = _stitch_segments_with_diagnostics(segments, level)
    cleaned = [clean_contour_line(contour, config) for contour in contours]
    candidates = [contour for contour in cleaned if contour is not None]
    candidates = validate_and_correct_contour_currents(part, candidates)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda contour: _geometric_match_score(
            contour,
            contour_a,
            contour_b,
            source_turn_multiplicities,
        ),
    )


def _geometric_match_score(
    candidate: ContourLine,
    contour_a: ContourLine,
    contour_b: ContourLine,
    source_turn_multiplicities: tuple[int, int],
) -> float:
    center_a = np.mean(_unique_points(contour_a.v), axis=0)
    center_b = np.mean(_unique_points(contour_b.v), axis=0)
    center_candidate = np.mean(_unique_points(candidate.v), axis=0)
    weight_a, weight_b = (float(value) for value in source_turn_multiplicities)
    total_weight = weight_a + weight_b
    desired_center = (weight_a * center_a + weight_b * center_b) / total_weight
    geometry_scale = max(
        float(np.linalg.norm(np.ptp(np.vstack((contour_a.v, contour_b.v)), axis=0))),
        1.0e-12,
    )
    center_score = float(np.linalg.norm(center_candidate - desired_center) / geometry_scale)
    desired_length = max((weight_a * contour_a.length + weight_b * contour_b.length) / total_weight, 1.0e-12)
    length_score = abs(float(np.log(max(candidate.length, 1.0e-12) / desired_length)))
    return center_score + 0.25 * length_score


def _unique_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1.0e-10:
        return points[:-1]
    return points


def _source_field_cache(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
) -> dict[tuple[int, int], np.ndarray]:
    return {
        (part_index, contour_index): float(contour.turn_multiplicity)
        * _unit_contour_field(part.mesh, contour, target_field, config)
        for part_index, part in enumerate(parts)
        for contour_index, contour in enumerate(part.contour_lines)
    }


def _unit_contour_field(
    mesh: Mesh,
    contour: ContourLine,
    target_field: TargetField,
    config: CoilGenConfig,
) -> np.ndarray:
    return biot_savart_wire_path(
        contour_evaluation_points(mesh, contour, config),
        target_field.coords,
        chunk_size=int(config.biot_savart_chunk_size),
        worker_count=1,
        min_work_size=int(config.parallel_min_work_size),
        max_chunk_memory_mb=float(config.parallel_max_chunk_memory_mb),
    )


def _proposal_bundles(
    parts: list[CoilPart],
    proposals: list[_MergeProposal],
    config: CoilGenConfig,
) -> list[tuple[list[_MergeProposal], bool]]:
    if not bool(getattr(config, "contour_sparsification_preserve_symmetry", True)):
        return [([proposal], True) for proposal in proposals]

    symmetry_groups, protected, symmetry_complete = _contour_loop_pruning_groups(parts, config)
    orbit_by_key: dict[tuple[int, int], int] = {}
    orbit_keys: dict[int, set[tuple[int, int]]] = {}
    for orbit_index, group in enumerate(symmetry_groups):
        keys = {tuple(key) for key in group}
        orbit_keys[orbit_index] = keys
        for key in keys:
            orbit_by_key[key] = orbit_index

    by_orbits: dict[tuple[int, ...], list[_MergeProposal]] = {}
    for proposal in proposals:
        source_keys = [(proposal.part_index, index) for index in proposal.source_indices]
        if any(key in protected or key not in orbit_by_key for key in source_keys):
            by_orbits.setdefault((-1, id(proposal)), []).append(proposal)
            continue
        orbit_signature = tuple(sorted(orbit_by_key[key] for key in source_keys))
        by_orbits.setdefault(orbit_signature, []).append(proposal)

    bundles: list[tuple[list[_MergeProposal], bool]] = []
    for signature, bundle in by_orbits.items():
        if signature and signature[0] == -1:
            bundles.append((bundle, False))
            continue
        expected = set().union(*(orbit_keys[index] for index in set(signature)))
        consumed = {
            (proposal.part_index, contour_index) for proposal in bundle for contour_index in proposal.source_indices
        }
        bundles.append((bundle, bool(symmetry_complete and consumed == expected)))
    return sorted(
        bundles,
        key=lambda item: min((proposal.part_index, min(proposal.source_indices)) for proposal in item[0]),
    )


def _within_error_budget(rms_percent: float, peak_percent: float, config: CoilGenConfig) -> bool:
    rms_limit = float(getattr(config, "contour_sparsification_max_rms_field_error_percent", 1.0))
    peak_limit = float(getattr(config, "contour_sparsification_max_peak_field_error_percent", 5.0))
    return bool(
        np.isfinite(rms_percent)
        and np.isfinite(peak_percent)
        and rms_percent <= rms_limit + 1.0e-12
        and peak_percent <= peak_limit + 1.0e-12
    )


def _relative_field_error_percent(
    candidate_field: np.ndarray,
    reference_field: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
) -> tuple[float, float]:
    candidate_values, mask = _field_values_and_mask(candidate_field, target_field, config)
    reference_values, _ = _field_values_and_mask(reference_field, target_field, config)
    finite = mask & np.isfinite(candidate_values) & np.isfinite(reference_values)
    if not np.any(finite):
        return float("inf"), float("inf")
    diff = candidate_values[finite] - reference_values[finite]
    reference = reference_values[finite]
    rms_reference = max(_rms(reference), 1.0e-30)
    peak_reference = max(float(np.max(np.abs(reference))), 1.0e-30)
    return 100.0 * _rms(diff) / rms_reference, 100.0 * float(np.max(np.abs(diff))) / peak_reference


def _field_error_values(
    residual_field: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
) -> tuple[float, float]:
    values, mask = _field_values_and_mask(residual_field, target_field, config)
    finite = mask & np.isfinite(values)
    if not np.any(finite):
        return float("inf"), float("inf")
    selected = values[finite]
    return _rms(selected), float(np.max(np.abs(selected)))


def _field_values_and_mask(
    field: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    component = str(getattr(config, "target_field_component", "Bz"))
    b0_direction: Any = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    values = field_component_values(field, component, b0_direction=b0_direction).reshape(-1)
    mask = component_sample_values(dsv_target_mask(target_field), component).astype(bool).reshape(-1)
    return values, mask


def _sum_fields(fields: Any, target_count: int) -> np.ndarray:
    total = np.zeros((target_count, 3), dtype=float)
    for field in fields:
        total += np.asarray(field, dtype=float)
    return total


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _apply_pass_proposals(
    parts: list[CoilPart],
    source_members: list[list[tuple[int, ...]]],
    proposals: list[_MergeProposal],
) -> list[list[tuple[int, ...]]]:
    updated_members: list[list[tuple[int, ...]]] = []
    for part_index, part in enumerate(parts):
        part_proposals = [proposal for proposal in proposals if proposal.part_index == part_index]
        accepted_by_first = {min(proposal.source_indices): proposal for proposal in part_proposals if proposal.accepted}
        consumed = {
            contour_index
            for proposal in part_proposals
            if proposal.accepted
            for contour_index in proposal.source_indices
        }
        active: list[ContourLine] = []
        active_members: list[tuple[int, ...]] = []
        for contour_index, contour in enumerate(part.contour_lines):
            proposal = accepted_by_first.get(contour_index)
            if proposal is not None:
                proposal.representative_contour_index = len(active)
                active.append(proposal.candidate)
                active_members.append(proposal.source_original_indices)
            elif contour_index not in consumed:
                active.append(contour)
                active_members.append(source_members[part_index][contour_index])
        part.contour_lines = active
        updated_members.append(active_members)
        _clear_topology_state(part)
    return updated_members


def _merge_records(
    proposals: list[_MergeProposal],
    source_snapshot: list[ContourLine],
) -> list[ContourMergeRecord]:
    return [
        ContourMergeRecord(
            representative_contour_index=proposal.representative_contour_index,
            source_contour_indices=proposal.source_original_indices,
            source_potentials=tuple(
                float(source_snapshot[index].potential) for index in proposal.source_original_indices
            ),
            selected_potential=float(proposal.candidate.potential),
            turn_multiplicity=int(proposal.candidate.turn_multiplicity),
            positioning_method=proposal.positioning_method,
            pair_field_rms_error_T=float(proposal.pair_rms_error),
            pair_field_peak_error_T=float(proposal.pair_peak_error),
            accepted=bool(proposal.accepted),
            rejection_reason=proposal.rejection_reason,
            pass_index=int(proposal.pass_index),
            source_turn_multiplicities=proposal.source_turn_multiplicities,
        )
        for proposal in proposals
    ]


def _finalize_sparsification(
    parts: list[CoilPart],
    source_snapshots: list[list[ContourLine]],
    source_members: list[list[tuple[int, ...]]],
    proposals: list[_MergeProposal],
    *,
    mode: str,
    final_rms_percent: float,
    final_peak_percent: float,
    requested_passes: int,
    completed_passes: int,
    accepted_merges_by_pass: list[int],
    max_turn_multiplicity: int,
) -> None:
    for part_index, part in enumerate(parts):
        part_proposals = [proposal for proposal in proposals if proposal.part_index == part_index]
        records = _merge_records(part_proposals, source_snapshots[part_index])
        accepted_part_count = sum(record.accepted for record in records)
        part_pass_counts = tuple(
            sum(record.accepted and record.pass_index == pass_index for record in records)
            for pass_index in range(1, completed_passes + 1)
        )
        achieved_max_multiplicity = max(
            (int(contour.turn_multiplicity) for contour in part.contour_lines),
            default=0,
        )
        diagnostics: dict[str, object] = {
            "contour_sparsification_mode": mode,
            "contour_sparsification_realization": "ideal_weighted",
            "contour_sparsification_source_geometry_count": int(len(source_snapshots[part_index])),
            "contour_sparsification_nominal_geometry_count": int(len(part.contour_lines)),
            "contour_sparsification_physical_turn_count": int(
                sum(contour.turn_multiplicity for contour in part.contour_lines)
            ),
            "contour_sparsification_proposed_merge_count": int(len(records)),
            "contour_sparsification_accepted_merge_count": int(accepted_part_count),
            "contour_sparsification_rejected_merge_count": int(len(records) - accepted_part_count),
            "contour_sparsification_requested_pass_count": int(requested_passes),
            "contour_sparsification_completed_pass_count": int(completed_passes),
            "contour_sparsification_accepted_merges_by_pass": part_pass_counts,
            "contour_sparsification_global_accepted_merges_by_pass": tuple(accepted_merges_by_pass),
            "contour_sparsification_max_turn_multiplicity": int(max_turn_multiplicity),
            "contour_sparsification_achieved_max_turn_multiplicity": int(achieved_max_multiplicity),
            "contour_sparsification_final_turn_multiplicities": tuple(
                int(contour.turn_multiplicity) for contour in part.contour_lines
            ),
            "contour_sparsification_global_rms_field_error_percent": float(final_rms_percent),
            "contour_sparsification_global_peak_field_error_percent": float(final_peak_percent),
            "contour_sparsification_physical_realization_available": False,
        }
        part.contour_sparsification = ContourSparsificationResult(
            source_contours=source_snapshots[part_index],
            merge_records=records,
            final_source_contour_indices=list(source_members[part_index]),
            realized_turn_contours=[],
            diagnostics=diagnostics,
        )
        part.combined_loop_length = part.effective_turn_length
        part.loop_significance = None
        part.contour_level_diagnostics.update(diagnostics)
        _clear_topology_state(part)


def _clear_topology_state(part: CoilPart) -> None:
    part.groups = []
    part.loop_groups = []
    part.loop_in_loop_matrix = None
    part.topology_diagnostics = {}
    part.group_levels = []
    part.level_positions = []
    part.group_centers_v = None
    part.connected_groups = []
    part.opening_cuts_among_groups = []
    part.wire_path = None
