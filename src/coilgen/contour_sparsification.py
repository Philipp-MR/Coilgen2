from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CoilGenConfig
from .contours import (
    _contour_loop_metric_context,
    _contour_loop_pruning_groups,
    _contour_loop_residual_metric,
    _segments_for_level,
    _stitch_segments_with_diagnostics,
    clean_contour_line,
    contour_evaluation_points,
    regularize_contour_levels,
    validate_and_correct_contour_currents,
)
from .field_components import component_sample_values, field_component_values
from .layout import find_minimal_contour_distance, topological_loop_grouping
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


@dataclass(frozen=True)
class _PruningGroup:
    group_index: int
    keys: tuple[tuple[int, int], ...]
    field: np.ndarray
    significance_percent: float
    nominal_length_m: float
    effective_turn_length_m: float
    turn_count: int
    nonmanufacturable_geometry_count: int


@dataclass(frozen=True)
class _PruningSelection:
    removed_group_indices: tuple[int, ...]
    keep_keys: frozenset[tuple[int, int]]
    field: np.ndarray
    residual_metric: float
    current_A: float
    total_rms_error_percent: float
    total_peak_error_percent: float
    incremental_rms_error_percent: float
    incremental_peak_error_percent: float
    retained_geometry_count: int
    retained_nonmanufacturable_geometry_count: int
    retained_nominal_length_m: float


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
    When configured, the cumulative target-residual budget uses that same
    baseline across merging and later pruning. This stage deliberately does
    not invent separated physical tracks.
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
    residual_tolerances = _sparsification_residual_tolerances(config)
    residual_context = None
    baseline_residual_metric = float("nan")
    baseline_residual_current = float("nan")
    allowed_residual_metric = float("inf")
    if residual_tolerances is not None:
        residual_context = _contour_loop_metric_context(parts, target_field, config)
        baseline_residual_metric, baseline_residual_current = _contour_loop_residual_metric(
            baseline_field,
            residual_context,
        )
        relative_tolerance, absolute_tolerance = residual_tolerances
        allowed_residual_metric = baseline_residual_metric * (1.0 + relative_tolerance) + absolute_tolerance
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
            if residual_context is not None:
                candidate_metric, _ = _contour_loop_residual_metric(candidate_field, residual_context)
                numerical_tolerance = 1.0e-12 * max(
                    abs(allowed_residual_metric),
                    abs(baseline_residual_metric),
                    1.0e-30,
                )
                if (
                    not np.isfinite(candidate_metric)
                    or candidate_metric > allowed_residual_metric + numerical_tolerance
                ):
                    for proposal in bundle:
                        proposal.rejection_reason = "residual_budget"
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
    final_residual_metric = float("nan")
    final_residual_current = float("nan")
    if residual_context is not None:
        final_residual_metric, final_residual_current = _contour_loop_residual_metric(
            accepted_field,
            residual_context,
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
        baseline_residual_metric=baseline_residual_metric,
        allowed_residual_metric=allowed_residual_metric,
        final_residual_metric=final_residual_metric,
        baseline_residual_current=baseline_residual_current,
        final_residual_current=final_residual_current,
        residual_tolerances=residual_tolerances,
    )
    return parts


def prune_sparsified_contours(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
    *,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Remove redundant weighted representatives without spending field fidelity.

    Selection operates on complete symmetry orbits of the final
    multiplicity-weighted contours. For small layouts every orbit subset is
    evaluated globally; larger layouts use a deterministic beam search. The
    selected subset must stay within both the target-residual budget and the
    original sparsification field envelope measured against the retained
    unsparsified source contours.
    """

    mode = _normalise_sparsification_pruning_mode(getattr(config, "contour_sparsification_pruning_mode", "off"))
    sparsified_part_indices = [
        part_index for part_index, part in enumerate(parts) if part.contour_sparsification is not None
    ]
    if mode == "off" or not sparsified_part_indices:
        return parts

    field_by_key = _source_field_cache(parts, target_field, config)
    prunable_keys = {
        (part_index, contour_index)
        for part_index in sparsified_part_indices
        for contour_index in range(len(parts[part_index].contour_lines))
    }
    if not prunable_keys:
        return parts

    source_indices_by_key = _sparsification_source_indices_by_key(parts, sparsified_part_indices)
    sparse_baseline_field = _sum_fields(field_by_key.values(), len(target_field.coords))
    source_reference_field = _sparsification_source_reference_field(parts, target_field, config)
    metric_context = _contour_loop_metric_context(parts, target_field, config)
    baseline_metric, baseline_current = _contour_loop_residual_metric(sparse_baseline_field, metric_context)
    if not np.isfinite(baseline_metric):
        return parts

    source_baseline_metric, source_baseline_current = _contour_loop_residual_metric(
        source_reference_field,
        metric_context,
    )
    residual_tolerances = _sparsification_residual_tolerances(config)
    if residual_tolerances is None:
        relative_tolerance = max(
            float(getattr(config, "contour_sparsification_pruning_residual_relative_tolerance", 0.0)),
            0.0,
        )
        absolute_tolerance = max(
            float(getattr(config, "contour_sparsification_pruning_residual_absolute_tolerance_T", 0.0)),
            0.0,
        )
        residual_budget_baseline = baseline_metric
    else:
        relative_tolerance, absolute_tolerance = residual_tolerances
        residual_budget_baseline = source_baseline_metric
    allowed_metric = residual_budget_baseline * (1.0 + relative_tolerance) + absolute_tolerance
    maximum_removed = max(
        int(getattr(config, "contour_sparsification_pruning_max_removed_geometries", 0)),
        0,
    )
    target_geometry_count = max(
        int(getattr(config, "contour_sparsification_pruning_target_geometry_count", 0)),
        0,
    )
    preserve_symmetry = bool(getattr(config, "contour_sparsification_pruning_preserve_symmetry", True))
    groups, protected_keys, symmetry_complete = _contour_loop_pruning_groups(
        parts,
        config,
        preserve_symmetry=preserve_symmetry,
    )
    symmetry_validation_mode = str(getattr(config, "contour_symmetry_validation", "auto")).strip().lower()
    if preserve_symmetry and symmetry_validation_mode == "strict" and not symmetry_complete:
        raise ValueError(
            "Strict contour symmetry validation failed before post-sparsification pruning: "
            "at least one weighted contour has no complete geometry/potential/multiplicity orbit."
        )
    protected_keys.update(set(field_by_key) - prunable_keys)
    nonmanufacturable_keys = _nonmanufacturable_contour_keys(parts, config, prunable_keys)
    pruning_groups, protected_keys = _build_sparsification_pruning_groups(
        parts,
        groups,
        protected_keys,
        prunable_keys,
        field_by_key,
        sparse_baseline_field,
        metric_context,
        nonmanufacturable_keys,
        len(target_field.coords),
    )
    reporter = progress or ProgressReporter.from_config(config, label="Sparse contour pruning")
    selection, search_diagnostics = _select_sparsification_pruning_subset(
        pruning_groups,
        all_keys=set(field_by_key),
        prunable_keys=prunable_keys,
        protected_keys=protected_keys,
        nonmanufacturable_keys=nonmanufacturable_keys,
        sparse_baseline_field=sparse_baseline_field,
        source_reference_field=source_reference_field,
        target_field=target_field,
        config=config,
        metric_context=metric_context,
        baseline_metric=baseline_metric,
        baseline_current=baseline_current,
        allowed_metric=allowed_metric,
        maximum_removed=maximum_removed,
        target_geometry_count=target_geometry_count,
        reporter=reporter,
    )
    keep_keys = set(selection.keep_keys)
    active_field = selection.field
    selected_metric = selection.residual_metric
    selected_current = selection.current_A
    best_metric = float(search_diagnostics["best_metric_T"])
    removal_records = _selected_pruning_removal_records(
        pruning_groups,
        selection.removed_group_indices,
        parts,
        source_indices_by_key,
        sparse_baseline_field,
        source_reference_field,
        target_field,
        config,
        metric_context,
        baseline_metric,
    )

    subset_validation = _validate_symmetry_closed_pruning_subset(
        groups,
        protected_keys,
        prunable_keys,
        keep_keys,
        preserve_symmetry=preserve_symmetry,
    )
    if preserve_symmetry and not subset_validation["ok"]:
        raise ValueError(
            "Post-sparsification pruning selected an incomplete symmetry orbit: " + str(subset_validation["violations"])
        )

    final_total_rms, final_total_peak = _relative_field_error_percent(
        active_field,
        source_reference_field,
        target_field,
        config,
    )
    final_incremental_rms, final_incremental_peak = _relative_field_error_percent(
        active_field,
        sparse_baseline_field,
        target_field,
        config,
    )
    _apply_sparsified_pruning_result(
        parts,
        keep_keys,
        field_by_key,
        active_field,
        metric_context,
        source_indices_by_key,
        removal_records,
        mode=mode,
        baseline_metric=baseline_metric,
        selected_metric=selected_metric,
        best_metric=best_metric,
        allowed_metric=allowed_metric,
        source_baseline_metric=source_baseline_metric,
        source_baseline_current=source_baseline_current,
        baseline_current=baseline_current,
        selected_current=selected_current,
        final_total_rms=final_total_rms,
        final_total_peak=final_total_peak,
        final_incremental_rms=final_incremental_rms,
        final_incremental_peak=final_incremental_peak,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_removed=maximum_removed,
        target_geometry_count=target_geometry_count,
        preserve_symmetry=preserve_symmetry,
        symmetry_complete=symmetry_complete,
        protected_key_count=len(protected_keys & prunable_keys),
        search_diagnostics=search_diagnostics,
        subset_symmetry_validation=subset_validation,
    )
    for part_index in sparsified_part_indices:
        provenance = parts[part_index].contour_sparsification
        if provenance is None:
            continue
        provenance.diagnostics["contour_sparsification_pruning_nonmanufacturable_candidate_count"] = int(
            len(nonmanufacturable_keys)
        )
        provenance.diagnostics["contour_sparsification_pruning_retained_nonmanufacturable_count"] = int(
            len(nonmanufacturable_keys & keep_keys)
        )
    return parts


def consolidate_pruned_contours(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
    *,
    progress: ProgressReporter | None = None,
) -> list[CoilPart]:
    """Merge newly adjacent retained contours after sparsification pruning.

    This opt-in stage deliberately relaxes only the original-level adjacency
    rule. Candidate pairs must still be adjacent inside one retained intrinsic
    topology group, and every accepted symmetry bundle is checked against the
    original unsparsified field, the cumulative residual budget, spacing, and
    (when enabled) swept-body manufacturing constraints.
    """

    mode = _normalise_post_pruning_merge_mode(getattr(config, "contour_sparsification_post_pruning_merge_mode", "off"))
    sparsified_part_indices = [
        part_index for part_index, part in enumerate(parts) if part.contour_sparsification is not None
    ]
    if mode == "off" or not sparsified_part_indices:
        return parts

    source_snapshots: list[list[ContourLine]] = []
    source_members: list[list[tuple[int, ...]]] = []
    for part in parts:
        provenance = part.contour_sparsification
        if provenance is None:
            source_snapshots.append(list(part.contour_lines))
            source_members.append([(index,) for index in range(len(part.contour_lines))])
            continue
        if len(provenance.final_source_contour_indices) != len(part.contour_lines):
            raise ValueError("Sparsification provenance must align with retained contours before post-pruning merging.")
        source_snapshots.append(provenance.source_contours)
        source_members.append(list(provenance.final_source_contour_indices))

    before_geometry_counts = [len(part.contour_lines) for part in parts]
    before_turn_counts = [part.physical_turn_count for part in parts]
    source_reference_field = _sparsification_source_reference_field(parts, target_field, config)
    field_cache = _source_field_cache(parts, target_field, config)
    accepted_field = _sum_fields(field_cache.values(), len(target_field.coords))
    metric_context = _contour_loop_metric_context(parts, target_field, config)
    baseline_metric, baseline_current = _contour_loop_residual_metric(accepted_field, metric_context)
    source_baseline_metric, _ = _contour_loop_residual_metric(source_reference_field, metric_context)
    allowed_candidates = [
        float(part.contour_sparsification.diagnostics.get("contour_sparsification_residual_allowed_metric_T", np.nan))
        for part in parts
        if part.contour_sparsification is not None
    ]
    finite_allowed = [value for value in allowed_candidates if np.isfinite(value)]
    if finite_allowed:
        allowed_metric = min(finite_allowed)
    else:
        residual_tolerances = _sparsification_residual_tolerances(config)
        if residual_tolerances is None:
            residual_tolerances = (
                max(
                    float(
                        getattr(
                            config,
                            "contour_sparsification_pruning_residual_relative_tolerance",
                            0.0,
                        )
                    ),
                    0.0,
                ),
                max(
                    float(
                        getattr(
                            config,
                            "contour_sparsification_pruning_residual_absolute_tolerance_T",
                            0.0,
                        )
                    ),
                    0.0,
                ),
            )
            budget_baseline = baseline_metric
        else:
            budget_baseline = source_baseline_metric
        allowed_metric = budget_baseline * (1.0 + residual_tolerances[0]) + residual_tolerances[1]

    max_passes = max(
        int(getattr(config, "contour_sparsification_post_pruning_max_passes", 1)),
        1,
    )
    max_merges = max(int(getattr(config, "contour_sparsification_max_merges", 0)), 0)
    prior_accepted_count = sum(
        int(record.accepted)
        for part in parts
        if part.contour_sparsification is not None
        for record in part.contour_sparsification.merge_records
    )
    accepted_count = 0
    completed_passes = 0
    accepted_merges_by_pass: list[int] = []
    audit_proposals: list[_MergeProposal] = []
    audit_rows: list[dict[str, Any]] = []
    reporter = progress or ProgressReporter.from_config(config, label="Post-pruning contour merge")

    for pass_index in range(1, max_passes + 1):
        if max_merges > 0 and prior_accepted_count + accepted_count >= max_merges:
            break
        for part in parts:
            _clear_topology_state(part)
        topological_loop_grouping(parts, config=config, progress=reporter)
        field_cache = _source_field_cache(parts, target_field, config)
        option_sets = _build_post_pruning_merge_options(
            parts,
            target_field,
            config,
            field_cache,
            source_members,
            source_snapshots,
            pass_index=pass_index,
        )
        if not option_sets:
            break
        completed_passes = pass_index
        representative_options = [options[0] for options in option_sets]
        options_by_representative_id = {id(options[0]): options for options in option_sets}
        selected_this_pass: list[_MergeProposal] = []
        accepted_this_pass = 0

        for representative_bundle, complete in _proposal_bundles(parts, representative_options, config):
            if not complete:
                for proposal in representative_bundle:
                    proposal.rejection_reason = "incomplete_symmetry_orbit"
                audit_proposals.extend(representative_bundle)
                audit_rows.extend(_post_pruning_audit_rows(representative_bundle, accepted=False))
                continue
            if max_merges > 0 and prior_accepted_count + accepted_count + len(representative_bundle) > max_merges:
                for proposal in representative_bundle:
                    proposal.rejection_reason = "maximum_merge_count"
                audit_proposals.extend(representative_bundle)
                audit_rows.extend(_post_pruning_audit_rows(representative_bundle, accepted=False))
                continue

            bundle_option_sets = [options_by_representative_id[id(proposal)] for proposal in representative_bundle]
            candidate_evaluations: list[tuple[tuple[Any, ...], list[_MergeProposal], dict[str, Any]]] = []
            rejection_reasons: list[str] = []
            for option_index in range(min(len(options) for options in bundle_option_sets)):
                candidate_bundle = [options[option_index] for options in bundle_option_sets]
                evaluation = _evaluate_post_pruning_bundle(
                    parts,
                    selected_this_pass,
                    candidate_bundle,
                    accepted_field,
                    source_reference_field,
                    target_field,
                    config,
                    metric_context,
                    allowed_metric,
                )
                if not evaluation["ok"]:
                    rejection_reasons.append(str(evaluation["reason"]))
                    continue
                score = (
                    float(evaluation["total_rms_error_percent"]),
                    float(evaluation["total_peak_error_percent"]),
                    float(evaluation["residual_metric_T"]),
                    sum(float(proposal.pair_rms_error) for proposal in candidate_bundle),
                    tuple(float(proposal.candidate.potential) for proposal in candidate_bundle),
                )
                candidate_evaluations.append((score, candidate_bundle, evaluation))

            if not candidate_evaluations:
                reason = rejection_reasons[0] if rejection_reasons else "no_feasible_candidate"
                for proposal in representative_bundle:
                    proposal.rejection_reason = reason
                audit_proposals.extend(representative_bundle)
                audit_rows.extend(_post_pruning_audit_rows(representative_bundle, accepted=False))
                continue

            _score, selected_bundle, evaluation = min(candidate_evaluations, key=lambda item: item[0])
            for proposal in selected_bundle:
                proposal.accepted = True
                proposal.rejection_reason = ""
            selected_this_pass.extend(selected_bundle)
            audit_proposals.extend(selected_bundle)
            audit_rows.extend(
                _post_pruning_audit_rows(
                    selected_bundle,
                    accepted=True,
                    evaluation=evaluation,
                )
            )
            accepted_field = np.asarray(evaluation["field"], dtype=float)
            accepted_count += len(selected_bundle)
            accepted_this_pass += len(selected_bundle)

        accepted_merges_by_pass.append(accepted_this_pass)
        if accepted_this_pass == 0:
            break
        source_members = _apply_pass_proposals(parts, source_members, selected_this_pass)

    final_field_cache = _source_field_cache(parts, target_field, config)
    final_field = _sum_fields(final_field_cache.values(), len(target_field.coords))
    final_rms, final_peak = _relative_field_error_percent(
        final_field,
        source_reference_field,
        target_field,
        config,
    )
    final_metric, final_current = _contour_loop_residual_metric(final_field, metric_context)
    final_scale = _field_significance_scale(final_field, metric_context)

    for part_index, part in enumerate(parts):
        provenance = part.contour_sparsification
        if provenance is None:
            continue
        if part.physical_turn_count != before_turn_counts[part_index]:
            raise ValueError("Post-pruning contour consolidation changed the represented physical turn count.")
        part_proposals = [proposal for proposal in audit_proposals if proposal.part_index == part_index]
        part_records = _merge_records(
            part_proposals,
            source_snapshots[part_index],
            stage="post_pruning",
        )
        provenance.merge_records.extend(part_records)
        provenance.final_source_contour_indices = list(source_members[part_index])
        accepted_part_count = sum(record.accepted for record in part_records)
        proposed_part_count = len(part_records)
        existing_accepted = int(provenance.diagnostics.get("contour_sparsification_accepted_merge_count", 0))
        existing_proposed = int(provenance.diagnostics.get("contour_sparsification_proposed_merge_count", 0))
        provenance.diagnostics.update(
            {
                "contour_sparsification_post_pruning_merge_mode": mode,
                "contour_sparsification_post_pruning_requested_pass_count": int(max_passes),
                "contour_sparsification_post_pruning_completed_pass_count": int(completed_passes),
                "contour_sparsification_post_pruning_accepted_merges_by_pass": tuple(accepted_merges_by_pass),
                "contour_sparsification_post_pruning_proposed_merge_count": int(proposed_part_count),
                "contour_sparsification_post_pruning_accepted_merge_count": int(accepted_part_count),
                "contour_sparsification_post_pruning_rejected_merge_count": int(
                    proposed_part_count - accepted_part_count
                ),
                "contour_sparsification_post_pruning_pre_geometry_count": int(before_geometry_counts[part_index]),
                "contour_sparsification_post_pruning_final_geometry_count": int(len(part.contour_lines)),
                "contour_sparsification_post_pruning_pre_turn_count": int(before_turn_counts[part_index]),
                "contour_sparsification_post_pruning_final_turn_count": int(part.physical_turn_count),
                "contour_sparsification_post_pruning_baseline_metric_T": float(baseline_metric),
                "contour_sparsification_post_pruning_selected_metric_T": float(final_metric),
                "contour_sparsification_post_pruning_metric_change_T": float(final_metric - baseline_metric),
                "contour_sparsification_post_pruning_allowed_metric_T": float(allowed_metric),
                "contour_sparsification_post_pruning_baseline_current_A": float(baseline_current),
                "contour_sparsification_post_pruning_selected_current_A": float(final_current),
                "contour_sparsification_post_pruning_total_rms_field_error_percent": float(final_rms),
                "contour_sparsification_post_pruning_total_peak_field_error_percent": float(final_peak),
                "contour_sparsification_post_pruning_merge_records": [
                    row for row in audit_rows if int(row["part_index"]) == part_index
                ],
                "contour_sparsification_proposed_merge_count": int(existing_proposed + proposed_part_count),
                "contour_sparsification_accepted_merge_count": int(existing_accepted + accepted_part_count),
                "contour_sparsification_rejected_merge_count": int(
                    existing_proposed + proposed_part_count - existing_accepted - accepted_part_count
                ),
                "contour_sparsification_global_rms_field_error_percent": float(final_rms),
                "contour_sparsification_global_peak_field_error_percent": float(final_peak),
                "contour_sparsification_nominal_geometry_count": int(len(part.contour_lines)),
                "contour_sparsification_physical_turn_count": int(part.physical_turn_count),
                "contour_sparsification_achieved_max_turn_multiplicity": max(
                    (int(contour.turn_multiplicity) for contour in part.contour_lines),
                    default=0,
                ),
                "contour_sparsification_final_turn_multiplicities": tuple(
                    int(contour.turn_multiplicity) for contour in part.contour_lines
                ),
            }
        )
        part.loop_significance = np.asarray(
            [
                _field_significance_percent(
                    final_field_cache[(part_index, contour_index)],
                    metric_context,
                    final_scale,
                )
                for contour_index in range(len(part.contour_lines))
            ],
            dtype=float,
        )
        part.combined_loop_length = part.effective_turn_length
        part.contour_level_diagnostics.update(provenance.diagnostics)
        _clear_topology_state(part)
    return parts


def _build_post_pruning_merge_options(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
    field_cache: dict[tuple[int, int], np.ndarray],
    source_members: list[list[tuple[int, ...]]],
    source_snapshots: list[list[ContourLine]],
    *,
    pass_index: int,
) -> list[list[_MergeProposal]]:
    option_sets: list[list[_MergeProposal]] = []
    for part_index, part in enumerate(parts):
        if part.contour_sparsification is None:
            continue
        used: set[int] = set()
        for group in part.groups:
            ordered = sorted(
                (int(index) for index in group.loop_indices),
                key=lambda index: (part.contour_lines[index].potential, index),
            )
            position = 0
            while position + 1 < len(ordered):
                first, second = ordered[position], ordered[position + 1]
                potential_a = float(part.contour_lines[first].potential)
                potential_b = float(part.contour_lines[second].potential)
                compatible_polarity = potential_a * potential_b > 0.0
                if (
                    first in used
                    or second in used
                    or not compatible_polarity
                    or not _eligible_pair(
                        part,
                        first,
                        second,
                        source_members[part_index],
                        {},
                        config,
                        require_original_adjacency=False,
                    )
                ):
                    position += 1
                    continue
                options = _proposal_options_for_pair(
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
                if options:
                    option_sets.append(options)
                    used.update((first, second))
                    position += 2
                else:
                    position += 1
    return option_sets


def _evaluate_post_pruning_bundle(
    parts: list[CoilPart],
    already_selected: list[_MergeProposal],
    bundle: list[_MergeProposal],
    accepted_field: np.ndarray,
    source_reference_field: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
    metric_context: dict[str, Any],
    allowed_metric: float,
) -> dict[str, Any]:
    candidate_field = accepted_field + _sum_fields(
        (proposal.field_delta for proposal in bundle),
        len(target_field.coords),
    )
    total_rms, total_peak = _relative_field_error_percent(
        candidate_field,
        source_reference_field,
        target_field,
        config,
    )
    if not _within_error_budget(total_rms, total_peak, config):
        return {"ok": False, "reason": "field_error_budget"}
    residual_metric, residual_current = _contour_loop_residual_metric(
        candidate_field,
        metric_context,
    )
    numerical_tolerance = 1.0e-12 * max(
        abs(allowed_metric),
        abs(residual_metric) if np.isfinite(residual_metric) else 0.0,
        1.0e-30,
    )
    if not np.isfinite(residual_metric) or residual_metric > allowed_metric + numerical_tolerance:
        return {"ok": False, "reason": "residual_budget"}
    for proposal in bundle:
        if not _candidate_preserves_source_topology(parts[proposal.part_index], proposal, config):
            return {"ok": False, "reason": "topology_mismatch"}

    tentative_parts = _parts_with_merge_proposals(parts, [*already_selected, *bundle])
    preserve_symmetry = bool(getattr(config, "contour_sparsification_preserve_symmetry", True))
    if preserve_symmetry:
        _groups, _protected, symmetry_complete = _contour_loop_pruning_groups(
            tentative_parts,
            config,
            preserve_symmetry=True,
        )
        if not symmetry_complete:
            return {"ok": False, "reason": "final_symmetry_validation"}

    minimum_spacing = float("inf")
    required_spacing = max(float(getattr(config, "min_turn_distance", 0.0)), 0.0)
    if required_spacing > 0.0:
        spacing_config = deepcopy(config)
        spacing_config.skip_calculation_min_winding_distance = False
        find_minimal_contour_distance(tentative_parts, spacing_config)
        measured = [
            float(part.minimal_contour_distance)
            for part in tentative_parts
            if len(part.contour_lines) > 1 and np.isfinite(part.minimal_contour_distance)
        ]
        minimum_spacing = min(measured) if measured else float("inf")
        if minimum_spacing + 1.0e-12 < required_spacing:
            return {"ok": False, "reason": "minimum_spacing"}

    manufacturing_checked = bool(
        getattr(config, "contour_only_sweep_mode", False) and not bool(getattr(config, "skip_sweep", True))
    )
    manufacturing_rows: list[dict[str, Any]] = []
    if manufacturing_checked:
        from .manufacturing import create_swept_contour_body

        for proposal in bundle:
            try:
                body = create_swept_contour_body(
                    parts[proposal.part_index],
                    proposal.candidate,
                    config,
                    part_index=proposal.part_index + 1,
                    contour_index=min(proposal.source_indices) + 1,
                    validate=False,
                )
            except (RuntimeError, ValueError):
                return {"ok": False, "reason": "manufacturing_validation"}
            row = {
                "manufacturing_ok": bool(body.diagnostics.get("manufacturing_ok", False)),
                "max_curvature_envelope_ratio": float(
                    body.diagnostics.get("sweep_max_curvature_envelope_ratio", np.nan)
                ),
                "curvature_envelope_limit": float(body.diagnostics.get("sweep_curvature_envelope_limit", np.nan)),
                "capsule_self_intersection_count": int(
                    body.diagnostics.get("sweep_capsule_self_intersection_count", 0)
                ),
                "mesh_self_intersection_count": int(body.diagnostics.get("sweep_mesh_self_intersection_count", 0)),
            }
            manufacturing_rows.append(row)
            if not row["manufacturing_ok"]:
                return {"ok": False, "reason": "manufacturing_validation"}

    return {
        "ok": True,
        "reason": "",
        "field": candidate_field,
        "total_rms_error_percent": float(total_rms),
        "total_peak_error_percent": float(total_peak),
        "residual_metric_T": float(residual_metric),
        "residual_current_A": float(residual_current),
        "minimum_spacing_m": float(minimum_spacing),
        "required_spacing_m": float(required_spacing),
        "manufacturing_checked": manufacturing_checked,
        "manufacturing": manufacturing_rows,
    }


def _candidate_preserves_source_topology(
    part: CoilPart,
    proposal: _MergeProposal,
    config: CoilGenConfig,
) -> bool:
    first, second = proposal.source_indices
    contains = part.loop_in_loop_matrix
    if contains is None or contains.shape[0] <= max(first, second):
        return False
    first_contains_second = bool(contains[first, second])
    second_contains_first = bool(contains[second, first])
    if first_contains_second and second_contains_first:
        return False
    trial = deepcopy(part)
    trial.contour_lines = [
        deepcopy(part.contour_lines[first]),
        deepcopy(proposal.candidate),
        deepcopy(part.contour_lines[second]),
    ]
    _clear_topology_state(trial)
    try:
        topological_loop_grouping([trial], config=config)
    except (RuntimeError, ValueError):
        return False
    trial_contains = trial.loop_in_loop_matrix
    if trial_contains is None or trial_contains.shape != (3, 3):
        return False
    same_group = any(set(group.loop_indices) == {0, 1, 2} for group in trial.groups)
    if not same_group:
        return False
    if first_contains_second:
        return bool(trial_contains[0, 1] and trial_contains[1, 2])
    if second_contains_first:
        return bool(trial_contains[2, 1] and trial_contains[1, 0])
    # Boundary-associated essential loops do not have a containment ordering.
    # For those parallel-surface families, remaining in one intrinsic group is
    # the topology-preserving equivalent of lying between nested loops.
    return True


def _parts_with_merge_proposals(
    parts: list[CoilPart],
    proposals: list[_MergeProposal],
) -> list[CoilPart]:
    tentative = deepcopy(parts)
    for part_index, part in enumerate(tentative):
        part_proposals = [proposal for proposal in proposals if proposal.part_index == part_index]
        candidate_by_first = {min(proposal.source_indices): proposal.candidate for proposal in part_proposals}
        consumed = {contour_index for proposal in part_proposals for contour_index in proposal.source_indices}
        part.contour_lines = [
            deepcopy(candidate_by_first[index]) if index in candidate_by_first else contour
            for index, contour in enumerate(part.contour_lines)
            if index not in consumed or index in candidate_by_first
        ]
        _clear_topology_state(part)
    return tentative


def _post_pruning_audit_rows(
    proposals: list[_MergeProposal],
    *,
    accepted: bool,
    evaluation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    evaluation = evaluation or {}
    manufacturing = list(evaluation.get("manufacturing", ()))
    rows: list[dict[str, Any]] = []
    for proposal_index, proposal in enumerate(proposals):
        manufacturing_row = manufacturing[proposal_index] if proposal_index < len(manufacturing) else {}
        rows.append(
            {
                "part_index": int(proposal.part_index),
                "pass_index": int(proposal.pass_index),
                "source_representative_indices": [int(index) for index in proposal.source_indices],
                "source_contour_indices": [int(index) for index in proposal.source_original_indices],
                "source_turn_multiplicities": [int(value) for value in proposal.source_turn_multiplicities],
                "selected_potential_A": float(proposal.candidate.potential),
                "turn_multiplicity": int(proposal.candidate.turn_multiplicity),
                "accepted": bool(accepted),
                "rejection_reason": "" if accepted else str(proposal.rejection_reason),
                "pair_rms_field_error_T": float(proposal.pair_rms_error),
                "pair_peak_field_error_T": float(proposal.pair_peak_error),
                "total_rms_field_error_percent": float(evaluation.get("total_rms_error_percent", np.nan)),
                "total_peak_field_error_percent": float(evaluation.get("total_peak_error_percent", np.nan)),
                "residual_metric_T": float(evaluation.get("residual_metric_T", np.nan)),
                "minimum_spacing_m": float(evaluation.get("minimum_spacing_m", np.nan)),
                "required_spacing_m": float(evaluation.get("required_spacing_m", np.nan)),
                "manufacturing_checked": bool(evaluation.get("manufacturing_checked", False)),
                "manufacturing_ok": bool(manufacturing_row.get("manufacturing_ok", False)),
                "max_curvature_envelope_ratio": float(manufacturing_row.get("max_curvature_envelope_ratio", np.nan)),
                "curvature_envelope_limit": float(manufacturing_row.get("curvature_envelope_limit", np.nan)),
            }
        )
    return rows


def _build_sparsification_pruning_groups(
    parts: list[CoilPart],
    symmetry_groups: list[list[tuple[int, int]]],
    protected_keys: set[tuple[int, int]],
    prunable_keys: set[tuple[int, int]],
    field_by_key: dict[tuple[int, int], np.ndarray],
    sparse_baseline_field: np.ndarray,
    metric_context: dict[str, Any],
    nonmanufacturable_keys: set[tuple[int, int]],
    target_count: int,
) -> tuple[list[_PruningGroup], set[tuple[int, int]]]:
    """Normalize symmetry orbits into disjoint atomic pruning decisions."""

    protected = set(protected_keys)
    active_member_keys: set[tuple[int, int]] = set()
    normalized: list[tuple[tuple[int, int], ...]] = []
    for symmetry_group in symmetry_groups:
        full_group = tuple(sorted(set(symmetry_group)))
        active_group = tuple(key for key in full_group if key in prunable_keys)
        if not active_group:
            continue
        overlap = active_member_keys.intersection(active_group)
        if overlap:
            raise ValueError(f"Contour symmetry groups overlap at {sorted(overlap)}.")
        active_member_keys.update(active_group)
        if any(key not in prunable_keys for key in full_group):
            # Removing only the sparsified side of a cross-part orbit would be
            # a unilateral symmetry change. Keep that complete orbit fixed.
            protected.update(active_group)
        normalized.append(active_group)

    missing = prunable_keys - active_member_keys
    if missing:
        # A contour omitted by the matcher has no proven complete orbit.
        # Protect it instead of silently treating it as independently removable.
        protected.update(missing)
        normalized.extend((key,) for key in sorted(missing))

    pruning_groups: list[_PruningGroup] = []
    for group_index, keys in enumerate(normalized):
        contours = [parts[part_index].contour_lines[contour_index] for part_index, contour_index in keys]
        pruning_groups.append(
            _PruningGroup(
                group_index=group_index,
                keys=keys,
                field=_sum_fields((field_by_key[key] for key in keys), target_count),
                significance_percent=_group_significance_percent(
                    keys,
                    field_by_key,
                    sparse_baseline_field,
                    metric_context,
                ),
                nominal_length_m=float(sum(contour.length for contour in contours)),
                effective_turn_length_m=float(sum(contour.effective_turn_length for contour in contours)),
                turn_count=int(sum(contour.turn_multiplicity for contour in contours)),
                nonmanufacturable_geometry_count=int(len(set(keys) & nonmanufacturable_keys)),
            )
        )
    return pruning_groups, protected


def _select_sparsification_pruning_subset(
    pruning_groups: list[_PruningGroup],
    *,
    all_keys: set[tuple[int, int]],
    prunable_keys: set[tuple[int, int]],
    protected_keys: set[tuple[int, int]],
    nonmanufacturable_keys: set[tuple[int, int]],
    sparse_baseline_field: np.ndarray,
    source_reference_field: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
    metric_context: dict[str, Any],
    baseline_metric: float,
    baseline_current: float,
    allowed_metric: float,
    maximum_removed: int,
    target_geometry_count: int,
    reporter: ProgressReporter,
) -> tuple[_PruningSelection, dict[str, Any]]:
    """Select a smallest feasible union of complete symmetry orbits.

    The retained geometry target is a lower bound. If its exact value is not
    reachable by complete orbits, the smallest feasible count above it wins.
    This is what prevents an odd target from splitting a mirror pair.
    """

    minimum_geometry_count = max(target_geometry_count, 1)
    removable_groups = [group for group in pruning_groups if not (set(group.keys) & protected_keys)]
    group_by_index = {group.group_index: group for group in pruning_groups}
    exact_max_orbits = max(
        int(getattr(config, "contour_sparsification_pruning_exact_max_orbits", 16)),
        0,
    )
    beam_width = max(
        int(getattr(config, "contour_sparsification_pruning_beam_width", 128)),
        1,
    )
    requested_mode = (
        str(getattr(config, "contour_sparsification_pruning_search_mode", "auto")).strip().lower().replace("-", "_")
    )
    if requested_mode not in {"auto", "exact", "beam"}:
        requested_mode = "auto"
    search_mode = (
        "exact"
        if requested_mode == "exact" or (requested_mode == "auto" and len(removable_groups) <= exact_max_orbits)
        else "beam"
    )

    baseline_total_rms, baseline_total_peak = _relative_field_error_percent(
        sparse_baseline_field,
        source_reference_field,
        target_field,
        config,
    )
    baseline_selection = _PruningSelection(
        removed_group_indices=(),
        keep_keys=frozenset(all_keys),
        field=sparse_baseline_field.copy(),
        residual_metric=float(baseline_metric),
        current_A=float(baseline_current),
        total_rms_error_percent=float(baseline_total_rms),
        total_peak_error_percent=float(baseline_total_peak),
        incremental_rms_error_percent=0.0,
        incremental_peak_error_percent=0.0,
        retained_geometry_count=int(len(prunable_keys)),
        retained_nonmanufacturable_geometry_count=int(len(prunable_keys & nonmanufacturable_keys)),
        retained_nominal_length_m=float(sum(group.nominal_length_m for group in pruning_groups)),
    )

    if not removable_groups:
        return baseline_selection, {
            "search_mode": search_mode,
            "requested_search_mode": requested_mode,
            "removable_orbit_count": 0,
            "evaluated_subset_count": 1,
            "feasible_subset_count": 1,
            "optimality_proven": True,
            "target_count_feasible": bool(target_geometry_count == 0 or len(prunable_keys) == target_geometry_count),
            "best_metric_T": float(baseline_metric),
        }

    def evaluate(removed_group_indices: tuple[int, ...]) -> _PruningSelection | None:
        removed_groups = [group_by_index[index] for index in removed_group_indices]
        removed_keys = {key for group in removed_groups for key in group.keys}
        removed_geometry_count = len(removed_keys)
        retained_geometry_count = len(prunable_keys) - removed_geometry_count
        if retained_geometry_count < minimum_geometry_count:
            return None
        if maximum_removed and removed_geometry_count > maximum_removed:
            return None
        candidate_field = sparse_baseline_field - _sum_fields(
            (group.field for group in removed_groups),
            len(target_field.coords),
        )
        candidate_metric, candidate_current = _contour_loop_residual_metric(
            candidate_field,
            metric_context,
        )
        if not np.isfinite(candidate_metric):
            return None
        total_rms, total_peak = _relative_field_error_percent(
            candidate_field,
            source_reference_field,
            target_field,
            config,
        )
        incremental_rms, incremental_peak = _relative_field_error_percent(
            candidate_field,
            sparse_baseline_field,
            target_field,
            config,
        )
        keep_keys = all_keys - removed_keys
        return _PruningSelection(
            removed_group_indices=removed_group_indices,
            keep_keys=frozenset(keep_keys),
            field=candidate_field,
            residual_metric=float(candidate_metric),
            current_A=float(candidate_current),
            total_rms_error_percent=float(total_rms),
            total_peak_error_percent=float(total_peak),
            incremental_rms_error_percent=float(incremental_rms),
            incremental_peak_error_percent=float(incremental_peak),
            retained_geometry_count=int(retained_geometry_count),
            retained_nonmanufacturable_geometry_count=int(len(keep_keys & prunable_keys & nonmanufacturable_keys)),
            retained_nominal_length_m=float(
                sum(
                    group.nominal_length_m for group in pruning_groups if group.group_index not in removed_group_indices
                )
            ),
        )

    numerical_tolerance = 1.0e-12 * max(abs(allowed_metric), abs(baseline_metric), 1.0e-30)

    def feasible(candidate: _PruningSelection) -> bool:
        return bool(
            candidate.residual_metric <= allowed_metric + numerical_tolerance
            and _within_error_budget(
                candidate.total_rms_error_percent,
                candidate.total_peak_error_percent,
                config,
            )
        )

    def selection_score(candidate: _PruningSelection) -> tuple[Any, ...]:
        return (
            candidate.retained_geometry_count,
            candidate.retained_nonmanufacturable_geometry_count,
            candidate.residual_metric,
            candidate.total_rms_error_percent,
            candidate.total_peak_error_percent,
            -candidate.retained_nominal_length_m,
            candidate.removed_group_indices,
        )

    evaluated_count = 1
    feasible_selections = [baseline_selection]
    best_metric = float(baseline_metric)
    if search_mode == "exact":
        subset_count = 1 << len(removable_groups)
        task = reporter.task("exact symmetry-orbit subset search", max(subset_count - 1, 0))
        for mask in range(1, subset_count):
            removed = tuple(
                removable_groups[index].group_index for index in range(len(removable_groups)) if mask & (1 << index)
            )
            candidate = evaluate(removed)
            evaluated_count += 1
            if candidate is not None:
                if feasible(candidate):
                    best_metric = min(best_metric, candidate.residual_metric)
                    feasible_selections.append(candidate)
            task.update(mask)
        task.done()
        optimality_proven = True
    else:
        task = reporter.task("beam symmetry-orbit subset search", len(removable_groups))
        frontier: list[tuple[int, ...]] = [()]
        visited: set[tuple[int, ...]] = {()}
        for depth in range(1, len(removable_groups) + 1):
            expanded: list[_PruningSelection] = []
            for removed in frontier:
                for group in removable_groups:
                    if group.group_index in removed:
                        continue
                    candidate_indices = tuple(sorted((*removed, group.group_index)))
                    if candidate_indices in visited:
                        continue
                    visited.add(candidate_indices)
                    candidate = evaluate(candidate_indices)
                    evaluated_count += 1
                    if candidate is None:
                        continue
                    expanded.append(candidate)
                    if feasible(candidate):
                        best_metric = min(best_metric, candidate.residual_metric)
                        feasible_selections.append(candidate)
            if not expanded:
                task.update(depth)
                break

            def beam_score(candidate: _PruningSelection) -> tuple[Any, ...]:
                residual_excess = max(candidate.residual_metric - allowed_metric, 0.0) / max(
                    abs(allowed_metric), 1.0e-30
                )
                rms_limit = max(
                    float(getattr(config, "contour_sparsification_max_rms_field_error_percent", 0.0)),
                    1.0e-30,
                )
                peak_limit = max(
                    float(getattr(config, "contour_sparsification_max_peak_field_error_percent", 0.0)),
                    1.0e-30,
                )
                budget_excess = (
                    residual_excess
                    + max(candidate.total_rms_error_percent / rms_limit - 1.0, 0.0)
                    + max(candidate.total_peak_error_percent / peak_limit - 1.0, 0.0)
                )
                return (
                    budget_excess,
                    candidate.residual_metric,
                    candidate.retained_nonmanufacturable_geometry_count,
                    candidate.total_rms_error_percent,
                    candidate.removed_group_indices,
                )

            expanded.sort(key=beam_score)
            frontier = [candidate.removed_group_indices for candidate in expanded[:beam_width]]
            task.update(depth)
        task.done()
        optimality_proven = False

    selection = min(feasible_selections, key=selection_score)
    target_count_feasible = bool(
        target_geometry_count == 0
        or any(candidate.retained_geometry_count == target_geometry_count for candidate in feasible_selections)
    )
    return selection, {
        "search_mode": search_mode,
        "requested_search_mode": requested_mode,
        "removable_orbit_count": int(len(removable_groups)),
        "evaluated_subset_count": int(evaluated_count),
        "feasible_subset_count": int(len(feasible_selections)),
        "optimality_proven": bool(optimality_proven),
        "target_count_feasible": target_count_feasible,
        "best_metric_T": float(best_metric),
        "selected_removed_orbit_indices": [int(index) for index in selection.removed_group_indices],
    }


def _selected_pruning_removal_records(
    pruning_groups: list[_PruningGroup],
    removed_group_indices: tuple[int, ...],
    parts: list[CoilPart],
    source_indices_by_key: dict[tuple[int, int], tuple[int, ...]],
    sparse_baseline_field: np.ndarray,
    source_reference_field: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
    metric_context: dict[str, Any],
    baseline_metric: float,
) -> list[dict[str, Any]]:
    """Build a deterministic audit trail for the globally selected subset."""

    group_by_index = {group.group_index: group for group in pruning_groups}
    active_field = sparse_baseline_field.copy()
    active_metric = float(baseline_metric)
    records: list[dict[str, Any]] = []
    for step, group_index in enumerate(removed_group_indices, start=1):
        group = group_by_index[group_index]
        candidate_field = active_field - group.field
        candidate_metric, candidate_current = _contour_loop_residual_metric(candidate_field, metric_context)
        total_rms, total_peak = _relative_field_error_percent(
            candidate_field,
            source_reference_field,
            target_field,
            config,
        )
        incremental_rms, incremental_peak = _relative_field_error_percent(
            candidate_field,
            sparse_baseline_field,
            target_field,
            config,
        )
        records.append(
            {
                "step": int(step),
                "symmetry_orbit_index": int(group_index),
                "representative_indices": [
                    [int(part_index), int(contour_index)] for part_index, contour_index in group.keys
                ],
                "source_contour_indices": [
                    {
                        "part_index": int(part_index),
                        "indices": [int(index) for index in source_indices_by_key[(part_index, contour_index)]],
                    }
                    for part_index, contour_index in group.keys
                ],
                "removed_turn_count": int(group.turn_count),
                "nominal_length_m": float(group.nominal_length_m),
                "effective_turn_length_m": float(group.effective_turn_length_m),
                "significance_percent": float(group.significance_percent),
                "nonmanufacturable_geometry_count": int(group.nonmanufacturable_geometry_count),
                "residual_metric_before_T": float(active_metric),
                "residual_metric_after_T": float(candidate_metric),
                "residual_metric_change_T": float(candidate_metric - active_metric),
                "evaluation_current_A": float(candidate_current),
                "total_rms_field_error_percent": float(total_rms),
                "total_peak_field_error_percent": float(total_peak),
                "incremental_rms_field_error_percent": float(incremental_rms),
                "incremental_peak_field_error_percent": float(incremental_peak),
            }
        )
        active_field = candidate_field
        active_metric = float(candidate_metric)
    return records


def _validate_symmetry_closed_pruning_subset(
    symmetry_groups: list[list[tuple[int, int]]],
    protected_keys: set[tuple[int, int]],
    prunable_keys: set[tuple[int, int]],
    keep_keys: set[tuple[int, int]],
    *,
    preserve_symmetry: bool,
) -> dict[str, Any]:
    """Verify the prospective final layout is a union of complete orbits."""

    if not preserve_symmetry:
        return {
            "enabled": False,
            "ok": True,
            "checked_orbit_count": 0,
            "violations": [],
        }
    violations: list[dict[str, Any]] = []
    checked_orbits = 0
    for orbit_index, group in enumerate(symmetry_groups):
        orbit = set(group) & prunable_keys
        if not orbit:
            continue
        checked_orbits += 1
        retained = orbit & keep_keys
        if retained and retained != orbit:
            violations.append(
                {
                    "orbit_index": int(orbit_index),
                    "members": [list(key) for key in sorted(orbit)],
                    "retained_members": [list(key) for key in sorted(retained)],
                }
            )
    removed_protected = (protected_keys & prunable_keys) - keep_keys
    if removed_protected:
        violations.append(
            {
                "reason": "protected_unmatched_contour_removed",
                "members": [list(key) for key in sorted(removed_protected)],
            }
        )
    return {
        "enabled": True,
        "ok": not violations,
        "checked_orbit_count": int(checked_orbits),
        "protected_geometry_count": int(len(protected_keys & prunable_keys)),
        "violations": violations,
    }


def _nonmanufacturable_contour_keys(
    parts: list[CoilPart],
    config: CoilGenConfig,
    candidate_keys: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return sparse representatives that cannot form strict swept solids."""

    if (
        not bool(getattr(config, "contour_only_sweep_mode", False))
        or bool(getattr(config, "skip_sweep", True))
        or not candidate_keys
    ):
        return set()
    from .manufacturing import create_swept_contour_body

    invalid: set[tuple[int, int]] = set()
    for part_index, contour_index in sorted(candidate_keys):
        try:
            body = create_swept_contour_body(
                parts[part_index],
                parts[part_index].contour_lines[contour_index],
                config,
                part_index=part_index + 1,
                contour_index=contour_index + 1,
                validate=False,
            )
        except (RuntimeError, ValueError):
            invalid.add((part_index, contour_index))
            continue
        if not bool(body.diagnostics.get("manufacturing_ok", False)):
            invalid.add((part_index, contour_index))
    return invalid


def _normalise_sparsification_mode(value: object) -> str:
    key = str(value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    return "pairwise" if key in {"pairwise", "pairs", "pair"} else "off"


def _normalise_sparsification_pruning_mode(value: object) -> str:
    key = str(value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    return "residual_budget" if key in {"residual_budget", "backward_elimination"} else "off"


def _normalise_post_pruning_merge_mode(value: object) -> str:
    key = str(value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    return "retained_adjacent" if key in {"retained_adjacent", "retained_neighbors"} else "off"


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
    *,
    require_original_adjacency: bool = True,
) -> bool:
    contour_a = part.contour_lines[first]
    contour_b = part.contour_lines[second]
    max_multiplicity = max(int(getattr(config, "contour_sparsification_max_turn_multiplicity", 2)), 2)
    if contour_a.turn_multiplicity + contour_b.turn_multiplicity > max_multiplicity:
        return False
    if require_original_adjacency:
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
    proposals = _proposal_options_for_pair(
        part,
        part_index,
        first,
        second,
        target_field,
        config,
        field_cache,
        source_members,
        source_snapshot,
        pass_index=pass_index,
    )
    return proposals[0] if proposals else None


def _proposal_options_for_pair(
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
) -> list[_MergeProposal]:
    """Return deterministic field-ranked representatives for one source pair."""

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
        candidate_field = float(total_multiplicity) * _unit_contour_field(part.mesh, candidate, target_field, config)
        residual = candidate_field - source_field
        rms_error, peak_error = _field_error_values(residual, target_field, config)
        if any(
            np.isclose(candidate.potential, existing[2].potential, rtol=1.0e-12, atol=1.0e-15)
            for existing in candidates
        ):
            continue
        candidates.append((rms_error, peak_error, candidate, candidate_field))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1], abs(item[2].potential - weighted_level)))
    return [
        _MergeProposal(
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
        for rms_error, peak_error, candidate, candidate_field in candidates
    ]


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


def _sparsification_source_indices_by_key(
    parts: list[CoilPart],
    sparsified_part_indices: list[int],
) -> dict[tuple[int, int], tuple[int, ...]]:
    mapping: dict[tuple[int, int], tuple[int, ...]] = {}
    for part_index in sparsified_part_indices:
        part = parts[part_index]
        provenance = part.contour_sparsification
        if provenance is None:
            continue
        if len(provenance.final_source_contour_indices) != len(part.contour_lines):
            raise ValueError("Sparsification provenance must align with active representative contours before pruning.")
        for contour_index, source_indices in enumerate(provenance.final_source_contour_indices):
            mapping[(part_index, contour_index)] = tuple(int(index) for index in source_indices)
    return mapping


def _sparsification_source_reference_field(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
) -> np.ndarray:
    fields = []
    for part in parts:
        provenance = part.contour_sparsification
        contours = provenance.source_contours if provenance is not None else part.contour_lines
        fields.extend(
            float(contour.turn_multiplicity) * _unit_contour_field(part.mesh, contour, target_field, config)
            for contour in contours
        )
    return _sum_fields(fields, len(target_field.coords))


def _group_significance_percent(
    keys: tuple[tuple[int, int], ...],
    field_by_key: dict[tuple[int, int], np.ndarray],
    active_field: np.ndarray,
    metric_context: dict[str, Any],
) -> float:
    mask = np.asarray(metric_context["mask"], dtype=bool).reshape(-1)
    active_values = field_component_values(
        active_field,
        metric_context["component"],
        b0_direction=metric_context["b0_direction"],
    ).reshape(-1)
    finite = mask & np.isfinite(active_values)
    if not np.any(finite):
        return float("inf")
    scale = max(float(np.mean(np.abs(active_values[finite]))), 1.0e-30)
    significance = 0.0
    for key in keys:
        values = field_component_values(
            field_by_key[key],
            metric_context["component"],
            b0_direction=metric_context["b0_direction"],
        ).reshape(-1)
        finite_values = finite & np.isfinite(values)
        if np.any(finite_values):
            significance = max(significance, float(np.max(np.abs(values[finite_values]))) / scale * 100.0)
    return float(significance)


def _apply_sparsified_pruning_result(
    parts: list[CoilPart],
    keep_keys: set[tuple[int, int]],
    field_by_key: dict[tuple[int, int], np.ndarray],
    final_field: np.ndarray,
    metric_context: dict[str, Any],
    source_indices_by_key: dict[tuple[int, int], tuple[int, ...]],
    removal_records: list[dict[str, Any]],
    *,
    mode: str,
    baseline_metric: float,
    selected_metric: float,
    best_metric: float,
    allowed_metric: float,
    source_baseline_metric: float,
    source_baseline_current: float,
    baseline_current: float,
    selected_current: float,
    final_total_rms: float,
    final_total_peak: float,
    final_incremental_rms: float,
    final_incremental_peak: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_removed: int,
    target_geometry_count: int,
    preserve_symmetry: bool,
    symmetry_complete: bool,
    protected_key_count: int,
    search_diagnostics: dict[str, Any],
    subset_symmetry_validation: dict[str, Any],
) -> None:
    sparsified_keys = set(source_indices_by_key)
    removed_keys = sparsified_keys - keep_keys
    pre_geometry_count = len(sparsified_keys)
    pre_turn_count = int(
        sum(
            parts[part_index].contour_lines[contour_index].turn_multiplicity
            for part_index, contour_index in sparsified_keys
        )
    )
    removed_turn_count = int(sum(record["removed_turn_count"] for record in removal_records))
    retained_geometry_count = pre_geometry_count - len(removed_keys)
    retained_turn_count = pre_turn_count - removed_turn_count
    global_diagnostics: dict[str, Any] = {
        "contour_sparsification_pruning_mode": mode,
        "contour_sparsification_pruning_metric": str(metric_context["metric"]),
        "contour_sparsification_pruning_current_source": str(metric_context["source"]),
        "contour_sparsification_pruning_pre_geometry_count": int(pre_geometry_count),
        "contour_sparsification_pruning_retained_geometry_count": int(retained_geometry_count),
        "contour_sparsification_pruning_removed_geometry_count": int(len(removed_keys)),
        "contour_sparsification_pruning_pre_turn_count": int(pre_turn_count),
        "contour_sparsification_pruning_retained_turn_count": int(retained_turn_count),
        "contour_sparsification_pruning_removed_turn_count": int(removed_turn_count),
        "contour_sparsification_pruning_baseline_metric_T": float(baseline_metric),
        "contour_sparsification_pruning_best_metric_T": float(best_metric),
        "contour_sparsification_pruning_selected_metric_T": float(selected_metric),
        "contour_sparsification_pruning_metric_change_T": float(selected_metric - baseline_metric),
        "contour_sparsification_pruning_allowed_metric_T": float(allowed_metric),
        "contour_sparsification_residual_baseline_metric_T": float(source_baseline_metric),
        "contour_sparsification_residual_baseline_current_A": float(source_baseline_current),
        "contour_sparsification_residual_allowed_metric_T": float(allowed_metric),
        "contour_sparsification_pruning_residual_relative_tolerance": float(relative_tolerance),
        "contour_sparsification_pruning_residual_absolute_tolerance_T": float(absolute_tolerance),
        "contour_sparsification_pruning_baseline_current_A": float(baseline_current),
        "contour_sparsification_pruning_selected_current_A": float(selected_current),
        "contour_sparsification_pruning_max_removed_geometries": int(maximum_removed),
        "contour_sparsification_pruning_target_geometry_count": int(target_geometry_count),
        "contour_sparsification_pruning_target_reached": bool(
            target_geometry_count == 0 or retained_geometry_count == target_geometry_count
        ),
        "contour_sparsification_pruning_preserve_symmetry": bool(preserve_symmetry),
        "contour_sparsification_pruning_symmetry_grouping_complete": bool(symmetry_complete),
        "contour_sparsification_pruning_protected_geometry_count": int(protected_key_count),
        "contour_sparsification_pruning_search_mode": str(search_diagnostics["search_mode"]),
        "contour_sparsification_pruning_requested_search_mode": str(search_diagnostics["requested_search_mode"]),
        "contour_sparsification_pruning_removable_orbit_count": int(search_diagnostics["removable_orbit_count"]),
        "contour_sparsification_pruning_evaluated_subset_count": int(search_diagnostics["evaluated_subset_count"]),
        "contour_sparsification_pruning_feasible_subset_count": int(search_diagnostics["feasible_subset_count"]),
        "contour_sparsification_pruning_optimality_proven": bool(search_diagnostics["optimality_proven"]),
        "contour_sparsification_pruning_target_count_feasible": bool(search_diagnostics["target_count_feasible"]),
        "contour_sparsification_pruning_selected_removed_orbit_indices": list(
            search_diagnostics.get("selected_removed_orbit_indices", ())
        ),
        "contour_sparsification_pruning_final_symmetry_validation": dict(subset_symmetry_validation),
        "contour_sparsification_pruning_final_symmetry_ok": bool(subset_symmetry_validation["ok"]),
        "contour_sparsification_pruning_total_rms_field_error_percent": float(final_total_rms),
        "contour_sparsification_pruning_total_peak_field_error_percent": float(final_total_peak),
        "contour_sparsification_pruning_incremental_rms_field_error_percent": float(final_incremental_rms),
        "contour_sparsification_pruning_incremental_peak_field_error_percent": float(final_incremental_peak),
        "contour_sparsification_pruning_removal_records": list(removal_records),
    }
    final_scale = _field_significance_scale(final_field, metric_context)
    for part_index, part in enumerate(parts):
        provenance = part.contour_sparsification
        if provenance is None:
            continue
        original_contours = list(part.contour_lines)
        retained_indices = [
            contour_index for contour_index in range(len(original_contours)) if (part_index, contour_index) in keep_keys
        ]
        removed_indices = [
            contour_index
            for contour_index in range(len(original_contours))
            if (part_index, contour_index) in removed_keys
        ]
        merge_rms = float(provenance.diagnostics.get("contour_sparsification_global_rms_field_error_percent", 0.0))
        merge_peak = float(provenance.diagnostics.get("contour_sparsification_global_peak_field_error_percent", 0.0))
        part.contour_lines = [original_contours[index] for index in retained_indices]
        provenance.final_source_contour_indices = [
            source_indices_by_key[(part_index, index)] for index in retained_indices
        ]
        part.loop_significance = np.asarray(
            [
                _field_significance_percent(field_by_key[(part_index, index)], metric_context, final_scale)
                for index in retained_indices
            ],
            dtype=float,
        )
        part.combined_loop_length = part.effective_turn_length
        part_diagnostics = dict(global_diagnostics)
        part_diagnostics.update(
            {
                "contour_sparsification_pruning_part_removed_geometry_count": int(len(removed_indices)),
                "contour_sparsification_pruning_part_retained_geometry_count": int(len(retained_indices)),
                "contour_sparsification_merge_rms_field_error_percent": merge_rms,
                "contour_sparsification_merge_peak_field_error_percent": merge_peak,
                "contour_sparsification_global_rms_field_error_percent": float(final_total_rms),
                "contour_sparsification_global_peak_field_error_percent": float(final_total_peak),
                "contour_sparsification_nominal_geometry_count": int(part.nominal_contour_count),
                "contour_sparsification_physical_turn_count": int(part.physical_turn_count),
                "contour_sparsification_achieved_max_turn_multiplicity": max(
                    (int(contour.turn_multiplicity) for contour in part.contour_lines),
                    default=0,
                ),
                "contour_sparsification_final_turn_multiplicities": tuple(
                    int(contour.turn_multiplicity) for contour in part.contour_lines
                ),
            }
        )
        provenance.diagnostics.update(part_diagnostics)
        part.contour_level_diagnostics.update(part_diagnostics)
        _clear_topology_state(part)


def _field_significance_scale(field: np.ndarray, metric_context: dict[str, Any]) -> float:
    values = field_component_values(
        field,
        metric_context["component"],
        b0_direction=metric_context["b0_direction"],
    ).reshape(-1)
    mask = np.asarray(metric_context["mask"], dtype=bool).reshape(-1) & np.isfinite(values)
    return max(float(np.mean(np.abs(values[mask]))) if np.any(mask) else 0.0, 1.0e-30)


def _field_significance_percent(
    field: np.ndarray,
    metric_context: dict[str, Any],
    scale: float,
) -> float:
    values = field_component_values(
        field,
        metric_context["component"],
        b0_direction=metric_context["b0_direction"],
    ).reshape(-1)
    mask = np.asarray(metric_context["mask"], dtype=bool).reshape(-1) & np.isfinite(values)
    return float(np.max(np.abs(values[mask])) / scale * 100.0) if np.any(mask) else float("inf")


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
    *,
    stage: str = "pre_pruning",
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
            stage=str(stage),
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
    baseline_residual_metric: float,
    allowed_residual_metric: float,
    final_residual_metric: float,
    baseline_residual_current: float,
    final_residual_current: float,
    residual_tolerances: tuple[float, float] | None,
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
            "contour_sparsification_residual_budget_enabled": residual_tolerances is not None,
            "contour_sparsification_residual_baseline_metric_T": float(baseline_residual_metric),
            "contour_sparsification_residual_allowed_metric_T": float(allowed_residual_metric),
            "contour_sparsification_merge_selected_residual_metric_T": float(final_residual_metric),
            "contour_sparsification_merge_residual_metric_change_T": float(
                final_residual_metric - baseline_residual_metric
            ),
            "contour_sparsification_residual_baseline_current_A": float(baseline_residual_current),
            "contour_sparsification_merge_selected_current_A": float(final_residual_current),
            "contour_sparsification_residual_relative_tolerance": float(
                residual_tolerances[0] if residual_tolerances is not None else np.nan
            ),
            "contour_sparsification_residual_absolute_tolerance_T": float(
                residual_tolerances[1] if residual_tolerances is not None else np.nan
            ),
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


def _sparsification_residual_tolerances(config: CoilGenConfig) -> tuple[float, float] | None:
    relative = getattr(config, "contour_sparsification_residual_relative_tolerance", None)
    absolute = getattr(config, "contour_sparsification_residual_absolute_tolerance_T", None)
    if relative is None and absolute is None:
        return None
    return (
        max(float(relative or 0.0), 0.0),
        max(float(absolute or 0.0), 0.0),
    )


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
