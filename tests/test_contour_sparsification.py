from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.config_validation import validate_config
from coilgen.contour_sparsification import (
    consolidate_pruned_contours,
    prune_sparsified_contours,
    sparsify_contours,
)
from coilgen.contours import clean_contour_line, extract_contours, reverse_contour_line
from coilgen.evaluation import calculate_unconnected_contour_field
from coilgen.field_maps import field_error_scalar_maps, field_scalar_maps
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.matplotlib_visualization import build_result_plot_specs, plot_error_metrics_matplotlib
from coilgen.models import CoilPart, ContourLine, ContourSparsificationResult, Mesh, TargetField
from coilgen.visualization import save_contours_isometric


def _target() -> TargetField:
    coords = np.asarray([[0.0, 0.0, z] for z in np.linspace(-0.05, 0.05, 9)], dtype=float)
    return TargetField(
        coords=coords,
        b=np.zeros((len(coords), 3), dtype=float),
        weights=np.ones(len(coords), dtype=float),
        suppressed=np.zeros(len(coords), dtype=bool),
        target_field_group_inds=np.arange(len(coords), dtype=np.int64),
    )


def _ring_part(level_count: int = 8) -> CoilPart:
    mesh = normalize_mesh(build_cylinder_mesh(0.4, 0.2, 24, 8, 1, 0, 0, 0))
    part = CoilPart(mesh=mesh)
    part.stream_function = mesh.vertices[:, 2].copy()
    part.potential_level_list = np.linspace(-0.12, 0.12, level_count)
    part.contour_step = float(np.diff(part.potential_level_list).mean())
    extract_contours([part], CoilGenConfig(min_point_loop_number=8, contour_symmetry_validation="off"))
    assert len(part.contour_lines) == level_count
    return part


def _sparsification_config(**overrides: object) -> CoilGenConfig:
    values: dict[str, object] = {
        "min_point_loop_number": 8,
        "skip_postprocessing": True,
        "contour_sparsification_mode": "pairwise",
        "contour_sparsification_max_rms_field_error_percent": 1.0,
        "contour_sparsification_max_peak_field_error_percent": 5.0,
        "contour_sparsification_preserve_symmetry": False,
        "contour_symmetry_validation": "off",
        "topological_grouping_method": "parallel_surface",
    }
    values.update(overrides)
    return CoilGenConfig(**values)


def _sparsified_plot_result() -> SimpleNamespace:
    part = _ring_part()
    config = _sparsification_config(
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
    )
    target = _target()
    sparsify_contours([part], target, config)
    field = calculate_unconnected_contour_field([part], config, target)
    return SimpleNamespace(
        coil_parts=[part],
        input_data=config,
        target_field=target,
        target_field_1A=target,
        field_by_layout=field,
        field_by_unconnected_loops=field.copy(),
        field_layout_per1Amp=field,
        field_loops_per1Amp=field.copy(),
        b_field_opt_sf=np.zeros_like(field),
        b_field_opt_sf_1A=np.zeros_like(field),
        error_vals={
            "max_rel_error_layout_vs_target": 1.0,
            "max_rel_error_unconnected_contours_vs_target": 1.0,
            "max_rel_error_layout_vs_stream_function_field": 2.0,
            "max_rel_error_unconnected_contours_vs_stream_function_field": 2.0,
        },
    )


def _mock_field_contour(
    bz: np.ndarray,
    *,
    length: float,
    potential: float,
    multiplicity: int = 1,
) -> ContourLine:
    contour = ContourLine(
        v=np.asarray([[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=float),
        potential=potential,
        turn_multiplicity=multiplicity,
    )
    contour._test_unit_field = np.column_stack((np.zeros((len(bz), 2)), np.asarray(bz, dtype=float)))
    return contour


def _post_pruning_problem(
    representative_specs: list[tuple[np.ndarray, float, int]],
    target_bz: np.ndarray,
) -> tuple[CoilPart, TargetField]:
    representatives: list[ContourLine] = []
    source_contours: list[ContourLine] = []
    source_members: list[tuple[int, ...]] = []
    for representative_index, (field_bz, length, multiplicity) in enumerate(representative_specs):
        unit_bz = np.asarray(field_bz, dtype=float) / float(multiplicity)
        representatives.append(
            _mock_field_contour(
                unit_bz,
                length=length,
                potential=float(representative_index),
                multiplicity=multiplicity,
            )
        )
        member_indices: list[int] = []
        for _ in range(multiplicity):
            member_indices.append(len(source_contours))
            source_contours.append(
                _mock_field_contour(
                    unit_bz,
                    length=length,
                    potential=float(representative_index),
                )
            )
        source_members.append(tuple(member_indices))
    mesh = Mesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    part = CoilPart(mesh=mesh, contour_lines=representatives)
    part.contour_step = 1.0
    part.contour_sparsification = ContourSparsificationResult(
        source_contours=source_contours,
        final_source_contour_indices=source_members,
        diagnostics={
            "contour_sparsification_global_rms_field_error_percent": 0.0,
            "contour_sparsification_global_peak_field_error_percent": 0.0,
        },
    )
    coords = np.column_stack((np.arange(len(target_bz), dtype=float), np.zeros((len(target_bz), 2))))
    target = TargetField(
        coords=coords,
        b=np.column_stack((np.zeros((len(target_bz), 2)), np.asarray(target_bz, dtype=float))),
        weights=np.ones(len(target_bz), dtype=float),
        suppressed=np.zeros(len(target_bz), dtype=bool),
        target_field_group_inds=np.arange(len(target_bz), dtype=np.int64),
    )
    return part, target


def _post_pruning_config(**overrides: object) -> CoilGenConfig:
    values: dict[str, object] = {
        "skip_postprocessing": True,
        "contour_sparsification_mode": "pairwise",
        "contour_sparsification_max_rms_field_error_percent": 5.0,
        "contour_sparsification_max_peak_field_error_percent": 10.0,
        "contour_sparsification_pruning_mode": "residual_budget",
        "contour_sparsification_pruning_residual_relative_tolerance": 0.0,
        "contour_sparsification_pruning_residual_absolute_tolerance_T": 0.0,
        "contour_sparsification_pruning_preserve_symmetry": False,
        "contour_symmetry_validation": "off",
        "field_evaluation_current_source": "stream_function",
        "target_error_constraint_metric": "std",
    }
    values.update(overrides)
    return CoilGenConfig(**values)


def _gapped_retained_ring_problem() -> tuple[CoilPart, TargetField]:
    part = _ring_part(level_count=6)
    source_contours = deepcopy(part.contour_lines)
    retained_source_indices = (3, 5)
    part.contour_lines = [deepcopy(source_contours[index]) for index in retained_source_indices]
    part.contour_sparsification = ContourSparsificationResult(
        source_contours=source_contours,
        final_source_contour_indices=[(index,) for index in retained_source_indices],
        diagnostics={
            "contour_sparsification_global_rms_field_error_percent": 0.0,
            "contour_sparsification_global_peak_field_error_percent": 0.0,
            "contour_sparsification_residual_allowed_metric_T": 1.0e9,
            "contour_sparsification_accepted_merge_count": 0,
            "contour_sparsification_proposed_merge_count": 0,
        },
    )
    return part, _target()


def test_contour_turn_multiplicity_is_positive_integer() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

    assert ContourLine(v=points, potential=0.0, turn_multiplicity=2).turn_multiplicity == 2
    with pytest.raises(ValueError, match="positive integer"):
        ContourLine(v=points, potential=0.0, turn_multiplicity=0)
    with pytest.raises(ValueError, match="positive integer"):
        ContourLine(v=points, potential=0.0, turn_multiplicity=1.5)


def test_contour_transformations_preserve_turn_multiplicity() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    contour = ContourLine(v=points, potential=0.5, turn_multiplicity=3)
    config = CoilGenConfig(min_point_loop_number=3)

    assert reverse_contour_line(contour).turn_multiplicity == 3
    cleaned = clean_contour_line(contour, config)
    assert cleaned is not None
    assert cleaned.turn_multiplicity == 3


def test_unconnected_contour_field_applies_turn_multiplicity() -> None:
    mesh = Mesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    contour = ContourLine(
        v=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        potential=0.0,
        turn_multiplicity=2,
    )
    part = CoilPart(mesh=mesh, contour_lines=[contour])
    expected_unit_field = np.ones((len(_target().coords), 3), dtype=float)

    with patch("coilgen.evaluation.biot_savart_wire_path", return_value=expected_unit_field):
        field = calculate_unconnected_contour_field([part], CoilGenConfig(), _target())

    np.testing.assert_allclose(field, 2.0 * expected_unit_field)


def test_pairwise_sparsification_replaces_eight_rings_by_four_double_turn_contours() -> None:
    part = _ring_part()

    sparsify_contours([part], _target(), _sparsification_config())

    assert part.nominal_contour_count == 4
    assert part.physical_turn_count == 8
    assert [contour.turn_multiplicity for contour in part.contour_lines] == [2, 2, 2, 2]
    result = part.contour_sparsification
    assert result is not None
    assert len(result.source_contours) == 8
    assert len(result.merge_records) == 4
    assert all(record.accepted for record in result.merge_records)
    assert result.diagnostics["contour_sparsification_global_rms_field_error_percent"] <= 1.0
    assert result.diagnostics["contour_sparsification_physical_realization_available"] is False
    assert part.wire_path is None


def test_iterative_sparsification_replaces_eight_rings_by_one_eight_turn_contour() -> None:
    part = _ring_part()
    config = _sparsification_config(
        contour_sparsification_max_passes=3,
        contour_sparsification_max_turn_multiplicity=8,
        contour_sparsification_field_fit_candidate_count=9,
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
    )

    sparsify_contours([part], _target(), config)

    assert part.nominal_contour_count == 1
    assert part.physical_turn_count == 8
    assert [contour.turn_multiplicity for contour in part.contour_lines] == [8]
    result = part.contour_sparsification
    assert result is not None
    assert len(result.merge_records) == 7
    assert [
        sum(record.accepted for record in result.merge_records if record.pass_index == pass_index)
        for pass_index in (1, 2, 3)
    ] == [4, 2, 1]
    assert result.final_source_contour_indices == [tuple(range(8))]
    assert result.diagnostics["contour_sparsification_completed_pass_count"] == 3
    assert result.diagnostics["contour_sparsification_accepted_merges_by_pass"] == (4, 2, 1)
    assert result.diagnostics["contour_sparsification_achieved_max_turn_multiplicity"] == 8


def test_iterative_sparsification_respects_turn_multiplicity_cap() -> None:
    part = _ring_part()
    config = _sparsification_config(
        contour_sparsification_max_passes=5,
        contour_sparsification_max_turn_multiplicity=4,
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
    )

    sparsify_contours([part], _target(), config)

    assert part.nominal_contour_count == 2
    assert part.physical_turn_count == 8
    assert [contour.turn_multiplicity for contour in part.contour_lines] == [4, 4]
    result = part.contour_sparsification
    assert result is not None
    assert result.diagnostics["contour_sparsification_completed_pass_count"] == 2


def test_iterative_sparsification_uses_one_cumulative_original_field_budget() -> None:
    part = _ring_part()
    target = _target()
    config = _sparsification_config(
        contour_sparsification_max_passes=3,
        contour_sparsification_max_turn_multiplicity=8,
        contour_sparsification_max_rms_field_error_percent=2.0,
        contour_sparsification_max_peak_field_error_percent=10.0,
    )
    baseline_field = calculate_unconnected_contour_field([part], config, target)

    sparsify_contours([part], target, config)

    final_field = calculate_unconnected_contour_field([part], config, target)
    baseline_bz = baseline_field[:, 2]
    final_bz = final_field[:, 2]
    rms_percent = 100.0 * np.sqrt(np.mean(np.square(final_bz - baseline_bz))) / np.sqrt(np.mean(np.square(baseline_bz)))
    peak_percent = 100.0 * np.max(np.abs(final_bz - baseline_bz)) / np.max(np.abs(baseline_bz))
    result = part.contour_sparsification
    assert result is not None
    assert rms_percent == pytest.approx(result.diagnostics["contour_sparsification_global_rms_field_error_percent"])
    assert peak_percent == pytest.approx(result.diagnostics["contour_sparsification_global_peak_field_error_percent"])
    assert rms_percent <= 2.0
    assert peak_percent <= 10.0
    assert part.physical_turn_count == 8


def test_maximum_merge_count_applies_across_all_passes() -> None:
    part = _ring_part()
    config = _sparsification_config(
        contour_sparsification_max_passes=3,
        contour_sparsification_max_turn_multiplicity=8,
        contour_sparsification_max_merges=5,
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
    )

    sparsify_contours([part], _target(), config)

    result = part.contour_sparsification
    assert result is not None
    assert sum(record.accepted for record in result.merge_records) == 5
    assert part.nominal_contour_count == 3
    assert part.physical_turn_count == 8


def test_zero_error_budget_keeps_original_contours_and_records_rejections() -> None:
    part = _ring_part()
    config = _sparsification_config(
        contour_sparsification_max_rms_field_error_percent=0.0,
        contour_sparsification_max_peak_field_error_percent=0.0,
    )

    sparsify_contours([part], _target(), config)

    assert part.nominal_contour_count == 8
    assert part.physical_turn_count == 8
    result = part.contour_sparsification
    assert result is not None
    assert len(result.merge_records) == 4
    assert not any(record.accepted for record in result.merge_records)
    assert all(record.rejection_reason == "field_error_budget" for record in result.merge_records)


def test_cumulative_residual_budget_rejects_merge_that_degrades_target_fit(monkeypatch) -> None:
    part = _ring_part(level_count=2)
    target = _target()
    target.b[:, 2] = np.linspace(-1.0, 1.0, len(target.coords))

    def synthetic_field(_mesh, contour, field_target, _config):
        if contour.turn_multiplicity == 1:
            bz = 0.5 * field_target.b[:, 2]
        else:
            bz = np.zeros(len(field_target.coords), dtype=float)
        return np.column_stack((np.zeros((len(bz), 2)), bz))

    monkeypatch.setattr("coilgen.contour_sparsification._unit_contour_field", synthetic_field)
    config = _sparsification_config(
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
        contour_sparsification_residual_relative_tolerance=0.0,
        contour_sparsification_residual_absolute_tolerance_T=0.0,
        field_evaluation_current_source="stream_function",
        target_error_constraint_metric="rms",
    )

    sparsify_contours([part], target, config)

    assert part.nominal_contour_count == 2
    result = part.contour_sparsification
    assert result is not None
    assert result.diagnostics["contour_sparsification_residual_budget_enabled"]
    assert result.diagnostics["contour_sparsification_merge_selected_residual_metric_T"] == pytest.approx(
        result.diagnostics["contour_sparsification_residual_baseline_metric_T"]
    )
    assert all(record.rejection_reason == "residual_budget" for record in result.merge_records)


def test_post_sparsification_pruning_removes_only_residual_neutral_weighted_geometry(monkeypatch) -> None:
    relevant = np.asarray([-1.0, 1.0, -1.0, 1.0])
    irrelevant_offset = np.full(4, 0.01)
    part, target = _post_pruning_problem(
        [(relevant, 2.0, 4), (irrelevant_offset, 0.1, 1)],
        target_bz=relevant,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )

    prune_sparsified_contours([part], target, _post_pruning_config())

    assert part.nominal_contour_count == 1
    assert part.physical_turn_count == 4
    assert part.contour_lines[0].turn_multiplicity == 4
    provenance = part.contour_sparsification
    assert provenance is not None
    assert provenance.final_source_contour_indices == [(0, 1, 2, 3)]
    diagnostics = provenance.diagnostics
    assert diagnostics["contour_sparsification_pruning_removed_geometry_count"] == 1
    assert diagnostics["contour_sparsification_pruning_removed_turn_count"] == 1
    assert (
        diagnostics["contour_sparsification_pruning_selected_metric_T"]
        <= diagnostics["contour_sparsification_pruning_baseline_metric_T"] + 1.0e-12
    )
    assert diagnostics["contour_sparsification_pruning_removal_records"][0]["source_contour_indices"] == [
        {"part_index": 0, "indices": [4]}
    ]


def test_post_sparsification_pruning_retains_small_geometry_that_improves_shim_std(monkeypatch) -> None:
    large_field = np.asarray([-1.0, 1.0, -1.0, 1.0])
    small_but_relevant = np.asarray([-0.1, -0.1, 0.1, 0.1])
    part, target = _post_pruning_problem(
        [(large_field, 2.0, 4), (small_but_relevant, 0.05, 1)],
        target_bz=large_field + small_but_relevant,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )

    prune_sparsified_contours([part], target, _post_pruning_config())

    assert part.nominal_contour_count == 2
    assert part.physical_turn_count == 5
    provenance = part.contour_sparsification
    assert provenance is not None
    assert provenance.final_source_contour_indices == [(0, 1, 2, 3), (4,)]
    assert provenance.diagnostics["contour_sparsification_pruning_removed_geometry_count"] == 0


def test_post_sparsification_pruning_respects_cumulative_source_field_budget(monkeypatch) -> None:
    relevant = np.asarray([-1.0, 1.0, -1.0, 1.0])
    residual_neutral_offset = np.full(4, 0.01)
    part, target = _post_pruning_problem(
        [(relevant, 2.0, 4), (residual_neutral_offset, 0.1, 1)],
        target_bz=relevant,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )
    config = _post_pruning_config(
        contour_sparsification_max_rms_field_error_percent=0.0,
        contour_sparsification_max_peak_field_error_percent=0.0,
    )

    prune_sparsified_contours([part], target, config)

    assert part.nominal_contour_count == 2
    provenance = part.contour_sparsification
    assert provenance is not None
    assert provenance.diagnostics["contour_sparsification_pruning_removed_geometry_count"] == 0


def test_post_sparsification_pruning_removes_complete_symmetry_orbit(monkeypatch) -> None:
    relevant = np.asarray([-1.0, 1.0, -1.0, 1.0])
    offset = np.full(4, 0.005)
    part, target = _post_pruning_problem(
        [(relevant, 2.0, 4), (offset, 0.1, 1), (offset, 0.1, 1)],
        target_bz=relevant,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._contour_loop_pruning_groups",
        lambda _parts, _config, preserve_symmetry: (
            [[(0, 0)], [(0, 1), (0, 2)]],
            set(),
            True,
        ),
    )
    config = _post_pruning_config(contour_sparsification_pruning_preserve_symmetry=True)

    prune_sparsified_contours([part], target, config)

    assert part.nominal_contour_count == 1
    assert part.physical_turn_count == 4
    provenance = part.contour_sparsification
    assert provenance is not None
    assert provenance.final_source_contour_indices == [(0, 1, 2, 3)]
    assert provenance.diagnostics["contour_sparsification_pruning_removed_geometry_count"] == 2
    assert provenance.diagnostics["contour_sparsification_pruning_symmetry_grouping_complete"]


def test_post_sparsification_pruning_prioritizes_nonmanufacturable_geometry(monkeypatch) -> None:
    relevant = np.asarray([-1.0, 1.0, -1.0, 1.0])
    offset = np.full(4, 0.005)
    part, target = _post_pruning_problem(
        [(relevant, 2.0, 4), (offset, 0.1, 1), (offset, 0.2, 1)],
        target_bz=relevant,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._nonmanufacturable_contour_keys",
        lambda _parts, _config, _keys: {(0, 1)},
    )
    config = _post_pruning_config(
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
        contour_sparsification_pruning_max_removed_geometries=1,
    )

    prune_sparsified_contours([part], target, config)

    provenance = part.contour_sparsification
    assert provenance is not None
    assert provenance.final_source_contour_indices == [(0, 1, 2, 3), (5,)]
    diagnostics = provenance.diagnostics
    assert diagnostics["contour_sparsification_pruning_nonmanufacturable_candidate_count"] == 1
    assert diagnostics["contour_sparsification_pruning_retained_nonmanufacturable_count"] == 0
    assert diagnostics["contour_sparsification_pruning_removal_records"][0]["nonmanufacturable_geometry_count"] == 1


def test_post_sparsification_pruning_stops_at_explicit_retained_geometry_target(monkeypatch) -> None:
    relevant = np.asarray([-1.0, 1.0, -1.0, 1.0])
    offset = np.full(4, 0.002)
    part, target = _post_pruning_problem(
        [(relevant, 2.0, 4), (offset, 0.1, 1), (offset, 0.2, 1), (offset, 0.3, 1)],
        target_bz=relevant,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )
    config = _post_pruning_config(
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
        contour_sparsification_pruning_target_geometry_count=2,
    )

    prune_sparsified_contours([part], target, config)

    assert part.nominal_contour_count == 2
    provenance = part.contour_sparsification
    assert provenance is not None
    diagnostics = provenance.diagnostics
    assert diagnostics["contour_sparsification_pruning_target_geometry_count"] == 2
    assert diagnostics["contour_sparsification_pruning_target_reached"]


def test_global_subset_search_can_cross_an_infeasible_single_removal(monkeypatch) -> None:
    relevant = np.asarray([-1.0, 1.0, -1.0, 1.0])
    neutral_offset = np.full(4, 0.01)
    cancelling_a = np.asarray([-0.2, -0.2, 0.2, 0.2])
    cancelling_b = -cancelling_a
    part, target = _post_pruning_problem(
        [
            (relevant, 2.0, 1),
            (neutral_offset, 1.0, 1),
            (cancelling_a, 0.1, 1),
            (cancelling_b, 0.1, 1),
        ],
        target_bz=relevant,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )
    config = _post_pruning_config(
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
        contour_sparsification_pruning_target_geometry_count=2,
        contour_sparsification_pruning_search_mode="exact",
    )

    prune_sparsified_contours([part], target, config)

    assert part.nominal_contour_count == 2
    provenance = part.contour_sparsification
    assert provenance is not None
    assert provenance.final_source_contour_indices == [(0,), (1,)]
    diagnostics = provenance.diagnostics
    assert diagnostics["contour_sparsification_pruning_search_mode"] == "exact"
    assert diagnostics["contour_sparsification_pruning_optimality_proven"]
    assert diagnostics["contour_sparsification_pruning_target_count_feasible"]


def test_odd_target_does_not_split_complete_symmetry_orbit(monkeypatch) -> None:
    relevant_a = np.asarray([-0.5, 0.5, -0.5, 0.5])
    relevant_b = relevant_a.copy()
    cancelling_a = np.asarray([-0.01, -0.01, 0.01, 0.01])
    cancelling_b = -cancelling_a
    part, target = _post_pruning_problem(
        [
            (relevant_a, 1.0, 1),
            (relevant_b, 1.0, 1),
            (cancelling_a, 0.2, 1),
            (cancelling_a, 0.2, 1),
            (cancelling_b, 0.2, 1),
            (cancelling_b, 0.2, 1),
        ],
        target_bz=relevant_a + relevant_b,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._unit_contour_field",
        lambda _mesh, contour, _target, _config: contour._test_unit_field,
    )
    monkeypatch.setattr(
        "coilgen.contour_sparsification._contour_loop_pruning_groups",
        lambda _parts, _config, preserve_symmetry: (
            [[(0, 0), (0, 1)], [(0, 2), (0, 4)], [(0, 3), (0, 5)]],
            set(),
            True,
        ),
    )
    config = _post_pruning_config(
        contour_sparsification_pruning_preserve_symmetry=True,
        contour_sparsification_pruning_target_geometry_count=5,
        contour_sparsification_pruning_search_mode="exact",
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
    )

    prune_sparsified_contours([part], target, config)

    assert part.nominal_contour_count == 6
    provenance = part.contour_sparsification
    assert provenance is not None
    diagnostics = provenance.diagnostics
    assert not diagnostics["contour_sparsification_pruning_target_reached"]
    assert not diagnostics["contour_sparsification_pruning_target_count_feasible"]
    assert diagnostics["contour_sparsification_pruning_final_symmetry_ok"]


def test_post_pruning_merge_combines_retained_neighbors_across_original_level_gap() -> None:
    part, target = _gapped_retained_ring_problem()
    config = _sparsification_config(
        contour_sparsification_pruning_mode="residual_budget",
        contour_sparsification_post_pruning_merge_mode="retained_adjacent",
        contour_sparsification_post_pruning_max_passes=1,
        contour_sparsification_max_turn_multiplicity=8,
        contour_sparsification_field_fit_candidate_count=9,
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
        min_turn_distance=0.0,
        skip_sweep=True,
    )

    consolidate_pruned_contours([part], target, config)

    assert part.nominal_contour_count == 1
    assert part.physical_turn_count == 2
    assert part.contour_lines[0].turn_multiplicity == 2
    provenance = part.contour_sparsification
    assert provenance is not None
    assert provenance.final_source_contour_indices == [(3, 5)]
    assert provenance.merge_records[-1].stage == "post_pruning"
    assert provenance.merge_records[-1].accepted
    diagnostics = provenance.diagnostics
    assert diagnostics["contour_sparsification_post_pruning_accepted_merge_count"] == 1
    assert diagnostics["contour_sparsification_post_pruning_pre_geometry_count"] == 2
    assert diagnostics["contour_sparsification_post_pruning_final_geometry_count"] == 1


def test_post_pruning_merge_is_disabled_by_default() -> None:
    part, target = _gapped_retained_ring_problem()
    config = _sparsification_config(
        contour_sparsification_pruning_mode="residual_budget",
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
    )

    consolidate_pruned_contours([part], target, config)

    assert part.nominal_contour_count == 2
    assert part.contour_sparsification is not None
    assert part.contour_sparsification.final_source_contour_indices == [(3,), (5,)]


def test_post_pruning_merge_respects_cumulative_residual_budget() -> None:
    part, target = _gapped_retained_ring_problem()
    assert part.contour_sparsification is not None
    part.contour_sparsification.diagnostics["contour_sparsification_residual_allowed_metric_T"] = -1.0
    config = _sparsification_config(
        contour_sparsification_pruning_mode="residual_budget",
        contour_sparsification_post_pruning_merge_mode="retained_adjacent",
        contour_sparsification_max_turn_multiplicity=8,
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
        min_turn_distance=0.0,
        skip_sweep=True,
    )

    consolidate_pruned_contours([part], target, config)

    assert part.nominal_contour_count == 2
    records = part.contour_sparsification.diagnostics["contour_sparsification_post_pruning_merge_records"]
    assert records[0]["rejection_reason"] == "residual_budget"


def test_post_pruning_merge_tries_alternate_field_fit_candidate(monkeypatch) -> None:
    part, target = _gapped_retained_ring_problem()
    evaluated_potentials: list[float] = []

    def evaluate_bundle(
        _parts,
        _already_selected,
        bundle,
        accepted_field,
        _source_reference_field,
        _target_field,
        _config,
        _metric_context,
        _allowed_metric,
    ):
        evaluated_potentials.append(float(bundle[0].candidate.potential))
        if len(evaluated_potentials) == 1:
            return {"ok": False, "reason": "manufacturing_validation"}
        candidate_field = accepted_field + sum(
            (proposal.field_delta for proposal in bundle),
            np.zeros_like(accepted_field),
        )
        return {
            "ok": True,
            "reason": "",
            "field": candidate_field,
            "total_rms_error_percent": float(len(evaluated_potentials)),
            "total_peak_error_percent": float(len(evaluated_potentials)),
            "residual_metric_T": 0.0,
            "residual_current_A": 1.0,
            "minimum_spacing_m": float("inf"),
            "required_spacing_m": 0.0,
            "manufacturing_checked": False,
            "manufacturing": [],
        }

    monkeypatch.setattr(
        "coilgen.contour_sparsification._evaluate_post_pruning_bundle",
        evaluate_bundle,
    )
    config = _sparsification_config(
        contour_sparsification_pruning_mode="residual_budget",
        contour_sparsification_post_pruning_merge_mode="retained_adjacent",
        contour_sparsification_max_turn_multiplicity=8,
        contour_sparsification_positioning="field_fit",
        contour_sparsification_field_fit_candidate_count=9,
        contour_sparsification_max_rms_field_error_percent=1.0e9,
        contour_sparsification_max_peak_field_error_percent=1.0e9,
        min_turn_distance=0.0,
        skip_sweep=True,
    )

    consolidate_pruned_contours([part], target, config)

    assert len(evaluated_potentials) > 1
    assert part.nominal_contour_count == 1
    assert part.contour_sparsification is not None
    assert part.contour_sparsification.merge_records[-1].accepted
    assert part.contour_lines[0].potential == pytest.approx(evaluated_potentials[1])


def test_sparsification_requires_ideal_only_postprocessing_contract() -> None:
    report = validate_config(CoilGenConfig(contour_sparsification_mode="pairwise", skip_postprocessing=False))

    assert not report.ok
    assert "skip_postprocessing" in report.format()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("contour_sparsification_max_passes", 0),
        ("contour_sparsification_max_turn_multiplicity", 1),
        ("contour_sparsification_field_fit_candidate_count", 0),
    ),
)
def test_iterative_sparsification_bounds_are_validated(field: str, value: int) -> None:
    report = validate_config(CoilGenConfig(**{field: value}))

    assert not report.ok
    assert field in report.format()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("contour_sparsification_residual_relative_tolerance", -1.0),
        ("contour_sparsification_residual_absolute_tolerance_T", -1.0),
        ("contour_sparsification_pruning_residual_relative_tolerance", -1.0),
        ("contour_sparsification_pruning_residual_absolute_tolerance_T", -1.0),
        ("contour_sparsification_pruning_max_removed_geometries", -1),
        ("contour_sparsification_pruning_target_geometry_count", -1),
        ("contour_sparsification_pruning_exact_max_orbits", -1),
        ("contour_sparsification_pruning_beam_width", 0),
        ("contour_sparsification_post_pruning_max_passes", 0),
    ),
)
def test_post_sparsification_pruning_bounds_are_validated(field: str, value: float | int) -> None:
    report = validate_config(CoilGenConfig(**{field: value}))

    assert not report.ok
    assert field in report.format()


def test_post_sparsification_pruning_requires_active_sparsification() -> None:
    report = validate_config(CoilGenConfig(contour_sparsification_pruning_mode="residual_budget"))

    assert not report.ok
    assert "contour_sparsification_pruning_mode" in report.format()


def test_post_sparsification_pruning_search_mode_is_validated() -> None:
    report = validate_config(CoilGenConfig(contour_sparsification_pruning_search_mode="greedy"))

    assert not report.ok
    assert "contour_sparsification_pruning_search_mode" in report.format()


def test_post_pruning_merge_requires_active_pruning() -> None:
    report = validate_config(
        CoilGenConfig(
            contour_sparsification_mode="pairwise",
            contour_sparsification_post_pruning_merge_mode="retained_adjacent",
        )
    )

    assert not report.ok
    assert "contour_sparsification_post_pruning_merge_mode" in report.format()


def test_svg_contour_plot_reports_nominal_geometries_and_physical_turns(tmp_path) -> None:
    part = _ring_part()
    sparsify_contours([part], _target(), _sparsification_config())
    output = tmp_path / "sparsified_contours.svg"

    save_contours_isometric(part, output, label_contours=True)

    text = output.read_text(encoding="utf-8")
    assert "4 geometries / 8 turns" in text
    assert "x2" in text
    assert 'stroke-dasharray="4 4"' in text


def test_plotbook_uses_final_sparsified_layout_geometry_field_and_labels() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    result = _sparsified_plot_result()
    fields = field_scalar_maps(result)
    errors = field_error_scalar_maps(result)

    np.testing.assert_allclose(fields["layout"].values, result.field_layout_per1Amp[:, 2])
    assert "Final Sparsified" in fields["layout"].label
    assert "Final Sparsified" in errors["layout_vs_target"].label
    assert "Connected Wire" not in errors["layout_vs_stream"].label

    specs = {
        spec.key: spec
        for spec in build_result_plot_specs(
            result,
            include=("wire", "raw_contours", "stream_function", "current_density", "regularization"),
        )
    }
    assert "Final Sparsified Weighted-Contour Layout" in specs["wire"].title
    assert "Original Contours Dashed" in specs["raw_contours"].title
    assert "Pre-Sparsification" in specs["stream_function"].title
    assert "Pre-Sparsification" in specs["current_density"].title
    assert "Pre-Sparsification" in specs["regularization"].title

    figure = specs["wire"].factory()
    try:
        assert "Final Sparsified Weighted-Contour Layout" in figure.axes[0].get_title().replace("\n", " ")
        assert all(line.get_linestyle() != "--" for line in figure.axes[0].lines)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_sparse_error_metric_plot_removes_duplicate_unconnected_field_metrics() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    result = _sparsified_plot_result()

    figure = plot_error_metrics_matplotlib(result, show=False)
    try:
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert any("final sparsified weighted contours" in label for label in labels)
        assert not any("unconnected contours" in label for label in labels)
        assert "final sparsified weighted-contour layout" in figure.axes[0].get_title().lower().replace("\n", " ")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
