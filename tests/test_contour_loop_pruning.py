from __future__ import annotations

from unittest.mock import patch

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.contours import (
    _single_mesh_contour_pruning_groups,
    filter_loop_significance,
)
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField


def _target(values: np.ndarray) -> TargetField:
    values = np.asarray(values, dtype=float)
    coords = np.column_stack((np.arange(len(values), dtype=float), np.zeros((len(values), 2))))
    field = np.zeros((len(values), 3), dtype=float)
    field[:, 2] = values
    return TargetField(
        coords=coords,
        b=field,
        weights=np.ones(len(values), dtype=float),
        suppressed=np.zeros(len(values), dtype=bool),
        target_field_group_inds=np.ones(len(values), dtype=np.int64),
    )


def _part(loop_count: int) -> CoilPart:
    mesh = Mesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    contours = [
        ContourLine(
            v=np.asarray([[float(index), 0.0, 0.0], [float(index), 1.0, 0.0]]),
            potential=float(index),
        )
        for index in range(loop_count)
    ]
    return CoilPart(mesh=mesh, contour_step=1.0, contour_lines=contours)


def _config(*, absolute_tolerance: float, maximum_removed: int = 0) -> CoilGenConfig:
    return CoilGenConfig(
        min_loop_signifcance=0.0,
        contour_loop_pruning_mode="residual_budget",
        contour_loop_pruning_residual_absolute_tolerance_T=absolute_tolerance,
        contour_loop_pruning_max_removed_loops=maximum_removed,
        contour_loop_pruning_preserve_symmetry=False,
        target_error_constraint_metric="std",
        field_evaluation_current_source="stream_function",
        target_field_component="Bz",
        parallel_worker_count=1,
        progress_flag=False,
    )


def _field_stub(fields: list[np.ndarray]):
    def evaluate(path: np.ndarray, target_coords: np.ndarray, **_kwargs) -> np.ndarray:
        index = int(round(float(path[0, 0])))
        result = np.zeros((len(target_coords), 3), dtype=float)
        result[:, 2] = fields[index]
        return result

    return evaluate


def test_residual_budget_pruning_removes_only_a_proven_harmful_loop() -> None:
    scale = 1.0e-6
    fields = [
        scale * np.asarray([1.0, -1.0, 0.0, 0.0]),
        scale * np.asarray([0.0, 0.0, 1.0, -1.0]),
        scale * np.asarray([0.05, -0.05, -0.05, 0.05]),
    ]
    target = _target(fields[0] + fields[1])
    part = _part(3)

    with patch("coilgen.contours.biot_savart_wire_path", side_effect=_field_stub(fields)):
        filter_loop_significance([part], target, _config(absolute_tolerance=0.0))

    assert [contour.potential for contour in part.contour_lines] == [0.0, 1.0]
    diagnostics = part.contour_level_diagnostics
    assert diagnostics["contour_loop_pruning_removed_loop_count"] == 1
    assert diagnostics["contour_loop_pruning_selected_metric_T"] < diagnostics["contour_loop_pruning_baseline_metric_T"]


def test_legacy_significance_filter_remains_active_without_a_symmetry_candidate() -> None:
    fields = [
        np.asarray([1.0, -1.0, 1.0, -1.0]) * 1.0e-6,
        np.asarray([1.0, -1.0, 1.0, -1.0]) * 1.0e-10,
    ]
    part = _part(2)
    config = CoilGenConfig(
        min_loop_signifcance=1.0,
        contour_loop_pruning_mode="off",
        parallel_worker_count=1,
        progress_flag=False,
    )

    with patch("coilgen.contours.biot_savart_wire_path", side_effect=_field_stub(fields)):
        filter_loop_significance([part], _target(fields[0]), config)

    assert [contour.potential for contour in part.contour_lines] == [0.0]


def test_residual_budget_is_a_hard_aggregate_limit() -> None:
    scale = 1.0e-6
    fields = [
        scale * np.asarray([1.0, -1.0, 0.0, 0.0]),
        scale * np.asarray([0.0, 0.0, 1.0, -1.0]),
        scale * np.asarray([0.05, -0.05, -0.05, 0.05]),
    ]
    target = _target(fields[0] + fields[1] + fields[2])

    strict_part = _part(3)
    with patch("coilgen.contours.biot_savart_wire_path", side_effect=_field_stub(fields)):
        filter_loop_significance(
            [strict_part],
            target,
            _config(absolute_tolerance=0.01 * scale, maximum_removed=1),
        )
    assert len(strict_part.contour_lines) == 3

    tolerant_part = _part(3)
    with patch("coilgen.contours.biot_savart_wire_path", side_effect=_field_stub(fields)):
        filter_loop_significance(
            [tolerant_part],
            target,
            _config(absolute_tolerance=0.06 * scale, maximum_removed=1),
        )
    assert len(tolerant_part.contour_lines) == 2
    diagnostics = tolerant_part.contour_level_diagnostics
    assert diagnostics["contour_loop_pruning_selected_metric_T"] <= diagnostics["contour_loop_pruning_allowed_metric_T"]


def test_single_mesh_pruning_groups_mirror_partners() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 33)
    right = np.column_stack((np.ones_like(angles), np.cos(angles), np.sin(angles)))
    left = right.copy()
    left[:, 0] *= -1.0
    mesh = Mesh(
        vertices=np.asarray([[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
        faces=np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64),
    )
    part = CoilPart(
        mesh=mesh,
        contour_step=1.0,
        contour_lines=[ContourLine(v=right, potential=0.0), ContourLine(v=left, potential=0.0)],
    )
    part.contour_level_diagnostics.update(
        {"symmetry_stream_function_sign": 1, "symmetry_stream_function_offset_A": 0.0}
    )
    config = CoilGenConfig(
        symmetry_reduction_source="single_mesh",
        contour_symmetry_validation="strict",
        symmetry_reduction_mirror_plane_point=(0.0, 0.0, 0.0),
        symmetry_reduction_mirror_plane_normal=(1.0, 0.0, 0.0),
        contour_symmetry_geometry_tolerance=1.0e-6,
    )

    groups, protected, complete = _single_mesh_contour_pruning_groups(part, config)

    assert complete
    assert not protected
    assert groups == [[(0, 0), (0, 1)]]
