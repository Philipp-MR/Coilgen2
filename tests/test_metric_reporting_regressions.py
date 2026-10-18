from __future__ import annotations

import csv
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.gradient_linearity import gradient_linearity_summary
from coilgen.optimization import (
    _dsv_constraint_block,
    _residual_constraint_diagnostics,
    _target_error_constraint_blocks,
)
from coilgen.reporting import _constraint_status_summary, write_design_summary_csv
import coilgen.reporting as reporting
from coilgen.visualization import SvgStyle, _save_horizontal_bar_chart, _save_point_scalar_view


@pytest.mark.parametrize("metric, expected", [("std", 1.0e-6), ("rms", np.sqrt(26.0) * 1.0e-6)])
@pytest.mark.parametrize("weight_scale", [0.001, 1.0, 73.0])
def test_physical_weighted_metric_is_invariant_to_global_weight_scale(metric, expected, weight_scale):
    weights = weight_scale * np.array([1.0, 1.0, 100.0, 100.0])
    target = 1.0e-6 * np.array([4.0, 6.0, 4.0, 6.0])
    config = CoilGenConfig(
        target_error_constraint_mode="separate",
        target_error_constraint_metric=metric,
        target_error_std_limit_T=3.0e-6,
        target_error_rms_limit_T=7.0e-6,
        target_error_constraint_acceptance_absolute_T=0.0,
    )
    block = _dsv_constraint_block(np.zeros((4, 1)), target, config, weights=weights)
    problem = SimpleNamespace(residual_norm=lambda _x: float(np.linalg.norm(block.target)))
    diagnostics = _residual_constraint_diagnostics(problem, np.zeros(1), (block,), config)

    assert diagnostics["dsv_constraint_weighting"] == "target_weights"
    assert diagnostics["dsv_constraint_field_metric_T"] == pytest.approx(expected)
    assert diagnostics["target_error_constraint_field_metric_T"] == pytest.approx(expected)
    assert diagnostics["dsv_constraint_met"] is True
    row = diagnostics["target_error_constraint_block_diagnostics"][0]
    assert row["field_metric_T"] / row["allowed_field_metric_T"] == pytest.approx(row["violation_ratio"])
    if metric == "std":
        assert diagnostics["dsv_constraint_std_T"] == pytest.approx(expected)
    # The solver-facing weighted residual and budget retain their original
    # sqrt(sum(weights)) scaling; the fix affects reported physical units.
    assert block.allowed_residual == pytest.approx(block.field_limit_T * np.sqrt(np.sum(weights)))


def test_weighted_std_value_and_effective_acceptance_status_agree():
    config = CoilGenConfig(
        target_error_constraint_mode="separate",
        target_error_constraint_metric="std",
        target_error_std_limit_T=0.8e-6,
        target_error_constraint_acceptance_absolute_T=0.0,
    )
    block = _dsv_constraint_block(
        np.zeros((4, 1)), np.array([-1, 1, -1, 1]) * 1.0e-6, config, weights=np.array([1, 1, 100, 100])
    )
    diagnostics = _residual_constraint_diagnostics(
        SimpleNamespace(residual_norm=lambda _x: 0.0), np.zeros(1), (block,), config
    )
    assert diagnostics["dsv_constraint_std_T"] == pytest.approx(1.0e-6)
    assert diagnostics["dsv_constraint_met"] is False
    assert (
        diagnostics["dsv_constraint_field_metric_T"]
        > diagnostics["dsv_constraint_allowed_with_acceptance_field_metric_T"]
    )


@pytest.mark.parametrize("weight_scale", [0.002, 1.0, 31.0])
def test_auxiliary_dsv_statistics_describe_physical_residuals_not_weight_scaled_rows(weight_scale):
    raw = np.array([-1.0, 2.0, -5.0, 3.0]) * 1.0e-6
    weights = weight_scale * np.array([1.0, 2.0, 100.0, 10.0])
    config = CoilGenConfig(target_error_constraint_mode="separate", target_error_constraint_metric="norm")
    block = _dsv_constraint_block(np.zeros((4, 1)), -raw, config, weights=weights)
    diagnostics = _residual_constraint_diagnostics(
        SimpleNamespace(residual_norm=lambda _x: 0.0), np.zeros(1), (block,), config
    )
    mean = np.dot(weights, raw) / weights.sum()
    assert diagnostics["dsv_constraint_mean_T"] == pytest.approx(mean)
    assert diagnostics["dsv_constraint_std_T"] == pytest.approx(
        np.sqrt(np.dot(weights, (raw - mean) ** 2) / weights.sum())
    )
    assert diagnostics["dsv_constraint_rms_residual_T"] == pytest.approx(
        np.sqrt(np.dot(weights, raw**2) / weights.sum())
    )
    assert diagnostics["dsv_constraint_linf_residual_T"] == pytest.approx(np.max(np.abs(raw)))
    assert diagnostics["dsv_constraint_residual_statistics_scope"] == "weighted_physical_field"
    assert diagnostics["dsv_constraint_field_metric_T"] == pytest.approx(np.linalg.norm(np.sqrt(weights) * raw))


def test_combined_weighted_constraint_labels_actual_nonuniform_weights():
    config = CoilGenConfig(target_error_constraint_mode="combined")
    problem = SimpleNamespace(
        target_z=np.array([1.0, 10.0]), reduced_sensitivity=np.zeros((2, 1)), target_weights=np.array([1.0, 100.0])
    )
    block = _target_error_constraint_blocks(problem, config)[0]
    assert block.name == "combined_weighted"
    assert block.weighting == "target_weights"
    assert block.row_weights is None  # mixed rows do not claim physical auxiliary moments


def _field_fixture(constant=False):
    coords = np.array(np.meshgrid(*([np.linspace(-0.08, 0.08, 3)] * 3), indexing="ij")).reshape(3, -1).T
    target = np.zeros_like(coords)
    target[:, 2] = 1.0e-3 if constant else 0.02 * coords[:, 1]
    field = SimpleNamespace(coords=coords, b=target, suppressed=np.zeros(len(coords), dtype=bool))
    return field


def test_disabled_gradient_requirements_are_not_reported_as_pass():
    field = _field_fixture()
    summary = gradient_linearity_summary(field.b, field, CoilGenConfig(target_gradient_strength=0.02))
    assert summary["linearity_status"] == "not_checked"
    assert summary["linearity_applicable"]
    assert summary["linearity_rms_percent"] == pytest.approx(0, abs=1.0e-10)
    assert summary["linearity_gradient_strength_status"] == "not_checked"
    assert summary["linearity_cross_gradient_status"] == "not_checked"


def test_constant_bz_has_no_relative_gradient_percentage_but_retains_absolute_residual():
    field = _field_fixture(constant=True)
    actual = field.b.copy()
    actual[:, 2] += 2.0e-6 * (field.coords[:, 0] / 0.08) ** 2
    config = CoilGenConfig(target_gradient_strength=0.0)
    summary = gradient_linearity_summary(actual, field, config)
    assert summary["linearity_status"] == "not_applicable"
    assert not summary["linearity_applicable"]
    assert np.isnan(summary["linearity_rms_percent"])
    assert np.isnan(summary["linearity_cross_gradient_percent"])
    assert summary["linearity_rms_residual_T"] > 0.0
    config.linearity_rms_tolerance_percent = 1.0
    assessed = gradient_linearity_summary(actual, field, config)
    assert assessed["linearity_status"] == "FAIL"
    assert not assessed["linearity_met"]
    assert "cannot be assessed" in assessed["linearity_failure_reasons"]


def test_constraint_status_distinguishes_absent_inactive_and_failed_evidence():
    result = SimpleNamespace(input_data=CoilGenConfig(), regularization_diagnostics={})
    statuses = _constraint_status_summary(result)
    assert statuses["target_error_constraint_status"] == "inactive"
    assert statuses["linearity_rms_constraint_status"] == "inactive"
    result.input_data.linearity_rms_tolerance_percent = 1.0
    assert _constraint_status_summary(result)["linearity_rms_constraint_status"] == "unknown"
    result.regularization_diagnostics.update(
        linearity_rms_constraint_met=False, suppression_constraint_enabled=False, suppression_constraint_met=False
    )
    statuses = _constraint_status_summary(result)
    assert statuses["linearity_rms_constraint_status"] == "FAIL"
    assert statuses["suppression_constraint_status"] == "inactive"


def test_quantity_value_csv_round_trips_punctuation_and_multiline_values(monkeypatch, tmp_path):
    summary = {"quantity,quoted": 'Copper, "two turns"\nsecond line', "unicode": "µT, Ω", "number": 0.1}
    monkeypatch.setattr(reporting, "result_summary_dict", lambda _result: summary)
    path = write_design_summary_csv(object(), tmp_path / "summary.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["quantity", "value"]
    assert all(len(row) == 2 for row in rows)
    assert dict(rows[1:]) == {key: str(value) for key, value in summary.items()}


def test_error_point_svg_discloses_sparse_sampling_has_color_scale_and_world_axes(tmp_path):
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [np.nan, 0]], dtype=float)
    values = np.array([-3, -1, 2, 3, 100], dtype=float)
    source_values = values.copy()
    path = _save_point_scalar_view(
        points,
        values,
        tmp_path / "points.svg",
        title="Signed Bz difference",
        style=SvgStyle(),
        scalar_label="signed Bz difference [µT]",
        axis_labels=None,
    )
    document = ET.parse(path)
    texts = [element.text for element in document.iter() if element.tag.endswith("text")]
    assert any("n=4 finite samples" in (text or "") for text in texts)
    assert all(axis in texts for axis in ("x", "y", "z"))
    assert "signed Bz difference [µT]" in texts
    assert all(value in texts for value in ("-3", "0", "3"))
    assert any(element.attrib.get("stroke-dasharray") == "4,4" for element in document.iter())
    np.testing.assert_array_equal(values, source_values)


def test_metric_svg_keeps_complete_wrapped_labels_and_numeric_values_inside_canvas(tmp_path):
    label = "max_rel_error_unconnected_contours_vs_requested_target_in_independent_sample_grid"
    path = _save_horizontal_bar_chart(
        [(label, 123.456), ("short", -2.0)], tmp_path / "bars.svg", title="Field error [%]", style=SvgStyle(width=640)
    )
    root = ET.parse(path).getroot()
    texts = [element for element in root.iter() if element.tag.endswith("text")]
    text = " ".join(element.text or "" for element in texts)
    assert label.replace("_", " ") in text
    assert "123.5" in text
    assert "..." not in text
    assert all(float(element.attrib["x"]) < 640 - 60 for element in texts)
