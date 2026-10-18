from __future__ import annotations

import json

import pytest

from coilgen import workflow


@pytest.mark.parametrize("diagnostic_value", [0, "x" * workflow.RUN_RECORD_MAX_STRING_CHARS])
def test_compact_logs_retain_engineering_metrics_before_optional_diagnostics(
    tmp_path, diagnostic_value: object
) -> None:
    critical = {
        "layout_gradient_efficiency_mT_per_m_per_A": 0.35,
        "layout_field_efficiency_mT_per_A": 0.063,
        "unconnected_contours_field_efficiency_mT_per_A": 0.064,
        "stream_function_field_efficiency_mT_per_A": 0.065,
        "layout_needed_current_A": 12.0,
        "coil_resistance_total_ohm": 0.8,
        "coil_inductance_same_current_H": 0.0002,
        "constrained_solver_contract_met": False,
        "constrained_solver_physical_constraints_met": False,
        "constrained_solver_diagnostic_result": True,
        "constrained_solver_failed_constraints": "surface_current_density",
        "layout_target_met": False,
        "shim_layout_residual_rms_T": 1.2e-6,
    }
    metrics = {
        **{f"optional_diagnostic_{index}": diagnostic_value for index in range(600)},
        **critical,
    }
    record = {
        "run": {"status": "success", "quality_status": "FAIL"},
        "metrics": metrics,
        "case_parameters": {f"optional_parameter_{index}": index for index in range(300)},
    }
    path = tmp_path / "run_summary.json"

    workflow._write_run_record_json(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.stat().st_size <= workflow.RUN_RECORD_MAX_SUMMARY_BYTES
    # The existing omitted-items marker is an extra bookkeeping entry.
    assert len(payload["metrics"]) <= workflow.RUN_RECORD_MAX_MAPPING_ITEMS + 1
    assert {key: payload["metrics"][key] for key in critical} == critical
    if diagnostic_value:
        assert payload["run_record_truncation"]["truncated"]
    else:
        assert payload["metrics"]["__omitted_items__"] > 0

    row = workflow._bounded_run_metrics_row(record)
    assert len(row) <= workflow.RUN_RECORD_MAX_CSV_COLUMNS + 1
    assert row["quality_status"] == "FAIL"
    assert row["run_record_omitted_metric_columns"] > 0
    for key, value in critical.items():
        assert row[key] == str(value)
