"""Example summaries must distinguish skipped circuit estimates from zero."""

from types import SimpleNamespace

import numpy as np
import pytest

from projects._shared.example_utils import print_result_summary


def circuit_result(*, skipped, use_arrays=True):
    return SimpleNamespace(
        input_data=SimpleNamespace(skip_inductance_calculation=skipped, target_gradient_strength=0.02),
        inductance_matrix=np.array([[2e-3, -0.2e-3], [-0.2e-3, 3e-3]]),
        resistance=np.array([0.2, 0.3]) if use_arrays else None,
        coil_lengths=np.array([7.0, 5.0]) if use_arrays else None,
        coil_parts=[
            SimpleNamespace(coil_resistance=0.2, coil_length=7.0, contour_lines=[]),
            SimpleNamespace(coil_resistance=0.3, coil_length=5.0, contour_lines=[]),
        ],
        regularization_diagnostics={"sheet_power_W": 4.5},
    )


@pytest.mark.parametrize("skipped", [True, False])
def test_console_summary_reports_calculation_availability(monkeypatch, capsys, skipped):
    import coilgen

    result = circuit_result(skipped=skipped)
    result.target_field = SimpleNamespace(coords=np.zeros((7, 3)))
    result.potential_step = 1.0
    result.needed_current_layout = 10.0
    result.regularization_diagnostics = {}
    result.error_vals = {}
    monkeypatch.setattr(
        coilgen, "result_quality_status", lambda _r: SimpleNamespace(status="pass", failures=(), warnings=())
    )
    monkeypatch.setattr(coilgen, "result_summary_dict", lambda _r: {"execution_status": "completed"})
    monkeypatch.setattr(coilgen, "symmetry_summary_dict", lambda _r: {})
    monkeypatch.setattr(coilgen, "result_performance_table_rows", lambda _r: [])
    print_result_summary(result)
    output = capsys.readouterr().out
    assert "coil length total: 12 m" in output
    if skipped:
        assert "unavailable (calculation disabled)" in output
        assert "coil resistance part 1" not in output
        assert "coil inductance part 1 self" not in output
        assert "coil inductance matrix [H]" not in output
    else:
        assert "coil resistance total: 0.5 ohm" in output
        assert "coil inductance same-current total: 0.0046 H" in output
