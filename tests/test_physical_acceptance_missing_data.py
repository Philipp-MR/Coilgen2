from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.evaluation import layout_target_residual_diagnostics
from coilgen.hardware_compiler import HardwareSpec, evaluate_discrete_hardware


def _target() -> SimpleNamespace:
    return SimpleNamespace(
        coords=np.zeros((2, 3)),
        b=np.zeros((2, 3)),
        weights=np.ones(2),
        suppressed=np.zeros(2, dtype=bool),
    )


@pytest.mark.parametrize("source", ["layout", "target"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_layout_acceptance_requires_every_active_field_sample(source: str, invalid: float) -> None:
    target = _target()
    layout = np.zeros((2, 3))
    (layout if source == "layout" else target.b)[1, 2] = invalid
    result = layout_target_residual_diagnostics(
        layout,
        target,
        CoilGenConfig(target_error_constraint_metric="rms", layout_target_error_rms_limit_T=1.0e-6),
    )

    assert result["layout_target_checked"]
    assert not result["layout_target_met"]
    assert result["layout_target_sample_count"] == 1
    assert result["layout_target_invalid_sample_count"] == 1


def test_inactive_field_sample_does_not_invalidate_layout_acceptance() -> None:
    target = _target()
    target.weights[1] = 0.0
    layout = np.zeros((2, 3))
    layout[1, 2] = np.nan
    result = layout_target_residual_diagnostics(
        layout,
        target,
        CoilGenConfig(target_error_constraint_metric="rms", layout_target_error_rms_limit_T=1.0e-6),
    )

    assert result["layout_target_met"]
    assert result["layout_target_invalid_sample_count"] == 0


def _hardware_result(parts: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        resistance=None,
        coil_lengths=None,
        coil_parts=parts,
        input_data=CoilGenConfig(),
        target_field=_target(),
        field_by_layout=np.zeros((2, 3)),
    )


def test_required_hardware_measurements_cannot_pass_when_unavailable() -> None:
    part = SimpleNamespace(
        coil_resistance=np.nan,
        coil_length=np.nan,
        physical_turn_count=1,
        contour_sparsification=None,
        minimal_contour_distance=np.nan,
        wire_path=None,
        contour_lines=[],
    )
    metrics = evaluate_discrete_hardware(
        _hardware_result([part]),
        HardwareSpec(wire_current_A=1.0, max_power_W=1.0, min_wire_spacing_m=0.001, min_bend_radius_m=0.01),
    )

    assert not metrics.hard_constraints_satisfied
    assert set(metrics.violations) == {"power_unavailable", "wire_spacing_unavailable", "bend_radius_unavailable"}


def test_empty_hardware_result_has_no_physical_realization() -> None:
    metrics = evaluate_discrete_hardware(_hardware_result([]), HardwareSpec(wire_current_A=1.0))

    assert not metrics.hard_constraints_satisfied
    assert "physical_turn_realization" in metrics.violations
