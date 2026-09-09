from __future__ import annotations

import numpy as np
import pytest

from coilgen.cohort_shim import (
    ChannelCurrentLimits,
    CohortFitOptions,
    RigidTransform,
    ShimScenario,
    fit_scenario_currents,
)


def _scenario(disturbance: np.ndarray) -> ShimScenario:
    return ShimScenario(
        scenario_id="physical_scale",
        subject_id="subject",
        coords_scanner_m=np.column_stack((np.arange(len(disturbance)) * 0.01, np.zeros((len(disturbance), 2)))),
        disturbance_T=disturbance,
        scanner_from_mask=RigidTransform("mask_cad", "scanner", np.eye(4)),
    )


@pytest.mark.parametrize("field_scale", [1.0, 1.0e-6, 1.0e-9])
def test_bounded_fit_is_invariant_to_field_units(field_scale: float) -> None:
    result = fit_scenario_currents(
        _scenario(np.array([3.0, 0.0]) * field_scale),
        np.array([[1.0], [0.0]]) * field_scale,
        ChannelCurrentLimits.symmetric(1, 2.0),
        options=CohortFitOptions(center_frequency=False),
    )

    np.testing.assert_allclose(result.channel_currents_A, [-2.0], atol=1.0e-8)
    assert result.rms_after_T == pytest.approx(field_scale / np.sqrt(2.0))
    assert result.saturated_channels.tolist() == [True]


@pytest.mark.parametrize("field_scale", [1.0, 1.0e-6, 1.0e-9])
def test_power_limited_fit_is_invariant_to_field_units(field_scale: float) -> None:
    pytest.importorskip("cvxpy")
    result = fit_scenario_currents(
        _scenario(np.array([2.0, 2.0]) * field_scale),
        np.eye(2) * field_scale,
        ChannelCurrentLimits.symmetric(2, 2.0, resistance_ohm=(1.0, 1.0), total_power_limit_W=1.0),
        options=CohortFitOptions(center_frequency=False),
    )

    np.testing.assert_allclose(result.channel_currents_A, -np.sqrt(0.5) * np.ones(2), atol=2.0e-4)
    assert result.power_W == pytest.approx(1.0, abs=1.0e-6)


@pytest.mark.parametrize("fixed_current", [0.0, 0.5])
def test_fixed_channel_contribution_is_retained_in_fit(fixed_current: float) -> None:
    fields = np.array([[1.0, 1.0], [1.0, -1.0]]) * 1.0e-6
    expected_currents = np.array([fixed_current, -1.0])
    result = fit_scenario_currents(
        _scenario(-(fields @ expected_currents)),
        fields,
        ChannelCurrentLimits(np.array([fixed_current, -2.0]), np.array([fixed_current, 2.0])),
        options=CohortFitOptions(center_frequency=False),
    )

    np.testing.assert_allclose(result.channel_currents_A, expected_currents, atol=1.0e-10)
    np.testing.assert_allclose(result.residual_T, 0.0, atol=1.0e-18)


def test_all_channels_can_be_fixed() -> None:
    currents = np.array([0.0, 0.5])
    result = fit_scenario_currents(
        _scenario(np.array([2.0, 3.0]) * 1.0e-6),
        np.eye(2) * 1.0e-6,
        ChannelCurrentLimits(currents, currents),
        options=CohortFitOptions(center_frequency=False),
    )

    np.testing.assert_array_equal(result.channel_currents_A, currents)
    np.testing.assert_allclose(result.residual_T, np.array([2.0, 3.5]) * 1.0e-6, atol=1.0e-18)
