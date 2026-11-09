"""Results lead with the drive current and the shape error left after it."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.gui.result_summary import FieldResultSummary, drive_metric_cards, summarize_field_result
from coilgen.metrics import current_scale_fit


def _target(n=200, seed=1):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.02, 0.02, (n, 3))
    return 0.02 * points[:, 1], rng


def test_pure_gain_error_has_no_shape_error():
    target, _ = _target()
    fit = current_scale_fit(target / 1.064, target)
    assert fit["best_fit_current_scale"] == pytest.approx(1.064)
    assert fit["best_fit_relative_rms_error"] == pytest.approx(0.0, abs=1e-12)


def test_shape_error_is_measured_after_the_current_fit():
    target, rng = _target()
    noise = rng.normal(0.0, 0.01 * np.sqrt(np.mean(target**2)), len(target))
    fit = current_scale_fit((target + noise) / 1.05, target)
    assert fit["best_fit_current_scale"] == pytest.approx(1.05, rel=2e-3)
    assert fit["best_fit_relative_rms_error"] == pytest.approx(0.01, rel=0.15)


def test_weights_and_vector_rows():
    target, _ = _target()
    field = target / 1.02
    field[:10] = 1.0  # wildly wrong, but zero weight
    weights = np.ones(len(target))
    weights[:10] = 0.0
    assert current_scale_fit(field, target, weights)["best_fit_relative_rms_error"] == pytest.approx(0.0, abs=1e-12)
    vectors = np.column_stack((np.zeros_like(target), np.zeros_like(target), target))
    assert current_scale_fit(vectors / 2.0, vectors)["best_fit_current_scale"] == pytest.approx(2.0)


def test_undefined_fits_are_empty():
    target, _ = _target()
    assert current_scale_fit(-target, target) == {}
    assert current_scale_fit(np.zeros_like(target), target) == {}
    assert current_scale_fit(target, target[:-1]) == {}
    assert current_scale_fit([], []) == {}
    assert current_scale_fit(np.empty((0, 3)), np.empty((0, 3))) == {}
    assert current_scale_fit([1, 1], [1, 0.5], [2, -1]) == {}


@pytest.mark.parametrize("magnitude", [1e-200, 1.0, 1e200])
def test_current_fit_is_invariant_to_field_and_weight_units(magnitude):
    field = magnitude * np.array([1.0, 2.0])
    fit = current_scale_fit(field, 2.0 * field, [1e308, 1e308])
    assert fit["best_fit_current_scale"] == pytest.approx(2.0)
    assert fit["best_fit_relative_rms_error"] == pytest.approx(0.0, abs=1e-12)


def test_zero_weight_samples_do_not_overflow_the_current_fit():
    fit = current_scale_fit([1e200, 1.0], [1.0, 2.0], [0.0, 1.0])
    assert fit["best_fit_current_scale"] == pytest.approx(2.0)
    assert fit["best_fit_relative_rms_error"] == pytest.approx(0.0, abs=1e-12)


def test_cards_state_the_drive_current_and_the_design_current():
    summary = FieldResultSummary(
        visible=True,
        source_key="fabricated",
        current_A=11.75,
        drive_current_A=12.41,
        shape_percent=1.684,
        gain_percent=-5.3,
    )
    cards = {card["key"]: card for card in drive_metric_cards(summary)}
    assert cards["drive"]["value"] == "12.41 A"
    assert cards["shape"]["value"] == "1.68 %"
    assert cards["gain"]["value"] == "-5.3 %" and cards["gain"]["name"] == "Field at 11.75 A"
    assert not any(card["unavailable"] for card in cards.values())
    sheet = drive_metric_cards(FieldResultSummary(visible=True, source_key="sheet"))
    assert sheet[0]["value"] == "Not applicable" and all(card["unavailable"] for card in sheet)


def test_drive_cards_preserve_current_polarity_relative_to_the_winding():
    summary = FieldResultSummary(
        visible=True,
        source_key="routed",
        current_A=-3.125,
        drive_current_A=-3.75,
        shape_percent=1.5,
        gain_percent=-16.67,
    )
    cards = {card["key"]: card for card in drive_metric_cards(summary)}
    assert cards["drive"]["value"] == "-3.75 A"
    assert cards["gain"]["name"] == "Field at -3.125 A"


def test_saved_gain_is_target_projection_instead_of_inverse_current_scale():
    actual, target = np.array([1.0, 1.0]), np.array([1.0, 0.0])
    fit = current_scale_fit(actual, target)
    evidence = {"available": True, "current_refitted": False, "current_A": -2.0, **fit}
    record = SimpleNamespace(
        metadata={"wire_fields": {"routed": evidence}},
        array_names=("target_points_m", "routed_field_T"),
    )
    summary = summarize_field_result(record)
    expected_gain = 100.0 * (np.dot(actual, target) / np.dot(target, target) - 1.0)
    assert expected_gain == 0.0
    assert summary.gain_percent == pytest.approx(expected_gain, abs=1e-12)
    assert summary.shape_percent == pytest.approx(100.0 / np.sqrt(2.0))
    assert summary.drive_current_A == pytest.approx(-1.0)
    for missing in fit:
        evidence.pop(missing)
        assert summarize_field_result(record).gain_percent is None
        evidence[missing] = fit[missing]
