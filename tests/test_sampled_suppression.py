"""Suppression design-surface diagnostics never imply independent acceptance."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.models import TargetField
from coilgen.physics import biot_savart_wire_path
from coilgen.research.suppression_validation import sampled_suppression
from coilgen.research.validation import validate_fabricated_constraints


def case(component="vector", *, area=True):
    config = CoilGenConfig(
        suppression_field_component=component,
        suppression_field_rms_limit_T=20e-6,
        suppression_field_peak_limit_T=35e-6,
        suppression_surface_weighting="area",
    )
    target = TargetField(
        coords=np.arange(12, dtype=float).reshape(4, 3),
        b=np.full((4, 3), 1e-6),
        weights=np.array([1, 1, 1, 0]),
        suppressed=np.array([False, True, True, True]),
        target_field_group_inds=np.ones(4),
        suppression_area_weights=np.array([0, 1, 3, 5], dtype=float) if area else None,
        suppression_normals=np.tile([2.0, 0, 0], (4, 1)),
    )
    residual = 1e-6 * np.array([[1e9, 0, 0], [3, 4, 0], [12, 16, 0], [1e9, 0, 0]])
    result = SimpleNamespace(input_data=config, target_field=target, b_field_opt_sf=target.b + residual)
    return result, target.b + 2 * residual


def test_vector_surface_rms_is_area_normalized_in_tesla_and_retains_failures():
    result, fabricated = case()
    report = sampled_suppression(result, fabricated)
    assert report["scope"] == "suppression_design_samples" and report["independent"] is False
    assert report["sample_count"] == 2 and report["weighting"] == "area"
    expected = np.sqrt((5**2 + 3 * 20**2) / 4) * 1e-6
    assert report["sheet"]["rms_T"] == pytest.approx(expected)
    assert report["fabricated"]["rms_T"] == pytest.approx(2 * expected)
    assert report["sheet"]["rms_passed"] and not report["fabricated"]["rms_passed"]
    assert report["sheet"]["peak_passed"] and not report["fabricated"]["peak_passed"]
    assert report["fabricated"]["peak_T"] == pytest.approx(40e-6)
    result.target_field.suppression_area_weights *= 0.4
    assert sampled_suppression(result, fabricated)["fabricated"]["rms_T"] == pytest.approx(2 * expected)


@pytest.mark.parametrize("component,peak", (("Bx", 24), ("components", 32), ("Bnormal", 24), ("Bmag", 40)))
def test_scalar_and_vector_peak_match_the_configured_component(component, peak):
    result, field = case(component)
    assert sampled_suppression(result, field)["fabricated"]["peak_T"] == pytest.approx(peak * 1e-6)


def test_missing_weights_or_field_do_not_create_a_surface_pass():
    result, _field = case(area=False)
    check = sampled_suppression(result, None)
    assert check["sheet"]["rms_T"] is None and check["sheet"]["rms_passed"] is None
    assert not check["sheet"]["available"] and "areas" in check["sheet"]["reason"]
    assert not check["fabricated"]["available"] and check["fabricated"]["peak_passed"] is None
    incomplete = SimpleNamespace(electrical_metrics={"bodies": []}, winding_steps=[])
    review = validate_fabricated_constraints(result, incomplete)
    assert not review["sampled_suppression"]["fabricated"]["available"]
    assert not review["verified"] and any("suppression" in reason for reason in review["unknown_reasons"])


@pytest.mark.parametrize("limit_factor,passes", ((1.1, True), (0.9, False)))
def test_fabricated_current_recheck_keeps_the_holdout_unknown(limit_factor, passes):
    loop = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], float)
    points = np.array([[0.1, 0.3, 0.4], [0.5, 0.5, 0.4]])
    field = 2 * biot_savart_wire_path(loop, points, worker_count=1)
    expected = np.linalg.norm(field[1])
    config = CoilGenConfig(
        suppression_field_component="vector",
        suppression_surface_weighting="area",
        suppression_field_rms_limit_T=limit_factor * expected,
    )
    target = TargetField(
        points,
        np.array([field[0], np.zeros(3)]),
        np.ones(2),
        np.array([False, True]),
        np.ones(2),
        suppression_area_weights=np.array([0, 0.04]),
    )
    result = SimpleNamespace(input_data=config, target_field=target, b_field_opt_sf=field)
    fabrication = SimpleNamespace(
        winding_steps=[{"body_id": "loop", "electrical_points_m": loop, "circuit_complete": True}],
        electrical_metrics={"bodies": [{"body_id": "loop", "current_A": 2.0}]},
    )
    review = validate_fabricated_constraints(result, fabrication)
    surface = review["sampled_suppression"]
    assert surface["independent"] is False
    assert surface["fabricated"]["rms_T"] == pytest.approx(expected)
    assert surface["fabricated"]["rms_passed"] is passes
    assert not review["verified"] and not review["passed"]
    assert any("not been independently verified" in reason for reason in review["unknown_reasons"])
