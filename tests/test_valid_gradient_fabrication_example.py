"""The passing Maxwell bench reference must retain geometry and drive polarity."""

from copy import deepcopy

import numpy as np
import pytest

from examples.research_valid_gradient_fabrication import main
from coilgen.research.validation import validate_winding_on_independent_grid


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return main(["--output-dir", str(tmp_path_factory.mktemp("maxwell_reference"))])


def test_reference_passes_actual_swept_geometry_and_fixed_current_gradient_checks(reference):
    result, fabrication, verification = reference
    assert fabrication.ready and fabrication.assembly_clearance["passed"]
    assert len(fabrication.paths) == 2
    assert all(record.diagnostics["manufacturing_ok"] for record in fabrication.paths)
    independent = verification["independent_grid"]
    assert independent["verified"] and independent["passed"]
    assert independent["new_point_count"] == independent["point_count"] == 913
    assert not independent["current_refitted"]
    assert list(independent["currents_A"].values()) == [5.0, -5.0]
    assert independent["linearity"]["linearity_rms_percent"] < 1.0
    assert independent["linearity"]["linearity_peak_percent"] < 5.0
    assert independent["linearity"]["linearity_gradient_strength_error_percent"] < 1.0
    assert independent["rms_error_T"] < result.input_data.target_error_rms_limit_T
    assert verification["analytic_axis_relative_L2_error_percent"] < 0.05
    assert not verification["construction_released"]
    assert not verification["drive_feed_designed"]


def test_reversing_one_current_fails_the_same_gradient_contract_without_refit(reference):
    result, fabrication, _verification = reference
    incorrect = deepcopy(fabrication)
    incorrect.electrical_metrics["bodies"][1]["current_A"] *= -1.0
    checked = validate_winding_on_independent_grid(result, incorrect, resolution=12)
    assert checked["verified"] and checked["passed"] is False
    assert not checked["current_refitted"]
    assert checked["rms_error_T"] > result.input_data.target_error_rms_limit_T
    assert checked["linearity"]["linearity_gradient_strength_error_percent"] > 99.0
    assert fabrication.electrical_metrics["bodies"][1]["current_A"] == -5.0


def test_independent_validation_does_not_change_support_target_or_recorded_paths(reference):
    result, fabrication, _verification = reference
    source_parts = [part.mesh.vertices.copy() for part in result.coil_parts]
    source_target = result.target_field.b.copy()
    paths = [record.points_m.copy() for record in fabrication.paths]
    checked = validate_winding_on_independent_grid(result, fabrication, resolution=10)
    assert checked["verified"] and checked["passed"]
    for part, original in zip(result.coil_parts, source_parts):
        np.testing.assert_array_equal(part.mesh.vertices, original)
    for record, original in zip(fabrication.paths, paths):
        np.testing.assert_array_equal(record.points_m, original)
    np.testing.assert_array_equal(result.target_field.b, source_target)
