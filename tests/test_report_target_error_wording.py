"""Absolute target-error limits are not reported as a 0 % relative tolerance."""

import numpy as np

from coilgen import reporting


def test_absolute_limit_detection():
    absolute = {"target_error_tolerance_percent": 0.0, "target_error_allowed_field_metric_T": 1.5e-6}
    relative = {"target_error_tolerance_percent": 3.0, "target_error_allowed_field_metric_T": 1.5e-6}
    assert reporting._target_error_limit_is_absolute(absolute)
    assert not reporting._target_error_limit_is_absolute(relative)
    assert not reporting._target_error_limit_is_absolute({"target_error_tolerance_percent": 0.0})
    assert reporting._target_error_limit_is_absolute(
        {"target_error_tolerance_percent": np.nan, "target_error_allowed_field_metric_T": 1e-6}
    )
