"""An explicit manufacturing corridor must never become an implicit waiver."""

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.config_validation import validate_config


PARAMETER = "sweep_corner_cluster_max_source_deviation_m"


def test_compound_corner_fallback_is_disabled_by_default():
    config = CoilGenConfig()
    assert getattr(config, PARAMETER) == 0.0
    assert config.manufacturing_settings.as_dict()[PARAMETER] == 0.0


@pytest.mark.parametrize("value", [-1e-3, np.nan, np.inf, -np.inf, True, False, None, "invalid"])
def test_compound_corner_corridor_rejects_invalid_or_boolean_distances(value):
    config = CoilGenConfig(sweep_bend_mode="fillet", **{PARAMETER: value})
    report = validate_config(config)
    assert any(message.field == PARAMETER for message in report.errors)


def test_positive_compound_corner_corridor_requires_fillet_mode():
    report = validate_config(CoilGenConfig(**{PARAMETER: 1e-3}))
    assert any(message.field == PARAMETER for message in report.errors)


def test_explicit_finite_compound_corner_corridor_is_accepted():
    report = validate_config(
        CoilGenConfig(coil_mesh_file="create cylinder mesh", sweep_bend_mode="fillet", **{PARAMETER: 1e-3})
    )
    assert report.ok, report.format()
