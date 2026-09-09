from __future__ import annotations

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.inductance import conductor_cross_section_area, equivalent_round_wire_radius


@pytest.mark.parametrize("matlab_orientation", [False, True])
def test_custom_conductor_area_supports_both_coordinate_orientations(matlab_orientation: bool) -> None:
    # A 6 mm by 4 mm polygon must supersede the 1 mm square fallback conductor.
    points = np.array([[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]]) * 1.0e-3
    if matlab_orientation:
        points = points.T
    config = CoilGenConfig(
        cross_sectional_points=points,
        conductor_cross_section_width=1.0e-3,
        conductor_cross_section_height=1.0e-3,
    )

    assert conductor_cross_section_area(config) == pytest.approx(24.0e-6)
    assert equivalent_round_wire_radius(config) == pytest.approx(np.sqrt(24.0e-6 / np.pi))
