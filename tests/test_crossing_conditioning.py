import numpy as np

from coilgen.crossing_conditioning import _taper_join_budget
from coilgen.models import WirePathGeometry, WireSegmentOwner


def test_conditioned_join_uses_shared_cut_budget_at_its_contour_handoff():
    points = np.array([[-3, 0], [-2, 0], [-1, 0], [0, 0], [-1.5, 0.2], [-1.5, 1.2], [-1.5, 2.2]]) * 1e-3
    stations = np.arange(len(points)) * 1e-3
    join = WireSegmentOwner("contour_group", 0, connector_role="joint_spiral_join")
    loop = WireSegmentOwner("contour_loop", 0, 0)
    owners = [join] * 4 + [loop] * 2
    geom = [WirePathGeometry.SURFACE] * len(owners)
    heights = np.zeros(len(stations))
    shift = np.tile([0.0, 0.003], (len(stations), 1))
    shift[4:, 1] = 0.006
    corrected = _taper_join_budget(points, stations, owners, geom, heights, shift, 0.003, 0.006, 0.003, np.array([]))
    np.testing.assert_array_equal(corrected[:4], shift[:4])
    np.testing.assert_allclose(corrected[4], [0.0, 0.003])
    assert 0.003 < corrected[5, 1] < 0.006
    inherited = _taper_join_budget(points, stations, owners, geom, heights, shift, 0.003, 0.006, 0.003, stations[3:4])
    np.testing.assert_array_equal(inherited, shift)
