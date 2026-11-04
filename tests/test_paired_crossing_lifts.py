from types import SimpleNamespace

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.models import FabricationPath, WirePath, WirePathGeometry, WireSegmentOwner
from coilgen.paired_crossing_lifts import paired_witness_lift_proposal, smooth_lift_ceiling


def test_paired_crossing_changes_both_heights_and_preserves_the_routed_footprint(monkeypatch):
    planar = np.array([[-20, 0], [-1, 0], [1, 0], [20, 0], [20, 20], [0, 20], [0, 1], [0, -1], [0, -20]])
    points = np.column_stack((planar, np.full(len(planar), 2.0))) * 1e-3
    normals = np.tile([0.0, 0.0, 1.0], (len(planar), 1))
    owners = [
        WireSegmentOwner("contour_loop", 0, 0) if i < 4 else WireSegmentOwner("within_group_return", 0)
        for i in range(len(planar) - 1)
    ]
    wire = WirePath(points, owners, [WirePathGeometry.OFFSET_SURFACE] * len(owners))
    section = 1e-3 * np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]])
    mounted = FabricationPath(
        points + 0.0005 * normals,
        normals,
        normals,
        np.zeros_like(normals),
        section,
        points,
        False,
        segment_owners=tuple(owners),
        segment_geometry=tuple(wire.segment_geometry),
    )
    part = SimpleNamespace(
        fabrication_path=mounted,
        manufacturing_diagnostics={
            "copper_clearance_witness_intervals": [(1, 6, 0.0)],
            "sweep_centerline_offset_m": 0.0005,
        },
    )
    seed = dict(
        source_geometry=[WirePathGeometry.SURFACE] * len(owners),
        source_positions=np.arange(len(points)),
        output_normals=normals,
        lift_distance_m=points[:, 2].copy(),
        selected_contacts=[],
    )
    monkeypatch.setattr("coilgen.paired_crossing_lifts.incumbent_lift_proposal", lambda *_: (wire, seed))
    config = CoilGenConfig(normal_shift_length=0.004, sweep_min_bend_radius=0.006, conductor_cross_section_width=0.002)
    for reverse in (False, True):
        candidate, detail = paired_witness_lift_proposal(part, config, reverse=reverse)
        heights = detail["lift_distance_m"]
        np.testing.assert_allclose(candidate.v[:, :2], points[:, :2], rtol=0, atol=0)
        np.testing.assert_array_equal(candidate.v[[0, -1]], points[[0, -1]])
        assert heights.min() >= 0 and heights.max() <= config.normal_shift_length
        assert (heights[1] > heights[6]) == reverse
        assert abs(heights[1] - heights[6]) > 0.003
        assert detail["selected_contacts"]


def test_smooth_ceiling_preserves_lower_layers_footprint_and_prescribed_terminals():
    normals = np.tile([0.0, 0.0, 1.0], (5, 1))
    footprint = np.column_stack((np.arange(5) * 0.001, np.zeros((5, 2))))
    for sign in (1, -1):
        original = sign * np.array([0.0, 0.002, 0.0034, 0.004, 0.0])
        wire = WirePath(footprint + original[:, None] * normals)
        config = CoilGenConfig(normal_shift_length=sign * 0.0035, conductor_cross_section_width=0.002)
        clipped, d = smooth_lift_ceiling((wire, dict(lift_distance_m=original, output_normals=normals)), config)
        height = d["lift_distance_m"]
        np.testing.assert_array_equal(original[[0, 1, -1]], height[[0, 1, -1]])
        assert 0.0033 < sign * height[2] < 0.0034
        assert sign * height[2] < sign * height[3] <= 0.0035
        np.testing.assert_allclose(clipped.v - height[:, None] * normals, footprint, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(wire.v, footprint + original[:, None] * normals)
