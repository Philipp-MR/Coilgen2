"""A tightly sampled ninety-degree bend can invert the inner edge of finite-width foil."""

from dataclasses import replace

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import _build_sweep_rings, _surface_adaptive_resample
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry, WireSegmentOwner
from coilgen.sweep_fold_relief import profile_fold_stations, repair_reversed_sweep, reversed_profile_edges


def _bent_foil():
    points = 1e-3 * np.array(
        [
            [-18, 0, 0],
            [-12, 0, 0],
            [-8, 0, 0],
            [-4, 0, 0],
            [-0.6, 0, 0],
            [-0.15, 0.04, 0],
            [0.04, 0.15, 0],
            [0.12, 0.5, 0],
            [0.12, 4, 0],
            [0.12, 8, 0],
            [0.12, 12, 0],
            [0.12, 18, 0],
        ],
        dtype=float,
    )
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]), np.array([[0, 1, 2], [0, 2, 3]])
        )
    )
    section = 1e-3 * np.array([[-1.0, -0.1], [1.0, -0.1], [1.0, 0.1], [-1.0, 0.1]])
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    owners = [WireSegmentOwner("contour_loop", 0, 0)] * len(labels)
    part = CoilPart(mesh=mesh, wire_path=WirePath(points.copy(), owners, labels))
    config = CoilGenConfig(
        sweep_bend_mode="fillet",
        sweep_min_bend_radius=0.006,
        sweep_corner_cluster_max_source_deviation_m=0.004,
        sweep_surface_realization="mesh",
    )
    return points, labels, part, section, config


def test_realized_foil_fold_repair_preserves_source_and_protected_joins():
    points, labels, part, section, config = _bent_foil()
    rings = _build_sweep_rings(part, points, labels, section, config, closed=False, mounting_offset=0.0001)
    assert len(reversed_profile_edges(rings[4], 4, closed=False)) > 0
    repair = repair_reversed_sweep(
        part, rings[0], labels, rings[4], section, config, closed=False, mounting_offset=0.0001, radius=0.006
    )
    fixed, typed, diagnostics = repair
    assert not len(reversed_profile_edges(fixed[4], 4, closed=False))
    assert 0 < diagnostics["sweep_fold_relief_max_source_bound_m"] <= 0.004
    assert set(typed) == {WirePathGeometry.SURFACE}
    np.testing.assert_array_equal(fixed[0][[0, -1]], points[[0, -1]])
    np.testing.assert_array_equal(part.wire_path.v, points)
    # No source point may be spliced across a raised return.
    protected = replace(
        part, wire_path=WirePath(points.copy(), [WireSegmentOwner("within_group_return")] * len(labels), labels)
    )
    assert (
        repair_reversed_sweep(
            protected, rings[0], labels, rings[4], section, config, closed=False, mounting_offset=0.0001, radius=0.006
        )
        is None
    )


def test_raised_connector_handoff_keeps_programmed_height_without_surface_turn_opt_in():
    points, _, part, section, config = _bent_foil()
    height = 0.004
    raised = points + np.array([0.0, 0.0, height])
    labels = [WirePathGeometry.OFFSET_SURFACE] * (len(raised) - 1)
    owners = [
        WireSegmentOwner("contour_loop", 0, 0)
        if i < 5
        else WireSegmentOwner("inter_group_connector", connector_index=0)
        for i in range(len(labels))
    ]
    part = replace(
        part,
        wire_path=WirePath(raised.copy(), owners, labels),
        normal_shift_provenance={"method": "arclength_crossing_layers"},
    )
    config = replace(
        config,
        sweep_corner_cluster_max_source_deviation_m=0.0,
        normal_shift_length=height,
        interconnection_cut_width=0.008,
    )
    rings = _build_sweep_rings(part, raised, labels, section, config, closed=False, mounting_offset=0.0001)
    assert len(reversed_profile_edges(rings[4], 4, closed=False)) > 0
    fixed, typed, diags = repair_reversed_sweep(
        part, rings[0], labels, rings[4], section, config, closed=False, mounting_offset=0.0001, radius=0.006
    )
    assert set(typed) == {WirePathGeometry.OFFSET_SURFACE}
    assert diags["sweep_fold_relief_max_source_bound_m"] <= height
    np.testing.assert_allclose(fixed[0][:, 2], height, atol=1e-9)
    np.testing.assert_array_equal(fixed[0][[0, -1]], raised[[0, -1]])
    assert len(reversed_profile_edges(fixed[4], 4, closed=False)) == 0


def test_nearby_facets_can_cross_even_if_no_single_profile_edge_reverses():
    _, _, part, section, config = _bent_foil()
    # The actual biplanar Gy zigzag, rotated into the fixture's XY plane.
    planar = np.array(
        [
            [-28.566, 10.634],
            [-30.739, 9.006],
            [-31.334, 8.333],
            [-31.274, 7.514],
            [-30.899, 6.686],
            [-30.841, 5.841],
            [-31.535, 5.570],
            [-32.467, 5.625],
            [-33.333, 5.400],
            [-34.344, 4.430],
        ]
    )
    points = 1e-3 * np.column_stack((planar, np.zeros(len(planar))))
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    rings = _build_sweep_rings(part, points, labels, section, config, closed=False, mounting_offset=0.0001)
    assert 5 not in reversed_profile_edges(rings[4], len(section), closed=False)
    assert 5 in profile_fold_stations(rings[4], len(section), closed=False)


def test_sharp_returning_contour_retains_its_tip_inside_the_source_corridor():
    _, _, part, section, config = _bent_foil()
    # A U-turn at this opening angle needs two displaced flanks to fit a
    # constant-radius nose inside the allowed three-millimetre corridor.
    planar = np.array(
        [
            [-25.0, -4.0],
            [-20.0, -3.2],
            [-15.0, -2.4],
            [-10.0, -1.6],
            [-6.0, -0.96],
            [-3.0, -0.48],
            [-0.5, -0.08],
            [0.0, 0.0],
            [-0.5, 0.08],
            [-3.0, 0.48],
            [-6.0, 0.96],
            [-10.0, 1.6],
            [-15.0, 2.4],
            [-20.0, 3.2],
            [-25.0, 4.0],
        ]
    )
    points = 1e-3 * np.column_stack((planar, np.zeros(len(planar))))
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    owners = [WireSegmentOwner("contour_loop", 0, 0)] * len(labels)
    part = replace(part, wire_path=WirePath(points.copy(), owners, labels))
    config = replace(config, sweep_min_bend_radius=0.003, sweep_corner_cluster_max_source_deviation_m=0.003)
    rings = _build_sweep_rings(part, points, labels, section, config, closed=False, mounting_offset=0.0001)
    assert len(profile_fold_stations(rings[4], 4, closed=False)) > 0
    fixed, _, diags = repair_reversed_sweep(
        part, rings[0], labels, rings[4], section, config, closed=False, mounting_offset=0.0001, radius=0.003
    )
    assert len(profile_fold_stations(fixed[4], 4, closed=False)) == 0
    assert diags["sweep_fold_relief_max_source_bound_m"] <= 0.003
    np.testing.assert_array_equal(fixed[0][[0, -1]], points[[0, -1]])


def test_positive_copper_gap_flags_an_under_radius_turn_without_a_reversed_edge():
    _, _, part, section, config = _bent_foil()
    planar = np.array(
        [
            [-39.7604, 35.8274],
            [-37.9357, 37.9668],
            [-35.9035, 39.8681],
            [-32.8223, 42.2998],
            [-32.6767, 42.4792],
            [-32.5721, 42.8265],
            [-32.7190, 43.1575],
            [-33.1148, 43.4732],
            [-34.8870, 44.5291],
            [-36.6592, 45.5851],
            [-38.4313, 46.6411],
        ]
    )
    points = 1e-3 * np.column_stack((planar, np.zeros(len(planar))))
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    section = section * np.array([0.5, 5.0])
    rings = _build_sweep_rings(part, points, labels, section, config, closed=False, mounting_offset=0.0005)
    assert 5 not in profile_fold_stations(rings[4], 4, closed=False, bend_radius=0.003, gap=0.0001)
    assert 5 in profile_fold_stations(rings[4], 4, closed=False, bend_radius=0.003, gap=0.00025)


def test_mesh_subdivision_uses_same_carrier_for_attached_knots_and_new_midpoints():
    _, _, part, section, config = _bent_foil()
    points = 1e-3 * np.array([[-20, 0, 0], [-10, 0, -0.216], [0, 0, 0.12], [10, 0, 2], [20, 0, 2]])
    surface, raised = WirePathGeometry.SURFACE, WirePathGeometry.OFFSET_SURFACE
    sampled, _, info = _surface_adaptive_resample(
        part, points, [surface, surface, raised, raised], section, config, closed=False
    )
    assert info["sweep_retracted_surface_knot_count"] == 1
    np.testing.assert_allclose(info["sweep_retracted_surface_knot_max_m"], 0.000216)
    assert np.min(np.linalg.norm(sampled - [-0.01, 0, 0], axis=1)) < 1e-12
    # Mixed surface/raised handoff and feeds remain prescribed physical stations.
    for i in (0, 2, 4):
        assert np.min(np.linalg.norm(sampled - points[i], axis=1)) < 1e-12
