"""Physical orientation, ownership and clearance regressions for routing lifts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.layout import (
    _crossing_lift_anchors,
    _crossing_shift_profile,
    _mark_crossing_anchor,
    _normal_shift_routes_by_segment,
    _repair_owned_offset_backtracks,
    _smooth_shift_levels_by_arclength,
    _smooth_wire_path_normals,
    _wire_backtrack_indices,
    cleanup_wire_paths,
    shift_return_paths,
    wire_path_normals,
)
from coilgen.models import CoilPart, ConnectedGroup, Mesh, WirePath, WireSegmentOwner


def _plane(z=0.0):
    return normalize_mesh(
        Mesh(
            vertices=np.array([[-0.1, -0.1, z], [0.1, -0.1, z], [0.1, 0.1, z], [-0.1, 0.1, z]]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )


def _owned_crossing_part():
    first = np.column_stack((np.linspace(-0.02, 0.02, 41), np.linspace(-0.02, 0.02, 41), np.zeros(41)))
    second = np.column_stack((np.linspace(-0.02, 0.02, 41), np.linspace(0.02, -0.02, 41), np.zeros(41)))
    points = np.vstack(
        (
            first,
            [[0.04, 0.02, 0], [0.04, 0.04, 0], [-0.02, 0.04, 0]],
            second,
            [[0.04, -0.02, 0], [0.04, -0.04, 0], [-0.02, -0.04, 0]],
            first[:1],
        )
    )
    owners = [WireSegmentOwner("contour_loop", loop_index=0) for _ in range(len(points) - 1)]
    for index in range(44, 84):
        owners[index] = WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="primary")
    return CoilPart(mesh=_plane(), wire_path=WirePath(points, segment_owners=owners))


def test_barycentric_lift_removes_nearest_vertex_voronoi_jump():
    support = normalize_mesh(
        Mesh(
            np.array(
                [
                    [-0.05, -0.05, 0],
                    [-0.05, 0.05, 0],
                    [0, -0.05, 0],
                    [0, 0.05, 0],
                    [0.05, -0.05, 0.05],
                    [0.05, 0.05, 0.05],
                ]
            ),
            np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]]),
        )
    )
    x = -0.025 + np.array([-0.0001, -0.00001, 0.00001, 0.0001])
    points = np.column_stack((x, np.full(4, -0.01), np.zeros(4)))
    part = CoilPart(mesh=support, wire_path=WirePath(points))
    normals = wire_path_normals(part)
    lifted = points + 0.003 * normals
    assert np.linalg.norm(lifted[2] - lifted[1]) < 25e-6  # Previously 1.590 mm for a 20 um interval.
    assert np.linalg.norm(normals[2] - normals[1]) < 1e-3
    refined_points = 0.5 * (points[:-1] + points[1:])
    part.wire_path = WirePath(np.vstack((points, refined_points)))
    np.testing.assert_allclose(wire_path_normals(part)[:4], normals, atol=1e-14)


def test_lift_respects_declared_inward_orientation_instead_of_center_flip():
    mesh = normalize_mesh(
        Mesh(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], dtype=float),
            np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
        )
    )
    points = mesh.vertices[mesh.faces[0]].mean(axis=0)[None, :]
    part = CoilPart(mesh=mesh, wire_path=WirePath(np.vstack((points, points + [1e-6, 0, 0]))))
    normals = wire_path_normals(part)
    assert normals[0] @ mesh.face_normals[0] > 0
    assert normals[0] @ (points[0] - mesh.vertices.mean(axis=0)) < 0
    # Check orientation against input geometry rather than an inferred centre.
    from coilgen.manufacturing import _SurfaceProjector

    expected = _SurfaceProjector(mesh).project(part.wire_path.v)[1]
    np.testing.assert_allclose(normals, expected, atol=1e-14)


def test_normal_smoothing_never_flips_material_hemisphere():
    normals = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    np.testing.assert_array_equal(_smooth_wire_path_normals(normals, 0), normals)


def test_overpass_anchor_survives_nearby_underpass_ramp():
    points = np.column_stack((np.linspace(0, 0.02, 21), np.zeros((21, 2))))
    high = np.zeros(21, dtype=bool)
    low = np.zeros(21, dtype=bool)
    high[8] = True
    low[12] = True
    profile = _crossing_shift_profile(np.zeros(21), high, low, points, 0.01, (1, 1))
    assert profile[8] == 1
    assert profile[12] == 0
    assert np.all(np.diff(profile[8:13]) <= 0)


def test_saved_gz_endpoint_underpass_does_not_flatten_adjacent_interior_anchor():
    # Exact source points6061..6063,15403..15404,14063..14064 from Sep16 GZ.
    # The terminal point is an underpass; the crossing on the first interval
    # is compatible with lifting the connector's sole interior sample.
    points = np.array(
        [
            [0.0839685493692387, 0.011448759900526887, -0.036289122121479785],
            [0.08015745425459582, 0.0013756147954094473, -0.045489927739504485],
            [0.07859034566800104, -0.00472950538868443, -0.04927326588759361],
            [0.08109578533737723, 0.011070347276967579, -0.04322459273172976],
            [0.08507900791150827, 0.001879026427576782, -0.03360824710765734],
            [0.07859034566800104, -0.00472950538868443, -0.04927326588759361],
            [0.08116263841965064, -0.004724142596598096, -0.04306319709865832],
        ]
    )
    anchors = _crossing_lift_anchors([((0,), (3,)), ((5,), (1,))], [(0, 3, 1.263e-9), (1, 5, 0.0)], points)
    assert anchors[1][3] == 2.0
    over, under = np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=bool)
    for over_segment, under_segment, over_position, under_position, _distance in anchors:
        _mark_crossing_anchor(over, over_segment, over_position, points)
        _mark_crossing_anchor(under, under_segment, under_position, points)
    assert over[1] and not under[1]
    assert under[2]
    allowed = np.array([False, True, False, False, False, False, False])
    profile = _crossing_shift_profile(np.zeros(len(points)), over, under, points, 0.015, (2, 3), allowed_points=allowed)
    assert profile[1] == 1.0
    np.testing.assert_array_equal(profile[[0, 2]], [0.0, 0.0])


def test_soft_spacing_contacts_cannot_suppress_physical_overpass_anchor():
    # Two owned returns are 4 mm apart: below the requested 6 mm design gap,
    # but above the 1.414 mm conductor envelope. The first return crosses a
    # contour at the origin and must retain its 3 mm lift there.
    points = np.array(
        [
            [0, -0.03, 0],
            [0, 0.03, 0],
            [-0.04, 0.03, 0],
            [-0.04, 0, 0],
            [-0.02, 0, 0],
            [0, 0, 0],
            [0.02, 0, 0],
            [0.04, 0, 0],
            [0.04, 0.03, 0],
            [0.006, 0.03, 0],
            [0.006, 0.004, 0],
            [0.01, 0.004, 0],
            [0.06, 0.004, 0],
            [0.06, -0.04, 0],
            [0, -0.04, 0],
            [0, -0.03, 0],
        ]
    )
    owners = [WireSegmentOwner("contour_loop", loop_index=0)] * (len(points) - 1)
    owners[4:6] = [WireSegmentOwner("within_group_return", group_index=0)] * 2
    owners[9:11] = [WireSegmentOwner("within_group_return", group_index=1)] * 2
    part = CoilPart(mesh=_plane(), wire_path=WirePath(points.copy(), segment_owners=owners))
    shift_return_paths(
        [part],
        CoilGenConfig(
            normal_shift_length=0.003,
            min_wire_distance_m=0.006,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
        ),
    )
    assert part.wire_path.v[5, 2] == pytest.approx(0.003)
    np.testing.assert_array_equal(part.wire_path.v[:2], points[:2])
    assert any(0.003 < contact[2] < 0.006 for contact in part.normal_shift_provenance["contacts_before"])
    physical = part.normal_shift_diagnostics["normal_shift_physical_clearance_m"]
    assert all(anchor["distance_m"] <= physical + 1e-8 for anchor in part.normal_shift_provenance["crossing_anchors"])
    assert part.normal_shift_diagnostics["normal_shift_lift_below_clearance_target"]


def test_effective_lift_route_memberships_roundtrip_including_extended_segments(tmp_path):
    from coilgen.manufacturing import save_manufacturing_diagnostics

    points = np.column_stack((np.arange(7) * 0.001, np.zeros((7, 2))))
    group = ConnectedGroup(points.copy(), points[2:4].copy(), points[:1], points[-1:])
    part = CoilPart(mesh=_plane(), wire_path=WirePath(points.copy()), connected_groups=[group])
    original_routes = _normal_shift_routes_by_segment(part)
    assert [index for index, routes in enumerate(original_routes) if routes] == [2]
    shift_return_paths(
        [part],
        CoilGenConfig(
            normal_shift_length=0.003, conductor_cross_section_width=0.001, conductor_cross_section_height=0.001
        ),
    )
    expected = [{"key": ["within_group_return", 0, 0], "priority": 2, "order": 0, "segment_indices": [1, 2, 3]}]
    assert part.normal_shift_provenance["lift_routes"] == expected
    saved_paths = save_manufacturing_diagnostics(SimpleNamespace(coil_parts=[part]), tmp_path, prefix="lift_routes")
    with np.load(saved_paths[-1], allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        np.testing.assert_array_equal(saved["points_before_m"], points)
    assert metadata["lift_routes"] == expected
    # The serialized effective memberships are enough to rebuild the original
    # per-segment route table; the stored reference alone would lose 1 and3.
    restored = [set() for _ in range(len(points) - 1)]
    for route in metadata["lift_routes"]:
        for index in route["segment_indices"]:
            restored[index].add(tuple(route["key"]))
    assert [index for index, routes in enumerate(restored) if routes] == [1, 2, 3]


def test_arclength_ramp_matches_dense_reference_without_quadratic_storage():
    positions = np.array([0, 0.001, 0.007, 0.03, 0.05, 0.1])
    points = np.column_stack((positions, np.zeros((6, 2))))
    levels = np.array([0, 1, 0, 1, 0, 0])
    distance = abs(positions[:, None] - positions[levels.astype(bool)][None, :]).min(axis=1)
    expected = 0.5 * (1 + np.cos(np.pi * np.clip(distance / 0.025, 0, 1)))
    actual = _smooth_shift_levels_by_arclength(levels, points, 0.025)
    np.testing.assert_allclose(actual, expected)
    # This would allocate >2 GB in the previous N by N_active distance matrix.
    x = np.linspace(0, 1, 30000)
    result = _smooth_shift_levels_by_arclength(np.arange(len(x)) % 2, np.column_stack((x, np.zeros((len(x), 2)))), 0.01)
    assert result.shape == x.shape
    assert np.all(np.isfinite(result))


def test_explicit_owners_disambiguate_repeated_coordinates():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
    owners = [WireSegmentOwner("contour_loop", loop_index=0) for _ in range(4)]
    owners[3] = WireSegmentOwner("inter_group_connector", connector_index=4, connector_role="closing")
    part = CoilPart(mesh=_plane(), wire_path=WirePath(points, segment_owners=owners))
    routes = _normal_shift_routes_by_segment(part)
    assert routes[0] == []
    assert routes[3][0].key == ("inter_group_connector", 4, 1)


def test_owned_crossing_has_clear_overpass_and_preserves_contours():
    part = _owned_crossing_part()
    source = part.wire_path.v.copy()
    config = CoilGenConfig(
        normal_shift_length=0.005,
        min_wire_distance_m=0.003,
        conductor_cross_section_width=0.001,
        normal_sweep_smooth_factor=3,
    )
    shift_return_paths([part], config)
    provenance = part.normal_shift_provenance
    np.testing.assert_array_equal(
        part.wire_path.v[~provenance["allowed_points"]], source[~provenance["allowed_points"]]
    )
    assert part.normal_shift_diagnostics["normal_shift_unresolved_contact_count"] == 0
    assert np.max(np.linalg.norm(part.wire_path.v - source, axis=1)) <= 0.005 + 1e-12
    assert part.wire_path.v[64, 2] == pytest.approx(0.005)
    assert provenance["normal_face_indices"].shape == (len(source),)
    assert provenance["segment_owners"][50]["kind"] == "inter_group_connector"
    assert provenance["crossing_events"]
    saved_before = provenance["points_before_m"].copy()
    cleanup_wire_paths([part], config=config)
    np.testing.assert_array_equal(part.normal_shift_provenance["points_before_m"], saved_before)


def test_insufficient_lift_is_explicit_unresolved_and_never_increased():
    part = _owned_crossing_part()
    source = part.wire_path.v.copy()
    config = CoilGenConfig(
        normal_shift_length=0.002,
        min_wire_distance_m=0.005,
        conductor_cross_section_width=0.001,
        normal_sweep_smooth_factor=3,
    )
    shift_return_paths([part], config)
    assert part.normal_shift_diagnostics["normal_shift_lift_below_clearance_target"]
    assert part.normal_shift_diagnostics["normal_shift_status"] == "unresolved"
    assert part.normal_shift_diagnostics["normal_shift_unresolved_contact_count"] > 0
    assert np.max(np.linalg.norm(part.wire_path.v - source, axis=1)) <= 0.002 + 1e-12
    cleanup_wire_paths([part], config=config)
    assert "normal_shift_unresolved" in part.route_diagnostics["route_quality_failure_reasons"]


def test_missing_normal_does_not_invent_world_z_lift():
    part = _owned_crossing_part()
    part.mesh = Mesh(np.array([[0, 0, 0]], dtype=float), np.zeros((0, 3), dtype=np.int64))
    source = part.wire_path.v.copy()
    shift_return_paths([part], CoilGenConfig(normal_shift_length=0.005, min_wire_distance_m=0.003))
    np.testing.assert_array_equal(part.wire_path.v, source)
    assert part.normal_shift_diagnostics["normal_shift_invalid_normal_count"] > 0


# Actual saved GX CSV points2158..2161, metres; no invented pre-shift state.
GX_KINK = (
    np.array(
        [
            [84.73665618415214, 4.922253521795158, 8.475143114840819],
            [86.96007118877326, 4.922150251399867, 2.510612838268154],
            [86.73112268779884, 5.318509367714385, 3.6600814535010264],
            [86.7289897474835, 5.948394095405975, 3.665231434267061],
        ]
    )
    * 0.001
)


def test_saved_gx_kink_is_detected_and_unowned_geometry_is_preserved():
    np.testing.assert_array_equal(_wire_backtrack_indices(GX_KINK), [1])
    part = CoilPart(mesh=_plane(), wire_path=WirePath(GX_KINK.copy(), segment_geometry=["offset_surface"] * 3))
    repaired, events = _repair_owned_offset_backtracks(part, CoilGenConfig(normal_shift_length=0.003))
    np.testing.assert_array_equal(repaired, GX_KINK)
    assert events == []


def test_backtrack_diagnostic_includes_closed_path_seam():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.1, 0, 0], [0, 0, 0]], dtype=float)
    assert 0 in _wire_backtrack_indices(points)


def test_bounded_owned_offset_spike_repair_preserves_endpoints_and_ownership():
    points = np.array([[0, 0, 0.003], [0.006, 0, 0.003], [0.004, 0.001, 0.003]])
    owner = WireSegmentOwner("inter_group_connector", connector_index=1)
    part = CoilPart(
        mesh=_plane(),
        wire_path=WirePath(points.copy(), segment_owners=[owner] * 2, segment_geometry=["offset_surface"] * 2),
    )
    repaired, events = _repair_owned_offset_backtracks(
        part, CoilGenConfig(normal_shift_length=0.003, interconnection_cut_width=0.005)
    )
    assert len(events) == 1
    assert not len(_wire_backtrack_indices(repaired))
    np.testing.assert_array_equal(repaired[[0, 2]], points[[0, 2]])
    assert events[0]["movement_m"] <= 0.003
    part.wire_path.segment_owners[0] = WireSegmentOwner("contour_loop", loop_index=1)
    unchanged, events = _repair_owned_offset_backtracks(part, CoilGenConfig(normal_shift_length=0.003))
    np.testing.assert_array_equal(unchanged, points)
    assert events == []


def test_owned_repair_requires_a_common_planar_support_triangle():
    points = np.array([[-0.006, 0.006, 0.003], [0.006, 0, 0.003], [0.004, 0.001, 0.003], [0.004, 0.004, 0.003]])
    assert len(_wire_backtrack_indices(points))
    owner = WireSegmentOwner("inter_group_connector", connector_index=1)
    part = CoilPart(
        mesh=_plane(),
        wire_path=WirePath(points.copy(), segment_owners=[owner] * 3, segment_geometry=["offset_surface"] * 3),
    )
    repaired, events = _repair_owned_offset_backtracks(
        part, CoilGenConfig(normal_shift_length=0.003, interconnection_cut_width=0.005)
    )
    np.testing.assert_array_equal(repaired, points)
    assert events == []


def test_owned_repair_cannot_create_crossing_inside_old_local_exclusion():
    points = np.array(
        [[0, 0, 0.003], [0.004, 0, 0.003], [0.003, 0.002, 0.003], [0.0035, 0.003, 0.003], [0.0005, 0.0002, 0.003]]
    )
    owner = WireSegmentOwner("inter_group_connector", connector_index=1)
    part = CoilPart(
        mesh=_plane(),
        wire_path=WirePath(points.copy(), segment_owners=[owner] * 4, segment_geometry=["offset_surface"] * 4),
    )
    repaired, events = _repair_owned_offset_backtracks(
        part, CoilGenConfig(normal_shift_length=0.003, interconnection_cut_width=0.005, min_wire_distance_m=0.003)
    )
    np.testing.assert_array_equal(repaired, points)
    assert events == []
