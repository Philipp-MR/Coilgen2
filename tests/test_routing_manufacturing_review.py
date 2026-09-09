from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from coilgen.geometry import build_cylinder_mesh, rotation_matrix_by_vector
from coilgen.layout import _greedy_fuse_connected_groups
from coilgen.models import CoilPart, ConnectedGroup, Mesh, PcbLayout, PcbWirePart, WireSegmentOwner
from coilgen.pcb import _save_single_pcb_dxf, _wire_parts_from_centerline, cylindrical_unwrap_frame, unwrap_points
from coilgen.surface_distance import SurfaceGeodesicSolver
from coilgen.wire_provenance import dedupe_path_with_segment_owners, nearest_source_segment_indices


@pytest.mark.parametrize("height", [0.04, 0.8])
@pytest.mark.parametrize("angle", [0.0, 0.71])
def test_cylindrical_artwork_frame_preserves_radius_for_all_aspect_ratios(height: float, angle: float) -> None:
    radius = 0.15
    axis = np.asarray([1.0, 2.0, 3.0])
    rotation = rotation_matrix_by_vector(axis, angle)
    mesh = build_cylinder_mesh(height, radius, 32, 6, *axis, angle)
    translation = np.asarray([0.23, -0.17, 0.45])
    mesh.vertices += translation

    frame = cylindrical_unwrap_frame(CoilPart(mesh=mesh))

    assert abs(float(np.dot(frame.axis_z, rotation[:, 2]))) == pytest.approx(1.0)
    assert frame.radius == pytest.approx(radius)
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    ring = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))
    unwrapped = unwrap_points(ring @ rotation.T + translation, frame)
    assert np.ptp(unwrapped[:, 0]) == pytest.approx(2.0 * np.pi * radius)
    assert np.ptp(unwrapped[:, 1]) < 1e-12


def test_pcb_artwork_keeps_a_long_straight_conducting_interval() -> None:
    centerline = np.column_stack((np.asarray([0.0, 0.001, 0.002, 0.102, 0.103]), np.zeros(5)))

    parts = _wire_parts_from_centerline(centerline, 0.0005)

    assert len(parts) == 1
    np.testing.assert_array_equal(parts[0].centerline, centerline)
    assert np.linalg.norm(np.diff(parts[0].centerline, axis=0), axis=1).sum() == pytest.approx(0.103)


def test_pcb_dxf_declares_the_same_millimetres_as_its_coordinates(tmp_path: Path) -> None:
    points = np.asarray([[0.0, 0.0], [0.025, 0.0], [0.025, 0.002], [0.0, 0.0]])
    layout = PcbLayout(upper_layer=[PcbWirePart(centerline=points, track_shape=points)])

    output = _save_single_pcb_dxf(layout, tmp_path / "coil.dxf")

    tokens = output.read_text(encoding="ascii").splitlines()
    unit_index = tokens.index("$INSUNITS")
    assert tokens[unit_index + 1 : unit_index + 3] == ["70", "4"]
    coordinate_values = [float(tokens[index + 1]) for index in range(0, len(tokens) - 1, 2) if tokens[index] == "10"]
    assert max(coordinate_values) == pytest.approx(25.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_trace_provenance_retains_distinct_visits_to_the_same_segment(reverse: bool) -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    target = source[::-1].copy() if reverse else source.copy()

    indices = nearest_source_segment_indices(source, target)

    np.testing.assert_array_equal(indices, [2, 1, 0] if reverse else [0, 1, 2])


def test_duplicate_removal_preserves_owners_of_coincident_route_branches() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    owners = [WireSegmentOwner(kind="contour", loop_index=index) for index in range(4)]

    deduped, transferred = dedupe_path_with_segment_owners(points, owners)

    np.testing.assert_array_equal(deduped, points[[0, 1, 2, 4]])
    assert [owner.loop_index for owner in transferred] == [0, 1, 3]


def _square_group(offset: float) -> ConnectedGroup:
    points = np.asarray(
        [[offset, 0.0, 0.0], [offset + 0.1, 0.0, 0.0], [offset + 0.1, 0.1, 0.0], [offset, 0.1, 0.0], [offset, 0.0, 0.0]]
    )
    return ConnectedGroup(v=points, return_path_v=np.empty((0, 3)), spiral_in_v=points, spiral_out_v=points)


def test_greedy_route_closes_through_the_requested_surface_connector() -> None:
    groups = [_square_group(0.0), _square_group(1.0)]
    solver = SimpleNamespace(
        distance_between_points=lambda start, end: SimpleNamespace(distance=np.linalg.norm(end - start))
    )
    bridges = []

    def connector(start, end, geodesic_solver, *, allow_direct_fallback):
        assert geodesic_solver is solver
        assert not allow_direct_fallback
        midpoint = 0.5 * (start + end) + np.asarray([0.0, 0.0, 0.1 * (len(bridges) + 1)])
        bridge = np.vstack((start, midpoint, end))
        bridges.append(bridge)
        return bridge

    with patch("coilgen.layout._inter_group_connector_path", side_effect=connector):
        route = _greedy_fuse_connected_groups(groups, solver, allow_direct_fallback=False)

    assert len(bridges) == 2
    np.testing.assert_allclose(bridges[-1][-1], groups[0].v[0])
    np.testing.assert_allclose(route.v[0], route.v[-1])
    assert all(np.any(np.all(np.isclose(route.v, bridge[1]), axis=1)) for bridge in bridges)


def test_greedy_route_rejects_a_missing_strict_surface_return() -> None:
    groups = [_square_group(0.0), _square_group(1.0)]
    solver = SimpleNamespace(
        distance_between_points=lambda start, end: SimpleNamespace(distance=np.linalg.norm(end - start))
    )
    bridge = np.vstack((groups[0].v[-1], groups[1].v[0]))
    with patch("coilgen.layout._inter_group_connector_path", side_effect=[bridge, np.empty((0, 3))]):
        with pytest.raises(ValueError, match="Unable to close"):
            _greedy_fuse_connected_groups(groups, solver, allow_direct_fallback=False)


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_same_triangle_geodesics_are_exact_symmetric_and_zero_on_identity(scale: float) -> None:
    translation = np.asarray([0.23, -0.17, 0.45])
    mesh = Mesh(
        vertices=scale * np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) + translation,
        faces=np.asarray([[0, 1, 2]]),
    )
    solver = SurfaceGeodesicSolver(mesh)
    point_a = scale * np.asarray([0.25, 0.25, 0.0]) + translation
    point_b = scale * np.asarray([0.4, 0.3, 0.0]) + translation

    identity = solver.distance_between_points(point_a, point_a)
    forward = solver.distance_between_points(point_a, point_b)
    backward = solver.distance_between_points(point_b, point_a)

    assert identity.distance == 0.0
    assert np.linalg.norm(np.diff(identity.path_v, axis=0), axis=1).sum() == 0.0
    assert forward.distance == pytest.approx(np.linalg.norm(point_b - point_a))
    assert backward.distance == forward.distance
    np.testing.assert_array_equal(forward.path_v, np.vstack((point_a, point_b)))
    np.testing.assert_array_equal(backward.path_v, forward.path_v[::-1])


def test_geodesic_shortcut_does_not_join_disconnected_triangles() -> None:
    lower = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    upper = lower + np.asarray([0.0, 0.0, 0.01])
    solver = SurfaceGeodesicSolver(Mesh(vertices=np.vstack((lower, upper)), faces=np.asarray([[0, 1, 2], [3, 4, 5]])))

    result = solver.distance_between_points(np.asarray([0.25, 0.25, 0.0]), np.asarray([0.25, 0.25, 0.01]))

    assert np.isinf(result.distance)
    assert len(result.path_v) == 0


def test_geodesic_shortcut_does_not_cross_a_surface_hole() -> None:
    vertices = np.asarray([[x, y, 0.0] for y in range(4) for x in range(4)], dtype=float)
    faces = []
    for y in range(3):
        for x in range(3):
            if x == y == 1:
                continue
            corner = 4 * y + x
            faces.extend([[corner, corner + 1, corner + 5], [corner, corner + 5, corner + 4]])
    solver = SurfaceGeodesicSolver(Mesh(vertices=vertices, faces=np.asarray(faces)))
    start, end = np.asarray([0.8, 1.5, 0.0]), np.asarray([2.2, 1.5, 0.0])

    result = solver.distance_between_points(start, end)

    assert result.distance > np.linalg.norm(end - start)
    assert len(result.path_v) > 2
    assert result.distance == pytest.approx(np.linalg.norm(np.diff(result.path_v, axis=0), axis=1).sum())


def test_geodesic_shortcut_requires_endpoints_on_the_face() -> None:
    solver = SurfaceGeodesicSolver(
        Mesh(
            vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.asarray([[0, 1, 2]]),
        )
    )
    start, end = np.asarray([0.2, 0.2, 0.01]), np.asarray([0.4, 0.2, 0.01])

    result = solver.distance_between_points(start, end)

    assert len(result.path_v) > 2
    assert result.distance > np.linalg.norm(end - start)
