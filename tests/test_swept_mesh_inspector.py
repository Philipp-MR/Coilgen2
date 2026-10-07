from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import create_swept_coil_bodies
from coilgen.models import CoilPart, Mesh, SweptBody, WirePath
from coilgen.swept_mesh_inspector import SweptMeshInspector


def _triangle(z: float = 0.0) -> Mesh:
    return Mesh(np.array([[0.0, 0.0, z], [0.04, 0.0, z], [0.0, 0.04, z]]), np.array([[0, 1, 2]]))


@pytest.fixture
def figure():
    output = plt.figure()
    yield output
    inspector = getattr(output, "_coilgen_swept_mesh_inspector", None)
    if inspector is not None:
        inspector.disconnect()
    plt.close(output)


def test_prefer_per_contour_bodies_and_preserve_geometry_during_controls(figure) -> None:
    first = _triangle()
    second = _triangle(0.01)
    part = CoilPart(
        mesh=_triangle(-0.01),
        swept_body=first,
        swept_bodies=[SweptBody(first, contour_index=3), SweptBody(second, contour_index=7, turn_multiplicity=2)],
    )
    snapshots = [(mesh.vertices.copy(), mesh.faces.copy()) for mesh in (first, second, part.mesh)]
    inspector = SweptMeshInspector(figure, [part])
    assert len(inspector.bodies) == 2
    assert inspector.rendered_face_count == 2
    assert np.isinf(inspector.ax._focal_length)
    inspector.select_body(1)
    assert inspector.visible_indices == (1,)
    assert "contour 7" in inspector.selection_label.get_text()
    assert "multiplicity 2" in inspector.selection_label.get_text()
    assert inspector.rendered_face_count == 1
    inspector.set_mode("Edges only")
    inspector.set_opacity(0.35)
    inspector.set_view("XY")
    inspector.layers.set_active(0)
    inspector.layers.set_active(2)
    inspector.fit_all()
    figure.canvas.draw()
    assert inspector.rendered_face_count == 2
    assert [body.topology_counts for body in inspector.bodies] == [(3, 0), (3, 0)]
    artist_count = len(inspector.ax.collections)
    inspector.set_mode("Surface")
    inspector.set_mode("Surface + edges")
    assert len(inspector.ax.collections) == artist_count
    for mesh, (vertices, faces) in zip((first, second, part.mesh), snapshots):
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)


def test_legacy_mesh_and_unknown_or_skipped_checks_are_reported_honestly(figure) -> None:
    part = CoilPart(mesh=_triangle(), swept_body=_triangle())
    inspector = SweptMeshInspector(figure, [part])
    assert len(inspector.bodies) == 1
    assert "manufacturing: unknown" in inspector.summary.get_text()
    assert "watertight: unknown" in inspector.summary.get_text()
    assert "mesh collision: unknown" in inspector.summary.get_text()
    inspector.bodies[0].diagnostics.update(
        {
            "manufacturing_ok": False,
            "watertight": np.nan,
            "sweep_mesh_self_intersection": False,
            "sweep_mesh_collision_check_skipped": True,
        }
    )
    inspector.update(draw=False)
    assert "manufacturing: no" in inspector.summary.get_text()
    assert "watertight: unknown" in inspector.summary.get_text()
    assert "mesh collision: not checked" in inspector.summary.get_text()


def test_nonmanifold_edges_and_invalid_faces_are_visible_in_diagnostics(figure) -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float) * 0.01
    mesh = Mesh(vertices, np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4], [0, 1, 99]]))
    inspector = SweptMeshInspector(figure, [CoilPart(mesh=_triangle(), swept_body=mesh)])
    inspector.layers.set_active(2)
    assert inspector.bodies[0].topology_counts == (6, 1)
    assert inspector.rendered_face_count == 3
    assert "1 invalid/nonfinite triangles omitted" in inspector.summary.get_text()
    assert "not a collision or solid-validity certificate" in inspector.summary.get_text()
    figure.canvas.draw()


def test_actual_swept_mesh_and_fabrication_path_remain_full_resolution(figure) -> None:
    support = normalize_mesh(Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), np.array([[0, 1, 2]])))
    part = CoilPart(mesh=support, wire_path=WirePath(v=np.array([[0, 0, 0], [0.05, 0, 0], [0.05, 0.05, 0]])))
    part = create_swept_coil_bodies(
        [part],
        CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.002,
            conductor_cross_section_height=0.001,
        ),
    )[0]
    vertices = part.swept_body.vertices.copy()
    faces = part.swept_body.faces.copy()
    inspector = SweptMeshInspector(figure, [part])
    assert inspector.rendered_face_count == len(faces)
    inspector.layers.set_active(1)
    assert inspector.bodies[0].path_artist is not None
    np.testing.assert_allclose(
        np.column_stack(inspector.bodies[0].path_artist.get_data_3d()), part.fabrication_path.wire_points_m
    )
    inspector.layers.set_active(0)
    inspector.layers.set_active(2)
    inspector.set_mode("Edges only")
    figure.canvas.draw()
    np.testing.assert_array_equal(part.swept_body.vertices, vertices)
    np.testing.assert_array_equal(part.swept_body.faces, faces)
    inspector.disconnect()
    inspector.disconnect()
    assert inspector._disconnected


def test_missing_sweep_is_an_explicit_error(figure) -> None:
    with pytest.raises(ValueError, match="No nonempty swept conductor meshes"):
        SweptMeshInspector(figure, [SimpleNamespace(swept_body=None, swept_bodies=[])])


def test_fit_uses_drawable_faces_and_preserves_original_part_identity(figure) -> None:
    mesh = _triangle()
    mesh.vertices = np.vstack((mesh.vertices, np.full((1, 3), 1e6)))
    inspector = SweptMeshInspector(figure, [CoilPart(mesh=_triangle(), swept_body=mesh)], part_indices=(3,))
    inspector.select_body(0)
    assert "part 3" in inspector.selection_label.get_text()
    assert inspector.ax.get_xlim()[1] < 0.1
    assert "1 vertices outside drawable faces excluded from fit" in inspector.summary.get_text()


def test_all_invalid_faces_have_an_explicit_empty_state(figure) -> None:
    mesh = Mesh(np.zeros((3, 3)), np.array([[0, 1, 99]]))
    inspector = SweptMeshInspector(figure, [CoilPart(mesh=_triangle(), swept_body=mesh)])
    assert inspector.rendered_face_count == 0
    assert inspector.empty_label.get_visible()
    assert "1 invalid/nonfinite triangles omitted" in inspector.summary.get_text()
    figure.canvas.draw()


def test_fit_all_includes_enabled_support_but_fit_selected_stays_on_conductor(figure) -> None:
    support = _triangle()
    support.vertices *= 10.0
    inspector = SweptMeshInspector(figure, [CoilPart(mesh=support, swept_body=_triangle())])
    inspector.layers.set_active(0)
    inspector.fit_all()
    assert inspector.ax.get_xlim()[1] >= 0.4
    inspector.fit_selected()
    assert inspector.ax.get_xlim()[1] < 0.1


def test_stored_manufacturing_failures_follow_body_selection_without_implying_mesh_collisions(figure) -> None:
    reasons = (
        "route_clearance_violation;sweep_clearance_violation;sweep_self_intersection;"
        "curvature_envelope_violation;poor_sweep_triangle_quality"
    )
    failing = {"manufacturing_ok": False, "watertight": True, "manufacturing_failure_reasons": reasons}
    second = {"manufacturing_ok": False, "manufacturing_failure_reasons": "sweep_clearance_violation"}
    part = CoilPart(
        mesh=_triangle(),
        swept_bodies=[
            SweptBody(_triangle(), contour_index=1, diagnostics=failing.copy()),
            SweptBody(_triangle(0.01), contour_index=2, diagnostics=second.copy()),
            SweptBody(_triangle(0.02), contour_index=3, diagnostics={"manufacturing_ok": False}),
            SweptBody(_triangle(0.03), contour_index=4, diagnostics={"manufacturing_ok": True}),
        ],
    )
    inspector = SweptMeshInspector(figure, [part])
    failure_text = " ".join(inspector.failure_label.get_text().split())
    assert "sweep_clearance_violation (2)" in failure_text
    assert "reasons unavailable (1)" in failure_text
    inspector.select_body(0)
    for reason in reasons.split(";"):
        assert reason in inspector.failure_label.get_text()
    assert "watertight: yes" in inspector.summary.get_text()
    assert "mesh collision: unknown" in inspector.summary.get_text()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert inspector.failure_label.get_window_extent(renderer).y0 > inspector.ax.get_window_extent(renderer).y1
    assert part.swept_bodies[0].diagnostics == failing
    assert part.swept_bodies[1].diagnostics == second
    inspector.select_body(3)
    assert inspector.failure_label.get_text() == ""


def _long_ring_mesh(offset: float = 0.0) -> Mesh:
    section = 0.001 * np.array([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]])
    centers = np.column_stack((offset + 0.001 * np.arange(100), np.zeros(100), np.zeros(100)))
    vertices = (centers[:, None, :] + section[None, :, :]).reshape((-1, 3))
    faces = []
    for ring in range(99):
        for side in range(4):
            first, following = ring * 4 + side, ring * 4 + (side + 1) % 4
            faces.extend(([first, following, first + 4], [following, following + 4, first + 4]))
    return Mesh(vertices, np.asarray(faces))


def test_inspect_issue_cycles_stored_locations_and_fits_actual_rings_without_changing_geometry(figure):
    first = _long_ring_mesh()
    second = _long_ring_mesh(0.2)
    data = {
        "sweep_ring_count": 100,
        "sweep_section_point_count": 4,
        "sweep_closed_path": False,
        "sweep_local_fold_violation": True,
        "sweep_worst_quad_ring": 10,
        "sweep_worst_quad_section": 2,
        "sweep_profile_jacobian_violation": True,
        "sweep_profile_jacobian_worst_ring": 70,
        "sweep_profile_jacobian_worst_section": 1,
        "sweep_final_frame_rotation_violation": True,
        "sweep_final_frame_rotation_worst_interval": 95,
    }
    second_data = {
        "sweep_ring_count": 100,
        "sweep_section_point_count": 4,
        "sweep_local_fold_violation": True,
        "sweep_worst_quad_ring": 20,
    }
    part = CoilPart(
        mesh=_triangle(), swept_bodies=[SweptBody(first, diagnostics=data), SweptBody(second, diagnostics=second_data)]
    )
    snapshots = [(mesh.vertices.copy(), mesh.faces.copy()) for mesh in (first, second)]
    inspector = SweptMeshInspector(figure, [part])
    inspector.inspect_issue()
    assert inspector.selected_index == 0
    assert "folded quad" in inspector.failure_label.get_text()
    assert "ring 11, section 3" in inspector.failure_label.get_text()
    assert inspector.ax.get_xlim()[1] - inspector.ax.get_xlim()[0] < 0.01
    np.testing.assert_allclose(inspector._issue_marker.get_data_3d(), first.vertices[10 * 4 + 2, :, None])
    inspector.inspect_issue()
    assert "profile Jacobian" in inspector.failure_label.get_text()
    assert inspector.ax.get_xlim()[0] > 0.06
    inspector.inspect_issue()
    assert "frame rotation" in inspector.failure_label.get_text()
    inspector.inspect_issue()
    assert inspector.selected_index == 1
    assert inspector.ax.get_xlim()[0] > 0.21
    inspector.inspect_issue()
    assert inspector.selected_index == 0
    assert "Issue 1 / 4" in inspector.failure_label.get_text()
    inspector.select_body(1)
    inspector.inspect_issue()
    assert "Issue 1 / 1" in inspector.failure_label.get_text()
    inspector.fit_all()
    assert not inspector._issue_marker.get_visible()
    assert "Issue" not in inspector.failure_label.get_text()
    assert inspector.rendered_face_count == len(first.faces) + len(second.faces)
    for mesh, snapshot in zip((first, second), snapshots):
        np.testing.assert_array_equal(mesh.vertices, snapshot[0])
        np.testing.assert_array_equal(mesh.faces, snapshot[1])
    figure.canvas.draw()


@pytest.mark.parametrize(
    "metadata",
    [
        {"manufacturing_ok": False, "manufacturing_failure_reasons": "folded_sweep_quad"},
        {
            "sweep_ring_count": 100,
            "sweep_section_point_count": 4,
            "sweep_worst_quad_ring": 10,
            "sweep_local_fold_violation": False,
        },
        {
            "sweep_ring_count": 100,
            "sweep_section_point_count": 4,
            "sweep_worst_quad_ring": 999,
            "sweep_local_fold_violation": True,
        },
    ],
)
def test_inspect_issue_reports_unavailable_for_legacy_inactive_or_invalid_locations(figure, metadata):
    inspector = SweptMeshInspector(
        figure, [CoilPart(mesh=_triangle(), swept_body=_long_ring_mesh(), manufacturing_diagnostics=metadata)]
    )
    before = (inspector.ax.get_xlim(), inspector.ax.get_ylim(), inspector.ax.get_zlim())
    inspector.inspect_issue()
    assert "No stored failure locations" in inspector.failure_label.get_text()
    assert inspector._issue_marker is None
    assert (inspector.ax.get_xlim(), inspector.ax.get_ylim(), inspector.ax.get_zlim()) == before


def test_inspect_issue_can_use_matching_stored_fabrication_path_without_mesh_ring_layout(figure):
    points = np.column_stack((0.001 * np.arange(10), np.zeros(10), np.zeros(10)))
    part = CoilPart(
        mesh=_triangle(),
        swept_body=_triangle(),
        manufacturing_diagnostics={
            "sweep_ring_count": 10,
            "sweep_final_frame_rotation_violation": True,
            "sweep_final_frame_rotation_worst_interval": 5,
        },
    )
    part.fabrication_path = SimpleNamespace(points_m=points, wire_points_m=points, closed=False)
    inspector = SweptMeshInspector(figure, [part])
    inspector.inspect_issue()
    assert "stored fabrication path" in inspector.failure_label.get_text()
    np.testing.assert_allclose(inspector._issue_marker.get_data_3d(), points[5, :, None])
    assert inspector.ax.get_xlim()[1] < 0.01


@pytest.mark.parametrize(("view", "collapsed_axis"), [("XY", 2), ("XZ", 1), ("YZ", 0)])
def test_axial_presets_hide_only_end_on_annotations_and_orbit_restores_them(figure, view, collapsed_axis) -> None:
    from matplotlib.backend_bases import MouseEvent

    inspector = SweptMeshInspector(figure, [CoilPart(mesh=_triangle(), swept_body=_triangle())])
    axes = (inspector.ax.xaxis, inspector.ax.yaxis, inspector.ax.zaxis)
    original_bounds = np.array([inspector.ax.get_xlim(), inspector.ax.get_ylim(), inspector.ax.get_zlim()])
    inspector.set_view(view)
    inspector.fit_selected()
    for index, axis in enumerate(axes):
        assert axis.label.get_visible() == (index != collapsed_axis)
        assert bool(len(axis.get_majorticklocs())) == (index != collapsed_axis)
    np.testing.assert_array_equal(
        [inspector.ax.get_xlim(), inspector.ax.get_ylim(), inspector.ax.get_zlim()], original_bounds
    )
    inspector.ax.view_init(elev=23, azim=-47)
    MouseEvent("button_release_event", figure.canvas, 100, 100)._process()
    assert all(axis.label.get_visible() and len(axis.get_majorticklocs()) for axis in axes)
    inspector.set_view(view)
    inspector.set_view("Isometric")
    assert all(axis.label.get_visible() and len(axis.get_majorticklocs()) for axis in axes)
