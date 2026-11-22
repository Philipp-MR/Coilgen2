"""Drawing, directed topology, and physical bridge proposals share one model."""

from copy import deepcopy
import numpy as np
import pytest
from coilgen.interconnection_editor import InterconnectionEditor
from coilgen.models import WirePath
from coilgen.winding_surface import WindingSurfaceChart
from _coilgen_test_support.interconnection_editor import routed as routed

from _coilgen_test_support.winding_sketch import (
    crossing_editor as crossing_editor,
)


def test_freehand_route_preserves_attachments_direction_and_exact_undo():
    editor = crossing_editor()
    before = editor.to_plan()
    row = editor.section("s0")
    stroke = np.array([row.points[0], [-0.01, 0.012, 0], [0.01, 0.008, 0], row.points[-1]])
    editor.draw_connection("s0", row.target, editor.chart.project(stroke), flat=True)
    changed = editor.section("s0")
    assert len(changed.guides) == 2 and editor.undo_label == "Draw connection"
    np.testing.assert_array_equal(changed.points[[0, -1]], row.points[[0, -1]])
    assert editor.connectivity["complete"] and changed.target == row.target
    replay = crossing_editor()
    replay.load_plan(editor.to_plan())
    np.testing.assert_array_equal(replay.wire_path().v, editor.wire_path().v)
    editor.undo()
    assert editor.to_plan() == before
    editor.redo()
    np.testing.assert_array_equal(editor.wire_path().v, replay.wire_path().v)


def test_freehand_checks_lock_and_occupied_destination_atomically():
    editor = crossing_editor()
    row = editor.section("s0")
    editor.set_locked("s0", True)
    before = editor.to_plan()
    with pytest.raises(ValueError, match="unlocked"):
        editor.draw_connection("s0", row.target, row.points)
    assert editor.to_plan() == before
    editor.set_locked("s0", False)
    editor.disconnect("s0")
    before = editor.to_plan()
    with pytest.raises(ValueError, match="free input"):
        editor.draw_connection("s0", editor.section("s2").target, row.points)
    assert editor.to_plan() == before
    editor.draw_connection("s0", row.target, row.points)
    assert editor.connectivity["complete"]


def test_direction_markers_follow_path_reversal_and_group_palette():
    from coilgen.winding_visuals import direction_markers, group_color

    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    forward, reverse = direction_markers(points, 0.02), direction_markers(points[::-1], 0.02)
    assert len(forward) == 5
    for (p, d), (q, e) in zip(forward, reversed(reverse)):
        np.testing.assert_allclose(p, q)
        np.testing.assert_allclose(d, -e)
    assert len({group_color(i) for i in range(4)}) == 4


def test_route_coalesces_geodesic_roundoff_without_moving_attachments():
    from types import SimpleNamespace
    from coilgen.models import WirePathGeometry

    editor = crossing_editor()
    start, end = editor.section("s0").points[[0, -1]]
    path = np.array([start, start + [4e-13, 0, 0], [0.0, 0.0, 0.0], end - [9e-13, 0, 0], end])
    editor._solver = SimpleNamespace(route_between_points=lambda a, b: SimpleNamespace(distance=0.06, path_v=path))
    result = editor._route(start, end, (), WirePathGeometry.SURFACE)
    np.testing.assert_array_equal(result[[0, -1]], [start, end])
    assert np.all(np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-12)
    assert np.max(np.abs(result[:, 1:])) == 0


def test_reconnection_changes_graph_and_rejects_occupied_ports(routed):
    editor = InterconnectionEditor(routed.part)
    original = editor.to_plan()
    edges = [s for s in editor.sections if s.editable][:3]
    for s in edges:
        editor.disconnect(s.key)
    assert len(editor.ports) == 6
    editor.reconnect_endpoints(edges[0].key, edges[1].target)
    with pytest.raises(ValueError, match="free"):
        editor.reconnect_endpoints(edges[1].key, edges[1].target)
    editor.reconnect_endpoints(edges[1].key, edges[2].target)
    editor.reconnect_endpoints(edges[2].key, edges[0].target)
    assert editor.connectivity["complete"] and not editor.ports
    assert not np.array_equal(editor.wire_path().v, routed.part.wire_path.v)
    loaded = InterconnectionEditor(routed.part)
    loaded.load_plan(editor.to_plan())
    np.testing.assert_array_equal(loaded.wire_path().v, editor.wire_path().v)
    for _ in range(6):
        editor.undo()
    assert editor.to_plan() == original


def test_reconnection_cannot_certify_two_accidental_closed_loops(routed):
    editor = InterconnectionEditor(routed.part)
    edges = [s for s in editor.sections if s.editable][:2]
    for s in edges:
        editor.disconnect(s.key)
    editor.reconnect_endpoints(edges[0].key, edges[1].target)
    editor.reconnect_endpoints(edges[1].key, edges[0].target)
    assert editor.connectivity["free_ends"] == 0
    assert editor.connectivity["components"] == 2 and not editor.connectivity["complete"]
    with pytest.raises(ValueError, match="one directed circuit"):
        editor.wire_path()


def test_seam_moves_multiple_cuts_atomically_and_keeps_source(routed):
    editor = InterconnectionEditor(routed.part)
    rows = [s for s in editor.sections if s.cut is not None]
    assert rows
    original = editor.to_plan()
    # A drawn planar stroke across one real contour, away from its existing cut.
    row = rows[0]
    curve = editor.cut_curve(row.key)
    midpoint = (curve[3] + curve[4]) / 2
    direction = curve[4] - curve[3]
    perpendicular = np.cross(direction, [0, 0, 1])
    perpendicular *= 0.0003 / np.linalg.norm(perpendicular)
    preview = editor.preview_seam([midpoint - perpendicular, midpoint + perpendicular], [row.key])
    assert len(preview["cuts"]) == 1
    editor.apply_seam(preview)
    assert editor.undo_label == "Move seam"
    for before, after in zip(routed.part.contour_lines, editor.build_part().contour_lines):
        np.testing.assert_array_equal(before.v, after.v)
    with pytest.raises(ValueError, match="stale"):
        editor.apply_seam(preview)
    editor.undo()
    assert editor.to_plan() == original
    editor.set_locked(row.key, True)
    preview = editor.preview_seam([midpoint - perpendicular, midpoint + perpendicular], [row.key])
    before = editor.to_plan()
    with pytest.raises(ValueError, match="unlocked"):
        editor.apply_seam(preview)
    assert editor.to_plan() == before


def test_bridge_over_under_and_roundtrip_preserve_attachments():
    editor = crossing_editor()
    baseline = editor.to_plan()
    ends = editor.section("s0").points[[0, -1]].copy()
    editor.bridge_crossing("s0", "s2", "over", 0.004, 0.04)
    raised = editor.section("s0")
    assert raised.points[:, 2].max() == pytest.approx(0.004)
    np.testing.assert_array_equal(raised.points[[0, -1]], ends)
    np.testing.assert_array_equal(editor.section("s2").points, editor.source.wire_path.v[2:4])
    loaded = crossing_editor()
    loaded.load_plan(editor.to_plan())
    np.testing.assert_array_equal(loaded.wire_path().v, editor.wire_path().v)
    damaged = deepcopy(editor.to_plan())
    damaged["sections"][0]["bridge"]["height_m"] *= 2
    with pytest.raises(ValueError, match="ramp definition"):
        loaded.load_plan(damaged)
    editor.undo()
    assert editor.to_plan() == baseline
    editor.bridge_crossing("s0", "s2", "under", 0.004, 0.04)
    assert editor.section("s2").points[:, 2].max() == pytest.approx(0.004)
    assert editor.section("s0").points[:, 2].max() == 0
    editor.remove_bridge("s2")
    assert editor.to_plan() == baseline


def test_bridge_locks_and_insufficient_ramp_room_are_atomic():
    editor = crossing_editor()
    editor.set_locked("s2", True)
    before = editor.to_plan()
    with pytest.raises(ValueError, match="unlocked"):
        editor.bridge_crossing("s0", "s2", "under", 0.004, 0.04)
    assert editor.to_plan() == before
    with pytest.raises(ValueError, match="ramps"):
        editor.bridge_crossing("s0", "s2", "over", 0.004, 0.1)
    assert editor.to_plan() == before


def test_chart_preserves_actual_triangle_lengths_and_inverse(routed):
    chart = WindingSurfaceChart(routed.part.mesh)
    for face, uv in zip(routed.part.mesh.faces, chart.face_uv):
        triangle = routed.part.mesh.vertices[face]
        np.testing.assert_allclose(
            np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1),
            np.linalg.norm(triangle - np.roll(triangle, 1, axis=0), axis=1),
            atol=1e-12,
        )
        np.testing.assert_allclose(chart.pick(uv.mean(axis=0)), triangle.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("taper", [0.0, 0.04, -0.04])
def test_curved_chart_isometry_and_inverse_after_rotation(taper):
    from coilgen.geometry import build_cylinder_mesh

    mesh = build_cylinder_mesh(0.3, 0.18, 32, 4, 1, 0, 0, 0)
    mesh.vertices[:, :2] *= ((0.18 - taper * (mesh.vertices[:, 2] + 0.15) / 0.3) / 0.18)[:, None]
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [4, 5, 7], [2, 1, 4]]))
    mesh.vertices = mesh.vertices @ rotation + [0.1, -0.2, 0.3]
    chart = WindingSurfaceChart(mesh)
    triangles = mesh.vertices[mesh.faces]
    np.testing.assert_allclose(
        np.linalg.norm(chart.face_uv - np.roll(chart.face_uv, 1, axis=1), axis=2),
        np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2),
        atol=1e-12,
    )
    for uv, triangle in zip(chart.face_uv[::13], triangles[::13]):
        np.testing.assert_allclose(chart.pick(uv.mean(axis=0)), triangle.mean(axis=0), atol=1e-12)


def test_crossing_at_sampled_vertex_is_detected_once():
    from coilgen.winding_bridges import crossing_stations

    editor = crossing_editor()
    first = np.array([[-0.03, 0, 0], [0, 0, 0], [0.03, 0, 0]])
    second = np.array([[0, -0.03, 0], [0, 0, 0], [0, 0.03, 0]])
    hits = crossing_stations(editor.chart, first, second)
    assert len(hits) == 1
    assert hits[0][0] == pytest.approx(0.03)


def test_offset_guide_drag_keeps_the_attachment_height():
    from coilgen.models import WirePathGeometry

    part = crossing_editor().source
    part.wire_path.v[:, 2] = 0.005
    part.wire_path.segment_geometry = [WirePathGeometry.OFFSET_SURFACE] * 4
    editor = InterconnectionEditor(part)
    editor.drag_guide("s0", 0, [0, 0.005, 0], flat=True)
    assert editor.section("s0").guides[0][2] == pytest.approx(0.005)
    np.testing.assert_allclose(editor.section("s0").points[:, 2], 0.005, atol=1e-12)
    editor.set_guides("s0", [[0, 0.005, 0.008]])
    assert editor.section("s0").guides[0][2] == pytest.approx(0.005)


def test_flat_free_space_drag_keeps_depth_and_numeric_entry_can_change_it():
    from coilgen.models import WirePathGeometry

    part = crossing_editor().source
    part.wire_path.v[:, 2] = 0.005
    part.wire_path.segment_geometry = [WirePathGeometry.FREE_3D] * 4
    editor = InterconnectionEditor(part)
    editor.drag_guide("s0", 0, [0, 0.005, 0], flat=True)
    assert editor.section("s0").guides[0][2] == pytest.approx(0.005)
    editor.set_guides("s0", [[0, 0.005, 0.008]])
    assert editor.section("s0").guides[0][2] == pytest.approx(0.008)


def test_multiple_crossings_require_the_picked_location():
    part = crossing_editor().source
    points = np.array(
        [[-0.03, 0, 0], [0.03, 0, 0], [0, -0.03, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0, 0.03, 0], [-0.03, 0, 0]]
    )
    owners = part.wire_path.segment_owners
    part.wire_path = WirePath(points, [owners[0], owners[1], owners[2], owners[2], owners[2], owners[3]])
    editor = InterconnectionEditor(part)
    with pytest.raises(ValueError, match="more than once"):
        editor.bridge_crossing("s0", "s2", "over", 0.003, 0.006)
    editor.bridge_crossing("s0", "s2", "over", 0.003, 0.006, [0.0075, 0, 0])
    assert editor.section("s0").bridge["center_m"] == pytest.approx(0.0375)
