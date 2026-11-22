"""Editing a doubly curved support must use physical facets, never a fake flat map."""

from copy import deepcopy
import numpy as np
import pytest

from coilgen.geometry import build_spherical_cap_mesh
from coilgen.interconnection_editor import InterconnectionEditor
from coilgen.models import CoilPart, WirePath, WireSegmentOwner
from _coilgen_test_support.workbench_usability import application as application


@pytest.fixture(params=[1.0, 0.7], ids=["hemisphere", "ellipsoidal-cap"])
def curved_part(request):
    mesh = build_spherical_cap_mesh(0.08, np.pi / 2, 24, 8, 1, 0, 0, 0, 0, 0, 0)
    mesh.vertices[:, 0] *= request.param
    # A real mesh-face boundary gives exact surface endpoints and a closed circuit.
    triangle = mesh.vertices[mesh.faces[len(mesh.faces) // 2]]
    points = np.vstack([triangle, triangle[0]])
    owners = [WireSegmentOwner("within_group_connector", group_index=0)]
    owners += [WireSegmentOwner("contour_group", group_index=0)] * 2
    return CoilPart(mesh=mesh, wire_path=WirePath(points, owners), contour_step=1.0)


def test_curved_surface_sketch_reconnect_lock_undo_and_replay(curved_part):
    editor = InterconnectionEditor(curved_part)
    original = deepcopy(editor.to_plan())
    row = editor.section("s0")
    with pytest.raises(ValueError, match="planar, cylindrical or conical"):
        _ = editor.chart
    # A deliberate lateral detour, projected onto the actual faceted support.
    guide = editor.project(row.points.mean(axis=0) + [0.003, -0.004, 0.002])
    editor.draw_connection(row.key, row.target, [row.points[0], guide, row.points[-1]])
    drawn = editor.section(row.key)
    assert drawn.guides and editor.connectivity["complete"]
    np.testing.assert_array_equal(drawn.points[[0, -1]], row.points[[0, -1]])
    t = np.linspace(0, 1, 7)[None, :, None]
    samples = (drawn.points[:-1, None] * (1 - t) + drawn.points[1:, None] * t).reshape(-1, 3)
    feet, _ = editor._projector.project(samples)
    assert np.linalg.norm(samples - feet, axis=1).max() < 1e-9
    saved = editor.to_plan()
    replay = InterconnectionEditor(curved_part)
    replay.load_plan(saved)
    np.testing.assert_array_equal(replay.wire_path().v, editor.wire_path().v)
    editor.undo()
    assert editor.to_plan() == original
    editor.redo()
    assert editor.to_plan() == saved
    editor.set_locked(row.key, True)
    with pytest.raises(ValueError, match="unlocked"):
        editor.draw_connection(row.key, row.target, row.points)
    editor.set_locked(row.key, False)
    editor.disconnect(row.key)
    assert not editor.connectivity["complete"]
    editor.reconnect_endpoints(row.key, row.target)
    assert editor.connectivity["complete"]


def test_native_curved_surface_disables_flat_and_seam(application, curved_part, tmp_path):
    from types import SimpleNamespace
    from coilgen.config import CoilGenConfig
    from coilgen.gui.interconnection_editor import InterconnectionEditorDialog
    from _coilgen_test_support.gui_interconnection_editor import Viewport, session_from_case, close

    session = session_from_case(SimpleNamespace(part=curved_part, config=CoilGenConfig()), tmp_path)
    dialog = InterconnectionEditorDialog(session, viewport_factory=Viewport)
    assert dialog.view_layout.currentText() == "3D"
    assert not dialog.view_layout.model().item(0).isEnabled()
    assert not dialog.view_layout.model().item(2).isEnabled()
    assert not dialog.mode.model().item(1).isEnabled()
    assert dialog.mode.model().item(4).isEnabled()
    assert "planar, cylindrical or conical" in dialog.view_layout.toolTip()
    close(dialog)
