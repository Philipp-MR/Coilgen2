"""Native editor controls, asynchronous edits, and real 3D picking."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from coilgen.gui.interconnection_editor import InterconnectionEditorDialog, WindingViewport
from coilgen.interconnection_validation import EditorValidation
from _coilgen_test_support.interconnection_editor import routed as routed, surface_connection, inner_cut, simple_session
from _coilgen_test_support.workbench_usability import application as application

from _coilgen_test_support.gui_interconnection_editor import (
    Viewport as Viewport,
    close as close,
    finish as finish,
    session_from_case as session_from_case,
)


@pytest.mark.parametrize("mode", ["draw", "connect", "select"])
def test_cut_outline_does_not_intercept_active_edit_tools(application, mode):
    class PickingViewport(WindingViewport):
        def __init__(self):
            QWidget.__init__(self)

    view = PickingViewport()
    outline = SimpleNamespace(GetAddressAsString=lambda _: "outline")
    handle = SimpleNamespace(GetAddressAsString=lambda _: "handle")
    view.plotter = SimpleNamespace(
        interactor=QWidget(view), renderer=SimpleNamespace(actors={"outline": outline, "handle": handle})
    )
    row = SimpleNamespace(
        points=np.array([[0.0, 0, 0], [0.01, 0, 0]]),
        locked=False,
        disconnected=False,
        bridge=None,
        editable=True,
        geometry=("surface",),
    )
    view.editor = SimpleNamespace(section=lambda key: row)
    view.busy, view.mode, view._drag, view._sketch = False, mode, None, None
    view._handles = {"handle": ("s0", "guide" if mode == "select" else "out", 0)}
    view._object_actors, view._issue_actors, view._actors = {"outline": "plane"}, {}, {}
    # Both actors overlap. Picking the outline first reproduces the real failure.
    view._pick = lambda position, actors: (actors[0] if actors else None, np.zeros(3))
    selected_objects = []
    view.object_selected.connect(selected_objects.append)
    event = SimpleNamespace(
        type=lambda: QEvent.Type.MouseButtonPress,
        button=lambda: Qt.MouseButton.LeftButton,
        modifiers=lambda: Qt.KeyboardModifier.NoModifier,
        position=lambda: QPointF(10, 10),
    )
    assert view.eventFilter(view.plotter.interactor, event)
    assert not selected_objects
    if mode == "select":
        assert view._drag[:3] == ("s0", "guide", 0)
    else:
        assert view._sketch["port"] == ("s0", "out", 0)
    view.deleteLater()


def test_native_cut_scope_can_narrow_an_existing_preview_to_one_turn(application, routed, tmp_path):
    session = session_from_case(routed, tmp_path)
    dialog = InterconnectionEditorDialog(session, viewport_factory=Viewport)
    dialog.show()
    cut = inner_cut(dialog.editor)
    dialog.select_section(cut.key)

    dialog.mode.setCurrentIndex(dialog.mode.findData("shape"))
    finish(application, dialog)
    dialog.shape_group.setCurrentIndex(dialog.shape_group.findData("selected"))
    finish(application, dialog)
    assert dialog._seam_preview["keys"] == [cut.key]
    assert [c["key"] for c in dialog._seam_preview["cuts"]] == [cut.key]
    dialog.route_action.trigger()
    finish(application, dialog)
    assert dialog.editor.cut_objects[0]["keys"] == [cut.key]
    close(dialog)


def test_graphical_freehand_gesture_and_compact_controls(application, routed, tmp_path):
    session = session_from_case(routed, tmp_path)
    dialog = InterconnectionEditorDialog(session, viewport_factory=Viewport)
    dialog.show()
    dialog.view_layout.setCurrentIndex(2)
    key = surface_connection(dialog.editor).key
    dialog.select_section(key)
    dialog.mode.setCurrentIndex(dialog.mode.findData("draw"))
    application.processEvents()
    assert not dialog.inspector.isVisible()
    assert dialog.route_action.text() == "Generate connections"
    row = dialog.editor.section(key)
    view = dialog.flat_view
    uv = view.chart.project([row.points[0], row.points.mean(axis=0) + [0, 0.0008, 0], row.points[-1]])
    positions = [view.screen(p).toPoint() for p in uv]
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=positions[0])
    QTest.mouseMove(view, positions[1], delay=10)
    QTest.mouseMove(view, positions[2], delay=10)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=positions[2])
    finish(application, dialog)
    assert dialog._seam_preview and dialog._seam_preview.get("drawing")
    np.testing.assert_array_equal(dialog.editor.section(key).points, row.points)
    dialog._generate()
    finish(application, dialog)
    assert dialog.editor.undo_label == "Draw connection", dialog.hint.text()
    assert dialog.editor.section(key).guides
    np.testing.assert_array_equal(dialog.editor.section(key).points[[0, -1]], row.points[[0, -1]])
    close(dialog)


def test_cut_reconnect_undo_redo_lock_and_stale_validation(application, routed, tmp_path):
    session = session_from_case(routed, tmp_path)
    applied = []
    dialog = InterconnectionEditorDialog(
        session, apply_plan=lambda p: applied.append(p) or True, viewport_factory=Viewport
    )
    key = surface_connection(dialog.editor).key
    dialog.select_section(key)
    assert not dialog.apply_button.isEnabled()
    dialog._disconnect()
    assert not dialog.validate_button.isEnabled()
    assert dialog.cut_action.text() == "Reconnect"
    dialog._undo()
    assert dialog.validate_button.isEnabled()
    dialog._redo()
    assert not dialog.editor.connectivity["complete"]
    dialog._disconnect()
    session.accept_validation(EditorValidation(session.identity, True, ()))
    dialog._refresh()
    assert dialog.apply_button.isEnabled()
    dialog._apply()
    assert len(applied) == 1
    dialog._lock()
    assert not dialog.cut_action.isEnabled() and not dialog.apply_button.isEnabled()
    close(dialog)


def test_numeric_guide_commits_off_thread_and_updates_view(application, routed, tmp_path):
    dialog = InterconnectionEditorDialog(session_from_case(routed, tmp_path), viewport_factory=Viewport)
    section = surface_connection(dialog.editor)
    dialog.select_section(section.key)
    for control, value in zip(dialog.xyz, section.points.mean(axis=0) + [0, 0.0006, 0]):
        control.setValue(value * 1e3)
    dialog._set_guide_numeric()
    assert dialog._busy and not dialog.toolbar.isEnabled()
    finish(application, dialog)
    assert dialog.editor.section(section.key).guides
    assert dialog.viewport.displayed[1] == section.key
    assert "pending" in dialog.validation_text.toPlainText()
    close(dialog)


def test_engineering_worker_can_be_cancelled_without_a_passing_receipt(application, tmp_path):
    session, _ = simple_session(tmp_path)
    dialog = InterconnectionEditorDialog(session, viewport_factory=Viewport)
    before = session.identity
    dialog._validate()
    dialog._cancel()
    finish(application, dialog)
    assert session.identity == before
    assert session.validation is None
    assert "cancel" in dialog.hint.text().lower()
    assert not dialog.apply_button.isEnabled()
    close(dialog)


@pytest.mark.skipif(
    os.environ.get("COILGEN_NATIVE_GUI_TESTS") != "1",
    reason="Native VTK picking requires a working OpenGL desktop; set COILGEN_NATIVE_GUI_TESTS=1.",
)
def test_native_3d_drag_creates_one_undoable_routing_edit(application, routed, tmp_path):
    pytest.importorskip("pyvistaqt")
    dialog = InterconnectionEditorDialog(session_from_case(routed, tmp_path))
    dialog.show()
    application.processEvents()
    section = surface_connection(dialog.editor)
    dialog.select_section(section.key)
    viewport = dialog.viewport
    # Handles stay readable in screen pixels when zooming close to a cut.
    handles = [a for a in viewport.plotter.renderer.actors.values() if a.GetAddressAsString("") in viewport._handles]
    assert handles and all(a.GetProperty().GetPointSize() == 15 for a in handles)
    viewport.plotter.view_xy()
    viewport.plotter.reset_camera()
    viewport.plotter.render()
    application.processEvents()
    renderer = viewport.plotter.renderer
    point = section.points[len(section.points) // 2]
    renderer.SetWorldPoint(*point, 1)
    renderer.WorldToDisplay()
    x, y, _ = renderer.GetDisplayPoint()
    ratio = viewport.plotter.interactor.devicePixelRatioF()
    position = QPoint(round(x / ratio), round(viewport.plotter.interactor.height() - y / ratio - 1))
    revision = dialog.editor.revision
    QTest.mousePress(viewport.plotter.interactor, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseMove(viewport.plotter.interactor, position + QPoint(8, -12), delay=30)
    assert dialog.editor.revision == revision
    QTest.mouseRelease(viewport.plotter.interactor, Qt.MouseButton.LeftButton, pos=position + QPoint(8, -12))
    finish(application, dialog)
    assert dialog.editor.revision == revision + 1, dialog.hint.text()
    np.testing.assert_array_equal(dialog.editor.section(section.key).points[[0, -1]], section.points[[0, -1]])
    dialog._undo()
    np.testing.assert_array_equal(dialog.editor.section(section.key).points, section.points)
    cut = inner_cut(dialog.editor)
    dialog.select_section(cut.key)

    def screen(point):
        renderer.SetWorldPoint(*point, 1)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        return QPoint(round(x / ratio), round(viewport.plotter.interactor.height() - y / ratio - 1))

    start = screen(dialog.editor.cut_point(cut.key))
    end = screen(dialog.editor.cut_at_position(cut.key, cut.cut[0] + 0.001))
    revision = dialog.editor.revision
    QTest.mousePress(viewport.plotter.interactor, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(viewport.plotter.interactor, end, delay=30)
    QTest.mouseRelease(viewport.plotter.interactor, Qt.MouseButton.LeftButton, pos=end)
    finish(application, dialog)
    assert dialog.editor.revision == revision + 1, dialog.hint.text()
    assert dialog.editor.section(cut.key).cut != cut.cut
    assert dialog.grab().save(str(tmp_path / "editor.png"))
    viewport.plotter.screenshot(str(tmp_path / "viewport.png"))
    close(dialog)
