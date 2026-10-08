"""Camera navigation must not activate object settings on mouse press."""

import numpy as np
import pytest
import test_native_workbench as native

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDockWidget


application = native.application
scientific_viewport = native.scientific_viewport
window = native.window


@pytest.fixture
def scene(scientific_viewport):
    viewport = scientific_viewport
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.01, 0.0, 0.0]])
    fields = np.array([[0.0, 0.0, 1e-6], [0.0, 0.0, 2e-6], [0.0, 0.0, 3e-6]])
    viewport.display_run({"target_points_m": points, "target_field_T": fields})
    viewport.units.setCurrentText("µT")
    actor = viewport.plotter.meshes[-1][2]
    viewport.plotter.picked_point = points[1]
    selected = []
    viewport.selected.connect(selected.append)
    return viewport, actor, selected


def mouse(
    viewport,
    kind,
    position=(30, 30),
    *,
    button=Qt.MouseButton.NoButton,
    buttons=Qt.MouseButton.NoButton,
    modifiers=Qt.KeyboardModifier.NoModifier,
):
    widget = viewport.plotter.interactor
    local = QPoint(*position)
    event = QMouseEvent(kind, QPointF(local), QPointF(widget.mapToGlobal(local)), button, buttons, modifiers)
    QCoreApplication.sendEvent(widget, event)


def press(viewport, actor, *, modifiers=Qt.KeyboardModifier.NoModifier):
    mouse(
        viewport,
        QEvent.Type.MouseButtonPress,
        button=Qt.MouseButton.LeftButton,
        buttons=Qt.MouseButton.LeftButton,
        modifiers=modifiers,
    )
    # VTK launches picking while handling the press, after the Qt event filter.
    viewport.plotter.pick(actor)


def release(viewport, position=(30, 30), *, modifiers=Qt.KeyboardModifier.NoModifier):
    mouse(viewport, QEvent.Type.MouseButtonRelease, position, button=Qt.MouseButton.LeftButton, modifiers=modifiers)


def test_pick_waits_for_stationary_release_and_keeps_sample_probe(scene):
    viewport, actor, selected = scene
    press(viewport, actor)
    assert selected == []
    assert viewport.probe.text() == "Probe: —"

    jitter = max(0, QApplication.startDragDistance() - 1)
    mouse(viewport, QEvent.Type.MouseMove, (30 + jitter, 30), buttons=Qt.MouseButton.LeftButton)
    release(viewport, (30 + jitter, 30))

    assert selected == ["target"]
    assert "Nearest displayed sample" in viewport.probe.text()
    assert "Bz [µT]: 2" in viewport.probe.text()


def test_rotation_drag_and_return_to_start_never_inspect(scene):
    viewport, actor, selected = scene
    press(viewport, actor)
    distance = max(1, QApplication.startDragDistance())
    mouse(viewport, QEvent.Type.MouseMove, (30 + distance, 30), buttons=Qt.MouseButton.LeftButton)
    mouse(viewport, QEvent.Type.MouseMove, buttons=Qt.MouseButton.LeftButton)
    release(viewport)

    assert selected == []
    assert viewport.probe.text() == "Probe: —"


@pytest.mark.parametrize("modifier", [Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.ControlModifier])
def test_modified_camera_gesture_does_not_inspect(scene, modifier):
    viewport, actor, selected = scene
    press(viewport, actor, modifiers=modifier)
    release(viewport, modifiers=modifier)
    assert selected == []


@pytest.mark.parametrize("interruption", [QEvent.Type.Leave, QEvent.Type.FocusOut, QEvent.Type.Hide])
def test_interrupted_gesture_does_not_dispatch_a_stale_pick(scene, interruption):
    viewport, actor, selected = scene
    press(viewport, actor)
    QCoreApplication.sendEvent(viewport.plotter.interactor, QEvent(interruption))
    release(viewport)
    assert selected == []

    press(viewport, actor)
    release(viewport)
    assert selected == ["target"]


def test_other_mouse_button_cancels_pending_left_pick(scene):
    viewport, actor, selected = scene
    press(viewport, actor)
    mouse(
        viewport,
        QEvent.Type.MouseButtonPress,
        button=Qt.MouseButton.RightButton,
        buttons=Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton,
    )
    mouse(
        viewport, QEvent.Type.MouseButtonRelease, button=Qt.MouseButton.RightButton, buttons=Qt.MouseButton.LeftButton
    )
    release(viewport)
    assert selected == []


def test_keyboard_or_programmatic_pick_still_inspects_immediately(scene):
    viewport, actor, selected = scene
    viewport.plotter.pick(actor)
    assert selected == ["target"]
    assert "Nearest displayed sample" in viewport.probe.text()


def test_drag_keeps_winding_settings_closed_and_model_unchanged(scene, window, application):
    viewport, actor, _selected = scene
    viewport._actors[viewport._actor_key(actor)] = "winding"
    viewport.selected.connect(window._inspect_node)
    window.show()
    application.processEvents()
    settings = window.findChild(QDockWidget, "settings_dock")
    settings.hide()
    before = window.project
    undo_count = window.undo_stack.count()

    press(viewport, actor)
    assert settings.isHidden()
    distance = max(1, QApplication.startDragDistance())
    mouse(viewport, QEvent.Type.MouseMove, (30 + distance, 30), buttons=Qt.MouseButton.LeftButton)
    release(viewport, (30 + distance, 30))
    application.processEvents()

    assert settings.isHidden()
    assert window.project is before
    assert window.project.revision == before.revision
    assert window.undo_stack.count() == undo_count

    press(viewport, actor)
    release(viewport)
    application.processEvents()
    assert not settings.isHidden()
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "winding"
    assert window.project is before
    assert window.undo_stack.count() == undo_count
