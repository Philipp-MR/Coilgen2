"""Native object manipulation updates the recipe and invalidates final meshes."""

import time

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("pyvistaqt")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from coilgen.application.presets import preset_config  # noqa: E402
from coilgen.application.project import ProjectDocument  # noqa: E402
from coilgen.gui.surface_editor import SurfaceEditorDialog  # noqa: E402


def wait(dialog, app):
    until = time.monotonic() + 20
    app.processEvents()
    while dialog.task is not None and time.monotonic() < until:
        QTest.qWait(10)
        app.processEvents()
    assert dialog.task is None, dialog.status.text()
    assert dialog.editor.scene is not None, dialog.status.text()


def test_native_drag_rotation_subtraction_face_deletion_mesh_and_apply(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    if app.platformName() == "offscreen":
        # Qt's offscreen platform has no native OpenGL context on Windows.
        # Exercise controls, actors and geometry here; the on-screen smoke run
        # additionally checks VTK rendering and picking on a real window.
        from pyvistaqt import QtInteractor

        monkeypatch.setattr(QtInteractor, "_render", lambda self: None)
    dialog = SurfaceEditorDialog(ProjectDocument.from_config(preset_config("cylinder")))
    dialog.show()
    try:
        wait(dialog, app)
        assert not dialog.apply_button.isEnabled()
        dialog.kind.setCurrentText("sphere")
        dialog._add()
        wait(dialog, app)
        selected = dialog.selected
        original = dialog.node["center_m"][0]
        dialog.mode.setCurrentIndex(dialog.mode.findData("move"))
        target = dialog.view.interactor
        center = QPoint(target.width() // 2, target.height() // 2)
        QTest.mousePress(target, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(target, center + QPoint(25, 0))
        assert dialog.node["center_m"][0] != original
        assert f"{dialog.node['center_m'][0] * 1000:.2f}" in dialog.bar_buttons[selected].text()
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, pos=center + QPoint(25, 0))
        wait(dialog, app)
        dialog.mode.setCurrentIndex(dialog.mode.findData("rotate"))
        QTest.mousePress(target, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(target, center + QPoint(20, 0))
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, pos=center + QPoint(20, 0))
        wait(dialog, app)
        assert dialog.node["rotation_rad"][0] != 0
        dialog.base.setCurrentIndex(dialog.base.findData("object1"))
        dialog.tool.setCurrentIndex(dialog.tool.findData(selected))
        dialog._subtract()
        wait(dialog, app)
        assert dialog.node["kind"] == "subtract"
        cap = next(f for f in dialog.editor.scene["faces"] if "Plane" in f["label"])
        if app.platformName() == "offscreen":
            dialog.face = cap["key"]
        else:
            dialog.mode.setCurrentIndex(dialog.mode.findData("face"))
            dialog.view.view_xy()
            renderer = dialog.view.renderer
            renderer.SetWorldPoint(0, 0, 0.12, 1)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            ratio = target.devicePixelRatioF()
            QTest.mouseClick(
                target, Qt.MouseButton.LeftButton, pos=QPoint(round(x / ratio), round(target.height() - y / ratio))
            )
            assert dialog.face is not None
            assert "Plane" in next(f["label"] for f in dialog.editor.scene["faces"] if f["key"] == dialog.face)
            dialog.view.view_isometric()
        dialog._delete_face()
        wait(dialog, app)
        dialog._start(dialog.editor.generate, "Mesh")
        wait(dialog, app)
        assert dialog.editor.mesh_scene["mesh_report"]["boundary_loops"] == 1
        assert dialog.apply_button.isEnabled()
        assert dialog.view.renderer.lights
        dialog.grab().save(str(tmp_path / "surface-editor.png"))
        if app.platformName() != "offscreen":
            pixels = dialog.view.screenshot(str(tmp_path / "surface-viewport.png"))
            assert pixels.std() > 5
        dialog._apply()
        assert dialog.result_value[0].identity == dialog.editor.model.identity
    finally:
        if dialog.isVisible():
            dialog.reject()
