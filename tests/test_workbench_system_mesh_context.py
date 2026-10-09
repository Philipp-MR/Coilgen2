"""System display context survives channel editing and portable GUI sessions."""

from PySide6.QtCore import QCoreApplication, QEvent

from test_native_workbench import application as application, scientific_viewport as scientific_viewport
from test_workbench_usability import Queue, Service

from coilgen.application.system_presets import system_assembly
from coilgen.gui.workbench import CoilGenWorkbench


def test_system_mesh_controls_follow_channel_switch_save_reopen_and_close(scientific_viewport, tmp_path):
    viewport = scientific_viewport
    assembly = system_assembly("cylindrical_xyz_gradient", project_root=tmp_path)
    window = CoilGenWorkbench(
        assembly,
        jobs=Queue(tmp_path / "jobs"),
        service=Service(),
        viewport_factory=lambda _parent: viewport,
    )
    window.timer.stop()
    try:
        assert set(viewport._system_supports) == {"gx", "gy", "gz"}
        assert viewport.system_meshes_button.text() == "System meshes (3/3)"
        viewport.system_mesh_checks["gy"].setChecked(False)
        window.channel_selector.setCurrentIndex(window.channel_selector.findData("gy"))
        assert window._active_channel_id == "gy"
        assert not viewport.system_mesh_checks["gy"].isChecked()
        window.channel_enabled.setChecked(False)
        assert not window.assembly.channel("gy").enabled
        assert viewport.system_meshes_button.text() == "System meshes (2/3)"
        path = window.save_project(tmp_path / "xyz.coilgen.json")
        assert window.close_project()
        assert not viewport._system_channels
        assert viewport.system_meshes_button.isHidden()
        window.open_project(path)
        assert window._active_channel_id == "gy"
        assert not window.channel_enabled.isChecked()
        assert not viewport.system_mesh_checks["gy"].isChecked()
        assert viewport.system_mesh_checks["gx"].isChecked()
        assert viewport.system_mesh_checks["gz"].isChecked()
        viewport.system_mesh_checks["gy"].setChecked(True)
        assert viewport.system_meshes_button.text() == "System meshes (3/3)"
        assert not window.assembly.channel("gy").enabled
        assert window._active_channel_id == "gy"
    finally:
        viewport.setParent(None)
        window._draft_dirty = False
        window.undo_stack.setClean()
        window.close()
        window.undo_stack.clear()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
