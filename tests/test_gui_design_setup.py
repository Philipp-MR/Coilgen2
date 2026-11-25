"""The main desktop editor has one scroll and explicit inline details."""

from _coilgen_test_support import workbench_empty as native
import pytest
from PySide6.QtWidgets import QDockWidget

application = native.application
window = native.window


def test_delayed_advanced_scroll_does_not_outlive_its_qt_owner(application, monkeypatch):
    import sys
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
    from coilgen.gui.design_setup import DesignSetupPanel

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    host = QWidget()
    host.readiness_banner = QLabel()
    host.apply_settings = lambda: True
    panel = DesignSetupPanel(host)
    editor = QWidget()
    layout = QVBoxLayout(editor)
    layout.addWidget(QLabel("Advanced settings"))
    layout.addLayout(QHBoxLayout())
    editor.configure_physics = QPushButton()
    editor.fields = {}
    panel.show_advanced(editor)
    host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    assert errors == []


def test_setup_replaces_tree_and_initial_plan_without_losing_tools(window, application):
    window.load_template("cylinder")
    application.processEvents()
    panel = window.setup_panel
    assert window.findChild(QDockWidget, "model_dock").widget() is panel
    assert panel.isVisible() and not window.tree.isVisible()
    assert window.design_plan_dock.isHidden()
    assert window.findChild(QDockWidget, "settings_dock").isHidden()
    assert set(panel.cards) == {"geometry", "target", "conductor", "optimization", "winding", "print", "validation"}
    assert panel.controls["cylinder_radius_mm"].isVisible()
    before = window.project
    toggle, details = panel.details["geometry"]
    toggle.click()
    assert details.isVisible()
    toggle.click()
    assert not details.isVisible()
    assert window.project is before and not panel.dirty and not window.service.submissions


def test_desktop_setup_save_uses_all_sections_and_undo_restores_source(window, tmp_path):
    window.load_template("cylinder")
    before = window.project
    panel = window.setup_panel
    panel._edit("cylinder_radius_mm", "142")
    panel._edit("target_amplitude_native", "32")
    panel._edit("validation_resolution", "8")
    assert window._has_input_drafts()
    path = window.save_project(tmp_path / "setup.coilgen")
    assert path
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.142)
    assert window.project.config.target_gradient_strength == 0.032
    assert window.project.study["validation_resolution"] == 8
    assert not panel.dirty
    window.undo_stack.undo()
    assert window.project.fingerprint == before.fingerprint


def test_incomplete_number_reveals_inline_details_without_opening_window(window):
    window.load_template("cylinder")
    before = window.project
    panel = window.setup_panel
    panel._edit("constrained_optimization_tolerance", "1e")
    assert not window.apply_settings()
    assert window.project is before
    assert panel.details["optimization"][0].isChecked()
    assert panel.controls["constrained_optimization_tolerance"].text() != ""  # input retains its draft
    assert window.findChild(QDockWidget, "settings_dock").isHidden()
    panel.reset()


def test_advanced_is_embedded_and_cannot_replace_pending_setup(window):
    window.load_template("cylinder")
    window._advanced_dialog()
    expert = window.expert_window
    assert not expert.isWindow()
    assert expert.parent() is window.setup_panel.advanced_host
    expert.fields["min_turn_distance"].control.setText("4")
    window.setup_panel._edit("cylinder_radius_mm", "91")
    before = window.project
    assert not expert.apply_changes()
    assert window.project is before
    assert window.setup_panel.draft.values["cylinder_radius_mm"] == "91"
    expert.close()
    assert window.expert_window is None
    assert window.apply_settings()
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.091)
