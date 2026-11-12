"""Shielded cylinder pair in the shared form, shielding target and confirm-before-replace."""

import math

import numpy as np
import pytest

from coilgen.application.cylinder_pair import SHIELDED_PAIR_SOURCE, project_geometry
from coilgen.application.gradient_geometry import CylindricalGeometry
from coilgen.application.session import WorkbenchSession
from coilgen.application.support_switch import IMPORTED_SUPPORT, support_switch_prompt
from coilgen.application.templates import template_assembly
from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh

try:  # Qt fixtures; the Qt tests skip without PySide6.
    from test_workbench_usability import application as application, windows as windows
except (ImportError, pytest.skip.Exception):  # pragma: no cover - installs without Qt

    @pytest.fixture
    def application():
        pytest.skip("PySide6 is not installed.")

    @pytest.fixture
    def windows():
        pytest.skip("PySide6 is not installed.")


def _shielded(tmp_path):
    return template_assembly("shielded_gy", project_root=tmp_path).channels[0].project


def _session(tmp_path, template="shielded_gy"):
    session = WorkbenchSession(tmp_path / "workspace")
    session.load_template(template)
    return session


def test_shielded_template_is_an_editable_cylinder_pair(tmp_path):
    project = _shielded(tmp_path)
    geometry = CylindricalGeometry.from_project(project)
    assert geometry.shield.radius_m == 0.11 and geometry.suppression.radius_m == 0.15
    mesh = load_mesh_from_config(project.config, tmp_path)
    assert CylindricalGeometry.from_dict(mesh.metadata["cylindrical_geometry"]) == geometry
    assert mesh.metadata["part_roles"] == ["primary", "active_shield"]
    assert project.revision == 1


def test_switch_prompt_names_the_bodies_and_offers_to_clear_shielding(tmp_path):
    project = _shielded(tmp_path)
    prompt = support_switch_prompt(project, "create cylinder mesh")
    assert prompt.message.startswith("This replaces 2 bodies (Primary, Active shield) with 1 cylinder.")
    assert prompt.clear_changes["secondary_target_mesh_file"] == "none"
    assert prompt.clear_changes["routing_circuit_scope"] == "single"
    assert prompt.clear_changes["suppression_field_rms_limit_T"] == 0.0
    assert prompt.to_state()["can_clear"] is True
    # Staying a shielded pair, or importing an STL, never clears the shielding target.
    assert support_switch_prompt(project, SHIELDED_PAIR_SOURCE) is None
    imported = support_switch_prompt(project, IMPORTED_SUPPORT)
    assert "the imported STL" in imported.message and not imported.clear_changes
    # Generated supports keep their parameters, so leaving them needs no question.
    plain = template_assembly("cylinder", project_root=tmp_path).channels[0].project
    assert support_switch_prompt(plain, "create bi-planary mesh") is None
    assert support_switch_prompt(plain, SHIELDED_PAIR_SOURCE) is None


def test_session_edits_pair_dimensions_openings_and_shielding_target(tmp_path):
    session = _session(tmp_path)
    values = session.settings_values("geometry")
    assert values["coil_mesh_file"] == SHIELDED_PAIR_SOURCE
    assert values["pair_primary_radius_mm"] == pytest.approx(80)
    assert values["pair_shield_radius_mm"] == pytest.approx(110)
    assert values["pair_openings"] == ""
    # Unchanged values commit nothing.
    session.apply_settings({key: str(value) for key, value in values.items()})
    assert session.project.revision == 1
    session.apply_settings({**values, "pair_primary_radius_mm": "85", "pair_openings": "90, 30, 0, 60, both"})
    geometry = project_geometry(session.project)
    assert geometry.primary.radius_m == pytest.approx(0.085) and geometry.shield.radius_m == 0.11
    assert geometry.primary.openings == geometry.shield.openings
    assert geometry.primary.openings[0].phi_width_rad == pytest.approx(math.radians(30))
    parts = split_disconnected_mesh(load_mesh_from_config(session.project.config, session.project.project_root))
    assert [len(part.boundaries) for part in parts] == [3, 3]

    target = session.settings_values("target")
    assert target["suppression_radius_mm"] == pytest.approx(150)
    assert target["suppression_rms_limit_uT"] == pytest.approx(20)
    assert target["suppression_field_component"] == "vector"
    assert target["suppression_surface_weighting"] == "area"
    session.apply_settings({"suppression_radius_mm": 160, "suppression_rms_limit_uT": 15})
    assert project_geometry(session.project).suppression.radius_m == pytest.approx(0.16)
    assert session.project.config.suppression_field_rms_limit_T == pytest.approx(15e-6)
    assert session.undo() and project_geometry(session.project).suppression.radius_m == pytest.approx(0.15)
    with pytest.raises(ValueError, match="enclose"):
        session.apply_settings({"suppression_radius_mm": 100})


def test_plain_cylinder_becomes_a_pair_and_the_pair_switches_back(tmp_path):
    session = _session(tmp_path, "cylinder")
    assert "suppression_rms_limit_uT" not in session.settings_values("target")
    values = session.settings_values("geometry", support_source=SHIELDED_PAIR_SOURCE)
    assert values["pair_shield_radius_mm"] > values["pair_primary_radius_mm"]
    session.apply_settings({**values, "coil_mesh_file": SHIELDED_PAIR_SOURCE})
    project = session.project
    assert project_geometry(project).shield is not None
    assert project.config.routing_circuit_scope == "per_part"
    assert project.config.suppression_field_component == "vector"
    assert session.settings_values("geometry")["coil_mesh_file"] == SHIELDED_PAIR_SOURCE

    prompt = support_switch_prompt(project, "create cylinder mesh")
    session.apply_settings({"coil_mesh_file": "create cylinder mesh"}, prompt.clear_changes)
    config = session.project.config
    assert config.coil_mesh_file == "create cylinder mesh" and config.secondary_target_mesh_file == "none"
    assert config.routing_circuit_scope == "single" and config.surface_is_cylinder_flag


def test_browser_asks_before_replacing_the_pair(tmp_path):
    pytest.importorskip("trame")
    from coilgen.web.app import create_app

    server, browser = create_app(tmp_path, user="tester", server_name="test-support-switch")
    state = server.state
    state.template = "shielded_gy"
    browser.load_template()
    sections = {field["key"]: field["section"] for field in state.settings_fields}
    assert sections["pair_primary_radius_mm"] == "Primary" and sections["pair_primary_length_mm"] == ""
    assert sections["pair_shield_radius_mm"] == "Active shield"
    labels = {choice["value"]: choice["title"] for choice in state.settings_fields[0]["choices"]}
    assert labels[SHIELDED_PAIR_SOURCE] == "Shielded cylinder pair"
    assert not any(title == "Project-owned support" for title in labels.values())

    before = browser.session.project
    state.settings_values = {**state.settings_values, "coil_mesh_file": "create cylinder mesh"}
    browser.apply_settings()
    assert state.support_switch_dialog and browser.session.project is before
    assert "2 bodies (Primary, Active shield)" in state.support_switch["message"]
    assert state.support_switch["can_clear"]
    browser.cancel_support_switch()
    assert not state.support_switch_dialog and browser.session.project is before

    browser.apply_settings()
    state.support_switch_clear = True
    browser.confirm_support_switch()
    config = browser.session.project.config
    assert config.coil_mesh_file == "create cylinder mesh" and config.secondary_target_mesh_file == "none"
    assert browser.session.undo() and browser.session.project.fingerprint == before.fingerprint


def test_browser_shielding_target_section(tmp_path):
    pytest.importorskip("trame")
    from coilgen.web.app import create_app

    server, browser = create_app(tmp_path, user="tester", server_name="test-shielding-target")
    state = server.state
    state.template = "shielded_gy"
    browser.load_template()
    state.node = "target"
    browser._refresh_settings()
    sections = {field["key"]: field["section"] for field in state.settings_fields}
    assert sections["suppression_radius_mm"] == "Shielding target"
    assert sections["suppression_rms_limit_uT"] == ""
    component = next(field for field in state.settings_fields if field["key"] == "suppression_field_component")
    assert {"title": "All three components (Bx, By, Bz)", "value": "vector"} in component["choices"]
    state.settings_values = {**state.settings_values, "suppression_rms_limit_uT": "12.5"}
    browser.apply_settings()
    assert not state.support_switch_dialog
    assert browser.session.project.config.suppression_field_rms_limit_T == pytest.approx(12.5e-6)


def test_browser_turns_a_cylinder_into_a_pair_without_asking(tmp_path):
    pytest.importorskip("trame")
    from coilgen.web.app import create_app

    server, browser = create_app(tmp_path, user="tester", server_name="test-pair-from-cylinder")
    state = server.state
    state.template = "cylinder"
    browser.load_template()
    state.settings_values = {**state.settings_values, "coil_mesh_file": SHIELDED_PAIR_SOURCE}
    browser._check_edits()
    assert "pair_shield_radius_mm" in state.settings_values
    assert "shielded" in state.geometry_help.lower() or "active-shield" in state.geometry_help
    state.settings_values = {**state.settings_values, "pair_shield_radius_mm": "120"}
    browser.apply_settings()
    assert not state.support_switch_dialog
    geometry = project_geometry(browser.session.project)
    assert geometry.shield.radius_m == pytest.approx(0.12)
    assert state.settings_values["coil_mesh_file"] == SHIELDED_PAIR_SOURCE


def test_qt_asks_before_replacing_the_pair_and_can_clear_shielding(windows, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QMessageBox

    project = _shielded(tmp_path)
    window = windows(project)
    texts = []

    def answer(box):
        # Cancel the first question, confirm the second (shielding clear stays ticked).
        texts.append(box.text())
        if len(texts) > 1:
            next(button for button in box.buttons() if button.text() == "Replace support").click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", answer)
    window.select_node("geometry")
    window.findChild(QComboBox, "support_type").setCurrentText("Cylinder")
    assert not window.apply_settings()
    assert window.project is project
    assert "2 bodies (Primary, Active shield)" in texts[0]
    assert window.apply_settings()
    config = window.project.config
    assert config.coil_mesh_file == "create cylinder mesh"
    assert config.secondary_target_mesh_file == "none" and config.routing_circuit_scope == "single"


def test_qt_geometry_dialog_edits_openings_and_rms_limit(application, tmp_path):
    from coilgen.application.cylinder_pair import apply_geometry
    from coilgen.gui.gradient_tools import GradientGeometryDialog

    project = _shielded(tmp_path)
    dialog = GradientGeometryDialog(project)
    try:
        assert dialog.value() == CylindricalGeometry.from_project(project)
        assert dialog.config_changes() == {}
        dialog.openings.setPlainText("90, 30, 0, 60, both\n270, 30, 0, 60, shield")
        dialog.rms_limit.setValue(12)
        geometry = dialog.value()
        assert len(geometry.primary.openings) == 1 and len(geometry.shield.openings) == 2
        changes = dialog.config_changes()
        assert changes["suppression_field_rms_limit_T"] == pytest.approx(12e-6)
        updated = apply_geometry(project, geometry, changes, expected_fingerprint=project.fingerprint)
        assert updated.revision == project.revision + 1
        assert CylindricalGeometry.from_project(updated) == geometry
        assert updated.config.suppression_field_rms_limit_T == pytest.approx(12e-6)
        dialog.openings.setPlainText("90, 400, 0, 60, both")
        with pytest.raises(ValueError, match="circumference"):
            dialog.value()
    finally:
        dialog.close()
    assert np.isfinite(project.config.suppression_field_rms_limit_T)
