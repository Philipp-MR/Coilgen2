"""Nine-channel example UI preserves target signs, ownership and full studies."""

import json
import math
import shutil

import pytest

import test_workbench_usability as usability_tests
from coilgen.application import ASSEMBLY_SCHEMA, DesignService, ProjectAssembly
from coilgen.application.system_presets import SYSTEM_EXAMPLE_TITLES
from coilgen.application.targets import recognize_solid_harmonic_target
from coilgen.gui.icons import TARGET_MODE_ICONS, icon
from PySide6.QtWidgets import QComboBox, QDockWidget

application = usability_tests.application
windows = usability_tests.windows

MODE_NAMES = ("Gx", "Gy", "Gz", "Z0", "Z2", "XZ", "YZ", "X²−Y²", "XY")
MODE_IDS = ("gx", "gy", "gz", "z0", "z2", "xz", "yz", "x2-y2", "xy")


def new_system(window):
    file_menu = next(action.menu() for action in window.menuBar().actions() if action.text() == "File")
    templates = next(action.menu() for action in file_menu.actions() if action.text() == "Load template")
    new_menu = next(action.menu() for action in templates.actions() if action.text() == "Coil systems")
    title = SYSTEM_EXAMPLE_TITLES["biplanar_gradient_shim"]
    action = next(action for action in new_menu.actions() if action.text() == title)
    assert not action.icon().isNull()
    action.trigger()
    return window.assembly


def fingerprints(assembly):
    return {channel.id: channel.project.fingerprint for channel in assembly.channels}


def test_menu_example_creates_nine_owned_channels_with_full_physical_studies(windows):
    window = windows()
    assembly = new_system(window)
    assert window._collection_mode
    assert tuple(channel.id for channel in assembly.channels) == MODE_IDS
    assert tuple(channel.name for channel in assembly.channels) == MODE_NAMES
    assert len({channel.project.id for channel in assembly.channels}) == 9
    assert all(channel.enabled for channel in assembly.channels)
    assert window._active_channel_id == "gx"
    assert window.channel_selector.count() == 9
    for index, channel in enumerate(assembly.channels):
        config = channel.project.config
        assert channel.project.study == {
            "pipeline_stage": "evaluated",
            "validation_resolution": 18,
            "design_margin": 0.8,
            "fabricate": True,
            "kernel_backend": "auto",
        }
        assert config.coil_mesh_file == "create bi-planary mesh"
        assert config.biplanar_mesh_parameter_list[:2] == pytest.approx((0.3, 0.3))
        assert config.biplanar_mesh_parameter_list[10] == pytest.approx(0.10 + 0.012 * index)
        assert config.target_region_radius == pytest.approx(0.02)
        assert config.target_field_component == "Bz" and config.target_field_units == "T"
        assert config.b_0_strength_T == 3.0 and tuple(config.b_0_direction) == (0, 0, 1)
        assert config.routing_circuit_scope == "per_part"
        assert not config.skip_postprocessing and not config.skip_sweep and not config.contour_only_sweep_mode
        width = 0.002 if index < 3 else 0.001
        assert config.conductor_cross_section_width == width
        assert config.conductor_cross_section_height == width
        assert config.sweep_insulation_thickness == pytest.approx(0.0001)
        assert config.sweep_min_bend_radius == pytest.approx(3 * width)
        if index < 3:
            assert config.target_gradient_strength == pytest.approx(0.02)
            assert config.field_shape_function == "xyz"[index]
            assert config.target_gradient_normalization == "linear_fit"
        elif index == 3:
            assert float(config.field_shape_function) == pytest.approx(100e-6)
            assert config.target_gradient_normalization == "none"
        else:
            assert recognize_solid_harmonic_target(config.field_shape_function).degree == 2
        mode = "X2-Y2" if channel.id == "x2-y2" else channel.name
        actual = window.channel_selector.itemIcon(index).pixmap(24, 24).toImage()
        expected = icon(TARGET_MODE_ICONS[mode]).pixmap(24, 24).toImage()
        assert actual == expected
    assert "combined hardware pass" in window.messages.toPlainText()


def test_xz_target_editor_keeps_real_harmonic_sign_and_isolated_channel_undo(windows):
    window = windows()
    before = fingerprints(new_system(window))
    window.channel_selector.setCurrentIndex(window.channel_selector.findData("xz"))
    window.select_node("target")
    assert window.project is window.assembly.channel("xz").project
    assert window.findChild(QComboBox, "target_mode").currentText() == "Spherical harmonic"
    controls = window._draft_controls
    assert controls["harmonic_degree"].value() == 2
    assert controls["harmonic_order"].value() == 1
    coefficient_uT = -10 / math.sqrt(15 / (16 * math.pi))
    assert controls["harmonic_coefficient_uT"].value() == pytest.approx(coefficient_uT)
    assert controls["harmonic_coefficient_uT"].suffix() == " µT"
    assert controls["harmonic_reference_radius_mm"].value() == 20
    assert controls["target_region_radius"].value() == 20
    controls["harmonic_coefficient_uT"].setValue(0.9 * coefficient_uT)
    assert window._draft_dirty
    assert fingerprints(window.assembly) == before
    assert window.apply_settings()
    after = fingerprints(window.assembly)
    assert after["xz"] != before["xz"]
    assert all(after[id_] == value for id_, value in before.items() if id_ != "xz")
    recipe = recognize_solid_harmonic_target(window.project.config.field_shape_function)
    assert recipe.coefficient_T == pytest.approx(0.9 * coefficient_uT * 1e-6)
    assert recipe.reference_radius_m == 0.02
    window.undo_stack.undo()
    assert fingerprints(window.assembly) == before
    assert window._active_channel_id == "xz"
    assert window._draft_controls["harmonic_coefficient_uT"].value() == pytest.approx(coefficient_uT)


def test_all_nine_channels_save_and_reload_as_one_owned_json(windows, tmp_path):
    window = windows()
    assembly = new_system(window)
    window.channel_selector.setCurrentIndex(window.channel_selector.findData("xz"))
    window.select_node("target")
    before = fingerprints(assembly)
    path = window.save_project(tmp_path / "portable" / "biplanar_system.coilgen.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema"] == ASSEMBLY_SCHEMA
    assert len(value["channels"]) == 9
    assert all(isinstance(channel["project"], dict) for channel in value["channels"])
    assert list(path.parent.glob("*.json")) == [path]
    relocated = tmp_path / "relocated" / path.name
    relocated.parent.mkdir()
    shutil.copy2(path, relocated)
    assert list(relocated.parent.iterdir()) == [relocated]
    loaded = ProjectAssembly.load(relocated)
    assert fingerprints(loaded) == before
    for original, restored in zip(assembly.channels, loaded.channels):
        assert restored.project.id == original.project.id
        assert restored.project.values == original.project.values
        assert restored.project.study == original.project.study
        assert restored.project.session["system_example"] == original.project.session["system_example"]
    reopened = windows()
    reopened.open_project(relocated)
    assert reopened._collection_mode
    assert reopened._active_channel_id == "xz"
    assert fingerprints(reopened.assembly) == before
    assert reopened.channel_selector.currentText() == "XZ"
    assert reopened.channel_selector.count() == 9


def test_compute_channels_snapshots_each_model_and_full_study_and_keeps_plot_visible(windows, tmp_path):
    window = windows()
    new_system(window)
    captured = {}
    mock_service = window.service

    class CapturingDesignService(DesignService):
        @classmethod
        def submit(cls, project, jobs, **controls):
            request = project.snapshot(tmp_path / "requests" / project.id, **controls)
            captured[project.id] = (project, request)
            return mock_service.submit(
                project,
                jobs,
                **{
                    key: getattr(request, key)
                    for key in ("pipeline_stage", "validation_resolution", "design_margin", "fabricate")
                },
            )

    mock_service.submit_assembly = CapturingDesignService.submit_assembly
    window.evidence.setCurrentWidget(window.messages)
    window.evidence.close_all()
    states = window.compute_channels()
    assert states is not None
    assert set(states) == set(MODE_IDS)
    assert len(captured) == len(mock_service.submissions) == 9
    for channel in window.assembly.channels:
        submitted, request = captured[channel.project.id]
        assert submitted is channel.project
        assert request.project_id == channel.project.id
        assert request.project_fingerprint == channel.project.fingerprint
        assert request.config["field_shape_function"] == channel.project.config.field_shape_function
        assert request.config["biplanar_mesh_parameter_list"] == list(
            channel.project.config.biplanar_mesh_parameter_list
        )
        assert request.config["routing_circuit_scope"] == "per_part"
        assert not request.config["skip_postprocessing"] and not request.config["skip_sweep"]
        assert request.pipeline_stage == "evaluated" and request.fabricate
        assert request.validation_resolution == 18 and request.design_margin == 0.8
    assert window.evidence.currentWidget() is window.jobs_table
    assert not window.evidence.window_for(window.jobs_table).isVisible()
    assert window.live_preview.following
    assert window.live_preview.state["project_id"] == window.project.id
    assert window.jobs_table.rowCount() == 9
    assert window._session_job_ids == {state["id"] for state in states.values()}
    assert "Coupled fields" in window.messages.toPlainText()


def test_channel_overview_action_opens_selected_channel_settings_without_computation(windows):
    window = windows()
    before = fingerprints(new_system(window))

    window.channel_overview_action.trigger()
    panel = window.channels_window
    assert panel.table.rowCount() == 9
    assert panel.isVisible() and not panel.isModal()
    panel.table.selectRow(8)
    panel.open_button.click()
    assert window._active_channel_id == "xy"
    assert not window.findChild(QDockWidget, "settings_dock").isHidden()
    assert fingerprints(window.assembly) == before
    assert not window.jobs.states
    assert not window.service.submissions
