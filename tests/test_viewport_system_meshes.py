"""Independent-channel context is complete, in SI, and purely a display choice."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_native_workbench as native

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.geometry import load_mesh_from_config
from coilgen.gui.viewport import _SystemSupportTask

application = native.application
scientific_viewport = native.scientific_viewport


def _project(name, radius, *, rotated=False, missing=False):
    config = replace(
        preset_config(),
        coil_mesh_file="none" if missing else "create cylinder mesh",
        field_shape_function="none",
        cylinder_mesh_parameter_list=(0.2125, radius, 12, 6, 1, 0, 0, 0),
        generated_mesh_rotation_axis=(1, 0, 0),
        generated_mesh_rotation_angle=np.pi / 2 if rotated else 0,
    )
    return ProjectDocument.from_config(config, name=name)


def _assembly(*, missing=False):
    assembly = ProjectAssembly.new(
        "Nested gradient supports",
        (_project("Gx", 0.08), _project("Gy", 0.09, rotated=True), _project("Gz", 0.10, missing=missing)),
    )
    return assembly.update_channel(assembly.channels[2].id, enabled=False)


def _display(viewport, assembly, index=0):
    channel = assembly.channels[index]
    viewport.set_assembly_context(assembly, channel.id)
    viewport.render_project(channel.project)
    return channel


def _system_actor_keys(viewport):
    return {key for key in viewport._actors.values() if key.startswith("system:")}


def test_all_supports_include_disabled_channels_and_exact_world_geometry(scientific_viewport, monkeypatch):
    import coilgen.target as target_module

    monkeypatch.setattr(target_module, "define_target_field", lambda *_args: pytest.fail("Inactive target evaluated"))
    viewport, assembly = scientific_viewport, _assembly()
    before = deepcopy(assembly.to_dict())
    active = _display(viewport, assembly)
    assert "geometry" in viewport._actors.values()
    assert _system_actor_keys(viewport) == {f"system:{channel.id}" for channel in assembly.channels[1:]}
    assert all(check.isEnabled() and check.isChecked() for check in viewport.system_mesh_checks.values())
    assert "compute disabled" in viewport.system_mesh_checks[assembly.channels[2].id].text()
    assert "3/3 channel support meshes visible" in viewport.caption.text()
    assert not viewport.system_meshes_button.isHidden()
    for channel in assembly.channels:
        expected = load_mesh_from_config(channel.project.config, Path.cwd())
        mesh = viewport._support if channel.id == active.id else viewport._system_supports[channel.id]
        np.testing.assert_array_equal(mesh.points, expected.vertices)
        np.testing.assert_array_equal(mesh.faces.reshape(-1, 4)[:, 1:], expected.faces)
    assert assembly.to_dict() == before


def test_selection_is_display_only_and_hidden_active_support_can_be_restored(scientific_viewport):
    viewport, assembly = scientific_viewport, _assembly()
    active = _display(viewport, assembly)
    fingerprint = assembly.fingerprint
    other = assembly.channels[1]
    viewport.system_mesh_checks[other.id].setChecked(False)
    assert f"system:{other.id}" not in _system_actor_keys(viewport)
    assert "2/3 channel support meshes visible" in viewport.caption.text()
    viewport.system_mesh_checks[active.id].setChecked(False)
    assert "geometry" not in viewport._actors.values()
    assert viewport.dataset.currentData() == "support"
    assert not viewport.hidden_geometry.isHidden()
    viewport.show_support.click()
    assert viewport.system_mesh_checks[active.id].isChecked()
    assert "geometry" in viewport._actors.values()
    assert assembly.fingerprint == fingerprint
    assert assembly.channels[2].enabled is False


def test_show_and_hide_all_include_every_support_without_camera_or_input_edits(scientific_viewport):
    viewport, assembly = scientific_viewport, _assembly()
    _display(viewport, assembly)
    camera = [[0.4, 0.3, 0.2], [0, 0, 0], [0, 0, 1]]
    viewport.plotter.camera_position = deepcopy(camera)
    viewport.system_hide_all.click()
    assert viewport._actors == {}
    assert not any(viewport._system_visibility.values())
    viewport.support_layer.setChecked(False)
    viewport.system_show_all.click()
    assert viewport.support_layer.isChecked()
    assert all(viewport._system_visibility.values())
    assert len(_system_actor_keys(viewport)) == 2
    assert viewport.plotter.camera_position == camera


def test_active_support_visibility_preserves_selected_field_and_arrays(scientific_viewport):
    viewport, assembly = scientific_viewport, _assembly()
    active = _display(viewport, assembly)
    vertices = viewport._support.points.copy()
    faces = viewport._support.faces.reshape(-1, 4)[:, 1:].copy()
    arrays = {
        "vertices_m": vertices,
        "faces": faces,
        "target_points_m": np.array([[0.0, 0, 0], [0.01, 0, 0]]),
        "target_field_T": np.zeros((2, 3)),
        "sheet_field_T": np.array([[0, 0, -1e-6], [0, 0, 1e-6]]),
        "stream_function_A": np.linspace(-1, 1, len(vertices)),
    }
    viewport.display_run(arrays, {"project_id": active.project.id})
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("sheet"))
    original_values = viewport._dataset_mesh.active_scalars.copy()
    viewport.system_mesh_checks[active.id].setChecked(False)
    assert "geometry" not in viewport._actors.values()
    assert viewport.dataset.currentData() == "sheet"
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, original_values)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("stream_function"))
    assert viewport._dataset_mesh is None
    assert "active support mesh hidden" in viewport.caption.text()
    viewport.system_mesh_checks[active.id].setChecked(True)
    assert viewport._dataset_mesh.n_points == len(vertices)
    viewport.support_layer.setChecked(False)
    assert viewport._dataset_mesh is None
    assert "active support mesh hidden" in viewport.caption.text()
    assert "dataset unavailable" not in viewport.caption.text()
    assert "System meshes (0/3)" == viewport.system_meshes_button.text()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("sheet"))
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, original_values)
    for key, values in arrays.items():
        np.testing.assert_array_equal(viewport._arrays[key], values)


def test_visibility_and_cache_follow_channel_ids_across_selection_rename_and_mesh_edit(
    scientific_viewport, monkeypatch
):
    import coilgen.geometry as geometry_module

    viewport, assembly = scientific_viewport, _assembly()
    _display(viewport, assembly)
    hidden = assembly.channels[1]
    cached = viewport._system_supports[hidden.id]
    viewport.system_mesh_checks[hidden.id].setChecked(False)
    renamed = assembly.update_channel(hidden.id, name="Gy custom")
    _display(viewport, renamed, 1)
    assert not viewport.system_mesh_checks[hidden.id].isChecked()
    assert viewport.system_mesh_checks[hidden.id].text().startswith("Gy custom")
    assert viewport._system_supports[assembly.channels[2].id].n_points > 0
    changed = hidden.project.update_config({"cylinder_mesh_parameter_list": (0.2125, 0.115, 12, 6, 1, 0, 0, 0)})
    edited = renamed.update_channel(hidden.id, project=changed)
    _display(viewport, edited, 0)
    assert not viewport.system_mesh_checks[hidden.id].isChecked()
    assert viewport._system_supports[hidden.id] is not cached
    expected = load_mesh_from_config(changed.config, Path.cwd())
    np.testing.assert_array_equal(viewport._system_supports[hidden.id].points, expected.vertices)
    calls = []
    original = geometry_module.load_mesh_from_config
    monkeypatch.setattr(
        geometry_module, "load_mesh_from_config", lambda *args: (calls.append(args), original(*args))[1]
    )
    viewport.set_assembly_context(edited, edited.channels[0].id)
    assert not calls


def test_mesh_visibility_session_validation_is_atomic_and_new_channels_default_visible(scientific_viewport):
    viewport, assembly = scientific_viewport, _assembly()
    _display(viewport, assembly)
    hidden = assembly.channels[1].id
    viewport.system_mesh_checks[hidden].setChecked(False)
    state = viewport.view_state()
    viewport.system_show_all.click()
    assert viewport.restore_view_state(state)
    assert viewport.view_state() == state
    before = viewport.view_state()
    invalid = dict(state, system_mesh_visibility={hidden: "false"}, camera=[[9, 9, 9], [0, 0, 0], [0, 0, 1]])
    assert not viewport.restore_view_state(invalid)
    assert viewport.view_state() == before
    expanded = assembly.add_channel(_project("Additional support", 0.12))
    _display(viewport, expanded)
    assert viewport.restore_view_state(state)
    assert not viewport.system_mesh_checks[hidden].isChecked()
    assert viewport.system_mesh_checks[expanded.channels[-1].id].isChecked()


def test_missing_or_invalid_support_does_not_hide_valid_system_geometry(scientific_viewport, monkeypatch):
    import coilgen.geometry as geometry_module

    viewport, assembly = scientific_viewport, _assembly(missing=True)
    original = geometry_module.load_mesh_from_config

    def invalid_mesh(config, root):
        if config.cylinder_mesh_parameter_list[1] == 0.09:
            return SimpleNamespace(vertices=np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]), faces=[[0, 1, 2]])
        return original(config, root)

    monkeypatch.setattr(geometry_module, "load_mesh_from_config", invalid_mesh)
    _display(viewport, assembly)
    assert "geometry" in viewport._actors.values()
    assert _system_actor_keys(viewport) == set()
    for channel in assembly.channels[1:]:
        check = viewport.system_mesh_checks[channel.id]
        assert not check.isEnabled()
        assert "unavailable" in check.text()
        assert "Support unavailable" in check.toolTip()
        assert channel.name in viewport.system_meshes_note.text()
    assert "1/3 channel support meshes visible" in viewport.caption.text()
    assert "System meshes (1/3)" == viewport.system_meshes_button.text()
    assert "support unavailable: Gy, Gz" in viewport.caption.text()


def test_stale_system_preview_and_closed_context_cannot_reintroduce_old_geometry(scientific_viewport, monkeypatch):
    viewport, assembly = scientific_viewport, _assembly()
    viewport._synchronous_preview = False
    requests = []
    monkeypatch.setattr(viewport, "_start_system_preview", requests.append)
    viewport.set_assembly_context(assembly, assembly.channels[0].id)
    old = requests[-1]
    edited_project = assembly.channels[1].project.update_config(
        {"cylinder_mesh_parameter_list": (0.2125, 0.125, 12, 6, 1, 0, 0, 0)}
    )
    edited = assembly.update_channel(assembly.channels[1].id, project=edited_project)
    viewport.set_assembly_context(edited, edited.channels[0].id)
    current = requests[-1]

    def finish(request):
        task = _SystemSupportTask(*request)
        task.signals.finished.connect(viewport._system_preview_finished)
        task.run()

    finish(old)
    assert viewport._system_supports == {}
    finish(current)
    expected = load_mesh_from_config(edited_project.config, Path.cwd())
    np.testing.assert_array_equal(viewport._system_supports[edited.channels[1].id].points, expected.vertices)
    viewport.set_assembly_context(None)
    finish(current)
    assert viewport._system_supports == {}
    assert viewport.system_mesh_checks == {}
    assert viewport.system_meshes_button.isHidden()


def test_picking_context_mesh_reports_its_channel_without_editor_navigation(scientific_viewport):
    viewport, assembly = scientific_viewport, _assembly()
    active = _display(viewport, assembly)
    other = assembly.channels[1]
    actor = next(
        actor
        for _mesh, _options, actor in viewport.plotter.meshes
        if viewport._actors[id(actor)] == f"system:{other.id}"
    )
    selected = []
    viewport.selected.connect(selected.append)
    viewport.plotter.picked_point = np.array([0.01, 0.02, 0.03])
    viewport.plotter.pick(actor)
    assert selected == []
    assert "Gy support mesh" in viewport.probe.text()
    assert "no field value sampled" in viewport.probe.text()
    assert viewport._system_active_channel == active.id


def test_loading_supports_are_not_counted_as_visible(scientific_viewport, monkeypatch):
    viewport, assembly = scientific_viewport, _assembly()
    viewport._synchronous_preview = False
    requests = []
    monkeypatch.setattr(viewport, "_start_system_preview", requests.append)
    viewport.set_assembly_context(assembly, assembly.channels[0].id)
    assert viewport.system_meshes_button.text() == "System meshes (0/3)"
    viewport._synchronous_preview = True
    viewport.render_project(assembly.channels[0].project)
    assert viewport.system_meshes_button.text() == "System meshes (1/3)"
    for channel in assembly.channels[1:]:
        assert not viewport.system_mesh_checks[channel.id].isEnabled()
        assert "loading" in viewport.system_mesh_checks[channel.id].text()
    task = _SystemSupportTask(*requests[-1])
    task.signals.finished.connect(viewport._system_preview_finished)
    task.run()
    assert viewport.system_meshes_button.text() == "System meshes (3/3)"


def test_historical_result_uses_its_owner_support_and_labels_current_other_channel_context(scientific_viewport):
    viewport, assembly = scientific_viewport, _assembly()
    editing = _display(viewport, assembly)
    historical_owner = assembly.channels[1]
    historical = _project("Old Gy", 0.055, rotated=True)
    old_mesh = load_mesh_from_config(historical.config, Path.cwd())
    arrays = {
        "vertices_m": old_mesh.vertices,
        "faces": old_mesh.faces,
        "route_0_points_m": old_mesh.vertices[:3],
    }
    viewport.display_run(arrays, {"project_id": historical_owner.project.id, "project_revision": 1})
    assert viewport.dataset.currentData() == "routes"
    assert "other channels: current model supports; not part of this run" in viewport.caption.text()
    assert _system_actor_keys(viewport) == {f"system:{editing.id}", f"system:{assembly.channels[2].id}"}
    result_mesh = next(
        mesh for mesh, _options, actor in viewport.plotter.meshes if viewport._actors[id(actor)] == "geometry"
    )
    np.testing.assert_array_equal(result_mesh.points, old_mesh.vertices)
    assert "editing" in viewport.system_mesh_checks[editing.id].text()
    assert "result" in viewport.system_mesh_checks[historical_owner.id].text()
    viewport.system_mesh_checks[editing.id].setChecked(False)
    assert "geometry" in viewport._actors.values()
    assert f"system:{editing.id}" not in _system_actor_keys(viewport)
    viewport.system_mesh_checks[historical_owner.id].setChecked(False)
    assert "geometry" not in viewport._actors.values()
    assert "winding" in viewport._actors.values()
    assert viewport._system_active_channel == editing.id
    assert viewport.system_meshes_button.text() == "System meshes (1/3)"


def test_valid_stored_support_remains_selectable_when_its_current_model_support_is_missing(scientific_viewport):
    viewport, assembly = scientific_viewport, _assembly(missing=True)
    _display(viewport, assembly)
    owner = assembly.channels[2]
    old_mesh = load_mesh_from_config(_project("Old Gz", 0.10).config, Path.cwd())
    viewport.display_run(
        {"vertices_m": old_mesh.vertices, "faces": old_mesh.faces, "route_0_points_m": old_mesh.vertices[:3]},
        {"project_id": owner.project.id},
    )
    check = viewport.system_mesh_checks[owner.id]
    assert check.isEnabled()
    assert "current model unavailable" in check.text()
    assert "retains its stored support" in check.toolTip()
    assert "geometry" in viewport._actors.values()
    assert viewport.system_meshes_button.text() == "System meshes (3/3)"
    check.setChecked(False)
    assert "geometry" not in viewport._actors.values()
    assert viewport.system_meshes_button.text() == "System meshes (2/3)"
