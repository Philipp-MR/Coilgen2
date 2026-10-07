"""Current target intent and verified stage graphics retain their true source."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import test_native_workbench as native

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.geometry import load_mesh_from_config
from coilgen.gui.viewport import _preview_data
from coilgen.models import TargetField
from coilgen.object_registry import register_in_memory_target_field
from coilgen.target import define_target_field

application = native.application
scientific_viewport = native.scientific_viewport


def _saved_arrays():
    vertices = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    return {
        "vertices_m": vertices,
        "faces": np.array([[0, 1, 2]]),
        "target_points_m": np.array([[0, 0, 0], [0, 0.01, 0]]),
        "target_field_T": np.array([[0, 0, -1e-6], [0, 0, 1e-6]]),
        "stream_function_A": np.array([-1.0, 0.0, 1.0]),
    }


@pytest.mark.parametrize("kind", ["cylinder", "biplanar", "conical_harmonic"])
def test_selected_target_shows_exact_field_and_support_automatically(scientific_viewport, kind):
    viewport = scientific_viewport
    config = preset_config(kind)
    viewport.render_config(config)
    support = load_mesh_from_config(config, Path.cwd())
    target = define_target_field([support], None, None, config)
    assert viewport.dataset.currentData() == "preview_target"
    assert viewport.support_layer.isChecked() and viewport.roi_layer.isChecked()
    assert set(viewport._actors.values()) == {"geometry", "target"}
    np.testing.assert_array_equal(viewport._preview_arrays["target_field_T"], target.b)
    np.testing.assert_array_equal(viewport._dataset_mesh.points, target.coords)
    assert "current model target" in viewport.caption.text()
    assert "model samples" in viewport.caption.text()
    assert viewport.view_state()["source"] == "model"


def test_target_choice_reveals_missing_field_with_existing_support_and_camera(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    config = replace(preset_config(), field_shape_function="none")
    viewport.render_config(config)
    assert viewport.dataset.currentData() == "support" and viewport._support is not None
    assert "Choose a target field" in viewport.caption.text()
    camera = [[0.3, 0.2, 0.1], [0, 0, 0], [0, 0, 1]]
    viewport.plotter.camera_position = deepcopy(camera)
    viewport.component.setCurrentText("By")
    viewport.support_layer.setChecked(False)
    viewport.slice_axis.setCurrentText("z")
    viewport.clip.setChecked(True)
    monkeypatch.setattr(viewport.plotter, "reset_camera", lambda **_kwargs: pytest.fail("Target edit moved camera"))
    viewport.render_config(replace(config, field_shape_function="y"))
    assert viewport.dataset.currentData() == "preview_target"
    assert viewport.component.currentText() == "Bz" and viewport.slice_axis.currentText() == "All points"
    assert viewport.support_layer.isChecked() and not viewport.clip.isChecked()
    assert viewport.plotter.camera_position == camera
    assert np.ptp(viewport._dataset_mesh.active_scalars) > 0


def test_harmonic_and_roi_edits_use_shared_evaluator_in_world_coordinates(scientific_viewport):
    from coilgen.target_regions import TargetRegion

    viewport = scientific_viewport
    config = replace(
        preset_config("conical_harmonic"),
        field_shape_function="100e-6*solid_harmonic(2,0,x,y,z,0.05,0.01,0,0)",
        target_gradient_normalization="none",
        target_region_geometry=TargetRegion("sphere", radius_m=0.05, center_m=(0.01, 0, 0)).to_dict(),
    )
    viewport.render_config(config)
    expected = _preview_data(config, Path.cwd())[1]
    for key in ("target_points_m", "target_field_T"):
        np.testing.assert_array_equal(viewport._preview_arrays[key], expected[key])
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("support"))
    edited = replace(
        config, target_region_geometry=TargetRegion("sphere", radius_m=0.04, center_m=(0.02, 0, 0)).to_dict()
    )
    viewport.render_config(edited)
    assert viewport.dataset.currentData() == "preview_target"
    assert np.mean(viewport._dataset_mesh.points[:, 0]) == pytest.approx(0.02)


def test_owned_imported_target_changes_material_without_resetting_user_plot(scientific_viewport):
    viewport = scientific_viewport
    points = np.array([[0.0, 0, 0], [0.01, 0, 0]])
    field = np.array([[2e-6, 0, 0], [-3e-6, 0, 0]])
    imported = register_in_memory_target_field(
        TargetField(points, field, np.ones(2), np.zeros(2, bool), np.ones(2, int))
    )
    config = replace(
        preset_config(), field_shape_function="none", target_field_definition_file=imported, target_field_component="Bx"
    )
    project = ProjectDocument.from_config(config)
    viewport.render_project(project)
    assert viewport.dataset.currentData() == "preview_target" and viewport.component.currentText() == "Bx"
    np.testing.assert_array_equal(viewport._preview_arrays["target_field_T"], field)
    np.testing.assert_array_equal(viewport._dataset_mesh.points, points)
    saved = _saved_arrays()
    saved["sheet_field_T"] = saved["target_field_T"].copy()
    viewport.display_run(saved, {"name": "Previous run"})
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("sheet"))
    edited = project.update_config({"levels": project.config.levels + 1})
    assert project.config.target_field_definition_file != edited.config.target_field_definition_file
    viewport.render_project(edited)
    assert viewport.dataset.currentData() == "sheet"
    assert viewport.view_state()["source"] == "run"
    assert viewport._metadata["name"] == "Previous run"


def test_support_roi_context_never_uses_previous_runs_target(scientific_viewport):
    viewport = scientific_viewport
    viewport.render_config(preset_config())
    current_points = viewport._preview_arrays["target_points_m"].copy()
    viewport.display_run(_saved_arrays())
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("support"))
    viewport.roi_layer.setChecked(True)
    target_actor = next(actor for actor, key in viewport._actors.items() if key == "target")
    target_mesh = next(mesh for mesh, _options, actor in viewport.plotter.meshes if id(actor) == target_actor)
    np.testing.assert_array_equal(target_mesh.points, current_points)


@pytest.mark.parametrize("choice", ["support", "stream_function", "saved_run", "saved_view"])
def test_preview_completion_respects_later_user_view_choice(scientific_viewport, monkeypatch, choice):
    viewport = scientific_viewport
    viewport.render_config(preset_config())
    viewport.display_run(_saved_arrays())
    viewport._synchronous_preview = False
    requests = []
    monkeypatch.setattr(viewport, "_start_preview", lambda request: requests.append(request))
    changed = replace(preset_config(), target_gradient_strength=0.03)
    viewport.render_config(changed)
    assert viewport.dataset.currentData() == "preview_target"
    if choice == "saved_run":
        viewport.display_run(_saved_arrays())
        expected = "stream_function"
    elif choice == "saved_view":
        state = viewport.view_state()
        state.update(dataset="stream_function", source="run")
        assert viewport.restore_view_state(state)
        expected = "stream_function"
    else:
        viewport.dataset.setCurrentIndex(viewport.dataset.findData(choice))
        expected = choice
    generation, config, root, _owner = requests[-1]
    viewport._preview_finished(generation, _preview_data(config, root))
    assert viewport.dataset.currentData() == expected


def test_late_preview_and_missing_target_never_show_old_field(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    viewport._synchronous_preview = False
    requests = []
    monkeypatch.setattr(viewport, "_start_preview", lambda request: requests.append(request))
    viewport.render_config(preset_config())
    old = requests[-1]
    config = replace(preset_config(), field_shape_function="none")
    viewport.render_config(config)
    latest = requests[-1]
    viewport._preview_finished(old[0], _preview_data(old[1], old[2]))
    assert viewport._preview_arrays == {}
    assert viewport.dataset.currentData() == "support"
    viewport._preview_finished(latest[0], _preview_data(latest[1], latest[2]))
    assert viewport._support is not None and viewport._preview_arrays == {}
    assert set(viewport._actors.values()) == {"geometry"}


def test_target_failure_preserves_support_without_reusing_previous_target(scientific_viewport, monkeypatch):
    import coilgen.target as target_module

    viewport = scientific_viewport
    viewport.render_config(preset_config())

    def failed_target(*_args):
        raise ValueError("Target expression is nonfinite on this region")

    monkeypatch.setattr(target_module, "define_target_field", failed_target)
    errors = []
    viewport.error.connect(errors.append)
    viewport.render_config(replace(preset_config(), target_gradient_strength=0.04))
    assert viewport._support is not None and viewport._preview_arrays == {}
    assert set(viewport._actors.values()) == {"geometry"}
    assert "Target preview unavailable" in viewport.caption.text()
    assert errors and "nonfinite" in errors[-1]


def test_high_resolution_target_has_explicit_omission_with_actual_support(scientific_viewport):
    viewport = scientific_viewport
    viewport.render_config(preset_config())
    viewport.render_config(replace(preset_config(), target_region_resolution=41))
    assert viewport.dataset.currentData() == "support"
    assert not viewport.dataset.model().item(viewport.dataset.findData("preview_target")).isEnabled()
    assert viewport._support is not None and viewport._preview_arrays == {}
    assert "omitted above resolution 40" in viewport.caption.text()
    assert "full grid" in viewport.caption.text()
    assert set(viewport._actors.values()) == {"geometry"}


def test_target_without_support_prompts_for_geometry_and_starts_no_worker(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    monkeypatch.setattr(viewport, "_start_preview", lambda *_args: pytest.fail("Missing support launched a worker"))
    viewport.render_config(replace(preset_config(), coil_mesh_file="none"))
    assert viewport.dataset.currentData() == "support" and viewport._support is None
    assert viewport._preview_arrays == {} and viewport._actors == {}
    assert "Choose a support surface" in viewport.caption.text()


@pytest.mark.parametrize(
    "stage, added, preferred",
    [
        ("prepared", {}, "target"),
        ("sheet", {"stream_function_A": np.array([-1.0, 0.0, 1.0])}, "stream_function"),
        ("contours", {"contour_0_points_m": _saved_arrays()["vertices_m"]}, "contours"),
        ("connected", {"route_0_points_m": _saved_arrays()["vertices_m"]}, "routes"),
        (
            "swept",
            {"swept_0_0_vertices_m": _saved_arrays()["vertices_m"], "swept_0_0_faces": np.array([[0, 1, 2]])},
            "swept",
        ),
    ],
)
def test_live_stage_uses_exact_outputs_without_replacing_saved_run(scientific_viewport, stage, added, preferred):
    viewport = scientific_viewport
    saved = _saved_arrays()
    viewport.display_run(saved, {"name": "Older completed result"})
    camera = [[0.3, 0.2, 0.1], [0, 0, 0], [0, 0, 1]]
    viewport.plotter.camera_position = deepcopy(camera)
    stage_arrays = {key: value for key, value in _saved_arrays().items() if key != "stream_function_A"}
    stage_arrays.update(added)
    viewport.display_stage_preview(stage_arrays, {"stage": stage, "project_revision": 3, "acceptance_claim": True})
    assert viewport.dataset.currentData() == preferred and viewport.view_state()["source"] == "live"
    assert viewport.plotter.camera_position == camera
    assert "Live study preview" in viewport.caption.text() and "acceptance pending" in viewport.caption.text()
    assert viewport.stage_preview_metadata["acceptance_claim"] is False
    assert viewport._metadata == {"name": "Older completed result"}
    for key, values in saved.items():
        np.testing.assert_array_equal(viewport._arrays[key], values)
    assert "geometry" in viewport._actors.values() or preferred == "stream_function"
    assert "target" in viewport._actors.values()
    viewport.display_run(saved, {"name": "Completed now"})
    assert viewport.view_state()["source"] == "run" and viewport.stage_preview_metadata is None
    assert viewport._stage_arrays == {}


def test_clearing_live_preview_returns_to_inputs_preserving_saved_arrays(scientific_viewport):
    viewport = scientific_viewport
    viewport.render_config(preset_config())
    saved = _saved_arrays()
    viewport.display_run(saved)
    viewport.display_stage_preview(saved, {"stage": "sheet"})
    viewport.clear_stage_preview()
    assert viewport.dataset.currentData() == "preview_target" and viewport.view_state()["source"] == "model"
    assert viewport.stage_preview_metadata is None and viewport._stage_arrays == {}
    np.testing.assert_array_equal(viewport._arrays["stream_function_A"], saved["stream_function_A"])


@pytest.mark.parametrize("component, axis", [("Bx", 0), ("By", 1)])
def test_first_live_snapshot_displays_requested_component_then_keeps_manual_choice(
    scientific_viewport, component, axis
):
    viewport = scientific_viewport
    arrays = {key: value for key, value in _saved_arrays().items() if key != "stream_function_A"}
    arrays["target_field_T"] = np.zeros((2, 3))
    arrays["target_field_T"][:, axis] = [-2e-6, 3e-6]
    metadata = {"pipeline_stage": "prepared", "request_sha256": "a" * 64, "target_field_component": component}
    viewport.display_stage_preview(arrays, metadata)
    assert viewport.component.currentText() == component
    assert np.ptp(viewport._dataset_mesh.active_scalars) > 0
    viewport.component.setCurrentText("|B|")
    viewport.display_stage_preview(arrays, {**metadata, "pipeline_stage": "sheet"})
    assert viewport.component.currentText() == "|B|"
    viewport.display_stage_preview(arrays, {**metadata, "request_sha256": "b" * 64})
    assert viewport.component.currentText() == component


def test_live_stream_function_caption_never_claims_saved_run(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_stage_preview(_saved_arrays(), {"pipeline_stage": "sheet", "target_field_component": "Bz"})
    assert "live continuous-sheet solution" in viewport.caption.text()
    assert "immutable" not in viewport.caption.text() and "run /" not in viewport.caption.text()
