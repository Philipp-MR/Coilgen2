"""Swept conductor rendering uses stored triangle geometry without reconstruction."""

import numpy as np
import pytest

import test_native_workbench as workbench_tests

application = workbench_tests.application
scientific_viewport = workbench_tests.scientific_viewport


def body_arrays(index=0, shift=0.0):
    vertices = np.array([[0.1, 0.0, 0.0], [0.1, 0.002, 0.0], [0.1, 0.0, 0.003], [0.102, 0.0, 0.0]]) + shift
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    return {f"swept_0_{index}_vertices_m": vertices, f"swept_0_{index}_faces": faces}


def test_swept_dataset_renders_exact_saved_vertices_and_faces_without_centerline_recreation(
    scientific_viewport, monkeypatch
):
    viewport = scientific_viewport
    arrays = {**body_arrays(), **body_arrays(1, 0.01)}
    arrays["fabrication_0_points_m"] = np.array([[999, 999, 999], [1000, 1000, 1000]])
    original = {key: value.copy() for key, value in arrays.items()}
    monkeypatch.setattr(
        viewport.pv, "lines_from_points", lambda *_args, **_kwargs: pytest.fail("Centerline recreated as swept mesh")
    )
    viewport.display_run(arrays, {"summary": {"accepted": False, "scope": "fabrication_review"}})
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("swept"))
    assert len(viewport.plotter.meshes) == 2
    for index, (mesh, options, actor) in enumerate(viewport.plotter.meshes):
        np.testing.assert_array_equal(mesh.points, arrays[f"swept_0_{index}_vertices_m"])
        np.testing.assert_array_equal(mesh.faces.reshape(-1, 4)[:, 1:], arrays[f"swept_0_{index}_faces"])
        assert mesh.n_cells == 4
        assert options["show_edges"]
        assert viewport._actors[viewport._actor_key(actor)] == "winding"
    assert "2 bodies" in viewport.caption.text()
    assert "inspection only" in viewport.caption.text()
    assert "manufacturing acceptance requires separate checks" in viewport.caption.text()
    assert not viewport.component.isEnabled()
    assert not viewport.units.isEnabled()
    for key, values in original.items():
        np.testing.assert_array_equal(arrays[key], values)


def test_older_run_without_solid_arrays_remains_a_separate_centerline_view(scientific_viewport):
    viewport = scientific_viewport
    path = np.array([[0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]])
    errors = []
    viewport.error.connect(errors.append)
    viewport.display_run({"fabrication_0_points_m": path})
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("swept"))
    assert viewport.dataset.currentData() == "fabrication"
    assert not viewport.dataset.model().item(viewport.dataset.findData("swept")).isEnabled()
    assert errors and "No saved conductor solids" in errors[-1]
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("fabrication"))
    assert len(viewport.plotter.meshes) == 1
    np.testing.assert_array_equal(viewport.plotter.meshes[0][0].points, path)
    assert "centerlines" in viewport.caption.text()


def test_swept_body_clipping_receives_original_triangles(scientific_viewport):
    viewport = scientific_viewport
    arrays = {**body_arrays(), **body_arrays(1, 0.01)}
    viewport.display_run(arrays)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("swept"))
    viewport.clip.setChecked(True)
    assert len(viewport.plotter.plane_clipped_meshes) == 2
    for index, mesh in enumerate(viewport.plotter.plane_clipped_meshes):
        np.testing.assert_array_equal(mesh.points, arrays[f"swept_0_{index}_vertices_m"])
        np.testing.assert_array_equal(mesh.faces.reshape(-1, 4)[:, 1:], arrays[f"swept_0_{index}_faces"])


def test_swept_dataset_selection_roundtrips_without_changing_other_persisted_controls(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(body_arrays())
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("swept"))
    viewport.units.setCurrentText("µT")
    viewport.view.setCurrentText("YZ")
    state = viewport.view_state()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("fabrication"))
    assert viewport.restore_view_state(state)
    assert viewport.view_state() == state


def test_incomplete_saved_solid_is_a_visible_error(scientific_viewport):
    viewport = scientific_viewport
    arrays = body_arrays()
    arrays.pop("swept_0_0_faces")
    errors = []
    viewport.error.connect(errors.append)
    viewport.display_run(arrays)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("swept"))
    assert errors and "Missing swept_0_0_faces" in errors[-1]
    assert viewport.dataset.currentData() == "support"
    assert not viewport.dataset.model().item(viewport.dataset.findData("swept")).isEnabled()
