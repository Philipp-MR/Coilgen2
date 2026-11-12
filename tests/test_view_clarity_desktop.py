"""Desktop viewport: open edges, Primary / Shield filter, direction presets, Fit ROI and the unwrapped view."""

import numpy as np

import test_native_workbench as workbench_tests
from test_view_clarity import pair

application = workbench_tests.application
scientific_viewport = workbench_tests.scientific_viewport


def run_arrays():
    arrays = pair()
    count = len(arrays["vertices_m"])
    arrays.update(
        stream_function_A=np.linspace(-1, 1, count),
        target_points_m=np.array([[0, 0, 0], [0.01, 0, 0], [0.2, 0, 0]]),
        target_field_T=np.zeros((3, 3)),
        target_suppressed=np.array([False, False, True]),
    )
    return arrays


def outline_meshes(viewport):
    return [mesh for mesh, options, _ in viewport.plotter.meshes if options.get("color") == "#b4651e"]


def test_desktop_viewport_outlines_filters_bodies_and_fits_the_roi(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    assert viewport.view.itemText(0) == "Oblique" and viewport.view.findText("From +x (YZ)") >= 0
    viewport.display_run(run_arrays(), {})
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("stream_function"))
    assert len(outline_meshes(viewport)) == 1 and viewport.body_view.isEnabled() and viewport.unwrapped.isEnabled()
    viewport.body_view.setCurrentIndex(viewport.body_view.findData("active_shield"))
    surface = next(mesh for mesh, options, _ in viewport.plotter.meshes if options.get("scalars"))
    assert np.allclose(np.linalg.norm(np.asarray(surface.points)[:, :2], axis=1), 0.1)
    viewport.open_edges.setChecked(False)
    assert not outline_meshes(viewport)
    assert not viewport.fit_roi.isEnabled()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("target"))
    assert viewport.fit_roi.isEnabled()
    fits = []
    monkeypatch.setattr(viewport.plotter, "reset_camera", lambda bounds=None: fits.append(bounds))
    viewport.fit_roi.click()
    assert np.asarray(fits[-1]).reshape(3, 2)[0, 1] < 0.05
    dialog = viewport.show_unwrapped()
    assert dialog is not None
    dialog.close()


def test_saved_view_state_with_old_preset_name_is_accepted(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(run_arrays(), {})
    state = viewport.view_state()
    state["view"] = "YZ"
    state.pop("camera", None)
    assert viewport.restore_view_state(state)
    assert viewport.view.currentText() == "From +x (YZ)"
