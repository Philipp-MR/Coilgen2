"""Stage-aware viewing and evidence inspection preserve saved engineering data."""

from copy import deepcopy

import numpy as np
import pytest

import test_native_workbench as workbench_tests
from coilgen.application.diagnostics import ProblemIssue

application = workbench_tests.application
scientific_viewport = workbench_tests.scientific_viewport


def saved_geometry():
    points = np.array([[0.1, 0, 0], [0.1, 0.002, 0], [0.1, 0, 0.003]])
    return {
        "vertices_m": points,
        "faces": np.array([[0, 1, 2]]),
        "stream_function_A": np.array([0, 1, 2]),
        "target_points_m": points.copy(),
        "target_field_T": np.array([[0, 0, 0], [0, 0, 1e-6], [0, 0, 2e-6]]),
    }


@pytest.mark.parametrize(
    "key, preferred",
    [
        (None, "stream_function"),
        ("contour_0_points_m", "contours"),
        ("route_0_points_m", "routes"),
        ("evaluated_winding_0_points_m", "evaluated_winding"),
        ("fabrication_0_points_m", "fabrication"),
        ("review_conductor_0_points_m", "review_conductor"),
    ],
)
def test_first_result_view_selects_saved_completed_geometry(scientific_viewport, key, preferred):
    arrays = saved_geometry()
    if key is not None:
        arrays[key] = arrays["vertices_m"].copy()
    before = deepcopy(arrays)
    scientific_viewport.display_run(arrays)
    assert scientific_viewport.dataset.currentData() == preferred
    for name, values in before.items():
        np.testing.assert_array_equal(scientific_viewport._arrays[name], values)
        np.testing.assert_array_equal(arrays[name], values)


def test_exact_solid_is_preferred_and_strict_review_remains_distinct(scientific_viewport):
    viewport = scientific_viewport
    arrays = saved_geometry()
    arrays.update(
        swept_0_0_vertices_m=arrays["vertices_m"].copy(),
        swept_0_0_faces=arrays["faces"].copy(),
        fabrication_0_points_m=arrays["vertices_m"].copy(),
    )
    viewport.display_run(arrays, {"summary": {"scope": "continuous_sheet", "accepted": True}})
    assert viewport.dataset.currentData() == "swept"
    np.testing.assert_array_equal(viewport.plotter.meshes[-1][0].points, arrays["swept_0_0_vertices_m"])
    assert "inspection only" in viewport.caption.text()
    reviewed = arrays["vertices_m"] + [0, 0, 0.001]
    arrays["review_conductor_0_points_m"] = reviewed
    viewport.display_run(arrays, {"summary": {"scope": "fabrication_review", "accepted": False}})
    assert viewport.dataset.currentData() == "review_conductor"
    np.testing.assert_array_equal(viewport.plotter.meshes[-1][0].points, reviewed)
    assert "fabrication-review conductor" in viewport.caption.text()


def test_grouped_selector_retains_all_session_keys_and_skips_headers(scientific_viewport):
    viewport = scientific_viewport
    expected = {
        "support",
        "preview_target",
        "stream_function",
        "target",
        "sheet",
        "validation",
        "contour_field",
        "contour_residual",
        "routed_field",
        "routed_residual",
        "contours",
        "routes",
        "evaluated_winding",
        "fabrication",
        "swept",
        "review_conductor",
        "review_electrical",
        "fabricated",
    }
    keys = set()
    headers = []
    for index in range(viewport.dataset.count()):
        key = viewport.dataset.itemData(index)
        if key is None:
            headers.append(viewport.dataset.model().item(index))
        else:
            keys.add(key)
    assert keys == expected
    assert len(headers) == 4 and all(not header.isEnabled() for header in headers)
    assert viewport.dataset.maxVisibleItems() >= viewport.dataset.count()
    assert viewport.dataset.minimumContentsLength() == 24


def test_saved_view_overrides_first_result_choice_and_camera_without_refitting(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    arrays = saved_geometry()
    viewport.display_run(arrays)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("target"))
    viewport.plotter.camera_position = [[0.3, 0.2, 0.1], [0.1, 0, 0], [0, 0, 1]]
    viewport.units.setCurrentText("µT")
    state = viewport.view_state()
    arrays["swept_0_0_vertices_m"] = arrays["vertices_m"].copy()
    arrays["swept_0_0_faces"] = arrays["faces"].copy()
    viewport.display_run(arrays)
    assert viewport.dataset.currentData() == "swept"
    monkeypatch.setattr(viewport.plotter, "reset_camera", lambda **_kwargs: pytest.fail("Saved camera was refitted"))
    assert viewport.restore_view_state(state)
    assert viewport.view_state() == state


def test_geometry_view_uses_sparse_axes_and_reserves_sample_slab_for_fields(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    grids = []
    monkeypatch.setattr(viewport.plotter, "show_grid", lambda **kwargs: grids.append(kwargs))
    viewport.display_run(saved_geometry())
    assert viewport.slice_controls.isHidden()
    assert grids[-1]["n_xlabels"] == grids[-1]["n_ylabels"] == grids[-1]["n_zlabels"] == 2
    assert grids[-1]["grid"] is False
    assert not viewport.component.isEnabled() and not viewport.units.isEnabled()
    assert "metres" in viewport.units.toolTip()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("target"))
    assert not viewport.slice_controls.isHidden()


def test_evidence_fit_updates_orthographic_scale_fades_context_and_can_restore_view(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    arrays = saved_geometry()
    arrays["review_conductor_0_points_m"] = arrays["vertices_m"].copy()
    viewport.display_run(arrays)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("stream_function"))
    viewport.plotter.camera = viewport.pv.Camera()
    viewport.plotter.camera.parallel_scale = 0.3
    before_camera = deepcopy(viewport.plotter.camera_position)
    fits = []

    def fit(**kwargs):
        fits.append(kwargs["bounds"])
        viewport.plotter.camera.parallel_scale = 0.004

    monkeypatch.setattr(viewport.plotter, "reset_camera", fit)
    issue = ProblemIssue(
        "clearance",
        "Recorded clearance witness",
        "Inspect the saved intervals",
        body_id="part001_body0001",
        points_m=((0.1, 0, 0), (0.1, 0.000001, 0)),
        segments=((0, 1),),
        location_description="Actual recorded interval pair",
    )
    assert viewport.locate_issue(issue)
    assert len(fits) == 1
    bounds = np.asarray(fits[0]).reshape(3, 2)
    points = np.asarray(issue.points_m)
    assert np.all(points >= bounds[:, 0]) and np.all(points <= bounds[:, 1])
    assert viewport.plotter.camera.parallel_scale == 0.004
    assert not viewport.clear_highlight.isHidden()
    assert all(options["opacity"] <= 0.18 for _, options, _ in viewport.plotter.meshes[:-1])
    marker, options, _ = viewport.plotter.meshes[-1]
    np.testing.assert_array_equal(marker.points, points)
    assert options.get("opacity", 1) == 1 and marker.n_lines == 1
    viewport.clear_highlight.click()
    assert viewport._located_issue is None and viewport.clear_highlight.isHidden()
    assert viewport.dataset.currentData() == "stream_function"
    assert viewport.plotter.camera.parallel_scale == 0.3
    assert viewport.plotter.camera_position == before_camera
    for name, values in arrays.items():
        np.testing.assert_array_equal(viewport._arrays[name], values)


def test_switching_dataset_clears_witness_from_another_scope(scientific_viewport):
    viewport = scientific_viewport
    arrays = saved_geometry()
    arrays["review_conductor_0_points_m"] = arrays["vertices_m"].copy()
    viewport.display_run(arrays)
    issue = ProblemIssue("witness", "Witness", "Recorded", points_m=((0.1, 0, 0),))
    assert viewport.locate_issue(issue)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("target"))
    assert viewport._located_issue is None and viewport.clear_highlight.isHidden()
    assert "highlighted" not in viewport.caption.text()
    np.testing.assert_array_equal(viewport._dataset_mesh.points, arrays["target_points_m"])


def test_saved_orthographic_zoom_roundtrips_and_old_sessions_remain_supported(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(saved_geometry())
    viewport.plotter.camera = viewport.pv.Camera()
    viewport.plotter.camera.parallel_scale = 0.017
    state = viewport.view_state()
    viewport.plotter.camera.parallel_scale = 0.4
    assert viewport.restore_view_state(state)
    assert viewport.plotter.camera.parallel_scale == 0.017
    state.pop("parallel_scale")
    assert viewport.restore_view_state(state)


@pytest.mark.parametrize("scale", [0, -1, True, float("nan")])
def test_invalid_saved_zoom_is_rejected_atomically(scientific_viewport, scale):
    viewport = scientific_viewport
    before = viewport.view_state()
    assert not viewport.restore_view_state(dict(before, parallel_scale=scale))
    assert viewport.view_state() == before


def test_orthographic_fit_uses_projected_scene_extent_and_retains_label_margin(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    viewport.plotter.camera = viewport.pv.Camera()
    camera = viewport.plotter.camera
    camera.position, camera.focal_point, camera.up = (0, 0, 1), (0, 0, 0), (0, 1, 0)
    camera.parallel_scale = 0.5
    viewport.plotter.bounds = (-0.1, 0.1, -0.2, 0.2, -0.15, 0.15)
    viewport.plotter.window_size = (400, 300)
    monkeypatch.setattr(viewport.plotter, "reset_camera", lambda: None)
    viewport._fit_view()
    assert camera.parallel_scale == pytest.approx(0.25)
    # A tall narrow viewport must fit the horizontal extent as well.
    camera.parallel_scale = 2
    viewport.plotter.window_size = (100, 400)
    viewport._fit_view()
    assert camera.parallel_scale == pytest.approx(0.5)


def test_fit_leaves_perspective_projection_and_saved_scale_untouched(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    viewport.plotter.camera = viewport.pv.Camera()
    viewport.plotter.camera.parallel_scale = 0.5
    viewport.plotter.bounds = (-0.1, 0.1, -0.2, 0.2, -0.15, 0.15)
    viewport.plotter.window_size = (400, 300)
    monkeypatch.setattr(viewport.plotter, "reset_camera", lambda: None)
    viewport.orthographic.setChecked(False)
    viewport._fit_view()
    assert viewport.plotter.camera.parallel_scale == 0.5
