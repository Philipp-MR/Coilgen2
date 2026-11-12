"""Viewing clarity: open edges, body filter, direction presets, ROI fit, located checks, shielding limit,
unwrapped view and grouped exports."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.cad import (
    PRINT_FORMER_SCOPE,
    GeometryOption,
    export_draft_banner,
    export_option_groups,
    export_preset_selection,
    grouped_export_options,
)
from coilgen.application.scene_spec import highlight_layers, scene_layers
from coilgen.application.scene_views import (
    BOTH,
    LEGACY_VIEW_NAMES,
    VIEW_PRESETS,
    apply_view_preset,
    body_labels,
    body_view_choices,
    filter_body_view,
    opening_loops,
    outline_paths,
    roi_bounds,
    saved_part_roles,
    support_roles,
    view_preset_name,
)


def cylinder(radius, *, window=False, nphi=36, nz=12, height=0.2, offset=0):
    """Open cylinder about z; ``window`` removes a 60° × middle-third patch at φ = 0."""
    phi = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
    z = np.linspace(-height / 2, height / 2, nz)
    vertices = np.array([[radius * np.cos(p), radius * np.sin(p), h] for h in z for p in phi])
    faces = []
    for k in range(nz - 1):
        for j in range(nphi):
            a, b = k * nphi + j, k * nphi + (j + 1) % nphi
            c, d = a + nphi, b + nphi
            middle = (phi[j] + np.pi / nphi) % (2 * np.pi)
            centre = min(middle, 2 * np.pi - middle)
            if window and centre < np.radians(30) and nz // 3 <= k < 2 * nz // 3:
                continue
            faces += [[a, b, d], [a, d, c]]
    return vertices, np.asarray(faces) + offset


def pair(window=True):
    inner, inner_faces = cylinder(0.08, window=window)
    outer, outer_faces = cylinder(0.1, window=window, offset=len(inner))
    return {"vertices_m": np.vstack([inner, outer]), "faces": np.vstack([inner_faces, outer_faces])}


def test_open_edges_are_drawn_after_the_support_and_windows_are_found():
    arrays = pair()
    layers, reason = scene_layers(arrays, "support")
    assert not reason and [layer.name for layer in layers] == ["support", "support_edges"]
    edges = layers[1]
    # Two ends per cylinder plus one window each.
    assert edges.kind == "lines" and edges.extra["count"] == 6 and edges.extra["line_width"] > 2
    assert all(np.allclose(path[0], path[-1]) for path in edges.lines)
    assert len(outline_paths(arrays["vertices_m"], arrays["faces"])) == 6
    windows = opening_loops(arrays["vertices_m"], arrays["faces"])
    assert len(windows) == 2
    assert all(np.all(path[:, 0] > 0) for path in windows)  # both sit at φ = 0 (+x)


def test_body_filter_keeps_only_the_chosen_role():
    arrays = pair(window=False)
    count = len(arrays["vertices_m"]) // 2
    arrays["stream_function_A"] = np.r_[np.ones(count), 2 * np.ones(count)]
    arrays["contour_0_0_points_m"] = arrays["vertices_m"][: count // 2]
    arrays["contour_1_0_points_m"] = arrays["vertices_m"][count : count + 10] * 1.001
    labels, bodies = body_labels(arrays["faces"], len(arrays["vertices_m"]))
    assert bodies == 2 and set(labels[:count]) == {0} and set(labels[count:]) == {1}
    roles = support_roles(arrays)
    assert roles == ["primary", "active_shield"]
    assert [choice["title"] for choice in body_view_choices(roles)] == ["Both", "Primary", "Shield"]
    assert body_view_choices(["primary", "primary"]) == []
    shield = filter_body_view(arrays, "active_shield", roles)
    radii = np.linalg.norm(shield["vertices_m"][:, :2], axis=1)
    assert np.allclose(radii, 0.1) and shield["faces"].max() < len(shield["vertices_m"])
    assert np.all(shield["stream_function_A"] == 2)
    assert "contour_1_0_points_m" in shield and "contour_0_0_points_m" not in shield
    assert filter_body_view(arrays, BOTH, roles) is arrays
    # Recorded roles win over the radius guess.
    assert support_roles(arrays, ["active_shield", "primary"]) == ["active_shield", "primary"]
    metadata = {"fabrication": {"paths": [{"body_id": "part001_body0001", "role": "shield"}]}}
    assert saved_part_roles(metadata) == {0: "active_shield"}


def test_view_presets_say_where_they_look_from():
    pv = pytest.importorskip("pyvista")
    assert list(VIEW_PRESETS)[0] == "Oblique" and "From +x (YZ)" in VIEW_PRESETS
    assert {view_preset_name(name) for name in LEGACY_VIEW_NAMES} <= set(VIEW_PRESETS)
    plotter = pv.Plotter(off_screen=True)
    try:
        plotter.add_mesh(pv.Sphere())
        assert apply_view_preset(plotter, "From +x (YZ)")
        np.testing.assert_allclose(plotter.camera.direction, (-1, 0, 0), atol=1e-12)
        assert apply_view_preset(plotter, "YZ")
        np.testing.assert_allclose(plotter.camera.direction, (-1, 0, 0), atol=1e-12)
        assert apply_view_preset(plotter, "Oblique")
        # An oblique view looks well away from the cylinder axis.
        assert abs(plotter.camera.direction[2]) < 0.4
        assert not apply_view_preset(plotter, "Sideways")
    finally:
        plotter.close()


def test_roi_bounds_leave_out_shielding_samples():
    points = np.array([[0, 0, 0], [0.02, 0.02, 0.02], [0.15, 0, 0], [-0.15, 0, 0.2]])
    arrays = {
        "target_points_m": points,
        "target_field_T": np.zeros((4, 3)),
        "target_suppressed": np.array([False, False, True, True]),
    }
    bounds = np.asarray(roi_bounds(arrays, "target")).reshape(3, 2)
    assert bounds[0, 1] < 0.05 and bounds[0, 0] < 0 < bounds[0, 1]
    assert roi_bounds(arrays, "stream_function") is None


def test_highlight_layers_and_problem_groups_carry_the_witness(tmp_path):
    from coilgen.application.diagnostics import ProblemIssue
    from coilgen.application.service import RunRecord
    from coilgen.web.result_display import problem_groups

    layers = highlight_layers([[0, 0, 0], [0, 0, 0.001]], [[0, 1]])
    assert [layer.name for layer in layers] == ["highlight_points", "highlight_segments"]
    assert highlight_layers([[0, 0, np.nan]], []) == [] and highlight_layers([[0, 0, 0]], [[0, 3]]) == []
    issue = ProblemIssue(
        "fabrication:part001_body0001:clearance",
        "Clearance too small",
        "",
        body_id="part001_body0001",
        points_m=((0.1, 0, 0), (0.1, 0.001, 0)),
        segments=((0, 1),),
        location_description="Recorded interval pair",
        scope="fabrication_review",
    )
    (group,) = problem_groups(RunRecord(tmp_path, {}, {}), [issue])
    item = group["items"][0]
    assert item["locatable"] and item["points_m"] == [[0.1, 0, 0], [0.1, 0.001, 0]] and item["segments"] == [[0, 1]]
    assert item["dataset"] == "review_conductor" and item["where"] == "Recorded interval pair"


def test_browser_locates_a_check_and_offers_bodies_and_roi(tmp_path, monkeypatch):
    pytest.importorskip("trame")
    from coilgen.application.service import RunRecord
    from coilgen.web.app import create_app

    server, browser = create_app(tmp_path, server_name="test-browser-view-clarity")
    state = server.state
    arrays = {
        **pair(),
        "review_conductor_0_points_m": np.array([[0.081, 0, -0.05], [0.081, 0, 0.05]]),
        "target_points_m": np.array([[0, 0, 0], [0.01, 0, 0], [0.2, 0, 0]]),
        "target_field_T": np.zeros((3, 3)),
        "target_suppressed": np.array([False, False, True]),
    }
    record = RunRecord(tmp_path, {"config": {"b_0_direction": [0, 0, 1]}}, arrays)
    monkeypatch.setattr(browser.session, "load_run", lambda _key: record)
    monkeypatch.setattr(browser.session, "export_options", lambda _record: ())
    browser._open_run("saved")
    assert state.dataset == "review_conductor"
    assert [choice["value"] for choice in state.body_views] == ["both", "primary", "active_shield"]
    assert state.unwrapped_available
    assert "support_edges" in browser.plotter.renderer.actors
    state.body_view = "active_shield"
    browser._render()
    # The only wire is on the primary, so the shield alone would show nothing of this plot.
    assert state.body_note and "review_conductor" in browser.plotter.renderer.actors
    state.dataset = "target"
    browser._render()
    support = browser.plotter.renderer.actors["support"].mapper.dataset
    assert not state.body_note and np.allclose(np.linalg.norm(support.points[:, :2], axis=1), 0.1)
    state.body_view = "both"
    browser._render()
    # Off by default: a camera fitted to the ROI would sit inside the support.
    assert state.roi_available and not state.roi_fit and browser.fit_bounds is None
    server.controller.toggle_roi_fit()
    assert state.roi_fit and np.asarray(browser.fit_bounds)[1] < 0.05
    server.controller.locate_problem([[0.081, 0, -0.05], [0.081, 0, 0.05]], [[0, 1]], "review_conductor", "Gap")
    assert state.dataset == "review_conductor" and state.highlight_note == "Located: Gap"
    assert state.see_through_support  # the witness is never hidden behind an opaque support
    assert "highlight_segments" in browser.plotter.renderer.actors
    server.controller.clear_highlight()
    assert not state.highlight_note and "highlight_points" not in browser.plotter.renderer.actors
    server.controller.show_unwrapped()
    assert state.center_section == "unwrapped" and state.unwrapped_image.startswith("data:image/png;base64,")


def test_shielding_plot_marks_the_limit_and_outlines_openings():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from coilgen.application.shielding import ShieldingSurfaceFields
    from coilgen.gui.shielding_plot import draw_shielding_surface, limit_verdict, record_openings, shielding_limits

    metadata = {"fabricated_physics": {"sampled_suppression": {"rms_limit_T": 20e-6, "peak_limit_T": None}}}
    assert shielding_limits(metadata) == {"rms_T": 20e-6}
    assert shielding_limits({"config": {"suppression_field_peak_limit_T": 5e-5}}) == {"peak_T": 5e-5}
    assert limit_verdict(26.34, 40, {"rms_T": 20e-6}) == "Combined RMS 26.3 µT > limit 20 µT · Fails"
    assert "Passes" in limit_verdict(10, 40, {"rms_T": 20e-6})
    vertices, faces = cylinder(0.15, nphi=24, nz=8)
    total = np.tile([0, 0, 30e-6], (len(vertices), 1))
    fields = ShieldingSurfaceFields(vertices, total, None, None, None, None, faces, 1.0, "test", None)
    figure = Figure()
    windows = opening_loops(*pair().values())
    note = draw_shielding_surface(figure, fields, "magnitude", limits={"rms_T": 20e-6}, openings_m=windows)
    assert "Fails" in figure._suptitle.get_text() and "2 support openings outlined" in note
    record = SimpleNamespace(array_names=("vertices_m", "faces"), array=lambda name, copy=False: pair()[name])
    assert len(record_openings(record)) == 2
    assert record_openings(SimpleNamespace(array_names=())) == ()


def test_unwrapped_view_shows_primary_and_shield_side_by_side():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from coilgen.application.unwrapped_view import draw_unwrapped, is_cylindrical, unwrapped_image

    arrays = pair()
    arrays["stream_function_A"] = np.linspace(-1, 1, len(arrays["vertices_m"]))
    arrays["contour_0_0_points_m"] = arrays["vertices_m"][:36]
    assert is_cylindrical(arrays)
    assert not is_cylindrical({"vertices_m": np.eye(3), "faces": np.array([[0, 1, 2]])})
    figure = Figure()
    note = draw_unwrapped(figure, arrays)
    titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
    assert titles == ["Primary · r 80 mm", "Active shield · r 100 mm"]
    assert note.startswith("2 cylinders · 2 openings · stream function shown · 1 contours")
    image, _ = unwrapped_image(arrays)
    assert image.startswith("data:image/png;base64,")


def options():
    def option(name, fmt, role=None, scope="fabrication_review_geometry"):
        body = name.split(".")[0]
        return GeometryOption(name, body, fmt, scope, "passed", "m", ((0, 0, 0), (1, 1, 1)), False, "0", role)

    return (
        option("shield_DRAFT_fabrication.stl", "stl", "active_shield"),
        option("former.stl", "stl", scope=PRINT_FORMER_SCOPE),
        option("primary.step", "step", "primary"),
        option("primary_centroid.csv", "path_csv", "primary"),
    )


def test_exports_group_by_role_with_presets_and_one_draft_banner():
    grouped = grouped_export_options(options())
    assert [option.name for option in grouped] == [
        "primary.step",
        "primary_centroid.csv",
        "shield_DRAFT_fabrication.stl",
        "former.stl",
    ]
    assert [group["title"] for group in export_option_groups(options())] == ["Primary", "Active shield", "Print former"]
    assert export_preset_selection(options(), "step") == ("primary.step",)
    assert export_preset_selection(options(), "stl") == ("shield_DRAFT_fabrication.stl", "former.stl")
    assert export_preset_selection(options(), "csv") == ("primary_centroid.csv",)
    assert export_draft_banner(options()).startswith("DRAFT: 1 of 4 files")
    assert export_draft_banner(options()[1:]) == ""


def test_desktop_export_panel_groups_rows_and_applies_presets():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from coilgen.gui.export_panel import GeometryExportPanel

    QApplication.instance() or QApplication(["view-clarity"])
    record = SimpleNamespace(directory=Path("run"), metadata={"name": "run"})
    panel = GeometryExportPanel(record, options())
    try:
        headers = [panel.table.verticalHeaderItem(row).text() for row in range(panel.table.rowCount())]
        assert headers == ["Primary", "Primary", "Active shield", "Print former"]
        assert panel.draft_banner.text().startswith("DRAFT: 1 of 4") and not panel.draft_banner.isHidden()
        assert set(panel.preset_buttons) == {"step", "stl", "csv"}
        panel.preset_buttons["step"].click()
        assert panel.selected_names() == ("primary.step",)
        panel.preset_buttons["stl"].click()
        assert panel.selected_names() == ("shield_DRAFT_fabrication.stl", "former.stl")
        assert not panel.export_button.isEnabled()  # DRAFT still needs the confirmation
        assert panel.table.item(0, 0).checkState() == Qt.CheckState.Unchecked
    finally:
        panel.close()
