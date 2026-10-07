"""Professional viewer controls retain scope, actual witnesses and immutable runs."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_native_workbench as workbench_tests
from coilgen.application.diagnostics import ProblemIssue
from coilgen.gui.problems import ProblemsPanel

application = workbench_tests.application
scientific_viewport = workbench_tests.scientific_viewport


def test_problems_panel_locate_only_with_actual_witness(application):
    from PySide6.QtCore import QCoreApplication, QEvent

    panel = ProblemsPanel()
    record = SimpleNamespace(
        directory=Path.cwd(),
        arrays={},
        metadata={"constraints": [{"name": "Unsupported validation", "passed": None}], "warnings": []},
    )
    panel.set_run(record)
    assert len(panel.issues) == 1 and not panel.locate.isEnabled()
    assert "unverified" in panel.details.text()
    navigation = []
    panel.navigate_requested.connect(navigation.append)
    panel.navigate.click()
    assert navigation == []
    assert not panel.navigate.isEnabled()
    assert "original request.json" in panel.navigate.toolTip()
    record.metadata["artifacts"] = [{"name": "request.json", "sha256": "a" * 64}]
    panel.set_run(record)
    panel.navigate.click()
    assert navigation == ["optimization"]
    panel.set_run(record, diagnostic="Artifact checksum changed")
    assert not panel.navigate.isEnabled()
    panel.navigate.click()
    assert navigation == ["optimization"]
    panel.close()
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_known_integrity_failure_blocks_recorded_witness_navigation(application, monkeypatch):
    from coilgen.gui import problems

    issue = ProblemIssue("clearance", "Failed clearance", "Saved witness", points_m=((0.1, 0, 0),))
    monkeypatch.setattr(problems, "run_problems", lambda _record: (issue,))
    panel = ProblemsPanel()
    record = SimpleNamespace(metadata={"artifacts": [{"name": "request.json", "sha256": "a" * 64}]})
    panel.set_run(record)
    assert panel.locate.isEnabled()
    received = []
    panel.locate_requested.connect(received.append)
    panel.set_run(record, diagnostic="Artifact checksum changed")
    assert not panel.locate.isEnabled()
    assert "checksum" in panel.locate.toolTip()
    panel._locate()
    assert not received
    panel.close()
    panel.deleteLater()
    application.processEvents()


def test_locate_marks_recorded_intervals_and_focuses_without_mutating_run(scientific_viewport):
    viewport = scientific_viewport
    route = np.array([[0.1, 0, 0], [0, 0.1, 0], [-0.1, 0, 0]])
    viewport.display_run({"review_conductor_0_points_m": route})
    issue = ProblemIssue(
        "collision",
        "Recorded interval pair",
        "Inspect saved intervals",
        points_m=((0.1, 0, 0), (0, 0.1, 0)),
        segments=((0, 1),),
        body_id="part001_body0001",
    )
    assert viewport.locate_issue(issue)
    marker = viewport.plotter.meshes[-1][0]
    np.testing.assert_array_equal(marker.points, issue.points_m)
    assert marker.n_lines == 1
    assert "Located part001_body0001" in viewport.probe.text()
    np.testing.assert_array_equal(viewport._arrays["review_conductor_0_points_m"], route)
    assert not viewport.locate_issue(ProblemIssue("no witness", "No witness", "Unavailable"))


def test_view_state_roundtrip_and_clear_run_preserve_current_model(scientific_viewport):
    from coilgen.application.presets import preset_config

    viewport = scientific_viewport
    preflight = []
    viewport.preflight_ready.connect(preflight.append)
    viewport.render_config(preset_config())
    support = viewport._support
    assert preflight[-1]["target"]["available"]
    viewport.display_run({"target_points_m": [[0, 0, 0]], "target_field_T": [[0, 0, 1e-6]]}, {"name": "Old run"})
    viewport.units.setCurrentText("µT")
    viewport.component.setCurrentText("By")
    viewport.roi_layer.setChecked(True)
    viewport.slice_axis.setCurrentText("z")
    viewport.slice_position.setValue(33)
    state = viewport.view_state()
    assert state["source"] == "run"
    viewport.units.setCurrentText("T")
    viewport.roi_layer.setChecked(False)
    assert viewport.restore_view_state(state)
    assert viewport.view_state() == state
    viewport.clear_run()
    assert not viewport._arrays and not viewport._metadata
    assert viewport._support is support
    assert viewport.dataset.currentData() == "preview_target" and viewport.view_state()["source"] == "model"


@pytest.mark.parametrize(
    "bad",
    [
        {"units": "mm"},
        {"dataset": "missing"},
        {"slice_position": -1},
        {"slice_position": True},
        {"orthographic": "true"},
        {"camera": [[float("nan"), 0, 0], [0, 0, 0], [0, 0, 1]]},
        {"camera": [[1, 0, 0], [1, 0, 0], [0, 0, 1]]},
        {"camera": [[1, 0, 0], [0, 0, 0], [1, 0, 0]]},
        {"camera": [[True, 0, 0], [0, 0, 0], [0, 0, 1]]},
        {"source": "other"},
    ],
)
def test_invalid_session_view_is_rejected_atomically(scientific_viewport, bad):
    viewport = scientific_viewport
    before = viewport.view_state()
    errors = []
    viewport.error.connect(errors.append)
    state = deepcopy(before)
    state.update(bad)
    assert viewport.restore_view_state(state) is False
    assert viewport.view_state() == before
    assert errors and "rejected" in errors[-1]


def test_two_runs_share_actual_colour_range_controls_and_camera(scientific_viewport):
    from PySide6.QtCore import QCoreApplication, QEvent
    from coilgen.gui.viewport import ComparisonDialog, ScientificViewport

    # scientific_viewport installs the renderless VTK interactor fixture.
    left = SimpleNamespace(
        directory=Path("left"),
        metadata={"name": "Left"},
        arrays={
            "target_points_m": np.array([[0, 0, 0], [0, 0, 0.01]]),
            "target_field_T": np.array([[0, 0, 1e-6], [0, 0, 2e-6]]),
        },
    )
    right = SimpleNamespace(
        directory=Path("right"),
        metadata={"name": "Right"},
        arrays={
            "target_points_m": np.array([[0.1, 0, 0], [0.1, 0, 0.02]]),
            "target_field_T": np.array([[0, 0, -3e-6], [0, 0, 4e-6]]),
        },
    )
    dialog = ComparisonDialog(
        left, right, viewport_factory=lambda parent: ScientificViewport(parent, synchronous_preview=True)
    )
    dialog.left.units.setCurrentText("µT")
    assert dialog.right.units.currentText() == "µT"
    assert dialog.left._colour_limits == dialog.right._colour_limits == (-3, 4)
    assert dialog.left.plotter.meshes[-1][1]["clim"] == (-3, 4)
    assert dialog.right.plotter.meshes[-1][1]["clim"] == (-3, 4)
    np.testing.assert_array_equal(dialog.left._dataset_mesh.points, left.arrays["target_points_m"])
    np.testing.assert_array_equal(dialog.right._dataset_mesh.points, right.arrays["target_points_m"])
    assert dialog.left.plotter.camera is dialog.right.plotter.camera
    dialog.left.plotter.camera.SetPosition(1, 2, 3)
    assert dialog.right.plotter.camera.GetPosition() == (1, 2, 3)
    dialog.right.dataset.setCurrentIndex(dialog.right.dataset.findData("routed_field"))
    assert dialog.left.dataset.currentData() == dialog.right.dataset.currentData() == "target"
    assert dialog.left._dataset_mesh is not None and dialog.right._dataset_mesh is not None
    assert not dialog.left.dataset.model().item(dialog.left.dataset.findData("routed_field")).isEnabled()
    dialog.close()
    assert dialog._camera_observer is None
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_replacing_comparison_does_not_lose_new_dialog_after_old_deferred_close(scientific_viewport):
    from PySide6.QtCore import QCoreApplication, QEvent

    viewport = scientific_viewport
    run = SimpleNamespace(directory=Path("run"), metadata={}, arrays={})
    first = viewport.display_comparison(run, run)
    second = viewport.display_comparison(run, run)
    assert first is not second and viewport._comparison is second
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert viewport._comparison is second and not second._closed
    second.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert viewport._comparison is None


@pytest.mark.parametrize("font_points", [9, 12])
def test_comparison_controls_fit_two_panes_on_1280_pixel_display(
    scientific_viewport, application, monkeypatch, font_points
):
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QPushButton
    from coilgen.gui.viewport import ComparisonDialog

    viewport = scientific_viewport
    screen = SimpleNamespace(availableGeometry=lambda: QRect(0, 0, 1280, 700))
    monkeypatch.setattr(ComparisonDialog, "screen", lambda _self: screen)
    # Windows' offscreen platform supplies no system fonts. Register the actual
    # native UI font for meaningful geometry rather than fallback glyph widths.
    font_id = -1
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    original_font = QFont(application.font())
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        assert font_id >= 0
        application.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], font_points))
    elif not QFontDatabase.families():
        pytest.skip("Native UI font is unavailable on this offscreen platform.")
    record = SimpleNamespace(
        directory=Path("conical_full_run"),
        metadata={"name": "Conical Y20 harmonic study / strict fabrication review"},
        arrays={
            "vertices_m": np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]),
            "faces": np.array([[0, 1, 2]]),
            "stream_function_A": np.arange(3, dtype=float),
            "target_points_m": np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]),
            "target_field_T": np.zeros((3, 3)),
            "routed_field_T": np.zeros((3, 3)),
        },
    )
    dialog = viewport.display_comparison(record, record)
    try:
        if font_id >= 0:
            dialog.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], font_points))
        application.processEvents()
        assert dialog.minimumSizeHint().width() <= 1280
        assert dialog.width() == 1280
        assert dialog.height() == 700
        dialog.left.dataset.setCurrentIndex(dialog.left.dataset.findData("stream_function"))
        application.processEvents()
        assert all(not pane.component.isEnabled() for pane in dialog.panes)
        assert all(pane.field_options.isHidden() and not pane.current_controls.isHidden() for pane in dialog.panes)
        dialog.left.dataset.setCurrentIndex(dialog.left.dataset.findData("routed_field"))
        application.processEvents()
        for pane in dialog.panes:
            fit = next(button for button in pane.findChildren(QPushButton) if button.text() == "Fit")

            def rect(control):
                return QRect(control.mapTo(pane, QPoint(0, 0)), control.size())

            assert pane.plot_selection_bar.isVisible() and not pane.field_controls.isHidden()
            assert rect(pane.units).top() > rect(pane.dataset).bottom()
            assert rect(pane.wire_current).top() > rect(pane.units).bottom()
            assert rect(fit).top() > rect(pane.wire_current).bottom()
            for control in (pane.dataset, pane.component, pane.units, pane.wire_current, pane.view, fit):
                assert pane.contentsRect().contains(rect(control))
                assert control.width() >= control.minimumSizeHint().width()
    finally:
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        application.setFont(original_font)
        if font_id >= 0:
            QFontDatabase.removeApplicationFont(font_id)
