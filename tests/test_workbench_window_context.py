"""Integrated scientific controls keep the model and saved-run header honest."""

from copy import deepcopy
from dataclasses import replace
import tempfile

import numpy as np
import pytest

import test_native_workbench as native
import test_workbench_usability as usability

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest

from coilgen.application import ProjectDocument
from coilgen.application.diagnostics import ProblemIssue
from coilgen.application.history import HistoryEntry
from coilgen.application.presets import preset_config
from coilgen.application.service import RunRecord
from coilgen.gui.viewport import ScientificViewport
from coilgen.gui.workbench import CoilGenWorkbench


application = native.application
scientific_viewport = native.scientific_viewport


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    pytest.importorskip("pyvista")
    pyvistaqt = pytest.importorskip("pyvistaqt")
    monkeypatch.setattr(pyvistaqt, "QtInteractor", native.RenderlessPlotter)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("TEMP", str(runtime))
    monkeypatch.setenv("TMP", str(runtime))
    monkeypatch.setattr(tempfile, "tempdir", str(runtime))
    project = ProjectDocument.from_config(preset_config(), name="Initial Gy").rename("Applied Gy")
    instance = CoilGenWorkbench(
        project,
        jobs=usability.Queue(tmp_path / "jobs"),
        service=usability.Service(),
        viewport_factory=lambda parent: ScientificViewport(parent, synchronous_preview=True),
    )
    instance.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    instance.timer.stop()
    yield instance
    instance._draft_dirty = False
    instance.undo_stack.setClean()
    instance.close()
    instance.undo_stack.clear()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _run(window, tmp_path, *, matching=True, missing_original=False, with_review=False):
    points = np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
    target = np.array([[0.0, 0, 1e-6], [0, 0, 2e-6], [0, 0, 3e-6]])
    arrays = {
        "vertices_m": points,
        "faces": np.array([[0, 1, 2]]),
        "stream_function_A": np.array([0.0, 0.5, 1.0]),
        "target_points_m": points,
        "target_field_T": target,
        "sheet_field_T": target * 1.1,
        "contour_field_T": target * 1.4,
        "contour_field_at_stream_current_T": target * 1.2,
        "routed_field_T": target * 1.6,
        "routed_field_at_stream_current_T": target * 1.3,
        "validation_points_m": np.array([[0.0, 0.0, 0], [0.002, 0.003, 0.004]]),
        "validation_field_T": np.array([[0.0, 0, 1e-6], [0, 0, 2e-6]]),
    }
    if missing_original:
        del arrays["contour_field_at_stream_current_T"]
    if with_review:
        arrays["review_conductor_0_points_m"] = points
    metadata = {
        "name": "Saved Gy from yesterday",
        "project_id": window.project.id,
        "project_revision": window.project.revision if matching else window.project.revision - 1,
        "project_fingerprint": window.project.fingerprint if matching else "older-model-and-study",
        "project_model_fingerprint": window.project.model_fingerprint if matching else "older-model",
        "pipeline_stage": "evaluated",
        "job": {"created_at": "2026-10-04T12:00:00Z"},
        "summary": {"scope": "continuous_sheet"},
        "wire_fields": {
            "contour": {
                "available": True,
                "current_A": 4.0,
                "current_refitted": True,
                "current_source": "weighted_fit",
                "geometry_sources": ["oriented_unconnected_contours"],
                "sample_count": 3,
                "at_stream_current": {
                    "available": True,
                    "current_A": 3.0,
                    "current_refitted": False,
                    "current_source": "original_contour_step",
                },
            },
            "routed": {
                "available": True,
                "current_A": 6.0,
                "current_refitted": True,
                "current_source": "weighted_fit",
                "geometry_sources": ["connected_route"],
                "sample_count": 3,
                "at_stream_current": {
                    "available": True,
                    "current_A": 3.0,
                    "current_refitted": False,
                    "current_source": "original_contour_step",
                },
            },
        },
    }
    record = RunRecord(tmp_path / "saved-run", metadata, arrays)
    window.runs["saved"] = record
    window.show_run("saved")
    return record


def _select(window, key):
    combo = window.viewport.dataset
    combo.setCurrentIndex(combo.findData(key))
    return window.view_context.context


def _unchanged(window, project, record, arrays, metadata, undo_count):
    assert window.project is project
    assert window.project.revision == project.revision
    assert window.project.fingerprint == project.fingerprint
    assert window.undo_stack.count() == undo_count
    assert not window.service.submissions
    assert record.metadata == metadata
    for key, values in arrays.items():
        np.testing.assert_array_equal(record.arrays[key], values)
        np.testing.assert_array_equal(window.viewport._arrays[key], values)


@pytest.mark.parametrize("key", ["support", "preview_target"])
def test_applied_preview_header_ignores_selected_older_run(window, tmp_path, key):
    record = _run(window, tmp_path, matching=False)
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata
    context = _select(window, key)
    assert window._selected_run == "saved"
    assert context.source == "model"
    assert context.identity == "Applied model preview"
    assert f"applied revision {project.revision}" in context.viewing_text
    assert "Saved Gy from yesterday" not in context.viewing_text
    if key == "support":
        assert window.viewport._dataset_mesh is window.viewport._support
        assert window.viewport._dataset_mesh.n_points > 3
    else:
        np.testing.assert_array_equal(
            window.viewport._dataset_mesh.points, window.viewport._preview_arrays["target_points_m"]
        )
        assert window.viewport._dataset_mesh.n_points > len(arrays["target_points_m"])
    _unchanged(window, project, record, arrays, metadata, count)


def test_saved_field_header_follows_run_revision_and_exact_original_current_array(window, tmp_path):
    record = _run(window, tmp_path, matching=False)
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata
    window.viewport.units.setCurrentText("T")
    window.viewport.component.setCurrentText("Bz")
    window.viewport.wire_current.setCurrentIndex(0)
    context = _select(window, "contour_field")
    assert context.source == "run"
    assert context.identity == "Earlier model/study revision"
    assert "Saved Gy from yesterday" in context.viewing_text
    assert f"revision {metadata['project_revision']}" in context.viewing_text
    assert metadata["project_revision"] != project.revision
    assert f"revision {project.revision}" in context.editing_text
    assert "date unrecorded" not in context.viewing_text
    assert context.current_A == 4.0
    assert "Saved refitted current 4 A" in context.viewing_text
    np.testing.assert_array_equal(window.viewport._dataset_mesh.active_scalars, arrays["contour_field_T"][:, 2])
    window.viewport.wire_current.setCurrentIndex(1)
    original = window.view_context.context
    assert original is not context
    assert original.current_A == 3.0
    assert "Original contour current 3 A" in original.viewing_text
    assert "weighted_fit" not in original.viewing_tooltip
    np.testing.assert_array_equal(
        window.viewport._dataset_mesh.active_scalars, arrays["contour_field_at_stream_current_T"][:, 2]
    )
    _unchanged(window, project, record, arrays, metadata, count)


def test_routed_residual_and_design_scope_match_actual_saved_field_source(window, tmp_path):
    record = _run(window, tmp_path)
    window.viewport.units.setCurrentText("T")
    window.viewport.component.setCurrentText("Bz")
    window.viewport.wire_current.setCurrentIndex(0)
    context = _select(window, "routed_residual")
    assert context.current_A == 6.0
    assert context.sampling == "Design samples · 3 samples"
    assert "connected_route" in context.viewing_tooltip
    assert "oriented_unconnected_contours" not in context.viewing_tooltip
    expected = record.arrays["routed_field_T"][:, 2] - record.arrays["target_field_T"][:, 2]
    np.testing.assert_array_equal(window.viewport._dataset_mesh.active_scalars, expected)
    assert not window.service.submissions


@pytest.mark.parametrize("key", ["support", "preview_target", "sheet"])
def test_pending_geometry_draft_changes_editing_row_without_relabelling_applied_plot(window, tmp_path, key):
    record = _run(window, tmp_path)
    context = _select(window, key)
    project, count = window.project, window.undo_stack.count()
    support = np.array(window.viewport._support.points, copy=True)
    preview = deepcopy(window.viewport._preview_arrays)
    arrays, metadata = record.arrays, record.metadata
    radius = window._draft_controls["radius_mm"]
    radius.setValue(radius.value() + 13.25)
    pending = window.view_context.context
    assert pending is not context
    assert window._draft_dirty
    assert "changes pending" in pending.editing_text
    assert pending.source == ("run" if key == "sheet" else "model")
    if key == "sheet":
        assert pending.identity == "Applied revision · edits pending"
        assert "edits pending" in pending.viewing_text
    else:
        assert pending.identity == "Applied model preview"
        assert "Apply pending edits" in pending.viewing_tooltip
    np.testing.assert_array_equal(window.viewport._support.points, support)
    for name, values in preview.items():
        np.testing.assert_array_equal(window.viewport._preview_arrays[name], values)
    _unchanged(window, project, record, arrays, metadata, count)


def test_slice_component_and_unit_signals_refresh_context_without_numerical_edits(window, tmp_path):
    record = _run(window, tmp_path)
    _select(window, "sheet")
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata
    previous = window.view_context.context
    window.viewport.slice_axis.setCurrentText("x")
    sliced = window.view_context.context
    assert sliced is not previous
    assert window.viewport._dataset_mesh.n_points == 2
    assert "Displayed sample selection: x ≈ 0 m · 2 samples; no interpolation" in sliced.viewing_tooltip
    window.viewport.slice_position.setValue(100)
    moved = window.view_context.context
    assert moved is not sliced
    assert window.viewport._dataset_mesh.n_points == 1
    assert "1 samples; no interpolation" in moved.viewing_tooltip
    window.viewport.component.setCurrentText("By")
    component = window.view_context.context
    assert component is not moved
    window.viewport.units.setCurrentText("nT")
    units = window.view_context.context
    assert units is not component
    assert units.source == "run"
    assert units.sampling == "Design samples · 3 samples"
    _unchanged(window, project, record, arrays, metadata, count)


def test_independent_validation_header_uses_distinct_saved_grid_and_no_drive_current(window, tmp_path):
    record = _run(window, tmp_path)
    context = _select(window, "validation")
    assert context.available is True
    assert context.sampling == "Independent samples · 2 samples"
    assert context.current_A is None
    np.testing.assert_array_equal(window.viewport._dataset_mesh.points, record.arrays["validation_points_m"])
    assert "Solved sheet coefficients [A]" in context.viewing_text


def test_missing_original_array_does_not_relabel_or_render_refitted_map(window, tmp_path):
    record = _run(window, tmp_path, missing_original=True)
    viewport = window.viewport
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata
    errors = []
    viewport.error.connect(errors.append)
    # The routed original-current map exists, so this is a valid explicit
    # global policy. The unavailable contour variant must not use its refit.
    window.viewport.wire_current.setCurrentIndex(1)
    before = deepcopy(viewport.view_state())
    context = _select(window, "contour_field")
    assert viewport.view_state() == before
    assert viewport.dataset.currentData() == "stream_function"
    assert context.available is True and context.current_A is None
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, arrays["stream_function_A"])
    contour = viewport.dataset.model().item(viewport.dataset.findData("contour_field"))
    assert not contour.isEnabled()
    assert "contour_field_at_stream_current_T" in contour.toolTip()
    assert errors and "Plot selection unavailable" in errors[-1]
    # An unavailable policy request on an already displayed wire map also
    # leaves the explicit selection, field values, and drive label intact.
    viewport.wire_current.setCurrentIndex(0)
    stored = _select(window, "contour_field")
    before = deepcopy(viewport.view_state())
    scalars = viewport._dataset_mesh.active_scalars.copy()
    viewport.wire_current.setCurrentIndex(1)
    assert viewport.view_state() == before
    assert not viewport.wire_current.model().item(1).isEnabled()
    assert window.view_context.context.current_A == stored.current_A == 4.0
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, scalars)
    assert "Current selection unavailable" in errors[-1]
    _unchanged(window, project, record, arrays, metadata, count)


def test_model_button_is_view_only_and_retains_unapplied_draft_and_selected_run(window, tmp_path, application):
    record = _run(window, tmp_path)
    _select(window, "routed_field")
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata
    radius = window._draft_controls["radius_mm"]
    radius.setValue(radius.value() + 5)
    value = radius.value()
    assert window.view_context.model_button.isEnabled()
    QTest.mouseClick(window.view_context.model_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert window.viewport.dataset.currentData() == "preview_target"
    assert window.view_context.context.source == "model"
    assert window._selected_run == "saved"
    assert window._draft_dirty
    assert radius.value() == value
    _unchanged(window, project, record, arrays, metadata, count)


@pytest.mark.parametrize("open_method", ["explicit", "compute"])
def test_jobs_window_exposes_original_table_with_its_filter_and_progress_controls(window, application, open_method):
    original = window.jobs_table
    if open_method == "compute":
        window.compute_button.click()
        assert len(window.service.submissions) == 1
        assert not window.evidence.window_for(window.jobs_table).isVisible()
        window._open_evidence(window.jobs_table)
    else:
        window._open_evidence(window.jobs_table)
    application.processEvents()
    panel = window.evidence.window_for(window.jobs_table)
    assert panel.isVisible()
    assert window.jobs_filter.isVisibleTo(panel)
    assert window.execution_label.isVisibleTo(panel)
    assert window.jobs_table.isVisibleTo(panel)
    assert not window.jobs_table.isHidden()
    panel.close()
    window._open_evidence(window.jobs_table)
    assert window.jobs_table is original
    assert window.jobs_table.isVisibleTo(panel)


def test_history_alias_rename_refreshes_header_immediately_without_altering_saved_evidence(
    window, tmp_path, monkeypatch
):
    record = _run(window, tmp_path)
    entry = HistoryEntry("saved", window.project.id, None, "Initial alias", record.directory, "saved-digest", record)
    restored = [entry]
    window._history_entries = (entry,)
    window._history_ids = {"saved": "saved"}
    window._refresh_history_panel()
    _select(window, "sheet")
    assert "Viewing: Initial alias" in window.view_context.context.viewing_text
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata

    def rename(id_, alias):
        assert id_ == "saved"
        restored[0] = replace(restored[0], alias=alias)

    monkeypatch.setattr(window.history, "rename_run", rename)
    monkeypatch.setattr(window.history, "restore", lambda *_args, **_kwargs: tuple(restored))
    window._rename_history("saved", "Wider copper variant")
    assert "Viewing: Wider copper variant" in window.view_context.context.viewing_text
    assert "Initial alias" not in window.view_context.context.viewing_text
    _unchanged(window, project, record, arrays, metadata, count)


def _witness():
    return ProblemIssue(
        "clearance",
        "Recorded conductor separation",
        "Saved interval witness",
        points_m=((0.0, 0, 0), (0.01, 0, 0)),
        segments=((0, 1),),
        body_id="body001",
        dataset="review_conductor",
        location_description="Recorded conductor segment [0, 1]",
    )


def test_locate_recorded_witness_refreshes_actual_saved_source_after_blocked_dataset_change(window, tmp_path):
    record = _run(window, tmp_path, with_review=True)
    _select(window, "support")
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata
    window._locate_problem(_witness())
    assert window.viewport.dataset.currentData() == "review_conductor"
    assert window.view_context.context.source == "run"
    assert window.view_context.context.dataset_key == "review_conductor"
    assert "Saved Gy from yesterday" in window.view_context.context.viewing_text
    _unchanged(window, project, record, arrays, metadata, count)


def test_clear_highlight_restores_model_preview_context_after_blocked_dataset_change(window, tmp_path):
    _run(window, tmp_path, with_review=True)
    _select(window, "support")
    window._locate_problem(_witness())
    # Set up a matching visible identity to isolate the clear operation from the
    # locate notification regression covered by the preceding test.
    window._refresh_view_context()
    assert window.view_context.context.source == "run"
    window.viewport.clear_highlight.click()
    assert window.viewport.dataset.currentData() == "support"
    assert window.view_context.context.source == "model"
    assert window.view_context.context.dataset_key == "support"
    assert not window.service.submissions


def test_blocked_view_state_restore_refreshes_actual_display_source(window, tmp_path):
    record = _run(window, tmp_path)
    _select(window, "support")
    model_state = window.viewport.view_state()
    _select(window, "sheet")
    project, count = window.project, window.undo_stack.count()
    arrays, metadata = record.arrays, record.metadata
    assert window.viewport.restore_view_state(model_state)
    assert window.viewport.dataset.currentData() == "support"
    assert window.view_context.context.source == "model"
    assert window.view_context.context.dataset_key == "support"
    _unchanged(window, project, record, arrays, metadata, count)


@pytest.mark.parametrize("condition", ["available", "missing", "nonfinite"])
def test_viewport_context_notification_follows_finalized_caption_and_dataset(scientific_viewport, condition):
    viewport = scientific_viewport
    events = []
    viewport.context_changed.connect(
        lambda: events.append((viewport.dataset.currentData(), viewport.caption.text(), viewport._dataset_mesh))
    )
    arrays = {
        "target_points_m": np.array([[0.0, 0, 0]]),
        "target_field_T": np.array([[0.0, 0, 1e-6]]),
    }
    if condition == "missing":
        arrays = {}
    elif condition == "nonfinite":
        arrays["target_field_T"][0, 2] = np.nan
    viewport.display_run(arrays)
    assert len(events) == 1
    key, caption, mesh = events[0]
    assert caption == viewport.caption.text()
    if condition == "available":
        assert key == "target"
        assert mesh is not None and mesh.n_points == 1
        assert "1 saved samples; no interpolation" in caption
    else:
        assert key == "support"
        assert mesh is None
        assert not viewport.dataset.model().item(viewport.dataset.findData("target")).isEnabled()
        assert not viewport.dataset.model().item(viewport.dataset.findData("support")).isEnabled()
        assert "Saved run has no available plot data" in caption
        if condition == "nonfinite":
            assert "finite real numbers" in caption
