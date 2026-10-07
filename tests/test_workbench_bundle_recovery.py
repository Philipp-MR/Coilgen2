"""Portable GUI studies survive unavailable history and preserve visible failures."""

import shutil

import numpy as np
import pytest

import test_workbench_usability as usability
from test_application_backend import _config

from PySide6.QtCore import QCoreApplication, QEvent, Qt

from coilgen.application import DesignService, ProjectAssembly, ProjectDocument
from coilgen.application.bundles import RunArchive
from coilgen.gui.workbench import CoilGenWorkbench


application = usability.application


@pytest.fixture
def windows(application, tmp_path):
    created = []

    def make(project, directory=None):
        queue = usability.Queue(directory or tmp_path / "jobs")
        window = CoilGenWorkbench(project, jobs=queue, service=DesignService(), viewport_factory=usability.Preview)
        window.timer.stop()
        created.append(window)
        return window

    yield make
    for window in created:
        window._draft_dirty = False
        window.undo_stack.setClean()
        window.close()
        window.undo_stack.clear()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def prepared(tmp_path):
    project = ProjectDocument.from_config(_config()).update_study({"pipeline_stage": "prepared"})
    directory = tmp_path / "original-runtime"
    DesignService.run(project, directory, threads=1)
    record = DesignService.load_run(directory)
    archive = RunArchive.capture(record, alias="Engineering baseline")
    return project, record, archive


def _copy_json(path, destination):
    destination.parent.mkdir(parents=True)
    shutil.copyfile(path, destination)
    assert list(destination.parent.iterdir()) == [destination]
    return destination


def test_save_again_uses_embedded_evidence_after_original_runtime_disappears(windows, prepared, tmp_path):
    project, record, _ = prepared
    window = windows(project)
    window.load_run(record.directory)
    window._rename_history(window._history_entries[0].id, "Engineering baseline")
    first = window.save_project(tmp_path / "first" / "project.json")
    original = ProjectDocument.load(first).saved_runs[0]
    assert record.directory.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(record.directory)

    second = window.save_project(tmp_path / "second" / "project.json")

    assert list(second.parent.iterdir()) == [second]
    saved = ProjectDocument.load(second)
    assert len(saved.saved_runs) == 1
    assert saved.saved_runs[0].id == original.id
    assert saved.saved_runs[0].alias == "Engineering baseline"
    restored = saved.saved_runs[0].materialize(tmp_path / "verified-cache")
    assert restored.metadata == record.metadata
    for name, data in record.arrays.items():
        np.testing.assert_array_equal(restored.arrays[name], data)


def test_save_without_optional_history_preserves_custom_alias(windows, prepared, tmp_path):
    project, _, archive = prepared
    project = project.with_results((archive,), session={"selected_run_id": archive.id})
    window = windows(project)
    assert window._selected_run in window.runs
    window.history = None

    path = window.save_project(tmp_path / "portable" / "project.json")

    saved = ProjectDocument.load(path)
    assert len(saved.saved_runs) == 1
    assert saved.saved_runs[0].id == archive.id
    assert saved.saved_runs[0].alias == archive.alias
    assert saved.session["selected_run_id"] == archive.id


def test_corrupt_optional_history_does_not_block_saving_owned_evidence(windows, prepared, tmp_path):
    project, record, archive = prepared
    project = project.with_results((archive,), session={"selected_run_id": archive.id})
    window = windows(project)
    window.history.path.write_text("{malformed-history", encoding="utf-8")

    path = window.save_project(tmp_path / "portable" / "project.json")

    saved = ProjectDocument.load(path)
    assert len(saved.saved_runs) == 1
    assert saved.saved_runs[0].id == archive.id
    assert saved.saved_runs[0].alias == archive.alias
    assert "history cache unavailable" in window.messages.toPlainText().lower()
    restored = saved.saved_runs[0].materialize(tmp_path / "verification")
    assert restored.metadata == record.metadata
    moved = _copy_json(path, tmp_path / "moved" / "project.json")
    reopened = windows(ProjectDocument.load(moved), tmp_path / "new-runtime" / "jobs")
    assert reopened._selected_run in reopened.runs
    assert reopened.runs[reopened._selected_run].metadata == record.metadata


def test_no_results_project_restores_view_and_page_using_only_json(windows, tmp_path):
    project = ProjectDocument.from_config(_config())
    window = windows(project)
    window.select_node("optimization")
    view = {"dataset": "target", "units": "µT", "camera": {"position": [1.0, 2.0, 3.0], "viewup": [0.0, 0.0, 1.0]}}
    window.viewport.state = view
    window.evidence.setCurrentWidget(window.recompute)
    path = window.save_project(tmp_path / "original" / "project.json")
    moved = _copy_json(path, tmp_path / "moved" / "project.json")

    reopened = windows(ProjectDocument.load(moved), tmp_path / "fresh-runtime" / "jobs")

    assert reopened.project.saved_runs == ()
    assert reopened.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "optimization"
    assert reopened.viewport.state["dataset"] == view["dataset"]
    assert reopened.viewport.state["units"] == view["units"]
    assert reopened.viewport.state["camera"] == view["camera"]
    assert reopened.evidence.currentWidget() is reopened.recompute
    assert reopened._selected_run is None


def test_collection_restores_active_channel_and_its_settings_and_camera(windows, tmp_path):
    first = ProjectDocument.from_config(_config(), name="Gx")
    second = first.duplicate("Gy").update_config({"levels": first.config.levels + 1})
    assembly = ProjectAssembly.new("Independent axes", (first, second))
    window = windows(assembly)
    second_id = assembly.channels[1].id
    window.channel_selector.setCurrentIndex(window.channel_selector.findData(second_id))
    window.select_node("winding")
    window.viewport.state = {"dataset": "target", "camera": {"position": [0.25, 0.5, 0.75]}}
    path = window.save_project(tmp_path / "original" / "channels.json")
    moved = _copy_json(path, tmp_path / "moved" / "channels.json")
    saved = ProjectAssembly.load(moved)
    assert all(channel.project.session["active_channel_id"] == second_id for channel in saved.channels)

    reopened = windows(saved, tmp_path / "fresh-runtime" / "jobs")

    assert reopened._active_channel_id == second_id
    assert reopened.channel_selector.currentData() == second_id
    assert reopened.project.id == second.id
    assert reopened.project.config.levels == second.config.levels
    assert reopened.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "winding"
    assert reopened.viewport.state["camera"] == {"position": [0.25, 0.5, 0.75]}
    assert reopened.assembly.channels[0].project.id == first.id


def test_failed_embedded_cache_retains_unavailable_row_after_other_run_is_selected(windows, prepared, tmp_path):
    project, _, failed_archive = prepared
    second = project.update_study({"validation_resolution": 14})
    directory = tmp_path / "second-runtime"
    DesignService.run(second, directory, threads=1)
    good_archive = RunArchive.capture(DesignService.load_run(directory), alias="Independent validation baseline")
    runtime_root = tmp_path / "fresh-runtime"
    failed_cache = failed_archive.materialize(runtime_root / "embedded-runs").directory
    (failed_cache / "scientific_result.npz").write_bytes(b"corrupt cached array archive")
    project = project.with_results(
        (failed_archive, good_archive),
        session={"selected_run_id": good_archive.id, "view_state": {"model_node": "results"}},
    )

    window = windows(project, runtime_root / "jobs")

    assert len(window._history_entries) == 2
    failed = next(entry for entry in window._history_entries if entry.alias == failed_archive.alias)
    verified = next(entry for entry in window._history_entries if entry.alias == good_archive.alias)
    assert not failed.verified
    assert verified.verified
    assert "could not be restored" in failed.diagnostic
    assert "integrity" in failed.diagnostic
    assert failed_archive.alias in window.history_panel.diagnostics.text()
    assert not window.history_panel.diagnostics.isHidden()
    assert failed_archive.alias in window.messages.toPlainText()
    assert window._selected_run in window.runs
    assert window.runs[window._selected_run].metadata == good_archive.metadata

    window._history_selected(verified)

    assert len(window.history_panel.entries) == 2
    assert failed_archive.alias in window.history_panel.diagnostics.text()
    assert "integrity" in window.history_panel.diagnostics.text()
    table = window.history_panel.runs_table
    failed_row = next(
        row for row in range(table.rowCount()) if table.item(row, 0).data(Qt.ItemDataRole.UserRole) == failed.id
    )
    assert table.item(failed_row, 6).text() == "Unavailable"
    assert table.item(failed_row, 3).text() == "Unverified"
    assert len(project.saved_runs) == 2
