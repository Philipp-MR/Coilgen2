"""Export actions retain source, selection, units and durable completion feedback."""

from pathlib import Path
import threading
import time

import numpy as np
import pytest
import test_workbench_usability as usability
from test_application_cad import _run
from test_gui_export_panel import _options, _record

from PySide6.QtCore import QCoreApplication, QEvent, QThread, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from coilgen.application import DesignService
from coilgen.application import cad
from coilgen.gui.export_panel import (
    ArtifactVerificationError,
    ExportReceiptPanel,
    GeometryExportPanel,
    GeometryExportState,
    export_verified_artifacts,
)
from coilgen.gui.workbench import CoilGenWorkbench


application = usability.application
windows = usability.windows


def _wait(window, attribute):
    deadline = time.monotonic() + 30
    while getattr(window, attribute) is not None and time.monotonic() < deadline:
        QTest.qWait(10)
        time.sleep(0.001)  # Let Python file-verification workers acquire the GIL.
    assert getattr(window, attribute) is None


def _select(window, record, id_="saved"):
    window.runs[id_] = record
    window.show_run(id_)
    window.navigate_workflow("export")
    return window.geometry_export_panel


def test_panel_rebuild_restores_choices_and_receipt_only_for_same_source(application, tmp_path):
    state = GeometryExportState()
    panel = GeometryExportPanel(_record(), _options(step_unit="mm"), state=state)
    panel.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    panel.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    panel.units.setCurrentIndex(1)
    panel.include_project.setChecked(False)
    panel.receipt.set_failed("Destination already exists", directory=tmp_path / "package")
    rebuilt = GeometryExportPanel(_record(), _options(step_unit="mm"), state=state)
    assert rebuilt.selected_names() == (_options()[1].name,)
    assert rebuilt.units.currentData() == "m" and not rebuilt.include_project.isChecked()
    assert "already exists" in rebuilt.receipt.status.text()
    assert rebuilt.export_button.isEnabled()
    other = GeometryExportPanel(_record(), _options())
    assert other.selected_names() == (_options()[0].name,)
    assert other.units.currentData() == "mm" and other.include_project.isChecked()
    assert other.receipt.isHidden()
    for widget in (panel, rebuilt, other):
        widget.deleteLater()


def test_geometry_receipt_preserves_choices_and_reports_actual_committed_units(
    windows, application, tmp_path, monkeypatch
):
    record = _run(tmp_path)
    window = windows()
    panel = _select(window, record)
    for row in range(panel.table.rowCount()):
        panel.table.item(row, 0).setCheckState(Qt.CheckState.Checked)
    panel.include_project.setChecked(False)
    names = panel.selected_names()
    directory = tmp_path / "geometry-package"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(directory), ""))
    panel.export_button.click()
    _wait(window, "_geometry_export_task")
    assert window.geometry_export_panel is panel
    assert panel.selected_names() == names and not panel.include_project.isChecked()
    assert "STL [mm]" in panel.receipt.status.text()
    assert "STEP [m]" in panel.receipt.status.text()
    assert "2 solid files" in panel.receipt.status.text()
    assert str(directory) in panel.receipt.status.text()
    assert panel.receipt.open_button.isVisibleTo(panel)
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    panel.receipt.open_button.click()
    assert Path(opened[0]) == directory
    window.statusBar().clearMessage()
    assert "2 solid files" in panel.receipt.status.text()
    window.select_node("geometry")
    window.navigate_workflow("export")
    assert window.geometry_export_panel.selected_names() == names
    assert "STL [mm]" in window.geometry_export_panel.receipt.status.text()


def test_geometry_export_failure_is_inline_keeps_selection_and_allows_retry(windows, tmp_path, monkeypatch):
    record = _run(tmp_path)
    window = windows()
    panel = _select(window, record)
    for row, option in enumerate(panel.options):
        panel.table.item(row, 0).setCheckState(
            Qt.CheckState.Checked if option.format == "step" else Qt.CheckState.Unchecked
        )
    panel.units.setCurrentIndex(1)
    panel.include_project.setChecked(False)
    names = panel.selected_names()
    directory = tmp_path / "existing"
    directory.mkdir()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(directory), ""))
    panel.export_button.click()
    _wait(window, "_geometry_export_task")
    assert "Export failed" in panel.receipt.status.text()
    assert panel.receipt.status.property("semanticTone") == "error"
    assert panel.export_button.isEnabled() and panel.selected_names() == names
    assert panel.units.currentData() == "m" and not panel.include_project.isChecked()
    assert window.evidence.any_open() is False
    retry = tmp_path / "retry"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(retry), ""))
    panel.export_button.click()
    _wait(window, "_geometry_export_task")
    assert panel.selected_names() == names
    assert "STEP [m]" in panel.receipt.status.text()
    assert panel.receipt.receipt.complete


def test_geometry_completion_while_viewing_other_run_does_not_label_wrong_panel(
    windows, application, tmp_path, monkeypatch
):
    record = _run(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = _run(other_root)
    window = windows()
    panel = _select(window, record)
    panel.include_project.setChecked(False)
    directory = tmp_path / "geometry-package"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(directory), ""))
    started, release = threading.Event(), threading.Event()
    export = cad.export_geometry

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return export(*args, **kwargs)

    monkeypatch.setattr(cad, "export_geometry", delayed)
    panel.export_button.click()
    assert started.wait(2)
    other_panel = _select(window, other, "other")
    assert other_panel.receipt.isHidden()
    release.set()
    _wait(window, "_geometry_export_task")
    assert window.geometry_export_panel is other_panel and other_panel.receipt.isHidden()
    panel = _select(window, record)
    assert str(directory) in panel.receipt.status.text()


def test_geometry_completion_after_navigation_avoids_deleted_panel(windows, tmp_path, monkeypatch):
    record = _run(tmp_path)
    window = windows()
    panel = _select(window, record)
    panel.include_project.setChecked(False)
    directory = tmp_path / "geometry-package"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(directory), ""))
    started, release = threading.Event(), threading.Event()
    export = cad.export_geometry

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return export(*args, **kwargs)

    monkeypatch.setattr(cad, "export_geometry", delayed)
    panel.export_button.click()
    assert started.wait(2)
    window.select_node("geometry")
    QTest.qWait(20)
    release.set()
    _wait(window, "_geometry_export_task")
    window.navigate_workflow("export")
    assert str(directory) in window.geometry_export_panel.receipt.status.text()


def test_artifact_export_uses_background_verification_and_no_array_decompression(
    windows, application, tmp_path, monkeypatch
):
    record = _run(tmp_path)
    window = windows()
    window.runs["saved"] = record
    window.show_run("saved")
    directory = tmp_path / "artifacts"
    directory.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(directory))
    calls = []

    def loader(path, *, array_names=None):
        calls.append((QThread.currentThread() == application.thread(), array_names))
        return DesignService.load_run(path, array_names=array_names)

    window.service.load_run = loader

    def unexpected_array(*_args):
        raise AssertionError("Artifact export decompressed an unrelated scientific array")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", unexpected_array)
    window._export_run_dialog()
    _wait(window, "_artifact_export_task")
    assert calls and all(call == (False, ()) for call in calls)
    receipt = next(
        widget for widget in window.findChildren(ExportReceiptPanel) if widget.receipt.directory == directory
    )
    assert receipt.receipt.complete and "verified artifacts" in receipt.status.text()
    assert "STL" in receipt.status.text() and "NPZ" in receipt.status.text()
    assert "no coordinate conversion" in receipt.status.text()
    window.statusBar().clearMessage()
    assert str(directory) in receipt.status.text()
    for artifact in record.metadata["artifacts"]:
        assert (directory / artifact["name"]).read_bytes() == (record.directory / artifact["name"]).read_bytes()


def test_artifact_invalid_source_does_not_publish_and_retains_source_health(windows, tmp_path):
    record = _run(tmp_path)
    window = windows()
    window.runs["saved"] = record
    window.show_run("saved")
    (record.directory / record.metadata["artifacts"][0]["name"]).write_bytes(b"changed")
    directory = tmp_path / "artifacts"
    with pytest.raises(ArtifactVerificationError):
        window.export_run_artifacts(directory)
    assert not directory.exists()
    assert window._run_verification_error("saved")


def test_artifact_existing_destination_is_never_overwritten_and_publishes_nothing(tmp_path):
    record = _run(tmp_path)
    directory = tmp_path / "artifacts"
    artifact = record.metadata["artifacts"][-1]["name"]
    existing = directory / artifact
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        export_verified_artifacts(record, directory, loader=DesignService.load_run)
    assert existing.read_bytes() == b"keep"
    assert [path for path in directory.rglob("*") if path.is_file()] == [existing]


def test_artifact_publication_failure_rolls_back_own_files_only(tmp_path, monkeypatch):
    import coilgen.gui.export_panel as exports

    record = _run(tmp_path)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    unrelated = directory / "existing-note.txt"
    unrelated.write_text("keep")
    link = exports.os.link
    calls = []

    def failing(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("Publication failed")
        return link(source, target)

    monkeypatch.setattr(exports.os, "link", failing)
    with pytest.raises(OSError, match="Publication failed"):
        export_verified_artifacts(record, directory, loader=DesignService.load_run)
    assert unrelated.read_text() == "keep"
    assert [path for path in directory.rglob("*") if path.is_file()] == [unrelated]


def test_artifact_retry_retains_original_source_after_selection_changes(windows, tmp_path, monkeypatch):
    record = _run(tmp_path)
    window = windows()
    window.runs["saved"] = record
    window.show_run("saved")
    failed = tmp_path / "failed"
    failed.mkdir()
    artifact = record.metadata["artifacts"][0]["name"]
    existing = failed / artifact
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")
    retry = tmp_path / "retry"
    retry.mkdir()
    directories = iter((str(failed), str(retry)))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: next(directories))
    window._export_run_dialog()
    _wait(window, "_artifact_export_task")
    receipt = next(widget for widget in window.findChildren(ExportReceiptPanel) if widget.receipt.directory == failed)
    assert receipt.receipt.error and not window._run_verification_error(record)
    window._selected_run = None
    receipt.retry_button.click()
    _wait(window, "_artifact_export_task")
    assert receipt.receipt.complete and receipt.receipt.directory == retry
    assert (retry / artifact).read_bytes() == (record.directory / artifact).read_bytes()


def test_artifact_completion_after_project_owner_changes_retains_source_receipt(windows, tmp_path, monkeypatch):
    record = _run(tmp_path)
    window = windows()
    window.runs["saved"] = record
    window.show_run("saved")
    directory = tmp_path / "artifacts"
    directory.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(directory))
    started, release = threading.Event(), threading.Event()

    def loader(path, *, array_names=None):
        started.set()
        assert release.wait(5)
        return DesignService.load_run(path, array_names=array_names)

    window.service.load_run = loader
    window._export_run_dialog()
    assert started.wait(2)
    receipt = next(
        widget for widget in window.findChildren(ExportReceiptPanel) if widget.receipt.directory == directory
    )
    window.close_project()
    window.new_project(name="Different design")
    release.set()
    _wait(window, "_artifact_export_task")
    assert window.project.name == "Different design" and window._selected_run is None
    assert receipt.receipt.complete and receipt.receipt.directory == directory
    assert "saved_run" in receipt.window().windowTitle()


def test_artifact_writer_finishes_after_main_window_is_destroyed(application, tmp_path, monkeypatch):
    import shiboken6

    record = _run(tmp_path)
    window = CoilGenWorkbench(
        jobs=usability.Queue(tmp_path / "jobs"), service=usability.Service(), viewport_factory=usability.Preview
    )
    window.timer.stop()
    window.runs["saved"] = record
    window._selected_run = "saved"
    directory = tmp_path / "artifacts"
    directory.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(directory))
    started, release = threading.Event(), threading.Event()

    def loader(path, *, array_names=None):
        started.set()
        assert release.wait(5)
        return DesignService.load_run(path, array_names=array_names)

    window.service.load_run = loader
    window._export_run_dialog()
    assert started.wait(2)
    assert window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(window)
    release.set()
    _wait(window, "_artifact_export_task")
    for artifact in record.metadata["artifacts"]:
        assert (directory / artifact["name"]).is_file()
