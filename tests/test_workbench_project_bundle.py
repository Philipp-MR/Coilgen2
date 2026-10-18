"""GUI Save/Open moves completed engineering studies using only one JSON."""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
from coilgen.application import DesignService, ProjectAssembly, ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.workbench import CoilGenWorkbench

import test_native_workbench as native

application = native.application


def _window(project, jobs_directory):
    window = CoilGenWorkbench(
        project=project,
        jobs=native.FakeJobs(jobs_directory),
        service=DesignService(),
        viewport_factory=native.FakeViewport,
    )
    window.timer.stop()
    return window


def _close(window):
    window.undo_stack.setClean()
    window.close()


@pytest.mark.parametrize("collection", (False, True))
@pytest.mark.parametrize("duplicate", (False, True))
def test_gui_open_parses_json_once_and_retains_duplicate_key_rejection(
    application, tmp_path, monkeypatch, collection, duplicate
):
    project = ProjectDocument.from_config(preset_config(), name="Read once")
    document = ProjectAssembly.new("Collection", (project,)) if collection else project
    path = document.save(tmp_path / "single-read.coilgen.json")
    if duplicate:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"name":', '"name": "Duplicate", "name":', 1), encoding="utf-8")
    reads = []
    read_text = Path.read_text

    def read(source, *args, **kwargs):
        if source.resolve() == path.resolve():
            reads.append(source)
        return read_text(source, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    window = _window(project.duplicate("Previous editor"), tmp_path / "jobs")
    previous = window.project
    try:
        if duplicate:
            with pytest.raises(ValueError, match="Duplicate project JSON property: name"):
                window.open_project(path)
            assert window.project is previous
        else:
            restored = window.open_project(path)
            assert restored.fingerprint == document.fingerprint
        assert len(reads) == 1
    finally:
        _close(window)


@pytest.mark.parametrize("failure_stage", ("archive", "history"))
def test_embedded_restore_failure_keeps_one_clear_diagnostic_and_verified_fallback(
    application, tmp_path, monkeypatch, failure_stage
):
    from coilgen.application import RunArchive
    from test_application_project_bundle import _prepared_document

    project, _record, archive = _prepared_document(tmp_path)
    view = {"source": "run", "dataset": "fabricated", "model_node": "results"}
    project = project.with_results((archive,), session={"selected_run_id": archive.id, "view_state": view})
    path = project.save(tmp_path / "saved.coilgen.json")
    window = _window(project.duplicate("Empty editor"), tmp_path / "new-runtime" / "jobs")
    restored = []
    monkeypatch.setattr(
        window.viewport,
        "restore_view_state",
        lambda state: restored.append((state, window._selected_run)) or True,
        raising=False,
    )

    def fail(*_args, **_kwargs):
        raise OSError("Fixture cache inaccessible")

    if failure_stage == "archive":
        monkeypatch.setattr(RunArchive, "materialize", fail)
    else:
        monkeypatch.setattr(window.history, "restore", fail)
    try:
        window.open_project(path)
        if failure_stage == "archive":
            assert window._selected_run is None
            assert all(state.get("source") != "run" for state, _id in restored)
            assert "recorded run could not be restored" in window.messages.toPlainText()
        else:
            assert window._selected_run in window.runs
            assert any(state == view and id_ in window.runs for state, id_ in restored)
        assert "cache inaccessible" in window.messages.toPlainText()
    finally:
        _close(window)


@pytest.mark.parametrize("extension", (".coilgen.json", ".coilgen"))
def test_gui_save_json_reopens_completed_run_after_original_jobs_removed(application, tmp_path, extension):
    original = tmp_path / "original"
    config = preset_config()
    config.progress_flag = False
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "prepared"})
    directory = original / "jobs" / "prepared"
    DesignService.run(project, directory, threads=1)
    baseline = DesignService.load_run(directory)
    window = _window(project, original / "jobs")
    try:
        window.load_run(directory)
        window.navigate_workflow("review")
        entry = window._history_entries[0]
        window._rename_history(entry.id, "Prepared reference")
        path = window.save_project(original / f"project{extension}")
        assert not path.with_suffix(".assets").exists()
        assert len(ProjectDocument.load(path).saved_runs) == 1
        assert ProjectDocument.load(path).saved_runs[0].alias == "Prepared reference"
        moved = tmp_path / "portable" / path.name
        moved.parent.mkdir()
        shutil.copyfile(path, moved)
    finally:
        _close(window)
    shutil.rmtree(original)

    reopened = _window(ProjectDocument.load(moved), tmp_path / "new-runtime" / "jobs")
    try:
        assert reopened._selected_run in reopened.runs
        record = reopened.runs[reopened._selected_run]
        assert record.metadata == baseline.metadata
        assert set(record.arrays) == set(baseline.arrays)
        for key, expected in baseline.arrays.items():
            np.testing.assert_array_equal(record.arrays[key], expected)
            assert record.arrays[key].dtype == expected.dtype
        assert reopened.tree.currentItem() is reopened.tree_items["results"]
        assert reopened._history_entries[0].alias == "Prepared reference"
        assert reopened.view_results_button.isEnabled()
        second = reopened.save_project(tmp_path / "second" / "project.coilgen.json")
        assert list(second.parent.iterdir()) == [second]
        assert len(ProjectDocument.load(second).saved_runs) == 1
    finally:
        _close(reopened)


@pytest.mark.parametrize("extension", (".json", ".coilgen"))
def test_gui_collection_save_embeds_completed_runs_for_each_channel(application, tmp_path, extension):
    config = preset_config()
    config.progress_flag = False
    first = ProjectDocument.from_config(config, name="First").update_study({"pipeline_stage": "prepared"})
    second = first.duplicate("Second")
    assembly = ProjectAssembly.new("Independent pair", (first, second))
    root = tmp_path / "original"
    window = _window(assembly, root / "jobs")
    try:
        for channel in assembly.channels:
            directory = root / "jobs" / channel.id
            DesignService.run(channel.project, directory, threads=1)
            window.load_run(directory, select=False)
        path = window.save_project(root / f"pair{extension}")
        moved = tmp_path / "moved" / path.name
        moved.parent.mkdir()
        shutil.copyfile(path, moved)
    finally:
        _close(window)
    shutil.rmtree(root)
    restored = ProjectAssembly.load(moved)
    assert all(len(channel.project.saved_runs) == 1 for channel in restored.channels)
    reopened = _window(restored, tmp_path / "new-runtime" / "jobs")
    try:
        for channel in restored.channels:
            reopened.channel_selector.setCurrentIndex(reopened.channel_selector.findData(channel.id))
            owned = [
                record for record in reopened.runs.values() if record.metadata.get("project_id") == channel.project.id
            ]
            assert owned
            assert len({record.directory for record in owned}) == 1
        assert list(moved.parent.iterdir()) == [moved]
    finally:
        _close(reopened)


def test_gui_new_independent_turn_demo_selects_strict_complete_review(application, tmp_path):
    window = _window(ProjectDocument.from_config(preset_config()), tmp_path / "jobs")
    try:
        window.new_project("independent_turns")
        assert window.project.name == "Independent-turn sweep demo"
        assert window.stage_selector.currentData() == "evaluated"
        assert window.fabricate
        assert window.project.config.skip_postprocessing
        assert window.project.config.contour_only_sweep_mode
        assert window.project.config.field_shape_function == "1e-4"
        assert window.project.config.target_gradient_normalization == "none"
    finally:
        _close(window)


def test_gui_compact_open_controllers_and_save_as_filter_are_accessible(application, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    first = ProjectDocument.from_config(preset_config(), name="First")
    other = first.duplicate("Other")
    single = first.save(tmp_path / "single.coilgen")
    assembly = ProjectAssembly.new("Pair", (first, other))
    collection = assembly.save(tmp_path / "pair.coilgen")
    window = _window(other, tmp_path / "jobs")
    try:
        opened = window.open_project(single)
        assert opened.id == first.id and window.project.id == first.id
        opened = window.open_project(collection)
        assert opened.id == assembly.id and window.assembly.fingerprint == assembly.fingerprint
        assert window._collection_mode
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *_args: (str(tmp_path / "selected.coilgen.json"), "Compact CoilGen bundle (*.coilgen)"),
        )
        assert window._save_dialog(save_as=True)
        assert window.project_path.name == "selected.coilgen"
        assert ProjectAssembly.load(window.project_path).fingerprint == assembly.fingerprint
        script = window.export_assembly_python(tmp_path / "pair.py", compact=True)
        assert ProjectAssembly.load(script.with_suffix(".coilgen")).fingerprint == assembly.fingerprint
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
    finally:
        _close(window)


@pytest.mark.parametrize("method", ("export_python", "export_assembly_python"))
@pytest.mark.parametrize("compact", (False, True))
def test_inputs_only_export_never_collects_completed_evidence(application, tmp_path, monkeypatch, method, compact):
    from coilgen.application import RunArchive
    from test_application_project_bundle import _prepared_document

    project, _record, archive = _prepared_document(tmp_path)
    project = project.with_results((archive,), session={"selected_run_id": archive.id})
    assembly = ProjectAssembly.new("Pair", (project, project.duplicate("Second")))
    window = _window(assembly, tmp_path / "jobs")
    try:

        def forbidden(*_args, **_kwargs):
            pytest.fail("Inputs-only export must not restore or capture completed evidence")

        monkeypatch.setattr(window, "_portable_assembly", forbidden)
        monkeypatch.setattr(window.history, "restore", forbidden)
        monkeypatch.setattr(RunArchive, "capture", forbidden)
        path = getattr(window, method)(tmp_path / "inputs.py", compact=compact)
        suffix = ".coilgen" if compact else ".coilgen.json" if method == "export_python" else ".coilgen-assembly.json"
        if method == "export_python":
            loaded = ProjectDocument.load(path.with_suffix(suffix))
            assert loaded.fingerprint == project.fingerprint and loaded._assets == project._assets
            assert loaded.saved_runs == () and loaded.session == {}
        else:
            loaded = ProjectAssembly.load(path.with_suffix(suffix))
            assert loaded.fingerprint == assembly.fingerprint
            assert all(
                channel.project.saved_runs == () and channel.project.session == {} for channel in loaded.channels
            )
        assert window.assembly.channel(window._active_channel_id).project.saved_runs == (archive,)
        assert not window._foreground_work and window.computation_indicator.isHidden()
    finally:
        _close(window)


@pytest.mark.parametrize("method", ("export_python", "export_assembly_python"))
def test_explicit_evidence_export_collects_and_retains_history(application, tmp_path, method):
    from test_application_project_bundle import _prepared_document

    project, record, archive = _prepared_document(tmp_path)
    window = _window(project, tmp_path / "jobs")
    try:
        window.load_run(record.directory)
        path = getattr(window, method)(tmp_path / "full.py", compact=True, include_results=True)
        loaded = (
            ProjectDocument.load(path.with_suffix(".coilgen"))
            if method == "export_python"
            else ProjectAssembly.load(path.with_suffix(".coilgen")).channels[0].project
        )
        assert len(loaded.saved_runs) == 1
        assert loaded.saved_runs[0].id == archive.id and loaded.saved_runs[0]._files == archive._files
        assert loaded.saved_runs[0].alias == window._history_entries[0].alias
        assert loaded.saved_runs[0].project_id == project.id
    finally:
        _close(window)
