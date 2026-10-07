"""GUI Save/Open moves completed engineering studies using only one JSON."""

import shutil

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


def test_gui_save_json_reopens_completed_run_after_original_jobs_removed(application, tmp_path):
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
        path = window.save_project(original / "project.coilgen.json")
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


def test_gui_collection_save_embeds_completed_runs_for_each_channel(application, tmp_path):
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
        path = window.save_project(root / "pair.json")
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
