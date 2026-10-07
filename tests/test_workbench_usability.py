"""Draft, stage, verified reuse, and durable history integration regressions."""

from pathlib import Path
from types import SimpleNamespace
import os
from uuid import uuid4

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Signal, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QDockWidget, QStyle, QTableWidget, QWidget

from coilgen.application import DesignService, ProjectAssembly, ProjectDocument
from coilgen.application.presets import EXAMPLE_TITLES, preset_config, preset_study
from coilgen.application.service import RunRecord
from coilgen.gui.workbench import CoilGenWorkbench


class Preview(QWidget):
    selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.displayed = None
        self.state = {"dataset": "support"}
        self.clear_count = 0

    def render_project(self, project):
        self.project = project

    def display_run(self, arrays, metadata=None):
        self.displayed = metadata

    def clear_run(self):
        self.displayed = None
        self.clear_count += 1

    def view_state(self):
        return dict(self.state)

    def restore_view_state(self, state):
        self.state = dict(state)


class Queue:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.policy = SimpleNamespace(threads_per_worker=1)
        self.states = []

    def list(self):
        return [dict(state) for state in self.states]


class Service:
    def __init__(self):
        self.submissions = []

    def submit(self, project, jobs, **controls):
        self.submissions.append((project, controls))
        state = {
            "id": uuid4().hex,
            "name": project.name,
            "status": "queued",
            "pipeline_stage": controls["pipeline_stage"],
            "project_id": project.id,
            "project_fingerprint": project.fingerprint,
            "project_model_fingerprint": project.model_fingerprint,
            "project_revision": project.revision,
        }
        jobs.states.append(state)
        return dict(state)

    load_run = staticmethod(DesignService.load_run)


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["workbench-usability-test"])


@pytest.fixture
def windows(application, tmp_path):
    created = []

    def make(project=None, directory=None):
        if project is None:
            project = ProjectDocument.from_config(preset_config(), name=EXAMPLE_TITLES["cylinder"]).update_study(
                preset_study()
            )
        queue = Queue(directory or tmp_path / "jobs")
        instance = CoilGenWorkbench(project, jobs=queue, service=Service(), viewport_factory=Preview)
        instance.timer.stop()
        created.append(instance)
        return instance

    yield make
    for instance in created:
        instance._draft_dirty = False
        instance.undo_stack.setClean()
        instance.close()
        instance.undo_stack.clear()
        instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def prepared_run(tmp_path):
    config = preset_config()
    config.cylinder_mesh_parameter_list = (0.3, 0.16, 12, 6, 1, 0, 0, 0)
    config.target_region_resolution = 10
    config.progress_flag = False
    project = ProjectDocument.from_config(config, name="Prepared support").update_study({"pipeline_stage": "prepared"})
    directory = tmp_path / "run_prepared"
    DesignService.run(project, directory, threads=1)
    return project, DesignService.load_run(directory)


def test_pending_changes_and_revert_leave_model_and_history_unchanged(windows):
    window = windows()
    original, count = window.project, window.undo_stack.count()
    radius = window._draft_controls["radius_mm"]
    radius.setValue(174)
    assert not window.pending_panel.isHidden()
    assert window._draft_dirty
    assert window.project is original
    window.revert_draft()
    assert window.project is original
    assert window.undo_stack.count() == count
    initial_radius_mm = original.config.cylinder_mesh_parameter_list[1] * 1000
    assert window._draft_controls["radius_mm"].value() == pytest.approx(initial_radius_mm)
    assert window.pending_panel.isHidden()
    assert not window._draft_dirty
    window._draft_controls["radius_mm"].setValue(170)
    window._draft_controls["radius_mm"].setValue(initial_radius_mm)
    assert window.pending_panel.isHidden()
    assert not window._changed_controls
    assert window.apply_settings()
    assert window.project is original


def test_invalid_geometry_draft_has_inline_errors_survives_navigation_and_can_revert(windows):
    config = preset_config()
    config.coil_mesh_file = "create conical frustum mesh"
    config.surface_is_cylinder_flag = False
    window = windows(ProjectDocument.from_config(config))
    original, count = window.project, window.undo_stack.count()
    window._draft_controls["rotation_axis_x"].setValue(0)
    window._draft_controls["rotation_angle_deg"].setValue(45)
    assert not window.apply_settings()
    assert window.project is original
    assert window.undo_stack.count() == count
    assert window._draft_dirty
    assert not window.draft_error_label.isHidden()
    assert "rotation" in window.draft_error_label.text().lower()
    assert any(not label.isHidden() for label in window._field_errors.values())
    window.select_node("target")
    assert window.tree.currentItem() is window.tree_items["geometry"]
    assert window._draft_controls["rotation_axis_x"].value() == 0
    assert window._draft_controls["rotation_angle_deg"].value() == 45
    assert window.project is original
    window.revert_draft()
    assert window.draft_error_label.isHidden()
    assert not window._draft_dirty
    assert window.project is original


def test_valid_apply_clears_pending_state_and_creates_one_revision(windows):
    window = windows()
    original = window.project
    window._draft_controls["radius_mm"].setValue(172)
    assert window.apply_settings()
    assert window.project.revision == original.revision + 1
    assert window.pending_panel.isHidden()
    assert not window._changed_controls
    assert window.undo_stack.count() == 1


def _set_stage(window, stage):
    window.stage_selector.setCurrentIndex(window.stage_selector.findData(stage))
    assert window.project.study["pipeline_stage"] == stage


def test_partial_stages_disable_fabrication_and_same_project_restores_preference(windows):
    project = ProjectDocument.from_config(preset_config()).update_study(
        {"pipeline_stage": "evaluated", "fabricate": True}
    )
    window = windows(project)
    _set_stage(window, "sheet")
    assert not window.fabricate
    window.select_node("validation")
    assert not window._draft_controls["fabricate"].isEnabled()
    window.compute()
    assert window.service.submissions[-1][1]["fabricate"] is False
    _set_stage(window, "evaluated")
    assert window.fabricate
    assert window._draft_controls["fabricate"].isEnabled()
    window.compute()
    assert window.service.submissions[-1][1]["fabricate"] is True


def test_fabrication_preference_does_not_leak_between_projects(windows):
    strict = ProjectDocument.from_config(preset_config(), name="Strict evaluated").update_study({"fabricate": True})
    partial = ProjectDocument.from_config(preset_config(), name="Partial draft").update_study(
        {"pipeline_stage": "sheet"}
    )
    window = windows(strict)
    assert window.fabricate
    window._set_project(partial)
    _set_stage(window, "evaluated")
    assert not window.fabricate
    assert window.project.study["fabricate"] is False
    window._set_project(strict)
    assert window.fabricate


def test_fabrication_preference_does_not_leak_between_independent_channels(windows):
    strict = ProjectDocument.from_config(preset_config(), name="Strict channel").update_study({"fabricate": True})
    partial = ProjectDocument.from_config(preset_config(), name="Partial channel").update_study(
        {"pipeline_stage": "sheet"}
    )
    assembly = ProjectAssembly.new("Independent studies", (strict, partial))
    window = windows(assembly)
    window.channel_selector.setCurrentIndex(1)
    _set_stage(window, "evaluated")
    assert not window.fabricate
    window.channel_selector.setCurrentIndex(0)
    assert window.fabricate
    window.channel_selector.setCurrentIndex(1)
    assert not window.fabricate


def test_compute_reuses_verified_compatible_boundary_but_not_a_stale_model(windows, prepared_run):
    project, record = prepared_run
    window = windows(project)
    window.load_run(record.directory)
    assert window.compute_through("sheet")
    controls = window.service.submissions[-1][1]
    assert controls["pipeline_stage"] == "sheet"
    assert Path(controls["resume_checkpoint"]).is_relative_to(record.directory)
    assert controls["fabricate"] is False
    parameters = list(window.project.config.cylinder_mesh_parameter_list)
    parameters[1] += 0.01
    assert window.apply_config({"cylinder_mesh_parameter_list": parameters})
    assert "STALE" in window.result_state.text()
    assert window.compute()
    assert "resume_checkpoint" not in window.service.submissions[-1][1]


def test_compute_rechecks_checkpoint_file_integrity_before_automatic_reuse(windows, prepared_run):
    project, record = prepared_run
    window = windows(project)
    window.load_run(record.directory)
    for artifact in record.metadata["artifacts"]:
        if artifact["kind"] in {"checkpoint", "stage_checkpoint"}:
            path = record.directory / artifact["name"]
            path.write_bytes(path.read_bytes() + b"changed after verification")
    assert window.compute_through("sheet")
    assert "resume_checkpoint" not in window.service.submissions[-1][1]
    assert window._reuse_plan.diagnostics
    before = len(window.service.submissions)
    assert window.resume_selected_run() is None
    assert len(window.service.submissions) == before


def test_explicit_resume_applies_pending_model_before_checking_compatibility(windows, prepared_run):
    project, record = prepared_run
    window = windows(project)
    window.load_run(record.directory)
    _set_stage(window, "sheet")
    window._draft_controls["radius_mm"].setValue(172)
    assert window._draft_dirty
    assert window.resume_selected_run() is None
    assert not window.service.submissions
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.172)
    assert not window._draft_dirty
    assert "compatible" in window.messages.toPlainText().lower()


def test_enabled_channel_toggle_applies_valid_pending_scientific_edit(windows):
    window = windows()
    original = window.project
    window._draft_controls["radius_mm"].setValue(172)
    window.channel_enabled.setChecked(False)
    assert not window.assembly.channels[0].enabled
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.172)
    assert window.project.revision == original.revision + 1
    assert not window._draft_dirty


def test_enabled_channel_toggle_retains_invalid_draft_and_reverts_enabled_state(windows):
    config = preset_config()
    config.coil_mesh_file = "create conical frustum mesh"
    config.surface_is_cylinder_flag = False
    window = windows(ProjectDocument.from_config(config))
    original = window.project
    window._draft_controls["rotation_axis_x"].setValue(0)
    window._draft_controls["rotation_angle_deg"].setValue(45)
    window.channel_enabled.setChecked(False)
    assert window.channel_enabled.isChecked()
    assert window.assembly.channels[0].enabled
    assert not window._collection_mode
    assert window.project is original
    assert window._draft_dirty
    assert window._draft_controls["rotation_angle_deg"].value() == 45
    assert not window.draft_error_label.isHidden()


def test_durable_history_restores_scoped_selection_alias_and_view(windows, prepared_run):
    project, record = prepared_run
    first = windows(project)
    first.load_run(record.directory)
    entry = first._history_entries[0]
    first._rename_history(entry.id, "Mesh baseline")
    first.viewport.state = {"dataset": "target", "units": "µT"}
    first._save_session()
    restored = windows(project, first.jobs_directory)
    assert restored._selected_run is not None
    assert restored.runs[restored._selected_run].directory == record.directory
    assert restored._history_entries[0].label == "Mesh baseline"
    assert restored.viewport.state["dataset"] == "target"
    assert restored.viewport.state["units"] == "µT"
    assert restored.viewport.state["model_node"] == "geometry"
    assert restored.viewport.state["evidence_tab"] == "Jobs"
    unrelated = project.duplicate("Independent project")
    restored._set_project(unrelated)
    assert restored._history_entries == ()
    assert restored._selected_run is None
    assert restored.runs_table.rowCount() == 0
    assert restored.viewport.displayed is None


def test_session_restores_stage_tree_selection(windows):
    original = windows()
    original.tree.setCurrentItem(original.stage_items["sheet"])
    original._save_session()
    restored = windows(original.project, original.jobs_directory)
    assert restored.tree.currentItem() is restored.stage_items["sheet"]


def test_adding_independent_channels_keeps_existing_channel_history(windows, prepared_run):
    project, record = prepared_run
    window = windows(project)
    window.load_run(record.directory)
    entry = window._history_entries[0]
    window._rename_history(entry.id, "Standalone baseline")
    window.add_channel("Additional independent channel")
    window.channel_selector.setCurrentIndex(0)
    assert window.project.id == project.id
    assert any(
        entry.directory == record.directory and entry.label == "Standalone baseline"
        for entry in window._history_entries
    )
    reopened = windows(window.assembly, window.jobs_directory)
    assert any(
        entry.directory == record.directory and entry.label == "Standalone baseline"
        for entry in reopened._history_entries
    )


def test_enabled_conversion_preserves_standalone_history_and_session(windows, prepared_run):
    project, record = prepared_run
    window = windows(project)
    window.load_run(record.directory)
    window._rename_history(window._history_entries[0].id, "Initial independent design")
    window.viewport.state = {"dataset": "target", "units": "µT"}
    window.channel_enabled.setChecked(False)
    assert window._collection_mode
    assert not window.assembly.channels[0].enabled
    assert window.project.id == project.id
    assert window._selected_run is not None
    assert window.runs[window._selected_run].directory == record.directory
    assert window._history_entries[0].label == "Initial independent design"
    assert window.viewport.state["dataset"] == "target"
    reopened = windows(window.assembly, window.jobs_directory)
    assert reopened._selected_run is not None
    assert reopened.runs[reopened._selected_run].directory == record.directory
    assert reopened._history_entries[0].label == "Initial independent design"
    assert reopened.viewport.state["dataset"] == "target"


def test_inactive_channel_completion_persists_without_replacing_active_evidence(windows, prepared_run):
    active, active_record = prepared_run
    inactive = active.duplicate("Inactive independent design")
    assembly = ProjectAssembly.new("Concurrent independent studies", (active, inactive))
    window = windows(assembly)
    window.load_run(active_record.directory)
    selected = window._selected_run
    displayed_identity = window.viewport.displayed["request_sha256"]
    id_ = "inactive_completed"
    directory = window.jobs_directory / id_
    DesignService.run(inactive, directory, threads=1)
    window.jobs.states.append(
        {
            "id": id_,
            "name": inactive.name,
            "status": "completed",
            "pipeline_stage": "prepared",
            "project_id": inactive.id,
            "project_fingerprint": inactive.fingerprint,
            "project_model_fingerprint": inactive.model_fingerprint,
            "project_revision": inactive.revision,
        }
    )
    window.poll_jobs()
    assert window.project.id == active.id
    assert window._selected_run == selected
    assert window.viewport.displayed["request_sha256"] == displayed_identity
    assert window.runs_table.rowCount() == 1
    inactive_channel = assembly.channels[1]
    inactive_history = window.history.restore(inactive.id, channel_id=inactive_channel.id)
    assert len(inactive_history) == 1
    assert inactive_history[0].verified
    assert inactive_history[0].directory == directory
    assert all(entry.project_id == active.id for entry in window._history_entries)
    window.channel_selector.setCurrentIndex(1)
    assert window.project.id == inactive.id
    assert any(entry.directory == directory and entry.verified for entry in window._history_entries)
    reopened = windows(window.assembly, window.jobs_directory)
    assert all(entry.project_id == active.id for entry in reopened._history_entries)
    reopened.channel_selector.setCurrentIndex(1)
    assert any(entry.directory == directory and entry.verified for entry in reopened._history_entries)


def test_unavailable_history_does_not_restore_old_in_memory_verified_evidence(windows, prepared_run):
    project, record = prepared_run
    window = windows(project)
    window.load_run(record.directory)
    unrelated = project.duplicate("Other project")
    window._set_project(unrelated)
    artifact = next(
        artifact for artifact in record.metadata["artifacts"] if artifact["name"] == "scientific_result.npz"
    )
    path = record.directory / artifact["name"]
    path.write_bytes(path.read_bytes() + b"corrupted after verification")
    window._set_project(project)
    assert window._history_entries
    assert not window._history_entries[0].verified
    assert window._history_entries[0].diagnostic
    assert window._selected_run is None
    assert window.viewport.displayed is None


def _show_layout(window, application):
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if font.exists():
        font_id = QFontDatabase.addApplicationFont(str(font))
        family = QFontDatabase.applicationFontFamilies(font_id)[0]
        window.setFont(QFont(family, 9))
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1280, 800)
    window.show()
    for _ in range(3):
        application.processEvents()


def test_results_values_remain_fully_visible_in_narrow_inspector(windows, application, tmp_path):
    window = windows()
    window.runs["layout"] = SimpleNamespace(
        directory=tmp_path / "layout_run",
        arrays={},
        metadata={
            "summary": {
                "power_w": 0.0432765,
                "validated_relative_field_error": 0.0452778,
                "sheet_accepted": True,
                "accepted": False,
                "scope": "fabrication_review",
            },
            "wire_fields": {"contour": {"current_A": 1.56028}},
            "fabrication": {
                "ready": False,
                "electrical_metrics": {"bodies": [{"body_id": "part001_body001", "resistance_ohm": 0.668019}]},
            },
        },
    )
    window.show_run("layout")
    window._inspect_node("results")
    _show_layout(window, application)
    dock = window.findChild(QDockWidget, "settings_dock")
    window.resizeDocks([dock], [285], Qt.Orientation.Horizontal)
    table = window.settings.findChild(QTableWidget, "selected_run_metrics")
    table.setFixedWidth(260)
    for _ in range(3):
        application.processEvents()
    assert table.horizontalScrollBar().maximum() == 0
    assert table.item(0, 2).text() == "0.0432765"
    for row in range(table.rowCount()):
        item = table.item(row, 2)
        cell = table.visualItemRect(item)
        assert cell.left() >= 0
        assert cell.right() <= table.viewport().rect().right()
        padding = 2 * (table.style().pixelMetric(QStyle.PixelMetric.PM_FocusFrameHMargin) + 1)
        assert table.fontMetrics().horizontalAdvance(item.text()) <= cell.width() - padding
        assert item.toolTip()


def test_stage_status_fits_inside_narrow_model_tree(windows, application):
    window = windows()
    window.states["failed"] = {
        "project_id": window.project.id,
        "project_fingerprint": window.project.fingerprint,
        "project_revision": window.project.revision,
        "pipeline_stage": "evaluated",
        "status": "failed",
    }
    window._refresh_stage_status(refresh_reuse=False)
    _show_layout(window, application)
    item = window.stage_items["evaluated"]
    assert item.text(1) == "Execution failed"
    header = window.tree.header()
    assert header.sectionSize(1) >= window.tree.fontMetrics().horizontalAdvance(item.text(1)) + 6
    assert header.sectionViewportPosition(1) >= 0
    assert header.sectionViewportPosition(1) + header.sectionSize(1) <= window.tree.viewport().width()
    assert window.tree.horizontalScrollBar().maximum() == 0


def test_switching_to_no_history_project_clears_previous_scientific_evidence(windows, tmp_path):
    window = windows()
    project = window.project
    record = RunRecord(
        tmp_path / "detached_scientific_evidence",
        {
            "project_id": project.id,
            "project_fingerprint": project.fingerprint,
            "summary": {"scope": "routed", "accepted": False},
            "constraints": [{"name": "Route clearance", "value": -0.001, "limit": 0.0, "unit": "m", "passed": False}],
            "wire_fields": {"routed": {"current_A": 1.56, "rms_error_T": 2.94e-7, "relative_rms_error": 0.10}},
            "warnings": ["Previous conductor has unverified terminals."],
        },
        {},
    )
    window.runs["previous"] = record
    window.show_run("previous")
    assert window.checks_table.rowCount() == 1
    assert window.wire_fields_table.rowCount() == 1
    assert "Previous conductor" in window.messages.toPlainText()
    window._set_project(project.duplicate("Independent draft without results"))
    assert window._selected_run is None
    assert window.viewport.displayed is None
    assert window.checks_table.rowCount() == 0
    assert window.wire_fields_table.rowCount() == 0
    assert "Previous conductor" not in window.messages.toPlainText()
