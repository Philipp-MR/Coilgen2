"""Live stage display stays distinct from scientific result evidence."""

import time
from unittest.mock import Mock

import numpy as np
import pytest

import test_workbench_usability as usability

from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config, preset_study
from coilgen.application.service import RunRecord
from coilgen.application.stage_preview import StagePreview
from coilgen.gui.workbench import CoilGenWorkbench


application = usability.application
KEYS = ("prepared", "sheet", "contours", "routed", "evaluated")


class StageViewport(usability.Preview):
    context_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = QComboBox(self)
        for key in ("support", "preview_target", "target", "stream_function", "contours", "swept"):
            self.dataset.addItem(key, key)
        self.dataset.currentIndexChanged.connect(self.context_changed)
        self.stage_preview_metadata = None
        self.stage_arrays = None

    def display_stage_preview(self, arrays, metadata):
        self.stage_preview_metadata, self.stage_arrays = dict(metadata), arrays
        preferred = {"prepared": "target", "sheet": "stream_function", "contours": "contours"}.get(
            metadata["pipeline_stage"], "swept"
        )
        self.dataset.setCurrentIndex(self.dataset.findData(preferred))
        self.context_changed.emit()

    def clear_stage_preview(self):
        self.stage_preview_metadata = None
        self.stage_arrays = None
        self.dataset.setCurrentIndex(self.dataset.findData("preview_target"))
        self.context_changed.emit()

    def display_run(self, arrays, metadata=None):
        self.stage_preview_metadata = None
        super().display_run(arrays, metadata)

    def view_state(self):
        return {"dataset": self.dataset.currentData(), "source": "live" if self.stage_preview_metadata else "model"}


class StageService(usability.Service):
    def submit(self, *args, **kwargs):
        state = super().submit(*args, **kwargs)
        state["request_sha256"] = "a" * 64
        args[1].states[-1].update(state)
        return state


@pytest.fixture
def window(application, tmp_path):
    project = ProjectDocument.from_config(preset_config()).update_study(preset_study())
    instance = CoilGenWorkbench(
        project, jobs=usability.Queue(tmp_path / "jobs"), service=StageService(), viewport_factory=StageViewport
    )
    instance.timer.stop()
    yield instance
    instance.live_preview.close()
    QThreadPool.globalInstance().waitForDone(3000)
    application.processEvents()
    instance._draft_dirty = False
    instance.undo_stack.setClean()
    instance.close()
    instance.undo_stack.clear()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _wait(predicate):
    limit = time.monotonic() + 3
    while not predicate() and time.monotonic() < limit:
        QTest.qWait(5)
    assert predicate()


def _snapshot(window, state, stage):
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]])
    arrays = {"target_points_m": points, "target_field_T": np.ones((2, 3)) * 1e-4}
    if stage != "prepared":
        arrays.update(vertices_m=points, faces=np.zeros((1, 3), dtype=int), stream_function_A=np.array([0.0, 1.0]))
    if stage in {"contours", "routed", "evaluated"}:
        arrays["contour_0_0_points_m"] = points
    metadata = {
        **{key: state[key] for key in ("project_id", "project_revision", "project_fingerprint", "request_sha256")},
        "pipeline_stage": stage,
        "completed_stages": list(KEYS[: KEYS.index(stage) + 1]),
        "scope": "completed_boundary_preview",
        "acceptance_claim": False,
    }
    return StagePreview(window.jobs_directory / state["id"], metadata, arrays)


def _publish(window, state, stage):
    preview = _snapshot(window, state, stage)
    window.live_preview._loader = lambda *_args, **_kwargs: preview
    directory = preview.directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stage_preview.json").write_text(stage, encoding="utf-8")
    window.jobs.states[0].update(status="running", stage=f"Computing after {stage}", completed_stages=preview.metadata["completed_stages"])
    window.poll_jobs()
    return preview


def _record(window, state):
    return RunRecord(
        window.jobs_directory / state["id"],
        {**state, "summary": {"accepted": False}, "warnings": [], "constraints": [], "artifacts": []},
        {},
    )


def test_submit_keeps_plot_visible_and_progress_has_no_guessed_completion(window):
    state = window.compute()
    assert window.live_preview.job_id == state["id"]
    assert window.live_preview.following
    assert not window.evidence.any_open()
    assert window.output_scope_label.isHidden()
    assert all(stage.status == "waiting" for stage in window.design_progress.snapshot.stages)
    window.jobs.states[0].update(status="running", progress=0.99, stage="Contours done")
    window.poll_jobs()
    assert not any(stage.status == "complete" for stage in window.design_progress.snapshot.stages)


def test_actual_stage_previews_update_graphics_without_creating_results(window):
    state = window.compute()
    for stage in KEYS:
        preview = _publish(window, state, stage)
        _wait(lambda: window.viewport.stage_preview_metadata is not None and window.viewport.stage_preview_metadata["pipeline_stage"] == stage)
        np.testing.assert_array_equal(window.viewport.stage_arrays["target_field_T"], preview.arrays["target_field_T"])
        assert window.view_context.context.source == "live"
        assert f"Live {stage}" in window.view_context.context.viewing_text
        assert window.runs == {}
        assert window._selected_run is None
        assert sum(item.status == "complete" for item in window.design_progress.snapshot.stages) == KEYS.index(stage) + 1
        assert all(window.stage_items[key].text(1) == "Computed" for key in KEYS[: KEYS.index(stage) + 1])
        assert window._workbench_view_state()["source"] == "model"
        assert window._workbench_view_state()["dataset"] == "preview_target"


def test_manual_selection_pauses_until_follow_is_resumed(window):
    state = window.compute()
    _publish(window, state, "sheet")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    window.viewport.dataset.setCurrentIndex(window.viewport.dataset.findData("target"))
    window.viewport.dataset.activated.emit(window.viewport.dataset.currentIndex())
    assert not window.live_preview.following
    _publish(window, state, "contours")
    QTest.qWait(30)
    assert window.viewport.dataset.currentData() == "target"
    assert window.viewport.stage_preview_metadata["pipeline_stage"] == "sheet"
    window._set_live_follow(True)
    _wait(lambda: window.viewport.stage_preview_metadata["pipeline_stage"] == "contours")


def test_final_verification_does_not_override_paused_plot(window):
    state = window.compute()
    _publish(window, state, "sheet")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    window._pause_live_follow()
    window.service.load_run = Mock(return_value=_record(window, state))
    window.jobs.states[0].update(status="completed", stage="Complete", progress=1.0)
    window.poll_jobs()
    assert state["id"] in window.runs
    assert window.viewport.displayed is None
    assert window.viewport.stage_preview_metadata["pipeline_stage"] == "sheet"
    assert window._selected_run is None
    assert window.design_progress.snapshot.status == "completed"


def test_verified_final_result_exits_live_scope(window):
    state = window.compute()
    _publish(window, state, "sheet")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    window.service.load_run = Mock(return_value=_record(window, state))
    window.jobs.states[0].update(status="completed", stage="Complete", progress=1.0)
    window.poll_jobs()
    assert window.live_preview.job_id is None
    assert window.viewport.stage_preview_metadata is None
    assert window._selected_run == state["id"]
    assert window.runs[state["id"]].metadata["summary"]["accepted"] is False


def test_viewing_saved_run_pauses_active_follow_and_can_return_to_live_plot(window):
    state = window.compute()
    _publish(window, state, "sheet")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    window.runs["historical"] = _record(window, {**state, "id": "historical", "name": "Older study"})
    window.show_run("historical")
    assert window._selected_run == "historical"
    assert window.live_preview.job_id == state["id"]
    assert not window.live_preview.following
    assert window.viewport.stage_preview_metadata is None
    _publish(window, state, "contours")
    window._set_live_follow(True)
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    assert window.viewport.stage_preview_metadata["pipeline_stage"] == "contours"
    assert window.view_context.context.source == "live"
    assert window._selected_run == "historical"


def test_verification_failure_retains_last_preview_and_separate_failed_status(window):
    state = window.compute()
    _publish(window, state, "sheet")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    window.service.load_run = Mock(side_effect=ValueError("Broken final checksum"))
    window.jobs.states[0].update(status="completed", stage="Complete", progress=1.0)
    window.poll_jobs()
    assert window.viewport.stage_preview_metadata["pipeline_stage"] == "sheet"
    assert not window.runs
    assert "verification failed" in window.design_progress.snapshot.activity.lower()


@pytest.mark.parametrize("replacement", ["revision", "channel", "close"])
def test_model_replacement_invalidates_preview_scope(window, replacement):
    state = window.compute()
    _publish(window, state, "sheet")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    if replacement == "revision":
        window._set_project(window.project.update_config({"target_gradient_strength": 0.031}))
    elif replacement == "channel":
        window._set_project(ProjectDocument.from_config(preset_config()))
    else:
        window._clear_project()
    assert window.live_preview.job_id is None
    window._show_live_preview(state["id"], _snapshot(window, state, "contours"))
    assert window.viewport.stage_preview_metadata is None or replacement == "close"


def test_return_to_model_pauses_and_shows_current_target(window):
    state = window.compute()
    _publish(window, state, "sheet")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    window._view_applied_model()
    assert not window.live_preview.following
    assert window.viewport.stage_preview_metadata is None
    assert window.viewport.dataset.currentData() == "preview_target"
    assert window.view_context.context.source == "model"


def test_cancelled_job_keeps_completed_geometry_without_acceptance_claim(window):
    state = window.compute()
    _publish(window, state, "contours")
    _wait(lambda: window.viewport.stage_preview_metadata is not None)
    window.jobs.states[0].update(status="cancelled")
    window.poll_jobs()
    assert window.viewport.stage_preview_metadata["pipeline_stage"] == "contours"
    assert not window.runs
    statuses = [stage.status for stage in window.design_progress.snapshot.stages]
    assert statuses[:3] == ["complete"] * 3
    assert statuses[3] == "stopped"


def test_other_channel_with_identical_parameters_cannot_claim_current_stage_progress(window):
    state = window.compute()
    foreign = {**state, "id": "other-channel-run", "project_id": "other-channel", "status": "failed"}
    foreign["completed_stages"] = list(KEYS)
    window.states[foreign["id"]] = foreign
    window._refresh_stage_status(refresh_reuse=False)
    assert window.stage_items["prepared"].text(1) == "Not computed"
    assert window.stage_items["evaluated"].text(1) == "Queued"
    assert not any(stage.status == "complete" for stage in window.design_progress.snapshot.stages)
