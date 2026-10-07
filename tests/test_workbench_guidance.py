"""Guided actions use the shared study/controller while preserving scientific inputs."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import test_workbench_usability as usability

from PySide6.QtCore import Qt

from coilgen.application import ProjectDocument, dependency_plan
from coilgen.application.presets import preset_config, preset_study
from coilgen.gui import workbench
from coilgen.gui.readiness import ReadinessReport


application = usability.application
windows = usability.windows


@pytest.fixture(autouse=True)
def ready_environment(monkeypatch):
    # Dedicated readiness tests cover file/solver discovery. These integration
    # contracts exercise navigation and use the real model/draft validators.
    monkeypatch.setattr(workbench, "check_readiness", lambda *_args, **_kwargs: ReadinessReport())


def _page(window):
    return window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole)


def _install_run(window, tmp_path, *, metadata=None):
    project = window.project
    payload = {
        "name": "Saved trial",
        "project_id": project.id,
        "project_fingerprint": project.fingerprint,
        "project_model_fingerprint": project.model_fingerprint,
        "project_revision": project.revision,
        "pipeline_stage": "evaluated",
        "summary": {"accepted": True, "sheet_accepted": True, "scope": "continuous_sheet"},
        "fabrication": None,
        "pipeline_sweeps": {"checks_passed": None, "body_count": 0},
        "artifacts": [],
    }
    payload.update(metadata or {})
    record = SimpleNamespace(directory=tmp_path / "saved_trial", metadata=payload, arrays={})
    window.service.load_run = lambda _path: record
    window.load_run(record.directory)
    return record


def test_ready_next_action_is_visible_without_changing_model_or_study(windows):
    project = ProjectDocument.from_config(preset_config()).update_study(preset_study())
    window = windows(project)
    assert window.design_overview._state.action_key == "compute_full"
    assert "Compute full design" in window.design_overview.action_button.text()
    assert window.overview_window.isHidden()
    window.overview_window.open_panel()
    assert window.overview_window.isVisible()
    assert not window.design_overview.details.isHidden()
    assert window.design_status_label.text() == window.design_overview._state.title
    assert window.project is project
    assert window.undo_stack.count() == 0
    assert not window.service.submissions


def test_seven_physical_overview_sections_map_to_existing_settings(windows):
    window = windows()
    original = window.project
    expected = {
        "target": "target",
        "support": "geometry",
        "conductor": "material",
        "solve": "optimization",
        "winding": "winding",
        "check": "results",
        "export": "export",
    }
    assert tuple(window.design_overview.stage_buttons) == tuple(expected)
    for key, page in expected.items():
        window.design_overview.stage_buttons[key].click()
        assert _page(window) == page
    assert window.project is original
    assert window.undo_stack.count() == 0
    assert not window.service.submissions


def test_pending_guided_compute_preserves_selected_boundary_and_requested_skips(windows):
    config = replace(preset_config(), skip_postprocessing=True, skip_sweep=True)
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "sheet", "fabricate": False})
    window = windows(project)
    window._draft_controls["radius_mm"].setValue(90)
    assert window.design_overview._state.action_key == "compute_full"
    assert "Pending model edits" in window.design_overview.title.text()

    window.design_overview.action_button.click()

    assert len(window.service.submissions) == 1
    submitted, controls = window.service.submissions[0]
    assert submitted.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.09)
    assert controls["pipeline_stage"] == "sheet"
    assert controls["fabricate"] is False
    assert submitted.config.skip_postprocessing is True
    assert submitted.config.skip_sweep is True
    assert submitted.config.levels == config.levels


def test_invalid_draft_guided_action_retains_values_and_stops_submission(windows):
    project = ProjectDocument.from_config(preset_config("conical_harmonic")).update_study(
        preset_study("conical_harmonic")
    )
    window = windows(project)
    original = window.project
    window._draft_controls["rotation_axis_x"].setValue(0)
    window._draft_controls["rotation_angle_deg"].setValue(45)

    window.design_overview.action_button.click()

    assert not window.service.submissions
    assert window.project is original
    assert window._draft_dirty
    assert window._draft_controls["rotation_angle_deg"].value() == 45
    assert not window.readiness_panel.isHidden()
    assert window.undo_stack.count() == 0


def test_stale_guidance_contains_actual_numerical_dependency_plan(windows, tmp_path):
    window = windows()
    original = window.project
    _install_run(window, tmp_path)
    assert window.apply_config({"levels": original.config.levels + 2})
    plan = dependency_plan(original.config, window.project.config)

    state = window.design_overview._state
    assert state.action_key == "recompute"
    assert "levels" in state.detail
    assert " → ".join(plan.stages) in state.detail
    assert "prepared" not in plan.stages and "sheet" not in plan.stages
    assert not window.service.submissions


def test_failed_native_geometry_opens_problems_without_invented_witness(windows, tmp_path):
    window = windows()
    window.viewport.locate_issue = Mock()
    _install_run(
        window,
        tmp_path,
        metadata={
            "pipeline_sweeps": {
                "checks_passed": False,
                "body_count": 1,
                "bodies": [
                    {
                        "body_id": "native_body",
                        "manufacturing_ok": False,
                        "diagnostics": {"manufacturing_failure_reasons": "folded_sweep_quad"},
                    }
                ],
            }
        },
    )
    assert window.design_overview._state.action_key == "inspect_problems"

    window.design_overview.action_button.click()

    assert _page(window) == "results"
    assert window.evidence.currentWidget() is window.problems_panel
    assert window.problems_panel.issues
    assert all(not issue.locatable for issue in window.problems_panel.issues)
    assert not window.problems_panel.locate.isEnabled()
    window.viewport.locate_issue.assert_not_called()


def test_failed_review_only_locates_hash_verified_saved_frame_witness(windows, tmp_path):
    window = windows()
    directory = tmp_path / "saved_trial"
    directory.mkdir()
    frames = directory / "body_fabrication_frames.npz"
    points = np.array([[0.08, 0, 0], [0.08, 0.005, 0], [0.075, 0.005, 0]])
    np.savez(
        frames,
        points_m=points,
        normals=np.tile([0, 0, 1.0], (3, 1)),
        transverse=np.tile([0, 1.0, 0], (3, 1)),
        cross_section_m=np.array([[-0.0005, -0.0005], [0.0005, -0.0005], [0.0005, 0.0005], [-0.0005, 0.0005]]),
        closed=False,
    )
    window.viewport.locate_issue = Mock()
    record = _install_run(
        window,
        tmp_path,
        metadata={
            "fabrication": {
                "ready": False,
                "paths": [
                    {
                        "body_id": "body",
                        "diagnostics": {
                            "manufacturing_failure_reasons": "folded_sweep_quad",
                            "sweep_ring_count": 3,
                            "sweep_worst_quad_ring": 0,
                            "sweep_worst_quad_section": 1,
                        },
                    }
                ],
            },
            "artifacts": [{"name": frames.name, "sha256": sha256(frames.read_bytes()).hexdigest()}],
        },
    )
    assert any(issue.locatable for issue in window.problems_panel.issues)

    window.design_overview.action_button.click()

    selected = window.viewport.locate_issue.call_args.args[0]
    assert selected in window.problems_panel.issues and selected.locatable
    assert selected.run_directory == str(record.directory)
    assert len(selected.points_m) == 4
    assert np.asarray(selected.points_m)[:, 0].min() == pytest.approx(0.08)


def test_export_navigation_preserves_current_inputs_and_does_not_compute(windows, tmp_path):
    window = windows()
    _install_run(window, tmp_path)
    original = window.project

    window.navigate_workflow("export")

    assert _page(window) == "export"
    assert window.project is original
    assert window.project.study == original.study
    assert window.undo_stack.count() == 0
    assert not window.service.submissions
    assert not window.geometry_export_panel.export_button.isEnabled()
