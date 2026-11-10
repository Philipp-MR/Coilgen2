"""Browser regressions for source selection, engineering verdicts and editable state."""

import asyncio
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("trame")

from coilgen.application.diagnostics import run_verdicts
from coilgen.application.cad import PRINT_FORMER_SCOPE, GeometryOption
from coilgen.application.service import RunRecord
from coilgen.web.app import create_app


def test_browser_former_export_preselects_only_former_and_requires_draft_ack(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-browser-print-former")
    state = server.state
    former = "fabrication/part001_grooved_support_DRAFT_fabrication.stl"
    conductor = "fabrication/part001_body0001_fabrication.stl"
    vertices, faces = np.eye(3), np.array([[0, 1, 2]])
    record = RunRecord(
        tmp_path,
        {
            "fabrication": {"support_bodies": [{"body_id": "part001_grooved_support", "support_body_ok": True}]},
            "artifacts": [{"name": former}],
        },
        {"vertices_m": vertices, "faces": faces, "stream_function_A": np.zeros(3)},
    )
    options = tuple(
        GeometryOption(name, "part001", "stl", scope, "passed", "m", ((0, 0, 0), (1, 1, 1)), True, "digest")
        for name, scope in ((former, PRINT_FORMER_SCOPE), (conductor, "fabrication_review_geometry"))
    )
    monkeypatch.setattr(browser.session, "load_run", lambda job_id: record)
    monkeypatch.setattr(browser.session, "export_options", lambda record: options)
    export = Mock(return_value=None)
    monkeypatch.setattr(browser.session, "export", export)
    plot = Mock(return_value={"review_grooved_support_0_vertices_m": vertices, "review_grooved_support_0_faces": faces})
    monkeypatch.setattr("coilgen.web.app.saved_print_former_arrays", plot)
    try:
        template = state.trame__template_main
        assert "results_metrics_expanded = false; dataset = 'print_former'" in template
        assert 'aria-controls="coilgen-result-metrics"' in template
        assert 'v-if="export_options.filter(option => option.former).length > 1"' in template
        state.selected_run = "with-former"
        browser._open_run("with-former")
        assert state.summary["print_former_status"] == "Print former · geometry checks passed"
        plot.assert_not_called()
        state.dataset = "print_former"
        browser._render()
        browser._render()
        plot.assert_called_once_with(record)
        assert "saved fabrication STL" in state.scene_caption
        assert state.export_selected == [conductor]
        browser.open_print_former()
        assert state.center_section == "export_former" and state.former_export_selected == [former]
        browser.export_geometry()
        export.assert_not_called()
        assert "DRAFT confirmation" in state.message
        state.former_draft_ack = True
        browser.export_geometry()
        export.assert_called_once_with(record, [former], unit="mm", allow_draft=True, include_project=True)
        assert state.export_selected == [conductor] and not state.draft_ack
    finally:
        browser.session.close()


def test_shielding_plot_uses_saved_channel_and_clears_on_switch(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-browser-shielding")
    state = server.state
    arrays = {
        "target_points_m": np.eye(3) * 0.15,
        "target_suppressed": np.array([True, False, True]),
        "target_weights": np.ones(3),
        "target_field_T": np.zeros((3, 3)),
        "sheet_field_T": np.eye(3) * 3e-6,
    }
    records = {
        "shielded": RunRecord(tmp_path, {"name": "Gx", "pipeline_stage": "sheet"}, arrays),
        "unshielded": RunRecord(tmp_path, {"name": "Gy"}, {}),
    }
    monkeypatch.setattr(browser.session, "load_run", records.__getitem__)
    monkeypatch.setattr(browser.session, "export_options", lambda record: ())
    browser._open_run("shielded")
    assert [s["value"] for s in state.shielding_sources] == ["sheet"]
    asyncio.run(browser.show_shielding("sheet", "Bx"))
    assert state.center_section == "shielding" and state.shielding_image.startswith("data:image/png;base64,")
    assert state.shielding_note.startswith("2 ") and "equal-sample RMS" in state.shielding_note
    assert not state.shielding_normal
    browser._open_run("unshielded")
    assert state.center_section == "results" and not state.shielding_image and not state.shielding_sources


def test_view_completed_batch_job_verifies_that_channel_without_changing_editor(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-batch-job-review")
    session, state = browser.session, server.state
    browser.load_template("cylindrical_xyz_gradient")
    editing = session.active_channel_id
    jobs = {}
    for index, channel in enumerate(session.channels):
        project = channel.project
        job_id = f"batch-{index}"
        job = {
            "id": job_id,
            "status": "completed",
            "name": channel.name,
            "project_id": project.id,
            "project_revision": project.revision,
            "project_fingerprint": project.fingerprint,
            "pipeline_stage": project.study["pipeline_stage"],
        }
        session.states[job_id] = job
        session.job_ids.append(job_id)
        jobs[job_id] = RunRecord(tmp_path, job, {})
    monkeypatch.setattr(session, "load_run", jobs.__getitem__)
    monkeypatch.setattr(session, "export_options", lambda _record: ())
    browser._refresh_jobs()
    assert not state.workspace_history and len(state.runs) == 1
    state.settings_dirty = True
    for job_id in jobs:
        browser.view_job_result(job_id)
        assert state.selected_run == job_id
        browser._open_run(state.selected_run)  # Vue's selected-run callback
    assert state.workspace_history and len(state.runs) == 3
    assert session.active_channel_id == editing and state.settings_dirty
    assert state.plan_summary == "15/15 steps recorded · 3/3 channels with verified results"
    assert state.run_context == f"Viewing channel {session.channels[-1].name} · editing {session.project.name}"
    browser.view_job_result("missing")
    assert state.selected_run == "batch-2" and state.message_color == "warning"


def test_browser_legend_and_camera_use_the_rendered_data(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-browser-plot-parity")
    state = server.state
    browser.load_template("cylinder")
    browser.set_camera_view("XY")
    np.testing.assert_allclose(browser.plotter.camera.direction, (0, 0, -1))
    np.testing.assert_allclose(browser.plotter.bounds, [-0.08, 0.08, -0.08, 0.08, -0.1, 0.1], atol=1e-7)
    # Compass markers must not move or expand the physical model bounds.
    assert all(widget.GetEnabled() for widget in browser.view.get_widgets())
    arrays = {
        "target_points_m": np.array([[0, 0, 0], [0, 0, 0.01]]),
        "target_field_T": np.zeros((2, 3)),
        "routed_field_T": np.array([[0, 0, -2e-6], [0, 0, 5e-6]]),
        "validation_points_m": np.array([[0, 0.01, 0], [0, 0.02, 0]]),
        "validation_field_T": np.array([[0, 0, 2e-6], [0, 0, 3e-6]]),
        "validation_target_T": np.array([[0, 0, -8e-6], [0, 0, 6e-6]]),
    }
    record = RunRecord(tmp_path, {"config": {"b_0_direction": [0, 0, 1]}}, arrays)
    monkeypatch.setattr(browser.session, "load_run", lambda _key: record)
    monkeypatch.setattr(browser.session, "export_options", lambda _record: ())
    state.dataset = "routed_residual"
    browser._open_run("saved")
    assert (state.field_legend["low"], state.field_legend["high"]) == ("-5", "5")
    assert state.field_legend["label"] == "ΔBz [µT]"
    state.component = "Bmag"
    browser._render()
    assert (state.field_legend["low"], state.field_legend["high"]) == ("2", "5")
    assert state.field_legend["label"] == "|B − target| [µT]"
    assert "includes transverse components" in state.field_source_header
    state.dataset, state.component = "validation", "Bz"
    browser._render()
    assert state.field_has_pair and "Independent grid · 2 samples" in state.field_source_header
    assert (state.field_legend["low"], state.field_legend["high"]) == ("-8", "8")
    state.field_show_target = True
    browser._render()
    assert state.field_legend["label"] == "Target Bz [µT]"
    assert (state.field_legend["low"], state.field_legend["high"]) == ("-8", "8")
    assert "Target Bz [µT] -8 … 6" in state.scene_caption
    state.dataset = "support"
    browser._render()
    assert state.field_legend is None


def test_only_outside_mounted_review_defaults_to_a_solid_support(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-browser-outer-winding")
    state = server.state
    points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
    arrays = {
        "vertices_m": points,
        "faces": np.array([[0, 1, 2]]),
        "review_conductor_0_points_m": points + [0, 0, 0.001],
    }
    state.dataset = "review_conductor"
    for mode, see_through, expected_opacity in (
        ("surface_outer", False, 1),
        ("surface_outer", True, 0.38),
        (None, False, 0.38),
    ):
        browser.record = RunRecord(tmp_path, {"config": {"conductor_mounting_mode": mode}}, arrays)
        state.see_through_support = see_through
        browser._render()
        assert state.outer_winding_support == (mode == "surface_outer")
        support = browser.plotter.renderer.actors["support"]
        wire = browser.plotter.renderer.actors["review_conductor"]
        assert support.prop.opacity == pytest.approx(expected_opacity)
        np.testing.assert_array_equal(wire.mapper.dataset.points, arrays["review_conductor_0_points_m"])


def test_historical_projection_uses_the_record_and_switching_documents_clears_results(tmp_path, monkeypatch):
    server, workbench = create_app(tmp_path, server_name="test-historical-projection")
    state = server.state
    workbench.load_template("cylinder")
    project = workbench.session.project
    record = RunRecord(
        tmp_path,
        {
            "name": project.name,
            "project_id": project.id,
            "project_revision": project.revision,
            "project_fingerprint": project.fingerprint,
            "pipeline_stage": "evaluated",
            "config": {"b_0_direction": [0, 0, 1]},
            "summary": {"accepted": True},
        },
        {"target_points_m": np.array([[0, 0, 0]]), "target_field_T": np.array([[2e-6, 0, 7e-6]])},
    )
    monkeypatch.setattr(workbench.session, "load_run", lambda _key: record)
    monkeypatch.setattr(workbench.session, "export_options", lambda _record: ())
    state.component = "BdotB0"
    workbench._open_run("saved-run")
    assert [step["code"] for step in state.plan_channels[0]["steps"]] == ["complete"] * 5
    workbench.session.apply_inputs({"b_0_direction": (1, 0, 0)})
    workbench._render()
    assert state.b0_text == "B0 ∥ (0, 0, 1)"
    assert "BdotB0 [µT] 7" in state.scene_caption
    assert state.run_context.startswith("Earlier")
    assert all(step["code"] == "stale" for step in state.plan_channels[0]["steps"])
    state.dataset = "support"
    workbench._render()
    assert state.b0_text == "B0 ∥ (1, 0, 0)" and state.scene_caption.startswith("Editing model")
    state.download_url = "downloads/previous-run.zip"
    workbench.new_project()
    assert workbench.record is None and state.summary is None
    assert all(step["code"] == "waiting" for step in state.plan_channels[0]["steps"])
    assert not state.export_options and not state.download_url and not state.run_context

    # A legacy record can still show Cartesian components without inventing a projection.
    legacy = RunRecord(tmp_path, {"summary": {"scope": "prepared_geometry"}}, record.arrays)
    monkeypatch.setattr(workbench.session, "load_run", lambda _key: legacy)
    state.dataset = "target"
    workbench._open_run("legacy")
    assert state.b0_text == "B0 unavailable" and "no recorded B0 direction" in state.scene_caption
    assert state.summary["label"] == "Prepare model completed"
    state.component = "Bx"
    workbench._render()
    assert "Bx [µT] 2" in state.scene_caption


def test_pending_edits_advanced_json_and_physical_geometry(tmp_path):
    server, workbench = create_app(tmp_path, server_name="test-browser-pending")
    state = server.state
    workbench.load_template("cylinder")
    assert state.settings_values["cylinder_length_mm"] == pytest.approx(200)
    state.settings_values = {**state.settings_values, "cylinder_radius_mm": "135"}
    workbench.compute()
    workbench.save_project()
    assert not workbench.session.job_ids and not state.project_download and state.settings_dirty
    workbench.apply_settings()
    assert not state.settings_dirty
    assert workbench.session.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.135)
    workbench.open_advanced()
    state.advanced_study = "[]"
    workbench.apply_advanced()
    assert state.advanced_dialog and "JSON object" in state.message


def test_batch_uses_one_boundary_without_modifying_disabled_channel(tmp_path, monkeypatch):
    server, workbench = create_app(tmp_path, server_name="test-browser-batch")
    state, session = server.state, workbench.session
    state.template = "cylindrical_xyz_gradient"
    workbench.load_template()
    excluded = session.channels[-1]
    workbench.toggle_channel(excluded.id, False)
    observed = []

    def submit(assembly, _jobs):
        observed.extend(channel.project.study["pipeline_stage"] for channel in assembly.channels if channel.enabled)
        return {}

    monkeypatch.setattr(session.service, "submit_assembly", submit)
    state.stage = "prepared"
    workbench.compute_channels()
    assert observed == ["prepared", "prepared"]
    assert session.channels[-1].project.study == excluded.project.study
    assert session.channels[-1].project.revision == excluded.project.revision


def test_early_boundaries_do_not_report_unperformed_checks_as_failures():
    meta = {
        "pipeline_stage": "prepared",
        "summary": {"accepted": False, "sheet_accepted": False},
        "pipeline_sweeps": {"checks_passed": False},
    }
    assert run_verdicts(meta) == {"sheet": "Not run", "sweeps": "Not run", "fabrication": "Not run"}


def test_empty_results_explains_only_the_latest_matching_attempt(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-web-stopped-attempt")
    browser.load_template("cylinder")
    session, state = browser.session, server.state
    project = session.project
    identity = {
        "project_id": project.id,
        "project_revision": project.revision,
        "project_fingerprint": project.fingerprint,
        "pipeline_stage": state.stage,
    }

    def add(job_id, status, **changes):
        session.states[job_id] = {**identity, "id": job_id, "status": status, **changes}
        session.job_ids.append(job_id)
        browser._refresh_jobs()

    add("failed-here", "failed", error="Could not publish results", completed_stages=["prepared"])
    assert "failed" in state.attempt_notice["title"] and "worker error" in state.attempt_notice["action"]
    channel = state.plan_channels[0]
    assert channel["steps"][0]["status"] == "Completed"
    assert channel["error"] and "Execution failed" in channel["activity"]
    add("foreign", "failed", project_id="other")
    add("older", "cancelled", project_revision=project.revision - 1)
    add("branched", "failed", project_fingerprint="different")
    add("other-stage", "cancelled", pipeline_stage="prepared")
    assert "failed-h" in state.attempt_notice["title"]
    assert state.summary is None
    add("retry", "queued")
    assert state.current_job_active and state.attempt_notice is None
    session.states["retry"]["status"] = "cancelled"
    browser._refresh_jobs()
    assert "retry cancelled" in state.attempt_notice["title"] and "Choose Compute" in state.attempt_notice["action"]
    add("successful", "completed")
    assert not state.current_job_active and state.attempt_notice is None
    assert state.jobs[0]["id"] == "successful"
    assert len(state.jobs) == 3 and "failed-here" in {job["id"] for job in state.older_jobs}
    session.apply_settings({"levels": 9})
    assert state.attempt_notice is None


def test_activity_keeps_background_jobs_visible_without_inventing_progress(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-web-background-activity")
    session, state = browser.session, server.state
    try:
        browser.load_template("cylinder")
        job = {
            "id": "background",
            "project_id": "other-project-in-this-workspace",
            "pipeline_stage": "prepared",
            "status": "queued",
            "progress": 0.2,
        }
        session.states[job["id"]] = job
        session.job_ids.append(job["id"])
        browser._refresh_jobs()
        assert not state.jobs and not state.current_job_active
        assert state.active_jobs[0]["progress"] is None

        job.update(status="running", progress_kind="allocated_work", progress_indeterminate=True)
        browser._refresh_jobs()
        assert state.active_jobs[0]["progress"] is None
        assert not state.active_jobs[0]["progress_label"]

        job.update(progress_indeterminate=False)
        browser._refresh_jobs()
        assert state.active_jobs[0]["progress_label"] == "Allocated work 20%"
        job.update(phase_progress=0.5)
        browser._refresh_jobs()
        assert state.active_jobs[0]["progress"] == 50
        assert state.active_jobs[0]["progress_label"] == "Operation 50%"

        job.update(status="cancelling")
        browser._refresh_jobs()
        assert state.active_jobs[0]["progress"] is None
        job.update(status="cancelled")
        browser._refresh_jobs()
        assert not state.active_jobs
    finally:
        browser.plotter.close()
        session.close()


def test_single_runs_entry_reports_strict_fabrication_without_repeating_it_in_design_plan(tmp_path):
    from coilgen.application.session import PIPELINE_STAGES

    server, browser = create_app(tmp_path, server_name="test-web-finalizing-fabrication")
    session, state = browser.session, server.state
    try:
        browser.load_template("cylinder")
        assert state.trame__template_main.count("runs_dialog = true") == 1
        project = session.project
        job = {
            "id": "fabrication-run",
            "project_id": project.id,
            "project_revision": project.revision,
            "project_fingerprint": project.fingerprint,
            "pipeline_stage": "evaluated",
            "completed_stages": list(dict(PIPELINE_STAGES)),
            "status": "running",
            "fabricate": True,
            "progress_operation": "Strict fabrication review",
            "stage": "Build grooved support",
        }
        session.states[job["id"]] = job
        session.job_ids.append(job["id"])
        browser._refresh_jobs()
        assert state.active_jobs[0]["caption"] == "Finalizing fabrication"
        assert state.active_jobs[0]["stage"] == "Build grooved support"
        assert state.plan_channels[0]["activity"] == ""
        assert all(step["status"] == "Completed" for step in state.plan_channels[0]["steps"])

        job.update(progress_operation="Publish result", stage="Write archive")
        browser._refresh_jobs()
        assert state.active_jobs[0]["caption"] == "Computing"
        assert state.active_jobs[0]["stage"] == "Publish result · Write archive"

        job.update(status="completed", embedded=True)
        browser._refresh_jobs()
        assert not state.active_jobs
        assert state.jobs[0]["caption"] == "Completed" and state.jobs[0]["embedded"]
    finally:
        browser.plotter.close()
        session.close()


def test_batch_click_cannot_repeat_an_active_enabled_channel(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-web-batch-inflight")
    session, state = browser.session, server.state
    session.load_template("cylindrical_xyz_gradient")
    first, second, third = session.channels
    session.set_channel_enabled(second.id, False)
    session.set_channel_enabled(third.id, False)
    calls = []

    def submit(assembly, _jobs):
        included = [channel for channel in assembly.channels if channel.enabled]
        calls.append([c.id for c in included])
        return {
            c.id: {
                "id": "queued-" + c.id,
                "project_id": c.project.id,
                "project_revision": c.project.revision,
                "project_fingerprint": c.project.fingerprint,
                "pipeline_stage": c.project.study["pipeline_stage"],
                "status": "running",
            }
            for c in included
        }

    monkeypatch.setattr(session.service, "submit_assembly", submit)
    try:
        browser.compute_channels()
        assert state.batch_busy_channels == [first.name]
        browser.compute_channels()
        assert calls == [[first.id]] and "repeat" in state.message
        session.set_channel_enabled(first.id, False)
        session.set_channel_enabled(second.id, True)
        assert state.batch_busy_channels == []
        browser.compute_channels()
        assert calls == [[first.id], [second.id]]
        session.set_stage("prepared")
        assert state.batch_busy_channels == []
    finally:
        session.close()


def test_workspace_picker_reopens_only_a_listed_project(tmp_path, monkeypatch):
    server, workbench = create_app(tmp_path, server_name="test-workspace-picker")
    workbench.load_template("cylinder")
    original = workbench.session.project.id
    workbench.save_project()
    name = server.state.workspace_file
    assert name in server.state.saved_projects and name.endswith(".coilgen")
    project = workbench.session.project
    job = {
        "id": "only-run",
        "name": project.name,
        "project_id": project.id,
        "project_revision": project.revision,
        "project_fingerprint": project.fingerprint,
        "pipeline_stage": "prepared",
    }
    monkeypatch.setattr(workbench.session, "completed_runs", lambda: [job])
    server.state.workspace_history = True
    workbench.new_project()
    server.state.workspace_file = "../outside.coilgen"
    asyncio.run(workbench.open_workspace_project())
    assert workbench.session.project.id != original
    server.state.workspace_file = name
    asyncio.run(workbench.open_workspace_project())
    assert workbench.session.project.id == original
    assert server.state.selected_run == "only-run"


def test_worker_thread_count_is_explicit_or_derived_from_seats(tmp_path, monkeypatch):
    from coilgen.web.hub import Hub

    monkeypatch.setattr("coilgen.web.hub.os.cpu_count", lambda: 16)
    assert Hub(tmp_path, local_user="local", password=None, max_users=4).threads == 4
    assert Hub(tmp_path, local_user="local", password=None, max_users=4, threads_per_worker=2).threads == 2
    with pytest.raises(ValueError, match="at least one"):
        Hub(tmp_path, local_user="local", password=None, threads_per_worker=0)
