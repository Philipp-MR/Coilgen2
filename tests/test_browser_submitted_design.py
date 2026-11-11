import asyncio
import json
import time

import pytest

pytest.importorskip("trame")

from coilgen.application.service import DesignService
from coilgen.web.app import create_app


def test_plot_caption_only_counts_known_contour_outcomes_across_supports():
    from coilgen.application.submitted_design import contour_fit_caption

    parts = [
        {"smoothed_count": 4, "fit_rejected_count": 1, "spacing_rejected_count": 1, "local_fit_count": 2},
        {"smoothed_count": 3, "fit_rejected_count": 0, "spacing_rejected_count": 2, "local_fit_count": 0},
    ]
    caption = contour_fit_caption({"contour_smoothing": parts})
    for expected in ("7 of 11", "2 using local corner blends", "1 kept after fit checks", "3 kept for spacing"):
        assert expected in caption
    parts[0].pop("local_fit_count")
    assert "local" not in contour_fit_caption({"contour_smoothing": parts})
    parts[0].pop("smoothed_count")
    assert not contour_fit_caption({"contour_smoothing": parts})
    assert not contour_fit_caption({})


def test_review_uses_original_request_after_edit_and_never_substitutes_on_error(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-web-submitted-choices")
    state, session = server.state, browser.session
    browser.load_template("shim_z0")
    project = session.set_stage("prepared")
    directory = tmp_path / "prepared"
    DesignService.run(project, directory, threads=1, **session.controls("prepared"))
    browser.record = DesignService.load_run(directory)
    session.apply_settings({"contour_treatment": "off", "cylinder_radius_mm": 95})
    assert session.project.config.contour_path_regularization_mode == "off"
    asyncio.run(browser.review_submitted_design())
    review = state.design_review
    assert review["error"] == ""
    assert json.loads(review["request_json"]) == json.loads((directory / "request.json").read_text())
    values = {row["name"]: row["value"] for group in review["groups"] for row in group["rows"]}
    assert values["Radius"] == "80 mm" and values["Length"] == "200 mm"
    assert values["Contour path treatment"] == "Surface B-spline smoothing"
    assert values["Maximum allowed displacement"] == "0.5 mm"
    assert values["Minimum centerline spacing"] == "No specified minimum (0 mm)"
    assert "preserve the extracted minimum spacing" in values["Guard inter-turn spacing"]
    assert json.loads(review["request_json"])["config"]["min_turn_distance"] == 0
    assert values["Support surface"] == "Cylinder" and values["Design context"] == "MRI shim"
    assert (
        values["Stream-function method"] == "Constrained quadratic solve"
        and values["Conductor side of support"] == "Outer side (along normal)"
    )
    assert values["Smoothing scale"] == "Local mesh wavelength"
    assert values["Requested boundary"] == "Prepare model" and values["Fabrication review requested"] == "No"
    monkeypatch.setattr("coilgen.web.app.INLINE_REQUEST_BYTES", 512)
    asyncio.run(browser.review_submitted_design())
    large = state.design_review
    assert not large["request_json"] and large["groups"] == review["groups"]
    assert json.loads((browser.exports / large["download"].split("/")[-1]).read_text(encoding="utf-8")) == json.loads(
        review["request_json"]
    )
    (directory / "request.json").write_text("{}")
    asyncio.run(browser.review_submitted_design())
    assert (
        state.design_review["error"] and not state.design_review["request_json"] and not state.design_review["groups"]
    )
    browser.new_project()
    assert state.design_review is None and state.review_dialog is False


def test_winding_controls_preserve_saved_aliases_and_custom_smoothing_values(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-web-smoothing-controls")
    state, session = server.state, browser.session
    browser.load_template("shim_z0")
    session.apply_config(
        {
            "contour_only_sweep_mode": True,
            "skip_postprocessing": True,
            "contour_path_regularization_mode": "smooth",
            "contour_smoothing_scale_mode": "mesh_coupled",
            "contour_smoothing_mesh_gain": 0.25,
            "contour_smoothing_max_displacement": 0.0002,
        }
    )
    state.node = "winding"
    browser._refresh_settings()
    original = session.project
    assert state.settings_values["contour_treatment"] == "mesh_coupled"
    browser.apply_settings()
    assert session.project is original
    state.settings_values = {**state.settings_values, "contour_treatment": "off"}
    browser.apply_settings()
    assert session.project.config.contour_path_regularization_mode == "off"
    state.settings_values = {**state.settings_values, "contour_treatment": "mesh_coupled"}
    browser.apply_settings()
    assert session.project.config.contour_path_regularization_mode == "parameterized_bspline"
    assert session.project.config.contour_smoothing_mesh_gain == 0.25
    assert session.project.config.contour_smoothing_max_displacement == pytest.approx(0.0002)
    assert "interconnection_axis" not in session.settings_values("winding")
    assert "normal_shift_mm" not in session.settings_values("conductor")
    assert any(issue["informational"] and "independent closed turns" in issue["title"] for issue in state.preflight)


def test_measured_smoothing_displacement_uses_readable_units_including_zero():
    from coilgen.application.submitted_design import design_choice_groups, displacement_label

    assert [displacement_label(value) for value in (None, float("nan"), 0, 9.73e-9, 1e-6, 5e-4, 1e-3)] == [
        "Unrecorded",
        "Unrecorded",
        "0 mm",
        "9.73 nm",
        "1 µm",
        "500 µm",
        "1 mm",
    ]
    groups = design_choice_groups(
        {
            "config": {
                "min_turn_distance": 0.002,
                "contour_path_regularization_mode": "parameterized_bspline",
                "contour_smoothing_preserve_min_turn_distance": False,
            }
        }
    )
    rows = {row["name"]: row["value"] for group in groups for row in group["rows"]}
    assert rows["Minimum centerline spacing"] == "2 mm" and rows["Guard inter-turn spacing"] == "No"


def test_halbach_rotation_axis_is_editable_and_atomic(tmp_path):
    _, browser = create_app(tmp_path, server_name="test-web-halb-axis")
    session = browser.session
    session.load_template("halbach_gz")
    original = session.project
    values = session.settings_values("geometry")
    axis = tuple(values[f"cylinder_rotation_axis_{a}"] for a in "xyz")
    assert axis == (1, 0, 0) and abs(values["cylinder_rotation_deg"]) == pytest.approx(90)
    with pytest.raises(ValueError, match="axis must be nonzero"):
        session.apply_settings({"cylinder_rotation_axis_x": 0})
    with pytest.raises(ValueError, match="finite"):
        session.apply_settings({"cylinder_rotation_axis_x": float("nan")})
    assert session.project is original
    updated = session.apply_settings({"cylinder_rotation_axis_x": 0, "cylinder_rotation_axis_z": 1})
    assert tuple(updated.config.cylinder_mesh_parameter_list[4:7]) == (0, 0, 1)


def test_named_regional_target_is_the_source_for_preview_and_design_review(tmp_path):
    from coilgen.application.gradient_studies import MeasuredFieldMap
    from coilgen.application.research_requirements import DesiredDataset, RegionalRequirement, ResearchRequirements
    from coilgen.application.submitted_design import design_choice_groups

    server, browser = create_app(tmp_path, server_name="test-web-named-region")
    browser.load_template("shim_z0")
    expression = browser.session.project.config.field_shape_function
    desired = DesiredDataset(MeasuredFieldMap([[0, 0, 0]], [[0, 0, 0.002]]))
    requirements = ResearchRequirements((RegionalRequirement("Imaging ROI", desired, rms_limit_T=4e-6),)).to_dict()
    project = browser.session.apply_inputs({}, {"requirements": requirements})
    assert project.values["field_shape_function"] == expression
    server.state.dataset = "preview_target"
    browser._render()
    assert "Bz [µT] 2000" in server.state.scene_caption
    groups = design_choice_groups({"config": project.values, **project.study})
    rows = {row["name"]: row["value"] for group in groups for row in group["rows"]}
    assert rows["Imaging ROI"] == "Bz · RMS limit 4 µT"
    assert "Target expression (world x, y, z in metres)" not in rows and "Target region" not in rows


def test_native_selection_survives_browser_relocation_and_local_job_deduplication(tmp_path):
    from coilgen.application.session_evidence import load_assembly

    source_state, local = create_app(tmp_path / "original", server_name="test-web-selection-local")
    local.load_template("cylinder")
    session = local.session
    try:
        session.set_stage("prepared")
        jobs = []
        records = []
        for levels in (5, 7):
            session.apply_settings({"levels": levels})
            job = session.compute()
            jobs.append(job["id"])
            deadline = time.monotonic() + 90
            while session.active() and time.monotonic() < deadline:
                time.sleep(0.1)
                session.poll()
            records.append(session.load_run(job["id"]))
        session.save_project(format="bundle", selected_run=records[0])
        project = session.project
        archived = {run.metadata["project_revision"]: run.id for run in project.saved_runs}
        earlier = archived[records[0].metadata_view["project_revision"]]
        current = archived[records[1].metadata_view["project_revision"]]
        # Native single-design saves carry this archive ID even when the applied revision is newer.
        native = project.with_results(project.saved_runs, {"selected_run_id": earlier}).save(
            tmp_path / "native.coilgen"
        )
        local._open_document(native)
        assert source_state.state.selected_run == jobs[0]
        assert len(session.completed_runs()) == 2
        local._open_run(jobs[1])
        local.save_project()
        browser_save = session.project_path
        saved = load_assembly(browser_save).channels[0].project
        assert saved.session["selected_run_id"] == current
        assert any(run.id == current and run.project_id == project.id for run in saved.saved_runs)

        destination_state, relocated = create_app(tmp_path / "fresh", server_name="test-web-selection-fresh")
        relocated._open_document(native)
        assert destination_state.state.selected_run == earlier
        relocated._open_run(earlier)
        assert relocated.server.state.run_context.startswith("Earlier")
        relocated._open_document(browser_save)
        assert destination_state.state.selected_run == current

        # Invalid or foreign selections must not select a random historical archive.
        changed = project.update_inputs({"levels": 9}).with_results(project.saved_runs, {"selected_run_id": jobs[0]})
        relocated._open_document(changed.save(tmp_path / "bad-selection.coilgen"))
        assert destination_state.state.selected_run is None
        fallback = project.with_results(project.saved_runs, {"selected_run_id": "not-an-archive"})
        relocated._open_document(fallback.save(tmp_path / "fallback.coilgen"))
        assert destination_state.state.selected_run == current
        relocated.new_project()
        relocated.server.state.workspace_history = True
        relocated._open_run(earlier)
        relocated.save_project()
        own = load_assembly(relocated.session.project_path).channels[0].project
        assert not own.saved_runs and own.session["selected_run_id"] is None
    finally:
        session.close()
