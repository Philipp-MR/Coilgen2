"""Scientific parity, checkpoint ownership and bounded worker stage execution."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pytest

from coilgen import CoilGenConfig, run_coilgen
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, invalidated_stage
from coilgen.persistence import numerical_source_digest
from coilgen.research.application import run_design
from coilgen.research.jobs import JobManager
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import request_identity


def config():
    return CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=8,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power"},
        constrained_quadratic_fallback_policy="strict",
        target_error_tolerance_percent=5,
        stream_function_smoothness_factor=0,
        turn_spacing_regularization_factor=0,
        levels=6,
        min_point_loop_number=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        progress_flag=False,
        plot_flag=False,
    )


def wait(jobs, id_):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        value = jobs.get(id_)
        if value["status"] not in {"queued", "running", "cancelling"}:
            return value
        time.sleep(0.05)
    pytest.fail("Stage worker did not finish.")


def test_stage_save_resume_has_same_scientific_outputs_as_script(tmp_path):
    original = config()
    expected = run_coilgen(original)
    checkpoint = None
    for stage in PipelineStage:
        checkpoint = CoilGenPipeline(original).run_stage(stage, checkpoint)
        path = checkpoint.save(tmp_path / f"{stage.value}.cgstage")
        checkpoint = PipelineCheckpoint.load(path)
    actual = checkpoint.result
    for name in ("b_field_opt_sf", "field_by_unconnected_loops", "field_by_layout", "resistance", "inductance_matrix"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        actual.combined_mesh.stream_function, expected.combined_mesh.stream_function, rtol=1e-12, atol=1e-15
    )
    for actual_part, expected_part in zip(actual.coil_parts, expected.coil_parts):
        assert len(actual_part.contour_lines) == len(expected_part.contour_lines)
        for first, second in zip(actual_part.contour_lines, expected_part.contour_lines):
            np.testing.assert_allclose(first.v, second.v, rtol=1e-12, atol=1e-15)
    original.research_physics["objective"] = "minimax"
    assert actual.input_data.research_physics["objective"] == "power"
    detached = checkpoint.state
    detached.parts[0].stream_function[:] = 999
    assert not np.all(checkpoint.state.parts[0].stream_function == 999)


def test_stage_reuse_respects_material_contour_and_hardware_coupling():
    before = config()
    prepared = CoilGenPipeline(before).run_stage("prepared")
    sheet = CoilGenPipeline(before).run_stage("sheet", prepared)
    after = replace(before, levels=8)
    assert invalidated_stage(before, after) is PipelineStage.CONTOURS
    changed = CoilGenPipeline(after).run_stage("contours", sheet)
    np.testing.assert_array_equal(changed.sf_b_field, sheet.sf_b_field)
    with pytest.raises(ValueError, match="Cannot reuse sheet"):
        CoilGenPipeline(replace(before, specific_conductivity_conductor=3e-8)).run_stage("sheet", sheet)
    updated = CoilGenPipeline(replace(before, specific_conductivity_conductor=3e-8)).run_stage("sheet", prepared)
    assert updated.stage is PipelineStage.SHEET
    linked = replace(before, max_wire_current_A=30)
    assert invalidated_stage(linked, replace(linked, levels=8)) is PipelineStage.SHEET
    with pytest.raises(ValueError, match="rewound"):
        CoilGenPipeline(before).run_stage("prepared", sheet)


def test_contour_manufacturing_and_resampling_modes_invalidate_contours():
    before = config()
    assert invalidated_stage(before, replace(before, skip_postprocessing=False)) is PipelineStage.CONTOURS
    manufacturing = replace(
        before, contour_only_sweep_mode=True, skip_sweep=False, contour_sparsification_mode="field_fit"
    )
    assert (
        invalidated_stage(manufacturing, replace(manufacturing, sweep_min_bend_radius=0.002)) is PipelineStage.CONTOURS
    )


def test_numerical_source_identity_tracks_engine_and_excludes_presentation(tmp_path):
    (tmp_path / "gui").mkdir()
    ui = tmp_path / "gui" / "workbench.py"
    engine = tmp_path / "optimization.py"
    credits = tmp_path / "attribution.py"
    ui.write_text("theme = 'light'\n")
    engine.write_text("objective = 'power'\n")
    credits.write_text("AUTHOR_NAME = 'Philipp Amrein'\n")
    baseline = numerical_source_digest(tmp_path)
    ui.write_text("theme = 'dark'\n")
    credits.write_text("AUTHOR_NAME = 'Updated software credits'\n")
    assert numerical_source_digest(tmp_path) == baseline
    engine.write_text("objective = 'minimax'\n")
    assert numerical_source_digest(tmp_path) != baseline


def test_changed_input_content_prevents_checkpoint_reuse(tmp_path):
    target = tmp_path / "target.csv"
    target.write_text("x,y,z,bz\n0,0,.01,1e-5\n0,.01,.02,2e-5\n", encoding="utf-8")
    imported = replace(config(), target_field_definition_file=str(target), target_gradient_normalization="off")
    checkpoint = CoilGenPipeline(imported).run_stage("prepared")
    target.write_text("x,y,z,bz\n0,0,.01,9e-5\n0,.01,.02,2e-5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot reuse prepared"):
        CoilGenPipeline(imported).run_stage("sheet", checkpoint)


def test_checkpoint_manifest_and_arrays_are_integrity_checked(tmp_path):
    path = CoilGenPipeline(config()).run_stage("prepared").save(tmp_path / "state.cgstage")
    with zipfile.ZipFile(path) as source:
        manifest = json.loads(source.read("manifest.json"))
        data = source.read("arrays.npz")
    manifest["stage"] = "sheet"
    with zipfile.ZipFile(path, "w") as destination:
        destination.writestr("manifest.json", json.dumps(manifest))
        destination.writestr("arrays.npz", data)
    with pytest.raises(ValueError, match="manifest.*integrity"):
        PipelineCheckpoint.load(path)


def test_stage_request_and_resume_keep_geometry_scope_explicit(tmp_path):
    request = ResearchRequest.from_config(
        config(), pipeline_stage="prepared", project_id="owned", project_revision=3, project_fingerprint="model-study"
    )
    prepared = run_design(request, tmp_path / "prepared")
    assert prepared["summary"]["scope"] == "prepared_geometry"
    assert prepared["summary"]["accepted"] is False
    assert prepared["project_id"] == "owned" and prepared["project_revision"] == 3
    checkpoint = tmp_path / "prepared" / prepared["checkpoint"]
    solved = run_design(replace(request, pipeline_stage="sheet", resume_checkpoint=str(checkpoint)), tmp_path / "sheet")
    assert solved["summary"]["scope"] == "continuous_sheet"
    assert solved["validation"]["available"]
    assert solved["pipeline_stage"] == "sheet"
    with np.load(tmp_path / "sheet" / "scientific_result.npz", allow_pickle=False) as data:
        assert {"target_suppressed", "target_weights", "sheet_field_T", "stream_function_A"} <= set(data.files)


def test_completed_worker_checkpoint_can_resume_after_runtime_restart(tmp_path):
    request = ResearchRequest.from_config(config(), pipeline_stage="prepared", project_id="project", project_revision=2)
    with JobManager(tmp_path) as jobs:
        submitted = jobs.submit(request)
        finished = wait(jobs, submitted["id"])
        assert finished["status"] == "completed", finished
        assert jobs.get(submitted["id"])["project_id"] == "project"
        checkpoint = jobs.artifact(submitted["id"], jobs.result(submitted["id"])["checkpoint"])
    with JobManager(tmp_path) as jobs:
        resumed = jobs.submit(replace(request, pipeline_stage="sheet", resume_checkpoint=str(checkpoint)))
        finished = wait(jobs, resumed["id"])
        assert finished["status"] == "completed", finished
        result = jobs.result(resumed["id"])
        assert result["pipeline_stage"] == "sheet" and result["summary"]["scope"] == "continuous_sheet"
        (tmp_path / resumed["id"] / "scientific_result.npz").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="integrity"):
            jobs.result(resumed["id"])


def test_queued_inputs_are_owned_and_numerical_identity_survives_relocation(tmp_path):
    original = tmp_path / "target.csv"
    original.write_text("x,y,z,bz\n0,0,.01,1e-5\n", encoding="utf-8")
    first = ResearchRequest.from_config(replace(config(), target_field_definition_file=str(original)))
    with JobManager(tmp_path / "jobs") as jobs:
        with jobs._lock:
            submitted = jobs.submit(first)
            snapshot = json.loads(jobs.artifact(submitted["id"], "request.json").read_text())
            copied = Path(snapshot["config"]["target_field_definition_file"])
            assert copied != original and copied.is_relative_to(jobs.directory)
            digest = request_identity(first.to_dict(), asdict(first.to_config()))[0]
            assert digest == submitted["request_sha256"]
            display = replace(first.to_config(), output_directory=tmp_path / "other-output", plot_flag=True)
            assert request_identity(first.to_dict(), asdict(display))[0] == digest
            original.write_text("changed", encoding="utf-8")
            assert copied.read_text().endswith("1e-5\n")
            jobs.cancel(submitted["id"])


def test_broken_worker_progress_channel_does_not_stop_job_supervision(tmp_path):
    request = ResearchRequest.from_config(config(), pipeline_stage="prepared")
    with JobManager(tmp_path) as jobs:
        submitted = jobs.submit(request)
        deadline = time.monotonic() + 10
        while jobs.get(submitted["id"])["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.02)
        with jobs._lock:
            process, original_events, cancel, started = jobs._active[submitted["id"]]

            class BrokenEvents:
                def get_nowait(self):
                    raise EOFError("simulated queue feeder failure")

                def close(self):
                    original_events.close()

            jobs._active[submitted["id"]] = process, BrokenEvents(), cancel, started
        completed = wait(jobs, submitted["id"])
        assert completed["status"] == "completed", completed
        assert "simulated" in completed["ipc_error"] and jobs._thread.is_alive()
        next_job = jobs.submit(request)
        assert wait(jobs, next_job["id"])["status"] == "completed"
