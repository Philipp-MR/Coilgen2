"""A real bounded worker preserves stage visualization and final scientific evidence."""

from dataclasses import replace
import json
import time

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectDocument, RunArchive
from coilgen.application.presets import preset_config
from coilgen.application.stage_preview import MANIFEST_NAME, load_stage_preview
from coilgen.pipeline_stages import PipelineStage
from coilgen.research.jobs import JobManager, RuntimePolicy


def test_full_owned_design_publishes_all_boundaries_and_preserves_final_project_results(tmp_path):
    config = replace(
        preset_config(),
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=8,
        target_error_tolerance_percent=5,
        levels=6,
        min_point_loop_number=3,
        gauss_order=1,
        skip_inductance_calculation=True,
    )
    project = ProjectDocument.from_config(config, name="Compact live Gy regression")
    runtime = tmp_path / "jobs"
    observed_boundaries, seen_preview_boundaries = [], set()
    # This verifies completion/evidence, not speed; hosted runners need more
    # than 90 seconds for the unchanged full numerical/fabrication pipeline.
    timeout_seconds = 180
    with JobManager(runtime, RuntimePolicy(timeout_seconds=timeout_seconds)) as jobs:
        submitted = DesignService.submit(
            project,
            jobs,
            pipeline_stage="evaluated",
            fabricate=False,
            validation_resolution=6,
            kernel_backend="reference",
        )
        identity = {
            "expected_request_sha256": submitted["request_sha256"],
            "expected_project_id": project.id,
            "expected_project_fingerprint": project.fingerprint,
            "expected_project_revision": project.revision,
        }
        directory = runtime / submitted["id"]
        deadline = time.monotonic() + timeout_seconds + 15
        latest = None
        while time.monotonic() < deadline:
            state = jobs.get(submitted["id"])
            completed = state.get("completed_stages", [])
            if completed:
                assert completed[: len(observed_boundaries)] == observed_boundaries
                observed_boundaries = completed
            if (directory / MANIFEST_NAME).exists():
                latest = load_stage_preview(directory, **identity)
                seen_preview_boundaries.add(latest.metadata["pipeline_stage"])
            if state["status"] not in {"queued", "running", "cancelling"}:
                break
            time.sleep(0.02)
        else:
            pytest.fail(f"Compact full worker design exceeded its bounded runtime: {state}")
        assert state["status"] == "completed", state
        expected = [stage.value for stage in PipelineStage]
        assert state["completed_stages"] == observed_boundaries == expected
        assert state["completed_stage"] == state["stage_preview_stage"] == "evaluated"
        assert "stage_preview_error" not in state
        assert latest is not None and latest.metadata["pipeline_stage"] == "evaluated"
        assert seen_preview_boundaries
        assert latest.metadata["target_field_component"] == "Bz"

        # All stage files exist even if an inexpensive phase finishes between polls.
        for stage in PipelineStage:
            snapshot = directory / "stage_previews" / f"{stage.value}.npz"
            assert snapshot.is_file()
            with np.load(snapshot, allow_pickle=False) as data:
                assert {"vertices_m", "faces", "target_field_T", "target_points_m"} <= set(data.files)
                if stage.index >= PipelineStage.SHEET.index:
                    assert {"stream_function_A", "sheet_field_T"} <= set(data.files)
                if stage.index >= PipelineStage.CONTOURS.index:
                    assert any(key.startswith("contour_") for key in data.files)
                if stage.index >= PipelineStage.ROUTED.index:
                    assert any(key.startswith("route_") for key in data.files)
                    assert any(key.startswith("swept_") and key.endswith("_vertices_m") for key in data.files)

        # Completion still requires and loads the immutable final evidence.
        final = jobs.result(submitted["id"])
        record = DesignService.load_run(directory)
        final_arrays = record.arrays
        for key, array in latest.arrays.items():
            np.testing.assert_array_equal(array, final_arrays[key])
        assert record.metadata["request_sha256"] == submitted["request_sha256"]
        assert final["pipeline_stage"] == record.metadata["pipeline_stage"] == "evaluated"
        assert record.metadata.get("source") != "live_stage_preview"
        assert not any(
            artifact["name"] == MANIFEST_NAME or artifact["name"].startswith("stage_previews/")
            for artifact in final["artifacts"]
        )
        archive = RunArchive.capture(record)
        assert MANIFEST_NAME not in archive.to_dict()["files"]
        assert not any(name.startswith("stage_previews/") for name in archive.to_dict()["files"])
        saved = project.with_results((archive,)).save(tmp_path / "completed.coilgen.json")
        loaded = ProjectDocument.load(saved)
        assert len(loaded.saved_runs) == 1
        assert loaded.saved_runs[0].metadata["request_sha256"] == record.metadata["request_sha256"]
        results = json.loads(saved.read_text(encoding="utf-8"))["results"]
        assert not any(name.startswith("stage_previews/") for name in results[0]["files"])
