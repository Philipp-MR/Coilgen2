"""Shared progress line and stage preview watch used by both workbenches."""

from datetime import datetime, timezone

from coilgen.application.live_preview import StagePreviewWatch, preferred_preview_dataset, stage_preview_request
from coilgen.application.run_progress import clock, run_progress
from coilgen.application.stage_preview import MANIFEST_NAME, load_stage_preview, publish_stage_preview

from test_stage_previews import DIGEST, checkpoint, request


NOW = datetime(2026, 11, 10, 12, 3, 10, tzinfo=timezone.utc)


def _job(**changes):
    return {
        "id": "job-1",
        "status": "running",
        "pipeline_stage": "evaluated",
        "active_stage": "routed",
        "started_at": "2026-11-10T12:00:00Z",
        "progress": 0.35,
        "progress_kind": "allocated_work",
        "progress_indeterminate": True,
        "phase_progress": None,
        **changes,
    }


def test_label_names_stage_and_elapsed_time_and_bar_keeps_one_scale():
    progress = run_progress(_job(), now=NOW)
    assert progress.label == "Stage 4/5 · Route / sweep · 3:10 elapsed"
    # Indeterminate routing: no percentage from the overall work estimate.
    assert progress.percent is None and progress.detail == ""
    measured = run_progress(_job(progress_indeterminate=False, phase_progress=0.42), now=NOW)
    assert measured.label == progress.label
    assert measured.percent == 42 and measured.detail == "Operation 42%"


def test_stage_count_follows_the_requested_boundary_and_completed_stages():
    assert run_progress(_job(pipeline_stage="contours", active_stage="sheet"), now=NOW).label.startswith(
        "Stage 2/3 · Stream function"
    )
    inferred = run_progress(_job(active_stage=None, completed_stage="sheet"), now=NOW)
    assert inferred.label.startswith("Stage 3/5 · Contours")
    unknown = run_progress(_job(active_stage=None, started_at=None), now=NOW)
    assert unknown.label == "Computing" and unknown.stage is None
    queued = run_progress(_job(status="queued", progress_indeterminate=False, phase_progress=0.5), now=NOW)
    assert queued.percent is None and "elapsed" not in queued.label


def test_clock_format():
    assert clock(190) == "3:10" and clock(5) == "0:05" and clock(3723) == "1:02:03"


def _state(job_id="job-1"):
    value = request()
    return {
        "id": job_id,
        "status": "running",
        "request_sha256": DIGEST,
        "project_id": value.project_id,
        "project_fingerprint": value.project_fingerprint,
        "project_revision": value.project_revision,
    }


def test_watch_reports_each_new_manifest_once_and_prefers_the_newest_stage(tmp_path):
    jobs = tmp_path / "jobs"
    directory = jobs / "job-1"
    watch = StagePreviewWatch(jobs)
    assert watch.changed(_state()) is None
    publish_stage_preview(directory, checkpoint(tmp_path, "sheet"), request(), DIGEST)
    first = watch.changed(_state())
    assert first is not None and first[0] == directory.resolve()
    assert watch.changed(_state()) is None
    preview = load_stage_preview(first[0], **first[1])
    assert preferred_preview_dataset(preview.arrays, preview.metadata) == "stream_function"

    publish_stage_preview(directory, checkpoint(tmp_path, "contours"), request(), DIGEST)
    second = watch.changed(_state())
    assert second is not None and second != first
    preview = load_stage_preview(second[0], **second[1])
    assert preferred_preview_dataset(preview.arrays, preview.metadata) == "contours"
    assert (directory / MANIFEST_NAME).exists()


def test_request_rejects_unsafe_job_ids_and_missing_identity(tmp_path):
    assert stage_preview_request(tmp_path, {**_state(), "id": "../escape"}) is None
    assert stage_preview_request(tmp_path, {**_state(), "request_sha256": "short"}) is None
    assert stage_preview_request(tmp_path, {**_state(), "project_id": " "}) is None
