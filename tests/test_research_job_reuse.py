"""Opt-in queue reuse requires exact, durable request provenance and controls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading

import pytest

from coilgen.config import CoilGenConfig
from coilgen.research.jobs import JobManager, RuntimePolicy
from coilgen.research.requests import ResearchRequest


def request(**controls):
    return ResearchRequest.from_config(
        CoilGenConfig(coil_mesh_file="create cylinder mesh", sf_opt_method="tikhonov", research_physics=None),
        name="Identified request",
        **{
            "project_id": "channel-a",
            "project_fingerprint": "a" * 64,
            "project_revision": 3,
            **controls,
        },
    )


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    # These checks concern queue admission, not solver execution or IPC.
    monkeypatch.setattr(JobManager, "_monitor", lambda _self: None)
    with JobManager(tmp_path / "jobs", RuntimePolicy(max_pending=8)) as manager:
        yield manager


def job_directories(manager):
    return {path.name for path in manager.directory.iterdir() if path.is_dir()}


def test_controls_are_flat_and_durable_in_submitted_job_state(jobs):
    value = request(validation_resolution=17, design_margin=0.7, fabricate=True, kernel_backend="reference")
    state = jobs.submit(value)
    persisted = json.loads((jobs.directory / state["id"] / "job.json").read_text(encoding="utf-8"))
    for key in ("validation_resolution", "design_margin", "fabricate", "kernel_backend"):
        assert state[key] == persisted[key] == getattr(value, key)


@pytest.mark.parametrize("status", ["queued", "running", "cancelling"])
def test_full_queue_returns_matching_active_request_without_output_creation(jobs, status):
    jobs.policy = replace(jobs.policy, max_pending=1)
    first = jobs.submit(request())
    jobs._states[first["id"]]["status"] = status
    before = job_directories(jobs)
    reused = jobs.submit(request(), reuse_active=True)
    assert reused["id"] == first["id"] and reused["status"] == status
    assert job_directories(jobs) == before and len(jobs.list()) == 1
    reused["log"].append("caller edit")
    assert jobs.get(first["id"])["log"] == []


@pytest.mark.parametrize(
    "controls",
    [
        {"pipeline_stage": "sheet"},
        {"validation_resolution": 17},
        {"design_margin": 0.8},
        {"fabricate": True},
        {"kernel_backend": "reference"},
        {"project_id": "channel-b"},
        {"project_fingerprint": "b" * 64},
        {"project_revision": 4},
    ],
)
def test_control_owner_and_revision_differences_create_independent_requests(jobs, controls):
    first = jobs.submit(request())
    second = jobs.submit(request(**controls), reuse_active=True)
    assert second["id"] != first["id"] and len(jobs.list()) == 2


@pytest.mark.parametrize(
    "missing",
    [
        "project_id",
        "project_fingerprint",
        "project_revision",
        "pipeline_stage",
        "validation_resolution",
        "design_margin",
        "fabricate",
        "kernel_backend",
        "request_sha256",
    ],
)
def test_missing_legacy_identity_or_control_never_establishes_active_match(jobs, missing):
    first = jobs.submit(request())
    jobs._states[first["id"]].pop(missing)
    second = jobs.submit(request(), reuse_active=True)
    assert second["id"] != first["id"] and len(jobs.list()) == 2


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
def test_terminal_requests_are_not_active_queue_reuse(jobs, status):
    first = jobs.submit(request())
    jobs._states[first["id"]]["status"] = status
    assert jobs.submit(request(), reuse_active=True)["id"] != first["id"]


@pytest.mark.parametrize(
    "controls",
    [{"project_id": ""}, {"project_id": " "}, {"project_fingerprint": ""}, {"project_revision": None}],
)
def test_unproven_request_identity_remains_independent(jobs, controls):
    value = request(**controls)
    first = jobs.submit(value)
    assert jobs.submit(value, reuse_active=True)["id"] != first["id"]


def test_default_submit_preserves_independent_jobs_and_queue_limit(jobs):
    jobs.policy = replace(jobs.policy, max_pending=2)
    first = jobs.submit(request())
    second = jobs.submit(request())
    assert first["id"] != second["id"]
    before = job_directories(jobs)
    with pytest.raises(ValueError, match="queue is full"):
        jobs.submit(request())
    assert job_directories(jobs) == before


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_active_reuse_flag_requires_an_explicit_boolean(jobs, value):
    with pytest.raises(TypeError, match="reuse_active must be a boolean"):
        jobs.submit(request(), reuse_active=value)
    assert not jobs.list() and not job_directories(jobs)


def test_concurrent_same_request_admission_is_atomic_even_with_one_queue_slot(jobs):
    jobs.policy = replace(jobs.policy, max_pending=1)
    gate = threading.Barrier(2)

    def submit():
        gate.wait(timeout=5)
        return jobs.submit(request(), reuse_active=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0]["id"] == results[1]["id"]
    assert len(jobs.list()) == len(job_directories(jobs)) == 1


def test_reused_fingerprint_cannot_hide_different_native_numerical_settings(jobs):
    value = request()
    first = jobs.submit(value)
    changed = replace(value, config={**value.config, "target_gradient_strength": 0.025})
    assert jobs.submit(changed, reuse_active=True)["id"] != first["id"]


def test_changed_input_contents_cannot_hide_behind_the_same_project_fingerprint(jobs, tmp_path):
    source = tmp_path / "target.csv"
    source.write_text("x,y,z,bz\n0,0,0,0.00001\n", encoding="utf-8")
    value = request()
    value = replace(value, config={**value.config, "target_field_definition_file": str(source)})
    first = jobs.submit(value)
    source.write_text("x,y,z,bz\n0,0,0,0.00002\n", encoding="utf-8")
    assert jobs.submit(value, reuse_active=True)["id"] != first["id"]


def test_input_relocation_with_identical_contents_still_matches_owned_request(jobs, tmp_path):
    source = tmp_path / "target.csv"
    moved = tmp_path / "moved.csv"
    source.write_text("x,y,z,bz\n0,0,0,0.00001\n", encoding="utf-8")
    moved.write_bytes(source.read_bytes())
    value = request()
    value = replace(value, config={**value.config, "target_field_definition_file": str(source)})
    first = jobs.submit(value)
    moved_request = replace(value, config={**value.config, "target_field_definition_file": str(moved)})
    assert jobs.submit(moved_request, reuse_active=True)["id"] == first["id"]


def test_changed_resume_checkpoint_contents_never_reuses_active_request(jobs, tmp_path):
    # Admission owns and hashes checkpoint bytes; typed checkpoint loading is
    # the numerical worker's independent compatibility check.
    checkpoint = tmp_path / "resume.cgstage"
    checkpoint.write_bytes(b"checkpoint v1")
    value = request(resume_checkpoint=str(checkpoint))
    first = jobs.submit(value)
    checkpoint.write_bytes(b"checkpoint v2")
    assert jobs.submit(value, reuse_active=True)["id"] != first["id"]
