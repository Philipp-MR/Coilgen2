"""Stopped jobs recover numerical boundaries without claiming final evidence."""

import json

import pytest

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.reuse import plan_stopped_reuse
from coilgen.pipeline_stages import PipelineStage
from coilgen.progress import DesignCancelled, progress_context
from coilgen.research.application import run_design
from coilgen.research import application as research
from coilgen.research.jobs import _record_completed_boundary
from test_pipeline_stages import config


@pytest.fixture
def stopped(tmp_path):
    project = ProjectDocument.from_config(config()).update_study({"pipeline_stage": "contours"})
    root = tmp_path / "jobs"
    directory = root / "stopped-sheet"
    submitted = DesignService.prepare(project, directory, pipeline_stage="contours")
    state = {
        "id": directory.name,
        "status": "cancelled",
        **{
            key: getattr(submitted, key)
            for key in (
                "project_id",
                "project_revision",
                "project_fingerprint",
                "project_model_fingerprint",
                "pipeline_stage",
            )
        },
    }

    def stop_after_sheet(event):
        _record_completed_boundary(state, event)
        if event.get("completed_stage") == "sheet":
            raise DesignCancelled("Stopped after a retained sheet boundary")

    with progress_context(stop_after_sheet), pytest.raises(DesignCancelled):
        run_design(submitted, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    state["request_sha256"] = manifest["request_sha256"]
    return project, state, root, directory


def test_stopped_job_has_verified_recovery_without_saved_result(stopped):
    project, state, root, directory = stopped
    assert not (directory / "result.json").exists()
    plan = plan_stopped_reuse(project, [state], root)
    assert plan.through("evaluated").stage is PipelineStage.SHEET
    assert plan.through("prepared").stage is PipelineStage.PREPARED
    assert not plan.diagnostics
    changed = project.update_config({"levels": 9})
    assert plan_stopped_reuse(changed, [state], root).through("contours").stage is PipelineStage.SHEET
    unrelated = ProjectDocument.from_config(project.config)
    assert not plan_stopped_reuse(unrelated, [state], root).available


def test_corrupt_later_checkpoint_retains_verified_earlier_recovery(stopped):
    project, state, root, directory = stopped
    path = directory / "checkpoints/sheet.cgstage"
    path.write_bytes(path.read_bytes() + b"changed")
    plan = plan_stopped_reuse(project, [state], root)
    assert plan.through("evaluated").stage is PipelineStage.PREPARED
    assert any("integrity" in detail for detail in plan.diagnostics)


def test_optional_recovery_metadata_failure_preserves_verified_result(tmp_path, monkeypatch):
    project = ProjectDocument.from_config(config()).update_study({"pipeline_stage": "prepared"})
    directory = tmp_path / "prepared"
    submitted = DesignService.prepare(project, directory, pipeline_stage="prepared")
    write_json = research.atomic_json

    def fail_recovery_metadata(path, value):
        if path.name == "evidence.json":
            raise OSError("Recovery metadata storage unavailable")
        return write_json(path, value)

    monkeypatch.setattr(research, "atomic_json", fail_recovery_metadata)
    events = []
    with progress_context(events.append):
        result = run_design(submitted, directory)
    assert result["pipeline_stage"] == "prepared"
    assert DesignService.load_run(directory).metadata["pipeline_stage"] == "prepared"
    assert (directory / "checkpoints/prepared.cgstage").is_file()
    assert not (directory / "checkpoints/evidence.json").exists()
    assert any("recovery metadata unavailable" in event["message"] for event in events)


@pytest.mark.parametrize("change", ["request", "digest", "path", "unrecorded", "legacy"])
def test_unverified_or_foreign_recovery_is_unavailable(stopped, change):
    project, state, root, directory = stopped
    if change == "request":
        path = directory / "request.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["project_id"] = "another-channel"
    elif change == "digest":
        path = directory / "checkpoints/evidence.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["request_sha256"] = "b" * 64
    elif change == "path":
        path = directory / "checkpoints/evidence.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["artifacts"][0]["name"] = "../outside.cgstage"
    elif change == "legacy":
        (directory / "checkpoints/evidence.json").unlink()
    else:
        state["completed_stages"] = []
    if change in {"request", "digest", "path"}:
        path.write_text(json.dumps(value), encoding="utf-8")
    plan = plan_stopped_reuse(project, [state], root)
    assert not plan.available
    if change != "unrecorded":
        assert plan.diagnostics
