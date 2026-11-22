"""Shared fixtures for stopped checkpoint recovery."""

import json
import pytest
from coilgen.application import DesignService
from coilgen.application import ProjectDocument
from coilgen.progress import DesignCancelled
from coilgen.progress import progress_context
from coilgen.research.application import run_design
from coilgen.research.jobs import _record_completed_boundary
from _coilgen_test_support.pipeline_stages import config


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
