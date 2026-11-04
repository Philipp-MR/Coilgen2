"""Job requests retain the submitted numerical version across process boundaries."""

from dataclasses import asdict
from queue import Queue
from threading import Event

import pytest

from coilgen.application.presets import preset_config
from coilgen.persistence import atomic_json, numerical_source_digest, source_digest
from coilgen.research import application, jobs
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import request_identity


def test_browser_adapter_changes_only_full_source_provenance(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    adapter = web / "app.py"
    adapter.write_text("# Browser controls", encoding="utf-8")
    engine = tmp_path / "physics.py"
    engine.write_text("# Field operator", encoding="utf-8")
    full, numerical = source_digest(tmp_path), numerical_source_digest(tmp_path)
    adapter.write_text("# New browser controls", encoding="utf-8")
    assert source_digest(tmp_path) != full and numerical_source_digest(tmp_path) == numerical
    engine.write_text("# New field operator", encoding="utf-8")
    assert numerical_source_digest(tmp_path) != numerical


@pytest.mark.parametrize("when", ("before", "during"))
def test_worker_fails_explicitly_on_numerical_source_changes(monkeypatch, tmp_path, when):
    request = ResearchRequest.from_config(preset_config("cylinder"), pipeline_stage="prepared", fabricate=False)
    digest, manifest = request_identity(request.to_dict(), asdict(request.to_config(threads=1)))
    ran = []
    monkeypatch.setattr(application, "run_design", lambda *args, **kw: ran.append(True))
    monkeypatch.setattr(jobs, "numerical_source_digest", lambda _root: "changed")
    submitted_source = manifest["numerical_source_sha256"] if when == "during" else "changed"
    atomic_json(
        tmp_path / "job.json", {"request_sha256": digest, "submitted_numerical_source_sha256": submitted_source}
    )
    events = Queue()
    with pytest.raises(SystemExit) as exit_info:
        jobs._worker(request.to_dict(), str(tmp_path), events, Event(), 1)
    assert exit_info.value.code == 1
    terminal = [event for event in events.queue if event["kind"] in ("completed", "failed")]
    assert len(terminal) == 1 and terminal[0]["kind"] == "failed"
    assert "Numerical code changed" in terminal[0]["error"] and "Submit a new job" in terminal[0]["error"]
    assert bool(ran) is (when == "during")
