"""The browser shows the followed job's stage previews until a completed run opens."""

import asyncio

import pytest

pytest.importorskip("trame")

from coilgen.application.service import RunRecord
from coilgen.application.stage_preview import publish_stage_preview
from coilgen.web.app import create_app

from test_stage_previews import DIGEST, checkpoint, request


def _job(job_id="live-job"):
    value = request()
    return {
        "id": job_id,
        "name": "Owned study",
        "status": "running",
        "pipeline_stage": "routed",
        "active_stage": "contours",
        "request_sha256": DIGEST,
        "project_id": value.project_id,
        "project_fingerprint": value.project_fingerprint,
        "project_revision": value.project_revision,
    }


def _enabled(state):
    return {item["value"] for item in state.datasets if not item["props"]["disabled"]}


def test_followed_job_previews_switch_to_the_newest_stage_and_clear_when_it_ends(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-web-live-stage-preview")
    session, state = browser.session, server.state
    try:
        state.ready()  # Run the dataset watcher as a connected browser would.
        browser.load_template("cylinder")
        job = _job()
        session.states[job["id"]] = job
        session.job_ids.append(job["id"])
        browser.follow_job_id = job["id"]
        directory = session.jobs_directory / job["id"]

        asyncio.run(browser.live_preview.poll())
        assert browser.live_preview.preview is None and state.dataset == "support"

        publish_stage_preview(directory, checkpoint(tmp_path, "sheet"), request(), DIGEST)
        asyncio.run(browser.live_preview.poll())
        assert state.dataset == "stream_function"
        assert "stream_function" in _enabled(state) and "contours" not in _enabled(state)
        assert state.scene_caption.startswith("Live preview · Stream function [A] · not final")
        assert state.result_plot is None

        # An unchanged manifest is not reloaded.
        loaded = browser.live_preview.preview
        asyncio.run(browser.live_preview.poll())
        assert browser.live_preview.preview is loaded

        publish_stage_preview(directory, checkpoint(tmp_path, "contours"), request(), DIGEST)
        asyncio.run(browser.live_preview.poll())
        assert state.dataset == "contours"
        assert {"stream_function", "contours"} <= _enabled(state)
        assert state.scene_caption.startswith("Live preview · Contours before routing · not final")

        # A completed run, once open, takes precedence over the live preview.
        browser.record = RunRecord(tmp_path, {"name": "Owned study"}, dict(browser.live_preview.preview.arrays))
        browser._render()
        assert state.scene_caption.startswith("Saved run")
        browser.record = None

        # The run stops without a result: the preview is dropped.
        browser.follow_job_id = None
        asyncio.run(browser.live_preview.poll())
        assert browser.live_preview.preview is None
        assert state.dataset == "support"
        assert not state.scene_caption.startswith("Live preview")
    finally:
        browser.plotter.close()
        session.close()


def test_broken_preview_never_stops_polling(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-web-live-stage-preview-error")
    session, state = browser.session, server.state
    try:
        browser.load_template("cylinder")
        job = _job()
        session.states[job["id"]] = job
        browser.follow_job_id = job["id"]
        directory = session.jobs_directory / job["id"]
        directory.mkdir(parents=True)
        (directory / "stage_preview.json").write_text("{not json", encoding="utf-8")
        asyncio.run(browser.live_preview.poll())
        assert browser.live_preview.preview is None and browser.live_preview.error
        assert state.dataset == "support"
    finally:
        browser.plotter.close()
        session.close()
