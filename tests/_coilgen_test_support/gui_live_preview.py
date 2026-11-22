"""Shared fixtures for gui live preview."""

import time
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QEvent
from PySide6.QtTest import QTest
from _coilgen_test_support import native_workbench as native
from coilgen.application.stage_preview import MANIFEST_NAME
from coilgen.application.stage_preview import StagePreview
from coilgen.gui.live_preview import LivePreviewController


application = native.application


def _state(job="a" * 32, **changes):
    return {
        "id": job,
        "status": "running",
        "stage": "Preparing model",
        "progress": 0.2,
        "request_sha256": "b" * 64,
        "project_id": "c" * 32,
        "project_fingerprint": "d" * 64,
        "project_revision": 3,
        **changes,
    }


def _write_manifest(jobs, state, content="prepared"):
    directory = jobs / state["id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_text(content, encoding="utf-8")
    return directory


def _preview(directory, stage="prepared"):
    return StagePreview(directory, {"pipeline_stage": stage}, {"vertices_m": np.array([[0.0, 0.0, 0.0]])})


def _wait(application, predicate, message="Background preview did not finish", timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    pytest.fail(message)


@pytest.fixture
def controllers(application, tmp_path):
    created, releases = [], []
    jobs = tmp_path / "jobs"

    def create(loader):
        controller = LivePreviewController(jobs, loader=loader)
        created.append(controller)
        return controller

    yield jobs, create, releases
    for release in releases:
        release.set()
    for controller in created:
        controller.close()
    _wait(application, lambda: all(controller._worker is None for controller in created))
    for controller in created:
        controller.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
