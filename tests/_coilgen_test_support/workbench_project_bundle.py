"""Shared fixtures for workbench project bundle."""

import pytest

pytest.importorskip("PySide6")
from coilgen.application import DesignService
from coilgen.gui.workbench import CoilGenWorkbench
from _coilgen_test_support import native_workbench as native


application = native.application


def _window(project, jobs_directory):
    window = CoilGenWorkbench(
        project=project,
        jobs=native.FakeJobs(jobs_directory),
        service=DesignService(),
        viewport_factory=native.FakeViewport,
    )
    window.timer.stop()
    return window


def _close(window):
    window.undo_stack.setClean()
    window.close()
