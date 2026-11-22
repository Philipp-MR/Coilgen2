"""Shared fixtures for workbench empty."""

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from coilgen.gui.workbench import CoilGenWorkbench
from _coilgen_test_support.workbench_usability import Preview
from _coilgen_test_support.workbench_usability import Queue
from _coilgen_test_support.workbench_usability import Service


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["workbench-empty-test"])


@pytest.fixture
def window(application, tmp_path):
    instance = CoilGenWorkbench(jobs=Queue(tmp_path / "jobs"), service=Service(), viewport_factory=Preview)
    instance.timer.stop()
    instance.show()
    application.processEvents()
    yield instance
    instance._draft_dirty = False
    instance.undo_stack.setClean()
    instance.close()
    instance.undo_stack.clear()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
