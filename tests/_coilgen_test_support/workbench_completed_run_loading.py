"""Shared fixtures for workbench completed run loading."""

import time
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication


def wait(predicate):
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)  # Yield the GIL to the Python archive/hash worker.
    assert predicate()
