"""Shared fixtures for expert settings."""

from copy import deepcopy
import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.expert_settings import ExpertSettingsDialog


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["expert-settings-test"])


@pytest.fixture
def dialogs(application):
    created = []

    def make(config=None, callback=None):
        document = ProjectDocument.from_config(config or preset_config())
        state = {"document": document, "calls": []}

        def commit(changes, label):
            state["calls"].append((deepcopy(changes), label))
            state["document"] = state["document"].update_config(changes)
            return True

        dialog = ExpertSettingsDialog(config or preset_config(), callback or commit)
        created.append(dialog)
        return dialog, state

    yield make
    for dialog in created:
        dialog.reject()
        dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
