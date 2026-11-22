"""Shared fixtures for gui interconnection editor."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget
from coilgen.application.interconnection_editor import WindingEditorSession
from coilgen.models import TargetField
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, PipelineState
from _coilgen_test_support.interconnection_editor import routed as routed
from _coilgen_test_support.workbench_usability import application as application


class Viewport(QWidget):
    selected = Signal(str)
    cut_moved = Signal(str, object)
    guide_moved = Signal(str, int, object)
    message = Signal(str)

    def display(self, editor, key, **options):
        self.displayed = (editor.fingerprint, key)


def session_from_case(case, tmp_path):
    coords = np.array([[0, 0, 0.025], [0.003, 0, 0.025]])
    target = TargetField(coords, np.full_like(coords, 1e-6), np.ones(2), np.zeros(2, bool), np.zeros(2, int))
    state = PipelineState(parts=[case.part], target_field=target, combined_mesh=case.part.mesh, sf_b_field=target.b)
    return WindingEditorSession(PipelineCheckpoint(PipelineStage.ROUTED, state, case.config, tmp_path))


def finish(application, dialog):
    deadline = time.monotonic() + 15
    while dialog._busy and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert not dialog._busy, "Editor worker did not finish"
    application.processEvents()


def close(dialog):
    dialog._saved_identity = dialog.session.identity
    dialog.close()
