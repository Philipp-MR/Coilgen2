"""Shared fixtures for native workbench."""

from __future__ import annotations
import os
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QWidget
from dataclasses import dataclass
from PySide6.QtCore import Signal
from coilgen.application import ProjectDocument
from coilgen.application.presets import EXAMPLE_TITLES
from coilgen.application.presets import preset_config as preset_config
from coilgen.application.presets import preset_study
from coilgen.gui.workbench import CoilGenWorkbench


class FakeJobs:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.states = []
        self.cancelled = []

    def list(self):
        return [dict(state) for state in self.states]

    def cancel(self, id_):
        self.cancelled.append(id_)
        self.states[0]["status"] = "cancelled"


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["coilgen-gui-test"])


class RenderlessPlotter(QWidget):
    """Exercise VTK datasets and Qt controls without an OpenGL context."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.interactor = self
        self.iren = object()
        self.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]
        self.plane_widgets = []
        self._plane_clipped_meshes = []
        self.meshes = []
        self.picked_point = None
        self.closed = False

    def __getattr__(self, name):
        if name in {
            "set_background",
            "enable_parallel_projection",
            "disable_parallel_projection",
            "add_axes",
            "show_grid",
            "render",
            "reset_camera",
            "view_isometric",
            "view_xy",
            "view_xz",
            "view_yz",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def enable_mesh_picking(self, **kwargs):
        self.pick = kwargs["callback"]

    def add_mesh(self, mesh, **kwargs):
        actor = object()
        self.meshes.append((mesh, kwargs, actor))
        return actor

    def add_mesh_clip_plane(self, mesh, **kwargs):
        self.plane_widgets.append(object())
        self.plane_clipped_meshes.append(mesh)

    @property
    def plane_clipped_meshes(self):
        return self._plane_clipped_meshes

    def clear_plane_widgets(self):
        self.plane_widgets.clear()

    def clear(self):
        self.meshes.clear()

    def close(self):
        self.closed = True

    def render(self):
        pass


@pytest.fixture
def scientific_viewport(application, monkeypatch):
    pytest.importorskip("pyvista")
    pyvistaqt = pytest.importorskip("pyvistaqt")
    monkeypatch.setattr(pyvistaqt, "QtInteractor", RenderlessPlotter)
    from coilgen.gui.viewport import ScientificViewport

    viewport = ScientificViewport(synchronous_preview=True)
    yield viewport
    viewport.close()
    viewport.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class FakeViewport(QWidget):
    selected = Signal(str)

    def render_config(self, config):
        self.config = config

    def display_run(self, arrays, metadata=None, *, follow_latest=False):
        self.arrays = arrays
        self.metadata = metadata


@dataclass
class FakeRun:
    directory: Path
    metadata: dict
    arrays: dict


class FakeService:
    def __init__(self):
        self.submissions = []
        self.record = None

    def submit(self, project, jobs, **controls):
        self.submissions.append((project, controls))
        state = {
            "id": "f" * 32,
            "name": project.name,
            "status": "queued",
            "stage": "Waiting for worker",
            "project_revision": project.revision,
            "project_fingerprint": project.fingerprint,
        }
        jobs.states.append(state)
        return dict(state)

    def load_run(self, path):
        return self.record


@pytest.fixture
def window(application, tmp_path):
    jobs = FakeJobs(tmp_path / "jobs")
    service = FakeService()
    project = ProjectDocument.from_config(preset_config(), name=EXAMPLE_TITLES["cylinder"]).update_study(preset_study())
    instance = CoilGenWorkbench(project, jobs=jobs, service=service, viewport_factory=FakeViewport)
    instance.timer.stop()
    yield instance
    instance.undo_stack.setClean()
    instance.close()
    instance.undo_stack.clear()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
