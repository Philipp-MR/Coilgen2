"""Shared fixtures for workbench usability."""

from pathlib import Path
from types import SimpleNamespace
import os
from uuid import uuid4
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QEvent
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QWidget
from coilgen.application import DesignService
from coilgen.application import ProjectDocument
from coilgen.application.presets import EXAMPLE_TITLES
from coilgen.application.presets import preset_config
from coilgen.application.presets import preset_study
from coilgen.gui.workbench import CoilGenWorkbench


class Preview(QWidget):
    selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.displayed = None
        self.state = {"dataset": "support"}
        self.clear_count = 0

    def render_project(self, project):
        self.project = project

    def display_run(self, arrays, metadata=None, *, follow_latest=False):
        self.displayed = metadata

    def clear_run(self):
        self.displayed = None
        self.clear_count += 1

    def view_state(self):
        return dict(self.state)

    def restore_view_state(self, state):
        self.state = dict(state)


class Queue:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.policy = SimpleNamespace(threads_per_worker=1)
        self.states = []

    def list(self):
        return [dict(state) for state in self.states]


class Service:
    def __init__(self):
        self.submissions = []

    def submit(self, project, jobs, **controls):
        self.submissions.append((project, controls))
        state = {
            "id": uuid4().hex,
            "name": project.name,
            "status": "queued",
            "pipeline_stage": controls["pipeline_stage"],
            "project_id": project.id,
            "project_fingerprint": project.fingerprint,
            "project_model_fingerprint": project.model_fingerprint,
            "project_revision": project.revision,
        }
        jobs.states.append(state)
        return dict(state)

    load_run = staticmethod(DesignService.load_run)


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["workbench-usability-test"])


@pytest.fixture
def windows(application, tmp_path):
    created = []

    def make(project=None, directory=None):
        if project is None:
            project = ProjectDocument.from_config(preset_config(), name=EXAMPLE_TITLES["cylinder"]).update_study(
                preset_study()
            )
        queue = Queue(directory or tmp_path / "jobs")
        instance = CoilGenWorkbench(project, jobs=queue, service=Service(), viewport_factory=Preview)
        instance.timer.stop()
        created.append(instance)
        return instance

    yield make
    for instance in created:
        instance._draft_dirty = False
        instance.undo_stack.setClean()
        instance.close()
        instance.undo_stack.clear()
        instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def prepared_run(tmp_path):
    config = preset_config()
    config.cylinder_mesh_parameter_list = (0.3, 0.16, 12, 6, 1, 0, 0, 0)
    config.target_region_resolution = 10
    config.progress_flag = False
    project = ProjectDocument.from_config(config, name="Prepared support").update_study({"pipeline_stage": "prepared"})
    directory = tmp_path / "run_prepared"
    DesignService.run(project, directory, threads=1)
    return project, DesignService.load_run(directory)
