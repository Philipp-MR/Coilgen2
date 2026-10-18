"""Preview work stays visible until real workers finish, including stale loads."""

from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import test_gui_live_preview as live
import test_native_workbench as native
import test_viewport_system_meshes as system

application = native.application
scientific_viewport = native.scientific_viewport
controllers = live.controllers


@pytest.mark.parametrize("invalidate", ["pause", "stop", "close"])
@pytest.mark.parametrize("fail", [False, True])
def test_invalidated_live_load_reports_physical_work_until_callback(controllers, application, invalidate, fail):
    jobs, create, releases = controllers
    state = live._state()
    live._write_manifest(jobs, state)
    entered, release = Event(), Event()
    releases.append(release)

    def loader(path, **_identity):
        entered.set()
        assert release.wait(3)
        if fail:
            raise ValueError("Archive cannot be loaded")
        return live._preview(path)

    controller = create(loader)
    activities, ready = [], []
    controller.changed.connect(lambda: activities.append((controller.working, controller.loading)))
    controller.preview_ready.connect(lambda *_args: ready.append(True))
    controller.start(state)
    assert entered.wait(3)
    assert activities[-1] == (True, True)
    getattr(controller, invalidate)()
    assert activities[-1] == (True, False)
    release.set()
    live._wait(application, lambda: not controller.working)
    assert activities[-1] == (False, False)
    assert not ready and controller.preview is None and controller.error is None


def _capture_pool(monkeypatch):
    import coilgen.gui.viewport as viewport_module

    tasks = []
    pool = SimpleNamespace(start=tasks.append)
    monkeypatch.setattr(viewport_module, "QThreadPool", SimpleNamespace(globalInstance=lambda: pool))
    return tasks


def test_model_preview_coalescing_and_clear_keep_inflight_work_visible(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    viewport._synchronous_preview = False
    tasks = _capture_pool(monkeypatch)
    activities = []
    viewport.activity_changed.connect(
        lambda: activities.append((viewport._preview_worker is not None, viewport._pending_preview is not None))
    )
    original = system._project("Gx", 0.08)
    edited = original.update_config({"cylinder_mesh_parameter_list": (0.2125, 0.09, 12, 6, 1, 0, 0, 0)})
    viewport.render_project(original)
    assert activities[-1] == (True, False)
    viewport.render_project(edited)
    assert activities[-1] == (True, True)
    tasks[0].run()
    assert len(tasks) == 2 and activities[-1] == (True, False)
    assert viewport._support is None
    viewport.clear_model()
    assert activities[-1] == (True, False)
    tasks[1].run()
    assert activities[-1] == (False, False)
    assert viewport._support is None


def test_system_mesh_work_notifies_pending_and_stale_completion(scientific_viewport, monkeypatch):
    viewport, assembly = scientific_viewport, system._assembly()
    viewport._synchronous_preview = False
    tasks = _capture_pool(monkeypatch)
    activities = []
    viewport.activity_changed.connect(
        lambda: activities.append((viewport._system_worker is not None, viewport._pending_system_preview is not None))
    )
    viewport.set_assembly_context(assembly, assembly.channels[0].id)
    assert activities[-1] == (True, False)
    edited_project = assembly.channels[1].project.update_config(
        {"cylinder_mesh_parameter_list": (0.2125, 0.125, 12, 6, 1, 0, 0, 0)}
    )
    edited = assembly.update_channel(assembly.channels[1].id, project=edited_project)
    viewport.set_assembly_context(edited, edited.channels[0].id)
    assert activities[-1] == (True, True)
    tasks[0].run()
    assert len(tasks) == 2 and activities[-1] == (True, False)
    assert not viewport._system_supports
    tasks[1].run()
    assert activities[-1] == (False, False)
    from coilgen.geometry import load_mesh_from_config

    expected = load_mesh_from_config(edited_project.config, edited_project.project_root)
    np.testing.assert_array_equal(viewport._system_supports[edited.channels[1].id].points, expected.vertices)


def test_synchronous_preview_and_close_publish_activity(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    activities = []
    viewport.activity_changed.connect(lambda: activities.append(viewport._preview_worker is not None))
    viewport.render_project(system._project("Gx", 0.08))
    assert True in activities and activities[-1] is False
    viewport._synchronous_preview = False
    _capture_pool(monkeypatch)
    viewport.render_project(system._project("Gy", 0.09))
    assert activities[-1] is True
    viewport.close()
    assert activities[-1] is False
