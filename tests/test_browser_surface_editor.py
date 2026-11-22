"""Surface messages cannot cross projects or apply a stale display tessellation."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("gmsh")

from coilgen.application.presets import preset_config  # noqa: E402
from coilgen.application.project import ProjectDocument  # noqa: E402
from coilgen.application.surface_model import SurfaceModel, primitive  # noqa: E402
from coilgen.web.surface_editor import BrowserSurfaceEditor  # noqa: E402


class State(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def flush(self):
        pass


def adapter(tmp_path):
    session = SimpleNamespace(
        workspace=tmp_path, project=ProjectDocument.from_config(preset_config("cylinder")), active_channel_id="coil"
    )
    session._replace_project = lambda project: setattr(session, "project", project)
    browser = SimpleNamespace(session=session, server=SimpleNamespace(state=State()), _advanced_has_edits=lambda: False)
    return BrowserSurfaceEditor(browser)


def command(editor, kind, **extra):
    return editor.action(dict(kind=kind, identity=editor.editor.model.identity, **extra))


def test_edit_mesh_apply_reopen_and_persistent_draft(tmp_path):
    editor = adapter(tmp_path)

    async def run():
        await editor.open()
        assert editor.state.surface_scene and not editor.state.surface_error
        original = editor.browser.session.project.fingerprint
        await command(editor, "apply")
        assert "Mesh" in editor.state.surface_error
        value = editor.editor.model.to_dict()
        value["nodes"].append(primitive("sphere", "object2", center_m=(0.065, -0.035, 0.04)))
        value["nodes"].append(dict(id="cut1", kind="subtract", name="Cut", base="object1", tool="object2"))
        await command(editor, "update", model=value)
        value = editor.editor.model.to_dict()
        value["excluded_faces"] = [f["key"] for f in editor.state.surface_scene["faces"] if "Plane" in f["label"]]
        await command(editor, "update", model=value)
        await command(editor, "mesh")
        assert editor.state.surface_scene["can_apply"]
        assert editor.editor.mesh_scene["mesh_report"]["boundary_loops"] == 2
        await command(editor, "apply")
        assert not editor.state.surface_error
        assert editor.browser.session.project.fingerprint != original
        assert SurfaceModel.from_project(editor.browser.session.project).identity == editor.editor.model.identity
        reopened = BrowserSurfaceEditor(editor.browser)
        await reopened.open()
        assert reopened.editor.model.identity == editor.editor.model.identity
        assert not reopened.state.surface_scene["can_apply"]

    asyncio.run(run())


def test_stale_message_and_project_switch_do_not_publish_geometry(tmp_path):
    editor = adapter(tmp_path)

    async def run():
        await editor.open()
        old = editor.editor.model.identity
        await editor.action(dict(kind="mesh", identity="stale"))
        assert editor.editor.mesh is None
        assert "stale" in editor.state.surface_error
        editor.browser.session.active_channel_id = "other"
        await command(editor, "mesh")
        assert editor.editor.mesh is None and editor.editor.model.identity == old
        assert "stale" in editor.state.surface_error

    asyncio.run(run())


def test_failed_boolean_keeps_last_usable_draft(tmp_path):
    editor = adapter(tmp_path)

    async def run():
        await editor.open()
        previous = editor.editor.model.identity
        value = editor.editor.model.to_dict()
        tool = primitive("sphere", "tool")
        tool["parameters"]["radius"] = 1
        value["nodes"] += [tool, dict(id="cut", kind="subtract", name="Empty", base="object1", tool="tool")]
        await command(editor, "update", model=value)
        assert "complete object" in editor.state.surface_error
        assert editor.editor.model.identity == previous
        assert not editor.busy

    asyncio.run(run())
