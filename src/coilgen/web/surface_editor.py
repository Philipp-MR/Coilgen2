"""Per-project browser transport for the shared CAD surface editor."""

import asyncio
from hashlib import sha256
import json

from ..application.surface_model import SurfaceEditor, SurfaceModel, apply_surface
from ..persistence import atomic_json


class BrowserSurfaceEditor:
    def __init__(self, browser):
        self.browser, self.state = browser, browser.server.state
        self.editor, self.owner = None, None
        self.busy = False
        self.drafts = browser.session.workspace / "surface-drafts"
        self.drafts.mkdir(exist_ok=True)
        self.state.update(
            surface_open=False, surface_busy=False, surface_scene=None, surface_error="", surface_status=""
        )

    def _owner(self):
        project = self.browser.session.project
        return None if project is None else (project.id, self.browser.session.active_channel_id, project.fingerprint)

    def _draft_path(self):
        return self.drafts / (sha256(json.dumps(self.owner).encode()).hexdigest() + ".json")

    def _save(self):
        atomic_json(self._draft_path(), self.editor.model.to_dict())

    def refresh(self):
        self.state.surface_scene = self.editor.snapshot() if self.editor else None
        self.state.flush()

    async def _run(self, operation, label, complete=None):
        if self.busy:
            return
        self.busy = self.state.surface_busy = True
        self.state.surface_error, self.state.surface_status = "", label + "…"
        owner = self._owner()
        self.state.flush()
        try:
            result = await asyncio.to_thread(operation)
            if owner != self._owner():
                raise ValueError("The active project changed. Reopen Geometry studio before applying this draft.")
            if complete:
                complete(result)
            self._save()
            self.state.surface_status = label + "."
        except Exception as exc:
            self.state.surface_error = str(exc)
        finally:
            self.busy = self.state.surface_busy = False
            self.refresh()

    async def open(self):
        self.state.surface_open = True
        if self.busy:
            return
        owner = self._owner()
        if owner is None:
            self.state.surface_error = "Open a project before creating its winding surface."
            return
        if self.state.settings_dirty or self.browser._advanced_has_edits():
            self.state.surface_error = "Apply or discard the pending settings before opening Geometry studio."
            return
        if self.editor is not None and self.owner == owner:
            self.refresh()
            return
        self.owner, self.editor = owner, None
        project = self.browser.session.project
        path = self._draft_path()

        def load():
            model = (
                SurfaceModel(json.loads(path.read_text(encoding="utf-8")))
                if path.exists()
                else SurfaceModel.from_project(project)
            )
            editor = SurfaceEditor(model)
            editor.preview()
            return editor

        await self._run(load, "Opened geometry studio", lambda editor: setattr(self, "editor", editor))

    async def action(self, message):
        if self.busy:
            return
        if not isinstance(message, dict) or self.editor is None:
            self.state.surface_error = "Open Geometry studio first."
            return
        if self.owner != self._owner() or message.get("identity") != self.editor.model.identity:
            self.state.surface_error = "This draft is stale. Close and reopen Geometry studio."
            self.refresh()
            return
        kind = message.get("kind")
        if kind == "update":
            await self._run(lambda: self.editor.edit(message.get("model")), "Updated CAD surface")
        elif kind in {"undo", "redo"}:
            await self._run(lambda: self.editor.travel(kind == "redo"), "Restored geometry edit")
        elif kind == "mesh":
            await self._run(self.editor.generate, "Meshed retained surface")
        elif kind == "apply":
            project = self.browser.session.project

            def apply(updated):
                self.browser.session._replace_project(updated)
                self.owner = self._owner()

            if self.editor.mesh is None:
                self.state.surface_error = "Mesh the retained surface before applying it."
                return
            await self._run(
                lambda: apply_surface(project, self.editor.model, self.editor.mesh, expected_fingerprint=self.owner[2]),
                "Applied winding surface",
                apply,
            )
        else:
            self.state.surface_error = "Unknown geometry action."


def build_surface_editor(browser):
    from trame.widgets import vuetify3 as v3
    from trame_client.widgets.core import HtmlElement

    with v3.VDialog(v_model="surface_open", fullscreen=True, persistent=True):
        HtmlElement(
            "coilgen-surface-editor",
            scene=("surface_scene",),
            busy=("surface_busy",),
            error=("surface_error",),
            status=("surface_status",),
            __properties=["scene", "busy", "error", "status"],
            __events=["action", "close"],
            action=(browser.surface.action, "[$event]"),
            close="surface_open = false",
        )
