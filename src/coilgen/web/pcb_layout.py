"""Asynchronous browser adapter for the shared PCB workspace."""

import asyncio
import json
from uuid import uuid4

from ..application.pcb import PcbSession, pcb_available


class BrowserPcbLayout:
    def __init__(self, browser):
        self.browser, self.state = browser, browser.server.state
        self.session = None
        self.sessions = {}
        self.busy = False
        self.preview_request = 0
        self.state.update(
            pcb_open=False,
            pcb_scene=None,
            pcb_busy=False,
            pcb_error="",
            pcb_status="",
            pcb_download="",
            pcb_download_label="",
            pcb_preview=None,
        )

    async def open(self):
        self.state.pcb_open = True
        if self.busy:
            return
        record = self.browser.record
        if not pcb_available(record) or self.state.selected_run != self.state.displayed_run:
            self.session = None
            self.state.pcb_download = ""
            self.state.pcb_error = "Select a saved run with a complete routing checkpoint."
            self.refresh()
            return
        key = str(record.directory)
        if key in self.sessions:
            self.session = self.sessions[key]
            self.state.pcb_download = ""
            self.state.pcb_error = ""
            self.state.pcb_status = "PCB draft restored."
            self.refresh()
            return

        def install(session):
            if self.browser.record is not record:
                raise ValueError("The selected run changed while opening the PCB layout. Open it again.")
            self.sessions[key] = self.session = session
            self.state.pcb_download = ""

        self.session = None
        self.state.pcb_download = ""
        self.refresh()
        await self.run(lambda: PcbSession.from_record(record, workspace=self.browser.session.workspace), install)

    def refresh(self):
        self.state.pcb_scene = self.session.scene() if self.session else None

    async def run(self, operation, completed=None, *, success="PCB layout updated."):
        if self.busy:
            return
        self.busy = self.state.pcb_busy = True
        self.preview_request += 1
        self.state.pcb_preview = None
        self.state.pcb_error = ""
        self.state.pcb_status = "Updating PCB layout and checks…"
        self.state.flush()
        try:
            value = await asyncio.to_thread(operation)
            if completed:
                completed(value)
            self.state.pcb_status = success
        except Exception as error:
            self.state.pcb_error = str(error)
            self.state.pcb_status = "Action failed; layout unchanged."
        finally:
            self.busy = self.state.pcb_busy = False
            self.refresh()
            self.state.flush()

    async def action(self, message):
        if self.busy or self.session is None:
            return
        if not isinstance(message, dict) or message.get("identity") != self.session.identity:
            self.state.pcb_error = "The PCB layout changed. Use the refreshed layout."
            self.refresh()
            return
        kind = message.get("kind")
        session = self.session
        if kind == "preview":
            self.preview_request += 1
            request = self.preview_request
            values = message.get("values", {})
            if not isinstance(values, dict):
                values = {}
            try:
                result = await asyncio.to_thread(session.preview_connection, **values)
            except (ValueError, TypeError, KeyError) as error:
                result = dict(
                    identity=message["identity"], **{k: values.get(k) for k in ("a", "b", "layer")}, error=str(error)
                )
            if request == self.preview_request and self.session is session and session.identity == message["identity"]:
                self.state.pcb_preview = result
                self.state.flush()
            return
        self.state.pcb_download = ""
        if kind in {"undo", "redo"}:
            await self.run(lambda: session.undo(kind == "redo"))
        elif kind == "evaluate":
            await self.run(session.evaluate)
        elif kind == "load":
            payload = message.get("document")
            if len(json.dumps(payload)) > 10 * 1024**2:
                self.state.pcb_error = "PCB draft exceeds 10 MiB."
                return
            await self.run(lambda: session.load(payload))
        elif kind in {"save", "export"}:
            name = (
                session.filename("artwork" if kind == "export" else "draft")
                + "-"
                + uuid4().hex[:8]
                + (".cgpcb" if kind == "save" else ".zip")
            )
            path = self.browser.exports / name

            def write():
                if kind == "export":
                    return session.export(path, tuple(message.get("formats", ["svg", "dxf"])))
                path.write_text(json.dumps(session.document(), indent=2, allow_nan=False), encoding="utf-8")
                return path

            label = "Download SVG/DXF package" if kind == "export" else "Download PCB draft"

            def ready(_):
                self.state.pcb_download = "downloads/" + name
                self.state.pcb_download_label = label

            await self.run(write, ready, success=f"File ready. Click {label} to keep a copy on your computer.")
        else:
            values = message.get("values", {})
            if not isinstance(values, dict):
                self.state.pcb_error = "Invalid PCB edit."
                return
            await self.run(lambda: session.change(kind, **values))


def build_pcb_layout(browser):
    from trame.widgets import vuetify3 as v3
    from trame_client.widgets.core import HtmlElement

    with v3.VDialog(v_model="pcb_open", fullscreen=True, persistent=True, content_class="pcb-dialog-content"):
        HtmlElement(
            "coilgen-pcb-layout",
            scene=("pcb_scene",),
            busy=("pcb_busy",),
            error=("pcb_error",),
            status=("pcb_status",),
            download=("pcb_download",),
            download_label=("pcb_download_label",),
            preview=("pcb_preview",),
            __properties=["scene", "busy", "error", "status", "download", "download_label", "preview"],
            __events=["action", "close"],
            action=(browser.pcb.action, "[$event]"),
            close="pcb_open = false",
        )
