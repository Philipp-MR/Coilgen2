"""Browser transport for the shared, source-bound winding editor.

Client coordinates are proposals in metres. Only the Python editor can commit
them; stale messages, locks and validation receipts are checked here as well.
"""

import asyncio
import json
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np

from ..application.evidence import verify_record
from ..application.interconnection_editor import WindingEditorSession, editor_available
from ..persistence import json_value
from ..progress import DesignCancelled, ProgressRelay, progress_context
from ..winding_visuals import group_color
from ..winding_review import local_frame, review_issues, winding_sequence


class BrowserWindingEditor:
    def __init__(self, browser):
        self.browser, self.state = browser, browser.server.state
        self.session = None
        self.owner = None
        self.circuit, self.selected = 0, ""
        self.busy = False
        self.cancel = Event()
        self.saved_identity = ""
        self.seam_preview = None
        self.drafts = browser.session.workspace / "winding-drafts"
        self.drafts.mkdir(exist_ok=True)
        self.state.update(
            winding_open=False,
            winding_busy=False,
            winding_scene=None,
            winding_status="",
            winding_error="",
            winding_download="",
            winding_report_url="",
            winding_drafts=[],
            winding_receipt=None,
        )

    @property
    def editor(self):
        return self.session.editors[self.circuit]

    def _owner(self):
        project = self.browser.session.project
        return None if project is None else (project.id, self.browser.session.active_channel_id)

    def _matches_model(self):
        project = self.browser.session.project
        if project is None or self.owner != self._owner():
            return False
        return not self.session.model_changes(project)

    async def open(self):
        self.state.winding_open = True
        if self.busy:
            return
        record, owner = self.browser.record, self._owner()
        if (
            self.session is not None
            and self.owner == owner
            and (
                record is None
                or str(record.directory) == self.session.source_directory
                or self.session.identity != self.saved_identity
            )
        ):
            self.refresh()
            return  # Retain the current draft when the user closes the window.
        if record is None or not editor_available(record):
            self.state.winding_error = "Select a saved run with a complete automatic routing checkpoint."
            return
        if self.session is not None and self.session.identity != self.saved_identity:
            self.state.winding_error = "Save the retained winding draft before opening a different project."
            return

        def install(session):
            if owner != self._owner():
                raise ValueError("The project changed while opening the winding. Open it again.")
            self.session, self.owner = session, owner
            self.circuit, self.selected = 0, ""
            self.saved_identity = session.identity
            self.state.winding_download = self.state.winding_report_url = ""

        await self._run(
            lambda: WindingEditorSession.from_record(verify_record(record)),
            "Opening verified automatic winding",
            install,
        )

    def refresh(self):
        if self.session is None:
            return
        editor = self.editor
        if self.seam_preview and (
            self.seam_preview.get("identity") != self.session.identity
            or self.seam_preview.get("circuit") != self.circuit
            or self.seam_preview.get("revision") != editor.revision
        ):
            self.seam_preview = None
        sections = editor.sections
        if self.selected not in {s.key for s in sections}:
            self.selected = next((s.key for s in sections if s.cut is not None), sections[0].key)
        selection = editor.section(self.selected)
        mesh = editor.support
        try:
            chart, chart_error = editor.chart, ""
        except ValueError as error:
            chart, chart_error = None, str(error)
        rows = [
            dict(
                key=s.key,
                label=s.label,
                points=s.points.tolist(),
                editable=bool(s.editable),
                cut=None if s.cut is None else [float(v) for v in s.cut],
                locked=s.locked,
                disconnected=s.disconnected,
                guides=list(s.guides),
                geometry=sorted({g.value for g in s.geometry}),
                group=s.owner.group_index,
                color=group_color(s.owner.group_index),
                contour=not s.editable and s.owner.loop_index >= 0,
                target=s.target,
                bridge=None
                if not s.bridge
                else {k: v for k, v in s.bridge.items() if k not in {"base_points_m", "base_geometry"}},
                flat_points=chart.polyline(s.points) if chart else None,
                preview_flat=chart.polyline(
                    np.asarray(
                        next(
                            (r["points"] for r in (self.seam_preview or {}).get("routes", []) if r["key"] == s.key),
                            s.points,
                        )
                    )
                )
                if chart
                else None,
                flat_guides=chart.project(
                    s.guides if s.guides else s.points[len(s.points) // 2 : len(s.points) // 2 + 1]
                ).tolist()
                if chart
                else [],
            )
            for s in sections
        ]
        cut_curve = None
        if selection.cut is not None:
            # Original contour, not the already opened arc, for exact cut snapping.
            cut_curve = editor.cut_curve(self.selected).tolist()
        self.state.winding_scene = dict(
            identity=self.session.identity,
            source=editor.fingerprint,
            revision=editor.revision,
            circuit=self.circuit,
            circuits=len(self.session.editors),
            selected=self.selected,
            cut_objects=editor.cut_objects,
            bridge_handle=editor.bridge_handle(self.selected),
            local_frame=local_frame(editor, self.selected),
            sequence=winding_sequence(editor),
            issues=review_issues(
                editor,
                self.session.validation if self.session.validated else None,
                self.circuit,
                self.session.config.sweep_min_bend_radius,
            ),
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(),
            sections=rows,
            cut_point=editor.cut_point(self.selected).tolist() if selection.cut is not None else None,
            cut_curve=cut_curve,
            baseline=editor.automatic_path.v.tolist(),
            connectivity=editor.connectivity,
            undo=editor.undo_label,
            redo=editor.redo_label,
            dirty=self.session.identity != self.saved_identity,
            can_apply=self.session.can_apply and self._matches_model() and not self.seam_preview,
            can_validate=all(e.connectivity["complete"] for e in self.session.editors) and not self.seam_preview,
            model_matches=self._matches_model(),
            chart=None if chart is None else dict(label=chart.label, faces=chart.face_uv.tolist()),
            chart_error=chart_error,
            flat_cut_point=chart.project([editor.cut_point(self.selected)])[0].tolist()
            if chart and selection.cut
            else None,
            flat_cut_curve=chart.polyline(np.asarray(cut_curve)) if chart and cut_curve else None,
            flat_baseline=chart.polyline(editor.automatic_path.v) if chart else None,
            ports=[dict(**p, flat=chart.project([p["point"]])[0].tolist() if chart else None) for p in editor.ports],
            drawing_ports=[
                dict(**p, flat=chart.project([p["point"]])[0].tolist() if chart else None)
                for p in editor.drawing_ports(self.selected)
            ],
            seam_preview=self.seam_preview
            if self.seam_preview
            and self.seam_preview["identity"] == self.session.identity
            and self.seam_preview["circuit"] == self.circuit
            and self.seam_preview["revision"] == editor.revision
            else None,
        )
        self.state.winding_receipt = None
        if self.state.winding_scene["seam_preview"] and not self.seam_preview.get("error"):
            scene = dict(self.state.winding_scene)
            scene["issues"] = [i for i in scene["issues"] if i["category"] != "advisory"] + self.seam_preview.get(
                "warnings", []
            )
            self.state.winding_scene = scene
        if self.session.validated:
            report = self.session.validation
            self.state.winding_receipt = json_value(
                dict(
                    identity=report.identity,
                    accepted=report.accepted,
                    reasons=report.reasons,
                    diagnostics=report.diagnostics,
                )
            )
        self.state.winding_drafts = sorted(p.name for p in self.drafts.glob("*.cgwinding"))
        self.state.flush()

    async def _run(self, operation, label, complete=None):
        if self.busy:
            return
        self.busy = True
        self.cancel.clear()
        self.state.winding_busy, self.state.winding_error = True, ""
        self.state.winding_status = label + "…"
        self.state.flush()
        loop = asyncio.get_running_loop()

        def progress(value):
            def update():
                if self.busy:
                    self.state.winding_status = str(value.get("message") or value.get("stage") or label)
                    self.state.flush()

            loop.call_soon_threadsafe(update)

        def work():
            relay = ProgressRelay(progress)
            with progress_context(relay.report, cancel_check=self.cancel.is_set):
                if self.cancel.is_set():
                    raise DesignCancelled("Operation cancelled.")
                result = operation()
            relay.flush()
            return result

        try:
            result = await asyncio.to_thread(work)
            if complete:
                complete(result)
            self.state.winding_status = label + " completed."
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, IndexError) as error:
            if label.startswith("Previewing"):
                self.seam_preview = None
            self.state.winding_error = str(error)
        finally:
            self.busy = False
            self.state.winding_busy = False
            self.refresh()
            self.state.flush()

    async def action(self, message):
        if not isinstance(message, dict):
            return
        kind = message.get("kind")
        if kind == "cancel":
            self.cancel.set()
            return
        if self.busy or self.session is None:
            return
        if message.get("identity") != self.session.identity or message.get("circuit") != self.circuit:
            self.state.winding_error = "That edit refers to an older revision. Review the current winding and retry."
            self.refresh()
            return
        self.state.winding_error = ""
        try:
            if kind == "select":
                self.editor.section(message["key"])
                self.selected = message["key"]
            elif kind == "clear_preview":
                self.seam_preview = None
                self.state.winding_status = "Preview cleared"
            elif kind == "circuit":
                index = int(message["index"])
                if not 0 <= index < len(self.session.editors):
                    raise ValueError("Unknown circuit.")
                self.circuit, self.selected = index, ""
            elif kind == "save":
                name = f"winding-{self.session.identity[:12]}-{uuid4().hex[:6]}.cgwinding"
                self.session.save_draft(self.drafts / name)
                self.session.save_draft(self.browser.exports / name)
                self.saved_identity = self.session.identity
                self.state.winding_download = "downloads/" + name
                self.state.winding_status = "Draft saved: " + name
            elif kind in {"open", "import"}:
                if kind == "open":
                    name = message["name"]
                    if not isinstance(name, str) or name != Path(name).name or name not in self.state.winding_drafts:
                        raise ValueError("Choose a saved winding draft from this workspace.")
                    self.session.load_draft(self.drafts / name)
                else:
                    content = message["content"]
                    if not isinstance(content, str) or len(content.encode()) > 50 * 1024**2:
                        raise ValueError("Winding draft exceeds 50 MiB.")
                    payload = json.loads(content)
                    if not isinstance(payload, dict) or payload.get("schema") != "coilgen.winding-draft/1":
                        raise ValueError("Unsupported winding draft format.")
                    self.session.load_plan(payload["plan"])
                self.saved_identity = self.session.identity
                self.state.winding_report_url = ""
                self.seam_preview = None
                self.state.winding_status = "Draft reopened. Validate this revision before applying."
            elif kind == "apply":
                state = self.browser.server.state
                if not self.session.can_apply or not self._matches_model() or self.seam_preview:
                    raise ValueError("Validate this exact winding and open its matching source model before applying.")
                if (
                    state.setup_dirty
                    or state.settings_dirty
                    or state.groups_dirty
                    or self.browser._advanced_has_edits()
                ):
                    raise ValueError("Apply or discard the pending model settings before applying this winding.")
                self.browser.session.apply_config({"routing_editor_plan": self.session.plan()})
                self.state.winding_status = "Winding plan applied. Recompute the design to update fields and exports."
            elif kind == "validate":
                if self.seam_preview:
                    raise ValueError("Generate or discard the preview before checking the winding.")

                def finish(report):
                    if not self.session.accept_validation(report):
                        raise ValueError("The winding changed during validation; repeat the checks.")
                    name = f"winding-validation-{report.identity[:16]}.json"
                    (self.browser.exports / name).write_text(
                        json.dumps(
                            json_value(
                                dict(
                                    identity=report.identity,
                                    accepted=report.accepted,
                                    reasons=report.reasons,
                                    diagnostics=report.diagnostics,
                                )
                            ),
                            indent=2,
                            allow_nan=False,
                        ),
                        encoding="utf-8",
                    )
                    self.state.winding_report_url = "downloads/" + name

                await self._run(self.session.validation_job(), "Engineering validation", finish)
                return
            else:
                editor = self.editor
                section = editor.section(message.get("key", self.selected))
                key = section.key

                def edit():
                    if kind == "undo":
                        editor.undo()
                    elif kind == "redo":
                        editor.redo()
                    elif kind == "lock":
                        obj = next((o for o in editor.cut_objects if o["id"] == message.get("object_id")), None)
                        if obj:
                            editor.set_cut_object_locked(obj["id"], not obj["locked"])
                        else:
                            editor.set_locked(key, not section.locked)
                    elif kind == "disconnect":
                        editor.disconnect(key, not section.disconnected)
                    elif kind == "route":
                        editor.route_unlocked()
                    elif kind == "preview_seam":
                        group = message.get("group")
                        keys = [
                            s.key
                            for s in editor.sections
                            if s.cut is not None and (group is None or s.owner.group_index == int(group))
                        ]
                        if message.get("scope") == "selected":
                            keys = [key] if editor.section(key).cut is not None else []
                        preview = editor.preview_seam(message["stroke"], keys, flat=bool(message.get("flat")))
                        preview = editor.complete_preview(preview, self.session.config.sweep_min_bend_radius)
                        return dict(**preview, identity=self.session.identity, circuit=self.circuit)
                    elif kind in {"preview_shape", "shape_handle", "select_cut_object", "shape_choice", "place_shape"}:
                        from ..winding_cut_shapes import CutShape

                        group = message.get("group")
                        keys = [
                            s.key
                            for s in editor.sections
                            if s.cut is not None and (group is None or s.owner.group_index == int(group))
                        ]
                        if message.get("scope") == "selected":
                            keys = [key] if editor.section(key).cut is not None else []
                        previous = self.seam_preview or {}
                        object_id = message.get("object_id")
                        choices = {}
                        if kind == "select_cut_object":
                            obj = next((o for o in editor.cut_objects if o["id"] == object_id), None)
                            if obj is None:
                                raise ValueError("Unknown cut object.")
                            value, keys = obj["shape"], obj["keys"]
                            self.selected = keys[0]
                        elif kind == "shape_choice":
                            value = previous["shape"]
                            keys, object_id = previous["keys"], previous["object_id"]
                            choices = {**previous.get("choices", {}), message["cut_key"]: int(message["index"])}
                        elif kind == "shape_handle":
                            if not self.seam_preview or self.seam_preview["identity"] != self.session.identity:
                                raise ValueError("Place a cut shape on this revision first.")
                            value = (
                                CutShape.from_dict(self.seam_preview["shape"])
                                .moved(message["role"], message["point"])
                                .to_dict()
                            )
                            keys, object_id = previous["keys"], previous["object_id"]
                        else:
                            value = message.get("shape") or editor.default_cut_shape(
                                message.get("shape_kind", "plane"), key
                            )
                            if kind == "place_shape":
                                value = (
                                    CutShape.from_dict(value)
                                    .moved("center", editor.project(message["point"]))
                                    .to_dict()
                                )
                            existing = next((o for o in editor.cut_objects if o["id"] == object_id), None)
                            if existing:
                                keys = existing["keys"]
                        # New previews already have a proposed id, but are not yet persistent objects.
                        known = {o["id"] for o in editor.cut_objects}
                        preview = editor.preview_cut_shape(
                            value, keys, object_id=object_id if object_id in known else None, choices=choices
                        )
                        preview = editor.complete_preview(preview, self.session.config.sweep_min_bend_radius)
                        return dict(**preview, identity=self.session.identity, circuit=self.circuit)
                    elif kind == "preview_drawing":
                        preview = editor.preview_drawing(
                            message["source"],
                            message["target"],
                            message["stroke"],
                            flat=bool(message.get("flat")),
                            minimum_radius=self.session.config.sweep_min_bend_radius,
                        )
                        return dict(**preview, identity=self.session.identity, circuit=self.circuit)
                    elif kind == "apply_seam":
                        if not self.seam_preview or self.seam_preview["identity"] != self.session.identity:
                            raise ValueError("Draw and review the seam on the current revision first.")
                        editor.apply_preview(self.seam_preview)
                    elif kind == "reconnect_endpoints":
                        editor.reconnect_endpoints(message["source"], message["target"])
                    elif kind == "draw_connection":
                        editor.draw_connection(
                            message["source"], message["target"], message["stroke"], flat=bool(message.get("flat"))
                        )
                    elif kind == "bridge":
                        return editor.bridge_crossing(
                            key,
                            message["other"],
                            message["relation"],
                            float(message["separation_mm"]) / 1000,
                            float(message["span_mm"]) / 1000,
                            message.get("point"),
                        )
                    elif kind == "remove_bridge":
                        editor.remove_bridge(key)
                    elif kind == "bridge_height":
                        editor.set_bridge_height(key, message["point"])
                    elif kind == "drag_guide":
                        editor.drag_guide(key, int(message["index"]), message["point"], flat=bool(message.get("flat")))
                    elif kind == "cut":
                        center = float(message["position_mm"]) / 1e3
                        editor.move_cut(key, center, float(message["span_mm"]) / 1e3)
                    elif kind == "drag_cut":
                        point = np.asarray(message["point"], float)
                        if point.shape != (3,) or not np.isfinite(point).all():
                            raise ValueError("A cut point needs three finite coordinates.")
                        editor.move_cut(key, editor.nearest_cut_position(key, point))
                    elif kind in {"guide", "remove_guide"}:
                        guides = list(section.guides)
                        index = int(message["index"])
                        if kind == "remove_guide":
                            if not 0 <= index < len(guides):
                                raise ValueError("Unknown guide point.")
                            guides.pop(index)
                        else:
                            point = np.asarray(message["point"], float)
                            if point.shape != (3,) or not np.isfinite(point).all():
                                raise ValueError("A guide needs three finite coordinates in metres.")
                            if not 0 <= index <= len(guides):
                                raise ValueError("Unknown guide point.")
                            if index == len(guides):
                                guides.append(point)
                            else:
                                guides[index] = point
                        editor.set_guides(key, guides)
                    else:
                        raise ValueError("Unknown winding operation.")

                def completed(value):
                    if kind in {
                        "preview_seam",
                        "preview_shape",
                        "shape_handle",
                        "shape_choice",
                        "select_cut_object",
                        "place_shape",
                        "preview_drawing",
                    }:
                        self.seam_preview = value
                    else:
                        self.seam_preview = None
                        if kind == "bridge":
                            self.selected = value

                await self._run(
                    edit,
                    dict(
                        drag_cut="Moving contour cut",
                        guide="Routing connection",
                        drag_guide="Routing connection",
                        cut="Moving contour cut",
                        route="Routing unlocked connections",
                        preview_seam="Previewing seam",
                        preview_shape="Previewing cut shape",
                        select_cut_object="Previewing cut object",
                        shape_choice="Previewing chosen opening",
                        place_shape="Previewing cut shape",
                        preview_drawing="Previewing drawn connection",
                        shape_handle="Previewing cut shape",
                        apply_seam="Moving seam",
                        reconnect_endpoints="Reconnecting endpoints",
                        bridge="Constructing crossing ramps",
                        remove_bridge="Removing bridge",
                    ).get(kind, kind.capitalize()),
                    completed,
                )
                return
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            self.state.winding_error = str(error)
        self.refresh()


def build_winding_editor(browser):
    from trame.widgets import vuetify3 as v3
    from trame_client.widgets.core import HtmlElement

    with v3.VDialog(v_model="winding_open", fullscreen=True, persistent=True):
        HtmlElement(
            "coilgen-winding-editor",
            scene=("winding_scene",),
            busy=("winding_busy",),
            status=("winding_status",),
            error=("winding_error",),
            drafts=("winding_drafts",),
            download=("winding_download",),
            report_url=("winding_report_url",),
            receipt=("winding_receipt",),
            __properties=[
                "scene",
                "busy",
                "status",
                "error",
                "drafts",
                "download",
                ("report_url", "report-url"),
                "receipt",
            ],
            __events=["action", "close"],
            action=(browser.winding.action, "[$event]"),
            close="winding_open = false",
        )
