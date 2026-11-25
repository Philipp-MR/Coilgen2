"""Source-bound PCB drafts and exact artwork packages shared by both clients."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from html import escape
from io import BytesIO
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import zipfile

import numpy as np

from ..attribution import software_attribution
from ..pcb import _dxf_lwpolyline
from ..pcb_layout import PcbDesign, PcbSettings, polygon_rings
from ..pipeline_stages import PipelineCheckpoint
from .evidence import load_saved_request, read_verified_artifact, verify_record
from .pcb_review import connection_preview, issue_groups, review_summary


def pcb_available(record):
    if record is None:
        return False
    metadata = getattr(record, "metadata_view", None)
    if metadata is None:
        metadata = getattr(record, "metadata", {})
    return metadata.get("completed_stage", "evaluated") in {"routed", "evaluated"} and any(
        a.get("name") == "pipeline_checkpoint.cgstage" for a in metadata.get("artifacts", ())
    )


class PcbSession:
    def __init__(self, parts, config, *, target=None, source="", source_id="", design_margin=1.0):
        self.parts, self.config, self.target = deepcopy(parts), deepcopy(config), deepcopy(target)
        if not self.parts:
            raise ValueError("The saved run contains no winding parts.")
        self.source, self.source_id = str(source), source_id
        self.source_label = Path(self.source).name or "Saved winding"
        if not np.isfinite(design_margin) or not 0 < design_margin <= 1:
            raise ValueError("Saved design margin must lie in (0, 1].")
        self.design_margin = float(design_margin)
        if not source_id:
            digest = sha256()
            for p in self.parts:
                for a in (p.mesh.vertices, p.mesh.faces, p.wire_path.v if p.wire_path else []):
                    digest.update(np.asarray(a).tobytes())
                for g in p.connected_groups:
                    digest.update(g.spiral_in_v.tobytes())
                    digest.update(g.spiral_out_v.tobytes())
            self.source_id = digest.hexdigest()
        width = max(1e-6, min(0.1, float(config.conductor_cross_section_width)))
        self.drafts = [dict(settings=asdict(PcbSettings(width_m=width)), edits={}) for _ in parts]
        self.part = 0
        self.history, self.future = [], []
        self.evaluation = None
        self.design = None
        self.problem = ""
        self.storage_path = None
        self.storage_error = ""
        self.saved_identity = None
        self.rebuild()

    @classmethod
    def from_record(cls, record, *, workspace=None):
        record = verify_record(record)
        payload = read_verified_artifact(record, "pipeline_checkpoint.cgstage")
        checkpoint = PipelineCheckpoint.load(BytesIO(payload))
        request = load_saved_request(record)
        session = cls(
            checkpoint.parts,
            checkpoint.config,
            target=checkpoint.target_field,
            source=record.directory,
            source_id=sha256(payload).hexdigest(),
            design_margin=request.get("design_margin", 1.0),
        )
        session.source_label = str(record.metadata_view.get("name") or session.source_label)
        if workspace is not None:
            session.attach_storage(Path(workspace) / "pcb-drafts")
        return session

    def attach_storage(self, directory):
        """Restore a source-bound draft, without ever replacing an unreadable one."""
        path = Path(directory) / (self.source_id + ".cgpcb")
        try:
            if path.exists():
                if path.stat().st_size > 10 * 1024**2:
                    raise ValueError("Saved draft exceeds 10 MiB.")
                self.load(json.loads(path.read_text(encoding="utf-8")))
                self.history.clear()
                self.future.clear()
            self.storage_path = path
            self._persist()
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.storage_path = None
            self.storage_error = "Workspace draft could not be restored: " + str(error)
        return self

    def _persist(self):
        if self.storage_path is None:
            return
        from ..persistence import atomic_text

        try:
            atomic_text(self.storage_path, json.dumps(self.document(), indent=2, allow_nan=False))
            self.saved_identity, self.storage_error = self.identity, ""
        except OSError as error:
            self.storage_error = "Draft not saved: " + str(error)

    def filename(self, kind="draft"):
        label = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.source_label).strip("-")[:70] or "coil"
        return f"{label}-part{self.part + 1}-pcb-{kind}"

    def preview_connection(self, a, b, layer):
        identity, design = self.identity, self.design
        connections = deepcopy(self.drafts[self.part]["edits"].get("connections", []))
        preview = connection_preview(design, a, b, layer, connections)
        return dict(preview, identity=identity)

    @property
    def identity(self):
        return sha256(json.dumps(self.document(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def document(self):
        return dict(
            schema="coilgen.pcb-draft/1",
            source_id=self.source_id,
            source_directory=self.source,
            part=self.part,
            design_margin=self.design_margin,
            drafts=deepcopy(self.drafts),
        )

    def rebuild(self):
        self.design, self.problem = None, ""
        try:
            draft = self.drafts[self.part]
            self.design = PcbDesign(self.parts[self.part], PcbSettings(**draft["settings"]), draft["edits"])
        except ValueError as error:
            self.problem = str(error)
        self.evaluation = None

    def change(self, operation, **values):
        before = (self.part, deepcopy(self.drafts))
        previous_result = self.design, self.problem, self.evaluation
        draft = self.drafts[self.part]
        try:
            if operation == "part":
                index = values["index"]
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.parts):
                    raise ValueError("Unknown PCB part.")
                self.part = index
            elif operation == "settings":
                settings = replace(PcbSettings(**draft["settings"]), **values)
                if settings.mode != draft["settings"]["mode"]:
                    draft["edits"] = {}
                draft["settings"] = asdict(settings)
            elif operation == "layer":
                if self.design is None or values["key"] not in {
                    t.key for t in self.design.traces if t.source is not None
                }:
                    raise ValueError("Select a source winding trace.")
                if values["layer"] not in {"top", "bottom"}:
                    raise ValueError("Unknown copper layer.")
                draft["edits"].setdefault("layers", {})[values["key"]] = values["layer"]
            elif operation == "connect":
                if self.design is None:
                    raise ValueError(self.problem)
                if any(values[k] not in self.design.endpoints for k in ("a", "b")) or values["a"] == values["b"]:
                    raise ValueError("Choose two different source endpoints.")
                link = {k: values[k] for k in ("a", "b", "layer")}
                connections = draft["edits"].setdefault("connections", [])
                if any({c["a"], c["b"]} == {link["a"], link["b"]} for c in connections):
                    raise ValueError("These endpoints are already connected.")
                connections.append(link)
            elif operation == "remove_connection":
                draft["edits"].get("connections", []).pop(int(values["index"]))
            elif operation == "terminals":
                if values["a"] == values["b"]:
                    raise ValueError("Input and return must be different endpoints.")
                draft["edits"]["terminals"] = [values["a"], values["b"]]
            elif operation == "seam":
                self._choose_seam(draft)
            else:
                raise ValueError("Unknown PCB edit.")
            self.rebuild()
            if self.problem and operation not in {"part", "settings"}:
                raise ValueError(self.problem)
        except Exception:
            self.part, self.drafts = before
            self.design, self.problem, self.evaluation = previous_result
            raise
        self.history.append(before)
        self.future.clear()
        self._persist()

    def _choose_seam(self, draft):
        if self.design is None or self.design.frame.kind != "cylinder":
            raise ValueError("Seam placement applies to cylindrical boards.")
        frame = self.design.frame
        sources = [t.source for t in self.design.traces if t.source is not None]
        angles = [
            np.unwrap(np.arctan2(((p - frame.origin) @ frame.axes.T)[:, 1], ((p - frame.origin) @ frame.axes.T)[:, 0]))
            for p in sources
        ]
        best = None
        for angle in np.linspace(0, 2 * np.pi, 180, endpoint=False):
            cuts = sum(int(np.count_nonzero(np.diff(np.floor((a - angle) / (2 * np.pi))))) for a in angles)
            distance = min(float(np.min(abs(np.angle(np.exp(1j * (a - angle)))))) for a in angles)
            candidate = (cuts, -distance, float(np.degrees(angle)))
            if best is None or candidate < best:
                best = candidate
        draft["settings"]["seam_deg"] = best[2]

    def undo(self, redo=False):
        source, destination = (self.future, self.history) if redo else (self.history, self.future)
        if source:
            destination.append((self.part, deepcopy(self.drafts)))
            self.part, self.drafts = source.pop()
            self.rebuild()
            self._persist()

    def load(self, payload):
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "coilgen.pcb-draft/1"
            or payload.get("source_id") != self.source_id
        ):
            raise ValueError("This PCB draft belongs to a different saved winding.")
        if len(payload.get("drafts", [])) != len(self.parts):
            raise ValueError("PCB draft part count does not match the saved winding.")
        before = (self.part, deepcopy(self.drafts))
        previous_result = self.design, self.problem, self.evaluation
        try:
            part = payload["part"]
            if type(part) is not int or not 0 <= part < len(self.parts):
                raise ValueError("Unknown PCB part.")
            for p, draft in zip(self.parts, payload["drafts"]):
                PcbSettings(**draft["settings"])
                # Validate all edited parts, not only the visible one.
                if draft.get("edits"):
                    PcbDesign(p, PcbSettings(**draft["settings"]), draft["edits"])
            self.part, self.drafts = part, deepcopy(payload["drafts"])
            self.rebuild()
        except Exception:
            self.part, self.drafts = before
            self.design, self.problem, self.evaluation = previous_result
            raise
        self.history.append(before)
        self.future.clear()
        self._persist()

    def scene(self):
        scene = self.design.scene() if self.design else {}
        endpoints = {e["key"]: e["label"] for e in scene.get("endpoints", [])}
        scene.update(
            identity=self.identity,
            source=self.source_label,
            part=self.part,
            parts=[f"Part {i + 1}" for i in range(len(self.parts))],
            problem=self.problem,
            settings=self.drafts[self.part]["settings"],
            undo=bool(self.history),
            redo=bool(self.future),
            connections=self.drafts[self.part]["edits"].get("connections", []),
            evaluation=self.evaluation,
            review=review_summary(self.design, self.evaluation, self.problem),
            issue_groups=issue_groups(scene.get("issues", [])),
            connection_labels=[
                f"{endpoints.get(c['a'], c['a'])} → {endpoints.get(c['b'], c['b'])} · {c['layer']}"
                for c in self.drafts[self.part]["edits"].get("connections", [])
            ],
            persistence=(
                self.storage_error or "Saved in workspace · restored after restart"
                if self.saved_identity == self.identity
                else self.storage_error or "Unsaved draft · save a copy to keep it"
            ),
        )
        return scene

    def evaluate(self):
        from ..physics import biot_savart_wire_path
        from ..research.validation import field_budget_check
        from types import SimpleNamespace

        design = self.design
        if design is None or design.path is None:
            raise ValueError("Resolve continuity, seam cuts and copper overlaps before evaluating the PCB circuit.")
        if self.target is None:
            raise ValueError("The source run has no saved target samples.")
        field = np.zeros_like(self.target.b)
        baseline = np.zeros_like(field)
        for i, part in enumerate(self.parts):
            path = (
                part.fabrication_path.wire_points_m
                if part.fabrication_path is not None
                else part.wire_path.v
                if part.wire_path is not None
                else None
            )
            if path is None or part.contour_step is None or not np.isfinite(part.contour_step):
                raise ValueError("Every source circuit needs its saved path and current for field comparison.")
            current = float(part.contour_step)
            original = current * biot_savart_wire_path(path, self.target.coords, worker_count=1, max_chunk_memory_mb=32)
            baseline += original
            field += (
                current * biot_savart_wire_path(design.path, self.target.coords, worker_count=1, max_chunk_memory_mb=32)
                if i == self.part
                else original
            )
        mask = ~self.target.suppressed
        weights = np.where(mask, self.target.weights, 0.0)
        denom = float(np.sum(weights))
        if denom <= 0:
            raise ValueError("No positively weighted target samples are available.")

        def rms(a):
            return float(np.sqrt(np.sum(weights[:, None] * a * a) / denom))

        # Pad/via resistance requires a real drill and plating specification.
        length = sum(
            float(
                np.sum(
                    np.linalg.norm(
                        np.diff(design.frame.path_points(t.uv, t.layer, design.settings.layer_gap_m), axis=0), axis=1
                    )
                )
            )
            for t in design.traces
        )
        resistance = (
            float(self.config.specific_conductivity_conductor)
            * length
            / (design.settings.width_m * design.settings.copper_m)
        )
        current = float(self.parts[self.part].contour_step)
        check = field_budget_check(
            SimpleNamespace(input_data=self.config, target_field=self.target),
            field,
            self.target.b,
            weights=weights,
            design_margin=self.design_margin,
        )
        self.evaluation = dict(
            identity=self.identity,
            delta_rms_uT=rms(field - baseline) * 1e6,
            target_rms_uT=rms(field - self.target.b) * 1e6,
            current_A=current,
            trace_resistance_ohm=resistance,
            trace_power_W=current * current * resistance,
            field_budget=check,
            field_budget_summary=(
                f"Saved field budget: {'passed' if check['passed'] else 'failed'} · "
                f"{check['component']} · {check['value'] * 1e6:.4g} / {check['effective_limit'] * 1e6:.4g} µT RMS"
                if check.get("verified") and check.get("value") is not None
                else "Saved field budget: not verified · " + check.get("reason", "No supported acceptance criterion.")
            ),
            scope="Selected PCB part at saved current; other parts retain saved paths. Thin-wire field; trace resistance excludes vias and contacts.",
        )
        return self.evaluation

    def export(self, path, formats=("svg", "dxf")):
        """A new ZIP package only; no mutable source artifact is overwritten."""
        if self.design is None:
            raise ValueError(self.problem)
        if not formats or any(f not in {"svg", "dxf"} for f in formats):
            raise ValueError("Select SVG, DXF, or both.")
        path = Path(path)
        if path.exists():
            raise FileExistsError("Choose a new PCB package filename.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent, ignore_cleanup_errors=True) as tmp:
            output = Path(tmp)
            prefix = f"part{self.part + 1}"
            for name, shape in (
                ("top", self.design.copper["top"]),
                ("bottom", self.design.copper["bottom"]),
                ("outline", self.design.board),
            ):
                if "svg" in formats:
                    (output / f"{prefix}_{name}.svg").write_text(
                        _svg(shape, self.design.frame.size, outline=name == "outline"), encoding="utf-8"
                    )
                if "dxf" in formats:
                    (output / f"{prefix}_{name}.dxf").write_text(_dxf(shape, name), encoding="ascii")
            (output / "layout.cgpcb").write_text(
                json.dumps(self.document(), indent=2, allow_nan=False), encoding="utf-8"
            )
            manifest = dict(
                schema="coilgen.pcb-export/1",
                identity=self.identity,
                source_id=self.source_id,
                part=self.part,
                units="mm",
                scale=1,
                viewing="Both copper layers viewed from top; not mirrored.",
                scope="PCB CAD artwork; not a fabrication release",
                settings=asdict(self.design.settings),
                checks=[asdict(i) for i in self.design.issues],
                evaluation=self.evaluation,
                review=review_summary(self.design, self.evaluation, self.problem),
                via_centers_m=[p.tolist() for p in self.design.vias],
                software=software_attribution(),
            )
            manifest["files"] = {p.name: sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            (output / "pcb-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            (output / "READ_ME.txt").write_text(
                "CoilGen2 PCB CAD artwork\n\nImport at scale 1:1 in mm. All files use a common origin.\n"
                "SVG uses screen Y-down; DXF uses engineering Y-up. Both encode the same geometry.\n"
                "Top and bottom are viewed from above. Do not mirror a copper layer for PCB CAD.\n"
                "SVG copper contains filled polygons only. DXF contains closed copper outlines; assign/convert copper in PCB CAD.\n"
                "Outline files include support cutouts. Vias have landing copper but no drill/plating specification.\n"
                "Define pads, holes, layer stackup and fabrication rules in PCB CAD.\n"
                "See pcb-manifest.json for unresolved checks and source identity. This is not a fabrication release.\n",
                encoding="utf-8",
            )
            pending = output / (uuid4().hex + ".zip")
            files = list(output.iterdir())
            with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
                for p in files:
                    archive.write(p, p.name)
            from ..persistence import publish_exclusive

            publish_exclusive(pending, path)
        return path


def _svg(shape, size, *, outline=False):
    w, h = size * 1000
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.9g}mm" height="{h:.9g}mm" viewBox="0 0 {w:.9g} {h:.9g}">',
        f"<metadata>{escape(json.dumps(software_attribution()))}</metadata>",
    ]
    for group in polygon_rings(shape):
        d = " ".join("M " + " L ".join(f"{x * 1000:.9g},{h - y * 1000:.9g}" for x, y in ring) + " Z" for ring in group)
        style = (
            'fill="none" stroke="black" stroke-width="0.05"'
            if outline
            else 'fill="black" fill-rule="evenodd" stroke="none"'
        )
        lines.append(f'<path d="{d}" {style}/>')
    return "\n".join([*lines, "</svg>"])


def _dxf(shape, layer):
    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$ACADVER",
        "1",
        "AC1015",
        "9",
        "$INSUNITS",
        "70",
        "4",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]
    for group in polygon_rings(shape):
        for ring in group:
            lines.extend(_dxf_lwpolyline(ring, layer, closed=True))
    return "\n".join([*lines, "0", "ENDSEC", "0", "EOF"]) + "\n"
