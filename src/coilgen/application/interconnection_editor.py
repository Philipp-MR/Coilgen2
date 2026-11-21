"""Run-bound winding editing and fixed-current engineering validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import os
import uuid

import numpy as np

from ..interconnection_editor import InterconnectionEditor, PLAN_SCHEMA, validate_editor_plan
from ..interconnection_validation import validate_winding_revision
from ..persistence import json_value
from ..pipeline_stages import PipelineCheckpoint, PipelineStage
from .evidence import load_saved_request, read_verified_artifact
from .routing_analysis import routing_analysis_available


def winding_model_changes(recorded, current):
    """Compare physical inputs with worker-owned thread controls normalized.

    ResearchRequest resolves automatic thread counts when executing a run.
    Retaining those recorded counts for inspection does not authorize reuse of
    a checkpoint under a different numerical engine.
    """
    from ..pipeline_stages import changed_config_fields, config_field_stage

    inspection = replace(
        current,
        parallel_worker_count=recorded.parallel_worker_count,
        constrained_optimization_threads=recorded.constrained_optimization_threads,
    )
    return [
        name
        for name in changed_config_fields(recorded, inspection)
        if name != "routing_editor_plan"
        and config_field_stage(name, inspection) is not None
        # Compact project JSON sorts object keys; checkpoint mappings retain
        # insertion order. That transport detail is not a physical input edit.
        and json_value(getattr(recorded, name)) != json_value(getattr(inspection, name))
    ]


def editor_available(record):
    return (
        record is not None
        and routing_analysis_available(record)
        and any(item.get("name") == "pipeline_checkpoint.cgstage" for item in record.metadata_view.get("artifacts", ()))
        and record.metadata_view.get("completed_stage", "evaluated") in {"routed", "evaluated"}
    )


class WindingEditorSession:
    """Source settings stay pinned even if the active workbench model changes."""

    def __init__(self, checkpoint, *, source_directory="", design_margin=1.0):
        if checkpoint.stage not in {PipelineStage.ROUTED, PipelineStage.EVALUATED}:
            raise ValueError("Finish automatic routing before opening the interconnection editor.")
        self.config = checkpoint.config
        self.target = checkpoint.target_field
        if isinstance(design_margin, bool) or not np.isfinite(design_margin) or not 0 < design_margin <= 1:
            raise ValueError("The saved design budget ratio must lie in (0, 1].")
        self.design_margin = float(design_margin)
        self.source_directory = str(source_directory)
        self.editors = [InterconnectionEditor(part) for part in checkpoint.parts]
        self.validation = None
        if self.config.routing_editor_plan is not None:
            self.load_plan(self.config.routing_editor_plan)

    @classmethod
    def from_record(cls, record):
        checkpoint = PipelineCheckpoint.load(BytesIO(read_verified_artifact(record, "pipeline_checkpoint.cgstage")))
        request = load_saved_request(record)
        return cls(checkpoint, source_directory=record.directory, design_margin=request.get("design_margin", 1.0))

    def plan(self):
        return {"schema": PLAN_SCHEMA, "circuits": [editor.to_plan() for editor in self.editors]}

    def model_changes(self, project):
        changes = winding_model_changes(self.config, project.config)
        if self.design_margin != getattr(project, "study", {}).get("design_margin", 1.0):
            changes.append("design_margin")
        return changes

    @property
    def identity(self):
        payload = {
            "plan": self.plan(),
            "config": json_value(self.config.as_dict()),
            "design_margin": self.design_margin,
        }
        return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def validated(self):
        return self.validation is not None and self.validation.identity == self.identity

    @property
    def can_apply(self):
        return self.validated and self.validation.accepted

    def load_plan(self, plan):
        validate_editor_plan(plan)
        if len(plan["circuits"]) != len(self.editors):
            raise ValueError("Saved edits contain a different number of circuits.")
        # Loading is atomic across all circuits, including a late fingerprint error.
        drafts = [InterconnectionEditor(editor.source) for editor in self.editors]
        for editor, value in zip(drafts, plan["circuits"]):
            editor.load_plan(value)
        self.editors = drafts
        self.validation = None

    def save_draft(self, path):
        path = Path(path)
        payload = {"schema": "coilgen.winding-draft/1", "source_directory": self.source_directory, "plan": self.plan()}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load_draft(self, path):
        path = Path(path)
        if path.stat().st_size > 50 * 1024**2:
            raise ValueError("Winding draft exceeds 50 MiB.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "coilgen.winding-draft/1":
            raise ValueError("Unsupported winding draft format.")
        self.load_plan(payload["plan"])

    def validation_job(self):
        """Create a detached worker; its receipt cannot certify subsequent edits."""
        identity = self.identity
        sources = [editor.source for editor in self.editors]
        plan = self.plan()
        config, target = deepcopy(self.config), deepcopy(self.target)
        design_margin = self.design_margin

        def run():
            report = validate_winding_revision(identity, sources, plan, config, target)
            return _validate_final_field_budget(report, config, target, design_margin)

        return run

    def accept_validation(self, report):
        if report.identity != self.identity:
            return False
        self.validation = report
        return True


def _validate_final_field_budget(report, config, target, design_margin):
    """Apply the same physical field contract as the completed design review.

    Route-change and explicit layout limits alone cannot establish the final
    norm/RMS budget. Use the mounted paths at their recorded currents and the
    verified saved request's budget ratio, never the active form settings.
    """
    from types import SimpleNamespace

    from ..physics import biot_savart_wire_path
    from ..research.validation import field_budget_check

    if not report.accepted:
        return report
    reasons = list(report.reasons)
    diagnostics = dict(report.diagnostics)
    if target is None or not report.parts or any(part.fabrication_path is None for part in report.parts):
        check = {"verified": False, "passed": None, "reason": "Physical field geometry is incomplete."}
    else:
        field = sum(
            part.contour_step
            * biot_savart_wire_path(
                part.fabrication_path.wire_points_m, target.coords, worker_count=1, max_chunk_memory_mb=32.0
            )
            for part in report.parts
        )
        check = field_budget_check(
            SimpleNamespace(input_data=config, target_field=target),
            field,
            target.b,
            weights=np.where(target.suppressed, 0.0, target.weights),
            design_margin=design_margin,
        )
    diagnostics["final_field_budget"] = check
    if check.get("passed") is not True:
        reasons.append(
            "Physical winding exceeds the final field-error budget" if check.get("verified") else check["reason"]
        )
    return replace(report, accepted=not reasons, reasons=tuple(dict.fromkeys(reasons)), diagnostics=diagnostics)
