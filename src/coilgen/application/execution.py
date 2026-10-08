"""Headless execution and action plans over the existing checkpoint contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..pipeline_stages import PipelineStage
from ..persistence import json_value
from .reuse import ReusePlan, plan_reuse, plan_stopped_reuse
from .project import _freeze


ACTIVE_JOB_STATES = frozenset({"queued", "running", "cancelling"})
STOPPED_JOB_STATES = frozenset({"failed", "cancelled", "interrupted"})


def matches_project(state, project, *, revision=False):
    """A fingerprint alone cannot establish channel ownership."""
    return (
        project is not None
        and state.get("project_id") == project.id
        and state.get("project_fingerprint") == project.fingerprint
        and (not revision or state.get("project_revision") == project.revision)
    )


def run_compatibility(metadata, project=None):
    if project is None:
        return "unknown", "Identity not checked"
    job = metadata.get("job") or {}
    owner = metadata.get("project_id") or job.get("project_id")
    fingerprint = metadata.get("project_fingerprint") or job.get("project_fingerprint")
    if owner and owner != project.id:
        return "other", "Other channel"
    if not owner or not fingerprint:
        return "unknown", "Identity unverified"
    if fingerprint != project.fingerprint:
        return "earlier", "Earlier model/study"
    return "current", "Matches model/study"


def matches_active_request(state, project, controls=None):
    controls = {**project.study, **(controls or {})}
    return (
        state.get("status") in ACTIVE_JOB_STATES
        and matches_project(state, project, revision=True)
        and state.get("pipeline_stage") == controls["pipeline_stage"]
        and all(
            state.get(key) == controls[key]
            for key in ("validation_resolution", "design_margin", "fabricate", "kernel_backend")
        )
        and state.get("engineering") == controls.get("engineering")
    )


def execution_reuse(project, records=(), states=(), *, jobs_directory=None, threads=1, source_health=None):
    """Combine actual saved and stopped boundaries; live snapshots are excluded."""
    records = tuple(record for record in records if source_health is None or not source_health.error(record))
    saved = plan_reuse(project, records, threads=threads)
    jobs_directory = None if jobs_directory is None else Path(jobs_directory)
    states = tuple(
        state
        for state in states
        if source_health is None
        or jobs_directory is None
        or not isinstance(state.get("id"), str)
        or not source_health.error(jobs_directory / state["id"])
    )
    stopped = (
        plan_stopped_reuse(project, states, jobs_directory, threads=threads)
        if jobs_directory is not None
        else ReusePlan((), (), ())
    )
    return ReusePlan(
        (*saved.available, *stopped.available),
        tuple(dict.fromkeys((*saved.stale, *stopped.stale))),
        (*saved.diagnostics, *stopped.diagnostics),
    )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    project_id: str
    project_revision: int
    project_fingerprint: str
    controls: Mapping
    active_job_id: str | None
    reuse: ReusePlan

    def __post_init__(self):
        object.__setattr__(self, "controls", _freeze(json_value(self.controls, nonfinite="raise")))

    @property
    def checkpoint(self):
        return self.reuse.through(self.controls["pipeline_stage"])

    def request_controls(self):
        controls = json_value(self.controls, nonfinite="raise")
        if not controls.get("resume_checkpoint") and self.checkpoint is not None:
            controls["resume_checkpoint"] = str(self.checkpoint.path)
        return controls

    def require_current(self, project):
        if (project.id, project.revision, project.fingerprint) != (
            self.project_id,
            self.project_revision,
            self.project_fingerprint,
        ):
            raise ValueError("The execution plan belongs to an earlier project revision. Plan again before submitting.")


def plan_execution(project, *, states=(), reuse=None, **request_controls):
    """Describe execution without submitting work or changing the applied model.

    Reuse must come from verified checkpoint planning. The numerical worker
    revalidates a selected checkpoint before executing it.
    """
    controls = {**project.study, **request_controls}
    from ..research.requests import ResearchRequest

    for key, expected in (
        ("project_id", project.id),
        ("project_revision", project.revision),
        ("project_fingerprint", project.fingerprint),
        ("project_model_fingerprint", project.model_fingerprint),
    ):
        if key in controls and controls[key] != expected:
            raise ValueError(f"Execution {key} must identify this project revision.")
    ResearchRequest(name=project.name, config=project.values, **controls)
    controls["pipeline_stage"] = PipelineStage(controls["pipeline_stage"]).value
    active = next((state for state in states if matches_active_request(state, project, controls)), {})
    return ExecutionPlan(
        project.id,
        project.revision,
        project.fingerprint,
        controls,
        active.get("id"),
        reuse if reuse is not None else ReusePlan((), (), ()),
    )


@dataclass(frozen=True, slots=True)
class RunActions:
    can_cancel: bool
    can_view_results: bool
    resume_stage: PipelineStage | None
    reason: str = ""

    @property
    def can_resume(self):
        return self.resume_stage is not None


def run_actions(
    stage,
    *,
    state=None,
    record=None,
    reuse=None,
    source_error="",
    submitting=False,
    runtime_available=True,
):
    """Keep execution, source integrity and compatible recovery distinct."""
    stage = PipelineStage(stage).value
    state = state or {}
    reuse = reuse if reuse is not None else ReusePlan((), (), ())
    candidate = None if source_error or submitting or state.get("status") in ACTIVE_JOB_STATES else reuse.through(stage)
    return RunActions(
        runtime_available and state.get("status") in {"queued", "running"},
        record is not None and not source_error and state.get("status") in {None, "completed"},
        candidate.stage if candidate is not None else None,
        "Saved evidence failed verification: " + source_error
        if source_error
        else reuse.diagnostics[0]
        if reuse.diagnostics
        else "No compatible verified checkpoint is available."
        if candidate is None
        else "",
    )


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Typed delivery of existing engine events, without parsing activity text."""

    kind: str
    message: str = ""
    percent: float | None = None
    completed_stage: PipelineStage | None = None
    payload: Mapping | None = None

    def __post_init__(self):
        object.__setattr__(self, "payload", _freeze(dict(self.payload or {})))

    @classmethod
    def from_progress(cls, event):
        boundary = event.get("completed_stage")
        return cls(
            "stage_completed" if boundary is not None else "progress",
            event.get("message", ""),
            event.get("percent"),
            PipelineStage(boundary) if boundary is not None else None,
            event,
        )
