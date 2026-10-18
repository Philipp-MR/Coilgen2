"""Pure presentation of an independent-channel design plan.

No files, Qt objects or numerical kernels are opened here. Records and reuse
plans must already have passed the application's evidence checks. Structured
checkpoint boundaries establish computation; a percentage never does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .runtime_feedback import (
    RuntimeFeedback,
    fraction_percent as _percent,
    runtime_feedback,
    work_detail as _work_detail,
)
from .run_metadata import read_run_metadata


_STAGES = (
    ("prepared", "Prepare model", "Prepare support meshes and target samples", "mesh"),
    ("sheet", "Current sheet", "Optimize the continuous current sheet", "stream-function"),
    ("contours", "Contour paths", "Generate contour winding paths", "contours"),
    ("routed", "Route / sweep", "Connect winding paths and construct conductors", "conductor-sweep"),
    ("evaluated", "Evaluate field", "Evaluate fields and requested engineering checks", "independent-validation"),
)
_KEYS = tuple(stage[0] for stage in _STAGES)
_ACTIVE = frozenset({"queued", "running", "cancelling"})
_STOPPED = frozenset({"failed", "cancelled", "interrupted"})
_STAGE_TEXT = {
    "complete": "Recorded stage computation completed; engineering acceptance is separate",
    "reusable": "Verified compatible checkpoint available; the worker rechecks it before reuse",
    "current": "Current computation stage",
    "waiting": "Awaiting computation",
    "skipped": "Beyond this study's requested boundary",
    "failed": "Execution stopped with an error at this stage",
    "stopped": "Execution stopped before this stage completed",
    "stale": "Earlier model/study evidence; not completed for the current applied inputs",
}


@dataclass(frozen=True, slots=True)
class StagePlan:
    key: str
    label: str
    description: str
    icon_name: str
    status: str
    tooltip: str
    viewed: bool = False
    inferred: bool = False


@dataclass(frozen=True, slots=True)
class ChannelPlan:
    channel_id: str
    name: str
    project_id: str
    revision: int
    enabled: bool
    selected: bool
    requested_stage: str
    stages: tuple[StagePlan, ...]
    status: str
    activity: str
    percent: int | None
    busy: bool
    job_id: str | None
    completed_count: int
    requested_count: int
    reuse_from: str | None
    historical_count: int
    verification: str | None
    engineering_errors: int | None
    review_needed: bool
    tooltip: str
    pending: bool
    runtime: RuntimeFeedback = RuntimeFeedback()


@dataclass(frozen=True, slots=True)
class DesignPlanState:
    name: str
    channels: tuple[ChannelPlan, ...]
    completed_count: int
    requested_count: int
    completed_channels: int
    enabled_channels: int
    running_channels: int
    queued_channels: int
    failed_channels: int
    waiting_channels: int
    disabled_channels: int
    summary: str
    tooltip: str


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _items(values):
    return tuple(values.values() if isinstance(values, Mapping) else values or ())


def _records(values):
    entries = values.items() if isinstance(values, Mapping) else ((None, value) for value in values or ())
    records = []
    for id_, record in entries:
        metadata = dict(read_run_metadata(record))
        job = metadata.get("job") or {}
        for key in ("project_id", "project_fingerprint", "project_revision", "pipeline_stage"):
            if not metadata.get(key) and job.get(key) is not None:
                metadata[key] = job[key]
        metadata["id"] = (
            id_
            or metadata.get("id")
            or job.get("id")
            or getattr(record, "alias", None)
            or getattr(getattr(record, "directory", None), "name", None)
        )
        metadata["archive_id"] = getattr(record, "id", None)
        records.append(metadata)
    return tuple(records)


def _matches(metadata, project, *, revision=False):
    return (
        metadata.get("project_id") == project.id
        and metadata.get("project_fingerprint") == project.fingerprint
        and (not revision or metadata.get("project_revision") == project.revision)
    )


def _boundaries(value, target):
    if not isinstance(value, (tuple, list, set, frozenset)):
        return set()
    return {key for key in value if isinstance(key, str) and key in _KEYS[: _KEYS.index(target) + 1]}


def _record_boundaries(metadata):
    target = metadata.get("pipeline_stage")
    return set(_KEYS[: _KEYS.index(target) + 1]) if target in _KEYS else set()


def _latest(states):
    # The manager returns newest first, but callers may supply a dictionary.
    # ISO UTC timestamps make the primary ordering explicit; input order breaks
    # ties without comparing unrelated job names or identifiers.
    return max(
        enumerate(states),
        key=lambda item: (_text(item[1].get("created_at")), -item[0]),
        default=(None, {}),
    )[1]


def _lookup(mapping, channel_id, project_id, default=None):
    return mapping.get(channel_id, mapping.get(project_id, default))


def _review_needed(records):
    from ..application.diagnostics import run_verdicts

    for metadata in records:
        if "Failed" in run_verdicts(metadata).values() or (metadata.get("summary") or {}).get("accepted") is False:
            return True
        if any(row.get("passed") is False for row in metadata.get("constraints", ())):
            return True
        fabrication = metadata.get("fabrication") or {}
        if fabrication.get("manufacturing_ok") is False or fabrication.get("failure_reasons"):
            return True
        if any(body.get("manufacturing_ok") is False for body in fabrication.get("paths", ())):
            return True
    return False


def _channel_plan(
    channel,
    *,
    selected,
    states,
    records,
    reuse,
    errors,
    requested_stage,
    pending,
    unavailable,
    engineering_errors,
    now=None,
):
    project = channel.project
    current = [state for state in states if _matches(state, project, revision=True)]
    state = _latest([item for item in current if item.get("status") in _ACTIVE] or current)
    status = state.get("status", "idle")
    target = state.get("pipeline_stage") or requested_stage or project.study["pipeline_stage"]
    if target not in _KEYS:
        raise ValueError(f"Unknown design stage: {target!r}")
    allowed = set(_KEYS[: _KEYS.index(target) + 1])
    job_id = state.get("id")
    source_error = errors.get(job_id, "") if job_id else ""
    valid_records, historical, stale = [], set(), set(getattr(reuse, "stale", ()))
    rejected_current_error = ""
    for metadata in records:
        id_ = metadata.get("id")
        if metadata.get("project_id") != project.id:
            continue
        record_error = errors.get(id_) or errors.get(metadata.get("archive_id"))
        if record_error:
            if _matches(metadata, project):
                rejected_current_error = record_error
            continue
        if _matches(metadata, project):
            valid_records.append(metadata)
        elif metadata.get("project_fingerprint"):
            historical.add(metadata.get("request_sha256") or id_ or id(metadata))
            stale.update(_record_boundaries(metadata))
    for item in states:
        if item.get("project_id") == project.id and item.get("project_fingerprint") != project.fingerprint:
            boundaries = _boundaries(item.get("completed_stages", ()), target)
            if boundaries and not errors.get(item.get("id")):
                stale.update(boundaries)
                historical.add(item.get("request_sha256") or item.get("id") or id(item))
    if not state and not valid_records:
        source_error = rejected_current_error
    recorded = _boundaries(state.get("completed_stages", ()), target) if not source_error else set()
    own_records = (
        [
            record
            for record in valid_records
            if record.get("id") == job_id
            or (state.get("request_sha256") and record.get("request_sha256") == state["request_sha256"])
        ]
        if state
        else valid_records
    )
    verification = "failed" if source_error else "verified" if own_records else None
    if own_records:
        recorded.update(set().union(*(_record_boundaries(record) for record in own_records)) & allowed)
    elif status == "completed" and not source_error:
        verification = "pending"
    cached = {
        getattr(entry.stage, "value", entry.stage)
        for entry in getattr(reuse, "available", ())
        if getattr(entry.stage, "value", entry.stage) in allowed
    }
    reuse_from = max(cached, key=_KEYS.index, default=None)
    remaining = [key for key in _KEYS if key in allowed and key not in recorded]
    explicit_active = state.get("active_stage") in remaining
    active = state["active_stage"] if explicit_active else remaining[0] if remaining else None
    marking = not unavailable and status in {"running", "cancelling", *_STOPPED}
    stages = []
    for key, label, description, icon_name in _STAGES:
        if key == "evaluated":
            controls = {**project.study, **state}
            description += f"; independent sheet field validation resolution {controls['validation_resolution']}"
            if controls.get("fabricate"):
                description += "; strict conductor geometry review"
            if any(
                controls.get(name)
                for name in ("engineering", "requirements", "scene", "dynamic_analysis", "array_study")
            ):
                description += "; additional configured research / engineering studies"
        stage_status = (
            "skipped"
            if key not in allowed
            else "complete"
            if key in recorded
            else "failed"
            if marking and key == active and status == "failed"
            else "stopped"
            if marking and key == active and status in {"cancelled", "interrupted"}
            else "current"
            if marking and key == active
            else "reusable"
            if key in cached
            else "stale"
            if key in stale
            else "waiting"
        )
        inferred = marking and key == active and not explicit_active
        tooltip = f"{description}. {_STAGE_TEXT[stage_status]}."
        if inferred:
            tooltip += " Next unrecorded requested boundary; the worker operation below identifies actual activity."
        stages.append(StagePlan(key, label, description, icon_name, stage_status, tooltip, inferred=inferred))
    count = len(recorded)
    if not state and count == len(allowed) and verification == "verified":
        status = "completed"
    elif not state and stale and not recorded and not cached:
        status = "stale"
    runtime = runtime_feedback(state, now=now)
    if verification == "failed":
        activity = "Saved evidence verification failed"
    elif verification == "pending":
        activity = "Execution finished · verifying saved evidence"
    elif status == "queued":
        activity = "Queued · waiting for a worker"
    elif status == "running":
        activity = "Computing · " + (runtime.operation or "worker operation unreported")
    elif status == "cancelling":
        activity = "Cancelling · waiting for worker stop" + (" · " + runtime.operation if runtime.operation else "")
    elif status == "failed":
        activity = "Execution failed · " + (_text(state.get("error")) or "see Messages")
    elif status in {"cancelled", "interrupted"}:
        activity = status.capitalize() + " · earlier recorded stages retained"
    elif status == "completed":
        activity = (
            "Computation complete · review engineering checks" if count == len(allowed) else "Recorded stages available"
        )
    elif status == "stale":
        activity = "Earlier model/study evidence · current stages await computation"
    else:
        activity = "Ready · verified checkpoint available" if cached else "Stages await computation"
    if status in {"running", "cancelling"} and (detail := _work_detail(state)):
        activity += " · " + detail
    if pending:
        activity += " · unapplied edits"
    if not channel.enabled:
        activity += " · excluded from system compute"
    percent = _percent(state.get("progress")) if status in _ACTIVE and not unavailable else None
    busy = (
        verification == "pending"
        or status == "cancelling"
        or (status == "running" and (percent is None or state.get("progress_indeterminate", False)))
    )
    if unavailable and state:
        activity = "Runtime unavailable · last known: " + activity
        status, busy, percent = "unavailable", False, None
    review_needed = bool(engineering_errors) or _review_needed(own_records)
    tooltip = (
        f"{channel.name} · applied revision {project.revision}. {activity}. "
        f"{count}/{len(allowed)} requested stages have recorded computation. "
        "Completion establishes calculation, not engineering acceptance. "
        "Channels are independent computations; no combined electromagnetic or manufacturing acceptance is implied."
    )
    if historical:
        tooltip += (
            f" {len(historical)} earlier model/study run(s) remain in History; they do not complete current steps."
        )
    if unavailable:
        tooltip += " " + _text(unavailable)
    if source_error:
        tooltip += " " + _text(source_error)
    if runtime.line and not unavailable:
        tooltip += " " + runtime.line + ". " + runtime.tooltip
    return ChannelPlan(
        channel.id,
        channel.name,
        project.id,
        project.revision,
        channel.enabled,
        selected,
        target,
        tuple(stages),
        status,
        activity,
        percent,
        busy,
        job_id,
        count,
        len(allowed),
        reuse_from,
        len(historical),
        verification,
        engineering_errors,
        review_needed,
        tooltip,
        pending,
        RuntimeFeedback(operation=runtime.operation) if unavailable else runtime,
    )


def build_design_plan(
    assembly_or_project,
    *,
    active_channel_id=None,
    states=(),
    records=(),
    reuse_plans=None,
    verification_errors=None,
    requested_stage=None,
    pending_channel_ids=(),
    runtime_unavailable=None,
    engineering_errors=None,
    now=None,
):
    """Describe current requested work without submitting jobs or verifying files.

    ``records`` are already verified RunRecords. ``reuse_plans`` and optional
    engineering error counts are keyed by channel or project ID. A requested
    boundary override applies only to the selected channel; other channels keep
    their own study settings. Exact-owner/revision live jobs take precedence.
    Disabled channels remain visible, while system totals exclude them.
    """
    from types import SimpleNamespace

    channels = getattr(assembly_or_project, "channels", None)
    if channels is None:
        project = assembly_or_project
        channels = (SimpleNamespace(id=project.id, name=project.name, project=project, enabled=True),)
    channels = tuple(channels)
    active_channel_id = active_channel_id or (channels[0].id if channels else None)
    if requested_stage is not None and requested_stage not in _KEYS:
        raise ValueError(f"Unknown design stage: {requested_stage!r}")
    states, records = _items(states), _records(records)
    reuse_plans, errors, engineering_errors = reuse_plans or {}, verification_errors or {}, engineering_errors or {}
    plans = tuple(
        _channel_plan(
            channel,
            selected=channel.id == active_channel_id,
            states=states,
            records=records,
            reuse=_lookup(reuse_plans, channel.id, channel.project.id),
            errors=errors,
            requested_stage=requested_stage if channel.id == active_channel_id else None,
            pending=channel.id in pending_channel_ids,
            unavailable=runtime_unavailable,
            engineering_errors=_lookup(engineering_errors, channel.id, channel.project.id),
            now=now,
        )
        for channel in channels
    )
    enabled = tuple(channel for channel in plans if channel.enabled)
    completed = sum(channel.completed_count for channel in enabled)
    requested = sum(channel.requested_count for channel in enabled)
    complete_channels = sum(
        channel.completed_count == channel.requested_count and channel.verification == "verified" for channel in enabled
    )
    running = sum(channel.status in {"running", "cancelling"} for channel in enabled)
    queued = sum(channel.status == "queued" for channel in enabled)
    failed = sum(channel.status == "failed" or channel.verification == "failed" for channel in enabled)
    waiting = len(enabled) - complete_channels - running - queued - failed
    # A terminal error may retain all recorded stages. Categorize execution
    # separately from evidence rather than allowing a negative pending count.
    waiting = max(0, waiting)
    summary = f"{completed}/{requested} steps recorded · {complete_channels}/{len(enabled)} channels computed"
    if running or queued:
        summary += f" · {running} running · {queued} queued"
    disabled = len(plans) - len(enabled)
    if disabled:
        summary += f" · {disabled} excluded"
    tooltip = (
        summary + ". System counts include enabled channels only. Each channel keeps its own requested study boundary. "
        "Step counts are recorded checkpoints, not a duration or aggregate work percentage. "
        "Individual worker percentages estimate allocated work; engineering acceptance remains a separate review."
    )
    return DesignPlanState(
        assembly_or_project.name,
        plans,
        completed,
        requested,
        complete_channels,
        len(enabled),
        running,
        queued,
        failed,
        waiting,
        disabled,
        summary,
        tooltip,
    )
