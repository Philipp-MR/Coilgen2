"""Recorded operation, elapsed time and process observations for passive views."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Real


@dataclass(frozen=True, slots=True)
class RuntimeFeedback:
    operation: str = ""
    line: str = ""
    tooltip: str = ""
    phase_percent: int | None = None
    allocated: bool = False


def fraction_percent(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return min(99, round(value * 100)) if value < 1 else 100


def work_detail(state):
    """Keep local measurable units distinct from the allocated overall floor."""
    if state.get("progress_indeterminate"):
        return "current operation has no measurable work total"
    percent = fraction_percent(state.get("phase_progress"))
    detail = f"operation {percent}%" if percent is not None else ""
    completed, total = state.get("progress_completed"), state.get("progress_total")
    if (
        isinstance(completed, int)
        and not isinstance(completed, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and 0 <= completed <= total
        and total > 0
    ):
        detail += (" · " if detail else "") + f"{completed}/{total} work units"
    return detail


def _timestamp(value):
    if not isinstance(value, str) or not ("T" in value or " " in value):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
    except (ValueError, OverflowError):
        return None


def _seconds(start, end):
    if start is None or end is None:
        return None
    value = (end - start).total_seconds()
    return max(0, int(value)) if value >= -5 else None


def _duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def runtime_feedback(state, *, now=None):
    """A process observation is liveness evidence, never solver advancement."""
    main = state.get("progress_operation")
    substep = state.get("stage")
    main = " ".join(main.split()) if isinstance(main, str) else ""
    substep = " ".join(substep.split()) if isinstance(substep, str) else ""
    operation = main + (" · " + substep if substep and substep != main else "") if main else substep
    phase_percent = None if state.get("progress_indeterminate") else fraction_percent(state.get("phase_progress"))
    allocated = state.get("progress_kind") == "allocated_work"
    status = state.get("status")
    if status not in {"queued", "running", "cancelling", "completed", "failed", "cancelled", "interrupted"}:
        return RuntimeFeedback(operation=operation, phase_percent=phase_percent, allocated=allocated)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    active = status in {"queued", "running", "cancelling"}
    end = now if active else _timestamp(state.get("completed_at")) or _timestamp(state.get("updated_at"))
    started = _timestamp(state.get("started_at"))
    submitted = _timestamp(state.get("created_at"))
    elapsed = _seconds(started or submitted, end)
    parts, notes = [], []
    if state.get("persistence_error"):
        parts.append("Job state not saved · retrying")
        notes.append(
            "Worker supervision continues, but restart recovery lacks the latest state until saving succeeds: "
            + str(state["persistence_error"])
        )
    if elapsed is not None:
        parts.append(("Elapsed " if started else "Since submission ") + _duration(elapsed))
        notes.append(
            "Elapsed starts at actual worker start." if started else "Submission age includes waiting for a worker."
        )
    if status == "queued":
        parts.append("Waiting for a worker")
    elif active:
        observed = _seconds(_timestamp(state.get("worker_observed_at")), now)
        if observed is not None and isinstance(state.get("worker_alive"), bool):
            parts.append(
                ("Worker observed alive " if state["worker_alive"] else "Worker observed stopped ")
                + _duration(observed)
                + " ago"
            )
            notes.append("Process liveness does not measure CPU activity, numerical advancement or completion.")
        elif started or state.get("last_progress_at"):
            parts.append("Worker liveness unreported")
        report_age = _seconds(_timestamp(state.get("last_progress_at")), now)
        if report_age is not None:
            parts.append("Last progress " + _duration(report_age) + " ago")
        elif (updated := _seconds(_timestamp(state.get("updated_at")), now)) is not None:
            parts.append("Status updated " + _duration(updated) + " ago")
            notes.append("A status-file update is not a numerical progress event.")
        if status == "cancelling":
            parts.append("Cancellation requested; waiting for worker stop")
        operation_age = _seconds(_timestamp(state.get("operation_started_at")), now)
        if operation_age is not None:
            notes.append("Current operation elapsed: " + _duration(operation_age) + ".")
    return RuntimeFeedback(operation, " · ".join(parts), " ".join(notes), phase_percent, allocated)
