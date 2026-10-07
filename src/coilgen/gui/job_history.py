"""Display execution history without confusing wall-clock age with GUI ownership."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from collections.abc import Container


@dataclass(frozen=True, slots=True)
class TimestampDisplay:
    text: str
    tooltip: str
    timestamp: float | None


def job_origin(job_id: str, *, initial_ids: Container[str] = (), session_ids: Container[str] = ()) -> str:
    """Classify IDs explicitly; a recent timestamp cannot prove who submitted a job.

    ``initial_ids`` is the runtime snapshot before this GUI submits any work.
    ``session_ids`` contains successful submissions, including resumed studies,
    made by this GUI. Jobs discovered later without a matching submission are
    from another session. Active jobs in the initial snapshot remain historical.
    """
    if job_id in session_ids:
        return "current"
    if job_id in initial_ids:
        return "previous"
    return "other"


def format_job_timestamp(value: object, *, tz: tzinfo | None = None) -> TimestampDisplay:
    """Render a recorded ISO timestamp in the host's local timezone by default.

    The tooltip retains the full instant in local time and UTC. Naive legacy
    timestamp strings follow the existing workbench's UTC convention and are
    explicitly marked as assumed. Missing/invalid timestamps stay unknown;
    file times and run-import times are never substituted.
    """
    unknown = TimestampDisplay("Not recorded", "No valid execution timestamp is recorded.", None)
    if not isinstance(value, str) or not value.strip():
        return unknown
    # A date alone contains no recorded execution time. Do not invent midnight.
    source = value.strip()
    if "T" not in source and " " not in source:
        return unknown
    try:
        parsed = datetime.fromisoformat(source.replace("Z", "+00:00"))
        assumed_utc = parsed.tzinfo is None
        if assumed_utc:
            parsed = parsed.replace(tzinfo=timezone.utc)
        utc = parsed.astimezone(timezone.utc)
        local = parsed.astimezone(tz)
        timestamp = utc.timestamp()
    except (ValueError, OverflowError, OSError):
        return unknown
    tooltip = f"Local: {local.isoformat()} ({local.tzname()})\nUTC: {utc.isoformat()}"
    if assumed_utc:
        tooltip += "\nLegacy timestamp had no timezone offset; interpreted as UTC."
    return TimestampDisplay(local.strftime("%Y-%m-%d %H:%M:%S"), tooltip, timestamp)
