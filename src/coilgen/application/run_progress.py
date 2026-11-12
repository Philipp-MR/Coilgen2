"""One progress line for a running job, shared by the desktop and browser clients.

The primary label names the pipeline stage and the elapsed time. The bar only
ever shows the current operation's own percentage (or runs indeterminate), so it
never jumps between the operation scale and the overall work estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Real

from ..pipeline_stages import PipelineStage


STAGE_SHORT_LABELS = {
    "prepared": "Prepare",
    "sheet": "Stream function",
    "contours": "Contours",
    "routed": "Route / sweep",
    "evaluated": "Evaluate",
}
_KEYS = tuple(stage.value for stage in PipelineStage)


@dataclass(frozen=True, slots=True)
class RunProgress:
    label: str = ""
    detail: str = ""
    percent: int | None = None
    stage: str | None = None
    elapsed_seconds: int | None = None


def fraction_percent(value):
    """A fraction in [0, 1] as a whole percentage; 100 only when exactly complete."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return min(99, round(value * 100)) if value < 1 else 100


def parse_timestamp(value):
    if not isinstance(value, str) or not ("T" in value or " " in value):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
    except (ValueError, OverflowError):
        return None


def clock(seconds):
    """3:10 or 1:02:03."""
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def current_stage(state):
    """The stage the worker reports, else the one after the last completed stage."""
    active = state.get("active_stage")
    if active in _KEYS:
        return active
    completed = state.get("completed_stage")
    if completed in _KEYS:
        return _KEYS[min(_KEYS.index(completed) + 1, len(_KEYS) - 1)]
    return None


def run_progress(state, *, now=None):
    """Primary label, secondary operation text and bar value for one job."""
    target = state.get("pipeline_stage")
    total = _KEYS.index(target) + 1 if target in _KEYS else len(_KEYS)
    stage = current_stage(state)
    parts = []
    if stage is not None:
        number = min(_KEYS.index(stage) + 1, total)
        stage = _KEYS[number - 1]
        parts.append(f"Stage {number}/{total} · {STAGE_SHORT_LABELS[stage]}")
    elapsed = None
    started = parse_timestamp(state.get("started_at"))
    if started is not None and state.get("status") != "queued":
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        value = (now - started).total_seconds()
        if value >= -5:
            elapsed = max(0, int(value))
            parts.append(f"{clock(elapsed)} elapsed")
    label = " · ".join(parts) if stage is not None else " · ".join(["Computing", *parts])
    percent = None
    if state.get("status") == "running" and not state.get("progress_indeterminate"):
        percent = fraction_percent(state.get("phase_progress"))
    detail = f"Operation {percent}%" if percent is not None else ""
    return RunProgress(label, detail, percent, stage, elapsed)
