from __future__ import annotations

import time
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator


class DesignCancelled(RuntimeError):
    """Raised at a numerical stage boundary when a worker is cancelled."""


_progress_sink: ContextVar[Callable[[dict], None] | None] = ContextVar("coilgen_progress_sink", default=None)
_cancel_check: ContextVar[Callable[[], bool] | None] = ContextVar("coilgen_cancel_check", default=None)


def update_work_progress(state: dict, event: dict, *, completion_limit: float = 0.99) -> None:
    """Store overall/local work separately; completion evidence belongs to callers."""
    from .pipeline_stages import PipelineStage

    state["stage"] = str(event.get("message", "Computing"))[:512]
    if event.get("completed_stage") is not None:
        return
    value = event.get("overall_percent")
    if value is None and event.get("label") == "CoilGen" and event.get("progress_kind") != "allocated_work":
        value = event.get("percent")  # Older workers only report pipeline progress.
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        state["progress"] = max(state.get("progress", 0.0), min(completion_limit, max(0.0, value / 100)))
    local = event.get("phase_percent", event.get("percent"))
    known = isinstance(local, (int, float)) and not isinstance(local, bool) and math.isfinite(local)
    state.update(
        phase_progress=min(1.0, max(0.0, local / 100)) if known else None,
        progress_indeterminate=bool(event.get("progress_indeterminate", not known)),
        progress_label=str(event.get("label", "Computing"))[:128],
        progress_kind=event.get("progress_kind", "reported_work"),
    )
    for key in ("completed", "total"):
        item = event.get(key)
        state["progress_" + key] = item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None
    if event.get("active_stage") in {stage.value for stage in PipelineStage}:
        state["active_stage"] = event["active_stage"]


class ProgressRelay:
    """Bound callback traffic while retaining the last real worker update."""

    def __init__(self, emit):
        self.emit = emit
        self.last_time = float("-inf")
        self.identity = None
        self.pending = None

    def report(self, event):
        if "progress_indeterminate" not in event:
            local = event.get("phase_percent", event.get("percent", event.get("overall_percent")))
            event = {**event, "progress_indeterminate": local is None}
        now = time.monotonic()
        identity = (
            event.get("active_stage"),
            event.get("scope_label"),
            event.get("label"),
            event.get("progress_indeterminate"),
        )
        overall = event.get("overall_percent")
        # Entering opaque work must reach observers before the blocking call.
        # Repeated opaque entries remain coalesced while observers are already
        # busy, so rapid candidate scopes do not flood the event queue.
        entering_busy = bool(event.get("progress_indeterminate")) and (self.identity is None or not self.identity[-1])
        new_stage = identity[0] is not None and (self.identity is None or identity[0] != self.identity[0])
        critical = (
            event.get("completed_stage") is not None
            or isinstance(overall, (int, float))
            and overall >= 100
            or entering_busy
            or new_stage
        )
        changed = identity != self.identity
        self.pending = event
        if critical or now - self.last_time >= (0.05 if changed else 0.1):
            self.flush(now)

    def flush(self, now=None):
        if self.pending is None:
            return
        event, self.pending = self.pending, None
        self.last_time = time.monotonic() if now is None else now
        self.identity = (
            event.get("active_stage"),
            event.get("scope_label"),
            event.get("label"),
            event.get("progress_indeterminate"),
        )
        self.emit(event)


@contextmanager
def progress_context(
    sink: Callable[[dict], None],
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[None]:
    """Capture progress in this execution context without changing solver inputs."""
    sink_token = _progress_sink.set(sink)
    cancel_token = _cancel_check.set(cancel_check)
    try:
        yield
    finally:
        _cancel_check.reset(cancel_token)
        _progress_sink.reset(sink_token)


@contextmanager
def progress_scope(
    start_percent: float,
    end_percent: float,
    message: str = "",
    *,
    label: str | None = None,
    stage: str | None = None,
    overall: bool = False,
) -> Iterator[None]:
    """Map nested completed work into one explicit portion of an operation.

    Allocations estimate overall work, not remaining wall time. An opaque
    operation keeps its completed-work floor and marks its local progress
    indeterminate. Only a normally returned scope completes its allocation;
    cancelled/failed work never advances to the end. Scientific boundaries
    pass through unchanged and remain independent of percentages.
    """
    start, end = float(start_percent), float(end_percent)
    tolerance = 1e-12  # Allocating thirds can produce 100.00000000000001.
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < -tolerance
        or end > 100 + tolerance
        or start > end + tolerance
    ):
        raise ValueError("Progress scopes require finite ordered percentages from 0 to 100.")
    start = min(100.0, max(0.0, start))
    end = min(100.0, max(start, end))
    parent = _progress_sink.get()
    if parent is None:
        yield
        return
    previous = start

    def mapped(event):
        nonlocal previous
        if event.get("completed_stage") is not None:
            parent(event)
            return
        value = event.get("percent")
        known = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        fraction = min(100.0, max(0.0, float(value))) / 100 if known else 0.0
        previous = max(previous, min(end, start + (end - start) * fraction))
        result = {
            **event,
            "percent": previous,
            "phase_percent": event.get("phase_percent", value),
            "progress_indeterminate": bool(event.get("progress_indeterminate", False) or not known),
            "progress_kind": "allocated_work",
        }
        if label is not None:
            result["scope_label"] = label
        if stage is not None:
            result["active_stage"] = stage
        if overall:
            result["overall_percent"] = previous
        elif "overall_percent" in result:
            result["overall_percent"] = previous
        parent(result)

    token = _progress_sink.set(mapped)
    reporter = ProgressReporter(label=label or "CoilGen")
    try:
        reporter.report(None, message or label or "Computing")
        yield
        reporter._report_event(
            {
                "label": label or "CoilGen",
                "percent": 100.0,
                "message": message or label or "Operation completed",
                "scope_completed": True,
            }
        )
    finally:
        _progress_sink.reset(token)


def format_elapsed(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 1.0:
        return f"{seconds:.3f} s"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)} min {remainder:.1f} s"


@dataclass
class ProgressReporter:
    enabled: bool = False
    label: str = "CoilGen"
    min_interval_seconds: float = 1.0
    percent_step: float = 5.0
    _start_time: float = field(default_factory=time.perf_counter, init=False, repr=False)

    @classmethod
    def from_config(cls, config: object, *, label: str = "CoilGen") -> "ProgressReporter":
        return cls(
            enabled=bool(getattr(config, "progress_flag", False)),
            label=label,
            min_interval_seconds=float(getattr(config, "progress_min_interval_seconds", 1.0)),
            percent_step=float(getattr(config, "progress_percent_step", 5.0)),
        )

    def report(self, percent: float | None, message: str, *, force: bool = False) -> None:
        """Report activity without a percentage when its total work is unknown."""
        percent = None if percent is None else max(0.0, min(100.0, float(percent)))
        self._report_event({"label": self.label, "percent": percent, "message": message})

    def boundary(self, stage: str, *, preview: str | None = None, preview_error: str | None = None) -> None:
        """Record a completed scientific boundary independently of task percentages."""
        from .pipeline_stages import PipelineStage

        selected = PipelineStage(stage)
        event = {
            "label": "CoilGen",
            "percent": None,
            "message": f"{selected.value} boundary completed",
            "completed_stage": selected.value,
            "completed_stages": [item.value for item in PipelineStage if item.index <= selected.index],
        }
        if preview is not None:
            event["stage_preview"] = preview
        if preview_error is not None:
            event["stage_preview_error"] = str(preview_error)[:512]
        self._report_event(event)

    def _report_event(self, event: dict) -> None:
        cancel = _cancel_check.get()
        if cancel is not None and cancel():
            raise DesignCancelled("Design cancelled by the user.")
        sink = _progress_sink.get()
        if sink is not None:
            sink(event)
        if not self.enabled:
            return
        percent, label = event["percent"], event["label"]
        prefix = label if percent is None else f"{label} {percent:6.1f}%"
        print(f"{prefix} - {event['message']}", flush=True)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    def finish(self, message: str = "finished") -> None:
        self.report(100.0, f"{message}; execution time {format_elapsed(self.elapsed_seconds())}", force=True)

    def task(self, label: str, total: int, *, indeterminate: bool = False) -> "ProgressTask":
        return ProgressTask(
            reporter=self,
            label=label,
            total=max(int(total), 0),
            min_interval_seconds=max(float(self.min_interval_seconds), 0.0),
            percent_step=max(float(self.percent_step), 0.0),
            indeterminate=indeterminate,
        )


@dataclass
class ProgressTask:
    reporter: ProgressReporter
    label: str
    total: int
    min_interval_seconds: float
    percent_step: float
    indeterminate: bool = False
    completed: int = 0
    _start_time: float = 0.0
    _last_time: float = 0.0
    _last_percent: float = -1.0

    def __post_init__(self) -> None:
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self.update(0, force=True)

    def advance(self, amount: int = 1, *, message: str | None = None, force: bool = False) -> None:
        self.update(self.completed + int(amount), message=message, force=force)

    def update(self, completed: int, *, message: str | None = None, force: bool = False) -> None:
        if not self.reporter.enabled and _progress_sink.get() is None and _cancel_check.get() is None:
            self.completed = max(int(completed), 0)
            return
        self.completed = min(max(int(completed), 0), self.total)
        percent = 100.0 if self.total <= 0 else 100.0 * float(self.completed) / float(self.total)
        now = time.perf_counter()
        # Interactive sinks need useful updates even when console reporting is
        # disabled or configured with coarse logging thresholds.
        interval = (
            min(self.min_interval_seconds, 0.2) if _progress_sink.get() is not None else self.min_interval_seconds
        )
        step = min(self.percent_step, 1.0) if _progress_sink.get() is not None else self.percent_step
        enough_percent = self._last_percent < 0.0 or percent >= self._last_percent + step
        enough_time = now >= self._last_time + interval
        is_complete = self.completed >= self.total
        if force or enough_percent or enough_time or is_complete:
            text = message if message is not None else f"{self.completed}/{self.total}"
            if self.indeterminate:
                text = f"{self.label} {self.completed}/{self.total}" + (f": {message}" if message is not None else "")
            else:
                text = f"{self.label}: {text}"
            self.reporter._report_event(
                {
                    "label": self.reporter.label,
                    "percent": None if self.indeterminate else percent,
                    "message": text,
                    "completed": self.completed,
                    "total": self.total,
                }
            )
            self._last_percent = percent
            self._last_time = now

    def done(self, *, message: str = "done") -> None:
        elapsed = time.perf_counter() - self._start_time
        self.update(self.total, message=f"{message}; execution time {format_elapsed(elapsed)}", force=True)
