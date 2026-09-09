from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator


class DesignCancelled(RuntimeError):
    """Raised at a numerical stage boundary when a worker is cancelled."""


_progress_sink: ContextVar[Callable[[dict], None] | None] = ContextVar("coilgen_progress_sink", default=None)
_cancel_check: ContextVar[Callable[[], bool] | None] = ContextVar("coilgen_cancel_check", default=None)


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

    def report(self, percent: float, message: str, *, force: bool = False) -> None:
        cancel = _cancel_check.get()
        if cancel is not None and cancel():
            raise DesignCancelled("Design cancelled by the user.")
        sink = _progress_sink.get()
        if sink is not None:
            sink({"label": self.label, "percent": max(0.0, min(100.0, float(percent))), "message": message})
        if not self.enabled:
            return
        print(f"{self.label} {max(0.0, min(100.0, float(percent))):6.1f}% - {message}", flush=True)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    def finish(self, message: str = "finished") -> None:
        self.report(100.0, f"{message}; execution time {format_elapsed(self.elapsed_seconds())}", force=True)

    def task(self, label: str, total: int) -> "ProgressTask":
        return ProgressTask(
            reporter=self,
            label=label,
            total=max(int(total), 0),
            min_interval_seconds=max(float(self.min_interval_seconds), 0.0),
            percent_step=max(float(self.percent_step), 0.0),
        )


@dataclass
class ProgressTask:
    reporter: ProgressReporter
    label: str
    total: int
    min_interval_seconds: float
    percent_step: float
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
        enough_percent = self._last_percent < 0.0 or percent >= self._last_percent + self.percent_step
        enough_time = now >= self._last_time + self.min_interval_seconds
        is_complete = self.completed >= self.total
        if force or enough_percent or enough_time or is_complete:
            text = message if message is not None else f"{self.completed}/{self.total}"
            self.reporter.report(percent, f"{self.label}: {text}")
            self._last_percent = percent
            self._last_time = now

    def done(self, *, message: str = "done") -> None:
        elapsed = time.perf_counter() - self._start_time
        self.update(self.total, message=f"{message}; execution time {format_elapsed(elapsed)}", force=True)
