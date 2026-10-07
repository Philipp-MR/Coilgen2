"""Phase-local routing throughput, without claiming a full-search completion ETA."""

from __future__ import annotations

from math import ceil
from time import perf_counter

from .progress import DesignCancelled, ProgressReporter, _cancel_check, format_elapsed


def _format_eta(seconds: float) -> str:
    rounded = ceil(seconds) if seconds < 60.0 else 10 * ceil(seconds / 10.0)
    if rounded < 60:
        return f"{rounded} s"
    minutes, remainder = divmod(rounded, 60)
    return f"{minutes} min {remainder} s"


class RoutingProgress:
    def __init__(self, reporter: ProgressReporter, prefix: str = ""):
        self.reporter = reporter
        self.prefix = prefix.strip()
        self.phase = ""
        self.total: int | None = None
        self.completed = 0
        self.upper_bound = False
        self._finished = False
        self._finish_status = "finished"
        self._started = self._last_completed = 0.0
        self._timed_units = 0
        self._timed_seconds = 0.0
        self._last_reported = 0.0
        self._last_coverage = 0.0

    def start(self, phase: str, total: int | None, *, upper_bound: bool = False) -> None:
        self.phase = phase
        self.total = None if total is None else max(int(total), 0)
        self.completed = 0
        self.upper_bound = upper_bound
        self._finished = False
        self._finish_status = "finished"
        self._started = self._last_completed = perf_counter()
        self._timed_units = 0
        self._timed_seconds = 0.0
        self._last_reported = self._started
        self._last_coverage = 0.0
        self._report(self._started, force=True)

    def update(self, completed: int, *, message: str = "", force: bool = False) -> None:
        now = perf_counter()
        count = max(int(completed), 0)
        if count < self.completed:
            raise ValueError("Routing phase completion counts must be monotone; start a new phase to reset them.")
        added = count - self.completed
        if added:
            elapsed = max(now - self._last_completed, 0.0)
            if elapsed > 0.0:
                self._timed_units += added
                self._timed_seconds += elapsed
            self._last_completed = now
        self.completed = count
        self._report(now, message=message, force=force)

    def activity(self, message: str = "", *, force: bool = False) -> None:
        self._report(perf_counter(), message=message, force=force)

    def finish(self, message: str = "finished", *, status: str = "finished") -> None:
        if status not in {"finished", "stopped"}:
            raise ValueError("Routing phase finish status must be finished or stopped.")
        self._finished = True
        self._finish_status = status
        self._report(perf_counter(), message=message, force=True)

    def _coverage(self) -> float | None:
        if self.total is None:
            return None
        return 100.0 if self.total == 0 else 100.0 * self.completed / self.total

    def _remaining(self, now: float) -> str:
        if self._finished:
            return ""
        if self.total is None:
            return "phase ETA unavailable (total not known)"
        label = "full-coverage ETA" if self.upper_bound else "phase ETA"
        pruning = " (pruning may finish sooner)" if self.upper_bound else ""
        remaining = max(self.total - self.completed, 0)
        if not remaining:
            # Graph bookkeeping can remain after every expensive connection
            # has been evaluated; zero remaining units is not a finish event.
            return "planned units complete; phase completion pending"
        if self._timed_units < 2:
            return f"{label} estimating{pruning}"
        mean = self._timed_seconds / self._timed_units
        active = max(now - self._last_completed, 0.0)
        if active >= mean:
            # An unfinished slow unit has unknown residual work. Reusing its
            # elapsed time as a rough future-unit scale keeps the estimate from
            # counting down to zero while the algorithm is still working.
            estimate = remaining * active
            return f"current unit slower than prior samples; {label} revising, rough ~{_format_eta(estimate)}{pruning}"
        estimate = remaining * mean - active
        return f"{label} ~{_format_eta(estimate)}{pruning}"

    def _report(self, now: float, *, message: str = "", force: bool = False) -> None:
        # Cancellation must remain responsive even when no console/sink event
        # is due. The reporter repeats this check when actually publishing.
        cancel = _cancel_check.get()
        if cancel is not None and cancel():
            raise DesignCancelled("Design cancelled by the user.")
        coverage = self._coverage()
        enough_time = now - self._last_reported >= max(float(self.reporter.min_interval_seconds), 0.0)
        enough_progress = coverage is not None and coverage - self._last_coverage >= max(
            float(self.reporter.percent_step), 0.0
        )
        at_total = self.total is not None and self.completed >= self.total
        if not (force or enough_time or enough_progress or at_total):
            return
        if self.total is None:
            count = f"{self.completed} completed"
        elif self.upper_bound:
            count = f"{self.completed} of at most {self.total}"
        else:
            count = f"{self.completed}/{self.total}"
        label = f"{self.prefix} {self.phase}" if self.prefix else self.phase
        fields = [f"{label}: {count}", f"elapsed {format_elapsed(now - self._started)}"]
        if self._finished:
            fields.append(self._finish_status)
        remaining = self._remaining(now)
        if remaining:
            fields.append(remaining)
        if message and not (self._finished and message == "finished"):
            fields.append(message)
        percent = None if self.upper_bound else coverage
        self.reporter.report(percent, "; ".join(fields), force=force)
        self._last_reported = now
        self._last_coverage = 0.0 if coverage is None else coverage
