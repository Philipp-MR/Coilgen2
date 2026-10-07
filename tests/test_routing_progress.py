"""Routing ETA messages describe measured phase work and explicit uncertainty."""

import pytest

from coilgen.progress import DesignCancelled, ProgressReporter, progress_context
import coilgen.routing_progress as routing_progress


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def progress_case(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(routing_progress, "perf_counter", clock)
    reporter = ProgressReporter(enabled=False, min_interval_seconds=100.0, percent_step=100.0)
    progress = routing_progress.RoutingProgress(reporter, "part 1/2: existing_cuts")
    events = []
    with progress_context(events.append):
        yield progress, clock, events


def test_known_phase_warms_up_then_uses_measured_throughput(progress_case):
    progress, clock, events = progress_case
    progress.start("winding states", 10)
    assert events[-1]["percent"] == 0.0
    assert events[-1]["message"].startswith("part 1/2: existing_cuts winding states: 0/10")
    assert "phase ETA estimating" in events[-1]["message"]
    clock.advance(5.0)
    progress.update(1, force=True)
    assert "phase ETA estimating" in events[-1]["message"]
    clock.advance(5.0)
    progress.update(2, force=True)
    assert events[-1]["percent"] == 20.0
    assert "elapsed 10.00 s" in events[-1]["message"]
    assert "phase ETA ~40 s" in events[-1]["message"]


def test_stopped_audit_does_not_report_finished_or_complete_coverage(progress_case):
    progress, _clock, events = progress_case
    progress.start("exact body collision audit", None)
    progress.update(4096)
    progress.finish("triangle_pair_budget_exceeded", status="stopped")
    assert events[-1]["percent"] is None
    assert "4096 completed" in events[-1]["message"]
    assert "stopped" in events[-1]["message"]
    assert "finished" not in events[-1]["message"]
    assert "ETA" not in events[-1]["message"]
    progress.start("next audit", None)
    progress.finish()
    assert "finished" in events[-1]["message"]


def test_overdue_active_unit_increases_rough_estimate(progress_case):
    progress, clock, events = progress_case
    progress.start("connections", 10)
    clock.advance(5.0)
    progress.update(1)
    clock.advance(5.0)
    progress.update(2, force=True)
    assert "phase ETA ~40 s" in events[-1]["message"]
    clock.advance(7.0)
    progress.activity("checking support", force=True)
    assert progress.completed == 2
    assert "current unit slower than prior samples" in events[-1]["message"]
    assert "phase ETA revising, rough ~56 s" in events[-1]["message"]
    assert events[-1]["message"].endswith("checking support")


def test_last_overdue_unit_does_not_count_down_to_zero(progress_case):
    progress, clock, events = progress_case
    progress.start("finalists", 3)
    clock.advance(5.0)
    progress.update(1)
    clock.advance(5.0)
    progress.update(2)
    clock.advance(20.0)
    progress.activity(force=True)
    assert "phase ETA revising, rough ~20 s" in events[-1]["message"]
    assert events[-1]["percent"] == pytest.approx(200 / 3)


def test_start_resets_phase_history(progress_case):
    progress, clock, events = progress_case
    progress.start("winding states", 10)
    clock.advance(4.0)
    progress.update(2, force=True)
    assert "phase ETA ~16 s" in events[-1]["message"]
    progress.start("connections", 30, upper_bound=True)
    assert progress.completed == 0
    assert "elapsed 0.000 s" in events[-1]["message"]
    assert "full-coverage ETA estimating" in events[-1]["message"]
    assert "16.00 s" not in events[-1]["message"]


def test_upper_bound_reports_coverage_uncertainty_and_finishes_at_actual_count(progress_case):
    progress, clock, events = progress_case
    progress.start("connections", 20, upper_bound=True)
    clock.advance(3.0)
    progress.update(1)
    clock.advance(3.0)
    progress.update(2, force=True)
    message = events[-1]["message"]
    assert events[-1]["percent"] is None
    assert "2 of at most 20" in message
    assert "full-coverage ETA ~54 s (pruning may finish sooner)" in message
    clock.advance(1.0)
    progress.finish("work budget exhausted")
    assert progress.completed == 2
    assert events[-1]["percent"] is None
    assert "2 of at most 20" in events[-1]["message"]
    assert "finished; work budget exhausted" in events[-1]["message"]
    assert "ETA" not in events[-1]["message"]


def test_exact_phase_early_finish_preserves_completed_count_and_percentage(progress_case):
    progress, clock, events = progress_case
    progress.start("finalists", 6)
    clock.advance(12.0)
    progress.update(2, force=True)
    progress.finish("no further proposals")
    assert progress.completed == 2
    assert events[-1]["percent"] == pytest.approx(100 / 3)
    assert "2/6" in events[-1]["message"]
    assert "finished; no further proposals" in events[-1]["message"]
    assert "ETA" not in events[-1]["message"]


def test_throttled_updates_still_record_all_phase_timing(progress_case):
    progress, clock, events = progress_case
    progress.start("winding states", 10)
    clock.advance(2.0)
    progress.update(1)
    clock.advance(8.0)
    progress.update(2)
    assert len(events) == 1
    progress.activity(force=True)
    assert len(events) == 2
    assert "phase ETA ~40 s" in events[-1]["message"]


def test_reporter_time_and_percent_steps_throttle_independently(progress_case):
    progress, clock, events = progress_case
    progress.reporter.min_interval_seconds = 10.0
    progress.reporter.percent_step = 25.0
    progress.start("winding states", 4)
    clock.advance(1.0)
    progress.update(1)
    assert len(events) == 2  # The 25% step publishes before the time interval.
    clock.advance(9.0)
    progress.activity()
    assert len(events) == 2
    clock.advance(1.0)
    progress.activity()
    assert len(events) == 3


def test_unknown_total_explains_why_phase_eta_is_unavailable(progress_case):
    progress, clock, events = progress_case
    progress.start("support checks", None)
    clock.advance(4.0)
    progress.update(8, force=True)
    assert events[-1]["percent"] is None
    assert "8 completed; elapsed 4.00 s" in events[-1]["message"]
    assert "phase ETA unavailable (total not known)" in events[-1]["message"]
    progress.finish()
    assert "ETA" not in events[-1]["message"]
    assert "8 completed" in events[-1]["message"]
    assert "finished" in events[-1]["message"]


def test_zero_total_exact_phase_is_complete(progress_case):
    progress, _clock, events = progress_case
    progress.start("finalists", 0)
    assert events[-1]["percent"] == 100.0
    progress.finish()
    assert progress.completed == 0
    assert "0/0" in events[-1]["message"]
    assert "ETA" not in events[-1]["message"]


def test_completed_work_units_do_not_claim_zero_time_before_phase_finishes(progress_case):
    progress, clock, events = progress_case
    progress.start("connections", 2, upper_bound=True)
    clock.advance(5.0)
    progress.update(1)
    clock.advance(5.0)
    progress.update(2, force=True)
    clock.advance(20.0)
    progress.activity("assembling circuit proposals", force=True)
    assert "2 of at most 2" in events[-1]["message"]
    assert "planned units complete; phase completion pending" in events[-1]["message"]
    assert "ETA" not in events[-1]["message"]
    progress.finish()
    assert "finished" in events[-1]["message"]
    assert "phase completion pending" not in events[-1]["message"]


@pytest.mark.parametrize(
    "seconds, expected",
    [(0.0, "0 s"), (0.01, "1 s"), (2.1, "3 s"), (59.2, "1 min 0 s"), (60.1, "1 min 10 s"), (138.1, "2 min 20 s")],
)
def test_eta_rounds_up_without_precision_beyond_its_throughput_estimate(seconds, expected):
    assert routing_progress._format_eta(seconds) == expected


def test_cancel_check_is_respected_even_when_reporting_is_throttled(progress_case):
    progress, clock, events = progress_case
    with progress_context(events.append, cancel_check=lambda: clock.now >= 2.0):
        progress.start("support checks", None)
        clock.advance(1.0)
        progress.update(1)
        assert len(events) == 1
        clock.advance(1.0)
        with pytest.raises(DesignCancelled, match="cancelled"):
            progress.activity()
