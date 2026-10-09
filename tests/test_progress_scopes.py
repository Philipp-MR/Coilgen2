"""Real completed work maps monotonically without claiming scientific evidence."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.pipeline import CoilGenPipeline
from coilgen.progress import DesignCancelled, ProgressReporter, progress_context, progress_scope
from coilgen.research.jobs import _record_progress
from test_pipeline_stages import config


def test_nested_work_allocations_retain_unknown_activity_and_exact_counts():
    events = []
    with progress_context(events.append), progress_scope(20, 80, "Pipeline", overall=True):
        with progress_scope(0, 50, "Sensitivity"):
            task = ProgressReporter(label="Sensitivity").task("nodes", 4)
            task.advance(message="first node")
            ProgressReporter(label="Factorization").report(None, "opaque solve")
            task.done()
    first = next(event for event in events if "first node" in event["message"])
    assert first["overall_percent"] == 27.5
    assert first["phase_percent"] == 25
    assert (first["completed"], first["total"]) == (1, 4)
    opaque = next(event for event in events if event["message"] == "opaque solve")
    assert opaque["overall_percent"] == 27.5 and opaque["progress_indeterminate"]
    assert opaque["phase_percent"] is None
    values = [event["overall_percent"] for event in events]
    assert values == sorted(values) and values[-1] == 80


@pytest.mark.parametrize("error", [ValueError("failure"), DesignCancelled("stop")])
def test_failed_or_cancelled_scope_does_not_advance_to_completion(error):
    events = []
    with pytest.raises(type(error)), progress_context(events.append), progress_scope(10, 90, overall=True):
        ProgressReporter().report(25, "partial work")
        raise error
    assert events[-1]["overall_percent"] == 30
    assert not any(event.get("scope_completed") for event in events)


def test_scientific_boundary_is_never_inferred_or_rewritten_from_percentage():
    events = []
    with progress_context(events.append), progress_scope(0, 50, overall=True):
        reporter = ProgressReporter()
        reporter.report(100, "task done")
        reporter.boundary("prepared", preview="stage_preview.json")
    boundary = next(event for event in events if event.get("completed_stage"))
    assert boundary["percent"] is None and "overall_percent" not in boundary
    assert boundary["completed_stages"] == ["prepared"]
    assert sum(bool(event.get("completed_stage")) for event in events) == 1


def test_interactive_work_updates_ignore_coarse_console_thresholds():
    events = []
    with progress_context(events.append):
        task = ProgressReporter(enabled=False, min_interval_seconds=9999, percent_step=50).task("nodes", 100)
        for _ in range(100):
            task.advance()
    assert [event["percent"] for event in events] == list(range(101))


def test_allocation_roundoff_is_clamped_but_invalid_scopes_are_rejected():
    events = []
    with progress_context(events.append), progress_scope(100 / 3, 100.00000000000001, overall=True):
        pass
    assert events[-1]["overall_percent"] == 100
    for bounds in ((-1, 50), (0, 100.001), (60, 50), (0, float("nan"))):
        with pytest.raises(ValueError), progress_scope(*bounds):
            pass


def test_transport_keeps_overall_local_work_and_stage_evidence_separate():
    state = {"progress": 0, "pipeline_stage": "evaluated"}
    _record_progress(
        state,
        {
            "label": "Sensitivity",
            "percent": 80,
            "overall_percent": 45,
            "phase_percent": 80,
            "completed": 8,
            "total": 10,
            "active_stage": "prepared",
            "message": "node work",
        },
    )
    assert state["progress"] == 0.45 and state["phase_progress"] == 0.8
    assert state["active_stage"] == "prepared" and state.get("completed_stage") is None
    _record_progress(
        state,
        {
            "label": "Factorization",
            "percent": 45,
            "overall_percent": 45,
            "phase_percent": None,
            "progress_indeterminate": True,
            "message": "SVD",
        },
    )
    assert state["progress"] == 0.45 and state["progress_indeterminate"]
    assert state["phase_progress"] is None and state["progress_total"] is None
    _record_progress(state, {"label": "Research", "overall_percent": 30, "message": "late work"})
    assert state["progress"] == 0.45
    _record_progress(state, {"label": "Research", "overall_percent": 100, "message": "saved"})
    assert state["progress"] == 0.99  # Final result verification owns 100%.


@pytest.mark.parametrize("invalid", [True, None, float("nan"), float("inf"), "45"])
def test_invalid_transport_percentage_does_not_corrupt_completed_work(invalid):
    state = {"progress": 0.2}
    _record_progress(state, {"label": "Research", "overall_percent": invalid, "message": "working"})
    assert state["progress"] == 0.2


def test_pipeline_reports_actual_intermediate_work_with_console_disabled():
    events = []
    settings = replace(config(), research_physics=None, sf_opt_method="tikhonov")
    expected = CoilGenPipeline(settings).run_stage("prepared")
    with progress_context(events.append):
        actual = CoilGenPipeline(settings).run_stage("prepared")
    values = [event["overall_percent"] for event in events if "overall_percent" in event]
    assert values == sorted(values) and values[-1] == 100
    assert len({round(value, 1) for value in values if 5 < value < 95}) > 20
    assert {event["active_stage"] for event in events if "active_stage" in event} == {"prepared"}
    np.testing.assert_array_equal(actual.parts[0].sensitivity_matrix, expected.parts[0].sensitivity_matrix)
    resumed = []
    with progress_context(resumed.append):
        CoilGenPipeline(settings).run_stage("sheet", actual)
    assert {event["active_stage"] for event in resumed if "active_stage" in event} == {"sheet"}
    assert resumed[-1]["percent"] == 100
