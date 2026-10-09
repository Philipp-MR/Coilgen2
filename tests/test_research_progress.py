"""Scientific work counts remain scoped across regional and array studies."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.array_studies import ArrayCase, ArrayChannel, ArrayStudySpec, fit_array_fields
from coilgen.application.research_requirements import (
    DesiredDataset,
    PhysicalScenario,
    RegionalRequirement,
    ResearchRequirements,
    evaluate_requirements,
)
from coilgen.application.gradient_studies import MeasuredFieldMap, RigidTransform
from coilgen.config import CoilGenConfig
from coilgen.progress import DesignCancelled, ProgressReporter, progress_context
from coilgen.research.application import _design_work_ranges, run_design
from coilgen.research.requests import ResearchRequest


def _channel():
    return ArrayChannel("c", "Coil", "run", "a" * 64, {"body": 1}, {"body": "b" * 64}, -2, 2)


def test_each_completed_array_case_advances_its_allocation_without_changing_fit():
    cases = tuple(ArrayCase(str(i), [[0, 0, 0], [1, 0, 0]], [i * 1e-7, -i * 1e-7]) for i in range(1, 5))
    spec = ArrayStudySpec(cases, (_channel(),))
    fields = {case.name: np.array([[1e-6], [-1e-6]]) for case in cases}
    expected = fit_array_fields(spec, fields)
    events = []
    with progress_context(events.append):
        actual = fit_array_fields(spec, fields)
    assert actual == expected
    complete = [event["overall_percent"] for event in events if "Completed array fit" in event["message"]]
    assert complete == [25, 50, 75, 100]
    solving = [event for event in events if "Solving bounded signed currents" in event["message"]]
    assert all(event["progress_indeterminate"] for event in solving)
    assert [event["overall_percent"] for event in solving] == [0, 25, 50, 75]


def test_regional_field_updates_cover_every_dataset_and_pose_and_cancel_between_chunks(monkeypatch):
    import coilgen.application.research_requirements as requirements
    import coilgen.evaluation as evaluation

    calls = []

    def operator(parts, points, *_args):
        calls.append(points.copy())
        reporter = ProgressReporter(label="Regional test operator")
        for value in (0, 25, 50, 75, 100):
            reporter.report(value, "Evaluating regional field chunk")
        return np.zeros((3, len(points), 2))

    monkeypatch.setattr(requirements, "_field_operator", operator)
    monkeypatch.setattr(evaluation, "layout_field_geometry", lambda *_args: (None, "unavailable"))
    first = DesiredDataset(MeasuredFieldMap([[0, 0, 0.01]], [[0, 0, 1e-6]]))
    second = DesiredDataset(MeasuredFieldMap([[0, 0, 0.02]], [[0, 0, 2e-6]]))
    spec = ResearchRequirements(
        (RegionalRequirement("Near", first, peak_limit_T=3e-6), RegionalRequirement("Far", second, peak_limit_T=3e-6)),
        (PhysicalScenario("Shifted", sample_transform=RigidTransform(translation_m=(0.001, 0, 0))),),
    )
    mesh = SimpleNamespace(vertices=np.zeros((2, 3)), faces=np.array([[0, 1, 1]]))
    parts = [SimpleNamespace(mesh=mesh, stream_function=np.zeros(2), contour_step=None)]
    config = CoilGenConfig(progress_flag=False)
    events = []
    with progress_context(events.append):
        result, _ = evaluate_requirements(parts, config, spec)
    assert len(calls) == 4 and len(result["rows"]) == 8
    completed = [event["overall_percent"] for event in events if "Regional datasets reviewed" in event["message"]]
    assert completed == [25, 50, 75, 100]
    intermediate = [
        event["overall_percent"] for event in events if "Evaluating regional field chunk" in event["message"]
    ]
    assert any(0 < value < 25 for value in intermediate)
    assert any(50 < value < 75 for value in intermediate)
    calls.clear()
    cancelled = []
    with progress_context(cancelled.append, lambda: any("chunk" in event["message"] for event in cancelled)):
        with pytest.raises(DesignCancelled):
            evaluate_requirements(parts, config, spec)
    assert len(calls) == 1
    assert not any(event.get("overall_percent") == 100 for event in cancelled)


def test_partial_design_finishes_only_after_checkpoint_artifacts_and_metadata_are_saved(tmp_path, monkeypatch):
    import coilgen.research.application as application

    request = ResearchRequest(
        name="Progress ownership",
        pipeline_stage="prepared",
        config={
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": [0.3, 0.12, 8, 4, 1, 0, 0, 0],
            "target_region_resolution": 3,
            "progress_flag": False,
        },
    )
    events, writes = [], []
    original = application.atomic_json

    def write(path, payload):
        if path.name == "result.json":
            writes.append(
                (payload.get("metadata_sha256"), max((e.get("overall_percent", 0) for e in events), default=0))
            )
        original(path, payload)

    monkeypatch.setattr(application, "atomic_json", write)
    with progress_context(events.append):
        result = run_design(request, tmp_path / "run")
    assert result["pipeline_stage"] == "prepared"
    assert writes[0][0] is None and writes[-1][0] == result["metadata_sha256"]
    assert all(value < 100 for _, value in writes)
    progress = [event["overall_percent"] for event in events if "overall_percent" in event]
    assert progress == sorted(progress) and progress[-1] == 100
    assert any(
        "Compressing scientific arrays" in event["message"] and event["progress_indeterminate"] for event in events
    )
    assert any("Verifying checkpoint artifact" in event["message"] for event in events)
    ranges = _design_work_ranges(request)
    assert "validation" not in ranges and "fabrication" not in ranges
    assert ranges["pipeline"][1] < ranges["save"][0]


def test_cohort_solver_cancellation_is_never_retried_as_a_solver_failure(monkeypatch):
    import coilgen.cohort_shim as cohort

    calls = []

    class Problem:
        def get_problem_data(self, solver):
            calls.append(solver)
            raise DesignCancelled("Stop during canonicalization")

    monkeypatch.setattr(cohort, "_cvxpy_conic_solver_order", lambda *_args: ("FIRST", "SECOND"))
    with pytest.raises(DesignCancelled):
        cohort._solve_cvxpy(Problem(), object(), CoilGenConfig(progress_flag=False))
    assert calls == ["FIRST"]


def test_cancellation_while_publishing_optional_preview_does_not_publish_final_result(tmp_path, monkeypatch):
    import coilgen.research.application as application

    def cancel_preview(*_args):
        raise DesignCancelled("Stop while preparing viewport evidence")

    monkeypatch.setattr(application, "publish_stage_preview", cancel_preview)
    request = ResearchRequest(
        name="Cancelled preview",
        pipeline_stage="prepared",
        config={
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": [0.3, 0.12, 8, 4, 1, 0, 0, 0],
            "target_region_resolution": 3,
            "progress_flag": False,
        },
    )
    events = []
    with progress_context(events.append):
        with pytest.raises(DesignCancelled):
            run_design(request, tmp_path / "cancelled")
    assert (tmp_path / "cancelled/checkpoints/prepared.cgstage").is_file()
    assert not (tmp_path / "cancelled/result.json").exists()
    assert not any(event.get("overall_percent") == 100 for event in events)
