from dataclasses import replace

import numpy as np
import pytest

from coilgen import CylinderCoilDesign, ParameterAxis
from coilgen.study import DesignStudyCase, DesignStudyResult, DesignStudyTable, StudyMetric
from coilgen.workflow import CoilCaseResult, CoilCaseSpec, CoilVariantSpec, ParameterSweepSpec, ParameterSweepWorkflow


class RecordingWorkflow:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.partial_failure = False

    def run(self, spec):
        self.calls.append(spec.name)
        if self.fail:
            raise RuntimeError("transient solver failure")
        result = CoilCaseResult(spec=spec, metrics={"score": spec.variants[0].design.conductor_width_m})
        if self.partial_failure:
            result.failures["main"] = RuntimeError("variant failed")
        return result


def sweep_fixture(tmp_path):
    design = CylinderCoilDesign()

    def factory(point):
        return CoilCaseSpec(
            name=f"case_{point['radius_m']}",
            variants=(CoilVariantSpec("main", design, save_reports=False),),
            output_dir=tmp_path,
            parameters=point,
            write_case_table=False,
        )

    spec = ParameterSweepSpec(
        name="test_sweep",
        axes=(ParameterAxis("radius_m", (0.1, 0.2)),),
        case_factory=factory,
        output_dir=tmp_path,
        retain_case_results=True,
    )
    runner = RecordingWorkflow()
    return spec, design, runner, ParameterSweepWorkflow(runner)


def test_resume_reuses_success_but_invalidates_changed_design_and_revision(tmp_path):
    spec, design, runner, workflow = sweep_fixture(tmp_path)
    first = workflow.run(spec)
    second = workflow.run(spec)
    assert len(runner.calls) == 2
    assert second.rows() == first.rows()
    design.conductor_width_m *= 2
    third = workflow.run(spec)
    assert len(runner.calls) == 4
    assert third.rows()[0]["score"] == design.conductor_width_m
    workflow.run(replace(spec, cache_revision="changed custom objective"))
    assert len(runner.calls) == 6


def test_resume_invalidates_changed_target_file_contents(tmp_path):
    spec, design, runner, workflow = sweep_fixture(tmp_path)
    target_path = tmp_path / "target.csv"
    target_path.write_text("first target", encoding="utf-8")
    design.target_field_file = str(target_path)
    workflow.run(spec)
    target_path.write_text("other target", encoding="utf-8")
    workflow.run(spec)
    assert len(runner.calls) == 4


def test_resume_invalidates_nested_source_change(tmp_path, monkeypatch):
    import coilgen.workflow as module

    package = tmp_path / "package"
    (package / "research").mkdir(parents=True)
    (package / "workflow.py").write_text("version = 1\n", encoding="utf-8")
    solver = package / "research" / "solver.py"
    solver.write_text("tolerance = 1e-6\n", encoding="utf-8")
    monkeypatch.setattr(module, "__file__", str(package / "workflow.py"))
    spec, _, runner, workflow = sweep_fixture(tmp_path)
    workflow.run(spec)
    workflow.run(spec)
    assert len(runner.calls) == 2
    solver.write_text("tolerance = 1e-9\n", encoding="utf-8")
    workflow.run(spec)
    assert len(runner.calls) == 4


@pytest.mark.parametrize("partial", [False, True])
def test_resume_retries_failed_cases_and_preserves_failure_rows(tmp_path, partial):
    spec, _, runner, workflow = sweep_fixture(tmp_path)
    runner.fail = not partial
    runner.partial_failure = partial
    first = workflow.run(spec)
    assert len(first.failures) == 2
    assert len(first.rows()) == 2
    assert all(row["status"] == "failed" for row in first.rows())
    runner.fail = runner.partial_failure = False
    second = workflow.run(spec)
    assert len(runner.calls) == 4
    assert not second.failures
    assert all(row["status"] == "success" for row in second.rows())


@pytest.mark.parametrize("corrupt", ["{", "[]", '{"status": "success"}'])
def test_resume_recovers_invalid_cache_and_keeps_axis_order(tmp_path, corrupt):
    spec, _, runner, workflow = sweep_fixture(tmp_path)
    first = workflow.run(spec)
    path = next((tmp_path / "case_cache").glob("case_0001_*.json"))
    path.write_text(corrupt, encoding="utf-8")
    second = workflow.run(spec)
    assert len(runner.calls) == 3
    assert second.rows() == first.rows()
    assert len(second.case_results) == len(second.cached_rows) == 1


def test_sweep_names_do_not_share_cached_results(tmp_path):
    spec, _, runner, workflow = sweep_fixture(tmp_path)
    workflow.run(spec)
    workflow.run(replace(spec, name="different_study"))
    assert len(runner.calls) == 4


def test_sweep_stop_on_failure_also_honors_failed_variants(tmp_path):
    spec, _, runner, workflow = sweep_fixture(tmp_path)
    runner.partial_failure = True
    with pytest.raises(RuntimeError, match="failed variants"):
        workflow.run(replace(spec, continue_on_failure=False))
    assert len(runner.calls) == 1


def test_freshly_constructed_custom_metric_functions_allow_resume(tmp_path):
    spec, _, runner, workflow = sweep_fixture(tmp_path)
    factory = spec.case_factory

    def with_metrics(point):
        return replace(factory(point), metric_functions=(lambda result: {"extra": 1.0},))

    spec.case_factory = with_metrics
    workflow.run(spec)
    workflow.run(spec)
    assert len(runner.calls) == 2


def test_incremental_write_failure_does_not_duplicate_parameter_rows(tmp_path, monkeypatch):
    import coilgen.workflow as module

    spec, _, _, workflow = sweep_fixture(tmp_path)
    original = module._append_jsonl
    calls = 0

    def fail_once(payload, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary write failure")
        original(payload, path)

    monkeypatch.setattr(module, "_append_jsonl", fail_once)
    result = workflow.run(spec)
    assert len(result.rows()) == 2
    assert [row["status"] for row in result.rows()] == ["failed", "success"]
    assert len(result.failures) == 1


def test_best_by_does_not_recommend_failed_or_unchecked_designs():
    def candidate(name, score, status):
        return DesignStudyResult(DesignStudyCase(name), {"efficiency": StudyMetric("efficiency", score)}, status=status)

    valid = candidate("valid", 2.0, "PASS")
    warning = candidate("inspect", 3.0, "WARNING")
    failed = candidate("collision", 20.0, "FAIL")
    unchecked = candidate("unchecked", 30.0, "UNKNOWN")
    missing = candidate("missing", np.nan, "PASS")
    table = DesignStudyTable.from_results([failed, unchecked, missing, valid, warning])
    assert table.best_by("efficiency") is warning
    assert table.best_by("efficiency", allowed_statuses=("PASS",)) is valid
    assert table.best_by("efficiency", allowed_statuses=None) is unchecked
    assert table.best_by("efficiency", allowed_statuses=()) is None
    assert DesignStudyTable.from_results([failed, unchecked]).best_by("efficiency") is None
