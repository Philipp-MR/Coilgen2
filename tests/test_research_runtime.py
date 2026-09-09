from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.progress import DesignCancelled, ProgressReporter, progress_context
from coilgen.research.jobs import JobManager, RuntimePolicy
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import (
    DirectoryOwner,
    atomic_json,
    file_digest,
    json_value,
    operator_fingerprint,
    request_identity,
    verified_result,
    verify_input_files,
)
from coilgen.research.studies import ResearchStudy, ToleranceStudy, pareto_front


def request(**config):
    return ResearchRequest.from_config(
        CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
            field_shape_function="y",
            target_gradient_strength=0.01,
            target_region_radius=0.045,
            target_region_resolution=12,
            sf_opt_method="constrained_quadratic",
            research_physics={"objective": "power"},
            constrained_quadratic_fallback_policy="strict",
            target_error_tolerance_percent=5,
            stream_function_smoothness_factor=0,
            turn_spacing_regularization_factor=0,
            levels=8,
            min_point_loop_number=3,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance_calculation=True,
            plot_flag=False,
            **config,
        ),
        name="Runtime test",
        design_margin=0.8,
    )


def wait_terminal(jobs, id_, timeout=45):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = jobs.get(id_)
        if state["status"] not in {"queued", "running", "cancelling"}:
            return state
        time.sleep(0.05)
    pytest.fail(f"Worker did not complete: {jobs.get(id_)}")


def _claim_directory(directory, events):
    try:
        with JobManager(directory):
            events.put("acquired")
    except RuntimeError as exc:
        events.put(str(exc))


def _hold_directory(directory, events):
    with DirectoryOwner(directory):
        events.put("acquired")
        time.sleep(20)


@pytest.mark.parametrize(
    "values",
    [
        {"max_workers": 1.5},
        {"threads_per_worker": True},
        {"max_pending": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
    ],
)
def test_runtime_budget_requires_finite_positive_integers(values):
    with pytest.raises(ValueError):
        RuntimePolicy(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"validation_resolution": True},
        {"validation_resolution": 1},
        {"fabricate": "false"},
        {"design_margin": 0},
        {"design_margin": float("nan")},
        {"config": {"conductor_thickness": float("nan")}},
        {"config": {"mesh_resolution": "fine"}},
    ],
)
def test_request_uses_native_controls_and_rejects_ambiguous_values(values):
    with pytest.raises((ValueError, TypeError)):
        ResearchRequest.from_dict(values)


def test_mesh_density_never_silently_disables_energy():
    value = ResearchRequest.from_dict(
        {
            "config": {
                "coil_mesh_file": "create cylinder mesh",
                "cylinder_mesh_parameter_list": [0.3, 0.12, 48, 24, 1, 0, 0, 0],
                "research_physics": {"objective": "power", "evaluate_energy": True},
            }
        }
    )
    assert value.to_config().research_physics["evaluate_energy"] is True


def test_process_local_inputs_cannot_cross_workers():
    value = ResearchRequest.from_dict({"config": {"coil_mesh_file": "memory:mesh:test"}})
    with pytest.raises(ValueError, match="process-local"):
        value.to_config()


def test_identity_canonical_paths_controls_and_file_content(tmp_path):
    mesh = tmp_path / "coil.stl"
    mesh.write_text("solid test\nendsolid test\n")
    value = ResearchRequest.from_dict({"config": {"coil_mesh_file": mesh.name, "geometry_source_path": str(tmp_path)}})
    config = value.to_config()
    assert Path(config.coil_mesh_file) == mesh
    first, manifest = request_identity(value.to_dict(), asdict(config))
    canonical = ResearchRequest.from_config(config, name="Renamed")
    assert request_identity(canonical.to_dict(), asdict(config))[0] == first
    assert manifest["input_files"]["coil_mesh_file"]["sha256"] == file_digest(mesh)
    controls = dict(value.to_dict(), fabricate=True)
    assert request_identity(controls, asdict(config))[0] != first
    mesh.write_text("solid changed\nendsolid changed\n")
    assert request_identity(value.to_dict(), asdict(config))[0] != first


def test_atomic_json_and_artifact_verification_reject_corruption(tmp_path):
    assert json_value({"bad": np.array([np.nan, np.inf])}) == {"bad": [None, None]}
    artifact = tmp_path / "evidence.json"
    atomic_json(artifact, {"actual": 1})
    atomic_json(
        tmp_path / "result.json",
        {
            "request_sha256": "abc",
            "artifacts": [{"name": artifact.name, "sha256": file_digest(artifact)}],
        },
    )
    assert verified_result(tmp_path, "abc") is not None
    artifact.write_text("corrupt")
    assert verified_result(tmp_path, "abc") is None


def test_external_input_changes_invalidate_result(tmp_path):
    path = tmp_path / "field.csv"
    path.write_text("initial")
    manifest = {"input_files": {"target": {"path": str(path), "sha256": file_digest(path)}}}
    verify_input_files(manifest)
    path.write_text("different")
    with pytest.raises(RuntimeError, match="changed during execution"):
        verify_input_files(manifest)


def test_operator_fingerprint_tracks_dtype_shape_content_without_json_array():
    array = np.arange(100).reshape(10, 10)
    original = operator_fingerprint(array)
    assert len(json.dumps(original)) < 150
    assert original != operator_fingerprint(array.astype(float))
    assert original != operator_fingerprint(array.ravel())
    array[0, 0] = -1
    assert original != operator_fingerprint(array)


def test_progress_context_cancels_inner_tasks_without_console_and_restores():
    events = []
    reporter = ProgressReporter(enabled=False)
    with progress_context(events.append):
        task = reporter.task("quadrature", 10)
        task.advance(1, force=True)
        with progress_context(events.append, lambda: True):
            with pytest.raises(DesignCancelled):
                task.advance(1, force=True)
        reporter.report(50, "Validate")
    assert [e["percent"] for e in events] == [0, 10, 50]


def test_competing_manager_cannot_recover_live_jobs_same_process(tmp_path):
    with JobManager(tmp_path) as jobs:
        with jobs._lock:
            state = jobs.submit(request())
            before = (tmp_path / state["id"] / "job.json").read_bytes()
            with pytest.raises(RuntimeError, match="already owns"):
                JobManager(tmp_path)
            assert (tmp_path / state["id"] / "job.json").read_bytes() == before
            jobs.cancel(state["id"])
    with JobManager(tmp_path):
        pass


def test_competing_spawned_process_cannot_recover_live_jobs(tmp_path):
    context = mp.get_context("spawn")
    events = context.Queue()
    with JobManager(tmp_path) as jobs:
        with jobs._lock:
            state = jobs.submit(request())
            path = tmp_path / state["id"] / "job.json"
            before = path.read_bytes()
            contender = context.Process(target=_claim_directory, args=(str(tmp_path), events))
            contender.start()
            try:
                assert "already owns" in events.get(timeout=15)
                contender.join(timeout=5)
                assert contender.exitcode == 0
                assert path.read_bytes() == before
            finally:
                if contender.is_alive():
                    contender.terminate()
                    contender.join(timeout=5)
                jobs.cancel(state["id"])
    events.close()


def test_os_releases_directory_lock_when_owner_crashes(tmp_path):
    context = mp.get_context("spawn")
    events = context.Queue()
    owner = context.Process(target=_hold_directory, args=(str(tmp_path), events))
    owner.start()
    try:
        assert events.get(timeout=15) == "acquired"
        with pytest.raises(RuntimeError, match="already owns"):
            DirectoryOwner(tmp_path)
    finally:
        owner.terminate()
        owner.join(timeout=5)
        events.close()
    with DirectoryOwner(tmp_path):
        pass


def test_queue_cancellation_recovery_and_artifact_confinement(tmp_path):
    with JobManager(tmp_path) as jobs:
        with jobs._lock:
            state = jobs.submit(request())
            assert jobs.cancel(state["id"])["status"] == "cancelled"
        assert jobs.artifact(state["id"], "request.json").is_file()
        with pytest.raises(FileNotFoundError):
            jobs.artifact(state["id"], "../../outside")
    state["status"] = "running"
    atomic_json(tmp_path / state["id"] / "job.json", state)
    with JobManager(tmp_path) as restarted:
        assert restarted.get(state["id"])["status"] == "interrupted"


def test_one_process_start_failure_does_not_stop_queue(tmp_path, monkeypatch):
    with JobManager(tmp_path) as jobs:
        original = jobs._context.Process

        class BrokenProcess:
            def start(self):
                raise OSError("simulated process launch failure")

        monkeypatch.setattr(jobs._context, "Process", lambda **_kwargs: BrokenProcess())
        bad = jobs.submit(request())
        state = wait_terminal(jobs, bad["id"])
        assert state["status"] == "failed" and "simulated" in state["error"]
        assert jobs._thread.is_alive()
        monkeypatch.setattr(jobs._context, "Process", original)
        good = jobs.submit(request())
        state = wait_terminal(jobs, good["id"])
        assert state["status"] == "completed", state
        result = jobs.result(good["id"])
        assert result["summary"]["accepted"] is True
        assert result["runtime"]["native_threads_requested"] == 1
        assert "geometry" not in result


def test_active_job_cancel_terminates_worker(tmp_path):
    with JobManager(tmp_path) as jobs:
        state = jobs.submit(request())
        end = time.monotonic() + 10
        while jobs.get(state["id"])["status"] == "queued" and time.monotonic() < end:
            time.sleep(0.02)
        assert jobs.cancel(state["id"])["status"] in {"cancelling", "cancelled"}
        assert wait_terminal(jobs, state["id"], timeout=10)["status"] == "cancelled"


def test_timeout_terminates_worker(tmp_path):
    with JobManager(tmp_path, RuntimePolicy(timeout_seconds=0.01)) as jobs:
        state = jobs.submit(request())
        result = wait_terminal(jobs, state["id"], timeout=10)
        assert result["status"] == "failed" and "exceeded" in result["error"]


def test_study_resume_verifies_all_evidence(tmp_path):
    study = ResearchStudy(tmp_path)
    first = study.run([request()])
    row = first["candidates"][0]
    assert row["status"] == "completed", row
    second = study.run([request()])
    assert second["candidates"][0]["reused"] is True
    evidence = tmp_path / "jobs" / row["job_id"] / "solver_diagnostics.json"
    evidence.write_text("corrupted")
    third = study.run([request()])
    assert third["candidates"][0]["reused"] is False


def test_independent_design_mixture_requires_explicit_common_scales(tmp_path):
    value = request().to_dict()
    value["config"]["research_physics"] = {"objective": "mixture"}
    with pytest.raises(ValueError, match="explicit objective_scales"):
        ResearchStudy(tmp_path).run([value])


def test_pareto_excludes_failed_unknown_nonfinite_and_mixed_scopes():
    def row(error, power, accepted=True, scope="sheet"):
        return {
            "summary": {"accepted": accepted, "scope": scope, "validated_relative_field_error": error, "power_w": power}
        }

    assert pareto_front([row(1, 1), row(2, 2), row(0, 0, False), row(0, np.nan), row(True, 0)]) == [0]
    with pytest.raises(ValueError, match="one physical"):
        pareto_front([row(1, 1), row(2, 2, scope="wire")])


def _planar_problem():
    from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh
    from coilgen.geometry import compute_face_normals, compute_vertex_normals
    from coilgen.models import CoilPart, Mesh, TargetField
    from coilgen.optimization import StreamFunctionOptimizationProblem
    from coilgen.physics import calculate_resistance_matrix, calculate_sensitivity_matrix

    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 0]], float) * 0.05
    mesh = Mesh(
        vertices, np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]), boundaries=[np.array([0, 1, 2, 3, 0])]
    )
    mesh.face_normals = compute_face_normals(mesh)
    mesh.vertex_normals = compute_vertex_normals(mesh)
    parts = [CoilPart(mesh)]
    config = CoilGenConfig(
        sf_opt_method="constrained_quadratic",
        symmetry_reduction="off",
        progress_flag=False,
        gauss_order=3,
        target_error_constraint_metric="rms",
        target_error_rms_limit_T=1e-7,
        target_error_constraint_acceptance_absolute_T=0,
        constrained_optimization_tolerance=1e-9,
    )
    target = TargetField(
        np.array([[0, 0, 0.04]]), np.array([[0, 0, 1e-6]]), np.ones(1), np.zeros(1, bool), np.ones(1, int)
    )
    calculate_one_ring_by_mesh(parts)
    calculate_basis_functions(parts)
    calculate_resistance_matrix(parts, config)
    calculate_sensitivity_matrix(parts, target, config)
    return StreamFunctionOptimizationProblem.from_parts(parts, target, config), config


def test_tolerance_study_reuses_graph_and_preserves_fixed_reference_on_resume(tmp_path, monkeypatch):
    from coilgen.research import convex

    problem, config = _planar_problem()
    spec = convex.ResearchPhysicsSpec(objective="mixture")
    study = ToleranceStudy(tmp_path, problem, spec, config)
    first = study.run([1, 2])
    assert all(row["status"] == "completed" for row in first["candidates"])
    assert first["operator_assembly_count"] == 1 and first["cvxpy_graph_reused"]
    assert first["normalization_scales"]["power"] > 0
    assert first["normalization_reference"]["tolerance_scale"] == 1
    assert first["candidates"][0]["summary"]["power_w"] > first["candidates"][1]["summary"]["power_w"]
    original = convex.ParameterizedResearchProblem

    def unexpected_assembly(*_args, **_kwargs):
        raise AssertionError("Fully cached studies must not build operators.")

    monkeypatch.setattr(convex, "ParameterizedResearchProblem", unexpected_assembly)
    second = study.run([2, 1])
    assert all(row["reused"] for row in second["candidates"])
    assert second["operator_assembly_count"] == 0
    assert second["normalization_scales"] == first["normalization_scales"]
    assert second["normalization_reference"] == first["normalization_reference"]
    created_with = []

    def capture_reference(problem, spec, config):
        created_with.append(spec.objective_scales)
        return original(problem, spec, config)

    monkeypatch.setattr(convex, "ParameterizedResearchProblem", capture_reference)
    third = study.run([2, 0.5, 1])
    assert [row["reused"] for row in third["candidates"]] == [True, False, True]
    assert created_with == [first["normalization_scales"]]
    assert third["normalization_reference"] == first["normalization_reference"]
    assert third["normalization_scales"] == first["normalization_scales"]


def test_tolerance_study_rejects_corrupt_artifacts_and_changed_operator_input(tmp_path):
    from coilgen.research.convex import ResearchPhysicsSpec

    problem, config = _planar_problem()
    study = ToleranceStudy(tmp_path, problem, ResearchPhysicsSpec(), config)
    first = study.run([1])
    evidence = next(tmp_path.glob("*/stream_function.npz"))
    evidence.write_bytes(b"corrupt")
    second = study.run([1])
    assert not second["candidates"][0]["reused"]
    assert np.load(evidence)["stream_function_A"].shape == (5,)
    config.target_error_rms_limit_T *= 0.5
    third = study.run([1])
    assert not third["candidates"][0]["reused"]
    assert third["operators_sha256"] != first["operators_sha256"]
    assert third["candidates"][0]["summary"]["power_w"] > first["candidates"][0]["summary"]["power_w"]


def test_tolerance_study_snapshots_mapping_spec_and_resumes_new_points(tmp_path):
    problem, config = _planar_problem()
    matrix = problem.sensitivity_matrix[2].copy()
    supplied = {
        "objective": "mixture",
        "objective_weights": {"power": 1, "magnetic_energy": 1},
        "field_constraints": [{"name": "additional_field", "matrix": matrix, "target": [1e-6], "tolerance": 1e-6}],
    }
    study = ToleranceStudy(tmp_path, problem, supplied, config)
    # Changes to caller-owned dictionaries and arrays must not alter the
    # contract or its identity between construction and execution.
    supplied["objective_weights"]["power"] = 99
    matrix[:] = 0
    assert study.spec.objective_weights["power"] == 1
    assert np.any(study.spec.field_constraints[0].matrix)
    first = study.run([1])
    assert first["candidates"][0]["status"] == "completed"
    second = study.run([1, 2])
    assert [row["status"] for row in second["candidates"]] == ["completed", "completed"]
    assert [row["reused"] for row in second["candidates"]] == [True, False]
    assert second["normalization_scales"] == first["normalization_scales"]
    assert second["normalization_reference"] == first["normalization_reference"]


def test_tolerance_study_cancellation_preserves_finished_evidence(tmp_path):
    from coilgen.research.convex import ResearchPhysicsSpec

    problem, config = _planar_problem()
    completed = []
    result = ToleranceStudy(tmp_path, problem, ResearchPhysicsSpec(), config).run(
        [1, 2],
        progress=completed.append,
        cancel_check=lambda: bool(completed),
    )
    assert [row["status"] for row in result["candidates"]] == ["completed", "cancelled"]
