from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np

from .jobs import JobManager, RuntimePolicy
from .requests import ResearchRequest
from .storage import DirectoryOwner, atomic_json, file_digest, operator_fingerprint, request_identity, verified_result


def pareto_front(results: Iterable[dict], metrics=("validated_relative_field_error", "power_w")) -> list[int]:
    """Indices of accepted, finite, nondominated candidates (all metrics minimize).

    Compare one physical scope at a time: sheet and fabrication results are not
    interchangeable. Duplicate objective vectors are retained for inspection.
    """
    candidates = []
    scopes = set()
    for index, result in enumerate(results):
        summary = result.get("summary", {})
        values = [summary.get(metric) for metric in metrics]
        if summary.get("accepted") is True and all(
            not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in values
        ):
            candidates.append((index, values))
            scopes.add(summary.get("scope"))
    if len(scopes) > 1:
        raise ValueError("Pareto selection requires one physical validation scope.")
    return [
        i
        for i, values in candidates
        if not any(
            all(a <= b for a, b in zip(other, values)) and any(a < b for a, b in zip(other, values))
            for j, other in candidates
            if i != j
        )
    ]


class ResearchStudy:
    """Bounded process sweep with explicit, content-verified restart/reuse.

    Use ParameterizedResearchProblem for faster repeated solves on one mesh.
    This class also supports changes to geometry, contours or physics between
    candidates. Failure of one candidate does not erase the other observations.
    """

    def __init__(self, directory: str | Path, policy: RuntimePolicy | None = None):
        self.directory = Path(directory).resolve()
        self.policy = policy or RuntimePolicy()

    def run(
        self,
        requests: Iterable[ResearchRequest | dict],
        *,
        resume=True,
        progress: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict:
        with DirectoryOwner(self.directory):
            return self._run(requests, resume=resume, progress=progress, cancel_check=cancel_check)

    def _run(self, requests, *, resume, progress, cancel_check) -> dict:
        candidates = [
            ResearchRequest.from_dict(item.to_dict() if isinstance(item, ResearchRequest) else item)
            for item in requests
        ]
        if not candidates or len(candidates) > 256:
            raise ValueError("A research study needs between 1 and 256 candidates.")
        prepared = []
        for request in candidates:
            config = request.to_config(threads=self.policy.threads_per_worker)
            if config.research_physics and config.research_physics.get("objective") == "mixture":
                from .convex import ResearchPhysicsSpec

                spec = ResearchPhysicsSpec.from_dict(config.research_physics)
                required = {key for key, value in spec.objective_weights.items() if value > 0}
                if not required <= spec.objective_scales.keys():
                    raise ValueError(
                        "Independent-design mixture studies require explicit objective_scales for every active term. "
                        "Use ToleranceStudy for one fixed operator set and a shared automatic reference."
                    )
            config = replace(
                config,
                parallel_worker_count=self.policy.threads_per_worker,
                constrained_optimization_threads=self.policy.threads_per_worker,
                plot_flag=False,
                save_stl_flag=False,
            )
            digest, _ = request_identity(request.to_dict(), asdict(config))
            prepared.append((request, digest))
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self.directory / "study.json"
        old = {}
        if resume and checkpoint.is_file():
            try:
                previous = json.loads(checkpoint.read_text(encoding="utf-8"))
                old = {
                    row["request_sha256"]: row
                    for row in previous.get("candidates", [])
                    if row.get("status") == "completed"
                }
            except (ValueError, KeyError, TypeError):
                pass
        rows, active, results = [], {}, {}
        with JobManager(self.directory / "jobs", self.policy) as jobs:
            for index, (request, digest) in enumerate(prepared):
                row = {
                    "index": index,
                    "name": request.name,
                    "request_sha256": digest,
                    "status": "pending",
                    "reused": False,
                }
                previous = old.get(digest)
                if previous:
                    old_id = previous.get("job_id", "")
                    if len(old_id) == 32 and all(c in "0123456789abcdef" for c in old_id):
                        result_path = jobs.directory / old_id / "result.json"
                        result = (
                            verified_result(jobs.directory / old_id, digest)
                            if result_path.is_file() and previous.get("result_sha256") == file_digest(result_path)
                            else None
                        )
                        if result:
                            row.update(
                                status="completed",
                                job_id=old_id,
                                reused=True,
                                summary=result["summary"],
                                result_sha256=previous["result_sha256"],
                            )
                            results[index] = result
                rows.append(row)

            def save():
                atomic_json(
                    checkpoint,
                    {"schema": "coilgen-research-study-v1", "runtime": asdict(self.policy), "candidates": rows},
                )

            save()
            while any(row["status"] in {"pending", "queued", "running", "cancelling"} for row in rows):
                if cancel_check and cancel_check():
                    for id_ in active:
                        jobs.cancel(id_)
                    for row in rows:
                        if row["status"] == "pending":
                            row["status"] = "cancelled"
                for row in rows:
                    if len(active) >= self.policy.max_pending:
                        break
                    if row["status"] == "pending":
                        job = jobs.submit(prepared[row["index"]][0].to_dict())
                        row.update(status=job["status"], job_id=job["id"])
                        active[job["id"]] = row
                changed = False
                for id_, row in list(active.items()):
                    state = jobs.get(id_)
                    if row["status"] != state["status"]:
                        row.update(status=state["status"], error=state.get("error"))
                        changed = True
                    if state["status"] == "completed":
                        result = jobs.result(id_)
                        row["summary"] = result["summary"]
                        row["result_sha256"] = file_digest(jobs.directory / id_ / "result.json")
                        results[row["index"]] = result
                    if state["status"] not in {"queued", "running", "cancelling"}:
                        del active[id_]
                if changed:
                    save()
                    if progress:
                        progress({"completed": len(results), "total": len(rows), "candidates": rows})
                if active:
                    time.sleep(0.1)
            save()
        # A combined run can intentionally mix scopes; separate Pareto sets.
        pareto = {}
        for scope in sorted({r["summary"]["scope"] for r in results.values()}):
            group = [(i, result) for i, result in results.items() if result["summary"]["scope"] == scope]
            pareto[scope] = [group[i][0] for i in pareto_front([r for _, r in group])]
        payload = {
            "schema": "coilgen-research-study-v1",
            "runtime": asdict(self.policy),
            "candidates": rows,
            "pareto_indices": pareto,
        }
        atomic_json(checkpoint, payload)
        return payload


class ToleranceStudy:
    """One conic graph/operator assembly and one fixed objective reference.

    This solves a *continuous sheet* against the supplied sampled operators.
    It does not claim independent-grid or fabricated-wire acceptance. Different
    meshes belong in ResearchStudy; fixed operators belong here and reuse DPP.
    """

    def __init__(self, directory: str | Path, problem, spec, config, *, threads: int = 1):
        from .convex import ResearchPhysicsSpec

        self.directory = Path(directory).resolve()
        self.problem = problem
        # Normalize mappings once and detach nested arrays/weight dictionaries
        # from the caller so resumed normalization uses the same contract.
        self.spec = deepcopy(ResearchPhysicsSpec.from_dict(spec))
        self.config = config
        self.policy = RuntimePolicy(threads_per_worker=threads)

    def run(
        self,
        tolerance_scales: Iterable[float],
        *,
        resume: bool = True,
        progress: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict:
        from threadpoolctl import threadpool_limits
        from ..progress import DesignCancelled, progress_context
        from .convex import ParameterizedResearchProblem

        scales = list(tolerance_scales)
        if (
            not scales
            or len(scales) > 256
            or any(
                isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s < 0 for s in scales
            )
        ):
            raise ValueError("A tolerance study needs 1–256 nonnegative finite scales.")
        config = replace(
            self.config,
            parallel_worker_count=self.policy.threads_per_worker,
            constrained_optimization_threads=self.policy.threads_per_worker,
        )
        identity = {
            "config": asdict(config),
            "operators": operator_fingerprint(self.problem),
            "physics": operator_fingerprint(self.spec),
        }
        base_digest, manifest = request_identity(identity, asdict(config))
        rows, results, compiled = [], {}, None
        checkpoint = self.directory / "study.json"

        def candidate_digest(scale):
            return sha256(f"{base_digest}:{float(scale).hex()}".encode()).hexdigest()

        with DirectoryOwner(self.directory), threadpool_limits(self.policy.threads_per_worker):
            previous = {}
            cached = {}
            normalization = None
            if resume and checkpoint.is_file():
                try:
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if saved.get("operators_sha256") == base_digest:
                        previous = {row["tolerance_scale"]: row for row in saved["candidates"]}
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            # The checkpoint is only an index. Scientific normalization comes
            # from complete, checksum-verified candidate evidence, never from
            # editable top-level labels. Freeze it before solving any new point.
            for scale in scales:
                digest = candidate_digest(scale)
                result_path = self.directory / digest / "result.json"
                old = previous.get(scale, {})
                result = (
                    verified_result(result_path.parent, digest)
                    if old.get("status") == "completed"
                    and result_path.is_file()
                    and old.get("result_sha256") == file_digest(result_path)
                    else None
                )
                if result is not None and isinstance(result.get("normalization"), dict):
                    candidate = result["normalization"]
                    if normalization is not None and candidate != normalization:
                        raise ValueError("Cached tolerance points disagree about their fixed objective reference.")
                    normalization = candidate
                    cached[float(scale)] = result

            def save():
                payload = {
                    "schema": "coilgen-tolerance-study-v1",
                    "operators_sha256": base_digest,
                    "normalization_scales": normalization["scales"] if normalization is not None else None,
                    "normalization_reference": normalization["reference"] if normalization is not None else None,
                    "operator_assembly_count": int(compiled is not None),
                    "cvxpy_graph_reused": compiled is not None and compiled.solve_count > 1,
                    "runtime": asdict(self.policy),
                    "scope": "continuous_sheet_sampled_constraints",
                    "independent_validation_performed": False,
                    "candidates": rows,
                    "pareto_indices": [
                        list(results)[i]
                        for i in pareto_front(list(results.values()), metrics=("relative_field_error", "power_w"))
                    ],
                }
                atomic_json(checkpoint, payload)
                return payload

            for index, scale in enumerate(scales):
                if cancel_check and cancel_check():
                    rows.extend(
                        {"index": i, "tolerance_scale": s, "status": "cancelled", "reused": False}
                        for i, s in enumerate(scales[index:], index)
                    )
                    break
                digest = candidate_digest(scale)
                directory = self.directory / digest
                row = {"index": index, "tolerance_scale": float(scale), "status": "pending", "reused": False}
                result_path = directory / "result.json"
                result = cached.get(float(scale))
                if result is not None:
                    row.update(status="completed", reused=True, result_sha256=file_digest(result_path))
                else:
                    try:
                        # Stage-boundary cancellation still works with console output off.
                        with progress_context(lambda _event: None, cancel_check):
                            if compiled is None:
                                spec = (
                                    replace(self.spec, objective_scales=normalization["scales"])
                                    if normalization
                                    else self.spec
                                )
                                compiled = ParameterizedResearchProblem(self.problem, spec, config)
                                if normalization is None:
                                    normalization = {
                                        "scales": dict(compiled.objective_scales),
                                        "reference": dict(compiled.reference_diagnostics),
                                    }
                            solution = compiled.solve(tolerance_scale=float(scale))
                        x = solution.reduced_stream_function
                        target_norm = float(np.linalg.norm(self.problem.target_z))
                        relative = self.problem.residual_norm(x) / target_norm if target_norm > 0 else None
                        diagnostics = solution.diagnostics
                        diagnostics = dict(
                            diagnostics,
                            research_reference=normalization["reference"],
                            research_objective_scales=normalization["scales"],
                        )
                        directory.mkdir(parents=True, exist_ok=True)
                        arrays = directory / "stream_function.npz"
                        np.savez_compressed(
                            arrays, reduced_stream_function_A=x, stream_function_A=self.problem.expand(x)
                        )
                        atomic_json(directory / "solver_diagnostics.json", diagnostics)
                        atomic_json(directory / "manifest.json", dict(manifest, tolerance_scale=float(scale)))
                        artifacts = [
                            {"name": name, "sha256": file_digest(directory / name)}
                            for name in (arrays.name, "solver_diagnostics.json", "manifest.json")
                        ]
                        result = {
                            "request_sha256": digest,
                            "artifacts": artifacts,
                            "normalization": normalization,
                            "summary": {
                                "accepted": diagnostics.get("constrained_solver_contract_met") is True,
                                "scope": "continuous_sheet_sampled_constraints",
                                "relative_field_error": relative,
                                "power_w": diagnostics.get("sheet_power_W"),
                                "energy_j": diagnostics.get("energy_J"),
                            },
                        }
                        atomic_json(result_path, result)
                        cached[float(scale)] = result
                        row.update(status="completed", result_sha256=file_digest(result_path))
                    except DesignCancelled:
                        row.update(status="cancelled")
                    except (RuntimeError, ValueError) as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if result is not None:
                    row["summary"] = result["summary"]
                    results[index] = result
                rows.append(row)
                save()
                if progress:
                    progress({"completed": len(results), "total": len(scales), "candidates": rows})
            return save()
