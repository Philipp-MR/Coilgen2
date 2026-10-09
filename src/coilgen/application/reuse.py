"""Inspect verified stage evidence without loading its large numerical payload."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile

from ..persistence import file_digest
from ..pipeline_stages import PipelineCheckpoint, PipelineStage


@dataclass(frozen=True, slots=True)
class ReusableStage:
    stage: PipelineStage
    path: Path
    run_directory: Path


@dataclass(frozen=True, slots=True)
class ReusePlan:
    available: tuple[ReusableStage, ...]
    stale: tuple[str, ...]
    diagnostics: tuple[str, ...]

    def through(self, stage: str) -> ReusableStage | None:
        selected = PipelineStage(stage)
        eligible = [entry for entry in self.available if entry.stage.index <= selected.index]
        return max(eligible, key=lambda entry: entry.stage.index, default=None)


def _checkpoint_manifest(record, artifact):
    directory = Path(record.directory).resolve()
    name = artifact.get("name", "")
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Invalid checkpoint artifact path.")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Checkpoint artifact escapes its run directory.")
    if not artifact.get("sha256") or file_digest(path) != artifact["sha256"]:
        raise ValueError("Checkpoint artifact failed integrity verification.")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["arrays.npz", "manifest.json"]:
            raise ValueError("Malformed checkpoint archive.")
        if archive.getinfo("manifest.json").file_size > 64 * 1024**2:
            raise ValueError("Checkpoint manifest exceeds the permitted size.")
        manifest = json.loads(archive.read("manifest.json"))
    if not isinstance(manifest, dict):
        raise ValueError("Checkpoint manifest must be an object.")
    if manifest.get("schema") != PipelineCheckpoint.SCHEMA:
        raise ValueError("Unsupported checkpoint schema.")
    claimed = manifest.pop("payload_sha256", None)
    actual = sha256(json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if claimed != actual:
        raise ValueError("Checkpoint manifest failed integrity verification.")
    fingerprints = manifest.get("fingerprints")
    if not isinstance(fingerprints, dict) or any(not isinstance(value, str) for value in fingerprints.values()):
        raise ValueError("Checkpoint fingerprints must be a mapping of stage hashes.")
    return path, PipelineStage(manifest["stage"]), fingerprints


def plan_reuse(project, records, *, threads: int = 1) -> ReusePlan:
    """Use the engine's exact source/input/dependency contract, with worker settings.

    Candidates are actual saved boundaries, never an earlier interpretation of
    a later mutable state. The numerical worker fully reloads and revalidates the
    selected checkpoint before reuse. Caller-provided records must be verified
    RunRecords; checkpoint file hashes are checked again here.
    """
    matching = [record for record in records if record.metadata.get("project_id") == project.id]
    if not matching:
        return ReusePlan((), (), ())
    config = replace(
        project.config,
        parallel_worker_count=threads,
        constrained_optimization_threads=threads,
        plot_flag=False,
        save_stl_flag=False,
    )
    from .research_requirements import requirements_stage_fingerprints

    current = requirements_stage_fingerprints(config, project.project_root, project.study.get("requirements"))
    available, stale, diagnostics = [], set(), []
    seen = set()
    for record in matching:
        for artifact in record.metadata.get("artifacts", []):
            if artifact.get("kind") not in {"checkpoint", "stage_checkpoint"}:
                continue
            try:
                path, stage, fingerprints = _checkpoint_manifest(record, artifact)
                if path in seen:
                    continue
                seen.add(path)
                if fingerprints.get(stage.value) == current[stage.value]:
                    available.append(ReusableStage(stage, path, record.directory))
                else:
                    stale.add(stage.value)
            except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as exc:
                diagnostics.append(f"{record.directory.name}/{artifact.get('name', '')}: {exc}")
    return ReusePlan(
        tuple(available), tuple(sorted(stale, key=lambda name: PipelineStage(name).index)), tuple(diagnostics)
    )


def plan_stopped_reuse(project, states, jobs_directory, *, threads: int = 1) -> ReusePlan:
    """Verify retained numerical boundaries of stopped jobs, independently of results.

    A live display snapshot never provides recovery evidence. Older jobs without
    a committed per-boundary hash record remain unavailable. The worker still
    reloads the checkpoint and validates its typed numerical state before reuse.
    """
    from types import SimpleNamespace

    from ..research.storage import request_identity, verify_input_files

    root = Path(jobs_directory).resolve()
    config = replace(
        project.config,
        parallel_worker_count=threads,
        constrained_optimization_threads=threads,
        plot_flag=False,
        save_stl_flag=False,
    )
    matching = [
        state
        for state in states
        if state.get("status") in {"failed", "cancelled", "interrupted"} and state.get("project_id") == project.id
    ]
    if not matching:
        return ReusePlan((), (), ())
    from .research_requirements import requirements_stage_fingerprints

    current = requirements_stage_fingerprints(config, project.project_root, project.study.get("requirements"))
    available, stale, diagnostics = [], set(), []
    for state in matching:
        job_id = state.get("id")
        try:
            if not isinstance(job_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id) is None:
                raise ValueError("Invalid stopped job identity.")
            directory = (root / job_id).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("Stopped job leaves its runtime directory.")
            documents = []
            for name in ("request.json", "manifest.json", "checkpoints/evidence.json"):
                path = (directory / name).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("Recovery evidence leaves its job directory.")
                with path.open("rb") as source:
                    encoded = source.read(64 * 1024**2 + 1)
                if len(encoded) > 64 * 1024**2:
                    raise ValueError("Recovery evidence exceeds the permitted size.")
                value = json.loads(encoded)
                if not isinstance(value, dict):
                    raise ValueError("Malformed recovery evidence.")
                documents.append(value)
            request, manifest, evidence = documents
            fields = ("project_id", "project_revision", "project_fingerprint", "project_model_fingerprint")
            if manifest.get("request") != request or any(request.get(key) != state.get(key) for key in fields):
                raise ValueError("Recovery request does not match the submitted job identity.")
            digest = state.get("request_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Stopped job has no verified request digest.")
            if manifest.get("request_sha256") != digest or evidence.get("request_sha256") != digest:
                raise ValueError("Recovery evidence belongs to a different request.")
            if evidence.get("schema") != "coilgen.checkpoint-evidence/1" or any(
                evidence.get(key) != request.get(key) for key in fields
            ):
                raise ValueError("Recovery checkpoint provenance does not match its request.")
            verify_input_files(manifest)
            actual, _ = request_identity(request, manifest["resolved_config"])
            if actual != digest:
                raise ValueError("Recovery request, source, inputs or dependencies changed.")
            artifacts = evidence.get("artifacts")
            if not isinstance(artifacts, list):
                raise ValueError("Recovery checkpoint records are unavailable.")
            record = SimpleNamespace(directory=directory)
            seen = set()
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    raise ValueError("Malformed recovery checkpoint record.")
                stage = PipelineStage(artifact.get("stage"))
                if stage.value in seen or artifact.get("name") != f"checkpoints/{stage.value}.cgstage":
                    raise ValueError("Recovery checkpoint path or boundary is invalid.")
                seen.add(stage.value)
                if stage.value not in state.get("completed_stages", ()):
                    continue
                try:
                    path, saved_stage, fingerprints = _checkpoint_manifest(record, artifact)
                except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile) as exc:
                    diagnostics.append(f"{job_id}/{stage.value}: Stopped checkpoint unavailable: {exc}")
                    continue
                if saved_stage is not stage:
                    raise ValueError("Recovery checkpoint disagrees with its recorded boundary.")
                if fingerprints.get(stage.value) == current[stage.value]:
                    available.append(ReusableStage(stage, path, directory))
                else:
                    stale.add(stage.value)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            # Reject the whole job when its committed evidence is inconsistent.
            available = [entry for entry in available if entry.run_directory.name != job_id]
            diagnostics.append(f"{job_id}: Stopped checkpoint unavailable: {exc}")
    return ReusePlan(
        tuple(available), tuple(sorted(stale, key=lambda name: PipelineStage(name).index)), tuple(diagnostics)
    )
