"""Thin headless application service; numerical work stays in the shared engine."""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from types import MappingProxyType
from tempfile import TemporaryDirectory

import numpy as np

from ..research.storage import DirectoryOwner, request_identity, verified_result, verify_input_files
from .project import ProjectDocument
from .project import _freeze


@dataclass(frozen=True, slots=True)
class RunRecord:
    directory: Path
    _metadata: dict = field(repr=False)
    _arrays: dict[str, np.ndarray] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory).resolve())
        object.__setattr__(self, "_metadata", _freeze(deepcopy(self._metadata)))
        arrays = {key: np.array(value, copy=True) for key, value in self._arrays.items()}
        for value in arrays.values():
            value.flags.writeable = False
        object.__setattr__(self, "_arrays", MappingProxyType(arrays))

    @property
    def metadata(self) -> dict:
        from ..persistence import json_value

        return json_value(self._metadata)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return self.select_arrays()

    @property
    def array_names(self) -> tuple[str, ...]:
        return tuple(self._arrays)

    def array(self, name: str, *, copy: bool = True) -> np.ndarray:
        """Copy one saved array, or explicitly borrow a read-only view."""
        value = self._arrays[name]
        return value.copy() if copy else value.view()

    def select_arrays(self, names=None, *, copy: bool = True) -> dict[str, np.ndarray]:
        """Avoid copying unrelated field maps; existing ``arrays`` stays detached."""
        if isinstance(names, str):
            raise TypeError("Array names must be a collection, not a single string.")
        return {name: self.array(name, copy=copy) for name in (self.array_names if names is None else names)}


class DesignService:
    """Same operations for Qt, notebooks, batch scripts and future adapters."""

    @staticmethod
    def validate(project: ProjectDocument):
        """Shared lightweight checks; does not assemble operators or invoke a solver."""
        from ..config_validation import validate_config

        return validate_config(project.config, project_root=project.project_root)

    @classmethod
    def preflight(cls, project: ProjectDocument, *, pipeline_stage=None):
        from .preflight import check_preflight

        return check_preflight(project, validator=cls.validate, pipeline_stage=pipeline_stage)

    @staticmethod
    def plan(project: ProjectDocument, *, states=(), reuse=None, **request_controls):
        from .execution import plan_execution

        return plan_execution(project, states=states, reuse=reuse, **request_controls)

    @classmethod
    def submit_plan(cls, project: ProjectDocument, jobs, plan):
        """Reject stale plans and recheck execution state before submitting."""
        from .execution import matches_active_request
        from ..research.jobs import JobManager

        plan.require_current(project)
        controls = plan.request_controls()
        if isinstance(jobs, JobManager):
            return cls.submit(project, jobs, reuse_active=True, **controls)
        existing = next((state for state in jobs.list() if matches_active_request(state, project, controls)), None)
        return existing if existing is not None else cls.submit(project, jobs, **controls)

    @staticmethod
    def prepare(project: ProjectDocument, output_dir: str | Path, **request_controls):
        DesignService.preflight(project, pipeline_stage=request_controls.get("pipeline_stage")).raise_for_errors()
        request_controls.setdefault("project_fingerprint", project.fingerprint)
        return project.snapshot(output_dir, **request_controls)

    @classmethod
    def submit(cls, project: ProjectDocument, jobs, *, reuse_active=False, **request_controls) -> dict:
        root = Path(jobs.directory) / "inputs" / project.id / project.fingerprint
        request = cls.prepare(project, root, **request_controls)
        state = jobs.submit(request, reuse_active=True) if reuse_active else jobs.submit(request)
        state.update(
            project_id=project.id,
            project_revision=project.revision,
            project_fingerprint=project.fingerprint,
            project_model_fingerprint=project.model_fingerprint,
        )
        return state

    @classmethod
    def run(
        cls,
        project: ProjectDocument,
        output_dir: str | Path,
        *,
        threads: int = 1,
        progress=None,
        cancel_check=None,
        **request_controls,
    ) -> dict:
        """Run with optional typed events and cooperative engine cancellation.

        A returned cached result is evidence availability, not new computation.
        Cancellation uses the engine's existing checks; it cannot interrupt an
        external solver at an arbitrary instruction.
        """
        from ..progress import DesignCancelled, progress_context
        from .execution import ExecutionEvent

        for callback in (progress, cancel_check):
            if callback is not None and not callable(callback):
                raise TypeError("Progress and cancellation callbacks must be callable.")

        def emit(event):
            if progress is not None:
                try:
                    progress(event)
                except Exception as exc:
                    logging.getLogger(__name__).warning("Progress observer failed: %s", exc)

        def sink(event):
            emit(ExecutionEvent.from_progress(event))

        context = (
            progress_context(sink, cancel_check) if progress is not None or cancel_check is not None else nullcontext()
        )
        emit(ExecutionEvent("started", payload={"project_id": project.id, "project_revision": project.revision}))
        try:
            with context:
                if cancel_check is not None and cancel_check():
                    raise DesignCancelled("Design cancelled by the user.")
                result = cls._run(project, output_dir, threads=threads, **request_controls)
        except DesignCancelled as exc:
            emit(ExecutionEvent("cancelled", str(exc)))
            raise
        except Exception as exc:
            emit(ExecutionEvent("failed", str(exc)))
            raise
        emit(ExecutionEvent("result_available", payload={"request_sha256": result.get("request_sha256")}))
        return result

    @classmethod
    def _run(cls, project: ProjectDocument, output_dir: str | Path, *, threads: int = 1, **request_controls) -> dict:
        """Run or replay an identical verified result; never replace completed history.

        The same OS directory lock is used by legacy runners and the application.
        Identity checks happen before writing durable requests or input assets.
        """
        from ..research.application import execute_owned_design
        from ..research.jobs import RuntimePolicy

        RuntimePolicy(threads_per_worker=threads)
        directory = Path(output_dir).expanduser().resolve()
        with DirectoryOwner(directory):
            if (directory / "result.json").exists():
                # Content identity excludes these temporary asset locations.
                # Do not add assets to, or edit manifests in, historical output.
                with TemporaryDirectory(prefix="coilgen-replay-") as preview:
                    request = cls.prepare(project, preview, **request_controls)
                    digest, _ = request_identity(request.to_dict(), asdict(request.to_config(threads=threads)))
                completed = verified_result(directory, digest)
                if completed is None:
                    raise ValueError(
                        "This output directory already contains a completed result whose numerical request "
                        "differs or whose artifacts fail integrity verification. Choose a new output directory "
                        "to preserve the historical run."
                    )
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("request_sha256") != digest:
                    raise ValueError("Historical run manifest identity is invalid; choose a new output directory.")
                verify_input_files(manifest)
                return deepcopy(completed)
            request = cls.prepare(project, directory, **request_controls)
            return execute_owned_design(request, directory, threads=threads)

    @classmethod
    def submit_assembly(cls, assembly, jobs, **request_controls) -> dict[str, dict]:
        """Queue enabled channels independently, preserving their scientific scope."""
        from .assembly import ProjectAssembly

        if not isinstance(assembly, ProjectAssembly):
            raise TypeError("assembly must be a ProjectAssembly.")
        enabled = [channel for channel in assembly.channels if channel.enabled]
        for channel in enabled:
            cls.validate(channel.project).raise_for_errors()
        submitted = {}
        try:
            for channel in enabled:
                submitted[channel.id] = cls.submit(channel.project, jobs, **request_controls)
        except Exception:
            # A bounded queue can fill between submissions. Leave no unnoticed
            # channel running after an unsuccessful collection submission.
            for state in submitted.values():
                jobs.cancel(state["id"])
            raise
        return submitted

    @classmethod
    def run_assembly(cls, assembly, output_dir: str | Path, *, threads: int = 1, **request_controls) -> dict[str, dict]:
        """Run enabled channels; this does not evaluate channel interactions."""
        from .assembly import ProjectAssembly

        if not isinstance(assembly, ProjectAssembly):
            raise TypeError("assembly must be a ProjectAssembly.")
        enabled = [channel for channel in assembly.channels if channel.enabled]
        for channel in enabled:
            cls.validate(channel.project).raise_for_errors()
        return {
            channel.id: cls.run(channel.project, Path(output_dir) / channel.id, threads=threads, **request_controls)
            for channel in enabled
        }

    @staticmethod
    def load_run(directory: str | Path, *, array_names=None) -> RunRecord:
        """Verify all evidence, optionally decompressing only selected NPZ arrays."""
        from .evidence import verified_run_metadata

        if isinstance(array_names, str):
            raise TypeError("Array names must be a collection, not a single string.")
        names = None if array_names is None else tuple(array_names)
        directory = Path(directory).expanduser().resolve()
        result, _manifest, _verification = verified_run_metadata(directory)
        arrays = {}
        for artifact in result["artifacts"]:
            if artifact.get("kind") != "npz":
                continue
            path = directory / artifact["name"]
            if path.name != "scientific_result.npz":
                continue
            with np.load(path, allow_pickle=False) as data:
                selected = data.files if names is None else names
                arrays = {key: data[key] for key in selected}
        if names is not None and set(names) - set(arrays):
            raise KeyError(
                "Requested scientific arrays are unavailable: " + ", ".join(sorted(set(names) - set(arrays)))
            )
        job = directory / "job.json"
        if job.is_file():
            result["job"] = json.loads(job.read_text(encoding="utf-8"))
        return RunRecord(directory, deepcopy(result), arrays)
