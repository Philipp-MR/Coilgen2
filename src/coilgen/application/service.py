"""Thin headless application service; numerical work stays in the shared engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
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
        return {name: value.copy() for name, value in self._arrays.items()}


class DesignService:
    """Same operations for Qt, notebooks, batch scripts and future adapters."""

    @staticmethod
    def validate(project: ProjectDocument):
        """Shared lightweight checks; does not assemble operators or invoke a solver."""
        from ..config_validation import validate_config

        return validate_config(project.config, project_root=project.project_root)

    @staticmethod
    def prepare(project: ProjectDocument, output_dir: str | Path, **request_controls):
        DesignService.validate(project).raise_for_errors()
        request_controls.setdefault("project_fingerprint", project.fingerprint)
        return project.snapshot(output_dir, **request_controls)

    @classmethod
    def submit(cls, project: ProjectDocument, jobs, **request_controls) -> dict:
        root = Path(jobs.directory) / "inputs" / project.id / project.fingerprint
        request = cls.prepare(project, root, **request_controls)
        state = jobs.submit(request)
        state.update(
            project_id=project.id,
            project_revision=project.revision,
            project_fingerprint=project.fingerprint,
            project_model_fingerprint=project.model_fingerprint,
        )
        return state

    @classmethod
    def run(cls, project: ProjectDocument, output_dir: str | Path, *, threads: int = 1, **request_controls) -> dict:
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
    def load_run(directory: str | Path) -> RunRecord:
        directory = Path(directory).expanduser().resolve()
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Malformed run result.")
        digest = result.get("request_sha256")
        if not isinstance(digest, str) or verified_result(directory, digest) is None:
            raise ValueError("Run is incomplete or its artifact integrity check failed.")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Malformed run manifest.")
        if manifest.get("request_sha256") != digest:
            raise ValueError("Run request and manifest identity disagree.")
        verify_input_files(manifest, relocated_root=directory)
        arrays = {}
        for artifact in result["artifacts"]:
            if artifact.get("kind") != "npz":
                continue
            path = directory / artifact["name"]
            if path.name != "scientific_result.npz":
                continue
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.array(data[key], copy=True) for key in data.files}
        job = directory / "job.json"
        if job.is_file():
            result["job"] = json.loads(job.read_text(encoding="utf-8"))
        return RunRecord(directory, deepcopy(result), arrays)
