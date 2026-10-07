"""Portable project revisions shared by desktop tools and ordinary scripts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from ..config import CoilGenConfig
from ..geometry import geometry_factory
from ..object_registry import (
    get_in_memory_mesh,
    get_in_memory_target_field,
    is_in_memory_mesh_key,
    is_in_memory_target_key,
)
from ..persistence import atomic_json, atomic_text, json_value
from .assets import InputAsset, mesh_asset, target_asset


PROJECT_SCHEMA = "coilgen-project-v2"
LEGACY_PROJECT_SCHEMAS = frozenset({"coilgen-project-v1"})
_STUDY_DEFAULTS = {
    "pipeline_stage": "evaluated",
    "validation_resolution": 12,
    "design_margin": 1.0,
    "fabricate": False,
    "kernel_backend": "auto",
}
_INPUT_FIELDS = {
    "coil_mesh_file",
    "target_mesh_file",
    "secondary_target_mesh_file",
    "target_field_definition_file",
    "preoptimized_stream_function_file",
    "sf_source_file",
}
_CONFIG_FIELDS = {item.name for item in fields(CoilGenConfig)}
_ASSET_PREFIX = "asset:"


def _portable_identifier(value: str) -> bool:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    return (
        isinstance(value, str)
        and 0 < len(value) <= 120
        and value.upper() not in reserved
        and all(c.isascii() and (c.isalnum() or c == "-") for c in value)
    )


def _values(config: CoilGenConfig) -> dict:
    return json_value(asdict(deepcopy(config)), nonfinite="raise")


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate project JSON property: {key}")
        value[key] = item
    return value


def _study(values):
    from ..research.requests import ResearchRequest

    if not isinstance(values, Mapping) or set(values) - set(_STUDY_DEFAULTS):
        raise ValueError("Unknown or malformed study controls.")
    resolved = {**_STUDY_DEFAULTS, **json_value(values, nonfinite="raise")}
    try:
        ResearchRequest(name="Study", config={}, **resolved)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid study controls: {exc}") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class ProjectDocument:
    """A model revision with detached configuration and owned input bytes.

    Editing returns another document; saved runs can retain this revision safely.
    ``config`` is directly usable by the original Python pipeline and resolves
    assets to a temporary owned cache. ``snapshot`` materializes durable inputs
    for worker processes. Neither interface consults the original source files.
    """

    id: str
    name: str
    revision: int
    _values: dict = field(repr=False)
    _assets: dict[str, InputAsset] = field(repr=False)
    project_root: Path = field(default_factory=Path.cwd)
    _cache: Any = field(default=None, init=False, repr=False, compare=False)
    _study: Mapping[str, Any] = field(default_factory=lambda: dict(_STUDY_DEFAULTS), repr=False)
    _runs: tuple[Any, ...] = field(default=(), repr=False)
    _session: Mapping[str, Any] = field(default_factory=dict, repr=False)
    _cache_lock: Any = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 120:
            raise ValueError("Project name must contain 1–120 characters.")
        if not _portable_identifier(self.id):
            raise ValueError("Project ID must be a nonempty portable identifier.")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ValueError("Project revision must be a positive integer.")
        unknown = set(self._values) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(f"Unknown CoilGenConfig settings: {', '.join(sorted(unknown))}")
        if set(self._values) != _CONFIG_FIELDS:
            raise ValueError("A project revision must contain every resolved CoilGenConfig setting.")
        values = json_value(self._values, nonfinite="raise")
        json.dumps(values, allow_nan=False)
        object.__setattr__(self, "_values", _freeze(values))
        object.__setattr__(self, "_assets", MappingProxyType(dict(self._assets)))
        object.__setattr__(self, "_study", _freeze(_study(self._study)))
        if not isinstance(self._session, Mapping) or any(not isinstance(key, str) for key in self._session):
            raise ValueError("Project session must be a JSON object with string keys.")
        object.__setattr__(self, "_session", _freeze(json_value(self._session, nonfinite="raise")))
        runs = tuple(self._runs)
        if runs:
            from .bundles import RunArchive

            if not all(isinstance(run, RunArchive) for run in runs):
                raise TypeError("Saved project results must be immutable RunArchive objects.")
            if len({run.id for run in runs}) != len(runs):
                raise ValueError("Saved project result archive IDs must be unique.")
            if any(run.project_id is not None and run.project_id != self.id for run in runs):
                raise ValueError("Saved run belongs to a different project owner.")
        object.__setattr__(self, "_runs", runs)
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        for key in _INPUT_FIELDS:
            value = self._values.get(key)
            if isinstance(value, str) and value.startswith(_ASSET_PREFIX) and value[6:] not in self._assets:
                raise ValueError(f"Missing owned input asset for {key}.")
        for id_, asset in self._assets.items():
            if not isinstance(asset, InputAsset) or id_ != asset.id:
                raise ValueError("Owned asset identity does not match its content.")
        for key in _INPUT_FIELDS:
            value = self._values[key]
            if not isinstance(value, str):
                raise ValueError(f"Project input {key} must be a source string.")
            if value.startswith(_ASSET_PREFIX):
                expected = (
                    "mesh"
                    if key.endswith("mesh_file")
                    else "target"
                    if key == "target_field_definition_file"
                    else "stream_function"
                )
                if self._assets[value[6:]].kind != expected:
                    raise ValueError(f"Owned input kind is incompatible with {key}.")
            elif value.lower() not in {"", "none"} and not (
                key.endswith("mesh_file") and (value.lower().startswith("create ") or geometry_factory(value))
            ):
                raise ValueError(f"Project input {key} must be owned by the project.")

    @classmethod
    def new(cls, name: str = "Coil design", config: CoilGenConfig | None = None) -> ProjectDocument:
        return cls.from_config(config or CoilGenConfig(), name=name)

    @classmethod
    def from_config(
        cls, config: CoilGenConfig, *, name: str = "Coil design", project_root: str | Path | None = None
    ) -> ProjectDocument:
        if not isinstance(config, CoilGenConfig):
            raise TypeError("config must be a CoilGenConfig.")
        root = Path.cwd() if project_root is None else Path(project_root).resolve()
        values, assets = cls._capture(config, root)
        return cls(uuid4().hex, name, 1, values, assets, root)

    @classmethod
    def from_problem(
        cls, problem, *, name: str | None = None, project_root: str | Path | None = None, **conversion_options
    ) -> ProjectDocument:
        return cls.from_config(
            problem.to_config(**conversion_options), name=name or problem.metadata.name, project_root=project_root
        )

    @staticmethod
    def _capture(config: CoilGenConfig, root: Path, existing: Mapping[str, InputAsset] | None = None):
        values = _values(config)
        assets = dict(existing or {})
        for key in _INPUT_FIELDS:
            value = str(values[key])
            if value.startswith(_ASSET_PREFIX):
                if value[6:] not in assets:
                    raise ValueError(f"Unowned input reference: {key}={value}")
                continue
            if is_in_memory_mesh_key(value):
                asset = mesh_asset(get_in_memory_mesh(value))
            elif is_in_memory_target_key(value):
                asset = target_asset(get_in_memory_target_field(value))
            elif (
                value.lower() == "none"
                or not value
                or (key.endswith("mesh_file") and (value.lower().startswith("create ") or geometry_factory(value)))
            ):
                continue
            else:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    if key.endswith("mesh_file"):
                        source_root = config.resolved_geometry_source_path(root)
                        if not source_root.is_absolute():
                            source_root = root / source_root
                        path = source_root / path
                    else:
                        path = root / path
                if not path.is_file():
                    raise FileNotFoundError(f"Project input {key} not found: {path}")
                asset = InputAsset.from_file(
                    path,
                    kind="mesh"
                    if key.endswith("mesh_file")
                    else "target"
                    if key == "target_field_definition_file"
                    else "stream_function",
                )
            if asset.id in assets and (assets[asset.id].suffix, assets[asset.id].kind) != (asset.suffix, asset.kind):
                raise ValueError(
                    "Identical input bytes with different file formats or input kinds require distinct source content."
                )
            assets[asset.id] = asset
            values[key] = _ASSET_PREFIX + asset.id
        # Keep only assets referenced by the current revision. Older revisions
        # retain their own objects and saved content-addressed files.
        ids = {
            value[6:]
            for key, value in values.items()
            if key in _INPUT_FIELDS and isinstance(value, str) and value.startswith(_ASSET_PREFIX)
        }
        return values, {id_: assets[id_] for id_ in ids}

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            {"config": self._identity_values, "assets": self._asset_formats, "study": self.study},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return sha256(encoded).hexdigest()

    @property
    def _identity_values(self) -> dict:
        values = self.values
        # The opt-in region adds no physical input when inactive. Preserve
        # identities of pre-region project JSON and its saved result evidence.
        if values.get("target_region_geometry") is None:
            values.pop("target_region_geometry", None)
        return values

    @property
    def _asset_formats(self) -> dict:
        return {id_: {"suffix": asset.suffix, "kind": asset.kind} for id_, asset in self._assets.items()}

    @property
    def model_fingerprint(self) -> str:
        """Physical/numerical input identity, independent of study and display name."""
        from ..pipeline_stages import config_field_stage

        values = {
            key: value
            for key, value in self._identity_values.items()
            if key != "geometry_source_path" and config_field_stage(key) is not None
        }
        encoded = json.dumps(
            {"config": values, "assets": self._asset_formats}, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return sha256(encoded).hexdigest()

    @property
    def values(self) -> dict:
        """Detached canonical configuration; owned sources remain ``asset:`` IDs."""
        return json_value(self._values)

    @property
    def study(self) -> dict:
        return json_value(self._study)

    @property
    def saved_runs(self) -> tuple:
        """Immutable verified result archives carried by this project file."""
        return self._runs

    @property
    def session(self) -> dict:
        """Detached display and selection state; it does not identify the model."""
        return json_value(self._session)

    def with_results(self, runs, session=None) -> ProjectDocument:
        """Attach saved evidence without creating a new scientific revision."""
        return ProjectDocument(
            self.id,
            self.name,
            self.revision,
            self._values,
            self._assets,
            self.project_root,
            _study=self._study,
            _runs=tuple(runs),
            _session=self._session if session is None else session,
        )

    def update_study(self, changes: Mapping[str, Any]) -> ProjectDocument:
        if not isinstance(changes, Mapping):
            raise TypeError("Study changes must be a mapping.")
        values = _study({**self.study, **dict(changes)})
        if values == self.study:
            return self
        return ProjectDocument(
            self.id,
            self.name,
            self.revision + 1,
            self._values,
            self._assets,
            self.project_root,
            _study=values,
            _runs=self._runs,
            _session=self._session,
        )

    def rename(self, name: str) -> ProjectDocument:
        if name == self.name:
            return self
        return ProjectDocument(
            self.id,
            name,
            self.revision + 1,
            self._values,
            self._assets,
            self.project_root,
            _study=self._study,
            _runs=self._runs,
            _session=self._session,
        )

    def duplicate(self, name: str) -> ProjectDocument:
        """Start an independent design with the same owned inputs and provenance."""
        return ProjectDocument(uuid4().hex, name, 1, self._values, self._assets, self.project_root, _study=self._study)

    def _resolved_config(self, asset_root: Path) -> CoilGenConfig:
        values = json_value(self._values)
        for key in _INPUT_FIELDS:
            value = values[key]
            if isinstance(value, str) and value.startswith(_ASSET_PREFIX):
                values[key] = str(self._assets[value[6:]].write(asset_root).resolve())
        return CoilGenConfig(**values)

    @property
    def config(self) -> CoilGenConfig:
        with self._cache_lock:
            if self._cache is None:
                object.__setattr__(self, "_cache", TemporaryDirectory(prefix="coilgen-project-"))
        return self._resolved_config(Path(self._cache.name))

    def update_config(self, changes: Mapping[str, Any], *, allow_incomplete: bool = False) -> ProjectDocument:
        """Validate and commit settings, optionally retaining an unset support.

        Desktop models may be saved and edited before selecting their support.
        This opt-in permits only that missing-input diagnostic. Physical values,
        geometry definitions and all other configuration errors remain strict;
        numerical preparation still requires the ordinary complete validator.
        """
        if not isinstance(changes, Mapping):
            raise TypeError("Configuration changes must be a mapping.")
        if not isinstance(allow_incomplete, bool):
            raise TypeError("allow_incomplete must be a boolean.")
        unknown = set(changes) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(f"Unknown CoilGenConfig settings: {', '.join(sorted(unknown))}")
        values = json_value(self._values)
        values.update(deepcopy(dict(changes)))
        captured, assets = self._capture(CoilGenConfig(**values), self.project_root, self._assets)
        if captured == json_value(self._values):
            return self
        updated = ProjectDocument(
            self.id,
            self.name,
            self.revision + 1,
            captured,
            assets,
            self.project_root,
            _study=self._study,
            _runs=self._runs,
            _session=self._session,
        )
        from ..config_validation import ConfigValidationReport, validate_config

        try:
            config = updated.config
            report = validate_config(config, project_root=self.project_root)
            if allow_incomplete and str(config.coil_mesh_file).strip().lower() in {"", "none"}:
                report = ConfigValidationReport(
                    tuple(
                        message
                        for message in report.messages
                        if not (
                            message.level == "error"
                            and message.field == "coil_mesh_file"
                            and message.message == "No coil surface is configured."
                        )
                    )
                )
            report.raise_for_errors()
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid CoilGen configuration: {exc}") from exc
        return updated

    def replace_config(self, config: CoilGenConfig) -> ProjectDocument:
        return self.update_config(asdict(config))

    def with_mesh(self, mesh, *, field: str = "target_mesh_file", changes=None) -> ProjectDocument:
        """Own an SI mesh directly, preserving studies and saved evidence.

        No process-local registry or scratch file is needed to capture the
        input. The caller can preflight the mesh with its intended sampling
        and placement before publishing this immutable project revision.
        """
        if field not in {"coil_mesh_file", "target_mesh_file", "secondary_target_mesh_file"}:
            raise ValueError("Owned mesh field must be coil_mesh_file, target_mesh_file or secondary_target_mesh_file.")
        changes = {} if changes is None else changes
        if not isinstance(changes, Mapping):
            raise TypeError("Configuration changes must be a mapping.")
        unknown = set(changes) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(f"Unknown CoilGenConfig settings: {', '.join(sorted(unknown))}")
        if field in changes:
            raise ValueError("Mesh source is supplied by with_mesh; do not override its field in changes.")
        asset = mesh_asset(mesh)
        values = self.values
        values.update(deepcopy(dict(changes)))
        values[field] = _ASSET_PREFIX + asset.id
        captured, assets = self._capture(CoilGenConfig(**values), self.project_root, {**self._assets, asset.id: asset})
        if captured == self.values:
            return self
        return ProjectDocument(
            self.id,
            self.name,
            self.revision + 1,
            captured,
            assets,
            self.project_root,
            _study=self._study,
            _runs=self._runs,
            _session=self._session,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser().resolve()
        atomic_json(path, self.to_dict())
        return path

    def to_dict(self) -> dict:
        """Single-file representation: inputs and completed evidence are inline."""
        return {
            "schema": PROJECT_SCHEMA,
            "id": self.id,
            "name": self.name,
            "revision": self.revision,
            "config": self.values,
            "assets": {id_: asset.to_dict() for id_, asset in self._assets.items()},
            "study": self.study,
            "results": [run.to_dict() for run in self._runs],
            "session": self.session,
        }

    @classmethod
    def load(cls, path: str | Path) -> ProjectDocument:
        path = Path(path).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object)
        return cls.from_dict(value, project_root=path.parent)

    @classmethod
    def from_dict(cls, value: dict, *, project_root: str | Path | None = None) -> ProjectDocument:
        root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
        expected = {"schema", "id", "name", "revision", "config", "assets"}
        schema = value.get("schema") if isinstance(value, dict) else None
        optional = {"study", "results", "session"} if schema == PROJECT_SCHEMA else {"study"}
        if (
            not isinstance(value, dict)
            or not expected <= set(value)
            or set(value) - expected - optional
            or value.get("schema") not in {PROJECT_SCHEMA, *LEGACY_PROJECT_SCHEMAS}
        ):
            raise ValueError("Unsupported or malformed CoilGen project schema.")
        if not isinstance(value["config"], dict) or not isinstance(value["assets"], dict):
            raise ValueError("Project config and assets must be JSON objects.")
        json.dumps(value, allow_nan=False)
        # Narrow migration: older files predate this optional, inactive field.
        # Missing unrelated settings still fail the resolved-config contract.
        config_values = dict(value["config"])
        config_values.setdefault("target_region_geometry", None)
        assets = {}
        for id_, descriptor in value["assets"].items():
            if value["schema"] == PROJECT_SCHEMA:
                asset = InputAsset.from_dict(descriptor)
                if id_ != asset.id:
                    raise ValueError("Project input integrity check failed for inline asset identity.")
                assets[id_] = asset
                continue
            if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256", "kind", "source"}:
                raise ValueError("Malformed project asset descriptor.")
            if not isinstance(descriptor["path"], str) or not isinstance(descriptor["source"], str):
                raise ValueError("Asset paths and source provenance must be strings.")
            if not isinstance(descriptor["kind"], str) or descriptor["kind"] not in {
                "mesh",
                "target",
                "stream_function",
            }:
                raise ValueError("Unsupported project input asset kind.")
            location = (root / descriptor["path"]).resolve()
            if Path(descriptor["path"]).is_absolute() or not location.is_relative_to(root):
                raise ValueError("Project asset path must stay inside the project directory.")
            asset = InputAsset.from_file(location, kind=descriptor["kind"])
            if id_ != asset.id or id_ != descriptor["sha256"]:
                raise ValueError(f"Project input integrity check failed: {location}")
            assets[id_] = InputAsset(asset.data, asset.suffix, asset.kind, descriptor["source"])
        results = value.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Saved project results must be a JSON array.")
        runs = ()
        if results:
            from .bundles import RunArchive

            runs = tuple(RunArchive.from_dict(item) for item in results)
        return cls(
            value["id"],
            value["name"],
            value["revision"],
            config_values,
            assets,
            root,
            _study=value.get("study", _STUDY_DEFAULTS),
            _runs=runs,
            _session=value.get("session", {}),
        )

    def snapshot(self, directory: str | Path, **request_controls):
        from ..research.requests import ResearchRequest

        directory = Path(directory).expanduser().resolve()
        config = self._resolved_config(directory / "inputs")
        config.output_directory = directory
        controls = {**self.study, **request_controls}
        for key, value in (
            ("project_fingerprint", self.fingerprint),
            ("project_model_fingerprint", self.model_fingerprint),
            ("project_id", self.id),
            ("project_revision", self.revision),
        ):
            if key in request_controls and request_controls[key] != value:
                raise ValueError(f"Snapshot {key} must identify this project revision.")
            controls[key] = value
        return ResearchRequest.from_config(config, name=self.name, **controls)

    def export_script(self, path: str | Path) -> Path:
        """Export a directly runnable script plus its portable project inputs."""
        path = Path(path).expanduser().resolve()
        project_path = path.with_suffix(".coilgen.json")
        self.save(project_path)
        text = (
            '"""Reproducible CoilGen2 design exported from a project revision."""\n'
            "from pathlib import Path\n"
            "from coilgen.application import DesignService, ProjectDocument\n\n"
            'if __name__ == "__main__":\n'
            f"    project = ProjectDocument.load(Path(__file__).with_name({project_path.name!r}))\n"
            '    result = DesignService.run(project, Path(__file__).with_suffix(".results"))\n'
            "\n"
            "# Existing direct numerical scripts remain supported:\n"
            "# from coilgen import run_coilgen\n"
            "# result = run_coilgen(project.config, project_root=Path(__file__).parent)\n"
        )
        atomic_text(path, text)
        return path

    export_python = export_script
