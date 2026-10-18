"""Portable independent-channel projects; no implicit electromagnetic coupling.

Each channel is an ordinary owned ProjectDocument in the existing SI world
frame. This container neither infers a circuit nor certifies mutual inductance,
assembly clearance, combined fields, or joint drive limits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from ..persistence import atomic_json, file_digest
from .project import PROJECT_SCHEMA, ProjectDocument, _json_object, _portable_identifier


ASSEMBLY_SCHEMA = "coilgen-independent-channels-v2"
LEGACY_ASSEMBLY_SCHEMAS = frozenset({"coilgen-independent-channels-v1"})


def _identifier(value: str) -> None:
    if not _portable_identifier(value):
        raise ValueError("Channel and assembly IDs must be portable identifiers.")


def _name(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 120:
        raise ValueError("Channel and assembly names must contain 1–120 characters.")


def _project_digest(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ChannelDocument:
    id: str
    name: str
    project: ProjectDocument
    enabled: bool = True

    def __post_init__(self) -> None:
        _identifier(self.id)
        _name(self.name)
        if not isinstance(self.project, ProjectDocument):
            raise TypeError("A channel must own a ProjectDocument.")
        if not isinstance(self.enabled, bool):
            raise ValueError("Channel enabled must be a boolean.")


@dataclass(frozen=True, slots=True)
class ProjectAssembly:
    id: str
    name: str
    revision: int
    channels: tuple[ChannelDocument, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.id)
        _name(self.name)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ValueError("Assembly revision must be a positive integer.")
        channels = tuple(self.channels)
        if not all(isinstance(channel, ChannelDocument) for channel in channels):
            raise TypeError("Assembly channels must be ChannelDocument objects.")
        if len({channel.id for channel in channels}) != len(channels):
            raise ValueError("Channel IDs must be unique within an assembly.")
        if len({channel.name.strip().casefold() for channel in channels}) != len(channels):
            raise ValueError("Channel names must be unique within an assembly.")
        if len({channel.project.id for channel in channels}) != len(channels):
            raise ValueError(
                "Each independent channel must own a unique ProjectDocument ID; "
                "use project.duplicate(name) to create an independent channel with the same configuration."
            )
        object.__setattr__(self, "channels", channels)

    @classmethod
    def new(cls, name: str = "Coil assembly", projects: Iterable[ProjectDocument] = ()) -> ProjectAssembly:
        return cls(
            uuid4().hex, name, 1, tuple(ChannelDocument(uuid4().hex, project.name, project) for project in projects)
        )

    @classmethod
    def from_coil_project(cls, project) -> ProjectAssembly:
        """Freeze resolved inputs from the existing script-facing CoilProject."""
        from ..project import CoilProject

        if not isinstance(project, CoilProject):
            raise TypeError("project must be an existing CoilProject.")
        resolved = project.resolved_designs(include_disabled=True)
        root = Path.cwd() if project.project_root is None else Path(project.project_root)
        channels = []
        for channel in project.channels:
            design = resolved[channel.name]
            if not hasattr(design, "to_config"):
                raise TypeError(f"Channel {channel.name!r} does not expose a native CoilGenConfig.")
            document = ProjectDocument.from_config(design.to_config(), name=channel.name, project_root=root)
            channels.append(ChannelDocument(uuid4().hex, channel.name, document, channel.enabled))
        return cls(uuid4().hex, project.name, 1, tuple(channels))

    @property
    def fingerprint(self) -> str:
        payload = {
            # Scientific identity is independent of the persistence format.
            "scope": "coilgen-independent-channels-v1",
            "channels": [
                {
                    "id": channel.id,
                    "name": channel.name,
                    "enabled": channel.enabled,
                    "project": channel.project.fingerprint,
                }
                for channel in self.channels
            ],
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def channel(self, id_: str) -> ChannelDocument:
        for channel in self.channels:
            if channel.id == id_:
                return channel
        raise KeyError(id_)

    def add_channel(
        self, project: ProjectDocument, *, name: str | None = None, enabled: bool = True
    ) -> ProjectAssembly:
        channel = ChannelDocument(uuid4().hex, project.name if name is None else name, project, enabled)
        return replace(self, revision=self.revision + 1, channels=(*self.channels, channel))

    def update_channel(
        self, id_: str, *, project: ProjectDocument | None = None, name: str | None = None, enabled: bool | None = None
    ) -> ProjectAssembly:
        old = self.channel(id_)
        channel = ChannelDocument(
            old.id,
            old.name if name is None else name,
            old.project if project is None else project,
            old.enabled if enabled is None else enabled,
        )
        if channel == old:
            return self
        return replace(
            self,
            revision=self.revision + 1,
            channels=tuple(channel if item.id == id_ else item for item in self.channels),
        )

    def remove_channel(self, id_: str) -> ProjectAssembly:
        self.channel(id_)
        return replace(
            self, revision=self.revision + 1, channels=tuple(channel for channel in self.channels if channel.id != id_)
        )

    def rename(self, name: str) -> ProjectAssembly:
        return self if name == self.name else replace(self, name=name, revision=self.revision + 1)

    def save(self, path: str | Path, *, progress=None) -> Path:
        """Atomically save JSON or a compact ``.coilgen`` binary evidence bundle."""
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".coilgen":
            from .portable import save_bundle

            return save_bundle(self, path, progress=progress)
        atomic_json(path, self.to_dict())
        return path

    def to_dict(self) -> dict:
        channels = []
        for channel in self.channels:
            project = channel.project.to_dict()
            channels.append(
                {
                    "id": channel.id,
                    "name": channel.name,
                    "enabled": channel.enabled,
                    "project": project,
                    "sha256": _project_digest(project),
                }
            )
        return {
            "schema": ASSEMBLY_SCHEMA,
            "id": self.id,
            "name": self.name,
            "revision": self.revision,
            "units": "SI",
            "coordinate_frame": "world",
            "channels": channels,
        }

    @classmethod
    def load(cls, path: str | Path, *, progress=None) -> ProjectAssembly:
        path = Path(path).expanduser().resolve()
        from .portable import is_bundle, load_bundle

        if is_bundle(path):
            document = load_bundle(path, progress=progress)
            if not isinstance(document, cls):
                raise ValueError("Compact file contains a single project, not an independent-channel assembly.")
            return document
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object)
        return cls.from_dict(value, project_root=path.parent)

    @classmethod
    def from_dict(cls, value: dict, *, project_root: str | Path | None = None) -> ProjectAssembly:
        root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
        expected = {"schema", "id", "name", "revision", "units", "coordinate_frame", "channels"}
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value["schema"] not in {ASSEMBLY_SCHEMA, *LEGACY_ASSEMBLY_SCHEMAS}
        ):
            raise ValueError("Unsupported or malformed independent-channel project schema.")
        json.dumps(value, allow_nan=False)
        if value["units"] != "SI" or value["coordinate_frame"] != "world" or not isinstance(value["channels"], list):
            raise ValueError("Independent-channel projects require SI units in the world coordinate frame.")
        channels = []
        for item in value["channels"]:
            if not isinstance(item, dict) or set(item) != {"id", "name", "enabled", "project", "sha256"}:
                raise ValueError("Malformed channel descriptor.")
            if not isinstance(item["sha256"], str):
                raise ValueError("Malformed channel project digest.")
            if value["schema"] == ASSEMBLY_SCHEMA:
                if not isinstance(item["project"], dict) or item["project"].get("schema") != PROJECT_SCHEMA:
                    raise ValueError("Channel project must be embedded in the collection JSON.")
                if _project_digest(item["project"]) != item["sha256"]:
                    raise ValueError("Channel project integrity check failed.")
                project = ProjectDocument.from_dict(item["project"], project_root=root)
            else:
                if not isinstance(item["project"], str):
                    raise ValueError("Malformed channel project path.")
                location = (root / item["project"]).resolve()
                if Path(item["project"]).is_absolute() or not location.is_relative_to(root):
                    raise ValueError("Channel project path must stay inside the collection directory.")
                if file_digest(location) != item["sha256"]:
                    raise ValueError("Channel project integrity check failed.")
                project = ProjectDocument.load(location)
            channels.append(ChannelDocument(item["id"], item["name"], project, item["enabled"]))
        return cls(value["id"], value["name"], value["revision"], tuple(channels))
