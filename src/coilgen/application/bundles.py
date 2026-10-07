"""Portable, immutable completed-run evidence embedded in project JSON."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from ..research.storage import result_metadata_digest
from .service import DesignService, RunRecord


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Run archive requires a lowercase SHA-256 content hash.")
    return value


def _relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value or any(c in value for c in ':*?<>|"'):
        raise ValueError("Run archive file name is not a safe relative POSIX path.")
    path = PurePosixPath(value)
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        "conin$",
        "conout$",
        "clock$",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
        *(f"{prefix}{number}" for prefix in ("com", "lpt") for number in "¹²³"),
    }
    if (
        not path.parts
        or path.is_absolute()
        or str(path) != value
        or any(
            part in {".", ".."}
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].lower() in reserved
            or any(ord(c) < 32 for c in part)
            for part in path.parts
        )
        or value.casefold() == "job.json"
    ):
        raise ValueError("Run archive file name is unsafe or contains live job state.")
    return value


def _archive_identity(files):
    hashes = {name: sha256(payload).hexdigest() for name, payload in files}
    return sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RunArchive:
    id: str
    alias: str
    _files: tuple[tuple[str, bytes], ...] = field(repr=False)
    project_id: str | None = field(init=False)

    def __post_init__(self):
        _digest(self.id)
        if not isinstance(self.alias, str) or not self.alias:
            raise ValueError("Run archive alias must be a nonempty string.")
        files = tuple(sorted(self._files))
        names = set()
        for name, payload in files:
            _relative_name(name)
            if name.casefold() in names or not isinstance(payload, bytes):
                raise ValueError("Run archive contains duplicate file paths or a mutable payload.")
            names.add(name.casefold())
        if any(
            str(parent).casefold() in names
            for name, _ in files
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        ):
            raise ValueError("Run archive file paths conflict with a parent directory.")
        if _archive_identity(files) != self.id:
            raise ValueError("Run archive content identity does not match its files.")
        object.__setattr__(self, "_files", files)
        result = self._validate_contents()
        project_id = result.get("project_id")
        if project_id is not None and not isinstance(project_id, str):
            raise ValueError("Archived run project identity must be a string or null.")
        # Native headless requests historically use "" for an unspecified
        # owner. Normalize only the operational ownership view; archived JSON,
        # scientific provenance and all content digests remain byte-exact.
        object.__setattr__(self, "project_id", project_id or None)

    def _validate_contents(self):
        files = dict(self._files)
        try:
            result, manifest, request = (
                json.loads(files[name]) for name in ("result.json", "manifest.json", "request.json")
            )
        except (KeyError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Run archive is missing valid result, manifest or request JSON.") from exc
        if not all(isinstance(value, dict) for value in (result, manifest, request)):
            raise ValueError("Archived run metadata must contain JSON objects.")
        digest = _digest(result.get("request_sha256"))
        if manifest.get("request_sha256") != digest:
            raise ValueError("Archived run request and manifest identities disagree.")
        if "request" in manifest and manifest["request"] != request:
            raise ValueError("Archived request JSON disagrees with the original manifest request.")
        if "metadata_sha256" in result and result["metadata_sha256"] != result_metadata_digest(result):
            raise ValueError("Archived run result metadata integrity check failed.")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("Archived run has no completed artifacts.")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError("Archived artifact description is invalid.")
            name = _relative_name(artifact.get("name"))
            expected = _digest(artifact.get("sha256"))
            if name not in files or sha256(files[name]).hexdigest() != expected:
                raise ValueError(f"Archived artifact integrity check failed: {name}.")
        inputs = manifest.get("input_files", {})
        if not isinstance(inputs, dict):
            raise ValueError("Archived input manifest is invalid.")
        for item in inputs.values():
            if not isinstance(item, dict):
                raise ValueError("Archived input description is invalid.")
            expected = _digest(item.get("sha256"))
            name = f"portable_inputs/{expected}"
            if name not in files or sha256(files[name]).hexdigest() != expected:
                raise ValueError("Archived run is missing an exact portable input copy.")
        return result

    @property
    def metadata(self) -> dict:
        """Detached original result metadata, without transient live job state."""
        return json.loads(dict(self._files)["result.json"])

    @classmethod
    def capture(cls, record: RunRecord, alias: str | None = None):
        verified = DesignService.load_run(record.directory)
        directory = verified.directory
        result = json.loads((directory / "result.json").read_bytes())
        names = {"result.json", "manifest.json", "request.json"}
        names.update(_relative_name(item["name"]) for item in result["artifacts"])
        files = {}
        for name in names:
            path = (directory / _relative_name(name)).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Archived artifact leaves the source run directory.")
            files[name] = path.read_bytes()
        manifest = json.loads(files["manifest.json"])
        for item in manifest.get("input_files", {}).values():
            expected = _digest(item["sha256"])
            portable = directory / "portable_inputs" / expected
            source = portable if portable.exists() or portable.is_symlink() else Path(item["path"])
            payload = source.read_bytes()
            if sha256(payload).hexdigest() != expected:
                raise ValueError("Run input changed while its archive was being captured.")
            files[f"portable_inputs/{expected}"] = payload
        payloads = tuple(sorted(files.items()))
        if alias is None:
            name = verified.metadata.get("name")
            alias = name.strip()[:120] if isinstance(name, str) and name.strip() else directory.name[:120]
        return cls(_archive_identity(payloads), alias, payloads)

    def to_dict(self):
        return {
            "schema": "coilgen-run-archive-v1",
            "id": self.id,
            "alias": self.alias,
            "files": {
                name: {
                    "encoding": "base64",
                    "sha256": sha256(payload).hexdigest(),
                    "data": base64.b64encode(payload).decode("ascii"),
                }
                for name, payload in self._files
            },
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema") != "coilgen-run-archive-v1":
            raise ValueError("Unrecognized run archive schema.")
        if set(value) != {"schema", "id", "alias", "files"} or not isinstance(value["files"], dict):
            raise ValueError("Malformed run archive envelope.")
        files = []
        for name, item in value["files"].items():
            _relative_name(name)
            if (
                not isinstance(item, dict)
                or set(item) != {"encoding", "sha256", "data"}
                or item["encoding"] != "base64"
            ):
                raise ValueError("Malformed run archive file encoding.")
            if not isinstance(item["data"], str):
                raise ValueError("Run archive file data must be base64 text.")
            try:
                payload = base64.b64decode(item["data"], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid base64 in run archive.") from exc
            if sha256(payload).hexdigest() != _digest(item["sha256"]):
                raise ValueError(f"Run archive file integrity check failed: {name}.")
            files.append((name, payload))
        return cls(value["id"], value["alias"], tuple(files))

    def _verify_materialized(self, directory):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Archived run cache is not a regular directory.")
        for name, payload in self._files:
            path = directory / name
            if (
                not path.resolve().is_relative_to(directory)
                or not path.is_file()
                or sha256(path.read_bytes()).digest() != sha256(payload).digest()
            ):
                raise ValueError(f"Existing archived run cache failed integrity verification: {name}.")
        return DesignService.load_run(directory)

    def materialize(self, root: str | Path) -> RunRecord:
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / self.id
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != destination:
                raise ValueError("Archived run cache must not redirect to another directory.")
            return self._verify_materialized(destination)
        with TemporaryDirectory(prefix=f".{self.id[:12]}-", dir=root) as staging:
            directory = Path(staging)
            for name, payload in self._files:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(payload)
            self._verify_materialized(directory)
            try:
                directory.rename(destination)
            except FileExistsError:
                return self._verify_materialized(destination)
        return self._verify_materialized(destination)
