"""Single-file binary transport for owned inputs and immutable run evidence.

The readable ZIP manifest refers to content-addressed bytes. No array is recast,
no archive member is extracted, and no pickle is loaded. Scientific identities
remain independent of this transport, as they are for the original inline JSON.
"""

from __future__ import annotations

from hashlib import sha256
import json
from lzma import LZMAError
import os
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, is_zipfile
from zlib import error as DeflateError

from .assets import InputAsset
from .bundles import RunArchive, _digest
from .project import ProjectDocument, _json_object


BUNDLE_SCHEMA = "coilgen-portable-bundle-v1"
_PROJECT = "coilgen-binary-project-v1"
_ASSEMBLY = "coilgen-binary-independent-channels-v1"
_MAX_MANIFEST_BYTES = 16 * 1024**2
DEFAULT_MAX_BYTES = 8 * 1024**3


def is_bundle(path):
    return is_zipfile(path)


def _read_member(archive, name):
    """Present unsupported compression and damaged streams as input errors."""
    try:
        return archive.read(name)
    except (BadZipFile, NotImplementedError, RuntimeError, OSError, EOFError, DeflateError, LZMAError) as exc:
        raise ValueError("Compact bundle member is corrupt or uses unsupported compression.") from exc


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Malformed compact {label}.")


def _project_manifest(project, blobs):
    def blob(payload):
        digest = sha256(payload).hexdigest()
        blobs.setdefault(digest, payload)
        return {"sha256": digest, "size": len(payload)}

    return {
        "schema": _PROJECT,
        "id": project.id,
        "name": project.name,
        "revision": project.revision,
        "config": project.values,
        "study": project.study,
        "session": project.session,
        "assets": {
            id_: {**blob(asset.data), "suffix": asset.suffix, "kind": asset.kind, "source": asset.source}
            for id_, asset in project._assets.items()
        },
        "results": [
            {"id": run.id, "alias": run.alias, "files": {name: blob(payload) for name, payload in run._files}}
            for run in project.saved_runs
        ],
    }


def save_bundle(document, path, *, progress=None):
    """Atomically write exact payload bytes once, reporting processed raw bytes."""
    from .assembly import ProjectAssembly

    path = Path(path).expanduser().resolve()
    blobs = {}
    if isinstance(document, ProjectDocument):
        value = _project_manifest(document, blobs)
    elif isinstance(document, ProjectAssembly):
        channels = []
        for channel in document.channels:
            project = _project_manifest(channel.project, blobs)
            channels.append(
                {
                    "id": channel.id,
                    "name": channel.name,
                    "enabled": channel.enabled,
                    "project": project,
                    "sha256": sha256(_json_bytes(project)).hexdigest(),
                }
            )
        value = {
            "schema": _ASSEMBLY,
            "id": document.id,
            "name": document.name,
            "revision": document.revision,
            "units": "SI",
            "coordinate_frame": "world",
            "channels": channels,
        }
    else:
        raise TypeError("A compact bundle requires a ProjectDocument or ProjectAssembly.")
    envelope = {"schema": BUNDLE_SCHEMA, "document": value, "sha256": sha256(_json_bytes(value)).hexdigest()}
    manifest = _json_bytes(envelope)
    if len(manifest) > _MAX_MANIFEST_BYTES:
        raise ValueError("Compact project manifest exceeds its supported size.")
    total, completed = sum(map(len, blobs.values())), 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                archive.writestr("manifest.json", manifest)
                if progress:
                    progress(0, total, "Writing compact project")
                for digest, payload in blobs.items():
                    archive.writestr(f"blobs/{digest}", payload)
                    completed += len(payload)
                    if progress:
                        progress(completed, total, "Writing compact project")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_bundle(path, *, progress=None, max_bytes=DEFAULT_MAX_BYTES):
    """Verify ZIP contents before constructing models or interpreting evidence.

    ``max_bytes`` bounds expanded payloads, not compressed file size. Increase it
    explicitly for a legitimately larger study. Paths are never extracted.
    """
    from .assembly import ChannelDocument, ProjectAssembly

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("Compact bundle expanded-byte limit must be a positive integer.")
    path = Path(path).expanduser().resolve()
    try:
        with ZipFile(path) as archive:
            members = archive.infolist()
            names = {member.filename: member for member in members}
            if len(names) != len(members) or len({name.casefold() for name in names}) != len(names):
                raise ValueError("Compact bundle has duplicate archive members.")
            if "manifest.json" not in names or names["manifest.json"].file_size > _MAX_MANIFEST_BYTES:
                raise ValueError("Compact bundle has no supported readable manifest.")
            if any(member.flag_bits & 1 or member.is_dir() for member in members):
                raise ValueError("Compact bundle cannot contain encrypted members or directories.")
            envelope = json.loads(_read_member(archive, "manifest.json"), object_pairs_hook=_json_object)
            _exact(envelope, ("schema", "document", "sha256"), "bundle envelope")
            if envelope["schema"] != BUNDLE_SCHEMA:
                raise ValueError("Unsupported compact bundle schema.")
            value = envelope["document"]
            if sha256(_json_bytes(value)).hexdigest() != _digest(envelope["sha256"]):
                raise ValueError("Compact manifest integrity check failed.")
            references = {}

            def reference(item):
                _exact(item, ("sha256", "size"), "blob reference")
                digest, size = _digest(item["sha256"]), item["size"]
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError("Compact blob requires a nonnegative exact byte count.")
                if digest in references and references[digest] != size:
                    raise ValueError("Compact blob references disagree about byte count.")
                references[digest] = size

            def project_references(project):
                _exact(
                    project,
                    ("schema", "id", "name", "revision", "config", "study", "session", "assets", "results"),
                    "project",
                )
                if (
                    project["schema"] != _PROJECT
                    or not isinstance(project["assets"], dict)
                    or not isinstance(project["results"], list)
                ):
                    raise ValueError("Unsupported compact project schema.")
                for id_, item in project["assets"].items():
                    _exact(item, ("sha256", "size", "suffix", "kind", "source"), "input asset")
                    if id_ != item["sha256"]:
                        raise ValueError("Compact project asset identity disagrees with its blob.")
                    reference({key: item[key] for key in ("sha256", "size")})
                for run in project["results"]:
                    _exact(run, ("id", "alias", "files"), "run archive")
                    if not isinstance(run["files"], dict):
                        raise ValueError("Compact run files must be an object.")
                    for item in run["files"].values():
                        reference(item)

            if isinstance(value, dict) and value.get("schema") == _ASSEMBLY:
                _exact(
                    value,
                    ("schema", "id", "name", "revision", "units", "coordinate_frame", "channels"),
                    "channel assembly",
                )
                if (
                    value["units"] != "SI"
                    or value["coordinate_frame"] != "world"
                    or not isinstance(value["channels"], list)
                ):
                    raise ValueError("Compact channel assemblies require SI/world coordinates.")
                for channel in value["channels"]:
                    _exact(channel, ("id", "name", "enabled", "project", "sha256"), "channel")
                    if sha256(_json_bytes(channel["project"])).hexdigest() != _digest(channel["sha256"]):
                        raise ValueError("Compact channel project integrity check failed.")
                    project_references(channel["project"])
            else:
                project_references(value)
            if set(names) != {"manifest.json", *(f"blobs/{digest}" for digest in references)}:
                raise ValueError("Compact bundle contains missing or unreferenced archive members.")
            total = sum(references.values())
            if total > max_bytes:
                raise ValueError(
                    "Compact bundle exceeds the expanded-byte limit; increase max_bytes explicitly if expected."
                )
            blobs, completed = {}, 0
            if progress:
                progress(0, total, "Verifying compact project")
            for digest, size in references.items():
                name = f"blobs/{digest}"
                if names[name].file_size != size:
                    raise ValueError("Compact blob expanded size differs from its manifest.")
                payload = _read_member(archive, name)
                if len(payload) != size or sha256(payload).hexdigest() != digest:
                    raise ValueError("Compact blob integrity check failed.")
                blobs[digest] = payload
                completed += size
                if progress:
                    progress(completed, total, "Verifying compact project")
    except (BadZipFile, UnicodeDecodeError, json.JSONDecodeError, EOFError) as exc:
        raise ValueError("Compact bundle is corrupt or contains invalid JSON.") from exc

    def project(item):
        assets = {
            id_: InputAsset(blobs[asset["sha256"]], asset["suffix"], asset["kind"], asset["source"])
            for id_, asset in item["assets"].items()
        }
        runs = tuple(
            RunArchive(
                run["id"], run["alias"], tuple((name, blobs[blob["sha256"]]) for name, blob in run["files"].items())
            )
            for run in item["results"]
        )
        return ProjectDocument(
            item["id"],
            item["name"],
            item["revision"],
            item["config"],
            assets,
            path.parent,
            _study=item["study"],
            _runs=runs,
            _session=item["session"],
        )

    if value["schema"] == _ASSEMBLY:
        return ProjectAssembly(
            value["id"],
            value["name"],
            value["revision"],
            tuple(
                ChannelDocument(channel["id"], channel["name"], project(channel["project"]), channel["enabled"])
                for channel in value["channels"]
            ),
        )
    return project(value)
