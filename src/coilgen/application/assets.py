"""Owned, content-addressed numerical inputs; no pickle or mutable source links."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import uuid

import numpy as np

from ..models import Mesh, TargetField
from ..persistence import file_digest, json_value


def project_reference_path(root: Path, reference: str, *, error: str) -> Path:
    """Resolve a contained legacy reference using both platforms' path rules."""
    # Native POSIX Path treats C:/..., C:relative and Windows UNC/rooted paths
    # as relative filenames. Reject their anchors before native resolution.
    if not reference or PureWindowsPath(reference).anchor or PurePosixPath(reference).is_absolute():
        raise ValueError(error)
    # Legacy Windows-relative references must retain their directory meaning
    # when the project is moved to POSIX, including traversal containment.
    location = (root / reference.replace("\\", "/")).resolve()
    if not location.is_relative_to(root):
        raise ValueError(error)
    return location


@dataclass(frozen=True, slots=True)
class InputAsset:
    data: bytes
    suffix: str
    kind: str
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError("Owned asset content must be bytes.")
        object.__setattr__(self, "data", bytes(self.data))
        if not isinstance(self.kind, str) or self.kind not in {"mesh", "target", "stream_function"}:
            raise ValueError("Unsupported project input asset kind.")
        if (
            not isinstance(self.suffix, str)
            or not self.suffix.startswith(".")
            or (len(self.suffix) > 16 or not self.suffix[1:].isalnum())
        ):
            raise ValueError("Owned asset suffix must be a portable file extension.")
        object.__setattr__(self, "suffix", self.suffix.lower())
        if not isinstance(self.source, str):
            raise TypeError("Owned asset source provenance must be a string.")

    @property
    def id(self) -> str:
        return sha256(self.data).hexdigest()

    @property
    def filename(self) -> str:
        return self.id + self.suffix

    def write(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / self.filename
        if path.exists():
            if not path.is_file() or file_digest(path) != self.id:
                raise ValueError(f"Owned input asset is corrupt: {path}")
            return path
        temporary = root / f".{self.id}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as output:
                output.write(self.data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def from_file(cls, path: Path, *, kind: str) -> InputAsset:
        # Read once. Later changes to the user's source cannot mutate a project.
        return cls(path.read_bytes(), path.suffix.lower(), kind, str(path.resolve()))

    def to_dict(self) -> dict:
        """Lossless bytes carried by the project JSON, independent of source paths."""
        return {
            "encoding": "base64",
            "sha256": self.id,
            "suffix": self.suffix,
            "kind": self.kind,
            "source": self.source,
            "data": base64.b64encode(self.data).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, value: dict) -> InputAsset:
        expected = {"encoding", "sha256", "suffix", "kind", "source", "data"}
        if not isinstance(value, dict) or set(value) != expected or value.get("encoding") != "base64":
            raise ValueError("Malformed inline project asset descriptor.")
        if not isinstance(value["data"], str) or not isinstance(value["sha256"], str):
            raise ValueError("Inline project asset content and digest must be strings.")
        try:
            data = base64.b64decode(value["data"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Malformed base64 project asset content.") from exc
        asset = cls(data, value["suffix"], value["kind"], value["source"])
        if asset.id != value["sha256"]:
            raise ValueError("Project input integrity check failed for inline asset.")
        return asset


def mesh_asset(mesh: Mesh) -> InputAsset:
    if not isinstance(mesh, Mesh):
        raise TypeError("Owned mesh must be a Mesh.")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3 or not len(mesh.vertices):
        raise ValueError("Owned mesh requires nonempty (n, 3) coordinates.")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError("Owned mesh coordinates must be finite.")
    if np.any(mesh.faces < 0) or np.any(mesh.faces >= mesh.n_vertices):
        raise ValueError("Owned mesh faces must reference valid vertices.")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or not len(mesh.faces):
        raise ValueError("Owned mesh requires nonempty triangular faces.")
    if np.any(np.diff(np.sort(mesh.faces, axis=1), axis=1) == 0):
        raise ValueError("Owned mesh faces must reference three distinct vertices.")
    output = BytesIO()
    normals = {}
    for key, count in (("vertex_normals", mesh.n_vertices), ("face_normals", mesh.n_faces)):
        value = getattr(mesh, key)
        if value is not None:
            if value.shape != (count, 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"Owned mesh {key} must be finite and match mesh geometry.")
            normals[key] = value
    np.savez_compressed(
        output,
        schema="coilgen-mesh-v1",
        vertices_m=mesh.vertices,
        faces=mesh.faces,
        metadata_json=json.dumps(json_value(mesh.metadata, nonfinite="raise"), sort_keys=True),
        **normals,
    )
    return InputAsset(output.getvalue(), ".npz", "mesh")


def target_asset(target: TargetField) -> InputAsset:
    if not isinstance(target, TargetField):
        raise TypeError("Owned target must be a TargetField.")
    values = {
        key: value
        for key in (
            "coords",
            "b",
            "weights",
            "suppressed",
            "target_field_group_inds",
            "target_gradient_dbdxyz",
            "suppression_normals",
            "suppression_area_weights",
        )
        if (value := getattr(target, key)) is not None
    }
    count = len(target.coords)
    if count == 0:
        raise ValueError("Owned target requires at least one sample.")
    for key, value in values.items():
        expected = (count, 3) if key in {"coords", "b", "target_gradient_dbdxyz", "suppression_normals"} else (count,)
        if value.shape != expected or not np.all(np.isfinite(value)):
            raise ValueError(f"Owned target {key} must be finite and match the target point count.")
    if np.any(target.weights < 0):
        raise ValueError("Owned target weights must be nonnegative; use zero to disable a sample.")
    if target.suppression_area_weights is not None and np.any(target.suppression_area_weights < 0):
        raise ValueError("Owned suppression area weights must be nonnegative.")
    output = BytesIO()
    np.savez_compressed(
        output,
        schema="coilgen-target-v1",
        **values,
        target_diagnostics_json=json.dumps(json_value(target.target_diagnostics, nonfinite="raise"), sort_keys=True),
    )
    return InputAsset(output.getvalue(), ".npz", "target")
