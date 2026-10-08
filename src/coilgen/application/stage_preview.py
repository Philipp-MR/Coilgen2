"""Bounded, atomic display snapshots of completed scientific stage boundaries.

These files are optional live visualization, separate from final result evidence,
checkpoint reuse, project archives, acceptance and export.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
import uuid
import zipfile

import numpy as np

from ..field_components import normalize_field_component
from ..persistence import atomic_json, file_digest
from ..pipeline_stages import PipelineStage


SCHEMA = "coilgen.live-stage-preview/1"
MANIFEST_NAME = "stage_preview.json"
DEFAULT_MAX_BYTES = 256 * 1024**2
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_ARRAYS = 10000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PATH_ARRAY = re.compile(r"(?:contour_\d+_\d+|route_\d+|fabrication_\d+(?:_\d+)?)_points_m\Z")
_BODY_ARRAY = re.compile(r"swept_\d+_\d+_(?:vertices_m|faces)\Z")
_BASIC_ARRAYS = {
    "vertices_m",
    "faces",
    "target_points_m",
    "target_field_T",
    "target_suppressed",
    "target_weights",
    "stream_function_A",
    "sheet_field_T",
    "sheet_current_density_points_m",
    "sheet_current_density_A_per_m",
}
_MANIFEST_FIELDS = {
    "schema",
    "source",
    "live_stage_preview",
    "scope",
    "pipeline_stage",
    "completed_stages",
    "acceptance_claim",
    "name",
    "request_sha256",
    "project_id",
    "project_revision",
    "project_fingerprint",
    "project_model_fingerprint",
    "array_file",
    "array_sha256",
    "coordinate_units",
    "field_units",
    "stream_function_units",
}


@dataclass(frozen=True, slots=True)
class StagePreview:
    """A detached presentation snapshot; deliberately not a RunRecord."""

    directory: Path
    _metadata: dict = field(repr=False)
    _arrays: dict[str, np.ndarray] = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "directory", Path(self.directory).resolve())
        object.__setattr__(self, "_metadata", deepcopy(self._metadata))
        for array in self._arrays.values():
            array.flags.writeable = False
        object.__setattr__(self, "_arrays", MappingProxyType(dict(self._arrays)))

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    @property
    def arrays(self):
        return self._arrays


def _byte_limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Stage preview byte limit must be a positive integer.")
    return value


def _sha(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Stage preview file escapes its job directory.")
    return path


def _validate_arrays(arrays, *, max_bytes):
    if not 1 <= len(arrays) <= _MAX_ARRAYS:
        raise ValueError("Invalid stage preview array count.")
    required = {"vertices_m", "faces", "target_points_m", "target_field_T", "target_weights", "target_suppressed"}
    if not required <= arrays.keys():
        raise ValueError("Stage preview is missing its support or target arrays.")
    if sum(array.nbytes for array in arrays.values()) > max_bytes:
        raise ValueError("Stage preview arrays exceed the permitted size.")
    for key, array in arrays.items():
        if key not in _BASIC_ARRAYS and not _PATH_ARRAY.fullmatch(key) and not _BODY_ARRAY.fullmatch(key):
            raise ValueError(f"Unknown stage preview array: {key}.")
        if array.dtype.kind not in "bifu" or not np.all(np.isfinite(array)):
            raise ValueError(f"Stage preview array must contain finite real data: {key}.")
        if key.endswith(("_points_m", "_vertices_m")) or key == "vertices_m":
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"Stage preview coordinates must be Nx3: {key}.")
        if key.endswith("faces"):
            vertex_key = "vertices_m" if key == "faces" else key.removesuffix("faces") + "vertices_m"
            vertices = arrays.get(vertex_key)
            if (
                vertices is None
                or array.dtype.kind not in "iu"
                or array.ndim != 2
                or array.shape[1] != 3
                or (array.size and (array.min() < 0 or array.max() >= len(vertices)))
            ):
                raise ValueError(f"Stage preview triangular indices are invalid: {key}.")
    count = len(arrays["target_points_m"])
    if not count or not len(arrays["vertices_m"]) or arrays["target_field_T"].shape != (count, 3):
        raise ValueError("Stage preview support/target arrays have incompatible dimensions.")
    if arrays["target_weights"].shape not in {(count,), (count, 1)}:
        raise ValueError("Stage preview weights do not match target samples.")
    if arrays["target_suppressed"].shape != (count,) or arrays["target_suppressed"].dtype.kind != "b":
        raise ValueError("Stage preview suppression mask does not match target samples.")
    if "sheet_field_T" in arrays and arrays["sheet_field_T"].shape != (count, 3):
        raise ValueError("Stage preview sheet field does not match target samples.")
    density_keys = {"sheet_current_density_points_m", "sheet_current_density_A_per_m"}
    if density_keys & arrays.keys():
        if not density_keys <= arrays.keys():
            raise ValueError("Stage preview current density requires its recorded face coordinates and vectors.")
        density = arrays["sheet_current_density_A_per_m"]
        points = arrays["sheet_current_density_points_m"]
        if (
            density.dtype.kind not in "ifu"
            or points.dtype.kind not in "ifu"
            or density.shape != (len(arrays["faces"]), 3)
            or points.shape != density.shape
        ):
            raise ValueError("Stage preview current density must contain one real Nx3 vector per support face.")
    if "stream_function_A" in arrays and arrays["stream_function_A"].shape not in {
        (len(arrays["vertices_m"]),),
        (len(arrays["vertices_m"]), 1),
    }:
        raise ValueError("Stage preview stream function does not match support vertices.")
    for key in arrays:
        if (
            key.startswith("swept_")
            and key.endswith("vertices_m")
            and key.removesuffix("vertices_m") + "faces" not in arrays
        ):
            raise ValueError("Stage preview conductor vertices have no recorded triangular faces.")


def publish_stage_preview(directory, checkpoint, request, request_sha256, *, max_bytes=DEFAULT_MAX_BYTES) -> dict:
    """Publish actual recorded SI data only after the full NPZ is flushed.

    The caller treats failures as optional visualization failures. A stage's file
    is immutable once announced; an existing file is never replaced.
    """
    max_bytes = _byte_limit(max_bytes)
    if not _sha(request_sha256):
        raise ValueError("Stage preview requires the exact submitted request digest.")
    stage = PipelineStage(checkpoint.stage)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    arrays = checkpoint.scientific_arrays(max_bytes=max_bytes)
    _validate_arrays(arrays, max_bytes=max_bytes)
    relative = f"stage_previews/{stage.value}.npz"
    destination = _contained(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("A completed stage preview is already published.")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size > max_bytes:
            raise ValueError("Stage preview archive exceeds the permitted size.")
        digest = file_digest(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema": SCHEMA,
        "source": "live_stage_preview",
        "live_stage_preview": True,
        "scope": "completed_boundary_preview",
        "pipeline_stage": stage.value,
        "completed_stages": [item.value for item in PipelineStage if item.index <= stage.index],
        "acceptance_claim": False,
        "name": request.name,
        "request_sha256": request_sha256,
        "project_id": request.project_id,
        "project_revision": request.project_revision,
        "project_fingerprint": request.project_fingerprint,
        "project_model_fingerprint": request.project_model_fingerprint,
        "array_file": relative,
        "array_sha256": digest,
        "coordinate_units": "m",
        "field_units": "T",
        "stream_function_units": "A",
        "target_field_component": normalize_field_component(request.config.get("target_field_component", "Bz")),
    }
    atomic_json(_contained(root, MANIFEST_NAME), manifest)
    return manifest


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate stage preview manifest field.")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("Nonfinite stage preview manifest value.")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_stage_preview_manifest(
    directory,
    *,
    expected_request_sha256,
    expected_project_id,
    expected_project_fingerprint,
    expected_project_revision=None,
) -> dict | None:
    """Read only a small committed manifest, rejecting another submitted identity."""
    if not _sha(expected_request_sha256):
        raise ValueError("Stage preview requires a verified submitted request digest.")
    root = Path(directory).resolve()
    path = _contained(root, MANIFEST_NAME)
    try:
        with path.open("rb") as source:
            encoded = source.read(_MAX_MANIFEST_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise ValueError("Stage preview manifest exceeds the permitted size.")
    manifest = _strict_json(encoded.decode("utf-8"))
    if (
        not isinstance(manifest, dict)
        or not _MANIFEST_FIELDS <= manifest.keys()
        or manifest.keys() - _MANIFEST_FIELDS - {"target_field_component"}
    ):
        raise ValueError("Malformed stage preview manifest.")
    if (
        manifest["schema"] != SCHEMA
        or manifest["source"] != "live_stage_preview"
        or manifest["live_stage_preview"] is not True
        or manifest["acceptance_claim"] is not False
        or manifest["scope"] != "completed_boundary_preview"
        or (manifest["coordinate_units"], manifest["field_units"], manifest["stream_function_units"]) != ("m", "T", "A")
    ):
        raise ValueError("Unsupported stage preview schema or scientific scope.")
    stage = PipelineStage(manifest["pipeline_stage"])
    completed = [item.value for item in PipelineStage if item.index <= stage.index]
    if manifest["completed_stages"] != completed or manifest["array_file"] != f"stage_previews/{stage.value}.npz":
        raise ValueError("Stage preview filename or completed boundaries disagree.")
    if not _sha(manifest["array_sha256"]):
        raise ValueError("Invalid stage preview array digest.")
    expected = {
        "request_sha256": expected_request_sha256,
        "project_id": expected_project_id,
        "project_fingerprint": expected_project_fingerprint,
        "project_revision": expected_project_revision,
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("Stage preview does not match the submitted job identity.")
    for key in ("name", "project_id", "project_fingerprint", "project_model_fingerprint"):
        if not isinstance(manifest[key], str) or len(manifest[key]) > 512:
            raise ValueError("Malformed stage preview provenance.")
    revision = manifest["project_revision"]
    if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int) or revision < 1):
        raise ValueError("Malformed stage preview revision.")
    manifest["target_field_component"] = normalize_field_component(manifest.get("target_field_component", "Bz"))
    _contained(root, manifest["array_file"])
    return manifest


def load_stage_preview(
    directory,
    *,
    expected_request_sha256,
    expected_project_id,
    expected_project_fingerprint,
    expected_project_revision=None,
    expected_stage=None,
    max_bytes=DEFAULT_MAX_BYTES,
) -> StagePreview | None:
    """Load an announced immutable snapshot; never read partial final-result files."""
    max_bytes = _byte_limit(max_bytes)
    root = Path(directory).resolve()
    manifest = read_stage_preview_manifest(
        root,
        expected_request_sha256=expected_request_sha256,
        expected_project_id=expected_project_id,
        expected_project_fingerprint=expected_project_fingerprint,
        expected_project_revision=expected_project_revision,
    )
    if manifest is None:
        return None
    if expected_stage is not None and manifest["pipeline_stage"] != PipelineStage(expected_stage).value:
        raise ValueError("Stage preview boundary changed while loading.")
    path = _contained(root, manifest["array_file"])
    with path.open("rb") as source:
        encoded = source.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise ValueError("Stage preview archive exceeds the permitted size.")
    if sha256(encoded).hexdigest() != manifest["array_sha256"]:
        raise ValueError("Stage preview array integrity check failed.")
    try:
        arrays = _load_arrays(encoded, max_bytes)
    except zipfile.BadZipFile as exc:
        raise ValueError("Malformed stage preview numerical archive.") from exc
    _validate_arrays(arrays, max_bytes=max_bytes)
    stage = PipelineStage(manifest["pipeline_stage"])
    if (
        stage.index == 0
        and {"stream_function_A", "sheet_field_T", "sheet_current_density_A_per_m", "sheet_current_density_points_m"}
        & arrays.keys()
    ):
        raise ValueError("Prepared preview cannot claim solved fields.")
    if stage.index >= PipelineStage.SHEET.index and not {"stream_function_A", "sheet_field_T"} <= arrays.keys():
        raise ValueError("Solved stage preview is missing its stream function or sheet field.")
    if stage.index < PipelineStage.CONTOURS.index and any(
        key.startswith(("contour_", "route_", "fabrication_", "swept_")) for key in arrays
    ):
        raise ValueError("Stage preview contains geometry from a later boundary.")
    if stage.index < PipelineStage.ROUTED.index and any(
        key.startswith(("route_", "fabrication_", "swept_")) for key in arrays
    ):
        raise ValueError("Stage preview contains routed geometry before routing.")
    return StagePreview(root, manifest, arrays)


def _load_arrays(encoded, max_bytes):
    buffer = io.BytesIO(encoded)
    with zipfile.ZipFile(buffer) as archive:
        items = archive.infolist()
        if not 1 <= len(items) <= _MAX_ARRAYS or sum(item.file_size for item in items) > max_bytes:
            raise ValueError("Stage preview decompressed arrays exceed the permitted size.")
        names = [item.filename for item in items]
        if len(names) != len(set(names)) or any(
            not name.endswith(".npy") or "/" in name or "\\" in name for name in names
        ):
            raise ValueError("Malformed stage preview numerical archive.")
        # A forged NPY header can claim enormous shapes inside a tiny ZIP entry.
        # Inspect its declared storage before NumPy has an opportunity to allocate.
        for item in items:
            with archive.open(item) as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _order, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _order, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError("Unsupported stage preview NPY format.")
                if dtype.kind not in "bifu" or len(shape) > 2:
                    raise ValueError("Stage preview array requires a finite real numerical dtype.")
                declared_bytes = math.prod(shape) * dtype.itemsize
                if declared_bytes > max_bytes or declared_bytes + source.tell() != item.file_size:
                    raise ValueError("Stage preview array shape exceeds its recorded storage.")
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as numerical:
        arrays = {key: numerical[key] for key in numerical.files}
    return arrays
