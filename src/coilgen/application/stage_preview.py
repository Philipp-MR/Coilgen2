"""Bounded, atomic display snapshots of completed scientific stages and geometry.

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
from ..scientific_arrays import scientific_arrays


SCHEMA = "coilgen.live-stage-preview/1"
MANIFEST_NAME = "stage_preview.json"
DEFAULT_MAX_BYTES = 256 * 1024**2
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_ARRAYS = 10000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PATH_ARRAY = re.compile(r"(?:contour_\d+_\d+|route_\d+|fabrication_\d+(?:_\d+)?)_points_m\Z")
_BODY_ARRAY = re.compile(r"swept_\d+_\d+_(?:vertices_m|faces)\Z")
# Features are produced during routing. Each snapshot includes the support,
# fields, contours and any earlier geometry still present in the same solution.
_FEATURE_GEOMETRY = {
    # Required geometry prefix, prefixes belonging to later features.
    "interconnections": ("route_", ("fabrication_", "swept_")),
    "realized_centerlines": ("fabrication_", ("swept_",)),
    "sweep": ("swept_", ()),
}
# Ordered by the operation which first produces each family of plot arrays.
_EVALUATION_ARRAYS = {
    "evaluated_winding": re.compile(
        r"(?:contour|routed)_field(?:_at_stream_current)?_T|evaluated_winding_\d+_points_m"
    ),
    "independent_field": re.compile(r"validation_(?:points_m|field_T|target_T)"),
    "review_conductor": re.compile(r"review_(?:conductor|electrical)_\d+_points_m"),
    "fabricated_field": re.compile(r"fabricated_(?:points_m|field_T)"),
    "winding_validation": re.compile(r"winding_validation_(?:points_m|field_T|target_T)"),
}
_WIRE_METRICS = {
    "available",
    "reason",
    "component",
    "sample_count",
    "rms_error_T",
    "peak_error_T",
    "target_rms_T",
    "relative_rms_error",
    "best_fit_current_scale",
    "best_fit_relative_rms_error",
    "rms_convention",
    "current_A",
    "current_source",
    "current_refitted",
}
_WIRE_SCOPE = {
    "scope": "design_grid_wire_field",
    "field_model": "thin_wire_biot_savart",
    "independent_sampling": False,
    "acceptance_claim": False,
}
_STREAM_CURRENT_SOURCES = {"stream_function", "stream", "contour_step", "sf"}
_REFIT_CURRENT_SOURCES = {
    "target_fit",
    "target_projection",
    "projection",
    "least_squares",
    "least_squares_projection",
    "target_std_projection",
    "std_projection",
    "homogeneity_projection",
    "mean_free_projection",
}
_FABRICATED_CURRENT = {"current_source": "original_contour_current_no_refit"}
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


def _base_array(key):
    return key in _BASIC_ARRAYS or _PATH_ARRAY.fullmatch(key) or _BODY_ARRAY.fullmatch(key)


def _evaluation_family(key):
    return next((family for family, pattern in _EVALUATION_ARRAYS.items() if pattern.fullmatch(key)), None)


def _validate_arrays(arrays, *, max_bytes, feature=None):
    if not 1 <= len(arrays) <= _MAX_ARRAYS:
        raise ValueError("Invalid stage preview array count.")
    required = {"vertices_m", "faces", "target_points_m", "target_field_T", "target_weights", "target_suppressed"}
    if not required <= arrays.keys():
        raise ValueError("Stage preview is missing its support or target arrays.")
    if sum(array.nbytes for array in arrays.values()) > max_bytes:
        raise ValueError("Stage preview arrays exceed the permitted size.")
    for key, array in arrays.items():
        if not _base_array(key) and not (feature in _EVALUATION_ARRAYS and _evaluation_family(key)):
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
        raise ValueError("Stage preview stream function field does not match target samples.")
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
    stage = PipelineStage(checkpoint.stage)
    arrays = checkpoint.scientific_arrays(max_bytes=max_bytes)
    return _publish(directory, arrays, stage, None, request, request_sha256, max_bytes)


def publish_feature_preview(directory, state, feature, request, request_sha256, *, max_bytes=DEFAULT_MAX_BYTES) -> dict:
    """Synchronously snapshot recorded geometry without copying the solver state."""
    max_bytes = _byte_limit(max_bytes)
    if feature not in _FEATURE_GEOMETRY:
        raise ValueError(f"Unknown stage preview feature: {feature}.")
    arrays = scientific_arrays(state.parts, state.target_field, state.combined_mesh, state.sf_b_field)
    _, later = _FEATURE_GEOMETRY[feature]
    arrays = {key: value for key, value in arrays.items() if not key.startswith(later)}
    return _publish(directory, arrays, PipelineStage.CONTOURS, feature, request, request_sha256, max_bytes)


def publish_evaluation_preview(
    directory,
    arrays,
    feature,
    request,
    request_sha256,
    *,
    wire_fields=None,
    fabricated_current_source=None,
    max_bytes=DEFAULT_MAX_BYTES,
) -> dict:
    """Snapshot completed plot data from evaluation without copying full result artifacts."""
    if feature not in _EVALUATION_ARRAYS:
        raise ValueError(f"Unknown evaluation preview feature: {feature}.")
    arrays = {key: value for key, value in arrays.items() if _base_array(key) or _evaluation_family(key)}
    evidence = {}
    if wire_fields is not None:
        evidence["wire_fields"] = wire_fields
    if fabricated_current_source is not None:
        evidence["fabricated_physics"] = {"current_source": fabricated_current_source}
    return _publish(
        directory,
        arrays,
        PipelineStage.EVALUATED,
        feature,
        request,
        request_sha256,
        _byte_limit(max_bytes),
        evidence,
    )


def _validate_feature_arrays(arrays, feature):
    prefix, later = _FEATURE_GEOMETRY[feature]
    suffix = "_faces" if feature == "sweep" else "_points_m"
    minimum = 1 if feature == "sweep" else 2
    if not any(
        key.startswith(prefix) and key.endswith(suffix) and len(array) >= minimum for key, array in arrays.items()
    ):
        raise ValueError(f"Stage preview has no recorded geometry for {feature}.")
    if any(key.startswith(later) for key in arrays):
        raise ValueError("Stage preview contains geometry from a later feature.")


def _finite_number(value):
    return type(value) in (float, int) and -math.inf < value < math.inf


def _validate_wire_maps(arrays, evidence, component):
    if not isinstance(evidence, dict) or set(evidence) != {"contour", "routed"}:
        raise ValueError("Wire preview requires its recorded contour and routed field provenance.")
    design_shape = arrays["target_points_m"].shape
    active = np.count_nonzero(~arrays["target_suppressed"] & (arrays["target_weights"].reshape(-1) > 0))
    for kind, entry in evidence.items():
        if (
            not isinstance(entry, dict)
            or entry.keys() - _WIRE_METRICS - _WIRE_SCOPE.keys() - {"geometry_sources", "at_stream_current"}
            or any(type(entry.get(k)) is not type(v) or entry[k] != v for k, v in _WIRE_SCOPE.items())
        ):
            raise ValueError("Wire preview must describe design-grid thin-wire fields without an acceptance claim.")
        sources = entry.get("geometry_sources")
        if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
            raise ValueError("Invalid wire preview geometry provenance.")
        if kind == "contour" and sources != ["oriented_unconnected_contours"]:
            raise ValueError("Contour field must use oriented unconnected contours.")
        if kind == "routed":
            paths = {k for k in arrays if k.startswith("evaluated_winding_")}
            expected = set()
            if entry.get("available") is True:
                for index, source in enumerate(sources):
                    prefix = {"fabricated_centroid": "fabrication", "connected_wire": "route"}.get(source)
                    key = f"evaluated_winding_{index}_points_m"
                    if (
                        prefix is None
                        or key not in paths
                        or not np.array_equal(arrays[key], arrays.get(f"{prefix}_{index}_points_m"))
                    ):
                        raise ValueError("Evaluated wire path disagrees with its recorded geometry source.")
                    expected.add(key)
                if not expected:
                    raise ValueError("Routed field has no evaluated winding.")
            if paths != expected:
                raise ValueError("Evaluated winding paths disagree with routed field provenance.")
        source = entry.get("current_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Wire field requires its recorded current source.")
        policy = source.strip().lower().replace("-", "_")
        if policy not in _STREAM_CURRENT_SOURCES | _REFIT_CURRENT_SOURCES or entry.get("current_refitted") is not (
            policy in _REFIT_CURRENT_SOURCES
        ):
            raise ValueError("Wire current source and refit provenance disagree.")
        original = entry.get("at_stream_current", {"available": False})
        if not isinstance(original, dict) or original.keys() - _WIRE_METRICS:
            raise ValueError("Invalid original wire current provenance.")
        for suffix, metric in (("", entry), ("_at_stream_current", original)):
            name = f"{kind}_field{suffix}_T"
            if metric.get("available") is not (name in arrays) or (
                "reason" in metric and not isinstance(metric["reason"], str)
            ):
                raise ValueError("Wire field arrays disagree with recorded availability.")
            fit_keys = {"best_fit_current_scale", "best_fit_relative_rms_error"}
            if fit_keys & metric.keys() and (
                not fit_keys <= metric.keys()
                or not all(_finite_number(metric[key]) for key in fit_keys)
                or metric["best_fit_current_scale"] <= 0
                or metric["best_fit_relative_rms_error"] < 0
            ):
                raise ValueError("Invalid wire current-fit provenance.")
            if name not in arrays:
                continue
            if (
                arrays[name].shape != design_shape
                or not _finite_number(metric.get("current_A"))
                or (
                    suffix
                    and (
                        metric.get("current_source") != "original_contour_step"
                        or metric.get("current_refitted") is not False
                        or metric["current_A"] == 0
                        or not entry["available"]
                    )
                )
                or not isinstance(metric.get("component"), str)
                or normalize_field_component(metric["component"]) != component
                or metric.get("sample_count") != active
                or metric.get("rms_convention") != ("vector_magnitude" if component == "vector" else "scalar")
                or not all(
                    _finite_number(metric.get(k)) and metric[k] >= 0
                    for k in ("rms_error_T", "peak_error_T", "target_rms_T")
                )
                or (metric.get("relative_rms_error") is not None and not _finite_number(metric["relative_rms_error"]))
            ):
                raise ValueError("Invalid wire field sampling, current or residual provenance.")


def _validate_evaluation_arrays(arrays, feature, evidence, component):
    if not {"stream_function_A", "sheet_field_T"} <= arrays.keys():
        raise ValueError("Evaluation preview requires its solved stream function and design-grid field.")
    families = set()
    for key, value in arrays.items():
        family = _evaluation_family(key)
        if family is not None:
            families.add(family)
            minimum = 2 if key.startswith(("evaluated_winding_", "review_")) else 1
            if value.dtype.kind not in "fiu" or value.ndim != 2 or value.shape[1] != 3 or len(value) < minimum:
                raise ValueError(f"Invalid recorded evaluation vectors: {key}.")
    order = tuple(_EVALUATION_ARRAYS)
    if feature not in families or families.intersection(order[order.index(feature) + 1 :]):
        raise ValueError("Evaluation preview is missing the named feature or contains a later feature.")
    if "evaluated_winding" in families or "wire_fields" in evidence:
        _validate_wire_maps(arrays, evidence.get("wire_fields"), component)
    for family, prefix in (("independent_field", "validation"), ("winding_validation", "winding_validation")):
        if family in families:
            names = [f"{prefix}_{suffix}" for suffix in ("points_m", "field_T", "target_T")]
            if not all(key in arrays for key in names) or len({arrays[key].shape for key in names}) != 1:
                raise ValueError("Independent field, target and coordinates must share the recorded grid.")
    if "fabricated_field" in families or "fabricated_physics" in evidence:
        if (
            evidence.get("fabricated_physics") != _FABRICATED_CURRENT
            or not {"fabricated_field_T", "fabricated_points_m"} <= arrays.keys()
            or arrays["fabricated_field_T"].shape != arrays["fabricated_points_m"].shape
            or not np.array_equal(arrays["fabricated_points_m"], arrays["target_points_m"])
        ):
            raise ValueError("Fabricated field requires its recorded design coordinates and original current policy.")


def _publish(directory, arrays, stage, feature, request, request_sha256, max_bytes, evidence=None):
    if not _sha(request_sha256):
        raise ValueError("Stage preview requires the exact submitted request digest.")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    component = normalize_field_component(request.config.get("target_field_component", "Bz"))
    evidence = evidence or {}
    _validate_arrays(arrays, max_bytes=max_bytes, feature=feature)
    if feature in _EVALUATION_ARRAYS:
        _validate_evaluation_arrays(arrays, feature, evidence, component)
    elif feature is not None:
        _validate_feature_arrays(arrays, feature)
    relative = f"stage_previews/{feature or stage.value}.npz"
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
        "scope": "completed_feature_preview" if feature is not None else "completed_boundary_preview",
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
        "target_field_component": component,
        **evidence,
    }
    if feature is not None:
        manifest["preview_feature"] = feature
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
        or manifest.keys()
        - _MANIFEST_FIELDS
        - {"target_field_component", "preview_feature", "wire_fields", "fabricated_physics"}
    ):
        raise ValueError("Malformed stage preview manifest.")
    if (
        manifest["schema"] != SCHEMA
        or manifest["source"] != "live_stage_preview"
        or manifest["live_stage_preview"] is not True
        or manifest["acceptance_claim"] is not False
        or (manifest["coordinate_units"], manifest["field_units"], manifest["stream_function_units"]) != ("m", "T", "A")
    ):
        raise ValueError("Unsupported stage preview schema or scientific scope.")
    stage = PipelineStage(manifest["pipeline_stage"])
    feature = manifest.get("preview_feature")
    if "preview_feature" in manifest:
        if (
            not isinstance(feature, str)
            or feature not in (*_FEATURE_GEOMETRY, *_EVALUATION_ARRAYS)
            or manifest["scope"] != "completed_feature_preview"
            or stage != (PipelineStage.EVALUATED if feature in _EVALUATION_ARRAYS else PipelineStage.CONTOURS)
        ):
            raise ValueError("Invalid stage preview feature or scientific scope.")
    elif manifest["scope"] != "completed_boundary_preview":
        raise ValueError("Unsupported stage preview scientific scope.")
    if feature not in _EVALUATION_ARRAYS and {"wire_fields", "fabricated_physics"} & manifest.keys():
        raise ValueError("Wire field provenance outside an evaluation preview.")
    completed = [item.value for item in PipelineStage if item.index <= stage.index]
    if (
        manifest["completed_stages"] != completed
        or manifest["array_file"] != f"stage_previews/{feature or stage.value}.npz"
    ):
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
    stage = PipelineStage(manifest["pipeline_stage"])
    feature = manifest.get("preview_feature")
    _validate_arrays(arrays, max_bytes=max_bytes, feature=feature)
    if (
        stage.index == 0
        and {"stream_function_A", "sheet_field_T", "sheet_current_density_A_per_m", "sheet_current_density_points_m"}
        & arrays.keys()
    ):
        raise ValueError("Prepared preview cannot claim solved fields.")
    if stage.index >= PipelineStage.SHEET.index and not {"stream_function_A", "sheet_field_T"} <= arrays.keys():
        raise ValueError("Solved stage preview is missing its stream function or the corresponding magnetic field.")
    if stage.index < PipelineStage.CONTOURS.index and any(
        key.startswith(("contour_", "route_", "fabrication_", "swept_")) for key in arrays
    ):
        raise ValueError("Stage preview contains geometry from a later boundary.")
    if feature in _EVALUATION_ARRAYS:
        _validate_evaluation_arrays(arrays, feature, manifest, manifest["target_field_component"])
    elif feature is not None:
        _validate_feature_arrays(arrays, feature)
    elif stage.index < PipelineStage.ROUTED.index and any(
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
