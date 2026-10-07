"""Headless availability of scientific evidence; no calculations or acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from hashlib import sha256
import json
from zipfile import BadZipFile

import numpy as np


FIELD_KEYS = {
    "preview_target": "target_field_T",
    "target": "target_field_T",
    "sheet": "sheet_field_T",
    "validation": "validation_field_T",
    "contour_field": "contour_field_T",
    "contour_residual": "contour_field_T",
    "routed_field": "routed_field_T",
    "routed_residual": "routed_field_T",
    "fabricated": "fabricated_field_T",
}
WIRE_FIELDS = ("contour_field", "contour_residual", "routed_field", "routed_residual")
PATH_PREFIXES = {
    "contours": "contour_",
    "routes": "route_",
    "evaluated_winding": "evaluated_winding_",
    "fabrication": "fabrication_",
    "review_conductor": "review_conductor_",
    "review_electrical": "review_electrical_",
}
REQUIRED_STEPS = {
    "support": "support",
    "preview_target": "target",
    "target": "prepared",
    "stream_function": "sheet",
    "sheet": "sheet",
    "contours": "contours",
    "routes": "routed",
    "fabrication": "routed",
    "swept": "routed",
    "validation": "evaluated",
    "contour_field": "evaluated",
    "contour_residual": "evaluated",
    "routed_field": "evaluated",
    "routed_residual": "evaluated",
    "evaluated_winding": "evaluated",
    "review_conductor": "fabrication_review",
    "review_electrical": "fabrication_review",
    "fabricated": "fabrication_review",
}


@dataclass(frozen=True, slots=True)
class DatasetCapability:
    available: bool
    reason: str = ""
    required_step: str = ""
    invalid: bool = False


def _vectors(arrays, name, minimum=1):
    value = arrays.get(name)
    if value is None:
        return f"Missing {name}.", False
    try:
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[1] != 3:
            return f"Invalid {name}: expected Nx3 values.", True
        if len(value) < minimum:
            return f"Empty {name}." if not len(value) else f"Insufficient points in {name}.", False
        if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) or not np.all(np.isfinite(value)):
            return f"Invalid {name}: coordinates/values must be finite real numbers.", True
    except (TypeError, ValueError):
        return f"Invalid {name}: expected finite Nx3 values.", True
    return "", False


def _surface(arrays, vertices="vertices_m", faces="faces"):
    reason, invalid = _vectors(arrays, vertices)
    if reason:
        return reason, invalid
    if faces not in arrays:
        return f"Missing {faces}.", False
    try:
        triangles = np.asarray(arrays[faces])
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            return f"Invalid {faces}: expected Nx3 triangular indices.", True
        if not len(triangles):
            return f"Empty {faces}.", False
        if not np.issubdtype(triangles.dtype, np.integer):
            return f"Invalid {faces}: indices must be integers.", True
        if triangles.min() < 0 or triangles.max() >= len(arrays[vertices]):
            return f"Invalid {faces}: vertex index outside mesh.", True
    except (TypeError, ValueError):
        return f"Invalid {faces}: expected triangular indices.", True
    return "", False


def resolve_dataset(arrays, key, *, original=False):
    """Validate the exact geometry/map/current pair, including legitimate zeros.

    This checks display integrity only. Availability never implies independent
    field acceptance, feasible topology, or manufacturing acceptance.
    """
    required = REQUIRED_STEPS.get(key, "")
    reason, invalid = "", False
    if key == "preview_target":
        reason, invalid = _surface(arrays)
        if reason:
            return DatasetCapability(False, reason, required, invalid)
    if key in FIELD_KEYS:
        points = "validation_points_m" if key == "validation" else "target_points_m"
        if key == "fabricated" and "fabricated_points_m" in arrays:
            points = "fabricated_points_m"
        field = FIELD_KEYS[key]
        if key in WIRE_FIELDS and original:
            field = field.replace("_field_T", "_field_at_stream_current_T")
        for name in (points, field, *(("target_field_T",) if key.endswith("_residual") else ())):
            reason, invalid = _vectors(arrays, name)
            if reason:
                break
        if not reason:
            count = len(arrays[points])
            if len(arrays[field]) != count or key.endswith("_residual") and len(arrays["target_field_T"]) != count:
                reason, invalid = "Field coordinates, values and residual target have incompatible dimensions.", True
    elif key in {"support", "stream_function"}:
        reason, invalid = _surface(arrays)
        if not reason and key == "stream_function":
            values = arrays.get("stream_function_A")
            if values is None:
                reason = "Missing stream_function_A."
            else:
                try:
                    values = np.asarray(values)
                    count = len(arrays["vertices_m"])
                    if (
                        values.shape not in {(count,), (count, 1)}
                        or not np.issubdtype(values.dtype, np.number)
                        or np.iscomplexobj(values)
                        or not np.all(np.isfinite(values))
                    ):
                        reason, invalid = (
                            "Invalid stream function: expected one finite real coefficient per mesh vertex.",
                            True,
                        )
                except (TypeError, ValueError):
                    reason, invalid = "Invalid stream function coefficients.", True
    elif key in PATH_PREFIXES:
        names = sorted(name for name in arrays if name.startswith(PATH_PREFIXES[key]) and name.endswith("_points_m"))
        if not names:
            reason = "No saved paths for this dataset."
        for name in names:
            reason, invalid = _vectors(arrays, name, minimum=2)
            if reason:
                break
    elif key == "swept":
        names = sorted(name for name in arrays if name.startswith("swept_") and name.endswith("_vertices_m"))
        if not names:
            reason = "No saved conductor solids."
        for name in names:
            reason, invalid = _surface(arrays, name, name.removesuffix("_vertices_m") + "_faces")
            if reason:
                break
    else:
        reason = "Unknown dataset."
    return DatasetCapability(not reason, reason, required if reason else "", invalid)


def points_available(arrays, name="target_points_m"):
    return not _vectors(arrays, name)[0]


def dataset_capabilities(arrays, *, original=False):
    """Build a catalog when a source changes, rather than rechecking during camera edits."""
    return {key: resolve_dataset(arrays, key, original=original) for key in REQUIRED_STEPS}


class EvidenceIntegrityError(ValueError):
    """A selected saved source changed or failed its integrity checks."""


def source_directory(source) -> Path:
    """Canonical identity shared by every alias of a saved run directory."""
    return Path(getattr(source, "directory", source)).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """A verification outcome, separate from immutable cached arrays.

    Success records a check at one point in time; it cannot promise that files
    remain unchanged. Legacy results without a metadata checksum deliberately
    retain ``metadata_checked=False``.
    """

    directory: Path
    status: str
    error: str = ""
    metadata_checked: bool = False
    request_sha256: str = ""

    def __post_init__(self):
        object.__setattr__(self, "directory", source_directory(self.directory))
        if self.status not in {"unverified", "verified", "failed"}:
            raise ValueError("Unknown evidence verification status.")
        if self.status == "failed" and not self.error:
            raise ValueError("Failed evidence verification requires a reason.")
        if self.status != "failed" and self.error:
            raise ValueError("Only failed evidence verification has an error.")

    @property
    def verified(self) -> bool:
        return self.status == "verified"


class SourceHealth:
    """Sticky source failures across aliases, cleared by explicit verification.

    Callers resolve application-specific IDs to paths. Displaying a retained
    snapshot never calls ``mark_verified`` and cannot clear a source failure.
    """

    def __init__(self):
        self._reports: dict[Path, VerificationReport] = {}

    def report(self, source) -> VerificationReport:
        directory = source_directory(source)
        return self._reports.get(directory, VerificationReport(directory, "unverified"))

    def error(self, source) -> str:
        return self.report(source).error

    def mark_failed(self, source, error) -> VerificationReport:
        report = VerificationReport(source_directory(source), "failed", str(error))
        self._reports[report.directory] = report
        return report

    def mark_verified(self, source, report: VerificationReport | None = None) -> VerificationReport:
        """Record an explicit successful load/check; never infer from a view."""
        directory = source_directory(source)
        if report is None:
            metadata = getattr(source, "metadata", {})
            report = VerificationReport(
                directory,
                "verified",
                metadata_checked="metadata_sha256" in metadata,
                request_sha256=metadata.get("request_sha256", ""),
            )
        if not report.verified or report.directory != directory:
            raise ValueError("Source health can only be cleared by successful verification of the same source.")
        self._reports[directory] = report
        return report


def verified_run_metadata(directory):
    """Verify saved metadata, artifacts and relocated inputs without loading NPZ.

    Use the established storage contract, including compatibility with legacy
    results without metadata checksums. The returned report makes that reduced
    integrity strength explicit; it says nothing about engineering acceptance.
    """
    from ..research.storage import verified_result, verify_input_files

    directory = source_directory(directory)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Malformed run result.")
    digest = result.get("request_sha256")
    checked = verified_result(directory, digest) if isinstance(digest, str) else None
    if checked is None:
        raise ValueError("Run is incomplete or its artifact integrity check failed.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Malformed run manifest.")
    if manifest.get("request_sha256") != digest:
        raise ValueError("Run request and manifest identity disagree.")
    verify_input_files(manifest, relocated_root=directory)
    report = VerificationReport(
        directory, "verified", metadata_checked="metadata_sha256" in checked, request_sha256=digest
    )
    return checked, manifest, report


def _assert_same_metadata(record, metadata):
    from ..research.storage import result_metadata_digest

    old = {key: value for key, value in record.metadata.items() if key != "job"}
    current = {key: value for key, value in metadata.items() if key != "job"}
    if result_metadata_digest(old) != result_metadata_digest(current):
        raise EvidenceIntegrityError("Saved run metadata changed after it was selected. Reload the run.")


def _checked_snapshot_metadata(record):
    """Check a selected identity without decompressing unrelated scientific maps."""
    from .service import RunRecord

    if not isinstance(record, RunRecord):
        raise TypeError("Saved evidence operations require a RunRecord.")
    try:
        metadata, manifest, report = verified_run_metadata(record.directory)
        _assert_same_metadata(record, metadata)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, BadZipFile) as exc:
        if isinstance(exc, EvidenceIntegrityError):
            raise
        raise EvidenceIntegrityError(str(exc) or type(exc).__name__) from exc
    return metadata, manifest, report


def verify_record(record):
    """Reverify and load the complete unchanged selected scientific snapshot."""
    from .service import DesignService, RunRecord

    if not isinstance(record, RunRecord):
        raise TypeError("Saved evidence operations require a RunRecord.")
    try:
        verified = DesignService.load_run(record.directory)
        _assert_same_metadata(record, verified.metadata)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, BadZipFile) as exc:
        if isinstance(exc, EvidenceIntegrityError):
            raise
        raise EvidenceIntegrityError(str(exc) or type(exc).__name__) from exc
    return verified


def verify_run_source(source) -> VerificationReport:
    """Check existing storage contracts and report their verification strength."""
    from .service import RunRecord

    directory = source_directory(source)
    try:
        _metadata, _manifest, report = (
            _checked_snapshot_metadata(source) if isinstance(source, RunRecord) else verified_run_metadata(directory)
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        return VerificationReport(directory, "failed", str(exc) or type(exc).__name__)
    return report


def _artifact_bytes(record, name) -> bytes:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Artifact names must be portable relative paths.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or PureWindowsPath(name).drive:
        raise ValueError("Artifact paths cannot leave the saved run.")
    items = [item for item in record.metadata.get("artifacts", ()) if item.get("name") == name]
    if not items:
        raise ValueError(f"No verified saved artifact is recorded: {name}")
    if len(items) != 1 or not items[0].get("sha256"):
        raise EvidenceIntegrityError(f"Saved artifact has missing or ambiguous integrity information: {name}")
    path = (record.directory / name).resolve()
    if not path.is_relative_to(record.directory):
        raise EvidenceIntegrityError("Saved artifact path leaves its run directory.")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EvidenceIntegrityError(f"Saved artifact is no longer readable: {name}: {exc}") from exc
    if sha256(data).hexdigest() != items[0]["sha256"]:
        raise EvidenceIntegrityError(f"Saved artifact integrity check failed: {name}")
    return data


def read_verified_artifact(record, name: str) -> bytes:
    """Read hash-checked bytes from the unchanged selected scientific snapshot."""
    _checked_snapshot_metadata(record)
    return _artifact_bytes(record, name)


def load_saved_request(record) -> dict:
    """Read verified original settings; never infer from currently applied inputs."""
    from ..research.requests import ResearchRequest

    metadata, _manifest, _report = _checked_snapshot_metadata(record)
    if not any(item.get("name") == "request.json" for item in metadata.get("artifacts", ())):
        raise ValueError("This run has no verified original request.json; its saved settings cannot be inferred.")
    raw = _artifact_bytes(record, "request.json")
    try:
        request = ResearchRequest.from_dict(json.loads(raw))
    except (ValueError, TypeError, KeyError) as exc:
        raise EvidenceIntegrityError("The verified original request is invalid: " + str(exc)) from exc
    if request.project_id != metadata.get("project_id") or request.project_revision != metadata.get("project_revision"):
        raise EvidenceIntegrityError("The original request and result have different project/revision identities.")
    return request.to_dict()
