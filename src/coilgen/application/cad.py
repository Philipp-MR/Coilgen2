"""Export verified saved conductor solids without repeating physical design.

STL is unitless: the package explicitly records its coordinate unit and only
scales saved triangle coordinates. STEP is copied byte-for-byte with its
declared length unit. A package manifest is the final atomic commit marker;
files without that manifest are not a completed package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import struct
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np

from ..attribution import software_attribution
from ..geometry import _read_ascii_stl, _read_binary_stl, read_stl
from ..manufacturing import save_mesh_as_stl
from ..models import Mesh
from ..part_roles import body_display_name, normalize_part_role
from ..persistence import atomic_json, file_digest
from ..persistence import publish_exclusive as _publish_exclusive
from ..progress import ProgressReporter, progress_scope
from .service import RunRecord
from .diagnostics import readable_failure_reasons
from .evidence import verify_record


_MANIFEST_NAME = "geometry_manifest.json"
_UNIT_SCALE = {"m": 1.0, "mm": 1000.0}
_STEP_PREFIX = {
    "$": ("m", 1.0),
    ".MILLI.": ("mm", 1e-3),
    ".CENTI.": ("cm", 1e-2),
    ".MICRO.": ("um", 1e-6),
    ".KILO.": ("km", 1e3),
}
_UNITS_PER_M = {unit: 1 / scale for unit, scale in _STEP_PREFIX.values()}
_STEP_UNIT = re.compile(r"SI_UNIT\s*\(\s*(\$|\.[A-Z]+\.)\s*,\s*\.METRE\.\s*\)", re.IGNORECASE)
_STEP_POINT = re.compile(r"CARTESIAN_POINT\s*\(\s*'(?:[^']|'')*'\s*,\s*\(([^()]*)\)\s*\)", re.IGNORECASE)
_SOLID_STEP = re.compile(r"\b(?:FACETED_BREP|MANIFOLD_SOLID_BREP)\s*\(", re.IGNORECASE)
# Mounted conductor centerline (the swept profile's reference path).
_PATH_SUFFIXES = ("_fabrication_centroid.csv", "_fabrication_centroid.step")
PRINT_FORMER_SCOPE = "print_former_geometry"
PRINT_FORMER_EXPORT_HINT = "Enable Strict fabrication review and compute the full design to save a former STL."


def is_draft_artifact(name: str) -> bool:
    """Solids written although strict fabrication checks failed."""
    return "_DRAFT_" in PurePosixPath(name).name


GEOMETRY_FORMAT_LABELS = {
    "stl": "STL solid",
    "step": "STEP solid",
    "path_csv": "3D path · CSV",
    "path_step": "3D path · STEP",
}


def geometry_option_label(option, *, with_role: bool = True) -> str:
    """One label for an export choice, shared by every client.

    ``with_role=False`` is for lists already grouped under a role heading.
    """
    kind = GEOMETRY_FORMAT_LABELS.get(option.format, option.format.upper())
    former = "Print former · " if option.scope == PRINT_FORMER_SCOPE else ""
    body = body_display_name(option.body_id, getattr(option, "role", None) if with_role else None)
    return ("DRAFT · " if is_draft_artifact(option.name) else "") + former + f"{body} · {kind}"


# Export presets shared by every client: (key, button label, formats selected).
EXPORT_PRESETS = (
    ("step", "CAD · STEP solids", ("step",)),
    ("stl", "3D print · STL solids", ("stl",)),
    ("csv", "Centerlines · CSV", ("path_csv",)),
)
_EXPORT_GROUPS = ("Primary", "Active shield", "Other", "Conductors", "Print former")


def export_group(option) -> str:
    """Heading an export file is listed under: its body role, or "Print former"."""
    from ..part_roles import part_role_label

    if option.scope == PRINT_FORMER_SCOPE:
        return "Print former"
    return part_role_label(getattr(option, "role", None)) or "Conductors"


def grouped_export_options(options) -> tuple:
    """Options ordered by group (primary, shield, other, unassigned, former), file order kept within each."""
    return tuple(sorted(options, key=lambda option: _EXPORT_GROUPS.index(export_group(option))))


def export_option_groups(options) -> list[dict]:
    """``[{"title": "Primary", "names": [...]}, ...]`` in display order, empty groups left out."""
    groups = {}
    for option in grouped_export_options(options):
        groups.setdefault(export_group(option), []).append(option.name)
    return [{"title": title, "names": names} for title, names in groups.items()]


def export_preset_selection(options, preset) -> tuple[str, ...]:
    """Every file of the preset's formats (drafts included; exporting them still needs the DRAFT tick)."""
    formats = next((formats for key, _label, formats in EXPORT_PRESETS if key == preset), ())
    return tuple(option.name for option in grouped_export_options(options) if option.format in formats)


def export_draft_banner(options) -> str:
    """One line for the whole list when any file is a DRAFT."""
    options = tuple(options)
    drafts = sum(is_draft_artifact(option.name) for option in options)
    if not drafts:
        return ""
    return (
        f"DRAFT: {drafts} of {len(options)} files failed recorded checks and are not construction-released. "
        "They are marked DRAFT below."
    )


def print_former_status(metadata) -> str:
    """Describe the saved former geometry checks independently of coil acceptance."""
    reviewed = (metadata.get("fabrication") or {}).get("support_bodies") or ()
    pipeline = (metadata.get("pipeline_sweeps") or {}).get("grooved_supports") or ()
    bodies = reviewed or pipeline
    if not bodies:
        enabled = (metadata.get("config") or {}).get("support_body_mode", "off") != "off"
        return "Print former · no geometry saved" if enabled else ""
    checks = [body.get("support_body_ok", body.get("manufacturing_ok")) for body in bodies]
    status = (
        "failed"
        if any(check is False for check in checks)
        else "passed"
        if all(check is True for check in checks)
        else "unverified"
    )
    return f"Print former · geometry checks {status}"


def _saved_print_formers(metadata):
    names = {
        name
        for body in (metadata.get("fabrication") or {}).get("support_bodies") or ()
        if (body_id := body.get("body_id"))
        for name in (f"{body_id}_fabrication.stl", f"{body_id}_DRAFT_fabrication.stl")
    }
    for artifact in metadata.get("artifacts") or ():
        if isinstance(name := artifact.get("name"), str) and PurePosixPath(name).name in names:
            yield artifact


def has_saved_print_former(metadata) -> bool:
    """Cheap presence test. The export catalog still verifies the selected artifact."""
    return next(_saved_print_formers(metadata), None) is not None


def default_export_selection(options) -> tuple[str, ...]:
    """Released-candidate STL solids; drafts and paths are opt-in."""
    return tuple(option.name for option in options if option.format == "stl" and not is_draft_artifact(option.name))


def draft_export_notice(failure_reasons=()) -> str:
    """Wording that must accompany any DRAFT export, in every client."""
    reasons = readable_failure_reasons(dict.fromkeys(str(reason) for reason in failure_reasons))
    text = (
        "DRAFT geometry: recorded fabrication checks failed, so these files are NOT construction-released. "
        "Use them for inspection and prototyping only, and review clearance, bend radius and penetration "
        "before printing or winding."
    )
    if reasons:
        shown = reasons[:6]
        text += " Failed checks: " + "; ".join(shown) + (f"; … (+{len(reasons) - 6} more)" if len(reasons) > 6 else "")
    return text


@dataclass(frozen=True, slots=True)
class GeometryOption:
    name: str
    body_id: str
    format: str
    scope: str
    check_status: str
    source_unit: str
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    inspection_only: bool
    source_sha256: str
    role: str | None = None


@dataclass(frozen=True, slots=True)
class GeometryPackage:
    directory: Path
    manifest_path: Path
    paths: tuple[Path, ...]
    _manifest_json: str = field(repr=False)

    @property
    def manifest(self) -> dict:
        return json.loads(self._manifest_json)


def _status(value) -> str:
    return "passed" if value is True else "failed" if value is False else "unverified"


def _verified_record(record: RunRecord) -> RunRecord:
    if not isinstance(record, RunRecord):
        raise TypeError("CAD operations require a verified RunRecord.")
    return verify_record(record, array_names=())


def _artifact_path(record: RunRecord, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Artifact names must be portable relative paths.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or PureWindowsPath(name).drive:
        raise ValueError("Artifact paths cannot leave the saved run.")
    path = (record.directory / name).resolve()
    if not path.is_relative_to(record.directory) or not path.is_file():
        raise ValueError(f"Saved solid artifact is unavailable: {name}")
    return path


def _artifact_bytes(record: RunRecord, name: str, digest: str) -> bytes:
    data = _artifact_path(record, name).read_bytes()
    if not isinstance(digest, str) or sha256(data).hexdigest() != digest:
        raise ValueError(f"Saved solid artifact integrity check failed: {name}")
    return data


def _bounds(vertices: np.ndarray):
    values = np.asarray(vertices, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Saved solid coordinates must contain finite XYZ vertices.")
    return tuple(tuple(float(value) for value in row) for row in (values.min(axis=0), values.max(axis=0)))


def _stl_geometry(data: bytes) -> Mesh:
    """Parse the verified bytes, avoiding a second mutable source-file read."""
    if len(data) < 84:
        raise ValueError("Saved STL is too short.")
    face_count = struct.unpack("<I", data[80:84])[0]
    return _read_binary_stl(data, face_count) if len(data) == 84 + face_count * 50 else _read_ascii_stl(data)


def saved_print_former_arrays(record: RunRecord) -> dict:
    """Read the exact saved STL triangles for plotting, verifying their bytes."""
    arrays = {}
    for index, artifact in enumerate(_saved_print_formers(record.metadata_view)):
        mesh = _stl_geometry(_artifact_bytes(record, artifact["name"], artifact.get("sha256")))
        prefix = f"review_grooved_support_{index}"
        # STL repeats vertices for every triangle. Exact indexing keeps the
        # browser payload smaller without changing any saved triangle.
        vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
        arrays[f"{prefix}_vertices_m"] = vertices
        arrays[f"{prefix}_faces"] = inverse.reshape(-1)[mesh.faces]
    return arrays


def _step_geometry(data: bytes, *, solid: bool = True):
    text = data.decode("ascii")
    if solid and (not _SOLID_STEP.search(text) or not re.search(r"\bCLOSED_SHELL\s*\(", text, re.IGNORECASE)):
        raise ValueError("STEP artifact is a curve or surface, not a saved faceted solid.")
    if re.search(r"\bCONVERSION_BASED_UNIT\s*\(", text, re.IGNORECASE):
        raise ValueError("Conversion-based STEP units require an explicit CAD unit parser; they are not guessed.")
    units = {match.upper() for match in _STEP_UNIT.findall(text)}
    if len(units) != 1 or not units.issubset(_STEP_PREFIX):
        raise ValueError("STEP length units must be explicit and unambiguous SI metre units.")
    unit, scale_to_m = _STEP_PREFIX[next(iter(units))]
    points = []
    if not solid:
        # Curve bounds come from the curve itself, not placement origins.
        placements = set(re.findall(r"AXIS2_PLACEMENT_3D\s*\(\s*'(?:[^']|'')*'\s*,\s*#(\d+)", text, re.IGNORECASE))
        entities = re.findall(
            r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*'(?:[^']|'')*'\s*,\s*\(([^()]*)\)\s*\)", text, re.IGNORECASE
        )
        values_list = [values for ident, values in entities if ident not in placements]
    else:
        values_list = _STEP_POINT.findall(text)
    for values in values_list:
        coordinates = values.split(",")
        if len(coordinates) != 3:
            raise ValueError("Saved solid STEP points must be three-dimensional.")
        points.append([float(value.strip().replace("D", "E").replace("d", "e")) for value in coordinates])
    bounds = _bounds(np.asarray(points) * scale_to_m)
    return unit, scale_to_m, bounds


def _body_sources(metadata: dict) -> dict[str, tuple[str, dict]]:
    bodies = {}
    fabrication = metadata.get("fabrication") or {}
    for body in fabrication.get("paths", []):
        if isinstance(body, dict) and isinstance(body.get("body_id"), str):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", body["body_id"]):
                raise ValueError("Saved conductor body identifiers must be portable names.")
            bodies[body["body_id"]] = ("fabrication_review_geometry", body)
    # Grooved support formers are reviewed solids of their own, exported beside the conductors.
    for body in fabrication.get("support_bodies", []):
        if isinstance(body, dict) and isinstance(body.get("body_id"), str):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", body["body_id"]):
                raise ValueError("Saved support body identifiers must be portable names.")
            bodies[body["body_id"]] = (PRINT_FORMER_SCOPE, body)
    for body in (metadata.get("pipeline_sweeps") or {}).get("bodies", []):
        if isinstance(body, dict) and isinstance(body.get("body_id"), str):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", body["body_id"]):
                raise ValueError("Saved conductor body identifiers must be portable names.")
            bodies.setdefault(body["body_id"], ("pipeline_sweep_geometry", body))
    return bodies


def _geometry_catalog(record: RunRecord, kinds=("solid",)) -> tuple[GeometryOption, ...]:
    metadata = record.metadata
    bodies = _body_sources(metadata)
    report = metadata.get("fabrication") or {}
    summary = metadata.get("summary") or {}
    options = []
    seen = set()
    artifacts = metadata.get("artifacts", [])
    task = ProgressReporter(label="CAD catalog").task("saved artifact inspection", len(artifacts))
    for artifact_ind, artifact in enumerate(artifacts):
        task.update(artifact_ind, message="checking saved solid definitions")
        name = artifact.get("name")
        if not isinstance(name, str):
            raise ValueError("Malformed saved artifact name.")
        suffix = PurePosixPath(name).suffix.lower()
        filename = PurePosixPath(name).name
        # The connected conductor centerline is exportable as a 3D path.
        is_path = filename.lower().endswith(_PATH_SUFFIXES)
        if suffix not in {".stl", ".step", ".stp"} and not is_path:
            continue
        if ("path" if is_path else "solid") not in kinds:
            continue
        body_id = artifact.get("body_id")
        if body_id not in bodies:
            matches = [
                id_
                for id_ in bodies
                if filename
                in {
                    f"{id_}_fabrication.stl",
                    f"{id_}_fabrication_faceted.step",
                    f"{id_}_fabrication_faceted.stp",
                    f"{id_}_DRAFT_fabrication.stl",
                    f"{id_}_DRAFT_fabrication_faceted.step",
                    f"{id_}_fabrication_centroid.csv",
                    f"{id_}_fabrication_centroid.step",
                    f"{id_}.stl",
                    f"{id_}_faceted.step",
                    f"{id_}_faceted.stp",
                }
            ]
            if len(matches) != 1:
                # Support surfaces and centerline STEP files are deliberately
                # absent; filenames alone cannot invent a conductor body.
                continue
            body_id = matches[0]
        if not is_path and ("centroid" in filename.lower() or "centerline" in filename.lower()):
            continue
        if name in seen:
            raise ValueError(f"Duplicate saved solid artifact: {name}")
        seen.add(name)
        data = _artifact_bytes(record, name, artifact.get("sha256"))
        scope, body = bodies[body_id]
        diagnostics = body.get("diagnostics") or {}
        checks = body.get("manufacturing_ok", diagnostics.get("manufacturing_ok"))
        if body.get("mesh_solid_ok", diagnostics.get("mesh_solid_ok")) is False:
            checks = False
        check_status = _status(checks)
        if is_path and suffix == ".csv":
            points = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, ndmin=2)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
                raise ValueError(f"Saved conductor path is not a finite 3D polyline: {name}")
            unit, bounds = "m", _bounds(points)
            format_ = "path_csv"
        elif is_path:
            unit, _, bounds = _step_geometry(data, solid=False)
            format_ = "path_step"
        elif suffix == ".stl":
            mesh = _stl_geometry(data)
            if not len(mesh.faces):
                raise ValueError(f"Saved STL has no triangles: {name}")
            unit, bounds = "m", _bounds(mesh.vertices)
            format_ = "stl"
        else:
            text = data.decode("ascii")
            if not _SOLID_STEP.search(text) or not re.search(r"\bCLOSED_SHELL\s*\(", text, re.IGNORECASE):
                continue  # A mislabeled centerline still cannot become a solid.
            unit, _, bounds = _step_geometry(data)
            format_ = "step"
        inspection_only = (
            scope == PRINT_FORMER_SCOPE
            or check_status != "passed"
            or report.get("ready") is not True
            or summary.get("sheet_accepted") is False
            or summary.get("accepted") is not True
        )
        options.append(
            GeometryOption(
                name,
                body_id,
                format_,
                scope,
                check_status,
                unit,
                bounds,
                inspection_only,
                artifact["sha256"],
                normalize_part_role(body.get("role")),
            )
        )
    task.done()
    return tuple(sorted(options, key=lambda option: (option.body_id, option.format, option.name)))


def geometry_catalog(record: RunRecord) -> tuple[GeometryOption, ...]:
    """List identified saved conductor solids even when field acceptance failed.

    This verifies integrity and records existing check scope. It performs no
    geometry conditioning, electromagnetic calculation or acceptance promotion.
    """
    with progress_scope(0, 40, "Verifying saved geometry evidence", label="CAD catalog"):
        record = _verified_record(record)
    with progress_scope(40, 100, "Reading saved geometry definitions", label="CAD catalog"):
        return _geometry_catalog(record)


def path_catalog(record: RunRecord) -> tuple[GeometryOption, ...]:
    """List the saved mounted conductor centerlines (3D paths) per body.

    These are the sweep reference paths written by fabrication review, as CSV
    (metres) and STEP polylines. They are available whether or not checks
    passed; ``inspection_only`` carries the same review status as the solids.
    """
    with progress_scope(0, 40, "Verifying saved geometry evidence", label="CAD catalog"):
        record = _verified_record(record)
    with progress_scope(40, 100, "Reading saved conductor paths", label="CAD catalog"):
        return _geometry_catalog(record, kinds=("path",))


def export_catalog(record: RunRecord) -> tuple[GeometryOption, ...]:
    """Every exportable choice: solids (released or DRAFT) and 3D paths."""
    with progress_scope(0, 40, "Verifying saved geometry evidence", label="CAD catalog"):
        record = _verified_record(record)
    with progress_scope(40, 100, "Reading saved geometry definitions", label="CAD catalog"):
        return _geometry_catalog(record, kinds=("solid", "path"))


def _commit_package(stage: Path, destination: Path) -> None:
    """Reserve a new directory, publish without overwrite, commit manifest last."""
    destination.mkdir(exist_ok=False)
    published = []
    directories = [destination]
    try:
        files = []
        for source in sorted(stage.rglob("*")):
            if source.is_symlink():
                raise ValueError("Package companions cannot contain symbolic links.")
            relative = source.relative_to(stage)
            target = destination / relative
            if source.is_dir():
                target.mkdir(exist_ok=False)
                directories.append(target)
            elif source.is_file():
                files.append((source, target))
        files.sort(key=lambda pair: pair[1].relative_to(destination) == Path(_MANIFEST_NAME))
        task = ProgressReporter(label="CAD export").task("published package files", len(files))
        for source, target in files:
            _publish_exclusive(source, target)
            published.append(target)
            task.advance()
        task.done()
    except BaseException:
        for path in reversed(published):
            if path.resolve().is_relative_to(destination.resolve()):
                path.unlink(missing_ok=True)
        for path in reversed(directories):
            try:
                path.rmdir()
            except OSError:
                pass  # Never delete an unrelated concurrent writer's files.
        raise


def export_geometry(
    record: RunRecord,
    destination: str | Path,
    selected_names: Iterable[str] | None = None,
    unit: str = "mm",
    *,
    project=None,
    allow_draft: bool = False,
) -> GeometryPackage:
    """Package selected saved solids and optional portable project companion.

    ``destination`` must be a new directory outside the saved run. STL's
    requested unit is applied to saved coordinates; STEP retains its own unit
    and the manifest records both requested and actual units. Inspection exports
    are allowed, with their failures and review scope retained prominently.
    DRAFT solids (written although strict fabrication checks failed) require
    ``allow_draft=True``, i.e. an explicit acknowledgement by the client.
    """
    if unit not in _UNIT_SCALE:
        raise ValueError("STL export unit must be 'm' or 'mm'.")
    if not allow_draft and any(isinstance(name, str) and is_draft_artifact(name) for name in selected_names or ()):
        failures = (record.metadata.get("fabrication") or {}).get("failure_reasons", ())
        raise ValueError(draft_export_notice(failures) + " Confirm the DRAFT export explicitly to continue.")
    with progress_scope(0, 10, "Verifying saved geometry evidence", label="CAD export"):
        record = _verified_record(record)
    with progress_scope(10, 25, "Reading saved conductor solids", label="CAD export"):
        catalog = {option.name: option for option in _geometry_catalog(record, kinds=("solid", "path"))}
    # Without an explicit choice, export the solids that are not drafts.
    names = (
        [name for name, option in catalog.items() if option.format in {"stl", "step"} and not is_draft_artifact(name)]
        if selected_names is None
        else list(selected_names)
    )
    if not names or any(not isinstance(name, str) or name not in catalog for name in names):
        raise ValueError("Select at least one available saved solid STL or faceted STEP artifact.")
    if len(set(names)) != len(names):
        raise ValueError("A saved solid may be selected only once.")
    raw_destination = Path(destination).expanduser()
    if raw_destination.is_symlink():
        raise FileExistsError("An export package cannot replace a symbolic link.")
    destination = raw_destination.resolve()
    if destination.exists() or destination.is_relative_to(record.directory):
        raise FileExistsError("Choose a new package directory outside the immutable saved run.")
    filenames = [PurePosixPath(name).name for name in names]
    if len({name.casefold() for name in filenames}) != len(filenames):
        raise ValueError("Selected solids have colliding export filenames.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = record.metadata
    exported = []
    companion = None
    # Windows indexers can briefly hold the staging directory after publication.
    # Its cleanup must not turn a committed, verified package into a failed export.
    with TemporaryDirectory(prefix=".coilgen-cad-", dir=destination.parent, ignore_cleanup_errors=True) as temporary:
        stage = Path(temporary)
        for body_ind, (name, filename) in enumerate(zip(names, filenames)):
            with progress_scope(
                25 + 55 * body_ind / len(names),
                25 + 55 * (body_ind + 1) / len(names),
                f"Writing saved solid {body_ind + 1}/{len(names)}: {filename}",
                label="CAD export",
            ):
                option = catalog[name]
                data = _artifact_bytes(record, name, option.source_sha256)
                target = stage / filename
                if option.format == "stl" and unit != "m":
                    # Serialize the saved triangles after a pure unit transform;
                    # no resampling, sweep regeneration or current fitting occurs.
                    source = _stl_geometry(data)
                    save_mesh_as_stl(
                        Mesh(vertices=source.vertices * _UNIT_SCALE[unit], faces=source.faces),
                        target,
                        solid_name=option.body_id,
                    )
                    actual_unit = unit
                    bounds = _bounds(read_stl(target).vertices)
                elif option.format == "path_csv":
                    points = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, ndmin=2) * _UNIT_SCALE[unit]
                    np.savetxt(
                        target,
                        points,
                        delimiter=",",
                        header=",".join(f"{axis}_{unit}" for axis in "xyz"),
                        comments="",
                        fmt="%.17g",
                    )
                    actual_unit = unit
                    bounds = _bounds(points)
                else:
                    target.write_bytes(data)
                    actual_unit = option.source_unit
                    scale = _UNITS_PER_M[actual_unit]
                    bounds = tuple(tuple(value * scale for value in row) for row in option.bounds_m)
                # Windows' _commit requires a writable descriptor.
                with target.open("r+b") as saved:
                    os.fsync(saved.fileno())
                exported.append(
                    {
                        "name": filename,
                        "body_id": option.body_id,
                        "format": option.format,
                        "scope": option.scope,
                        "geometry_checks": option.check_status,
                        "inspection_only": option.inspection_only,
                        "draft": is_draft_artifact(name),
                        "source_artifact": name,
                        "source_sha256": option.source_sha256,
                        "export_sha256": file_digest(target),
                        "requested_unit": unit,
                        "actual_unit": actual_unit,
                        "source_unit": option.source_unit,
                        "original_bounds_m": option.bounds_m,
                        "exported_bounds": bounds,
                        "unit_conversion_only": option.format in {"stl", "path_csv"} and unit != "m",
                        "step_unit_policy": (
                            "preserved_declared_unit" if option.format in {"step", "path_step"} else None
                        ),
                    }
                )
        with progress_scope(80, 88, "Packaging project companion", label="CAD export"):
            if project is not None:
                project_path = stage / "project.coilgen.json"
                saved = Path(project.save(project_path)).resolve()
                if saved != project_path.resolve() or not saved.is_file():
                    raise ValueError("Project companion must save to the requested package location.")
                if any(path.is_symlink() for path in stage.rglob("*")):
                    raise ValueError("Project companions cannot contain symbolic links.")
                artifacts = [
                    {"name": path.relative_to(stage).as_posix(), "sha256": file_digest(path)}
                    for path in sorted(stage.rglob("*"))
                    if path.is_file() and path.name not in filenames
                ]
                companion = {
                    "name": project_path.name,
                    "sha256": file_digest(project_path),
                    "artifacts": artifacts,
                    "project_id": getattr(project, "id", None),
                    "revision": getattr(project, "revision", None),
                    "fingerprint": getattr(project, "fingerprint", None),
                    "model_fingerprint": getattr(project, "model_fingerprint", None),
                }
        manifest = {
            "schema": "coilgen-geometry-package-v1",
            "software": software_attribution(),
            "run_request_sha256": metadata.get("request_sha256"),
            "source_metadata_sha256": metadata.get("metadata_sha256"),
            "project_id": metadata.get("project_id"),
            "project_revision": metadata.get("project_revision"),
            "project_fingerprint": metadata.get("project_fingerprint"),
            "project_model_fingerprint": metadata.get("project_model_fingerprint"),
            "requested_stl_unit": unit,
            "geometry": exported,
            "project_companion": companion,
            "sheet_accepted": (metadata.get("summary") or {}).get("sheet_accepted"),
            "pipeline_sweep_checks_passed": (metadata.get("pipeline_sweeps") or {}).get("checks_passed"),
            "fabrication_review_ready": (metadata.get("fabrication") or {}).get("ready"),
            "engineering_accepted": (metadata.get("summary") or {}).get("accepted"),
            "fabrication_failure_reasons": (metadata.get("fabrication") or {}).get("failure_reasons", []),
            "construction_released": False,
            "export_scope": "inspection_export"
            if any(item["inspection_only"] for item in exported)
            else "saved_geometry_export",
            "notes": [
                "STL has no intrinsic units; import with each file's actual_unit.",
                "STEP retains its declared units and saved planar/faceted solid representation.",
                "Geometry export does not release a design for construction or alter scientific checks.",
            ],
        }
        with progress_scope(88, 90, "Writing geometry package manifest", label="CAD export"):
            atomic_json(stage / _MANIFEST_NAME, manifest)
        # Recheck after companion bundling, before publishing anything. A
        # change to a scientific input or selected artifact invalidates export.
        with progress_scope(90, 95, "Rechecking source evidence before publication", label="CAD export"):
            _verified_record(record)
        with progress_scope(95, 100, "Publishing geometry package files", label="CAD export"):
            _commit_package(stage, destination)
    return GeometryPackage(
        destination, destination / _MANIFEST_NAME, tuple(destination / name for name in filenames), json.dumps(manifest)
    )
