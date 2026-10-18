"""Durable, explicitly unqualified evidence from a failed numerical stage.

Arrays are ordinary NumPy data, never pickled objects or resumable checkpoints.
The original exception and acceptance policies remain authoritative.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def _metadata(value):
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return _metadata(value.tolist())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _metadata(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _metadata(getattr(value, item.name))
            for item in fields(value)
            if item.name not in {"progress_callback", "cancel_callback"}
        }
    if isinstance(value, dict):
        return {str(key): _metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unserialized_type": type(value).__name__}


def retain_pipeline_failure(error, *, state, config, stage, operation, timings, source_config=None, requirements=None):
    """Attach owned partial evidence without replacing the exception's type."""
    error.pipeline_failure = {
        "stage": stage.value,
        "operation": operation,
        "timings": list(timings),
        "config": config,
        "source_config": source_config,
        "requirements": requirements,
        "state": state,
    }


def write_failure_evidence(error, output_dir, *, prefix):
    """Save diagnostic geometry even when an assembly gate aborts the run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = getattr(error, "pipeline_failure", {})
    state = snapshot.get("state")
    arrays, parts = {}, []

    def array(name, value):
        if value is not None:
            data = np.asarray(value)
            if data.dtype.hasobject:
                raise TypeError(f"Failure evidence {name} cannot contain Python objects.")
            arrays[name] = data

    for index, part in enumerate(getattr(state, "parts", ())):
        key = f"part{index + 1:03d}"
        array(f"{key}_support_vertices_m", part.mesh.vertices)
        array(f"{key}_support_faces", part.mesh.faces)
        array(f"{key}_stream_function_A", part.stream_function)
        contours = []
        for contour_index, contour in enumerate(part.contour_lines):
            array_key = f"{key}_contour{contour_index + 1:04d}_points_m"
            array(array_key, contour.v)
            contours.append(
                {
                    "contour_index": contour_index + 1,
                    "points_array": array_key,
                    "potential_A": contour.potential,
                    "current_orientation": contour.current_orientation,
                    "turn_multiplicity": contour.turn_multiplicity,
                }
            )
        if part.wire_path is not None:
            array(f"{key}_source_wire_points_m", part.wire_path.v)
        if part.fabrication_path is not None:
            array(f"{key}_fabricated_wire_points_m", part.fabrication_path.wire_points_m)
        bodies = part.swept_bodies or ([part.swept_body] if part.swept_body is not None else [])
        for body_index, body in enumerate(bodies):
            mesh = getattr(body, "mesh", body)
            array(f"{key}_body{body_index + 1:04d}_vertices_m", mesh.vertices)
            array(f"{key}_body{body_index + 1:04d}_faces", mesh.faces)
        parts.append(
            {
                "part_index": index + 1,
                "contour_current_A": part.contour_step,
                "contours": contours,
                "topology": part.topology_diagnostics,
                "routing": part.route_diagnostics,
                "manufacturing": part.manufacturing_diagnostics,
                "source_segment_owners": getattr(part.wire_path, "segment_owners", ()),
                "source_segment_geometry": getattr(part.wire_path, "segment_geometry", ()),
            }
        )
    target = getattr(state, "target_field", None)
    if target is not None:
        for name in (
            "coords",
            "b",
            "weights",
            "suppressed",
            "target_field_group_inds",
            "target_gradient_dbdxyz",
            "suppression_normals",
            "suppression_area_weights",
        ):
            array(f"target_{name}", getattr(target, name, None))
    array("continuous_sheet_field_T", getattr(state, "sf_b_field", None))
    files = []
    array_path = output / f"{prefix}_failure_geometry.npz"
    if arrays:
        with array_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        files.append(array_path)
    document = {
        "execution_status": "failed",
        "engineering_acceptance": "not_qualified",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "failed_stage": snapshot.get("stage", "unavailable"),
        "failed_operation": snapshot.get("operation"),
        "completed_operations": snapshot.get("timings", []),
        "config": snapshot.get("config"),
        "config_scope": "Effective configuration of the failed stage; source configuration retained separately.",
        "source_config": snapshot.get("source_config"),
        "requirements": snapshot.get("requirements"),
        "parts": parts,
        "regularization_diagnostics": getattr(state, "regularization_diagnostics", {}),
        "target_diagnostics": getattr(target, "target_diagnostics", {}),
        "geometry_archive": array_path.name if arrays else None,
        "geometry_archive_sha256": sha256(array_path.read_bytes()).hexdigest() if arrays else None,
        "array_inventory": {
            name: {"shape": list(data.shape), "dtype": str(data.dtype)} for name, data in arrays.items()
        },
        "scope": "Partial stage diagnostics only. Not a completed result, reusable checkpoint, CAD export or construction release.",
    }
    path = output / f"{prefix}_failure_report.json"
    path.write_text(json.dumps(_metadata(document), indent=2, allow_nan=False), encoding="utf-8")
    files.append(path)
    error.coilgen_failure_files = tuple(files)
    return tuple(files)
