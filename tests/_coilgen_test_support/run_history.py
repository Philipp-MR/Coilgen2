"""Shared fixtures for run history."""

from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from coilgen.application.service import DesignService
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.storage import result_metadata_digest


def write_run(directory, *, revision=1, config=None, summary=None, refitted=False, grid_shift=0, input_path=None):
    """Small recorded evidence fixture using the real persistence verifier."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    config = {"levels": 8, "target_field_component": "Bz", "b_0_direction": [0, 0, 1], **(config or {})}
    request = {
        "name": "Original scientific name",
        "project_id": "source-project",
        "project_revision": revision,
        "config": config,
        "pipeline_stage": "evaluated",
        "validation_resolution": 12,
        "design_margin": 1,
        "fabricate": False,
        "kernel_backend": "auto",
    }
    digest = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    input_files = {}
    if input_path:
        input_files["target_field_definition_file"] = {"path": str(input_path), "sha256": file_digest(input_path)}
    atomic_json(directory / "request.json", request)
    atomic_json(
        directory / "manifest.json",
        {
            "request_sha256": digest,
            "request": request,
            "resolved_config": config,
            "input_files": input_files,
        },
    )
    points = np.array([[0, 0, 0.01], [0.01, 0, 0]]) + grid_shift
    target = np.array([[0, 0, 1e-5], [0, 0, -1e-5]])
    np.savez_compressed(
        directory / "scientific_result.npz",
        target_points_m=points,
        target_field_T=target,
        target_weights=np.ones(2),
        target_suppressed=np.zeros(2, bool),
        validation_points_m=points + 0.002,
        validation_target_T=target,
    )
    wire = {
        "available": True,
        "component": "Bz",
        "sample_count": 2,
        "current_A": 4.0,
        "current_source": "fit" if refitted else "stream_function",
        "current_refitted": refitted,
        "geometry_sources": ["oriented_unconnected_contours"],
        "field_model": "thin_wire_biot_savart",
        "rms_convention": "scalar",
        "rms_error_T": 1e-6,
        "peak_error_T": 2e-6,
        "relative_rms_error": 0.1,
        "independent_sampling": False,
        "acceptance_claim": False,
        "at_stream_current": {
            "available": True,
            "sample_count": 2,
            "current_A": 3.9,
            "current_source": "original_contour_step",
            "current_refitted": False,
            "rms_error_T": 1.2e-6,
        },
    }
    metadata = {
        "name": request["name"],
        "project_id": request["project_id"],
        "project_revision": revision,
        "request_sha256": digest,
        "pipeline_stage": "evaluated",
        # Deliberately not authoritative for parameter comparison.
        "config": {"levels": 999},
        "summary": {
            "accepted": False,
            "scope": "continuous_sheet",
            "relative_field_error": 0.02,
            "validated_relative_field_error": 0.025,
            "power_w": 1.2,
            "current_a": None,
            "resistance_ohm": None,
            "inductance_h": None,
            **(summary or {}),
        },
        "validation": {
            "available": True,
            "point_count": 2,
            "rms_error_T": 2.5e-7,
            "relative_field_error": 0.025,
            "peak_error_T": 3e-7,
            "independent_quadrature_check": False,
        },
        "wire_fields": {"contour": wire},
        "artifacts": [
            {"name": name, "kind": kind, "sha256": file_digest(directory / name)}
            for name, kind in (
                ("request.json", "request"),
                ("manifest.json", "provenance"),
                ("scientific_result.npz", "npz"),
            )
        ],
    }
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(directory / "result.json", metadata)
    return DesignService.load_run(directory)
