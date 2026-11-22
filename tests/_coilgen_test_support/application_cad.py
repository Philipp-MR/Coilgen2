"""Shared fixtures for application cad."""

from __future__ import annotations
import numpy as np
from coilgen.application import DesignService
from coilgen.manufacturing import _save_xyz_polyline_as_step, save_mesh_as_step, save_mesh_as_stl
from coilgen.models import Mesh
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.storage import result_metadata_digest


BODY = "part001_body0001"


def _solid():
    # Offset closed tetrahedron makes origin/unit/bounds errors visible.
    return Mesh(
        vertices=np.asarray([[0.01, 0.02, 0.03], [0.014, 0.02, 0.03], [0.01, 0.025, 0.03], [0.01, 0.02, 0.036]]),
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )


def _run(tmp_path, *, sheet_accepted=True, ready=True, manufacturing_ok=True, step_unit="m"):
    directory = tmp_path / "saved_run"
    geometry = directory / "fabrication"
    geometry.mkdir(parents=True)
    stl = save_mesh_as_stl(_solid(), geometry / f"{BODY}_fabrication.stl", solid_name=BODY)
    step = save_mesh_as_step(
        _solid(), geometry / f"{BODY}_fabrication_faceted.step", solid_name=BODY, surface_mode="faceted_brep"
    )
    if step_unit == "mm":
        points = _solid()
        points.vertices *= 1000
        save_mesh_as_step(points, step, solid_name=BODY, surface_mode="faceted_brep")
        step.write_text(step.read_text().replace("SI_UNIT($,.METRE.)", "SI_UNIT(.MILLI.,.METRE.)"), encoding="ascii")
    centerline = _save_xyz_polyline_as_step(
        _solid().vertices[:2], geometry / f"{BODY}_fabrication_centroid.step", wire_name=BODY, part_index=1
    )
    support = save_mesh_as_stl(_solid(), directory / "support.stl", solid_name="support")
    arrays = directory / "scientific_result.npz"
    np.savez(arrays, vertices_m=_solid().vertices, current_A=np.asarray([8.5]), faces=_solid().faces)
    request = directory / "request.json"
    atomic_json(request, {"name": "Engineering fixture", "config": {}})
    manifest = directory / "manifest.json"
    atomic_json(manifest, {"request_sha256": "fixture-request", "input_files": {}})
    artifacts = [
        {"name": path.relative_to(directory).as_posix(), "kind": path.suffix.lstrip("."), "sha256": file_digest(path)}
        for path in (stl, step, centerline, support, arrays, request, manifest)
    ]
    result = {
        "request_sha256": "fixture-request",
        "project_id": "fixture-project",
        "project_revision": 4,
        "project_fingerprint": "full-fingerprint",
        "project_model_fingerprint": "model-fingerprint",
        "artifacts": artifacts,
        "summary": {
            "accepted": ready and sheet_accepted,
            "sheet_accepted": sheet_accepted,
            "construction_released": False,
        },
        "pipeline_sweeps": {"checks_passed": manufacturing_ok, "bodies": []},
        "fabrication": {
            "ready": ready,
            "failure_reasons": [] if ready else ["missing_return"],
            "paths": [{"body_id": BODY, "part_index": 1, "diagnostics": {"manufacturing_ok": manufacturing_ok}}],
        },
    }
    result["metadata_sha256"] = result_metadata_digest(result)
    atomic_json(directory / "result.json", result)
    return DesignService.load_run(directory)
