"""Shared fixtures for application evidence."""

from __future__ import annotations
import numpy as np
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import result_metadata_digest


def _arrays():
    return {
        "vertices_m": np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
        "faces": np.asarray([[0, 1, 2]]),
        "stream_function_A": np.zeros(3),
        "target_points_m": np.asarray([[0, 0, 0.02], [0, 0.01, 0.03]]),
        "target_field_T": np.zeros((2, 3)),
        "sheet_field_T": np.zeros((2, 3)),
        "routed_field_T": np.zeros((2, 3)),
    }


def _saved_run(tmp_path, *, checksum=True, request=True):
    directory = tmp_path / "saved"
    directory.mkdir()
    np.savez(directory / "scientific_result.npz", **_arrays())
    names = ["scientific_result.npz"]
    if request:
        submitted = ResearchRequest(
            name="Saved winding", config={"levels": 7}, project_id="fixture-project", project_revision=3
        )
        atomic_json(directory / "request.json", submitted.to_dict())
        names.append("request.json")
    atomic_json(directory / "manifest.json", {"request_sha256": "fixture-request", "input_files": {}})
    result = {
        "request_sha256": "fixture-request",
        "project_id": "fixture-project",
        "project_revision": 3,
        "summary": {"accepted": False},
        "artifacts": [
            {
                "name": name,
                "kind": "npz" if name.endswith(".npz") else "request",
                "sha256": file_digest(directory / name),
            }
            for name in names
        ],
    }
    if checksum:
        result["metadata_sha256"] = result_metadata_digest(result)
    atomic_json(directory / "result.json", result)
    return directory
