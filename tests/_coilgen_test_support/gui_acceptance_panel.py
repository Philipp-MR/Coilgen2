"""Shared fixtures for gui acceptance panel."""

import os
from pathlib import Path
from types import SimpleNamespace
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _record(name="first", *, failed=True):
    return SimpleNamespace(
        directory=Path(name),
        metadata={
            "project_id": "owner",
            "project_revision": 4,
            "artifacts": [{"name": "request.json", "sha256": "0" * 64}],
            "pipeline_stage": "evaluated",
            "summary": {"sheet_accepted": True, "scope": "fabrication_review"},
            "constraints": [],
            "wire_fields": {},
            "pipeline_sweeps": {"checks_passed": not failed, "body_count": 1},
            "fabrication": {
                "ready": not failed,
                "paths": [
                    {
                        "body_id": "body1",
                        "diagnostics": {
                            "sweep_capsule_min_centerline_distance_m": 0.001,
                            "sweep_required_centerline_clearance_m": 0.003,
                            "sweep_capsule_clearance_violation": failed,
                        },
                    }
                ],
            },
        },
    )
