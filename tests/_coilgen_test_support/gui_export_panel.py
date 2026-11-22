"""Shared fixtures for gui export panel."""

import os
from pathlib import Path
from types import SimpleNamespace
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from coilgen.application.cad import GeometryOption


def _record(*, ready=True, sheet=True):
    return SimpleNamespace(
        directory=Path("immutable_run"),
        metadata={
            "name": "Cylinder Gy",
            "project_id": "owner",
            "project_revision": 4,
            "summary": {"construction_released": False, "sheet_accepted": sheet},
            "fabrication": {"ready": ready, "failure_reasons": ["missing_return"] if ready is False else []},
        },
    )


def _options(*, status="passed", inspection=False, step_unit="m"):
    common = dict(
        body_id="part001_body0001",
        scope="fabrication_review_geometry",
        check_status=status,
        bounds_m=((0.01, 0.02, 0.03), (0.014, 0.025, 0.036)),
        inspection_only=inspection,
        source_sha256="0" * 64,
    )
    return (
        GeometryOption(name="fabrication/conductor.stl", format="stl", source_unit="m", **common),
        GeometryOption(name="fabrication/conductor_faceted.step", format="step", source_unit=step_unit, **common),
    )
