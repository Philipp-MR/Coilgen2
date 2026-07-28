from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from coilgen.models import CoilPart, Mesh
from coilgen.manufacturing import mesh_manufacturing_diagnostics
from projects.halbach_cylindrical_3axis_gradient import gradient_project
from projects.halbach_cylindrical_3axis_gradient import run_halbach_cylindrical_3axis_gradient as run_script


def _closed_tetrahedron() -> Mesh:
    return Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 2, 1],
                [0, 1, 3],
                [0, 3, 2],
                [1, 2, 3],
            ],
            dtype=np.int64,
        ),
    )


def test_halbach_plotbooks_include_current_direction_and_route_sequence() -> None:
    for key in ("wire_sequence", "contour_current_orientation"):
        assert key in run_script.PLOT_INCLUDE
        assert key in run_script.PROJECT.plot_include
        assert key in gradient_project.STANDARD_ACTIVE_PLOT_KEYS


def test_warn_policy_writes_rejected_stl_and_diagnostics(tmp_path: Path) -> None:
    body = _closed_tetrahedron()
    part = CoilPart(mesh=body)
    part.swept_body = body
    part.manufacturing_diagnostics = mesh_manufacturing_diagnostics(body)
    part.route_diagnostics = {"route_quality_status": "fail"}
    result = SimpleNamespace(coil_parts=[part])
    channel_run = SimpleNamespace(
        result=result,
        channel=SimpleNamespace(safe_name="gx", name="GX"),
        design=SimpleNamespace(name="warning_fixture"),
        files=(),
    )
    project_run = SimpleNamespace(channel_runs=[channel_run], output_dir=tmp_path, files=())

    output = io.StringIO()
    with patch.object(run_script, "STL_EXPORT_REJECTION_POLICY", "warn"), redirect_stdout(output):
        files = run_script.export_project_stls(project_run)

    names = {path.name for path in files}
    assert "warning_fixture_swept_body_part1.stl" in names
    assert "warning_fixture_swept_body_part1_current_carrying_surface.stl" in names
    assert "warning_fixture_swept_body_manufacturing_diagnostics.json" in names
    assert "warning_fixture_swept_body_manufacturing_diagnostics.csv" in names
    assert "not manufacturing-ready" in output.getvalue()
    assert "Writing visualization STL files" in output.getvalue()
    assert not part.manufacturing_diagnostics["manufacturing_ok"]
