"""STEP generator credits preserve design authorship and SI geometry."""

import numpy as np
import pytest

from coilgen.attribution import AUTHOR_CREDIT, AUTHOR_ROLE, SOFTWARE_NAME
from coilgen.manufacturing import save_mesh_as_step
from coilgen.models import Mesh


@pytest.mark.parametrize("mode", ["advanced_brep", "faceted_brep"])
def test_solid_step_header_credits_software_without_claiming_design_authorship(tmp_path, mode):
    vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.03]])
    mesh = Mesh(vertices.copy(), np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))
    path = save_mesh_as_step(mesh, tmp_path / "conductor.step", surface_mode=mode)
    text = path.read_text(encoding="ascii")
    header = next(line for line in text.splitlines() if line.startswith("FILE_NAME("))

    assert (
        f"('CoilGen'),('CoilGen'),'{SOFTWARE_NAME} - {AUTHOR_CREDIT} ({AUTHOR_ROLE})','{SOFTWARE_NAME}','');"
    ) in header
    assert "SI_UNIT($,.METRE.)" in text
    assert "CARTESIAN_POINT('',(0.01,0.,0.))" in text
    assert "CARTESIAN_POINT('',(0.,0.02,0.))" in text
    assert "CARTESIAN_POINT('',(0.,0.,0.03))" in text
    np.testing.assert_array_equal(mesh.vertices, vertices)
