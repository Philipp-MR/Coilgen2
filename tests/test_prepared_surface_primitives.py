"""Every selectable winding primitive reaches the shared prepared boundary."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application import DesignService, ProjectDocument
from coilgen.gui.editors import SUPPORT_EDITORS
from coilgen.pipeline_stages import PipelineCheckpoint


BOUNDARIES_PER_PART = {
    "Cylinder": (2,),
    "Biplanar": (1, 1),
    "Conical frustum": (2,),
    "Rectangle": (1,),
    "Disc": (1,),
    "Annulus": (2,),
    "Partial cylinder": (1,),
    "Elliptical cylinder": (2,),
    "Superellipse": (2,),
    "Circular biplanar": (1, 1),
    "Spherical cap": (1,),
    "Ellipsoidal cap": (1,),
    "Toroidal segment": (2,),
}


def small_config(editor):
    config = CoilGenConfig(
        coil_mesh_file=editor.source,
        surface_is_cylinder_flag=editor.cylindrical,
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_geometry={"shape": "sphere", "radius_m": 0.015, "sampling": "volume"},
        target_region_resolution=4,
        sf_opt_method="tikhonov",
        research_physics=None,
        iteration_num_mesh_refinement=0,
        plot_flag=False,
        progress_flag=False,
    )
    edits = {
        field.key: max(field.minimum, 8 if "angular" in field.key or "major" in field.key else 4)
        for field in editor.fields
        if field.integer
    }
    parameters = getattr(config, editor.parameter_field)
    if editor.label == "Circular biplanar":
        edits["mesh_edge_mm"] = 2 * parameters[1] * 1000 / 8
    return replace(config, **{editor.parameter_field: editor.apply_edits(parameters, edits)})


@pytest.mark.parametrize("editor", SUPPORT_EDITORS, ids=lambda editor: editor.label)
def test_generated_surface_reaches_owned_prepared_stage_with_valid_boundaries_and_operators(editor, tmp_path):
    project = ProjectDocument.from_config(small_config(editor), name=f"Prepared {editor.label}").update_study(
        {"pipeline_stage": "prepared", "fabricate": False}
    )
    fingerprint = project.fingerprint
    result = DesignService.run(project, tmp_path / "run", threads=1)
    assert result["pipeline_stage"] == "prepared"
    assert result["project_id"] == project.id and result["project_fingerprint"] == fingerprint
    assert result["summary"]["scope"] == "prepared_geometry"
    assert result["summary"]["accepted"] is False
    record = DesignService.load_run(tmp_path / "run")
    checkpoint = PipelineCheckpoint.load(record.directory / result["checkpoint"])
    state = checkpoint.state
    assert tuple(len(part.mesh.boundaries) for part in state.parts) == BOUNDARIES_PER_PART[editor.label]
    assert np.isfinite(state.target_field.coords).all() and np.isfinite(state.target_field.b).all()
    for part in state.parts:
        assert part.sensitivity_matrix.shape == (3, len(state.target_field.coords), part.mesh.n_vertices)
        assert np.isfinite(part.sensitivity_matrix).all() and np.any(part.sensitivity_matrix)
        assert len(part.basis_elements) == part.mesh.n_vertices
        for basis in part.basis_elements:
            assert np.all(basis.area > 0) and np.isfinite(basis.current).all()
        for boundary in part.mesh.boundaries:
            assert len(boundary) >= 4 and boundary[0] == boundary[-1]
    assert np.isfinite(record.arrays["target_field_T"]).all()
    assert project.fingerprint == fingerprint
