import math

import numpy as np
import pytest

from coilgen.application.session import WorkbenchSession
from coilgen.geometry import load_mesh_from_config


def test_cylinder_form_edits_physical_mesh_in_one_undo_step(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    original = session.project
    form = session.settings_values("geometry")
    assert form["cylinder_length_mm"] == pytest.approx(200)
    session.apply_settings(
        {**form, "cylinder_length_mm": 230, "cylinder_radius_mm": 95, "cylinder_angular": 40, "cylinder_axial": 20}
    )
    edited = session.project
    assert edited.revision == original.revision + 1
    mesh = load_mesh_from_config(edited.config, edited.project_root)
    assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(0.230)
    np.testing.assert_allclose(np.linalg.norm(mesh.vertices[:, :2], axis=1), 0.095)
    assert len(np.unique(mesh.vertices[:, 2])) == 21
    assert len(mesh.vertices) == 21 * 40
    session.apply_settings({"cylinder_axial": 24})
    mesh = load_mesh_from_config(session.project.config, session.project.project_root)
    assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(0.230)
    assert session.undo() and session.project is edited
    assert session.undo() and session.project is original
    assert session.redo() and session.project is edited


def test_cylinder_edits_preserve_orientation_and_reject_invalid_drafts(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    params = list(session.project.config.cylinder_mesh_parameter_list)
    params[4:] = [0, 1, 0, math.pi / 7]
    session.apply_config({"cylinder_mesh_parameter_list": params})
    original = session.project
    session.apply_settings(session.settings_values("geometry"))
    assert session.project is original
    session.apply_settings({"cylinder_rotation_deg": 30, "cylinder_radius_mm": 105})
    np.testing.assert_allclose(session.project.config.cylinder_mesh_parameter_list[4:], [0, 1, 0, math.pi / 6])
    assert session.settings_values("geometry")["cylinder_radius_mm"] == pytest.approx(105)
    edited = session.project
    for invalid in (0, 15.5):
        with pytest.raises(ValueError):
            session.apply_settings({"cylinder_radius_mm": 50, "cylinder_axial": invalid})
        assert session.project is edited


def test_duplicate_channel_creates_independent_editable_copy_with_unique_name(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    source = session.channels[0]
    first = session.duplicate_channel()
    assert session.project is first and first.id != source.project.id
    assert first.model_fingerprint == source.project.model_fingerprint
    assert first.name == source.name + " copy"
    session.apply_settings({"levels": source.project.config.levels + 1})
    assert session.channels[0].project is source.project
    second = session.duplicate_channel(source.id)
    assert second.name == source.name + " copy 2"
    assert len({channel.project.id for channel in session.channels}) == 3
    assert session.undo() and session.project.id == first.id
    assert session.redo() and session.project.id == second.id


@pytest.mark.parametrize("axis", "xyz")
def test_halbach_uniform_mechanical_field_matches_reference_and_tracks_edits(tmp_path, axis):
    session = WorkbenchSession(tmp_path)
    session.load_template(f"halbach_g{axis}")
    project = session.project
    assert project.config.b_0_strength_T == 0.05
    np.testing.assert_allclose(project.config.research_physics["b0_T"], [0, 0, 0.05])
    session.apply_settings({"b_0_direction": "0, 3, 4", "b_0_strength_T": 0.1, "design_margin": 0.75})
    cfg = session.project.config
    assert session.project.revision == project.revision + 1
    assert session.project.study["design_margin"] == 0.75
    assert cfg.research_physics["objective"] == project.config.research_physics["objective"]
    np.testing.assert_allclose(cfg.research_physics["b0_T"], [0, 0.06, 0.08])
    with pytest.raises(ValueError):
        session.apply_settings({"b_0_direction": "0, 0, 0"})
    assert session.project.config.research_physics == cfg.research_physics
    assert session.undo() and session.project is project


def test_explicit_mechanical_field_is_independent_of_reference(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    custom = {"objective": "power", "b0_T": [2, 0, 0]}
    session.apply_inputs({"b_0_strength_T": 0.2, "research_physics": custom})
    session.apply_settings({"b_0_direction": "0, 2, 0", "b_0_strength_T": 0.5})
    assert session.project.config.research_physics == custom

    # A per-face background remains independent even if its nominal scalar field
    # initially agrees with the MRI reference.
    spatial = {**custom, "b0_T": [0, 0.5, 0], "background_B_per_face_T": [[1, 0, 2]]}
    session.apply_config({"research_physics": spatial})
    session.apply_settings({"b_0_strength_T": 0.8})
    assert session.project.config.research_physics == spatial
