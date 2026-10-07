"""Incomplete documents permit explicit drafting, never invalid numerical inputs."""

import pytest

from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService
from coilgen.config import CoilGenConfig


def blank():
    return ProjectDocument.from_config(CoilGenConfig(field_shape_function="none"), name="Blank design")


def test_missing_support_can_be_retained_only_with_explicit_opt_in():
    project = blank()
    changes = {"field_shape_function": "y", "target_gradient_strength": 0.035}
    with pytest.raises(ValueError, match="No coil surface"):
        project.update_config(changes)
    with pytest.raises(ValueError, match="No coil surface"):
        project.update_config(changes, allow_incomplete=False)
    updated = project.update_config(changes, allow_incomplete=True)
    assert updated.id == project.id and updated.revision == project.revision + 1
    assert updated.values == {**project.values, **changes}
    assert project.values["field_shape_function"] == "none"
    assert updated.model_fingerprint != project.model_fingerprint
    assert {error.field for error in DesignService.validate(updated).errors} == {"coil_mesh_file"}


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"target_gradient_strength": -1.0}, "target gradient strength"),
        ({"specific_conductivity_conductor": -1.0}, "conductor resistivity"),
        ({"generated_mesh_rotation_axis": [0.0, 0.0, 0.0], "generated_mesh_rotation_angle": 30.0}, "nonzero"),
        ({"coil_mesh_file": "create nonexistent mesh"}, "Unknown mesh source"),
        ({"coil_mesh_file": "NONE"}, "Unknown mesh source"),
        (
            {"coil_mesh_file": "create cylinder mesh", "cylinder_mesh_parameter_list": [1, -0.2, 8, 4, 0, 0, 0, 0]},
            "cylinder radius",
        ),
        ({"field_shape_function": ""}, "Target field expression is empty"),
    ],
)
def test_opt_in_never_suppresses_physical_or_selected_geometry_errors(changes, message):
    project = blank()
    identity = project.fingerprint
    with pytest.raises(ValueError, match=message):
        project.update_config(changes, allow_incomplete=True)
    assert project.fingerprint == identity and project.revision == 1


def test_opt_in_still_rejects_unknown_settings_and_non_boolean_flag():
    project = blank()
    with pytest.raises(ValueError, match="Unknown CoilGenConfig"):
        project.update_config({"unrecognized": 1}, allow_incomplete=True)
    with pytest.raises(TypeError, match="boolean"):
        project.update_config({"levels": 11}, allow_incomplete="true")


def test_partial_document_can_be_saved_restored_and_replayed_as_immutable_edits(tmp_path):
    project = blank()
    initial = project.to_dict()
    updated = project.update_config({"levels": 17, "field_shape_function": "z"}, allow_incomplete=True)
    edited = updated.to_dict()
    saved = ProjectDocument.load(updated.save(tmp_path / "partial.json"))
    assert saved.to_dict() == edited
    assert saved.fingerprint == updated.fingerprint and saved.model_fingerprint == updated.model_fingerprint
    # Undo/redo stores immutable revisions: restoring either snapshot is exact.
    restored = ProjectDocument.from_dict(initial)
    redone = ProjectDocument.from_dict(edited)
    assert restored.fingerprint == project.fingerprint and restored.model_fingerprint == project.model_fingerprint
    assert redone.fingerprint == updated.fingerprint and redone.model_fingerprint == updated.model_fingerprint
    assert project.to_dict() == initial
    # A missing support remains an execution blocker after reload.
    with pytest.raises(ValueError, match="No coil surface"):
        DesignService.prepare(saved, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_valid_support_finishes_partial_geometry_without_relaxing_normal_validation():
    project = blank().update_config({"field_shape_function": "x"}, allow_incomplete=True)
    finished = project.update_config({"coil_mesh_file": "create cylinder mesh"})
    assert DesignService.validate(finished).ok
    assert finished.values["field_shape_function"] == "x"
    assert finished.revision == 3
