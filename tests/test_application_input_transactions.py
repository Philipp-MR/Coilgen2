"""Config/study drafts publish together as one immutable project revision."""

from copy import deepcopy
from dataclasses import replace

import pytest

from coilgen.application import ProjectDocument
from coilgen.config import CoilGenConfig


def project():
    return ProjectDocument.from_config(CoilGenConfig(coil_mesh_file="create cylinder mesh"))


def test_combined_edit_publishes_one_revision_preserving_untouched_precision():
    original = ProjectDocument.from_config(
        replace(project().config, target_gradient_strength=0.012345678901234567, research_physics=None)
    ).with_results((), session={"view": {"dataset": "target"}})
    before = original.to_dict()
    updated = original.update_inputs(
        {"levels": 27},
        {"pipeline_stage": "contours", "validation_resolution": 17},
        expected_fingerprint=original.fingerprint,
    )
    assert updated.id == original.id and updated.revision == original.revision + 1
    assert updated.values == {**original.values, "levels": 27}
    assert updated.study == {**original.study, "pipeline_stage": "contours", "validation_resolution": 17}
    assert updated.session == original.session and updated.saved_runs == original.saved_runs
    assert original.to_dict() == before


@pytest.mark.parametrize(
    "config,study,message",
    [
        ({"levels": -1}, {"pipeline_stage": "sheet"}, "levels"),
        ({"levels": 27}, {"pipeline_stage": "unknown"}, "pipeline_stage"),
        ({"unrecognized": 1}, {"pipeline_stage": "sheet"}, "Unknown CoilGenConfig"),
        ({"levels": 27}, {"unrecognized": 1}, "Unknown or malformed study"),
    ],
)
def test_either_invalid_mapping_rejects_whole_transaction(config, study, message):
    original = project()
    before = original.to_dict()
    with pytest.raises(ValueError, match=message):
        original.update_inputs(config, study)
    assert original.to_dict() == before


def test_stale_baseline_rejects_before_new_source_is_read():
    original = project()
    current = original.update_study({"pipeline_stage": "sheet"})
    with pytest.raises(ValueError, match="project changed"):
        current.update_inputs({"coil_mesh_file": "missing.npz"}, expected_fingerprint=original.fingerprint)
    assert current.revision == original.revision + 1


def test_noops_return_same_document_and_existing_delegates_keep_revision_behavior():
    original = project()
    assert original.update_inputs() is original
    assert original.update_inputs({}, {}) is original
    assert original.update_inputs({"levels": original.values["levels"]}, original.study) is original
    assert original.update_config({}) is original and original.update_study({}) is original
    assert original.update_config({"levels": 27}).revision == original.revision + 1
    assert original.update_study({"pipeline_stage": "sheet"}).revision == original.revision + 1


def test_incomplete_draft_opt_in_does_not_suppress_other_errors():
    original = ProjectDocument.new()
    updated = original.update_inputs({"levels": 27}, {"pipeline_stage": "prepared"}, allow_incomplete=True)
    assert updated.revision == 2 and updated.values["coil_mesh_file"] == "none"
    with pytest.raises(ValueError, match="target gradient strength"):
        original.update_inputs({"target_gradient_strength": -1}, {"pipeline_stage": "prepared"}, allow_incomplete=True)
    # Study-only edits are allowed on existing incomplete projects, as before.
    assert original.update_inputs(study_changes={"pipeline_stage": "prepared"}).revision == 2


def test_owned_inputs_and_caller_mappings_are_detached(tmp_path):
    source = tmp_path / "target.csv"
    source.write_text("x,y,z,bz\n0,0,0,0.00001\n", encoding="utf-8")
    original = ProjectDocument.from_config(
        replace(project().config, target_field_definition_file=str(source), target_gradient_normalization="none")
    )
    source.unlink()
    config_changes = {"b_0_direction": [0.0, 1.0, 0.0]}
    study_changes = {"pipeline_stage": "sheet"}
    updated = original.update_inputs(config_changes, study_changes)
    expected = deepcopy(updated.to_dict())
    config_changes["b_0_direction"][1] = 99
    study_changes["pipeline_stage"] = "unknown"
    assert updated.to_dict() == expected
    assert updated._assets == original._assets


@pytest.mark.parametrize(
    "kwargs",
    [
        {"config_changes": []},
        {"study_changes": []},
        {"allow_incomplete": "true"},
        {"expected_fingerprint": ""},
        {"expected_fingerprint": 123},
    ],
)
def test_transaction_arguments_are_explicitly_typed(kwargs):
    with pytest.raises(TypeError):
        project().update_inputs(**kwargs)


@pytest.mark.parametrize(
    "controls,message",
    [
        ({"project_revision": 999}, "Snapshot project_revision"),
        ({"pipeline_stage": "unknown"}, "pipeline_stage"),
        ({"validation_resolution": 0}, "validation_resolution"),
        ({"fabricate": True, "pipeline_stage": "sheet"}, "Fabrication review"),
    ],
)
def test_invalid_snapshot_controls_create_no_output_inputs(tmp_path, controls, message):
    source = tmp_path / "target.csv"
    source.write_text("x,y,z,bz\n0,0,0,0.00001\n", encoding="utf-8")
    original = ProjectDocument.from_config(replace(project().config, target_field_definition_file=str(source)))
    directory = tmp_path / "rejected-snapshot"
    with pytest.raises(ValueError, match=message):
        original.snapshot(directory, **controls)
    assert not directory.exists()
