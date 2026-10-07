"""Design decisions remain portable; restoring variants preserves physics."""

from copy import deepcopy
from dataclasses import replace
import json
import shutil

import pytest

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.design_studies import (
    DesignStudyState,
    RunReference,
    create_variant,
    design_study_state,
    pin_baseline,
    set_study_notes,
    set_study_notes_batch,
)
from coilgen.application.project import _INPUT_FIELDS
from test_application_project_bundle import _prepared_document
from test_run_history import write_run


def test_baseline_and_decision_notes_do_not_change_numerical_identity(tmp_path):
    project, record, archive = _prepared_document(tmp_path)
    before = {
        path.relative_to(record.directory): path.read_bytes() for path in record.directory.rglob("*") if path.is_file()
    }
    pinned = pin_baseline(project, record)
    noted = set_study_notes(pinned, hypothesis="Check larger clearance", rationale="Keep as reference", record=record)
    assert noted.revision == project.revision
    assert noted.fingerprint == project.fingerprint
    assert noted.model_fingerprint == project.model_fingerprint
    assert noted.saved_runs[0].id == archive.id
    state = design_study_state(noted)
    assert state.baseline.matches(record)
    assert state.hypothesis == "Check larger clearance"
    assert state.note_for(record).rationale == "Keep as reference"
    assert {
        path.relative_to(record.directory): path.read_bytes() for path in record.directory.rglob("*") if path.is_file()
    } == before
    cleared = pin_baseline(noted, None)
    assert design_study_state(cleared).baseline is None
    assert design_study_state(cleared).notes == state.notes
    assert cleared.saved_runs == noted.saved_runs


def test_variant_restores_recorded_physics_study_and_owned_inputs_without_overwriting_source(tmp_path):
    source, record, archive = _prepared_document(tmp_path)
    project = source.update_config({"levels": 17, "conductor_cross_section_width": 0.003}).update_study(
        {"pipeline_stage": "evaluated"}
    )
    project = pin_baseline(project, record)
    before = (record.directory / "request.json").read_bytes()
    variant = create_variant(project, record, hypothesis="Test routing cut placement")
    assert variant.id == project.id
    assert variant.name == project.name
    assert variant.revision == project.revision + 1
    assert variant.model_fingerprint == source.model_fingerprint
    assert variant.study == source.study
    assert variant.values["output_directory"] == project.values["output_directory"]
    assert (record.directory / "request.json").read_bytes() == before
    assert variant.saved_runs[0].id == archive.id
    state = design_study_state(variant)
    assert state.baseline.matches(record)
    assert state.source.matches(record)
    assert state.hypothesis == "Test routing cut placement"
    for field in _INPUT_FIELDS:
        if source.values[field].startswith("asset:"):
            assert variant.values[field] == source.values[field]
            assert variant._assets[source.values[field][6:]].data == source._assets[source.values[field][6:]].data


def test_project_json_relocation_keeps_baseline_notes_and_reproducible_variant(tmp_path):
    source, record, _ = _prepared_document(tmp_path)
    project = set_study_notes(
        pin_baseline(source, record), hypothesis="Reduce contour count", rationale="Baseline", record=record
    )
    saved = project.save(tmp_path / "design.coilgen.json")
    moved = tmp_path / "moved" / saved.name
    moved.parent.mkdir()
    shutil.copyfile(saved, moved)
    shutil.rmtree(tmp_path / "sources")
    shutil.rmtree(record.directory)
    saved.unlink()
    reopened = ProjectDocument.load(moved)
    restored = reopened.saved_runs[0].materialize(tmp_path / "restored")
    state = design_study_state(reopened)
    assert state.baseline.matches(restored)
    assert state.note_for(restored).rationale == "Baseline"
    variant = create_variant(reopened, restored, hypothesis=state.hypothesis)
    assert variant.model_fingerprint == source.model_fingerprint
    DesignService.validate(variant).raise_for_errors()


def test_baseline_identity_distinguishes_request_revisions_and_ignores_live_job_state(tmp_path):
    first = write_run(tmp_path / "first")
    second = write_run(tmp_path / "second", revision=2)
    reference = RunReference.from_metadata(first.metadata)
    assert reference.matches(first)
    assert not reference.matches(second)
    metadata = first.metadata
    metadata["job"] = {"state": "completed", "elapsed": 999}
    assert RunReference.from_metadata(metadata) == reference


@pytest.mark.parametrize("operation", [pin_baseline, create_variant])
def test_foreign_or_unrecorded_owner_cannot_supply_baseline_or_variant(tmp_path, operation):
    project, record, _ = _prepared_document(tmp_path)
    foreign = project.duplicate("Another design")
    with pytest.raises(ValueError, match="owned"):
        operation(foreign, record)
    assert not foreign.saved_runs
    assert not foreign.session


def test_incomplete_legacy_request_is_viewable_but_not_a_reproducible_variant(tmp_path):
    record = write_run(tmp_path / "run")
    project = replace(ProjectDocument.new(), id="source-project")
    assert pin_baseline(project, record).saved_runs
    with pytest.raises(ValueError, match="every resolved setting"):
        create_variant(project, record)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(schema="unknown"),
        lambda value: value.update(hypothesis=None),
        lambda value: value.update(notes={}),
        lambda value: value.update(unrecognized=True),
        lambda value: value.update(
            baseline={"project_id": "foreign", "request_sha256": "a" * 64, "metadata_sha256": "b" * 64}
        ),
        lambda value: value.update(
            source={"project_id": "owner", "request_sha256": "not-a-hash", "metadata_sha256": "b" * 64}
        ),
    ],
)
def test_malformed_or_foreign_study_metadata_cannot_be_used(change):
    project = ProjectDocument.new()
    state = DesignStudyState().to_dict()
    change(state)
    malformed = project.with_results((), session={"design_study": state})
    with pytest.raises(ValueError):
        design_study_state(malformed)


def test_state_is_detached_and_rejects_duplicate_result_notes(tmp_path):
    project, record, _ = _prepared_document(tmp_path)
    project = set_study_notes(project, hypothesis="Trial", record=record)
    value = deepcopy(project.session)
    value["design_study"]["hypothesis"] = "Mutated"
    assert design_study_state(project).hypothesis == "Trial"
    data = design_study_state(project).to_dict()
    data["notes"] *= 2
    with pytest.raises(ValueError, match="unique"):
        DesignStudyState.from_dict(data)
    with pytest.raises(ValueError, match="Select"):
        set_study_notes(project, hypothesis="Trial", rationale="Use this design")


def test_changed_original_input_or_result_prevents_variant_restore(tmp_path):
    project, record, _ = _prepared_document(tmp_path)
    (record.directory / "scientific_result.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        create_variant(project, record)
    assert not project.saved_runs
    assert not project.session


def test_original_request_owner_and_result_owner_must_agree(tmp_path):
    # A metadata checksum does not authenticate ownership. Check the verified
    # original request too, so inconsistent project links cannot be restored.
    from coilgen.persistence import atomic_json
    from coilgen.research.storage import result_metadata_digest

    project, record, _ = _prepared_document(tmp_path)
    metadata = json.loads((record.directory / "result.json").read_bytes())
    metadata["project_id"] = "different-owner"
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(record.directory / "result.json", metadata)
    other = replace(project, id="different-owner")
    with pytest.raises(ValueError, match="identities disagree"):
        pin_baseline(other, record)


def test_batch_decisions_keep_original_candidate_hypotheses_and_all_evidence(tmp_path):
    project, first, _ = _prepared_document(tmp_path)
    project = set_study_notes(project, hypothesis="Original experiment", rationale="Initial reference", record=first)
    candidate = project.update_config({"levels": 14})
    DesignService.run(candidate, tmp_path / "second")
    second = DesignService.load_run(tmp_path / "second")
    updated = set_study_notes_batch(
        candidate,
        hypothesis="Next experiment",
        decisions=((first, "Retain baseline"), (second, "Investigate tighter routing")),
    )
    state = design_study_state(updated)
    assert state.hypothesis == "Next experiment"
    assert state.note_for(first).hypothesis == "Original experiment"
    assert state.note_for(first).rationale == "Retain baseline"
    assert state.note_for(second).hypothesis == "Next experiment"
    assert state.note_for(second).rationale == "Investigate tighter routing"
    assert len(updated.saved_runs) == 2
    assert updated.fingerprint == candidate.fingerprint
    assert updated.revision == candidate.revision
    reopened = ProjectDocument.from_dict(updated.to_dict())
    assert design_study_state(reopened) == state


def test_batch_decisions_are_atomic_when_later_evidence_is_missing(tmp_path):
    project, first, _ = _prepared_document(tmp_path)
    other = project.update_config({"levels": 14})
    DesignService.run(other, tmp_path / "second")
    second = DesignService.load_run(tmp_path / "second")
    (second.directory / "scientific_result.npz").unlink()
    with pytest.raises(ValueError, match="integrity"):
        set_study_notes_batch(project, hypothesis="Trial", decisions=((first, "Keep"), (second, "Reject")))
    assert project.session == {}
    assert project.saved_runs == ()
    with pytest.raises(ValueError, match="distinct"):
        set_study_notes_batch(project, hypothesis="Trial", decisions=((first, "One"), (first, "Two")))
    with pytest.raises(ValueError, match="contain"):
        set_study_notes_batch(project, hypothesis="Trial", decisions=((first,),))
