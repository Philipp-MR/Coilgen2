"""Portable baseline and design decisions, independent of numerical settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from ..config import CoilGenConfig
from ..pipeline_stages import config_field_stage
from ..research.requests import ResearchRequest
from ..research.storage import result_metadata_digest
from .assets import InputAsset
from .bundles import RunArchive
from .evidence import read_verified_artifact, verify_record
from .project import ProjectDocument, _CONFIG_FIELDS, _INPUT_FIELDS, _values
from .service import DesignService, RunRecord


STUDY_SESSION_KEY = "design_study"
STUDY_SESSION_SCHEMA = "coilgen-design-study-v1"


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Design study identities must be lowercase SHA-256 hashes.")
    return value


def _text(value, label):
    if not isinstance(value, str) or len(value) > 4000:
        raise ValueError(f"{label} must be text of at most 4000 characters.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class RunReference:
    """Relocation-safe identity of one result, including its original owner."""

    project_id: str
    request_sha256: str
    metadata_sha256: str

    def __post_init__(self):
        if not isinstance(self.project_id, str) or not self.project_id or len(self.project_id) > 128:
            raise ValueError("Design study reference requires a recorded project owner.")
        _hash(self.request_sha256)
        _hash(self.metadata_sha256)

    @classmethod
    def from_metadata(cls, metadata):
        if not isinstance(metadata, dict):
            raise ValueError("Design study run metadata must be a JSON object.")
        clean = {key: value for key, value in metadata.items() if key != "job"}
        return cls(clean.get("project_id"), clean.get("request_sha256"), result_metadata_digest(clean))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"project_id", "request_sha256", "metadata_sha256"}:
            raise ValueError("Malformed design study run reference.")
        return cls(**value)

    def to_dict(self):
        return {
            "project_id": self.project_id,
            "request_sha256": self.request_sha256,
            "metadata_sha256": self.metadata_sha256,
        }

    def matches(self, record):
        try:
            return self == self.from_metadata(record.metadata)
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class CandidateNote:
    reference: RunReference
    hypothesis: str = ""
    rationale: str = ""

    def __post_init__(self):
        if not isinstance(self.reference, RunReference):
            raise ValueError("Candidate note requires a run reference.")
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "Design hypothesis"))
        object.__setattr__(self, "rationale", _text(self.rationale, "Selection rationale"))

    def to_dict(self):
        return {"reference": self.reference.to_dict(), "hypothesis": self.hypothesis, "rationale": self.rationale}


@dataclass(frozen=True, slots=True)
class DesignStudyState:
    baseline: RunReference | None = None
    source: RunReference | None = None
    hypothesis: str = ""
    notes: tuple[CandidateNote, ...] = ()

    def __post_init__(self):
        if any(item is not None and not isinstance(item, RunReference) for item in (self.baseline, self.source)):
            raise ValueError("Baseline and variant source must be recorded run references.")
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "Design hypothesis"))
        notes = tuple(self.notes)
        if not all(isinstance(note, CandidateNote) for note in notes):
            raise ValueError("Design study notes must contain candidate notes.")
        if len({note.reference for note in notes}) != len(notes):
            raise ValueError("Design study candidate notes must have unique result identities.")
        object.__setattr__(self, "notes", notes)

    @classmethod
    def from_dict(cls, value):
        expected = {"schema", "baseline", "source", "hypothesis", "notes"}
        if not isinstance(value, dict) or set(value) != expected or value.get("schema") != STUDY_SESSION_SCHEMA:
            raise ValueError("Malformed design study session.")
        if not isinstance(value["notes"], list):
            raise ValueError("Design study candidate notes must be a JSON array.")
        notes = []
        for item in value["notes"]:
            if not isinstance(item, dict) or set(item) != {"reference", "hypothesis", "rationale"}:
                raise ValueError("Malformed design study candidate note.")
            notes.append(
                CandidateNote(RunReference.from_dict(item["reference"]), item["hypothesis"], item["rationale"])
            )
        return cls(
            baseline=None if value["baseline"] is None else RunReference.from_dict(value["baseline"]),
            source=None if value["source"] is None else RunReference.from_dict(value["source"]),
            hypothesis=value["hypothesis"],
            notes=tuple(notes),
        )

    def to_dict(self):
        return {
            "schema": STUDY_SESSION_SCHEMA,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "source": self.source.to_dict() if self.source else None,
            "hypothesis": self.hypothesis,
            "notes": [note.to_dict() for note in self.notes],
        }

    def note_for(self, record):
        try:
            reference = RunReference.from_metadata(record.metadata)
        except (TypeError, ValueError):
            return None
        return next((note for note in self.notes if note.reference == reference), None)


def design_study_state(project: ProjectDocument) -> DesignStudyState:
    value = project.session.get(STUDY_SESSION_KEY)
    state = DesignStudyState() if value is None else DesignStudyState.from_dict(value)
    references = [state.baseline, state.source, *(note.reference for note in state.notes)]
    if any(reference is not None and reference.project_id != project.id for reference in references):
        raise ValueError("Design study reference belongs to a different project owner.")
    return state


def _with_state(project, state):
    session = project.session
    session[STUDY_SESSION_KEY] = state.to_dict()
    return project.with_results(project.saved_runs, session=session)


def _owned_record(project, record):
    verified = verify_record(record) if isinstance(record, RunRecord) else DesignService.load_run(record)
    if verified.metadata.get("project_id") != project.id:
        raise ValueError("This operation requires a verified run owned by the current project.")
    if not any(
        artifact.get("name") == "request.json" and artifact.get("sha256")
        for artifact in verified.metadata.get("artifacts", ())
    ):
        raise ValueError("Design studies require an integrity-checked original request.json artifact.")
    request = json.loads(read_verified_artifact(verified, "request.json"))
    if (
        not isinstance(request, dict)
        or request.get("project_id") != project.id
        or request.get("project_revision") != verified.metadata.get("project_revision")
    ):
        raise ValueError("Original request and recorded result project identities disagree.")
    return verified


def pin_baseline(project: ProjectDocument, record: RunRecord | str | Path | None) -> ProjectDocument:
    """Pin verified evidence; display-only selection does not change physics."""
    state = design_study_state(project)
    if record is None:
        return _with_state(project, replace(state, baseline=None))
    verified = _owned_record(project, record)
    archive = RunArchive.capture(verified)
    saved = {run.id: run for run in project.saved_runs}
    saved.setdefault(archive.id, archive)
    state = replace(state, baseline=RunReference.from_metadata(archive.metadata))
    return _with_state(project.with_results(tuple(saved.values())), state)


def set_study_notes(
    project: ProjectDocument, *, hypothesis: str, rationale: str = "", record: RunRecord | str | Path | None = None
) -> ProjectDocument:
    """Record the next experiment and, when selected, a result decision."""
    state = design_study_state(project)
    hypothesis, rationale = _text(hypothesis, "Design hypothesis"), _text(rationale, "Selection rationale")
    notes = state.notes
    if record is None and rationale:
        raise ValueError("Select a verified owned run before recording a result decision.")
    if record is not None:
        verified = _owned_record(project, record)
        archive = RunArchive.capture(verified)
        saved = {run.id: run for run in project.saved_runs}
        saved.setdefault(archive.id, archive)
        project = project.with_results(tuple(saved.values()))
        reference = RunReference.from_metadata(archive.metadata)
        note = CandidateNote(reference, hypothesis, rationale)
        notes = tuple(previous for previous in notes if previous.reference != reference) + (note,)
    return _with_state(project, replace(state, hypothesis=hypothesis, notes=notes))


def set_study_notes_batch(project: ProjectDocument, *, hypothesis: str, decisions=()) -> ProjectDocument:
    """Commit all pending result decisions atomically with the design intent.

    Existing candidate hypotheses remain tied to their original records. Every
    referenced run is reverified and archived before a detached project is
    returned; one invalid decision leaves the original project unchanged.
    """
    state = design_study_state(project)
    hypothesis = _text(hypothesis, "Design hypothesis")
    notes = {note.reference: note for note in state.notes}
    saved = {run.id: run for run in project.saved_runs}
    seen = set()
    for decision in decisions:
        if not isinstance(decision, (tuple, list)) or len(decision) != 2:
            raise ValueError("A pending decision must contain its verified run and rationale.")
        record, rationale = decision
        rationale = _text(rationale, "Selection rationale")
        archive = RunArchive.capture(_owned_record(project, record))
        reference = RunReference.from_metadata(archive.metadata)
        if reference in seen:
            raise ValueError("Pending decisions must refer to distinct results.")
        seen.add(reference)
        previous = notes.get(reference)
        notes[reference] = CandidateNote(reference, previous.hypothesis if previous else hypothesis, rationale)
        saved.setdefault(archive.id, archive)
    updated = project.with_results(tuple(saved.values()))
    return _with_state(updated, replace(state, hypothesis=hypothesis, notes=tuple(notes.values())))


def create_variant(
    project: ProjectDocument, record: RunRecord | str | Path, *, hypothesis: str = ""
) -> ProjectDocument:
    """Restore recorded physical settings as a new revision with owned inputs.

    Keeps the same design owner and its evidence history. It does not create an
    independent assembly channel or reuse/modify numerical results. Future stage
    reuse remains subject to the engine's usual checkpoint checks.
    """
    state = design_study_state(project)
    verified = _owned_record(project, record)
    archive = RunArchive.capture(verified)
    files = dict(archive._files)
    request = ResearchRequest.from_dict(json.loads(files["request.json"]))
    if request.project_id != project.id or request.project_revision != archive.metadata.get("project_revision"):
        raise ValueError("Original request and recorded result project identities disagree.")
    if set(request.config) != _CONFIG_FIELDS:
        raise ValueError(
            "The original request does not contain every resolved setting; this legacy run cannot be restored as a reproducible variant."
        )
    manifest = json.loads(files["manifest.json"])
    values = _values(CoilGenConfig(**dict(request.config)))
    # Retain output/display preferences rather than pointing a new revision at
    # historical output. Physical and numerical settings come from the request.
    values.update({key: value for key, value in project.values.items() if config_field_stage(key) is None})
    values["geometry_source_path"] = project.values["geometry_source_path"]
    assets = {}
    inputs = manifest.get("input_files", {})
    for key in _INPUT_FIELDS:
        item = inputs.get(key)
        if item is None:
            # A path input without an archived content identity cannot be
            # restored from wherever the old source happens to exist now.
            value = values[key]
            if isinstance(value, str) and value.startswith("asset:"):
                raise ValueError(f"Original run is missing portable input provenance: {key}.")
            continue
        payload = files[f"portable_inputs/{item['sha256']}"]
        source = str(item["path"])
        kind = (
            "mesh"
            if key.endswith("mesh_file")
            else "target"
            if key == "target_field_definition_file"
            else "stream_function"
        )
        asset = InputAsset(payload, Path(source).suffix, kind, source)
        previous = assets.get(asset.id)
        if previous is not None and (previous.suffix, previous.kind) != (asset.suffix, asset.kind):
            raise ValueError("Recorded inputs use identical bytes with incompatible formats or kinds.")
        assets[asset.id] = asset
        values[key] = "asset:" + asset.id
    saved = {run.id: run for run in project.saved_runs}
    saved.setdefault(archive.id, archive)
    state = replace(
        state, source=RunReference.from_metadata(archive.metadata), hypothesis=_text(hypothesis, "Design hypothesis")
    )
    session = project.session
    session[STUDY_SESSION_KEY] = state.to_dict()
    variant = ProjectDocument(
        project.id,
        project.name,
        project.revision + 1,
        values,
        assets,
        project.project_root,
        _study={key: getattr(request, key) for key in project.study},
        _runs=tuple(saved.values()),
        _session=session,
    )
    DesignService.validate(variant).raise_for_errors()
    return variant
