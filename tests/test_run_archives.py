"""Portable completed runs retain exact evidence, inputs and engineering scope."""

import base64
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application import DesignService, ProjectDocument
from coilgen.application.bundles import RunArchive
from coilgen.models import Mesh, TargetField
from coilgen.object_registry import register_in_memory_mesh, register_in_memory_target_field
from coilgen.research.storage import verify_input_files


@pytest.fixture
def completed(tmp_path):
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
        faces=np.array([[0, 1, 2]]),
        metadata={"units": "SI"},
    )
    target = TargetField(
        coords=np.array([[0, 0, 0.02], [0, 0.01, 0.03]]),
        b=np.array([[0, 0, 1e-5], [0, 0, 2e-5]]),
        weights=np.array([1.0, 0.25]),
        suppressed=np.array([False, False]),
        target_field_group_inds=np.array([1, 2]),
    )
    config = CoilGenConfig(
        coil_mesh_file=register_in_memory_mesh(mesh),
        target_field_definition_file=register_in_memory_target_field(target),
        target_gradient_normalization="off",
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        plot_flag=False,
        progress_flag=False,
    )
    project = ProjectDocument.from_config(config, name="Portable evidence").update_study({"pipeline_stage": "prepared"})
    original = tmp_path / "original"
    DesignService.run(project, original / "run", threads=1)
    record = DesignService.load_run(original / "run")
    (record.directory / "job.json").write_text('{"status":"completed","id":"old-live-id"}', encoding="utf-8")
    return project, DesignService.load_run(record.directory), original


def test_one_project_json_relocates_owned_inputs_results_and_exact_artifacts(completed, tmp_path):
    project, record, original = completed
    archive = RunArchive.capture(record, alias="original-ui-selection")
    original_bytes = {name: payload for name, payload in archive._files}
    expected_arrays, expected_metadata = record.arrays, record.metadata
    expected_metadata.pop("job", None)
    manifest = json.loads(original_bytes["manifest.json"])
    assert len(manifest["input_files"]) == 2
    saved = project.with_results((archive,), {"selected_run": archive.alias}).save(original / "design.coilgen.json")
    moved = tmp_path / "relocated"
    moved.mkdir()
    only_file = moved / saved.name
    shutil.copyfile(saved, only_file)
    shutil.rmtree(original)
    assert all(not Path(item["path"]).exists() for item in manifest["input_files"].values())
    assert list(moved.iterdir()) == [only_file]
    reopened = ProjectDocument.load(only_file)
    assert reopened.fingerprint == project.fingerprint and reopened.revision == project.revision
    recovered = reopened.saved_runs[0].materialize(tmp_path / "private-cache")
    assert recovered.metadata == expected_metadata
    for name, values in expected_arrays.items():
        np.testing.assert_array_equal(recovered.arrays[name], values)
    for name, payload in original_bytes.items():
        assert (recovered.directory / name).read_bytes() == payload
    assert not (recovered.directory / "job.json").exists()
    assert recovered.metadata["summary"]["construction_released"] is False
    assert recovered.metadata["summary"]["scope"] == "prepared_geometry"
    assert reopened.session["selected_run"] == archive.alias


def test_archive_roundtrip_is_detached_content_addressed_and_reusable(completed, tmp_path):
    project, record, original = completed
    archive = RunArchive.capture(record, alias="ui-id")
    assert RunArchive.capture(record).alias == "Portable evidence"
    assert archive.project_id == project.id
    with pytest.raises(FrozenInstanceError):
        archive.alias = "changed"
    value = archive.to_dict()
    restored = RunArchive.from_dict(json.loads(json.dumps(value)))
    assert restored == archive
    assert "job.json" not in value["files"]
    metadata = archive.metadata
    assert "job" not in metadata
    original_summary = dict(metadata["summary"])
    metadata["summary"]["accepted"] = True
    metadata["request_sha256"] = "modified"
    assert archive.metadata["summary"] == original_summary
    assert archive.metadata["request_sha256"] == record.metadata["request_sha256"]
    hashes = {name: item["sha256"] for name, item in value["files"].items()}
    assert archive.id == sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    value["files"]["request.json"]["data"] = "modified"
    assert RunArchive.from_dict(archive.to_dict()) == archive
    shutil.rmtree(original)
    recovered = restored.materialize(tmp_path / "cache")
    assert restored.materialize(tmp_path / "cache").directory == recovered.directory
    recaptured = RunArchive.capture(recovered, alias="other-alias")
    assert recaptured.id == archive.id and recaptured.alias == "other-alias"


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        ".",
        "/absolute",
        "a\\b",
        "C:drive",
        "a/../b",
        "a//b",
        "a/./b",
        "CON.txt",
        "CONIN$",
        "COM¹.txt",
        "lpt1/file",
        "trailing.",
        "spaces ",
        "job.json",
    ],
)
def test_unsafe_archive_file_names_rejected_before_materialization(completed, name):
    archive = RunArchive.capture(completed[1])
    value = archive.to_dict()
    value["files"][name] = value["files"].pop("request.json")
    with pytest.raises(ValueError, match="safe|unsafe|job"):
        RunArchive.from_dict(value)


@pytest.mark.parametrize("change", ["base64", "hash", "identity", "required"])
def test_tampered_archive_rejected(completed, change):
    value = RunArchive.capture(completed[1]).to_dict()
    if change == "base64":
        value["files"]["request.json"]["data"] += "\n"
    elif change == "hash":
        value["files"]["request.json"]["sha256"] = "0" * 64
    elif change == "identity":
        value["id"] = "0" * 64
    else:
        del value["files"]["manifest.json"]
        hashes = {name: item["sha256"] for name, item in value["files"].items()}
        value["id"] = sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError):
        RunArchive.from_dict(value)


def test_corrupt_existing_materialization_is_never_repaired_or_overwritten(completed, tmp_path):
    archive = RunArchive.capture(completed[1])
    record = archive.materialize(tmp_path / "cache")
    array = record.directory / "scientific_result.npz"
    array.write_bytes(b"user-modified-content")
    before = {
        path.relative_to(record.directory): path.read_bytes() for path in record.directory.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="integrity"):
        archive.materialize(tmp_path / "cache")
    after = {
        path.relative_to(record.directory): path.read_bytes() for path in record.directory.rglob("*") if path.is_file()
    }
    assert after == before


def test_bad_portable_input_never_falls_back_to_an_unchanged_original(completed, tmp_path):
    archive = RunArchive.capture(completed[1])
    recovered = archive.materialize(tmp_path / "cache")
    manifest = json.loads((recovered.directory / "manifest.json").read_bytes())
    item = next(iter(manifest["input_files"].values()))
    assert Path(item["path"]).is_file()
    (recovered.directory / "portable_inputs" / item["sha256"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        DesignService.load_run(recovered.directory)
    # Runtime verification still checks original absolute inputs.
    verify_input_files(manifest)


def test_archive_metadata_tampering_fails_even_after_envelope_hash_is_recomputed(completed):
    value = RunArchive.capture(completed[1]).to_dict()
    result = json.loads(base64.b64decode(value["files"]["result.json"]["data"]))
    result["summary"]["accepted"] = True
    payload = json.dumps(result).encode()
    value["files"]["result.json"] = {
        "encoding": "base64",
        "sha256": sha256(payload).hexdigest(),
        "data": base64.b64encode(payload).decode(),
    }
    hashes = {name: item["sha256"] for name, item in value["files"].items()}
    value["id"] = sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="metadata integrity"):
        RunArchive.from_dict(value)
