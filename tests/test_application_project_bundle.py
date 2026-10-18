"""Single JSON relocation retains exact scientific inputs and immutable results."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import PurePosixPath
import shutil

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectAssembly, ProjectDocument
from coilgen.application.assets import InputAsset, mesh_asset, project_reference_path, target_asset
from coilgen.application.bundles import RunArchive
from coilgen.application.project import PROJECT_SCHEMA
from coilgen.application.assembly import ASSEMBLY_SCHEMA
from coilgen.geometry import load_mesh_from_config
from coilgen.persistence import atomic_json, file_digest
from coilgen.target import load_target_field_definition

from test_application_backend import _config, _mesh, _target


_UNSAFE_LEGACY_REFERENCES = (
    "../outside.npz",
    r"..\outside.npz",
    "C:/outside/input.npz",
    r"C:\outside\input.npz",
    "C:input.npz",
    r"\outside\input.npz",
    r"\\server\share\input.npz",
    "//server/share/input.npz",
    r"\\?\C:\outside\input.npz",
    "/outside/input.npz",
)


def _legacy_project(project, path):
    descriptors = {}
    for id_, asset in project._assets.items():
        owned = asset.write(path.parent / "old-assets")
        descriptors[id_] = {
            "path": owned.relative_to(path.parent).as_posix(),
            "sha256": id_,
            "kind": asset.kind,
            "source": asset.source,
        }
    value = {
        "schema": "coilgen-project-v1",
        "id": project.id,
        "name": project.name,
        "revision": project.revision,
        "config": project.values,
        "assets": descriptors,
        "study": project.study,
    }
    atomic_json(path, value)
    return value


def _prepared_document(tmp_path):
    sources = tmp_path / "sources"
    config = _config()
    config.coil_mesh_file = str(mesh_asset(_mesh()).write(sources))
    config.target_field_definition_file = str(target_asset(_target()).write(sources))
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "prepared"})
    directory = tmp_path / "runtime"
    DesignService.run(project, directory, threads=1)
    record = DesignService.load_run(directory)
    archive = RunArchive.capture(record, alias="prepared_baseline")
    return project, record, archive


def test_copy_only_json_then_remove_original_inputs_and_runs_preserves_exact_arrays(tmp_path):
    project, record, archive = _prepared_document(tmp_path)
    scientific_arrays = record.arrays
    original_manifest = (record.directory / "manifest.json").read_bytes()
    original_result = (record.directory / "result.json").read_bytes()
    project = project.with_results(
        (archive,),
        session={"selected_run": archive.alias, "view": {"dataset": "target", "units": "µT"}},
    )
    path = project.save(tmp_path / "original" / "design.coilgen.json")
    assert list(path.parent.iterdir()) == [path]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == PROJECT_SCHEMA
    assert all("data" in value and "path" not in value for value in payload["assets"].values())
    moved = tmp_path / "moved" / path.name
    moved.parent.mkdir()
    shutil.copyfile(path, moved)
    shutil.rmtree(tmp_path / "sources")
    shutil.rmtree(tmp_path / "runtime")
    shutil.rmtree(path.parent)

    reopened = ProjectDocument.load(moved)
    assert reopened.id == project.id
    assert reopened.revision == project.revision
    assert reopened.fingerprint == project.fingerprint
    assert reopened.session == project.session
    mesh = load_mesh_from_config(reopened.config, moved.parent)
    np.testing.assert_array_equal(mesh.vertices, _mesh().vertices)
    np.testing.assert_array_equal(mesh.faces, _mesh().faces)
    config = reopened.config
    target = load_target_field_definition(config.target_field_definition_file, None, config)
    for key in (
        "coords",
        "b",
        "weights",
        "suppressed",
        "target_field_group_inds",
        "suppression_normals",
        "suppression_area_weights",
    ):
        np.testing.assert_array_equal(getattr(target, key), getattr(_target(), key))
        assert getattr(target, key).dtype == getattr(_target(), key).dtype
    restored_record = reopened.saved_runs[0].materialize(tmp_path / "restored-cache")
    assert set(restored_record.arrays) == set(scientific_arrays)
    for name, data in scientific_arrays.items():
        np.testing.assert_array_equal(restored_record.arrays[name], data)
        assert restored_record.arrays[name].dtype == data.dtype
    assert (restored_record.directory / "manifest.json").read_bytes() == original_manifest
    assert (restored_record.directory / "result.json").read_bytes() == original_result
    assert restored_record.metadata == record.metadata
    assert list(moved.parent.iterdir()) == [moved]


def test_result_and_session_attachments_do_not_change_scientific_identity(tmp_path):
    project, _, archive = _prepared_document(tmp_path)
    session = {"view": {"dataset": "stream_function"}, "selected_run": archive.alias}
    saved = project.with_results((archive,), session=session)
    session["view"]["dataset"] = "edited"
    assert saved.id == project.id
    assert saved.revision == project.revision
    assert saved.fingerprint == project.fingerprint
    assert saved.model_fingerprint == project.model_fingerprint
    assert saved.saved_runs == (archive,)
    assert saved.session["view"]["dataset"] == "stream_function"
    detached = saved.session
    detached["view"]["dataset"] = "changed"
    assert saved.session["view"]["dataset"] == "stream_function"
    with pytest.raises(TypeError):
        saved._session["view"]["dataset"] = "changed"
    for changed in (
        saved.rename("Renamed baseline"),
        saved.update_study({"validation_resolution": 14}),
        saved.update_config({"levels": saved.config.levels + 1}),
    ):
        assert changed.saved_runs == (archive,)
        assert changed.session == saved.session
    duplicate = saved.duplicate("Independent design")
    assert duplicate.saved_runs == ()
    assert duplicate.session == {}
    assert duplicate.fingerprint == saved.fingerprint
    with pytest.raises(ValueError, match="different project owner"):
        duplicate.with_results((archive,))
    with pytest.raises(ValueError, match="unique"):
        saved.with_results((archive, archive))


def test_independent_channels_embed_inputs_and_results_in_one_json(tmp_path):
    project, _, archive = _prepared_document(tmp_path)
    project = project.with_results((archive,), session={"selected_run": archive.alias})
    other = project.duplicate("Independent channel")
    assembly = ProjectAssembly.new("Two designs", (project, other))
    path = assembly.save(tmp_path / "original" / "channels.json")
    assert list(path.parent.iterdir()) == [path]
    assert isinstance(json.loads(path.read_text())["channels"][0]["project"], dict)
    moved = tmp_path / "moved" / path.name
    moved.parent.mkdir()
    shutil.copyfile(path, moved)
    shutil.rmtree(path.parent)
    shutil.rmtree(tmp_path / "sources")
    shutil.rmtree(tmp_path / "runtime")
    restored = ProjectAssembly.load(moved)
    assert restored.fingerprint == assembly.fingerprint
    assert restored.channels[0].project.saved_runs[0].alias == archive.alias
    assert restored.channels[1].project.saved_runs == ()
    assert restored.channels[0].project.session == project.session
    for channel in restored.channels:
        np.testing.assert_array_equal(
            load_mesh_from_config(channel.project.config, moved.parent).vertices, _mesh().vertices
        )
    arrays = restored.channels[0].project.saved_runs[0].materialize(tmp_path / "cache").arrays
    assert "vertices_m" in arrays
    detached = restored.to_dict()
    detached["channels"][0]["project"]["config"]["levels"] += 1
    with pytest.raises(ValueError, match="integrity"):
        ProjectAssembly.from_dict(detached)


def test_legacy_project_and_collection_import_resave_as_single_file(tmp_path):
    project = ProjectDocument.from_config(_config())
    old_path = tmp_path / "legacy" / "project.json"
    _legacy_project(project, old_path)
    loaded = ProjectDocument.load(old_path)
    assert loaded.fingerprint == project.fingerprint
    output = loaded.save(tmp_path / "new" / "project.json")
    assert list(output.parent.iterdir()) == [output]
    assert ProjectDocument.load(output).fingerprint == project.fingerprint
    value = {
        "schema": "coilgen-independent-channels-v1",
        "id": "assembly-id",
        "name": "Legacy channel",
        "revision": 1,
        "units": "SI",
        "coordinate_frame": "world",
        "channels": [
            {
                "id": "channel-id",
                "name": "A",
                "enabled": True,
                "project": old_path.name,
                "sha256": file_digest(old_path),
            }
        ],
    }
    old_assembly_path = old_path.parent / "assembly.json"
    atomic_json(old_assembly_path, value)
    old_assembly = ProjectAssembly.load(old_assembly_path)
    migrated = old_assembly.save(tmp_path / "new-assembly" / "assembly.json")
    assert list(migrated.parent.iterdir()) == [migrated]
    assert json.loads(migrated.read_text())["schema"] == ASSEMBLY_SCHEMA
    assert ProjectAssembly.load(migrated).fingerprint == old_assembly.fingerprint


@pytest.mark.parametrize("reference", _UNSAFE_LEGACY_REFERENCES)
def test_legacy_asset_paths_cannot_escape_project_directory(tmp_path, reference):
    project = ProjectDocument.from_config(_config())
    path = tmp_path / "legacy" / "project.json"
    value = _legacy_project(project, path)
    next(iter(value["assets"].values()))["path"] = reference
    atomic_json(path, value)
    with pytest.raises(ValueError, match="inside the project directory"):
        ProjectDocument.load(path)


@pytest.mark.parametrize("reference", [value for value in _UNSAFE_LEGACY_REFERENCES if "\\" in value or ":" in value])
def test_windows_legacy_references_are_rejected_even_when_posix_considers_them_relative(tmp_path, reference):
    assert not PurePosixPath(reference).is_absolute()
    with pytest.raises(ValueError, match="inside the project directory"):
        project_reference_path(tmp_path, reference, error="Reference must stay inside the project directory.")


@pytest.mark.parametrize("separator", ["/", "\\"])
@pytest.mark.parametrize("prefix", ["", "./", "old-assets/../"])
def test_safe_legacy_asset_references_preserve_relative_directory_meaning(tmp_path, separator, prefix):
    project = ProjectDocument.from_config(_config())
    path = tmp_path / "legacy" / "project.json"
    value = _legacy_project(project, path)
    for descriptor in value["assets"].values():
        descriptor["path"] = (prefix + descriptor["path"]).replace("/", separator)
    atomic_json(path, value)

    loaded = ProjectDocument.load(path)

    assert loaded.fingerprint == project.fingerprint
    assert loaded._assets == project._assets


@pytest.mark.parametrize("reference", _UNSAFE_LEGACY_REFERENCES)
def test_legacy_collection_project_references_cannot_escape_directory(tmp_path, reference):
    value = {
        "schema": "coilgen-independent-channels-v1",
        "id": "assembly-id",
        "name": "Legacy channel",
        "revision": 1,
        "units": "SI",
        "coordinate_frame": "world",
        "channels": [{"id": "channel-id", "name": "A", "enabled": True, "project": reference, "sha256": "0" * 64}],
    }

    with pytest.raises(ValueError, match="inside the collection directory"):
        ProjectAssembly.from_dict(value, project_root=tmp_path)


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_legacy_collection_preserves_safe_relative_subdirectory_references(tmp_path, separator):
    project = ProjectDocument.from_config(_config())
    path = project.save(tmp_path / "channels" / "project.json")
    value = {
        "schema": "coilgen-independent-channels-v1",
        "id": "assembly-id",
        "name": "Legacy channel",
        "revision": 1,
        "units": "SI",
        "coordinate_frame": "world",
        "channels": [
            {
                "id": "channel-id",
                "name": "A",
                "enabled": True,
                "project": "channels/project.json".replace("/", separator),
                "sha256": file_digest(path),
            }
        ],
    }

    loaded = ProjectAssembly.from_dict(value, project_root=tmp_path)

    assert loaded.channels[0].project.fingerprint == project.fingerprint


def test_inline_assets_reject_bad_base64_digest_and_external_path_descriptors():
    asset = InputAsset(b"bytes", ".npz", "mesh", "provenance")
    restored = InputAsset.from_dict(asset.to_dict())
    assert restored == asset
    for key, value in (("data", "%%%"), ("sha256", "0" * 64), ("encoding", "pickle"), ("suffix", ".npz/escape")):
        descriptor = deepcopy(asset.to_dict())
        descriptor[key] = value
        with pytest.raises(ValueError):
            InputAsset.from_dict(descriptor)
    descriptor = asset.to_dict()
    descriptor["path"] = "external.npz"
    with pytest.raises(ValueError, match="descriptor"):
        InputAsset.from_dict(descriptor)


@pytest.mark.parametrize("value", [None, [], {}, {"schema": "unknown"}])
def test_public_from_dict_rejects_malformed_document_shapes(value):
    with pytest.raises(ValueError, match="schema"):
        ProjectDocument.from_dict(value)
    with pytest.raises(ValueError, match="schema"):
        ProjectAssembly.from_dict(value)


def test_v2_channel_cannot_reintroduce_legacy_external_input_references(tmp_path):
    project = ProjectDocument.from_config(_config())
    path = tmp_path / "legacy" / "project.json"
    legacy = _legacy_project(project, path)
    value = ProjectAssembly.new("Set", (project,)).to_dict()
    value["channels"][0]["project"] = legacy
    value["channels"][0]["sha256"] = sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="embedded"):
        ProjectAssembly.from_dict(value, project_root=path.parent)


def test_empty_headless_owner_archives_attach_without_rewriting_scientific_evidence(tmp_path):
    from coilgen.config import CoilGenConfig
    from coilgen.research.application import run_design
    from coilgen.research.requests import ResearchRequest

    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.16, 8, 4, 1, 0, 0, 0),
        target_region_radius=0.05,
        target_region_resolution=5,
    )
    request = ResearchRequest.from_config(config, name="Headless preparation", pipeline_stage="prepared")
    assert request.project_id == "" and request.project_revision is None
    directory = tmp_path / "headless"
    run_design(request, directory)
    record = DesignService.load_run(directory)
    original_result = (directory / "result.json").read_bytes()
    original_manifest = (directory / "manifest.json").read_bytes()
    original_metadata = record.metadata
    original_arrays = record.arrays
    assert original_metadata["project_id"] == ""
    archive = RunArchive.capture(record)
    assert archive.project_id is None
    assert archive.metadata == original_metadata
    project = ProjectDocument.from_config(config).with_results((archive,))
    restored = ProjectDocument.load(project.save(tmp_path / "attached.coilgen.json"))
    assert restored.saved_runs[0].project_id is None
    assert restored.saved_runs[0].metadata == original_metadata
    replay = restored.saved_runs[0].materialize(tmp_path / "restored-evidence")
    assert (replay.directory / "result.json").read_bytes() == original_result
    assert (replay.directory / "manifest.json").read_bytes() == original_manifest
    assert replay.metadata == original_metadata
    for name, value in original_arrays.items():
        np.testing.assert_array_equal(replay.arrays[name], value)
