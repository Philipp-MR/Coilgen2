from __future__ import annotations

from dataclasses import asdict
import base64
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from coilgen import (
    CoilDesignProblem,
    CoilGenConfig,
    CoilSupportSpec,
    Mesh,
    SupportMesh,
    TargetFieldSpec,
)
from coilgen.application import DesignService, ProjectDocument, PROJECT_SCHEMA, dependency_plan
from coilgen.application import ChannelDocument, ProjectAssembly
from coilgen.application.assets import InputAsset, mesh_asset, target_asset
from coilgen.geometry import load_mesh_from_config
from coilgen.models import TargetField
from coilgen.object_registry import (
    register_in_memory_mesh,
    register_in_memory_target_field,
)
from coilgen.persistence import atomic_json, file_digest
from coilgen.target import load_target_field_definition


def _mesh(scale=1.0):
    return Mesh(
        vertices=scale * np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        faces=np.asarray([[0, 1, 2]]),
        metadata={"units": "SI", "name": "support"},
    )


def _target():
    return TargetField(
        coords=np.asarray([[0.0, 0.0, 0.02], [0.0, 0.01, 0.03]]),
        b=np.asarray([[1e-5, 2e-5, 3e-5], [0.0, 0.0, 0.0]]),
        weights=np.asarray([1.0, 0.25]),
        suppressed=np.asarray([False, True]),
        target_field_group_inds=np.asarray([1, 7]),
        target_gradient_dbdxyz=None,
        suppression_normals=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        suppression_area_weights=np.asarray([0.0, 0.004]),
        target_diagnostics={"source": "engineer", "nested": {"frame": "world"}},
    )


def _config():
    return CoilGenConfig(
        coil_mesh_file=register_in_memory_mesh(_mesh()),
        target_field_definition_file=register_in_memory_target_field(_target()),
        imported_mesh_length_scale_to_m=0.001,
        research_physics={"objective": "power", "objective_weights": {"power": 1.0, "minimax": 2.0}},
    )


def test_save_reopen_relocation_preserves_full_numeric_inputs(tmp_path):
    project = ProjectDocument.from_config(_config(), name="Portable")
    path = project.save(tmp_path / "original" / "design.coilgen.json")
    moved = tmp_path / "relocated"
    shutil.copytree(path.parent, moved)
    reopened = ProjectDocument.load(moved / path.name)
    assert reopened.id == project.id
    assert reopened.revision == project.revision
    assert reopened.fingerprint == project.fingerprint
    config = reopened.config
    loaded_mesh = load_mesh_from_config(config, moved)
    np.testing.assert_array_equal(loaded_mesh.vertices, _mesh().vertices)
    np.testing.assert_array_equal(loaded_mesh.faces, _mesh().faces)
    target = load_target_field_definition(config.target_field_definition_file, None, config)
    for field in (
        "coords",
        "b",
        "weights",
        "suppressed",
        "target_field_group_inds",
        "suppression_normals",
        "suppression_area_weights",
    ):
        np.testing.assert_array_equal(getattr(target, field), getattr(_target(), field))
    assert target.target_gradient_dbdxyz is None
    assert target.target_diagnostics == _target().target_diagnostics


def test_input_mutation_and_config_mutation_do_not_change_project(tmp_path):
    original = _config()
    project = ProjectDocument.from_config(original)
    original.research_physics["objective_weights"]["power"] = 999
    detached = project.config
    detached.research_physics["objective_weights"]["minimax"] = 999
    assert project.config.research_physics["objective_weights"] == {"power": 1.0, "minimax": 2.0}
    changed = project.update_config({"levels": project.config.levels + 1})
    assert changed.id == project.id and changed.revision == 2
    assert project.revision == 1 and project.config.levels == 10
    assert changed.update_config({"levels": changed.config.levels}) is changed
    with pytest.raises(ValueError, match="Unknown"):
        project.update_config({"imaginary_solver": True})


def test_same_name_problems_are_isolated():
    def problem(scale):
        return CoilDesignProblem(
            support=CoilSupportSpec("single_mesh", (SupportMesh(_mesh(scale), role="primary"),)),
            target_field=TargetFieldSpec.from_arrays(np.asarray([[0.0, 0.0, 0.02]]), np.asarray([scale * 1e-5])),
        )

    first, second = problem(1), problem(2)
    assert first.metadata.name == second.metadata.name
    config1, config2 = first.to_config(), second.to_config()
    assert config1.coil_mesh_file != config2.coil_mesh_file
    assert config1.target_field_definition_file != config2.target_field_definition_file
    np.testing.assert_array_equal(load_mesh_from_config(config1, Path.cwd()).vertices, _mesh().vertices)
    np.testing.assert_array_equal(load_mesh_from_config(config2, Path.cwd()).vertices, _mesh(2).vertices)


def test_project_freezes_external_target_before_source_changes(tmp_path):
    source = tmp_path / "target.csv"
    source.write_text("x,y,z,bz\n0,0,.01,1e-5\n", encoding="utf-8")
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        target_field_definition_file="target.csv",
        target_gradient_normalization="off",
    )
    project = ProjectDocument.from_config(config, project_root=tmp_path)
    before = project.fingerprint
    source.write_text("x,y,z,bz\n0,0,.01,9e-5\n", encoding="utf-8")
    frozen = project.config
    target = load_target_field_definition(frozen.target_field_definition_file, None, frozen)
    assert target.b[0, 2] == 1e-5
    assert project.fingerprint == before
    refreshed = ProjectDocument.from_config(config, project_root=tmp_path)
    assert refreshed.fingerprint != before


def test_schema_unknown_fields_and_corrupt_assets_are_rejected(tmp_path):
    path = ProjectDocument.from_config(_config()).save(tmp_path / "project.json")
    value = json.loads(path.read_text())
    value["schema"] = "coilgen-project-v999"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="schema"):
        ProjectDocument.load(path)
    value["schema"] = PROJECT_SCHEMA
    value["config"]["new_unknown_property"] = 1
    atomic_json(path, value)
    with pytest.raises(ValueError, match="Unknown"):
        ProjectDocument.load(path)
    del value["config"]["new_unknown_property"]
    atomic_json(path, value)
    asset = next(iter(value["assets"].values()))
    asset["data"] = base64.b64encode(b"modified").decode("ascii")
    atomic_json(path, value)
    with pytest.raises(ValueError, match="integrity"):
        ProjectDocument.load(path)


def test_snapshot_is_portable_and_detached_from_future_revision(tmp_path):
    project = ProjectDocument.from_config(_config())
    request = project.snapshot(tmp_path / "run", validation_resolution=8)
    payload = request.to_dict()
    assert payload["project_fingerprint"] == project.fingerprint
    assert payload["config"]["coil_mesh_file"].endswith(".npz")
    assert not payload["config"]["coil_mesh_file"].startswith("memory:")
    changed = project.update_config({"levels": 17})
    assert payload["config"]["levels"] == 10 and changed.config.levels == 17
    assert Path(payload["config"]["coil_mesh_file"]).is_file()


def test_exported_script_compiles_and_project_reopens(tmp_path):
    from coilgen.attribution import AUTHOR_NAME

    project = ProjectDocument.new(config=CoilGenConfig(coil_mesh_file="create cylinder mesh"))
    path = project.export_python(tmp_path / "design.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    reopened = ProjectDocument.load(path.with_suffix(".coilgen.json"))
    assert reopened.fingerprint == project.fingerprint
    assert "run_coilgen" in path.read_text(encoding="utf-8")
    assert f"Software creator and maintainer: {AUTHOR_NAME}" in path.read_text(encoding="utf-8")


def test_dependency_plan_keeps_wire_and_hardware_constraints_upstream():
    before = CoilGenConfig(coil_mesh_file="create cylinder mesh")
    for key, value in (
        ("max_wire_current_A", 30.0),
        ("min_wire_distance_m", 0.002),
        ("conductor_cross_section_width", 0.004),
        ("conductor_thickness", 0.001),
    ):
        after = CoilGenConfig(**{**asdict(before), key: value})
        plan = dependency_plan(before, after)
        assert plan.earliest_stage in {"prepared", "sheet"}, key
        assert "sheet" in plan.stages
    assert not dependency_plan(before, CoilGenConfig(**{**asdict(before), "plot_flag": True})).requires_computation


def test_load_run_verifies_artifacts_and_returns_detached_arrays(tmp_path):
    array = tmp_path / "scientific_result.npz"
    np.savez(array, vertices_m=_mesh().vertices)
    manifest = {"request_sha256": "known", "input_files": {}}
    atomic_json(tmp_path / "manifest.json", manifest)
    artifacts = [
        {"name": path.name, "kind": "npz" if path == array else "provenance", "sha256": file_digest(path)}
        for path in (array, tmp_path / "manifest.json")
    ]
    atomic_json(
        tmp_path / "result.json", {"request_sha256": "known", "artifacts": artifacts, "summary": {"accepted": False}}
    )
    record = DesignService.load_run(tmp_path)
    data = record.arrays
    data["vertices_m"][0] = 100
    assert record.arrays["vertices_m"][0, 0] == 0
    metadata = record.metadata
    metadata["summary"]["accepted"] = True
    assert record.metadata["summary"]["accepted"] is False
    array.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        DesignService.load_run(tmp_path)


def test_study_controls_are_owned_persisted_and_used_by_snapshot(tmp_path):
    project = ProjectDocument.from_config(_config())
    controls = {"pipeline_stage": "sheet", "validation_resolution": 16, "design_margin": 0.9}
    changed = project.update_study(controls)
    controls["validation_resolution"] = 2
    assert changed.revision == 2
    assert changed.fingerprint != project.fingerprint
    assert changed.model_fingerprint == project.model_fingerprint
    assert changed.study["validation_resolution"] == 16
    detached = changed.study
    detached["validation_resolution"] = 2
    assert changed.study["validation_resolution"] == 16
    restored = ProjectDocument.load(changed.save(tmp_path / "study.json"))
    assert restored.study == changed.study
    request = restored.snapshot(tmp_path / "snapshot")
    assert request.pipeline_stage == "sheet"
    assert request.validation_resolution == 16
    assert request.design_margin == 0.9
    assert request.project_id == changed.id
    assert request.project_revision == changed.revision
    assert changed.update_study({"pipeline_stage": "sheet"}) is changed
    for changes in (
        {"pipeline_stage": "unknown"},
        {"validation_resolution": True},
        {"design_margin": float("nan")},
        {"design_margin": "x"},
        {"pipeline_stage": "sheet", "fabricate": True},
        {"resume_checkpoint": "old.npz"},
    ):
        with pytest.raises(ValueError):
            changed.update_study(changes)


def test_rename_does_not_change_input_identity_and_internals_are_readonly():
    project = ProjectDocument.from_config(_config())
    renamed = project.rename("Gradient Gy")
    assert renamed.revision == 2 and project.name == "Coil design"
    assert renamed.fingerprint == project.fingerprint
    assert renamed.rename("Gradient Gy") is renamed
    with pytest.raises(ValueError, match="name"):
        project.rename("")
    with pytest.raises(TypeError):
        project._values["levels"] = 100
    with pytest.raises(TypeError):
        project._values["research_physics"]["objective_weights"]["power"] = 100
    with pytest.raises(AttributeError):
        project._assets.clear()
    detached = project.values
    detached["levels"] = 100
    assert project.config.levels == 10
    changed_display = project.update_config({"plot_flag": not project.config.plot_flag, "field_display_units": "Hz"})
    assert changed_display.model_fingerprint == project.model_fingerprint
    assert changed_display.fingerprint != project.fingerprint


def test_physical_edits_fail_before_committing_new_revision():
    from coilgen.application.presets import preset_config

    project = ProjectDocument.from_config(preset_config())
    for changes in (
        {"target_region_radius": -1},
        {"conductor_thickness": -1},
        {"levels": 0},
        {"research_physics": {"objective": "unsupported"}},
    ):
        with pytest.raises(ValueError):
            project.update_config(changes)
    assert project.revision == 1


def test_owned_asset_content_cannot_be_mutated_or_escape_directory(tmp_path):
    source = bytearray(b"source bytes")
    asset = InputAsset(source, ".NPZ", "mesh")
    source[:] = b"changed data"
    assert asset.data == b"source bytes"
    assert asset.suffix == ".npz"
    assert asset.write(tmp_path).parent == tmp_path
    for suffix in ("../escape", ".npz/../../escape", ".", ""):
        with pytest.raises(ValueError):
            InputAsset(b"x", suffix, "mesh")
    with pytest.raises(ValueError):
        InputAsset(b"x", ".npz", "unknown")


def test_assets_validate_array_shape_weights_and_distinct_mesh_vertices():
    target = _target()
    target.weights = np.ones((2, 2))
    with pytest.raises(ValueError, match="weights"):
        target_asset(target)
    target = _target()
    target.weights[0] = -1
    with pytest.raises(ValueError, match="weights"):
        target_asset(target)
    target = _target()
    target.target_gradient_dbdxyz = np.ones((2, 4))
    with pytest.raises(ValueError, match="gradient"):
        target_asset(target)
    mesh = _mesh()
    mesh.faces = np.asarray([[0, 0, 1]])
    with pytest.raises(ValueError, match="distinct"):
        mesh_asset(mesh)


def test_project_load_rejects_duplicate_properties_and_external_references(tmp_path):
    project = ProjectDocument.from_config(_config())
    path = project.save(tmp_path / "project.json")
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace('"revision": 1,', '"revision": 1, "revision": 2,'), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        ProjectDocument.load(path)
    value = json.loads(original)
    value["config"]["coil_mesh_file"] = "some-external-file.stl"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="owned"):
        ProjectDocument.load(path)
    with pytest.raises(ValueError, match="revision"):
        project.snapshot(tmp_path / "bad-snapshot", project_revision=999)


def test_save_is_atomic_when_final_file_replace_fails(tmp_path, monkeypatch):
    from coilgen import persistence

    project = ProjectDocument.from_config(_config())
    path = project.save(tmp_path / "project.json")
    before = path.read_bytes()

    def failed_replace(*_args):
        raise OSError("Simulated replace failure")

    monkeypatch.setattr(persistence.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        project.rename("New name").save(path)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_owned_mesh_retains_explicit_normals_and_project_fingerprint_is_stable(tmp_path):
    from coilgen.geometry import read_mesh_npz

    mesh = _mesh()
    mesh.vertex_normals = np.tile([0.0, 0.0, -1.0], (mesh.n_vertices, 1))
    mesh.face_normals = np.asarray([[0.0, 0.0, -1.0]])
    asset = mesh_asset(mesh)
    loaded = read_mesh_npz(asset.write(tmp_path))
    np.testing.assert_array_equal(loaded.vertex_normals, mesh.vertex_normals)
    np.testing.assert_array_equal(loaded.face_normals, mesh.face_normals)
    # Independent source registries cannot change canonical content identity.
    assert ProjectDocument.from_config(_config()).fingerprint == ProjectDocument.from_config(_config()).fingerprint


def test_run_record_copies_direct_constructor_arrays_and_metadata(tmp_path):
    from coilgen.application import RunRecord

    arrays = {"field_T": np.asarray([[1.0, 2.0, 3.0]])}
    metadata = {"constraints": {"accepted": False}}
    record = RunRecord(tmp_path, metadata, arrays)
    arrays["field_T"][0] = 99
    metadata["constraints"]["accepted"] = True
    np.testing.assert_array_equal(record.arrays["field_T"], [[1.0, 2.0, 3.0]])
    assert record.metadata["constraints"]["accepted"] is False
    with pytest.raises(ValueError):
        record._arrays["field_T"][0] = 0


def test_duplicate_preserves_owned_source_provenance_after_source_removal(tmp_path):
    source = mesh_asset(_mesh()).write(tmp_path / "source")
    config = _config()
    config.coil_mesh_file = str(source)
    project = ProjectDocument.from_config(config, name="Original").update_study({"pipeline_stage": "sheet"})
    source.unlink()
    duplicate = project.duplicate("Independent")
    assert duplicate.id != project.id
    assert duplicate.revision == 1
    assert duplicate.fingerprint == project.fingerprint
    assert duplicate.study == project.study
    assert [(asset.source, asset.data) for asset in duplicate._assets.values()] == [
        (asset.source, asset.data) for asset in project._assets.values()
    ]
    np.testing.assert_array_equal(load_mesh_from_config(duplicate.config, tmp_path).vertices, _mesh().vertices)
    changed = duplicate.update_config({"levels": project.config.levels + 1})
    assert changed.config.levels != project.config.levels


def test_old_project_schema_without_study_loads_explicit_default_controls(tmp_path):
    path = ProjectDocument.from_config(_config()).save(tmp_path / "project.json")
    value = json.loads(path.read_text())
    del value["study"]
    atomic_json(path, value)
    restored = ProjectDocument.load(path)
    assert restored.study == {
        "pipeline_stage": "evaluated",
        "validation_resolution": 12,
        "design_margin": 1.0,
        "fabricate": False,
        "kernel_backend": "auto",
    }


def test_service_preflight_rejects_incomplete_draft_without_writing_run(tmp_path):
    project = ProjectDocument.new()
    report = DesignService.validate(project)
    assert not report.ok
    with pytest.raises(ValueError, match="No coil surface"):
        DesignService.prepare(project, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_channel_collection_edits_are_immutable_and_roundtrip_after_relocation(tmp_path):
    first = ProjectDocument.from_config(_config(), name="Gx")
    second = ProjectDocument.from_config(_config(), name="Gy")
    assembly = ProjectAssembly.new("Gradient set", (first, second))
    first_id, second_id = (channel.id for channel in assembly.channels)
    changed = assembly.update_channel(first_id, project=first.update_config({"levels": 12}))
    changed = changed.update_channel(second_id, enabled=False)
    assert assembly.channel(first_id).project.config.levels == 10
    assert changed.channel(first_id).project.config.levels == 12
    assert changed.revision == assembly.revision + 2
    assert changed.channel(second_id).enabled is False
    assert changed.update_channel(second_id, enabled=False) is changed
    original = changed.save(tmp_path / "original" / "channels.coilgen.json")
    shutil.copytree(original.parent, tmp_path / "moved")
    reopened = ProjectAssembly.load(tmp_path / "moved" / original.name)
    assert reopened.fingerprint == changed.fingerprint
    assert [channel.id for channel in reopened.channels] == [first_id, second_id]
    np.testing.assert_array_equal(
        load_mesh_from_config(reopened.channel(first_id).project.config, tmp_path).vertices, _mesh().vertices
    )
    assert len(changed.remove_channel(first_id).channels) == 1
    with pytest.raises(ValueError, match="unique"):
        changed.add_channel(first, name="gy")
    with pytest.raises(ValueError, match="unique"):
        ProjectAssembly("set", "Set", 1, (changed.channels[0], changed.channels[0]))


def test_channel_collection_save_preserves_older_collection_revisions(tmp_path):
    assembly = ProjectAssembly.new("Channels", (ProjectDocument.from_config(_config(), name="Gx"),))
    old_path = assembly.save(tmp_path / "old.json")
    channel = assembly.channels[0]
    updated = assembly.update_channel(
        channel.id, project=channel.project.rename("Renamed").update_config({"levels": 12})
    )
    new_path = updated.save(tmp_path / "new.json")
    assert ProjectAssembly.load(old_path).channels[0].project.config.levels == 10
    assert ProjectAssembly.load(new_path).channels[0].project.config.levels == 12
    value = json.loads(new_path.read_text())
    value["channels"][0]["project"]["revision"] = 999
    atomic_json(new_path, value)
    with pytest.raises(ValueError, match="integrity"):
        ProjectAssembly.load(new_path)


def test_channel_collection_adapter_freezes_existing_script_project():
    from coilgen import CoilProject, CylinderCoilDesign

    script_project = CoilProject("Three axes", template=CylinderCoilDesign())
    script_project.add_channel("Gx", target_bz_expression="x")
    script_project.add_channel("Gy", target_bz_expression="y", enabled=False)
    assembly = ProjectAssembly.from_coil_project(script_project)
    assert [channel.name for channel in assembly.channels] == ["Gx", "Gy"]
    assert assembly.channels[1].enabled is False
    assert assembly.channels[0].project.config.field_shape_function == "x"
    script_project.channels.clear()
    assert len(assembly.channels) == 2


def test_collection_submission_disables_channels_and_cancels_partial_queue(tmp_path):
    class Jobs:
        directory = tmp_path

        def __init__(self, capacity):
            self.requests = []
            self.cancelled = []
            self.capacity = capacity

        def submit(self, request):
            if len(self.requests) >= self.capacity:
                raise ValueError("Queue full")
            self.requests.append(request)
            return {"id": str(len(self.requests)), "status": "queued"}

        def cancel(self, id_):
            self.cancelled.append(id_)

    first = ProjectDocument.from_config(_config(), name="Gx")
    second = ProjectDocument.from_config(_config(), name="Gy")
    assembly = ProjectAssembly.new("Set", (first, second))
    jobs = Jobs(capacity=2)
    disabled = assembly.update_channel(assembly.channels[1].id, enabled=False)
    submitted = DesignService.submit_assembly(disabled, jobs)
    assert list(submitted) == [assembly.channels[0].id]
    assert len(jobs.requests) == 1
    jobs = Jobs(capacity=1)
    with pytest.raises(ValueError, match="full"):
        DesignService.submit_assembly(assembly, jobs)
    assert jobs.cancelled == ["1"]


def test_portable_identifiers_reject_reserved_device_names_and_path_components():
    project = ProjectDocument.from_config(_config())
    for id_ in ("CON", "nul", "LPT1", "../escape", "", "ü", "x" * 121):
        with pytest.raises(ValueError, match="identifier"):
            ProjectDocument(id_, project.name, 1, project.values, project._assets)
        with pytest.raises(ValueError, match="identifier"):
            ChannelDocument(id_, "Gx", project)


def _fake_executor(monkeypatch, *, accepted):
    from coilgen.research import application
    from coilgen.research.storage import request_identity

    executed = []

    def execute(request, directory, *, threads=1):
        executed.append(request)
        config = request.to_config(threads=threads)
        digest, manifest = request_identity(request.to_dict(), asdict(config))
        manifest["request_sha256"] = digest
        atomic_json(directory / "request.json", request.to_dict())
        atomic_json(directory / "manifest.json", manifest)
        np.savez(directory / "scientific_result.npz", field_T=np.asarray([[1e-5, 2e-5, 3e-5]]))
        artifacts = [
            {"name": name, "kind": kind, "sha256": file_digest(directory / name)}
            for name, kind in (
                ("request.json", "request"),
                ("manifest.json", "provenance"),
                ("scientific_result.npz", "npz"),
            )
        ]
        result = {
            "request_sha256": digest,
            "artifacts": artifacts,
            "summary": {"accepted": accepted, "scope": "continuous_sheet", "construction_released": False},
            "validation": {"verified": accepted, "passed": True if accepted else None},
        }
        atomic_json(directory / "result.json", result)
        return result

    monkeypatch.setattr(application, "execute_owned_design", execute)
    return executed


def _directory_bytes(path):
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file() and file.name != ".owner.lock"
    }


@pytest.mark.parametrize("accepted", [True, False])
def test_service_replays_verified_history_without_execution_or_any_file_changes(tmp_path, monkeypatch, accepted):
    executed = _fake_executor(monkeypatch, accepted=accepted)
    project = ProjectDocument.from_config(_config()).update_study({"pipeline_stage": "sheet"})
    directory = tmp_path / "run"
    original = DesignService.run(project, directory)
    before = _directory_bytes(directory)
    replayed = DesignService.run(project.rename("Display name change"), directory)
    assert replayed == original
    assert len(executed) == 1
    assert replayed["summary"]["accepted"] is accepted
    assert replayed["validation"]["verified"] is accepted
    assert _directory_bytes(directory) == before
    replayed["summary"]["accepted"] = not accepted
    assert json.loads((directory / "result.json").read_text())["summary"]["accepted"] is accepted
    with pytest.raises(ValueError, match="new output directory"):
        DesignService.run(project.update_config({"levels": 11}), directory)
    assert _directory_bytes(directory) == before
    assert len(executed) == 1


def test_service_preserves_history_when_solver_environment_or_engine_source_changes(tmp_path, monkeypatch):
    from coilgen.research import storage

    _fake_executor(monkeypatch, accepted=True)
    project = ProjectDocument.from_config(_config())
    directory = tmp_path / "run"
    DesignService.run(project, directory)
    before = _directory_bytes(directory)
    original_environment = storage.environment_manifest()
    changed_environment = {**original_environment, "packages": {**original_environment["packages"], "clarabel": "new"}}
    monkeypatch.setattr(storage, "environment_manifest", lambda: changed_environment)
    with pytest.raises(ValueError, match="new output directory"):
        DesignService.run(project, directory)
    assert _directory_bytes(directory) == before
    monkeypatch.setattr(storage, "environment_manifest", lambda: original_environment)
    monkeypatch.setattr(storage, "numerical_source_digest", lambda _: "changed numerical source")
    with pytest.raises(ValueError, match="new output directory"):
        DesignService.run(project, directory)
    assert _directory_bytes(directory) == before


def test_service_replay_checks_owned_input_and_artifact_integrity_without_repairing_history(tmp_path, monkeypatch):
    _fake_executor(monkeypatch, accepted=False)
    project = ProjectDocument.from_config(_config())
    directory = tmp_path / "run"
    DesignService.run(project, directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    source = Path(next(iter(manifest["input_files"].values()))["path"])
    source.write_bytes(b"corrupt input")
    before = _directory_bytes(directory)
    with pytest.raises(RuntimeError, match="changed during execution"):
        DesignService.run(project, directory)
    assert _directory_bytes(directory) == before
    directory = tmp_path / "artifact-run"
    DesignService.run(project, directory)
    (directory / "scientific_result.npz").write_bytes(b"corrupt result")
    before = _directory_bytes(directory)
    with pytest.raises(ValueError, match="integrity verification"):
        DesignService.run(project, directory)
    assert _directory_bytes(directory) == before


def test_service_uses_the_legacy_directory_lock_and_runs_real_prepared_stage(tmp_path):
    from coilgen.research.storage import DirectoryOwner

    project = ProjectDocument.from_config(_config()).update_study({"pipeline_stage": "prepared"})
    directory = tmp_path / "run"
    with DirectoryOwner(directory):
        with pytest.raises(RuntimeError, match="already owns"):
            DesignService.run(project, directory)
    assert not (directory / "request.json").exists()
    result = DesignService.run(project, directory)
    assert result["pipeline_stage"] == "prepared"
    assert result["summary"]["scope"] == "prepared_geometry"
    assert result["summary"]["accepted"] is False
    assert result["project_id"] == project.id
    before = _directory_bytes(directory)
    assert DesignService.run(project, directory) == result
    assert _directory_bytes(directory) == before
