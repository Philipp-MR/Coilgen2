"""Shared fixtures for application project bundle."""

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.assets import mesh_asset, target_asset
from coilgen.application.bundles import RunArchive
from coilgen.persistence import atomic_json
from _coilgen_test_support.application_backend import _config, _mesh, _target


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
