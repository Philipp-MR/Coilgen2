"""One field catalog, with explicit historical defaults and conservative reuse."""

from dataclasses import asdict, fields, replace

import pytest

from coilgen import CoilGenConfig
from coilgen.config_schema import config_fields, settings_groups, settings_view, PipelineStage
from coilgen.config_migrations import CONFIG_VERSION, migrate_project_config
from coilgen.application import ProjectDocument


def test_catalog_and_typed_groups_cover_the_declared_fields():
    config = CoilGenConfig()
    catalog = config_fields()
    assert set(catalog) == {item.name for item in fields(config)}
    for group, cls in settings_groups().items():
        assert asdict(settings_view(config, cls)) == asdict(getattr(config, group + "_settings"))
        assert set(asdict(settings_view(config, cls))) == {item.name for item in fields(cls)}
    assert catalog["levels"].affected_stage(config) is PipelineStage.CONTOURS
    assert catalog["levels"].affected_stage(replace(config, max_current_A=10)) is PipelineStage.SHEET


@pytest.mark.parametrize("version", [True, -1, CONFIG_VERSION + 1, "1", None])
def test_unknown_config_version_is_rejected(version):
    with pytest.raises(ValueError, match="version"):
        migrate_project_config({}, version=version)


@pytest.mark.parametrize("suffix", [".coilgen.json", ".coilgen"])
def test_versioned_project_roundtrip_preserves_numerical_identity(tmp_path, suffix):
    project = ProjectDocument.from_config(CoilGenConfig(coil_mesh_file="create cylinder mesh"))
    assert project.to_dict()["config_version"] == CONFIG_VERSION
    restored = ProjectDocument.load(project.save(tmp_path / ("design" + suffix)))
    assert restored.fingerprint == project.fingerprint
    assert restored.model_fingerprint == project.model_fingerprint
    assert restored.values == project.values


def test_new_schema_requires_resolved_values_and_legacy_uses_frozen_defaults():
    project = ProjectDocument.from_config(CoilGenConfig(coil_mesh_file="create cylinder mesh"))
    payload = project.to_dict()
    del payload["config"]["contour_smoothing_local_fallback"]
    with pytest.raises(ValueError, match="every resolved"):
        ProjectDocument.from_dict(payload)
    del payload["config_version"]
    restored = ProjectDocument.from_dict(payload)
    assert restored.config.contour_smoothing_local_fallback is False
    assert restored.fingerprint == project.fingerprint
