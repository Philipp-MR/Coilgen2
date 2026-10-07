"""Acceptance metadata integrity and unchanged owned target sample semantics."""

import json
from pathlib import Path

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application import DesignService, ProjectDocument
from coilgen.models import TargetField
from coilgen.object_registry import register_in_memory_target_field
from coilgen.persistence import atomic_json
from coilgen.research.storage import result_metadata_digest, verified_result
from coilgen.target import load_target_field_definition


def _project():
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        target_region_radius=0.045,
        target_region_resolution=8,
        field_shape_function="y",
        progress_flag=False,
        plot_flag=False,
    )
    return ProjectDocument.from_config(config).update_study({"pipeline_stage": "prepared"})


@pytest.mark.parametrize("section,key", [("summary", "accepted"), ("validation", "passed")])
def test_valid_json_acceptance_edit_rejects_new_result_load_and_replay(tmp_path, section, key):
    project = _project()
    directory = tmp_path / "run"
    original = DesignService.run(project, directory)
    assert original["metadata_sha256"] == result_metadata_digest(original)
    assert verified_result(directory, original["request_sha256"]) is not None
    modified = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    modified[section][key] = True
    atomic_json(directory / "result.json", modified)
    before = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert verified_result(directory, original["request_sha256"]) is None
    with pytest.raises(ValueError, match="integrity"):
        DesignService.load_run(directory)
    with pytest.raises(ValueError, match="new output directory"):
        DesignService.run(project, directory)
    assert {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


def test_metadata_digest_is_canonical_strict_and_excludes_its_own_field():
    first = {"summary": {"accepted": False}, "value": np.float64(1.0)}
    reordered = {"value": 1.0, "metadata_sha256": "ignored", "summary": {"accepted": False}}
    assert result_metadata_digest(first) == result_metadata_digest(reordered)
    with pytest.raises(ValueError, match="Nonfinite"):
        result_metadata_digest({"value": float("nan")})


def test_legacy_result_without_metadata_checksum_remains_readable(tmp_path):
    directory = tmp_path / "run"
    original = DesignService.run(_project(), directory)
    historical = {key: value for key, value in original.items() if key != "metadata_sha256"}
    atomic_json(directory / "result.json", historical)
    assert verified_result(directory, original["request_sha256"]) == historical
    assert DesignService.load_run(directory).metadata["summary"]["accepted"] is False


def test_all_zero_target_weights_roundtrip_without_inventing_a_constraint(tmp_path):
    target = TargetField(
        coords=np.asarray([[0.0, 0.0, 0.02], [0.0, 0.01, 0.03]]),
        b=np.asarray([[0.0, 0.0, 1e-5], [0.0, 0.0, 2e-5]]),
        weights=np.zeros(2),
        suppressed=np.asarray([False, True]),
        target_field_group_inds=np.asarray([1, 2]),
    )
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        target_field_definition_file=register_in_memory_target_field(target),
    )
    project = ProjectDocument.from_config(config)
    reopened = ProjectDocument.load(project.save(tmp_path / "project.json"))
    frozen = reopened.config
    assert Path(frozen.target_field_definition_file).is_file()
    loaded = load_target_field_definition(frozen.target_field_definition_file, None, frozen)
    for name in ("coords", "b", "weights", "suppressed", "target_field_group_inds"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(target, name))
    assert loaded.target_gradient_dbdxyz is None
