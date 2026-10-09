"""Software credits remain separate from physical identity and saved authorship."""

import json

import pytest

from coilgen import CoilGenConfig
from coilgen.application import DesignService, ProjectDocument
from coilgen.attribution import software_attribution
from coilgen.research import storage


def _project(stage="prepared"):
    return ProjectDocument.from_config(
        CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.2, 0.12, 8, 4, 1, 0, 0, 0),
            field_shape_function="y",
            target_region_radius=0.035,
            target_region_resolution=3,
            levels=4,
            min_point_loop_number=3,
            gauss_order=1,
            regularization_selection_method="fixed",
            tikonov_reg_factor=20.0,
            progress_flag=False,
            plot_flag=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance_calculation=True,
        ),
        name="Independent researcher design",
    ).update_study({"pipeline_stage": stage, "validation_resolution": 3})


def test_software_credit_changes_do_not_change_request_identity(monkeypatch):
    original_digest, original_manifest = storage.request_identity({"name": "Physical case"}, {})
    altered = {**software_attribution(), "creator": "Changed software attribution"}
    monkeypatch.setattr(storage, "software_attribution", lambda: dict(altered))
    new_digest, new_manifest = storage.request_identity({"name": "Physical case"}, {})

    assert original_digest == new_digest
    assert original_manifest["software"] == software_attribution()
    assert new_manifest["software"] == altered
    assert {key: value for key, value in original_manifest.items() if key != "software"} == {
        key: value for key, value in new_manifest.items() if key != "software"
    }


@pytest.mark.parametrize("stage", ["prepared", "evaluated"])
def test_new_run_evidence_retains_software_credit_under_its_metadata_checksum(tmp_path, stage):
    directory = tmp_path / stage
    project = _project(stage)
    result = DesignService.run(project, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert result["software"] == manifest["software"] == software_attribution()
    assert result["name"] == project.name
    assert "author" not in result and "designer" not in result
    assert result["metadata_sha256"] == storage.result_metadata_digest(result)
    assert DesignService.load_run(directory).metadata["software"] == software_attribution()
    if stage == "evaluated":
        assert "Created and maintained by Philipp Amrein" in (directory / "design_report.md").read_text(
            encoding="utf-8"
        )


def test_replaying_saved_run_never_rewrites_its_recorded_software_credit(tmp_path, monkeypatch):
    project = _project()
    directory = tmp_path / "saved"
    original = DesignService.run(project, directory)
    saved = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        storage, "software_attribution", lambda: {**software_attribution(), "creator": "Changed attribution"}
    )

    replayed = DesignService.run(project, directory)

    assert replayed == original
    assert {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == saved
