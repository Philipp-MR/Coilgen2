"""CAD packages retain saved geometry, units, scientific scope and provenance."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.cad import export_geometry, geometry_catalog
from coilgen.config import CoilGenConfig
from coilgen.geometry import read_stl
from coilgen.manufacturing import _save_xyz_polyline_as_step, save_mesh_as_step, save_mesh_as_stl
from coilgen.models import Mesh
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.storage import result_metadata_digest


BODY = "part001_body0001"


def _solid():
    # Offset closed tetrahedron makes origin/unit/bounds errors visible.
    return Mesh(
        vertices=np.asarray([[0.01, 0.02, 0.03], [0.014, 0.02, 0.03], [0.01, 0.025, 0.03], [0.01, 0.02, 0.036]]),
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )


def _run(tmp_path, *, sheet_accepted=True, ready=True, manufacturing_ok=True, step_unit="m"):
    directory = tmp_path / "saved_run"
    geometry = directory / "fabrication"
    geometry.mkdir(parents=True)
    stl = save_mesh_as_stl(_solid(), geometry / f"{BODY}_fabrication.stl", solid_name=BODY)
    step = save_mesh_as_step(
        _solid(), geometry / f"{BODY}_fabrication_faceted.step", solid_name=BODY, surface_mode="faceted_brep"
    )
    if step_unit == "mm":
        points = _solid()
        points.vertices *= 1000
        save_mesh_as_step(points, step, solid_name=BODY, surface_mode="faceted_brep")
        step.write_text(step.read_text().replace("SI_UNIT($,.METRE.)", "SI_UNIT(.MILLI.,.METRE.)"), encoding="ascii")
    centerline = _save_xyz_polyline_as_step(
        _solid().vertices[:2], geometry / f"{BODY}_fabrication_centroid.step", wire_name=BODY, part_index=1
    )
    support = save_mesh_as_stl(_solid(), directory / "support.stl", solid_name="support")
    arrays = directory / "scientific_result.npz"
    np.savez(arrays, vertices_m=_solid().vertices, current_A=np.asarray([8.5]), faces=_solid().faces)
    request = directory / "request.json"
    atomic_json(request, {"name": "Engineering fixture", "config": {}})
    manifest = directory / "manifest.json"
    atomic_json(manifest, {"request_sha256": "fixture-request", "input_files": {}})
    artifacts = [
        {"name": path.relative_to(directory).as_posix(), "kind": path.suffix.lstrip("."), "sha256": file_digest(path)}
        for path in (stl, step, centerline, support, arrays, request, manifest)
    ]
    result = {
        "request_sha256": "fixture-request",
        "project_id": "fixture-project",
        "project_revision": 4,
        "project_fingerprint": "full-fingerprint",
        "project_model_fingerprint": "model-fingerprint",
        "artifacts": artifacts,
        "summary": {
            "accepted": ready and sheet_accepted,
            "sheet_accepted": sheet_accepted,
            "construction_released": False,
        },
        "pipeline_sweeps": {"checks_passed": manufacturing_ok, "bodies": []},
        "fabrication": {
            "ready": ready,
            "failure_reasons": [] if ready else ["missing_return"],
            "paths": [{"body_id": BODY, "part_index": 1, "diagnostics": {"manufacturing_ok": manufacturing_ok}}],
        },
    }
    result["metadata_sha256"] = result_metadata_digest(result)
    atomic_json(directory / "result.json", result)
    return DesignService.load_run(directory)


def test_catalog_identifies_only_saved_conductor_solids_and_si_bounds(tmp_path):
    record = _run(tmp_path)
    options = geometry_catalog(record)
    assert {option.format for option in options} == {"stl", "step"}
    assert len(options) == 2
    for option in options:
        assert option.body_id == BODY
        assert option.scope == "fabrication_review_geometry"
        assert option.check_status == "passed"
        assert option.source_unit == "m"
        np.testing.assert_allclose(option.bounds_m, [_solid().vertices.min(axis=0), _solid().vertices.max(axis=0)])
    assert not any("centroid" in option.name or "support" in option.name for option in options)


def test_mm_stl_scales_saved_triangles_without_altering_result_or_step(tmp_path):
    record = _run(tmp_path)
    original_arrays = record.arrays
    source_hashes = {item["name"]: item["sha256"] for item in record.metadata["artifacts"]}
    package = export_geometry(record, tmp_path / "cad_package", unit="mm")
    exported = package.manifest["geometry"]
    stl = next(path for path in package.paths if path.suffix == ".stl")
    mesh = read_stl(stl)
    np.testing.assert_allclose(
        [mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)],
        [1000 * _solid().vertices.min(axis=0), 1000 * _solid().vertices.max(axis=0)],
    )
    for row in exported:
        assert row["requested_unit"] == "mm"
        assert row["actual_unit"] == ("mm" if row["format"] == "stl" else "m")
        assert file_digest(package.directory / row["name"]) == row["export_sha256"]
        if row["format"] == "step":
            assert row["export_sha256"] == row["source_sha256"]
            assert row["step_unit_policy"] == "preserved_declared_unit"
    assert package.manifest["project_revision"] == 4
    assert package.manifest["project_model_fingerprint"] == "model-fingerprint"
    assert package.manifest["construction_released"] is False
    assert json.loads(package.manifest_path.read_text()) == package.manifest
    for name, values in original_arrays.items():
        np.testing.assert_array_equal(record.arrays[name], values)
    for name, digest in source_hashes.items():
        assert file_digest(record.directory / name) == digest


def test_step_declared_mm_is_preserved_when_m_requested(tmp_path):
    record = _run(tmp_path, step_unit="mm")
    step = next(option for option in geometry_catalog(record) if option.format == "step")
    assert step.source_unit == "mm"
    np.testing.assert_allclose(step.bounds_m, [_solid().vertices.min(axis=0), _solid().vertices.max(axis=0)])
    package = export_geometry(record, tmp_path / "cad", [step.name], unit="m")
    row = package.manifest["geometry"][0]
    assert row["requested_unit"] == "m" and row["actual_unit"] == "mm"
    assert row["export_sha256"] == step.source_sha256
    np.testing.assert_allclose(row["exported_bounds"], np.asarray(step.bounds_m) * 1000)


@pytest.mark.parametrize("ready,sheet_accepted", [(False, True), (True, False), (False, False)])
def test_failed_review_or_em_checks_allow_explicit_inspection_export(tmp_path, ready, sheet_accepted):
    record = _run(tmp_path, ready=ready, sheet_accepted=sheet_accepted)
    options = geometry_catalog(record)
    assert options and all(option.inspection_only for option in options)
    package = export_geometry(record, tmp_path / "inspection")
    assert package.manifest["export_scope"] == "inspection_export"
    assert package.manifest["fabrication_review_ready"] is ready
    assert package.manifest["sheet_accepted"] is sheet_accepted
    assert package.manifest["construction_released"] is False


def test_export_rechecks_corrupted_source_and_stale_metadata_before_writing(tmp_path):
    record = _run(tmp_path)
    option = geometry_catalog(record)[0]
    (record.directory / option.name).write_bytes(b"corrupted geometry")
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match="integrity"):
        export_geometry(record, destination)
    assert not destination.exists()


def test_selection_rejects_centerlines_unknowns_and_duplicate_names(tmp_path):
    record = _run(tmp_path)
    destination = tmp_path / "rejected"
    for names in (
        [],
        [f"fabrication/{BODY}_fabrication_centroid.step"],
        ["../outside.stl"],
        [geometry_catalog(record)[0].name] * 2,
    ):
        with pytest.raises(ValueError):
            export_geometry(record, destination, names)
    with pytest.raises(ValueError, match="unit"):
        export_geometry(record, destination, unit="inches")
    assert not destination.exists()


def test_existing_destination_is_never_overwritten(tmp_path):
    record = _run(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "owned_by_user.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_geometry(record, destination)
    assert sentinel.read_text() == "preserve"
    assert list(destination.iterdir()) == [sentinel]
    with pytest.raises(FileExistsError):
        export_geometry(record, record.directory / "new_child")


def test_project_companion_is_portable_and_hashed(tmp_path):
    record = _run(tmp_path)
    project = ProjectDocument.from_config(CoilGenConfig(coil_mesh_file="create cylinder mesh"), name="Companion")
    package = export_geometry(record, tmp_path / "complete_package", project=project)
    companion = package.manifest["project_companion"]
    loaded = ProjectDocument.load(package.directory / companion["name"])
    assert loaded.fingerprint == project.fingerprint
    assert companion["fingerprint"] == project.fingerprint
    for artifact in companion["artifacts"]:
        assert file_digest(package.directory / artifact["name"]) == artifact["sha256"]


def test_companion_failure_and_publication_failure_leave_no_committed_package(tmp_path, monkeypatch):
    record = _run(tmp_path)

    class FailingCompanion:
        def save(self, path):
            path.write_text("partial companion", encoding="utf-8")
            raise RuntimeError("companion failed")

    destination = tmp_path / "failed_companion"
    with pytest.raises(RuntimeError, match="companion"):
        export_geometry(record, destination, project=FailingCompanion())
    assert not destination.exists()
    import coilgen.application.cad as cad

    original = cad.os.link
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        return original(source, target)

    monkeypatch.setattr(cad.os, "link", fail_second)
    destination = tmp_path / "failed_publication"
    with pytest.raises(OSError, match="publication"):
        export_geometry(record, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".coilgen-cad-*"))


def test_changed_metadata_requires_reload_even_when_new_checksum_is_valid(tmp_path):
    record = _run(tmp_path)
    metadata = deepcopy(record.metadata)
    metadata["summary"]["sheet_accepted"] = False
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(record.directory / "result.json", metadata)
    with pytest.raises(ValueError, match="changed"):
        geometry_catalog(record)
    assert geometry_catalog(DesignService.load_run(record.directory))


def _replace_artifacts(record, alter):
    metadata = record.metadata
    alter(metadata)
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(record.directory / "result.json", metadata)
    return DesignService.load_run(record.directory)


def test_centerline_mislabeled_as_faceted_step_still_is_not_exportable(tmp_path):
    record = _run(tmp_path)
    source = record.directory / "fabrication" / f"{BODY}_fabrication_centroid.step"
    mislabeled = record.directory / "fabrication" / f"{BODY}_fabrication_faceted.step"
    mislabeled.write_bytes(source.read_bytes())

    def alter(metadata):
        for artifact in metadata["artifacts"]:
            if artifact["name"].endswith("_faceted.step"):
                artifact["sha256"] = file_digest(mislabeled)

    record = _replace_artifacts(record, alter)
    assert [option.format for option in geometry_catalog(record)] == ["stl"]


def test_duplicate_output_filenames_fail_before_publication(tmp_path):
    record = _run(tmp_path)
    source = record.directory / "fabrication" / f"{BODY}_fabrication.stl"
    duplicate = record.directory / "second_directory" / source.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(source.read_bytes())

    def alter(metadata):
        metadata["artifacts"].append(
            {
                "name": duplicate.relative_to(record.directory).as_posix(),
                "kind": "stl",
                "sha256": file_digest(duplicate),
                "body_id": BODY,
            }
        )

    record = _replace_artifacts(record, alter)
    destination = tmp_path / "collision"
    with pytest.raises(ValueError, match="colliding"):
        export_geometry(record, destination)
    assert not destination.exists()


def test_package_manifest_is_the_last_atomic_commit_marker(tmp_path, monkeypatch):
    record = _run(tmp_path)
    import coilgen.application.cad as cad

    original = cad.os.link
    published = []

    def observe(source, target):
        if Path(target).name == "geometry_manifest.json":
            assert all((Path(target).parent / path.name).is_file() for path in published)
        else:
            assert not (Path(target).parent / "geometry_manifest.json").exists()
        original(source, target)
        published.append(Path(target))

    monkeypatch.setattr(cad.os, "link", observe)
    package = export_geometry(record, tmp_path / "atomic")
    assert published[-1] == package.manifest_path
