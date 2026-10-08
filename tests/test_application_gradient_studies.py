"""Studies retain independent axes and measured fields use recorded geometry/current."""

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from coilgen.application.gradient_studies import (
    ConvergenceCriterion,
    MeasuredFieldMap,
    RigidTransform,
    StudyCandidate,
    compare_measured_winding,
    convergence_candidates,
    saved_winding_currents,
    submit_convergence,
    summarize_convergence,
)
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService
from coilgen.geometry import load_mesh_from_config
from coilgen.persistence import atomic_json, file_digest
from coilgen.physics import biot_savart_wire_path
from coilgen.research.storage import result_metadata_digest


def _project():
    return ProjectDocument.from_config(
        replace(preset_config("cylinder"), cylinder_mesh_parameter_list=(0.3, 0.08, 8, 3, 1, 0, 0, 0))
    )


def _run(directory, project=None, *, power=1.0, independent_n=2, currents=(2.0,), complete=True, input_files=None):
    directory.mkdir(parents=True)
    points = np.asarray(((0, 0, 0), (0, 0, 0.03)))
    target = np.asarray(((0, 0, 1e-5), (0, 0, 2e-5)))
    path = np.asarray(((-0.1, -0.1, 0), (0.1, -0.1, 0), (0.1, 0.1, 0), (-0.1, 0.1, 0), (-0.1, -0.1, 0)))
    arrays = {
        "target_points_m": points,
        "target_field_T": target,
        "target_weights": np.ones(2),
        "target_suppressed": np.zeros(2, bool),
        "validation_points_m": points[:independent_n],
        "validation_target_T": target[:independent_n],
    }
    plans, bodies = [], []
    for index, current in enumerate(currents):
        id_ = f"body{index}"
        arrays[f"review_electrical_{index}_points_m"] = path
        arrays[f"evaluated_winding_{index}_points_m"] = path
        plans.append({"body_id": id_, "circuit_complete": complete, "electrical_points_m": path.tolist()})
        bodies.append({"body_id": id_, "current_A": current, "circuit_complete": complete})
    np.savez_compressed(directory / "scientific_result.npz", **arrays)
    atomic_json(directory / "request.json", {"name": "Fixture", "config": project.values if project else {}})
    atomic_json(directory / "manifest.json", {"request_sha256": "fixture", "input_files": input_files or {}})
    metadata = {
        "request_sha256": "fixture",
        "artifacts": [
            {"name": name, "kind": "npz" if name.endswith(".npz") else "json", "sha256": file_digest(directory / name)}
            for name in ("scientific_result.npz", "request.json", "manifest.json")
        ],
        "summary": {"power_w": power, "relative_field_error": 0.01},
        "validation": {"rms_error_T": 1e-6},
        "winding_steps": plans,
        "fabrication": {"electrical_metrics": {"bodies": bodies}},
        "wire_fields": {
            "routed": {
                "available": True,
                "current_A": 9,
                "current_refitted": True,
                "at_stream_current": {"available": True, "current_A": 2, "current_refitted": False},
            }
        },
    }
    if project:
        metadata.update(
            project_id=project.id, project_revision=project.revision, project_fingerprint=project.fingerprint
        )
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(directory / "result.json", metadata)
    return DesignService.load_run(directory)


def test_candidates_change_only_explicit_axis_keep_units_constraints_and_portable_inputs(tmp_path):
    project = _project()
    before = project.to_dict()
    candidates = convergence_candidates(
        project, mesh_refinements=(0, 1, 2), quadrature_orders=(2, 3), independent_resolutions=(12, 20)
    )
    assert len(candidates) == 5
    assert len({candidate.project.id for candidate in candidates}) == len(candidates)
    assert project.to_dict() == before
    for candidate in candidates:
        deltas = {key for key in project.values if project.values[key] != candidate.project.values[key]}
        study_deltas = {key for key in project.study if project.study[key] != candidate.project.study[key]}
        assert deltas == (
            {"iteration_num_mesh_refinement"}
            if candidate.axis == "mesh"
            else {"gauss_order"}
            if candidate.axis == "quadrature"
            else set()
        )
        assert study_deltas == ({"validation_resolution"} if candidate.axis == "independent_sampling" else set())
        restored = StudyCandidate.from_dict(candidate.to_dict(), project_root=tmp_path)
        assert restored.project.fingerprint == candidate.project.fingerprint
        np.testing.assert_allclose(
            load_mesh_from_config(restored.project.config, tmp_path).vertices,
            load_mesh_from_config(project.config, tmp_path).vertices,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"mesh_refinements": (4,)},
        {"quadrature_orders": (0,)},
        {"independent_resolutions": (True,)},
        {"quadrature_orders": (3, 3)},
        {"mesh_refinements": (3,), "max_mesh_faces": 100},
        {"mesh_refinements": (1, 2), "max_candidates": 2},
        {},
    ),
)
def test_studies_are_explicitly_bounded(kwargs):
    with pytest.raises(ValueError):
        convergence_candidates(_project(), **kwargs)


def test_submission_uses_existing_service_and_cancels_partial_queue_failure():
    candidates = convergence_candidates(_project(), quadrature_orders=(3,))

    class Jobs:
        cancelled = []

        def cancel(self, id_):
            self.cancelled.append(id_)

    class Service:
        count = 0

        @staticmethod
        def preflight(project):
            return DesignService.preflight(project)

        @classmethod
        def submit(cls, project, jobs):
            cls.count += 1
            if cls.count == 2:
                raise ValueError("queue full")
            return {"id": project.id}

    jobs = Jobs()
    with pytest.raises(ValueError, match="queue full"):
        submit_convergence(candidates, jobs, service=Service)
    assert jobs.cancelled == [candidates[0].project.id]


def test_summary_reverifies_scope_reports_change_and_excludes_independent_sampling_claims(tmp_path):
    candidates = convergence_candidates(_project(), quadrature_orders=(3, 4), independent_resolutions=(20, 24))
    records = {
        candidate.project.id: _run(
            tmp_path / str(index),
            candidate.project,
            power=(1, 1.001, 1.002, 1, 1)[index],
            independent_n=1 if candidate.axis == "independent_sampling" else 2,
        )
        for index, candidate in enumerate(candidates)
    }
    report = summarize_convergence(candidates, records)
    assert report.stable_axes == ("quadrature",)
    assert all(change.within_tolerance is True for change in report.changes if change.axis == "quadrature")
    rms = summarize_convergence(
        candidates, records, criteria=(ConvergenceCriterion("Sheet independent RMS error", absolute_tolerance=1),)
    )
    assert all(
        not change.comparable and change.within_tolerance is None
        for change in rms.changes
        if change.axis == "independent_sampling" and change.earlier == "Baseline"
    )
    strict = summarize_convergence(candidates, records, criteria=(ConvergenceCriterion("Sheet power", 1e-6),))
    assert strict.stable_axes == ()
    assert report.to_dict()["schema"] == "coilgen-convergence-study-v1"
    artifact = records[candidates[1].project.id].directory / "scientific_result.npz"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        summarize_convergence(candidates, records)


def test_wrong_revision_and_incomplete_studies_cannot_be_reported_as_converged(tmp_path):
    candidates = convergence_candidates(_project(), quadrature_orders=(3,))
    with pytest.raises(ValueError, match="missing"):
        summarize_convergence(candidates, {})
    records = {
        candidate.project.id: _run(tmp_path / str(i), candidate.project) for i, candidate in enumerate(candidates)
    }
    wrong = _run(tmp_path / "wrong", candidates[1].project.rename("Different revision"))
    records[candidates[1].project.id] = wrong
    with pytest.raises(ValueError, match="revision"):
        summarize_convergence(candidates, records)


def test_measured_square_loop_matches_analytic_field_and_preserves_signed_body_currents(tmp_path):
    record = _run(tmp_path / "saved", currents=(4.0, -1.0))
    expected = 4e-7 * np.sqrt(2) / 0.1 * 3
    measurement = MeasuredFieldMap(np.asarray(((0, 0, 0),)), np.asarray(((0, 0, expected),)), independent=True)
    report = compare_measured_winding(record, measurement, rms_limit_T=1e-15)
    assert report.passed is True
    assert report.provenance["source"] == "fabricated_complete_electrical_paths"
    assert report.provenance["recorded_currents_A"] == {"body0": 4, "body1": -1}
    assert report.rms_error_T < 1e-19 and report.peak_error_T < 1e-19
    assert report.provenance["current_refitted"] is False
    assert saved_winding_currents(record)["body_currents_A"] == {"body0": 4, "body1": -1}
    with pytest.raises(ValueError, match="every"):
        compare_measured_winding(record, measurement, body_currents_A={"body0": 3})
    calibrated = compare_measured_winding(record, measurement, body_currents_A={"body0": 4, "body1": -1.5})
    assert calibrated.provenance["recorded_currents_A"] == report.provenance["recorded_currents_A"]
    assert calibrated.predicted_T[0, 2] == pytest.approx(expected * 2.5 / 3)


def test_registration_rotates_field_vectors_converts_units_and_does_not_fit(tmp_path):
    record = _run(tmp_path / "saved", currents=(2,))
    rotation = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    transform = RigidTransform(rotation, (0, 0, 0.03))
    world_points = np.asarray(((0, 0, 0), (0, 0, 0.05)))
    path = record.array("review_electrical_0_points_m")
    world_field = 2 * biot_savart_wire_path(path, world_points)
    raw_points = (world_points - transform.translation_m) @ rotation
    raw_field = world_field @ rotation
    measured = MeasuredFieldMap(raw_points * 1000, raw_field * 1e6, "mm", "uT", True)
    result = compare_measured_winding(record, measured, transform=transform, rms_limit_T=1e-15)
    np.testing.assert_allclose(result.points_m, world_points, atol=1e-17)
    np.testing.assert_allclose(result.measured_T, world_field, atol=1e-19)
    assert result.rms_error_T < 1e-19 and result.passed
    with pytest.raises(ValueError, match="proper rigid"):
        RigidTransform(((-1, 0, 0), (0, 1, 0), (0, 0, 1)))
    with pytest.raises(ValueError, match="proper rigid"):
        RigidTransform(((2, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_measured_csv_retains_original_bytes_and_portable_roundtrip(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text("x_m,y_m,z_m,Bx_T,By_T,Bz_T\n0,0,.02,0,0,2e-5\n", encoding="utf-8")
    measured = MeasuredFieldMap.from_file(path, independent=True)
    original = path.read_bytes()
    path.write_text("changed", encoding="utf-8")
    assert measured.source_bytes == original
    for suffix in ("json", "npz"):
        loaded = MeasuredFieldMap.from_file(measured.save(tmp_path / f"portable.{suffix}"))
        np.testing.assert_array_equal(loaded.field, measured.field)
        assert loaded.source_bytes == original and loaded.independent
        assert loaded.data_sha256 == measured.data_sha256
    with pytest.raises(ValueError):
        measured.field[0, 0] = 1
    data = measured.to_dict()
    data["field"][0][0] = 1
    with pytest.raises(ValueError, match="integrity"):
        MeasuredFieldMap.from_dict(data)


def test_training_input_and_singular_samples_are_not_measured_validation(tmp_path):
    source = tmp_path / "fitted.csv"
    source.write_text("x_m,y_m,z_m,Bx_T,By_T,Bz_T\n0,0,.02,0,0,2e-5\n", encoding="utf-8")
    measured = MeasuredFieldMap.from_file(source, independent=True)
    record = _run(
        tmp_path / "saved",
        input_files={
            "target_field_definition_file": {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
        },
    )
    with pytest.raises(ValueError, match="design input"):
        compare_measured_winding(record, measured)
    plain = _run(tmp_path / "plain")
    with pytest.raises(ValueError, match="explicitly independent"):
        compare_measured_winding(plain, replace(measured, independent=False))
    on_wire = MeasuredFieldMap(np.asarray(((0.1, 0, 0),)), np.zeros((1, 3)), independent=True)
    with pytest.raises(ValueError, match="centerline"):
        compare_measured_winding(plain, on_wire)


def test_partial_winding_is_descriptive_and_uses_selected_recorded_current(tmp_path):
    record = _run(tmp_path / "partial", complete=False)
    measurement = MeasuredFieldMap(np.zeros((1, 3)), np.zeros((1, 3)), independent=True)
    original = compare_measured_winding(record, measurement, peak_limit_T=1)
    stored = compare_measured_winding(record, measurement, current_policy="recorded")
    assert original.passed is None and not original.provenance["complete_circuit"]
    np.testing.assert_allclose(stored.predicted_T, 4.5 * original.predicted_T)
    assert original.relative_rms_error is None
    json.dumps(original.to_dict(), allow_nan=False)
