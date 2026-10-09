"""Engineering contracts of opt-in multichannel and cohort studies."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.array_studies import (
    ArrayCase,
    ArrayChannel,
    ArrayStudySpec,
    CohortControls,
    evaluate_array,
    fit_array_fields,
    synthesize_cohort,
    synthesize_contour_cohort,
)
from coilgen.application.gradient_studies import RigidTransform
from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService, RunRecord
from coilgen.config import CoilGenConfig
from coilgen.gradient_engineering import winding_path_sha256


def _channel(index=0, *, resistance=1.0):
    return ArrayChannel(
        f"c{index}",
        f"Channel {index}",
        f"run{index}",
        "a" * 64,
        {"body": 1.0},
        {"body": str(index) * 64},
        -2.0,
        2.0,
        resistance,
    )


def _case(name="field", desired=(1e-6, -1e-6), **kwargs):
    points = np.column_stack((np.arange(len(desired), dtype=float), np.zeros((len(desired), 2))))
    return ArrayCase(name, points, desired, **kwargs)


def test_registration_rotates_scalar_axis_and_vector_components(tmp_path):
    rotation = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    transform = RigidTransform(rotation, (0.1, -0.2, 0.3))
    source = tmp_path / "field.csv"
    source.write_text("1000,0,0,2,3,4\n0,1000,0,3,4,5", encoding="utf-8")
    case = ArrayCase.from_csv(
        source, length_unit="mm", field_unit="uT", transform=transform, direction=(1, 0, 0), disturbance=False
    )
    np.testing.assert_allclose(case.coordinates_m, [[0.1, 0.8, 0.3], [-0.9, -0.2, 0.3]])
    np.testing.assert_allclose(case.direction, [0, 1, 0])
    np.testing.assert_allclose(case.desired_field_T, [2e-6, 3e-6])
    assert len(case.registration["source_sha256"]) == 64
    scalar = tmp_path / "scalar.csv"
    scalar.write_text("1000 0 0 2", encoding="utf-8")
    scalar_case = ArrayCase.from_csv(
        scalar, length_unit="mm", field_unit="uT", transform=transform, direction=(1, 0, 0)
    )
    np.testing.assert_allclose(scalar_case.desired_field_T, [-2e-6])
    np.testing.assert_allclose(scalar_case.direction, [0, 1, 0])


def test_named_si_header_rejects_non_si_units(tmp_path):
    source = tmp_path / "si.csv"
    source.write_text("x_m,y_m,z_m,B_T\n0,0,0,1e-6", encoding="utf-8")
    with pytest.raises(ValueError, match="Named SI"):
        ArrayCase.from_csv(source, field_unit="uT")


def test_joint_signed_currents_and_combined_field_have_correct_error_sign():
    fields = np.array([[1e-6, 0], [0, 1e-6]])
    case = _case(desired=(1e-6, -1.5e-6), rms_limit_T=1e-12)
    spec = ArrayStudySpec((case,), (_channel(), _channel(1)))
    row = fit_array_fields(spec, {case.name: fields})["cases"][0]
    np.testing.assert_allclose(row["currents_A"], [1, -1.5], atol=1e-9)
    np.testing.assert_allclose(row["combined_field_T"], case.desired_field_T, atol=1e-15)
    assert row["passed"] is True
    assert row["diagnostics"]["rank"] == 2
    assert row["power_W"] == pytest.approx(3.25)


def test_simultaneous_power_bound_and_saturation_are_distinct():
    fields = np.eye(2) * 1e-6
    case = _case(desired=(4e-6, 4e-6), rms_limit_T=1e-6)
    spec = ArrayStudySpec((case,), (_channel(), _channel(1)), total_power_limit_W=1.0)
    row = fit_array_fields(spec, {case.name: fields})["cases"][0]
    assert row["power_W"] <= 1.00001
    np.testing.assert_allclose(row["currents_A"], np.sqrt(0.5), atol=2e-4)
    assert row["passed"] is False
    assert not any(row["saturated_channels"])


def test_frequency_offset_is_explicit_and_conditioning_excludes_it():
    case = _case(desired=(3e-6, 5e-6))
    spec = ArrayStudySpec((case,), (_channel(),), center_frequency=True)
    row = fit_array_fields(spec, {case.name: np.array([[-1e-6], [1e-6]])})["cases"][0]
    assert row["frequency_offset_T"] == pytest.approx(4e-6)
    np.testing.assert_allclose(row["currents_A"], [1.0])
    np.testing.assert_allclose(row["combined_field_T"], [3e-6, 5e-6])
    rank_deficient = fit_array_fields(
        replace(spec, channels=(_channel(), _channel(1))), {case.name: np.ones((2, 2)) * 1e-6}
    )["cases"][0]
    assert rank_deficient["diagnostics"]["rank"] == 0
    assert rank_deficient["diagnostics"]["normalized_condition_number"] is None


def test_spec_roundtrip_is_owned_and_source_identity_is_enforced(tmp_path, monkeypatch):
    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0, 0, 0.0]])
    record = RunRecord(
        tmp_path,
        {
            "request_sha256": "a" * 64,
            "wire_fields": {"routed": {"available": True, "at_stream_current": {"available": True, "current_A": 2.0}}},
        },
        {"evaluated_winding_0_points_m": path},
    )
    monkeypatch.setattr(DesignService, "load_run", lambda *_args, **_kwargs: record)
    import coilgen.application.evidence as evidence

    monkeypatch.setattr(
        evidence, "verified_run_metadata", lambda *_args: ({}, {}, SimpleNamespace(metadata_checked=True))
    )
    channel = ArrayChannel.from_run("c", "Loop", record, body_current_ratios={"evaluated_winding_0": -1.0})
    case = ArrayCase("independent", [[0.05, 0.05, 0.03]], [1e-6])
    spec = ArrayStudySpec((case,), (channel,))
    payload = spec.to_dict()
    restored = ArrayStudySpec.from_dict(payload)
    payload["cases"][0]["coordinates_m"][0][0] = 999
    assert restored.cases[0].coordinates_m[0, 0] == 0.05
    result = evaluate_array(restored)
    assert result["provenance"][0]["complete_circuit"] is False
    assert result["provenance"][0]["path_sha256"]["evaluated_winding_0"] == winding_path_sha256(path)
    changed = replace(channel, request_sha256="d" * 64)
    with pytest.raises(ValueError, match="identity changed"):
        evaluate_array(replace(spec, channels=(changed,)))
    changed = replace(channel, path_sha256={"evaluated_winding_0": "d" * 64})
    with pytest.raises(ValueError, match="geometry changed"):
        evaluate_array(replace(spec, channels=(changed,)))


def test_missing_resistance_cannot_claim_a_power_constraint():
    with pytest.raises(ValueError, match="requires resistance"):
        ArrayStudySpec((_case(),), (_channel(resistance=None),), total_power_limit_W=1.0)
    with pytest.raises(ValueError, match="real numeric"):
        _case(desired=[1j, 2j])


def test_duplicate_physical_conductors_are_not_independent_current_actuators():
    with pytest.raises(ValueError, match="same saved physical conductor"):
        ArrayStudySpec((_case(),), (_channel(), replace(_channel(), id="other")))


def _cohort_project():
    from coilgen.models import Mesh

    xy = np.linspace(-0.1, 0.1, 5)
    vertices = np.array([[x, y, 0] for y in xy for x in xy])
    faces = []
    for y in range(4):
        for x in range(4):
            a = y * 5 + x
            faces.extend(([a, a + 1, a + 6], [a, a + 6, a + 5]))
    config = CoilGenConfig(
        regularization_selection_method="fixed", tikonov_reg_factor=0.1, target_gradient_normalization="none", levels=6
    )
    return ProjectDocument.from_config(config).with_mesh(Mesh(vertices, np.array(faces)), field="coil_mesh_file")


def test_pod_synthesis_uses_training_only_and_exports_bound_mesh_basis():
    from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
    from coilgen.models import CoilPart
    from coilgen.preoptimization import load_preoptimized_stream_function_values

    x = np.array([-0.03, 0.03, -0.03, 0.03])
    y = np.array([-0.03, -0.03, 0.03, 0.03])
    coordinates = np.column_stack((x, y, np.full(4, 0.04)))
    cases = (
        ArrayCase("x", coordinates, x * 1e-4),
        ArrayCase("y", coordinates, y * 1e-4),
        ArrayCase("held", coordinates, (x + y) * 2e-4, "held_out"),
    )
    spec = ArrayStudySpec(cases, cohort_controls=CohortControls(resistance_ohm=(1.0, 2.0), total_power_limit_W=0.5))
    project = _cohort_project()
    result = synthesize_cohort(project, spec, channel_count=2)
    assert result["training_cases"] == ["x", "y"]
    assert result["held_out_cases"] == ["held"]
    assert result["routing_validated"] is False
    assert result["cases"][2]["membership"] == "held_out"
    assert all(row["power_W"] <= 0.50001 for row in result["cases"])
    assert ArrayStudySpec.from_dict(spec.to_dict()).cohort_controls.to_dict() == spec.cohort_controls.to_dict()
    altered = replace(spec, cases=(*cases[:2], replace(cases[2], desired_field_T=(x - y) * 1e3)))
    other = synthesize_cohort(project, altered, channel_count=2)
    for left, right in zip(result["projects"], other["projects"]):
        config = left.config
        mesh = load_mesh_from_config(config, left.project_root)
        parts = [CoilPart(mesh=item) for item in split_disconnected_mesh(mesh)]
        values = load_preoptimized_stream_function_values(config.preoptimized_stream_function_file, parts, config)
        right_config = right.config
        right_values = load_preoptimized_stream_function_values(
            right_config.preoptimized_stream_function_file, parts, right_config
        )
        np.testing.assert_allclose(values, right_values, atol=1e-10)
        assert config.iteration_num_mesh_refinement == 0
        assert config.target_gradient_normalization == "none"
        assert Path(config.preoptimized_stream_function_file).suffix == ".csv"
        assert left.session["cohort_provenance"]["routing_validated"] is False


def test_held_out_training_reuse_is_rejected_before_synthesis():
    case = _case()
    spec = ArrayStudySpec((case, replace(case, name="reused", membership="held_out")))
    with pytest.raises(ValueError, match="duplicates training"):
        synthesize_cohort(_cohort_project(), spec, channel_count=1)


def test_contour_family_partition_reviews_held_out_cases():
    from coilgen.cohort_contours import ContourDictionary, ContourFamily, PhysicalContour

    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0, 0, 0]])
    dictionary = ContourDictionary((ContourFamily("loop", (PhysicalContour(path),)),), "scanner")
    coordinates = np.array([[0.03, 0.03, 0.05], [0.08, 0.05, 0.07], [0.05, 0.08, 0.1]])
    from coilgen.physics import biot_savart_wire_path

    desired = biot_savart_wire_path(path, coordinates)[:, 2]
    spec = ArrayStudySpec(
        (ArrayCase("training", coordinates, desired), ArrayCase("held", coordinates, desired * 1.2, "held_out"))
    )
    result = synthesize_contour_cohort(spec, dictionary, channel_count=1)
    assert result["routing_validated"] is False
    assert result["cases"][1]["membership"] == "held_out"
    np.testing.assert_allclose(result["cases"][1]["currents_A"], [1.2], atol=1e-8)
    np.testing.assert_allclose(result["channels"][0]["families"][0]["contours_m"][0], path)
    controls = CohortControls(
        strategy="contour_dictionary", channel_count=1, contour_dictionary=CohortControls.encode_dictionary(dictionary)
    )
    restored = ArrayStudySpec.from_dict(replace(spec, cohort_controls=controls).to_dict())
    np.testing.assert_allclose(restored.cohort_controls.dictionary().families[0].contours[0].points_m, path)
