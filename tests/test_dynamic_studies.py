"""Circuit source contracts, finite duty cycles and separately scoped coupled evidence."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.dynamic_studies import (
    ANALYSIS_SCHEMA,
    CircuitReference,
    DynamicStudySpec,
    ExternalAnalysis,
    import_circuit_reference,
    import_external_analysis,
    import_waveform_csv,
    review_dynamic_analyses,
    verified_circuit_references,
    waveform_digest,
)
from coilgen.gradient_engineering import (
    CircuitSpec,
    DriveSpec,
    GradientEngineeringSpec,
    InductanceMatrixSpec,
    ThermalSpec,
    evaluate_gradient_engineering,
    expand_drive_waveform,
    winding_path_sha256,
)


def _circuit(body):
    points = [[0, 0, 0], [0.01, 0, 0]]
    return CircuitSpec(
        body,
        winding_path_sha256(points),
        [[-0.01, -0.01, 0], points[0]],
        [points[-1], [-0.01, -0.01, 0]],
        "Recorded complete route",
    )


def _engineering(*, matrix=False, thermal=False, repetitions=1, rest_s=0):
    circuits = (_circuit("primary"), _circuit("shield"))
    drive = DriveSpec(
        ("primary", "shield"),
        (10, 10),
        (100, 100),
        "Explicit signed pulse",
        time_s=(0, 1, 2),
        currents_A=((0, 0), (2, -1), (0, 0)),
        repetitions=repetitions,
        rest_s=rest_s,
        inductance_matrix=InductanceMatrixSpec(
            ("primary", "shield"), ((2, 0.5), (0.5, 3)), ("e" * 64, "f" * 64), "External circuit model"
        )
        if matrix
        else None,
        thermal=ThermalSpec(1, 2, 293.15, 293.15, 333.15, "Measured RC fixture", True) if thermal else None,
    )
    return GradientEngineeringSpec(circuits, drive)


def _fabrication():
    return SimpleNamespace(
        winding_steps=[
            {
                "body_id": body,
                "electrical_path_sha256": digest,
                "expected_winding_path_sha256": _circuit(body).expected_path_sha256,
                "drive_feed_designed": True,
            }
            for body, digest in (("primary", "e" * 64), ("shield", "f" * 64))
        ],
        electrical_metrics={
            "bodies": [
                {"body_id": "primary", "resistance_ohm": 1, "inductance_H": 2},
                {"body_id": "shield", "resistance_ohm": 1, "inductance_H": 3},
            ]
        },
    )


def _reference():
    return CircuitReference(
        "inductance",
        ("primary", "shield"),
        ("e" * 64, "f" * 64),
        ((2, 0.5), (0.5, 3)),
        "External circuit model",
        "a" * 64,
        "Independent FEM export",
    )


def _analysis(**changes):
    values = dict(
        name="Conducting sleeve response",
        kind="frequency_response",
        body_ids=("primary", "shield"),
        path_sha256=("e" * 64, "f" * 64),
        material_sha256="a" * 64,
        environment_sha256="c" * 64,
        source="External response samples",
        source_sha256="b" * 64,
        solver="Reference FEM",
        solver_version="1.0",
        points_m=((0, 0, 0.02), (0, 0, 0.03)),
        axis=(100, 1000),
        axis_unit="Hz",
        quantity="field_response",
        unit="T/A",
        real=(((3, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 1))),
        imaginary=(((0, 4, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0))),
        limit=4.9,
        excitation_body_id="primary",
    )
    values.update(changes)
    return ExternalAnalysis(**values)


def test_import_converts_units_and_binds_actual_source_digest(tmp_path):
    filename = tmp_path / "matrix.json"
    payload = {
        "kind": "inductance",
        "body_ids": ["primary", "shield"],
        "path_sha256": ["e" * 64, "f" * 64],
        "values": [[2000, 500], [500, 3000]],
        "unit": "mH",
        "source": "External circuit model",
        "method": "FEM",
    }
    filename.write_text(json.dumps(payload))
    reference = import_circuit_reference(filename)
    np.testing.assert_allclose(reference.values, [[2, 0.5], [0.5, 3]])
    assert reference.source_sha256 == sha256(filename.read_bytes()).hexdigest()
    assert CircuitReference.from_dict(reference.to_dict()) == reference
    changed = reference.to_dict()
    changed["values"][0][0] = 5
    with pytest.raises(ValueError, match="digest"):
        CircuitReference.from_dict(changed)


@pytest.mark.parametrize("matrix", [[[1, 2], [2, 1]], [[1, 0.3], [0.1, 1]]])
def test_reference_rejects_non_psd_and_asymmetric_matrix(matrix):
    with pytest.raises(ValueError, match="symmetric"):
        replace(_reference(), values=matrix)


def test_mutual_reference_is_used_only_for_exact_complete_ordered_paths():
    engineering = _engineering(matrix=True)
    fabrication = _fabrication()
    spec = DynamicStudySpec(inductance_reference=_reference())
    references = verified_circuit_references(spec, fabrication, engineering)
    evaluated = evaluate_gradient_engineering(engineering, fabrication, **references)
    assert evaluated["drive"]["mutual_inductance_included"]
    # RI + L*dI/dt: first rising ramp end is (5.5, -3); second end is (-3.5, 2).
    np.testing.assert_allclose(evaluated["waveform"]["interval_voltage_end_V"][0], [5.5, -3])
    stale = replace(_reference(), path_sha256=("e" * 64, "d" * 64))
    assert verified_circuit_references(DynamicStudySpec(inductance_reference=stale), fabrication, engineering) == {}
    fabrication.winding_steps[0]["drive_feed_designed"] = False
    assert verified_circuit_references(spec, fabrication, engineering) == {}


def test_efficiency_enables_target_drive_after_matching_source_and_geometry():
    reference = CircuitReference(
        "efficiency",
        ("primary", "shield"),
        ("e" * 64, "f" * 64),
        (0.01, -0.02),
        "Independent path regression",
        "a" * 64,
        "Bz gradient regression",
        field_component="Bz",
        gradient_direction_world=(0, 0, 1),
    )
    engineering = _engineering()
    target = replace(
        engineering.drive,
        time_s=(),
        currents_A=(),
        target_gradient_T_per_m=0.04,
        target_slew_T_per_m_per_s=0.08,
        gradient_efficiencies_T_per_m_per_A=reference.values,
        gradient_efficiency_path_sha256=reference.path_sha256,
        gradient_efficiency_source_sha256=reference.source_sha256,
        gradient_efficiency_source=reference.source,
    )
    engineering = replace(engineering, drive=target)
    references = verified_circuit_references(
        DynamicStudySpec(efficiency_reference=reference), _fabrication(), engineering
    )
    result = evaluate_gradient_engineering(engineering, _fabrication(), **references)
    assert result["drive"]["available"]
    np.testing.assert_allclose([row["peak_current_A"] for row in result["drive"]["bodies"]], [4, 2])


def test_repeated_pulses_preserve_rest_and_thermal_initial_condition():
    engineering = _engineering(thermal=True, repetitions=2, rest_s=3)
    time, currents = expand_drive_waveform(engineering.drive)
    np.testing.assert_allclose(time, [0, 1, 2, 5, 6, 7])
    np.testing.assert_allclose(currents[2:4], 0)
    result = evaluate_gradient_engineering(engineering, _fabrication())
    assert result["waveform"]["duration_s"] == 7
    assert len(result["thermal"]["sample_temperatures_K"]) == len(time)
    # No Joule input during rest: temperature excess follows exp(-rest/tau).
    temperature = np.asarray(result["thermal"]["sample_temperatures_K"])
    assert temperature[3] - 293.15 == pytest.approx((temperature[2] - 293.15) * np.exp(-1.5))
    assert result["thermal"]["peak_temperature_K"] >= max(temperature)
    single = _engineering().to_dict()["drive"]
    assert "repetitions" not in single and "rest_s" not in single


def test_repetition_refuses_discontinuous_endpoints():
    with pytest.raises(ValueError, match="zero current"):
        replace(_engineering().drive, currents_A=((1, 0), (2, -1), (0, 0)), repetitions=2)
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(_engineering().drive, repetitions=2, rest_s=1e308)


def test_legacy_maximum_single_waveform_remains_valid():
    drive = DriveSpec(
        ("primary",),
        (1,),
        (1,),
        "Maximum existing waveform contract",
        time_s=tuple(np.arange(10000, dtype=float)),
        currents_A=tuple((0,) for _ in range(10000)),
    )
    assert len(drive.time_s) == 10000
    assert "repetitions" not in drive.to_dict()


def test_csv_requires_exact_order_and_never_sorts(tmp_path):
    filename = tmp_path / "pulse.csv"
    filename.write_text("time_ms,primary_mA,shield_mA\n0,0,0\n1,2000,-1000\n2,0,0\n")
    waveform = import_waveform_csv(filename, ("primary", "shield"), time_unit="ms", current_unit="mA")
    np.testing.assert_allclose(waveform["time_s"], [0, 0.001, 0.002])
    np.testing.assert_allclose(waveform["currents_A"][1], [2, -1])
    with pytest.raises(ValueError, match="header"):
        import_waveform_csv(filename, ("shield", "primary"), time_unit="ms", current_unit="mA")
    filename.write_text("time_s,primary_A,shield_A\n0,0,0\n0,2,-1\n")
    with pytest.raises(ValueError, match="strictly increase"):
        import_waveform_csv(filename, ("primary", "shield"))


def test_complex_response_preserved_and_acceptance_has_witness():
    analysis = _analysis()
    assert ExternalAnalysis.from_dict(analysis.to_dict()) == analysis
    spec = DynamicStudySpec(analyses=(analysis,), material_sha256="a" * 64, environment_sha256="c" * 64)
    result = review_dynamic_analyses(spec, _fabrication())
    row = result["analyses"][0]
    assert row["complex_response"] and row["binding_verified"] and not row["solver_accuracy_verified"]
    assert row["peak"] == pytest.approx(5)
    assert row["limiting_axis_value"] == 100
    assert row["limiting_point_m"] == [0, 0, 0.02]
    assert row["passed"] is False and not result["qualification_claim"]
    assert result["unassessed_physics"] == ["mechanical", "spatial_thermal"]
    assert "imaginary" not in row  # Kept in imported analysis payload, never real magnetostatic arrays.


def test_material_environment_and_waveform_changes_invalidate_external_results():
    engineering = _engineering()
    analysis = _analysis(
        name="Sleeve heating",
        kind="spatial_thermal",
        quantity="temperature",
        unit="K",
        axis_unit="s",
        axis=(0, 2),
        real=(((293,), (294,)), ((301,), (300,))),
        imaginary=None,
        excitation_body_id=None,
        waveform_sha256=waveform_digest(engineering.drive),
        limit=302,
    )
    spec = DynamicStudySpec(analyses=(analysis,), material_sha256="a" * 64, environment_sha256="c" * 64)
    row = review_dynamic_analyses(spec, _fabrication(), engineering)["analyses"][0]
    assert row["available"] and row["passed"]
    changed = replace(engineering, drive=replace(engineering.drive, currents_A=((0, 0), (3, -1), (0, 0))))
    row = review_dynamic_analyses(spec, _fabrication(), changed)["analyses"][0]
    assert not row["available"] and row["passed"] is None and "waveform" in row["reasons"][0]
    row = review_dynamic_analyses(replace(spec, environment_sha256="d" * 64), _fabrication(), engineering)["analyses"][
        0
    ]
    assert not row["available"] and "environment" in row["reasons"][0]


def test_import_external_analysis_binds_file_and_disallows_complex_temperature(tmp_path):
    filename = tmp_path / "solver.json"
    value = _analysis().to_dict()
    value.pop("content_sha256")
    value.pop("source_sha256")
    assert value["schema"] == ANALYSIS_SCHEMA
    filename.write_text(json.dumps(value))
    analysis = import_external_analysis(filename)
    assert analysis.source_sha256 == sha256(filename.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="real values"):
        _analysis(
            kind="spatial_thermal",
            quantity="temperature",
            unit="K",
            excitation_body_id=None,
            waveform_sha256="d" * 64,
            real=(((293,), (293,)), ((294,), (294,))),
            imaginary=(((0,), (0,)), ((0,), (0,))),
        )


def test_source_payload_unknown_fields_and_complex_magnetostatic_mix_are_rejected():
    value = DynamicStudySpec().to_dict()
    value["eddy_solver_passed"] = True
    with pytest.raises(ValueError, match="Unknown"):
        DynamicStudySpec.from_dict(value)
    with pytest.raises(ValueError, match="real numerical"):
        replace(_reference(), values=np.eye(2, dtype=complex))


def test_production_review_passes_geometry_verified_references_to_drive_evaluator(tmp_path, monkeypatch):
    """Exercise the actual saved-result application block with deterministic stage evidence."""
    from coilgen import CoilGenConfig, CoilPart, Mesh, TargetField, WirePath
    from coilgen.pipeline_stages import PipelineStage
    from coilgen.research import application, fabrication as fabrication_module
    from coilgen.research.requests import ResearchRequest

    engineering = replace(_engineering(matrix=True), independent_winding_validation=False)
    target = TargetField(
        coords=np.asarray([[0, 0, 0.02], [0, 0, 0.03]]),
        b=np.zeros((2, 3)),
        weights=np.ones(2),
        suppressed=np.zeros(2, bool),
        target_field_group_inds=np.zeros(2, int),
    )
    mesh = Mesh(vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), faces=np.asarray([[0, 1, 2]]))
    part = CoilPart(mesh=mesh, wire_path=WirePath(v=np.asarray([[0, 0, 0], [0.01, 0, 0]])))
    config = CoilGenConfig()
    result = SimpleNamespace(
        coil_parts=[part],
        target_field=target,
        input_data=config,
        combined_mesh=mesh,
        b_field_opt_sf=np.zeros((2, 3)),
        regularization_diagnostics={},
        timing_summary={},
        needed_current_layout=1,
        field_by_unconnected_loops=np.zeros((2, 3)),
        field_loops_per1Amp=np.zeros((2, 3)),
        field_by_layout=np.zeros((2, 3)),
        field_layout_per1Amp=np.zeros((2, 3)),
    )
    checkpoint = SimpleNamespace(
        stage=PipelineStage.EVALUATED,
        result=result,
        save=lambda path: Path(path).write_bytes(b"deterministic stage evidence"),
    )
    fabrication = _fabrication()
    fabrication.ready, fabrication.failure_reasons, fabrication.paths, fabrication.artifacts = False, [], [], []
    fabrication.summary = lambda: {"ready": False, "paths": []}
    for row in fabrication.electrical_metrics["bodies"]:
        row.update(length_m=1, current_A=1, circuit_complete=True, joule_power_W=1)
    fabrication.electrical_metrics["total_joule_power_W"] = 2
    monkeypatch.setattr(application.CoilGenPipeline, "run_stage", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(fabrication_module, "compile_fabrication", lambda *_args, **_kwargs: fabrication)
    monkeypatch.setattr(
        application,
        "validate_on_independent_grid",
        lambda *_args, **_kwargs: {
            "available": False,
            "passed": None,
            "reason": "Deterministic fixture has no validation grid",
        },
    )
    monkeypatch.setattr(
        application,
        "validate_fabricated_constraints",
        lambda *_args, **_kwargs: {"checks": [], "unknown_reasons": [], "field_T": None},
    )
    monkeypatch.setattr(
        application, "write_design_report", lambda _result, path: Path(path).write_text("fixture report")
    )
    request = ResearchRequest(
        name="Bound coupled circuit",
        config={
            "coil_mesh_file": "create cylinder mesh",
            "cylinder_mesh_parameter_list": [0.3, 0.12, 8, 4, 1, 0, 0, 0],
        },
        fabricate=True,
        engineering=engineering.to_dict(),
        dynamic_analysis=DynamicStudySpec(inductance_reference=_reference()).to_dict(),
    )
    report = application._run_design(request, tmp_path / "actual-application-review")
    assert report["engineering_review"]["drive"]["mutual_inductance_included"]
    np.testing.assert_allclose(report["engineering_review"]["waveform"]["interval_voltage_end_V"][0], [5.5, -3])
    assert report["dynamic_analysis_review"]["engineering_review"] == report["engineering_review"]
