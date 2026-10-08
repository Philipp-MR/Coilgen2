"""Explicit feed geometry and reproducible quasistatic gradient-drive screens."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig, CoilPart, ContourLine, Mesh, TargetField, WirePath
from coilgen.gradient_engineering import (
    CircuitSpec,
    DriveSpec,
    GradientEngineeringSpec,
    InductanceMatrixSpec,
    ThermalSpec,
    evaluate_gradient_engineering,
    winding_path_sha256,
)
from coilgen.physics import biot_savart_wire_path
from coilgen.research.fabrication import compile_fabrication, winding_instructions


BODY = "part001_body0001"


def _result(*, closed=False):
    mesh = Mesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]), faces=np.array([[0, 1, 2], [0, 2, 3]])
    )
    if closed:
        angle = np.linspace(0, 2 * np.pi, 17)
        points = np.column_stack((0.03 * np.cos(angle), 0.03 * np.sin(angle), np.zeros_like(angle)))
        points[-1] = points[0]
        part = CoilPart(mesh=mesh, contour_lines=[ContourLine(v=points, potential=1)], contour_step=2)
    else:
        points = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
        part = CoilPart(mesh=mesh, wire_path=WirePath(v=points), contour_step=2)
    coordinates = np.array([[0, 0, 0.02], [0, 0.01, 0.03], [0.01, 0.02, 0.03]])
    target = TargetField(
        coords=coordinates,
        b=2 * biot_savart_wire_path(points, coordinates),
        weights=np.ones(3),
        suppressed=np.zeros(3, bool),
        target_field_group_inds=np.zeros(3, int),
    )
    return SimpleNamespace(
        coil_parts=[part],
        target_field=target,
        input_data=CoilGenConfig(
            conductor_cross_section_width=0.001, conductor_cross_section_height=0.001, sweep_max_chord_error=0
        ),
    )


def _circuit(points, *, closed=False, body_id=BODY, polarity=1):
    cut = 3 if closed else None
    if closed:
        unique = points[:-1]
        retained = np.asarray([unique[(cut + 1 + index) % len(unique)] for index in range(len(unique))])
    else:
        retained = np.asarray(points)
    if polarity == -1:
        retained = retained[::-1]
    driver = np.array([0, -0.06, retained[0, 2]])
    return CircuitSpec(
        body_id=body_id,
        expected_path_sha256=winding_path_sha256(points),
        feed_break_segment=cut,
        polarity=polarity,
        supply_points_m=[driver, retained[0]],
        return_points_m=[retained[-1], [0.04, -0.04, driver[2]], driver],
        source="Recorded measured fixture routing",
    )


def _drive_report(circuits, *, resistance=(3,), inductance=(2,)):
    return SimpleNamespace(
        electrical_metrics={
            "bodies": [
                {"body_id": circuit.body_id, "resistance_ohm": resistance[index], "inductance_H": inductance[index]}
                for index, circuit in enumerate(circuits)
            ]
        },
        winding_steps=[
            {
                "body_id": circuit.body_id,
                "drive_feed_designed": True,
                "expected_winding_path_sha256": circuit.expected_path_sha256,
                "electrical_path_sha256": "e" * 64,
            }
            for circuit in circuits
        ],
    )


def _waveform(circuits, *, times=(0, 1, 3), currents=((0,), (3,), (0,)), **kwargs):
    return GradientEngineeringSpec(
        circuits=circuits,
        drive=DriveSpec(
            body_ids=tuple(circuit.body_id for circuit in circuits),
            current_limits_A=(2,) * len(circuits),
            voltage_limits_V=(14,) * len(circuits),
            source="Explicit gradient amplifier limits and waveform",
            time_s=times,
            currents_A=currents,
            **kwargs,
        ),
    )


@pytest.mark.parametrize("closed,polarity", [(False, 1), (False, -1), (True, 1), (True, -1)])
def test_circuit_preserves_all_retained_sampled_winding_intervals(closed, polarity):
    record = compile_fabrication(_result(closed=closed), kernel_backend="reference").paths[0]
    original = record.points_m.copy()
    circuit = _circuit(original, closed=closed, polarity=polarity)
    constructed = circuit.construct_path(original, closed=closed)
    if closed:
        winding = original[:-1]
        winding = np.asarray([winding[(4 + index) % len(winding)] for index in range(len(winding))])
    else:
        winding = original
    if polarity == -1:
        winding = winding[::-1]
    np.testing.assert_array_equal(constructed["winding_points_m"], winding)
    np.testing.assert_array_equal(record.points_m, original)
    np.testing.assert_array_equal(constructed["electrical_points_m"][0], constructed["electrical_points_m"][-1])
    plan = winding_instructions(record, circuit=circuit)
    assert plan["drive_feed_designed"] and plan["circuit_complete"]
    assert plan["current_direction_polarity"] == polarity
    assert {row["operation"] for row in plan["steps"]} >= {"route_supply", "route_return"}
    assert plan["winding_length_m"] == pytest.approx(np.linalg.norm(np.diff(winding, axis=0), axis=1).sum())


def test_explicit_feed_participates_in_fields_resistance_and_export_without_qualification(tmp_path):
    result = _result(closed=True)
    baseline = compile_fabrication(result, kernel_backend="reference")
    record = baseline.paths[0]
    circuit = _circuit(record.points_m, closed=True)
    report = compile_fabrication(result, tmp_path, kernel_backend="reference", circuits={BODY: circuit})
    electrical = report.winding_steps[0]["electrical_points_m"]
    saved = np.load(tmp_path / "fabrication_fields.npz")
    np.testing.assert_allclose(
        saved["fabrication_field_T"],
        2 * biot_savart_wire_path(electrical, result.target_field.coords),
        rtol=1e-13,
        atol=1e-20,
    )
    length = np.linalg.norm(np.diff(electrical, axis=0), axis=1).sum()
    row = report.electrical_metrics["bodies"][0]
    assert row["length_m"] == pytest.approx(length)
    assert row["resistance_ohm"] == pytest.approx(result.input_data.specific_conductivity_conductor * length / 1e-6)
    assert row["inductance_kind"] == "closed_path_self"
    assert not row["mutual_inductance_included"]
    assert report.field_metrics["drive_feed_designed"]
    assert not report.ready and any("physical_feed_break_not_swept" in reason for reason in report.failure_reasons)
    assert any("drive_lead_conductor_clearance_not_certified" in reason for reason in report.failure_reasons)
    assert not list(tmp_path.glob("*.stl"))
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / f"{BODY}_electrical_path_explicit_drive.csv", delimiter=",", skiprows=1), electrical
    )
    json.loads((tmp_path / "fabrication_report.json").read_text())


def test_closed_feed_needs_explicit_break_and_exact_geometry_digest():
    record = compile_fabrication(_result(closed=True), kernel_backend="reference").paths[0]
    circuit = _circuit(record.points_m, closed=True)
    with pytest.raises(ValueError, match="feed-break"):
        replace(circuit, feed_break_segment=None).construct_path(record.points_m, closed=True)
    with pytest.raises(ValueError, match="stale winding"):
        replace(circuit, expected_path_sha256="f" * 64).construct_path(record.points_m, closed=True)
    with pytest.raises(ValueError, match="feed-break"):
        replace(circuit, feed_break_segment=1000).construct_path(record.points_m, closed=True)


def test_endpoints_are_validated_without_automatic_lead_or_drive_closure():
    points = np.array([[0, 0, 0], [0.01, 0, 0]])
    circuit = _circuit(points)
    broken_supply = list(circuit.supply_points_m)
    broken_supply[-1] = [0.02, 0, 0]
    with pytest.raises(ValueError, match="winding start"):
        replace(circuit, supply_points_m=broken_supply).construct_path(points, closed=False)
    broken_return = list(circuit.return_points_m)
    broken_return[-1] = [0.005, -0.07, 0]
    distinct = replace(circuit, return_points_m=broken_return)
    with pytest.raises(ValueError, match="drive_closure"):
        distinct.construct_path(points, closed=False)
    closure = [broken_return[-1], [0, -0.07, 0], circuit.supply_points_m[0]]
    completed = replace(distinct, drive_closure_points_m=closure).construct_path(points, closed=False)
    assert "drive_closure" in completed["segment_roles"]


def test_circuits_do_not_accept_unknown_bodies_or_legacy_returns_for_same_body():
    result = _result()
    record = compile_fabrication(result, kernel_backend="reference").paths[0]
    circuit = _circuit(record.points_m)
    with pytest.raises(ValueError, match="unknown fabrication"):
        compile_fabrication(result, circuits={"other": replace(circuit, body_id="other")}, kernel_backend="reference")
    with pytest.raises(ValueError, match="same body"):
        compile_fabrication(
            result, circuits={BODY: circuit}, leads={BODY: circuit.return_points_m}, kernel_backend="reference"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"length_unit": "mm"},
        {"coordinate_frame": "local"},
        {"polarity": True},
        {"polarity": 1.0},
        {"supply_points_m": None},
        {"supply_points_m": [[0, 0, 0], [0, 0, 0]]},
        {"return_points_m": [[0, 0, 0], [0, 0, np.nan]]},
    ],
)
def test_circuit_validation_rejects_ambiguous_units_nonfinite_or_degenerate_geometry(change):
    value = _circuit(np.array([[0, 0, 0], [0.01, 0, 0]])).to_dict()
    value.update(change)
    with pytest.raises(ValueError):
        CircuitSpec.from_dict(value)


def test_spec_is_strict_serializable_and_does_not_mutate_authoring_inputs():
    circuit = _circuit(np.array([[0, 0, 0], [0.01, 0, 0]]))
    spec = _waveform((circuit,))
    value = spec.to_dict()
    restored = GradientEngineeringSpec.from_dict(json.loads(json.dumps(value)))
    assert restored == spec and restored.circuit_map[BODY] == circuit
    value["circuits"][0]["supply_points_m"][0][0] = 99
    assert spec.circuits[0].supply_points_m[0][0] != 99
    with pytest.raises(ValueError, match="Unknown"):
        GradientEngineeringSpec.from_dict({"unknown": True})


@pytest.mark.parametrize(
    "change",
    [
        {"time_s": [0, 0, 1]},
        {"currents_A": [[0], [1]]},
        {"current_limits_A": [0]},
        {"voltage_limits_V": [np.inf]},
        {"time_s": [0, 1, 2], "currents_A": [[0], [np.nan], [0]]},
        {"target_gradient_T_per_m": 0.04},
    ],
)
def test_waveform_validation_requires_finite_matching_strictly_increasing_si_samples(change):
    value = _waveform((_circuit(np.array([[0, 0, 0], [0.01, 0, 0]])),)).drive.to_dict()
    value.update(change)
    with pytest.raises(ValueError):
        DriveSpec.from_dict(value)


def test_waveform_voltage_extrema_rms_and_joule_loss_use_supplied_nonuniform_times():
    circuit = _circuit(np.array([[0, 0, 0], [0.01, 0, 0]]))
    spec = _waveform((circuit,))
    report = evaluate_gradient_engineering(spec, _drive_report((circuit,)))
    row = report["drive"]["bodies"][0]
    assert row["peak_current_A"] == 3 and row["current_margin_A"] == -1
    assert row["peak_voltage_V"] == pytest.approx(15)
    assert row["voltage_margin_V"] == pytest.approx(-1)
    assert row["rms_current_A"] == pytest.approx(np.sqrt(3))
    assert row["average_joule_loss_W"] == pytest.approx(9)
    assert not row["within_current_limit"] and not row["within_voltage_limit"]
    assert not report["thermal"]["available"] and not report["qualification_claim"]


def test_target_drive_needs_pipeline_verified_geometry_bound_efficiency():
    circuit = _circuit(np.array([[0, 0, 0], [0.01, 0, 0]]))
    drive = DriveSpec(
        body_ids=(BODY,),
        current_limits_A=(3,),
        voltage_limits_V=(30,),
        source="Gradient ramp",
        target_gradient_T_per_m=0.04,
        target_slew_T_per_m_per_s=0.08,
        gradient_efficiencies_T_per_m_per_A=(0.01,),
        gradient_efficiency_path_sha256=("e" * 64,),
        gradient_efficiency_source_sha256="a" * 64,
        gradient_efficiency_source="Independent physical-path regression",
    )
    spec = GradientEngineeringSpec(circuits=(circuit,), drive=drive)
    source = _drive_report((circuit,))
    assert not evaluate_gradient_engineering(spec, source)["drive"]["available"]
    reference = {
        "body_ids": [BODY],
        "path_sha256": ["e" * 64],
        "efficiencies_T_per_m_per_A": [0.01],
        "source_sha256": "a" * 64,
    }
    report = evaluate_gradient_engineering(spec, source, verified_efficiency_reference=reference)
    row = report["drive"]["bodies"][0]
    assert row["peak_current_A"] == pytest.approx(4)
    assert row["peak_voltage_V"] == pytest.approx(28)
    assert row["rms_current_A"] is None and row["average_joule_loss_W"] is None
    assert row["target_metrics"]["drive_voltage_target_slew_V"] == pytest.approx(28)
    reference["path_sha256"] = ["b" * 64]
    assert not evaluate_gradient_engineering(spec, source, verified_efficiency_reference=reference)["drive"][
        "available"
    ]


def test_mutual_terms_require_explicit_verified_geometry_matched_matrix():
    points = np.array([[0, 0, 0], [0.01, 0, 0]])
    first, second = _circuit(points), _circuit(points, body_id="other")
    matrix = InductanceMatrixSpec(
        body_ids=(BODY, "other"),
        values_H=((2, 0.5), (0.5, 3)),
        expected_electrical_path_sha256=("e" * 64,) * 2,
        source="Verified external coupled-circuit model",
        verified=True,
    )
    spec = _waveform((first, second), times=(0, 1), currents=((0, 0), (2, -1)), inductance_matrix=matrix)
    source = _drive_report((first, second), resistance=(1, 1), inductance=(2, 3))
    unverified = evaluate_gradient_engineering(spec, source)
    assert not unverified["drive"]["mutual_inductance_included"]
    np.testing.assert_allclose([row["peak_voltage_V"] for row in unverified["drive"]["bodies"]], [6, 4])
    reference = {
        "body_ids": [BODY, "other"],
        "path_sha256": ["e" * 64] * 2,
        "values_H": [[2, 0.5], [0.5, 3]],
        "source": matrix.source,
    }
    verified = evaluate_gradient_engineering(spec, source, verified_inductance_reference=reference)
    assert verified["drive"]["mutual_inductance_included"]
    np.testing.assert_allclose([row["peak_voltage_V"] for row in verified["drive"]["bodies"]], [5.5, 3])


@pytest.mark.parametrize("values", [((1, 2), (2, 1)), ((1, 0.5), (0.1, 1)), ((np.inf, 0), (0, 1))])
def test_inductance_matrix_rejects_nonphysical_or_nonfinite_inputs(values):
    with pytest.raises(ValueError):
        InductanceMatrixSpec(
            body_ids=("a", "b"),
            values_H=values,
            expected_electrical_path_sha256=("e" * 64,) * 2,
            source="External model",
        )


def test_optional_measured_thermal_model_matches_constant_power_rc_response():
    circuit = _circuit(np.array([[0, 0, 0], [0.01, 0, 0]]))
    thermal = ThermalSpec(
        thermal_resistance_K_per_W=0.5,
        heat_capacity_J_per_K=4,
        ambient_temperature_K=300,
        initial_temperature_K=300,
        max_temperature_K=310,
        source="Measured assembly RC fit",
        measured=True,
    )
    spec = _waveform((circuit,), times=(0, 1), currents=((2,), (2,)), thermal=thermal)
    report = evaluate_gradient_engineering(spec, _drive_report((circuit,)))
    predicted = 300 + 6 * (1 - np.exp(-0.5))
    assert report["thermal"]["sample_temperatures_K"][-1] == pytest.approx(predicted, rel=1e-12)
    assert report["thermal"]["peak_temperature_K"] == pytest.approx(predicted, rel=1e-12)
    assert report["thermal"]["temperature_margin_K"] == pytest.approx(310 - predicted)
    with pytest.raises(ValueError, match="measured"):
        replace(thermal, measured=False)


def test_thermal_peak_inside_falling_pulse_matches_independent_transient_solution():
    from scipy.integrate import solve_ivp

    circuit = _circuit(np.array([[0, 0, 0], [0.01, 0, 0]]))
    thermal = ThermalSpec(
        thermal_resistance_K_per_W=0.1,
        heat_capacity_J_per_K=1,
        ambient_temperature_K=300,
        initial_temperature_K=300,
        max_temperature_K=310,
        source="Measured pulse model",
        measured=True,
    )
    spec = _waveform((circuit,), times=(0, 1), currents=((3,), (0,)), thermal=thermal)
    evaluated = evaluate_gradient_engineering(spec, _drive_report((circuit,)))
    solved = solve_ivp(
        lambda t, temperature: 3 * (3 * (1 - t)) ** 2 - (temperature - 300) / 0.1,
        (0, 1),
        [300.0],
        rtol=1e-10,
        atol=1e-10,
        dense_output=True,
        max_step=0.005,
    )
    peak = np.max(solved.sol(np.linspace(0, 1, 10001)))
    assert evaluated["thermal"]["peak_temperature_K"] == pytest.approx(peak, abs=1e-7)
    assert evaluated["thermal"]["peak_temperature_K"] > max(evaluated["thermal"]["sample_temperatures_K"])
