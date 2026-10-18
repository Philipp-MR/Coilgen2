"""Reproduce independent SI field/current checks with the installed CoilGen2."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, localcontext
import json
from pathlib import Path
import platform

import numpy as np

import coilgen
from coilgen import validate_circular_loop_biot_savart
from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh, current_density_from_basis
from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.models import CoilPart, Mesh, TargetField
from coilgen.persistence import numerical_source_digest
from coilgen.physics import biot_savart_wire_path, calculate_resistance_matrix, calculate_sensitivity_matrix


def check_loop_refinement():
    rows = [
        validate_circular_loop_biot_savart(radius_m=0.1, z_offset_m=z, segment_count=n).as_dict()
        for z in (0.0, 0.03, 0.2)
        for n in (180, 720, 2880)
    ]
    ratios = [rows[i]["relative_error_percent"] / rows[i + 1]["relative_error_percent"] for i in (0, 1, 3, 4, 6, 7)]
    assert all(15.9 < value < 16.1 for value in ratios)
    assert max(rows[i]["relative_error_percent"] for i in (2, 5, 8)) < 5.6e-5
    return {"cases": rows, "error_reduction_per_4x_segment_count": ratios}


def check_nearly_collinear_wire():
    rows = []
    radii = np.array([1e-4, 1e-6, 1e-8, 1e-10])
    wire = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    for axial in (-0.1, 0.2):
        points = np.column_stack((radii, np.zeros_like(radii), np.full_like(radii, axial)))
        with localcontext() as context:
            context.prec = 60
            a, b = Decimal(str(axial)), Decimal("0.1") - Decimal(str(axial))
            expected = []
            for value in radii:
                rho = Decimal(str(value))
                expected.append(
                    float(Decimal("1e-7") / rho * (a / (rho * rho + a * a).sqrt() + b / (rho * rho + b * b).sqrt()))
                )
        measured = biot_savart_wire_path(wire, points)[:, 1]
        old_endpoint_difference = (
            1e-7 / radii * (axial / np.hypot(radii, axial) + (0.1 - axial) / np.hypot(radii, 0.1 - axial))
        )
        np.testing.assert_allclose(measured, expected, rtol=5e-15, atol=0)
        rows.append(
            {
                "axial_m": axial,
                "radial_m": radii.tolist(),
                "decimal_60_digit_field_T_per_A": expected,
                "evaluated_field_T_per_A": measured.tolist(),
                "relative_error": (measured / expected - 1).tolist(),
                "legacy_endpoint_difference_relative_error": (old_endpoint_difference / expected - 1).tolist(),
            }
        )
    return rows


def check_chunk_invariance():
    rng = np.random.default_rng(3490)
    angle = np.linspace(0, 4 * np.pi, 257)
    wire = np.column_stack((0.1 * np.cos(angle), 0.1 * np.sin(angle), np.linspace(-0.15, 0.15, len(angle))))
    targets = rng.normal(0, 0.01, size=(513, 3))
    reference = biot_savart_wire_path(wire, targets, worker_count=1, max_chunk_memory_mb=64)
    rows = []
    for workers in (1, 3):
        actual = biot_savart_wire_path(wire, targets, worker_count=workers, min_work_size=1, max_chunk_memory_mb=0.05)
        np.testing.assert_allclose(actual, reference, rtol=3e-14, atol=1e-20)
        rows.append(
            {
                "workers": workers,
                "chunk_memory_budget_mb": 0.05,
                "relative_L2_difference": float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)),
            }
        )
    return rows


def check_uniform_sheet():
    base = np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    # A uniform K=-y A/m sheet subtends this exact rectangular solid angle.
    expected = np.array([-4e-7 * np.arctan(0.01 / (0.2 * np.sqrt(0.06))), 0.0, 0.0])
    rotation, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 1.0], [3.0, 1.0, 4.0]]))
    rows = []
    for scale in (1e-3, 1.0, 10.0):
        for moved in (False, True):
            q = rotation if moved else np.eye(3)
            shift = np.array([0.3, -0.2, 0.4]) if moved else np.zeros(3)
            vertices = scale * base @ q.T + shift
            part = CoilPart(normalize_mesh(Mesh(vertices, faces)), stream_function=scale * base[:, 0])
            calculate_one_ring_by_mesh([part])
            calculate_basis_functions([part])
            current = current_density_from_basis(part)
            target = TargetField(
                np.array([[0.0, 0.0, scale * 0.2]]) @ q.T + shift,
                np.zeros((1, 3)),
                np.ones(1),
                np.zeros(1, bool),
                np.ones(1, int),
            )
            fields = []
            for method in ("scalar", "vectorized"):
                config = CoilGenConfig(
                    gauss_order=12,
                    sensitivity_calculation_method=method,
                    parallel_worker_count=1,
                    progress_flag=False,
                    specific_conductivity_conductor=1.0,
                    conductor_thickness=1.0,
                )
                calculate_sensitivity_matrix([part], target, config)
                fields.append(np.einsum("ctv,v->tc", part.sensitivity_matrix, part.stream_function)[0])
            calculate_resistance_matrix([part], config)
            gauge = np.einsum("ctv,v->tc", part.sensitivity_matrix, np.ones(4))[0]
            power = float(part.stream_function @ part.resistance_matrix @ part.stream_function)
            relative_error = float(np.linalg.norm(fields[1] - expected @ q.T) / np.linalg.norm(expected))
            assert relative_error < 5e-11
            np.testing.assert_allclose(fields[0], fields[1], rtol=1e-12, atol=1e-21)
            assert np.linalg.norm(gauge) < 1e-11 * np.linalg.norm(expected) / scale
            np.testing.assert_allclose(
                current, np.tile(np.array([0.0, -1.0, 0.0]) @ q.T, (2, 1)), atol=1e-11, rtol=1e-11
            )
            assert np.isclose(power, 0.04 * scale**2, rtol=1e-10)
            rows.append(
                {
                    "coordinate_scale": scale,
                    "rotated_and_translated": moved,
                    "sheet_field_relative_error": relative_error,
                    "scalar_vectorized_relative_difference": float(
                        np.linalg.norm(fields[0] - fields[1]) / np.linalg.norm(expected)
                    ),
                    "ohmic_power_W_for_sheet_resistivity_factor_1_ohm": power,
                    "constant_stream_gauge_residual_T_per_A": float(np.linalg.norm(gauge)),
                }
            )
    return rows


if __name__ == "__main__":
    report = {
        "passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numerical_source_sha256": numerical_source_digest(Path(coilgen.__file__).parent),
        "scope": "Independent analytic field checks, finite-wire conditioning, P1 sheet gauge/current and SI scaling. No optimized-mesh convergence or hardware qualification is asserted.",
        "circular_loop_refinement": check_loop_refinement(),
        "nearly_collinear_wire": check_nearly_collinear_wire(),
        "chunk_and_worker_invariance": check_chunk_invariance(),
        "uniform_sheet_solid_angle": check_uniform_sheet(),
    }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"All independent physical checks passed: {destination}")
