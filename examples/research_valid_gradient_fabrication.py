"""Verified bench axial-gradient geometry from two independent Maxwell turns.

Run: python examples/research_valid_gradient_fabrication.py

This additive reference has no optimization or connectors. Two independently
closed circular turns carry opposed, recorded currents. Strict swept-body
checks and distinct-grid thin-wire field checks do not qualify drive feeds,
mutual inductance, finite conductor cross-section, cooling or construction.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

if __package__:
    from ._example_support import example_options, example_runtime
else:
    from _example_support import example_options, example_runtime


def build_reference():
    import numpy as np
    from coilgen import CoilGenConfig, CoilPart, ContourLine, Mesh, TargetField
    from coilgen.target_regions import sample_target_region_from_config

    radius, current, diameter = 0.12, 5.0, 0.002
    half_separation = np.sqrt(3.0) * radius / 2
    mu0 = 4.0 * np.pi * 1.0e-7
    gradient = 3 * mu0 * current * radius**2 * half_separation / (radius**2 + half_separation**2) ** 2.5
    profile_angle = np.arange(16) * 2 * np.pi / 16
    profile = diameter / 2 * np.column_stack((np.cos(profile_angle), np.sin(profile_angle)))
    config = CoilGenConfig(
        field_shape_function=f"{gradient:.17g}*z",
        target_gradient_normalization="off",
        target_gradient_strength=gradient,
        field_evaluation_current_source="stream_function",
        target_region_geometry={"shape": "sphere", "radius_m": 0.02},
        target_region_radius=0.02,
        target_region_resolution=5,
        target_error_constraint_mode="separate",
        target_error_constraint_metric="rms",
        target_error_rms_limit_T=5.0e-9,
        layout_target_error_rms_limit_T=5.0e-9,
        linearity_reference_radius_m=0.02,
        linearity_rms_tolerance_percent=1.0,
        linearity_peak_tolerance_percent=5.0,
        gradient_strength_tolerance_percent=1.0,
        linearity_cross_gradient_tolerance_percent=1.0,
        cross_sectional_points=profile,
        conductor_cross_section_width=diameter,
        conductor_cross_section_height=diameter,
        conductor_mounting_mode="surface_outer",
        sweep_surface_realization="mesh",
        sweep_min_bend_radius=0.006,
        sweep_insulation_thickness=0.0001,
        sweep_manufacturing_tolerance=0.00005,
        parallel_worker_count=1,
        progress_flag=False,
    )
    angle = np.linspace(0.0, 2 * np.pi, 257)
    parts = []
    for sign in (1.0, -1.0):
        # Both planar supports have +z normals. Offset each source/support by
        # one profile radius so the mounted centroid is exactly at ±h.
        source_height = sign * half_separation - diameter / 2
        vertices = np.array(
            [
                [-0.15, -0.15, source_height],
                [0.15, -0.15, source_height],
                [0.15, 0.15, source_height],
                [-0.15, 0.15, source_height],
            ]
        )
        support = Mesh(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]))
        points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full_like(angle, source_height)))
        points[-1] = points[0]
        parts.append(
            CoilPart(
                mesh=support,
                contour_lines=[ContourLine(v=points, potential=sign * current)],
                contour_step=sign * current,
            )
        )
    coordinates = sample_target_region_from_config(config)
    target = np.zeros_like(coordinates)
    target[:, 2] = gradient * coordinates[:, 2]
    result = SimpleNamespace(
        coil_parts=parts,
        input_data=config,
        target_field=TargetField(
            coordinates, target, np.ones(len(target)), np.zeros(len(target), bool), np.ones(len(target), int)
        ),
        combined_mesh=SimpleNamespace(vertices=np.vstack([part.mesh.vertices for part in parts])),
        regularization_diagnostics={"target_gradient_normalization_scale": 1.0},
    )
    return result, {
        "radius_m": radius,
        "half_separation_m": float(half_separation),
        "signed_currents_A": [current, -current],
        "gradient_T_per_m": float(gradient),
        "wire_diameter_m": diameter,
        "segments_per_turn": 256,
    }


def main(argv=None):
    options = example_options("outputs/valid_gradient_fabrication_example", argv)
    with example_runtime(required=("numpy",)):
        import numpy as np
        from coilgen.persistence import atomic_json
        from coilgen.physics import biot_savart_wire_path
        from coilgen.research.fabrication import compile_fabrication
        from coilgen.research.validation import validate_winding_on_independent_grid

        result, parameters = build_reference()
        output = options.output_dir
        report = compile_fabrication(result, output, strict=True, surface_realization="mesh")
        if not report.ready:
            raise RuntimeError(f"Maxwell reference geometry/field check failed: {report.failure_reasons}")
        radius, half_separation = parameters["radius_m"], parameters["half_separation_m"]
        axis_points = np.column_stack((np.zeros((21, 2)), np.linspace(-0.04, 0.04, 21)))
        field = np.zeros_like(axis_points)
        analytic = np.zeros(len(axis_points))
        for record, body, sign in zip(report.paths, report.electrical_metrics["bodies"], (1.0, -1.0)):
            exported = np.loadtxt(output / f"{record.body_id}_fabrication_centroid.csv", delimiter=",", skiprows=1)
            np.testing.assert_array_equal(exported, record.points_m)
            np.testing.assert_allclose(np.linalg.norm(exported[:, :2], axis=1), radius, atol=1e-14, rtol=0)
            np.testing.assert_allclose(exported[:, 2], sign * half_separation, atol=1e-14, rtol=0)
            assert body["current_A"] == sign * 5.0
            field += body["current_A"] * biot_savart_wire_path(exported, axis_points, worker_count=1)
            distance = axis_points[:, 2] - sign * half_separation
            analytic += 4 * np.pi * 1e-7 * body["current_A"] * radius**2 / (2 * (radius**2 + distance**2) ** 1.5)
        analytic_error = float(np.linalg.norm(field[:, 2] - analytic) / np.linalg.norm(analytic))
        if analytic_error >= 5.0e-4:
            raise RuntimeError(f"Circular-loop analytic benchmark failed: relative L2 error={analytic_error:.6g}")
        independent = validate_winding_on_independent_grid(result, report, resolution=12)
        np.savez_compressed(
            output / "independent_gradient_fields.npz",
            points_m=independent["points_m"],
            field_T=independent["field_T"],
            target_T=independent["target_T"],
        )
        verification = {
            "parameters": parameters,
            "geometry_ready": report.ready,
            "analytic_axis_relative_L2_error_percent": 100 * analytic_error,
            "independent_grid": {
                key: value for key, value in independent.items() if key not in {"points_m", "field_T", "target_T"}
            },
            "construction_released": False,
            "drive_feed_designed": False,
            "scope": "Two independently closed benchmark turns; exact mounted centroid paths and original signed currents.",
            "limits": "No feed breaks, terminals, supply/return leads, mutual inductance, finite-cross-section field integration, cooling or construction qualification.",
        }
        atomic_json(output / "example_verification.json", verification)
        if not independent["verified"] or independent["passed"] is not True:
            raise RuntimeError(f"Independent fixed-current gradient validation failed: {independent['checks']}")
        print(json.dumps(verification, indent=2))
        print(f"Artifacts: {output}")
        return result, report, verification


if __name__ == "__main__":
    main()
