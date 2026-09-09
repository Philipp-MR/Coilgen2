"""Verified fabrication artifacts for one independently closed circular turn.

Run: python examples/research_valid_fabrication.py

The faceted copper body is a valid geometric example, not a complete driven
coil: a feed break, supply terminals and external leads still need design.
Reported L is the isolated turn's thin-wire self estimate; mutual coupling,
finite-cross-section field integration and RF/thermal effects are not included.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

if __package__:
    from ._example_support import example_options, example_runtime
else:
    from _example_support import example_options, example_runtime


def main(argv=None):
    options = example_options("outputs/valid_fabrication_example", argv)
    with example_runtime(required=("numpy",)):
        import numpy as np
        from coilgen import CoilGenConfig, CoilPart, ContourLine, Mesh, TargetField
        from coilgen.manufacturing import _polygon_area, _polygon_centroid
        from coilgen.persistence import atomic_json, atomic_text
        from coilgen.physics import biot_savart_wire_path
        from coilgen.research.fabrication import compile_fabrication
        from coilgen.research.materials import WireSheetMaterial

        radius, current, diameter = 0.04, 2.0, 0.001
        angle = np.linspace(0.0, 2.0 * np.pi, 129)
        points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)))
        points[-1] = points[0]
        support = Mesh(
            vertices=np.array([[-0.06, -0.06, 0], [0.06, -0.06, 0], [0.06, 0.06, 0], [-0.06, 0.06, 0]]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
        )
        part = CoilPart(mesh=support, contour_lines=[ContourLine(v=points, potential=current)], contour_step=current)
        profile_angle = np.arange(16) * 2 * np.pi / 16
        profile = np.column_stack((diameter / 2 * np.cos(profile_angle), diameter / 2 * np.sin(profile_angle)))
        config = CoilGenConfig(
            cross_sectional_points=profile,
            conductor_cross_section_width=diameter,
            conductor_cross_section_height=diameter,
            conductor_mounting_mode="surface_outer",
            sweep_insulation_thickness=0.0001,
            parallel_worker_count=1,
            layout_target_error_rms_limit_T=2e-8,
        )
        # A separate, explicit model for a locally parallel layer with 4 mm pitch.
        # This affects a future sheet optimization, not this turn's exact R/P.
        material = WireSheetMaterial(wire_area_m2=_polygon_area(profile), pitch_m=0.004)
        config = material.apply_to(config)
        heights = np.array([0.01, 0.02, 0.03, 0.04])
        coordinates = np.column_stack((np.zeros((4, 2)), heights))
        # Independent analytic circle benchmark, at the mounted centroid height.
        target_b = np.zeros((4, 3))
        target_b[:, 2] = (
            (4 * np.pi * 1e-7) * current * radius**2 / (2 * (radius**2 + (heights - diameter / 2) ** 2) ** 1.5)
        )
        target = TargetField(
            coords=coordinates,
            b=target_b,
            weights=np.ones(4),
            suppressed=np.zeros(4, dtype=bool),
            target_field_group_inds=np.ones(4, dtype=int),
        )
        output = options.output_dir
        report = compile_fabrication(
            SimpleNamespace(coil_parts=[part], input_data=config, target_field=target),
            output,
            surface_realization="mesh",
        )
        if not report.ready:
            raise RuntimeError(f"Fabrication verification failed: {report.failure_reasons}")
        record = report.paths[0]
        exported = np.loadtxt(output / "part001_body0001_fabrication_centroid.csv", delimiter=",", skiprows=1)
        np.testing.assert_array_equal(exported, record.points_m)
        with np.load(output / "part001_body0001_fabrication_frames.npz") as saved:
            centered_profile = saved["cross_section_m"] - _polygon_centroid(saved["cross_section_m"])
            reconstructed_rings = (
                saved["points_m"][:, None, :] + centered_profile[None, :, 0, None] * saved["transverse"][:, None, :]
            )
            reconstructed_rings += centered_profile[None, :, 1, None] * saved["normals"][:, None, :]
        np.testing.assert_allclose(reconstructed_rings.reshape(-1, 3), record.mesh.vertices, atol=2e-17, rtol=1e-14)
        with np.load(output / "fabrication_fields.npz") as fields:
            np.testing.assert_allclose(
                fields["fabrication_field_T"],
                current * biot_savart_wire_path(exported, coordinates),
                rtol=1e-14,
                atol=1e-20,
            )
        assert (output / "part001_body0001_fabrication.stl").is_file()
        verification = {
            "ready_geometry": report.ready,
            "closed_mathematical_turn": True,
            "drive_feed_designed": False,
            "centroid_export_and_frames_verified": True,
            "field_uses_exported_centroid": True,
            "analytic_circle_relative_error_percent": report.field_metrics["dsv_relative_residual_percent"],
            "electrical": report.electrical_metrics["bodies"][0],
            "winding_operation_count": report.winding_steps[0]["operation_count"],
            "sampled_interval_count": report.winding_steps[0]["sampled_interval_count"],
            "sheet_objective_material_assumption": material.summary(),
            "limits": "Feed break, drive terminals and external leads are not designed; isolated self L only, no mutual coupling or finite-cross-section field integration.",
        }
        atomic_json(output / "example_verification.json", verification)
        atomic_text(
            output / "README.md",
            "# Verified single-turn fabrication example\n\n"
            "The STL and faceted STEP describe one geometrically checked, independently closed copper turn. "
            "Centroid CSV/STEP, frame/profile NPZ, winding instructions and field NPZ refer to that same mounted geometry. "
            "The field is checked against an independent analytic circular-loop benchmark.\n\n"
            "The mathematical loop is closed. A feed break, drive terminals and external supply leads remain undesigned. "
            "The inductance is an isolated thin-wire self estimate; this example makes no mutual-circuit, RF or thermal claim.\n",
        )
        print(json.dumps(verification, indent=2))
        print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
