"""Verified fabrication artifacts for one independently closed circular turn.

Run: python examples/research_valid_fabrication.py

The faceted copper body is a valid geometric example, not a complete driven
coil: a feed break, supply terminals and external leads still need design.
Reported L is the isolated turn's thin-wire self estimate; mutual coupling,
finite-cross-section field integration and RF/thermal effects are not included.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

# Import after the checkout and native-thread bootstrap for standalone use.
import numpy as np  # noqa: E402
from coilgen import CoilGenConfig, CoilPart, ContourLine, Mesh, TargetField  # noqa: E402
from coilgen.manufacturing import _polygon_area, _polygon_centroid  # noqa: E402
from coilgen.physics import biot_savart_wire_path  # noqa: E402
from coilgen.research.fabrication import compile_fabrication  # noqa: E402
from coilgen.research.materials import WireSheetMaterial  # noqa: E402


def main():
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
    target_b[:, 2] = (4 * np.pi * 1e-7) * current * radius**2 / (2 * (radius**2 + (heights - diameter / 2) ** 2) ** 1.5)
    target = TargetField(
        coords=coordinates,
        b=target_b,
        weights=np.ones(4),
        suppressed=np.zeros(4, dtype=bool),
        target_field_group_inds=np.ones(4, dtype=int),
    )
    output = ROOT / "outputs" / "valid_fabrication_example"
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
    saved = np.load(output / "part001_body0001_fabrication_frames.npz")
    centered_profile = saved["cross_section_m"] - _polygon_centroid(saved["cross_section_m"])
    reconstructed_rings = (
        saved["points_m"][:, None, :] + centered_profile[None, :, 0, None] * saved["transverse"][:, None, :]
    )
    reconstructed_rings += centered_profile[None, :, 1, None] * saved["normals"][:, None, :]
    np.testing.assert_allclose(reconstructed_rings.reshape(-1, 3), record.mesh.vertices, atol=2e-17, rtol=1e-14)
    fields = np.load(output / "fabrication_fields.npz")
    np.testing.assert_allclose(
        fields["fabrication_field_T"], current * biot_savart_wire_path(exported, coordinates), rtol=1e-14, atol=1e-20
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
    (output / "example_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Verified single-turn fabrication example\n\n"
        "The STL and faceted STEP describe one geometrically checked, independently closed copper turn. "
        "Centroid CSV/STEP, frame/profile NPZ, winding instructions and field NPZ refer to that same mounted geometry. "
        "The field is checked against an independent analytic circular-loop benchmark.\n\n"
        "The mathematical loop is closed. A feed break, drive terminals and external supply leads remain undesigned. "
        "The inductance is an isolated thin-wire self estimate; this example makes no mutual-circuit, RF or thermal claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
