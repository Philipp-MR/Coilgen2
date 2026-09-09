from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig, CoilPart, ContourLine, Mesh, TargetField, WirePath, create_swept_coil_meshes
from coilgen.physics import biot_savart_wire_path
from coilgen.research.fabrication import (
    FabricationRecord,
    assembly_clearance,
    compile_fabrication,
    winding_instructions,
)


def _support():
    return Mesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]), faces=np.array([[0, 1, 2], [0, 2, 3]])
    )


def _config(**kwargs):
    return CoilGenConfig(
        conductor_cross_section_width=0.001, conductor_cross_section_height=0.001, sweep_max_chord_error=0, **kwargs
    )


def _record(y, body_id, *, width=0.001, part_index=1):
    part = CoilPart(mesh=_support(), wire_path=WirePath(v=np.array([[-0.01, y, 0], [0.01, y, 0]])), contour_step=2.0)
    create_swept_coil_meshes(
        [part], replace(_config(), conductor_cross_section_width=width, conductor_cross_section_height=width)
    )
    assert part.fabrication_path is not None
    return FabricationRecord(
        body_id, part_index, None, part.fabrication_path, part.swept_body, part.manufacturing_diagnostics
    )


def _result(*, closed=False, current=2.0, config=None):
    if closed:
        angle = np.linspace(0, 2 * np.pi, 49)
        points = np.column_stack((0.03 * np.cos(angle), 0.03 * np.sin(angle), np.zeros_like(angle)))
        points[-1] = points[0]
        part = CoilPart(mesh=_support(), contour_lines=[ContourLine(v=points, potential=1.0)], contour_step=current)
    else:
        points = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
        part = CoilPart(mesh=_support(), wire_path=WirePath(v=points), contour_step=current)
    coords = np.array([[0.0, 0.0, 0.02], [0.0, 0.01, 0.03], [0.01, 0.02, 0.03]])
    field = (current or 1.0) * biot_savart_wire_path(points, coords)
    target = TargetField(
        coords=coords,
        b=field,
        weights=np.array([1.0, 2.0, 3.0]),
        suppressed=np.zeros(3, dtype=bool),
        target_field_group_inds=np.zeros(3, dtype=int),
    )
    return SimpleNamespace(coil_parts=[part], input_data=config or _config(), target_field=target)


def test_capsule_overlap_is_not_a_confirmed_conductor_intersection():
    first, second = _record(0, "a"), _record(0.0012, "b", part_index=2)
    report = assembly_clearance([first, second], backend="reference")
    assert report["checked_body_pairs"] == 1
    assert report["passed"]
    pair = report["pairs"][0]
    assert pair["conservative_envelope_overlap"]
    assert not pair["confirmed_conductor_intersection"]
    assert not pair["confirmed_clearance_violation"]


def test_insulation_failure_between_different_coil_parts():
    first, second = _record(0, "a"), _record(0.0012, "b", part_index=2)
    report = assembly_clearance([first, second], insulation_thickness_m=0.00015, backend="reference")
    assert not report["passed"]
    pair = report["pairs"][0]
    assert not pair["confirmed_conductor_intersection"]
    assert pair["status"] == "insulation_or_margin_violation"
    assert pair["minimum_surface_distance_m"] == pytest.approx(0.0002, abs=1e-12)


@pytest.mark.parametrize("offset,width", [(0.0008, 0.001), (0, 0.0005)])
def test_actual_copper_intersection_and_containment(offset, width):
    report = assembly_clearance([_record(0, "a"), _record(offset, "b", width=width)], backend="reference")
    assert not report["passed"]
    assert report["pairs"][0]["confirmed_conductor_intersection"]


def test_mounted_centroid_export_matches_swept_rings_and_field(tmp_path):
    profile = ((0.0, 0.0), (0.001, 0.0), (0.001, 0.002), (0.0, 0.002))
    result = _result(config=_config(cross_sectional_points=profile, conductor_mounting_mode="surface_outer"))
    original = result.coil_parts[0].wire_path.v.copy()
    report = compile_fabrication(result, tmp_path, kernel_backend="reference")
    record = report.paths[0]
    rings = record.mesh.vertices[: len(record.path.points_m) * 4].reshape(-1, 4, 3)
    np.testing.assert_allclose(record.path.points_m, rings.mean(axis=1), rtol=0, atol=1e-15)
    exported = np.loadtxt(tmp_path / "part001_body0001_fabrication_centroid.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(exported, record.points_m)
    frames = np.load(tmp_path / "part001_body0001_fabrication_frames.npz")
    np.testing.assert_array_equal(frames["normals"], record.path.normals)
    np.testing.assert_array_equal(frames["cross_section_m"], record.path.cross_section_m)
    np.testing.assert_array_equal(result.coil_parts[0].wire_path.v, original)
    assert not record.path.points_m.flags.writeable
    data = np.load(tmp_path / "fabrication_fields.npz")
    expected = 2.0 * biot_savart_wire_path(exported, result.target_field.coords)
    np.testing.assert_allclose(data["fabrication_field_T"], expected, rtol=1e-14, atol=1e-20)
    assert report.field_metrics["dsv_source_to_fabrication_delta_rms_T"] > 0
    assert report.electrical_metrics["bodies"][0]["cross_section_area_m2"] == pytest.approx(2e-6)
    assert report.electrical_metrics["bodies"][0]["length_m"] == pytest.approx(record.path.length_m)
    assert not report.ready
    assert not list(tmp_path.glob("*.stl"))
    assert any("external_return_not_modeled" in reason for reason in report.failure_reasons)
    assert json.loads((tmp_path / "fabrication_report.json").read_text())["ready"] is False


def test_valid_closed_loop_exports_strict_solids_and_closing_step(tmp_path):
    report = compile_fabrication(_result(closed=True), tmp_path, kernel_backend="reference")
    assert report.ready, report.failure_reasons
    assert len(list(tmp_path.glob("*.stl"))) == 1
    assert len(list(tmp_path.glob("*_faceted.step"))) == 1
    plan = report.winding_steps[0]
    assert plan["circuit_role"] == "closed_independent_turn"
    assert not plan["drive_feed_designed"]
    assert plan["steps"][-1]["to_marker"] == 1
    assert plan["steps"][-1]["instruction"].endswith("to 1.")
    assert sum(step["interval_length_m"] for step in plan["steps"]) == pytest.approx(report.paths[0].path.length_m)


def test_return_continuity_and_explicit_build_steps():
    record = _record(0, "a")
    start, finish = record.points_m[[0, -1]]
    missing = winding_instructions(record)
    assert not missing["circuit_complete"]
    with pytest.raises(ValueError, match="continuously"):
        winding_instructions(record, [start, finish])
    with pytest.raises(ValueError, match="zero-length"):
        winding_instructions(record, [finish, finish, start])
    return_points = [finish, finish + [0, 0.02, 0], start + [0, 0.02, 0], start]
    plan = winding_instructions(record, return_points)
    assert plan["circuit_complete"]
    assert plan["electrical_path_length_m"] > plan["winding_length_m"]
    assert plan["steps"][-1]["to_marker"] == 1
    assert sum(step["segment_role"] == "external_return" for step in plan["steps"]) == 1
    assert plan["steps"][-1]["operation"] == "route_return"
    assert len(plan["checkpoints"]) >= 4  # Return corners remain physical checkpoints.
    for before, after in zip(plan["steps"][:-1], plan["steps"][1:]):
        np.testing.assert_array_equal(before["end_xyz_m"], after["start_xyz_m"])


def test_explicit_return_participates_in_export_field_and_resistance(tmp_path):
    result = _result()
    start, finish = result.coil_parts[0].wire_path.v[[0, -1]]
    return_points = [finish, finish + [0, 0.02, 0], start + [0, 0.02, 0], start]
    report = compile_fabrication(
        result, tmp_path, kernel_backend="reference", leads={"part001_body0001": return_points}
    )
    points = np.loadtxt(tmp_path / "part001_body0001_electrical_path_with_return.csv", delimiter=",", skiprows=1)
    data = np.load(tmp_path / "fabrication_fields.npz")
    np.testing.assert_allclose(
        data["fabrication_field_T"], 2.0 * biot_savart_wire_path(points, result.target_field.coords), rtol=1e-14
    )
    assert report.electrical_metrics["bodies"][0]["length_m"] == pytest.approx(0.08)
    assert report.electrical_metrics["bodies"][0]["circuit_complete"]
    assert "part001_body0001:return_conductor_clearance_not_certified" in report.failure_reasons
    assert not report.ready


@pytest.mark.parametrize("current", [None, 0.0, float("nan")])
def test_physical_current_required(current):
    result = _result()
    result.coil_parts[0].contour_step = current
    with pytest.raises(ValueError, match="physical contour current"):
        compile_fabrication(result, kernel_backend="reference")


def test_fabrication_field_budget_is_rechecked_after_mounting():
    config = _config(conductor_mounting_mode="surface_outer", layout_target_error_rms_limit_T=1e-16)
    report = compile_fabrication(_result(config=config), kernel_backend="reference")
    assert "fabrication_field_rms_budget_exceeded" in report.failure_reasons
    assert not report.field_metrics["layout_budget_checks"]["rms"]["passed"]
