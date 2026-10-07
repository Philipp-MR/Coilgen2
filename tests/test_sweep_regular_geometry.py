"""Regression screens for sampled frames and finite swept profiles."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.engineering_metrics import CurrentDensityReport
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import (
    _final_frame_and_profile_diagnostics,
    _SurfaceProjector,
    _polyline_circumradii,
    _sweep_corner_candidates,
    _validated_mesh_surface_fillet,
    create_swept_coil_bodies,
    save_manufacturing_diagnostics,
)
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry


def _plane():
    return normalize_mesh(
        Mesh(
            np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )


@pytest.mark.parametrize("geometry", list(WirePathGeometry))
def test_full_frame_audit_catches_a_planar_corner_for_every_route_type(geometry):
    points = np.array([[0.0, 0.0, 0.01], [0.05, 0.0, 0.01], [0.05, 0.05, 0.01]])
    if geometry == WirePathGeometry.SURFACE:
        points[:, 2] = 0.0
    part = CoilPart(mesh=_plane(), wire_path=WirePath(points, segment_geometry=[geometry] * 2))
    create_swept_coil_bodies([part], CoilGenConfig(skip_sweep=False, sweep_max_frame_rotation_deg=5.0))
    data = part.manufacturing_diagnostics
    assert data["sweep_final_frame_rotation_violation"]
    assert data["sweep_adaptive_max_frame_rotation_deg"] > 40.0
    assert "unresolved_frame_sampling" in data["manufacturing_failure_reasons"]


def test_fillet_respects_actual_frame_bound_including_free_spans():
    points = np.array([[0.0, 0.0, 0.01], [0.05, 0.0, 0.01], [0.05, 0.05, 0.01]])
    part = CoilPart(mesh=_plane(), wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * 2))
    create_swept_coil_bodies(
        [part],
        CoilGenConfig(
            skip_sweep=False,
            sweep_max_frame_rotation_deg=3.0,
            sweep_bend_arc_angle_deg=30.0,
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.005,
        ),
    )
    data = part.manufacturing_diagnostics
    assert data["manufacturing_ok"], data["manufacturing_failure_reasons"]
    assert data["sweep_final_max_frame_rotation_deg"] <= 3.0 + 1e-6
    assert data["sweep_profile_min_jacobian"] > 0.0
    np.testing.assert_array_equal(part.wire_path.v, points)


@pytest.mark.parametrize("half_width, mounting", [(0.002, 0.0), (0.012, 0.0), (0.002, 0.011)])
def test_profile_jacobian_matches_analytic_circle_and_mounting(half_width, mounting):
    radius = 0.01
    theta = np.arange(720) * 2 * np.pi / 720
    points = radius * np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    tangents = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros_like(theta)))
    normals = -points / radius  # Signed inward radial profile axis.
    transverse = np.cross(tangents, normals)
    section = np.array([[-0.001, -half_width], [0.001, -half_width], [0.001, half_width], [-0.001, half_width]])
    data = _final_frame_and_profile_diagnostics(
        points,
        tangents,
        normals,
        transverse,
        section,
        CoilGenConfig(),
        closed=True,
        mounting_offset=mounting,
    )
    expected = 1.0 - (half_width + mounting) / radius
    assert data["sweep_profile_min_jacobian"] == pytest.approx(expected, abs=2e-5)
    assert data["sweep_profile_jacobian_violation"] == (expected <= 0.0)


@pytest.mark.parametrize("step_degrees", [30.0, 5.0, 0.5])
def test_mesh_crease_radius_decision_is_independent_of_output_density(step_degrees):
    mesh = normalize_mesh(
        Mesh(
            np.array(
                [
                    [-0.05, -0.05, 0.0],
                    [-0.05, 0.05, 0.0],
                    [0.0, -0.05, 0.0],
                    [0.0, 0.05, 0.0],
                    [0.05, -0.05, 0.05],
                    [0.05, 0.05, 0.05],
                ]
            ),
            np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]]),
        )
    )
    points = np.array([[-0.03, 0.0, 0.0], [0.0, 0.0, 0.0], [0.03, 0.0, 0.03]])
    config = CoilGenConfig(sweep_bend_mode="fillet", sweep_min_bend_radius=0.005)
    candidate = _sweep_corner_candidates(points, [WirePathGeometry.SURFACE] * 2, 0.005, config, closed=False)[0]
    assert (
        _validated_mesh_surface_fillet(*points, candidate, 0.005, _SurfaceProjector(mesh), np.deg2rad(step_degrees))
        is None
    )


def test_mesh_facet_fillet_uses_bounded_analytic_offset_only_for_small_support_gap():
    def case(crease_degrees):
        slope = np.tan(np.deg2rad(crease_degrees))
        vertices = np.array([[x, y, max(x, 0.0) * slope] for x in (-0.02, 0.0, 0.02) for y in (-0.02, 0.02)])
        mesh = normalize_mesh(Mesh(vertices, np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]])))
        points = np.array([[-0.006, -0.006, 0.0], [0.0, 0.0, 0.0], [0.006, -0.006, 0.006 * slope]])
        config = CoilGenConfig(sweep_bend_mode="fillet", sweep_min_bend_radius=0.002)
        candidate = _sweep_corner_candidates(points, [WirePathGeometry.SURFACE] * 2, 0.002, config, closed=False)[0]
        return points, candidate, mesh, _SurfaceProjector(mesh)

    points, candidate, mesh, projector = case(3.0)
    assert _validated_mesh_surface_fillet(*points, candidate, 0.002, projector, np.deg2rad(8.0)) is None
    assert (
        _validated_mesh_surface_fillet(*points, candidate, 0.002, projector, np.deg2rad(8.0), max_support_gap_m=20e-6)
        is None
    )
    fillet = _validated_mesh_surface_fillet(
        *points, candidate, 0.002, projector, np.deg2rad(8.0), max_support_gap_m=50e-6
    )
    assert fillet is not None
    _start, _end, arc, gap_bound, _inflation, offset_from_surface = fillet
    assert offset_from_surface
    assert gap_bound <= 50e-6
    radii = _polyline_circumradii(np.asarray(arc), closed=False, minimum_turn_angle_deg=1e-7)
    assert len(radii) and np.min(radii) >= 0.99 * 0.002

    part = CoilPart(mesh=mesh, wire_path=WirePath(points, segment_geometry=[WirePathGeometry.SURFACE] * 2))
    create_swept_coil_bodies(
        [part],
        CoilGenConfig(
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.002,
            sweep_max_chord_error=50e-6,
            sweep_manufacturing_tolerance=50e-6,
        ),
    )
    assert part.manufacturing_diagnostics["sweep_offset_surface_fillet_count"] == 1
    assert WirePathGeometry.OFFSET_SURFACE in part.fabrication_path.segment_geometry
    assert len(part.fabrication_path.segment_geometry) == len(part.fabrication_path.points_m) - 1

    points, candidate, _mesh, projector = case(45.0)
    assert (
        _validated_mesh_surface_fillet(*points, candidate, 0.002, projector, np.deg2rad(8.0), max_support_gap_m=50e-6)
        is None
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_trapezoid_current_density_uses_conducting_area(transpose):
    profile = np.array([[-0.00098, -0.0009], [0.00098, -0.0009], [0.0008, 0.0009], [-0.0008, 0.0009]])
    config = CoilGenConfig(
        cross_sectional_points=profile.T if transpose else profile,
        conductor_cross_section_width=0.00196,
        conductor_cross_section_height=0.0018,
    )
    report = CurrentDensityReport.from_result(SimpleNamespace(input_data=config, needed_current_layout=10.0))
    assert report.conductor_area_m2 == pytest.approx(3.204e-6)
    assert report.current_density_A_per_mm2 == pytest.approx(10.0 / 3.204)
    assert "configured" in report.assumption


def test_shift_provenance_survives_export_without_pickle(tmp_path):
    before = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    after = before + [0.0, 0.0, 0.003]
    part = CoilPart(mesh=_plane(), wire_path=WirePath(after))
    part.normal_shift_diagnostics = {"normal_shift_status": "complete"}
    part.normal_shift_provenance = {
        "points_before_m": before.copy(),
        "points_after_m": after.copy(),
        "normal_face_indices": np.array([0, 1]),
        "index_convention": "zero_based_before_cleanup",
        "segment_owners": [{"kind": "inter_group_connector", "connector_index": 1}],
        "repair_events": [{"point_index": np.int64(1), "after_m": after[1]}],
    }
    files = save_manufacturing_diagnostics(SimpleNamespace(coil_parts=[part]), tmp_path, prefix="shift")
    assert len(files) == 3
    with np.load(files[-1], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["points_before_m"], before)
        np.testing.assert_array_equal(saved["points_after_m"], after)
        for key in saved.files:
            assert saved[key].dtype != object
        metadata = json.loads(str(saved["metadata_json"]))
    assert metadata["schema_version"] == 1
    assert metadata["length_unit"] == "m"
    assert metadata["index_convention"] == "zero_based_before_cleanup"
    assert metadata["segment_owners"][0]["connector_index"] == 1
    assert metadata["repair_events"][0]["after_m"] == after[1].tolist()
    np.testing.assert_array_equal(part.normal_shift_provenance["points_before_m"], before)
