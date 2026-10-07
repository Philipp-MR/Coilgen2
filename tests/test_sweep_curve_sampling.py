"""Preserve validated spatial curves when other route corners need fillets."""

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.connector_bridges import generate_connector_bridges
from coilgen.manufacturing import (
    _bend_analysis_skeleton,
    _build_sweep_rings,
    _coarsening_preserves_frame_bound,
    _frame_rotation_angles_deg,
    _polygon_centroid,
    _polyline_circumradii,
    _prepare_sweep_centerline,
    _safe_fold_coarsening_removals,
)
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry


def _saved_gz_bridge_with_distant_corner():
    # Actual pre-lift GZ connector 0 closing and its two contour neighbors.
    # Frozen metres, independent of the large outputs directory.
    source = np.array(
        [
            [0.08387906698195316, 0.029974521551022263, 0.03650515970337379],
            [0.08387734188330911, 0.030279507089369388, 0.03650932444320956],
            [0.0801581097810807, 0.04007697594782451, 0.045488345912875865],
            [0.08015406373548829, 0.0401555649884186, 0.04549811387967778],
            [0.08012142743593897, 0.040757863384129, 0.045576904490308996],
        ]
    )
    normal = np.array([0.9238795540451957, 7.471430119899305e-8, 0.3826833803776234])
    before, after = source[1] - source[0], source[-1] - source[-2]
    before /= np.linalg.norm(before)
    after /= np.linalg.norm(after)
    generated = generate_connector_bridges(
        source[1:4],
        before,
        after,
        normal,
        np.array([[[0.0806022838, 0.0301445586, 0.04441601545], [0.08381019733, 0.04027796716, 0.03667142031]]]),
        maximum_height_m=0.003,
        minimum_radius_m=0.002715436306690595,
        maximum_segment_length_m=0.005,
        chord_tolerance_m=5e-5,
    )
    assert generated.candidates
    bridge = generated.candidates[0].points
    side = np.cross(normal, after)
    # A distant corner forces the resolver to assemble a filleted
    # output from its analysis spine, exposing accidental bridge coarsening.
    corner = bridge[-1] + 0.03 * after
    path = np.vstack((bridge[0] - 0.02 * before, bridge, bridge[-1] + 0.01 * after, corner, corner + 0.03 * side))
    geometry = (
        [WirePathGeometry.SURFACE]
        + [WirePathGeometry.FREE_3D] * (len(bridge) - 1)
        + [WirePathGeometry.SURFACE]
        + [WirePathGeometry.FREE_3D] * 2
    )
    u = before - normal * np.dot(before, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    vertices = bridge[0] + np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]) @ np.stack((u, v))
    mesh = Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]]))
    part = CoilPart(mesh=mesh, wire_path=WirePath(path, segment_geometry=geometry))
    section = np.array([[-0.00098, -0.0009], [0.00098, -0.0009], [0.0008, 0.0009], [-0.0008, 0.0009]])
    config = CoilGenConfig(
        surface_is_cylinder_flag=False,
        cross_sectional_points=section,
        sweep_bend_mode="fillet",
        sweep_min_bend_radius=0.0025,
        sweep_max_curvature_envelope_ratio=0.5,
        sweep_max_chord_error=5e-5,
        sweep_max_frame_rotation_deg=5,
        sweep_max_segment_length=0.005,
        sweep_path_simplification="exact",
    )
    return part, bridge, section, config


def test_sampled_gz_bridge_keeps_frame_and_radius_bounds_through_fillet_and_ring_build():
    part, bridge, section, config = _saved_gz_bridge_with_distant_corner()
    original = part.wire_path.v.copy()
    points, geometry, diagnostics = _prepare_sweep_centerline(
        part, original, part.wire_path.segment_geometry, section, config, closed=False
    )
    assert diagnostics["sweep_corner_resolved_count"] >= 1
    points, tangent, normal, transverse, _vertices, _repair = _build_sweep_rings(
        part, points, geometry, section, config, closed=False, mounting_offset=0.0
    )
    first = int(np.argmin(np.linalg.norm(points - bridge[0], axis=1)))
    last = int(np.argmin(np.linalg.norm(points - bridge[-1], axis=1)))
    np.testing.assert_allclose(points[[first, last]], bridge[[0, -1]], atol=1e-12)
    angles = _frame_rotation_angles_deg(tangent, normal, transverse, closed=False)
    assert float(angles[first:last].max()) <= config.sweep_max_frame_rotation_deg + 1e-6
    center = _polygon_centroid(section)
    centroids = points + center[0] * transverse + center[1] * normal
    radii = _polyline_circumradii(centroids[first : last + 1], closed=False, minimum_turn_angle_deg=1e-6)
    assert radii.min() >= diagnostics["sweep_required_bend_radius_m"]
    np.testing.assert_array_equal(part.wire_path.v, original)


@pytest.mark.parametrize("geometry", [WirePathGeometry.FREE_3D, WirePathGeometry.OFFSET_SURFACE])
def test_spatial_analysis_keeps_curvature_but_still_reduces_straight_runs(geometry):
    angle = np.linspace(0, 0.8, 81)
    curve = np.column_stack((0.004 * np.sin(angle), 0.004 * (1 - np.cos(angle)), np.zeros(len(angle))))
    direction = np.array([np.cos(angle[-1]), np.sin(angle[-1]), 0])
    tail = curve[-1] + np.linspace(0, 0.03, 51)[1:, None] * direction
    points = np.vstack((curve, tail))
    result, result_geometry = _bend_analysis_skeleton(
        points, [geometry] * (len(points) - 1), CoilGenConfig(sweep_max_chord_error=5e-5), closed=False
    )
    assert len(result) >= len(curve)
    assert len(result) < len(points) - 40
    assert set(result_geometry) == {geometry}
    np.testing.assert_array_equal(result[[0, -1]], points[[0, -1]])


def test_spatial_analysis_retains_a_sharp_real_corner():
    points = np.vstack((np.column_stack((np.linspace(-0.02, 0, 21), np.zeros((21, 2)))), [[0, 0.01, 0], [0, 0.02, 0]]))
    result, _geometry = _bend_analysis_skeleton(
        points, [WirePathGeometry.FREE_3D] * (len(points) - 1), CoilGenConfig(), closed=False
    )
    np.testing.assert_allclose(result, [[-0.02, 0, 0], [0, 0, 0], [0, 0.02, 0]], atol=1e-15)


def test_coarsening_cannot_hide_new_local_frame_failure_behind_old_distant_crease():
    current = np.array([2.0, 2.0, 2.0, 35.0])
    kept = np.array([0, 2, 3, 4])
    assert not _coarsening_preserves_frame_bound(current, np.array([6.0, 2.0, 35.0]), kept, 5.0, closed=False)
    assert _coarsening_preserves_frame_bound(current, np.array([4.0, 2.0, 35.0]), kept, 5.0, closed=False)


def test_closed_coarsening_checks_merged_interval_across_removed_seam():
    current = np.array([2.0, 35.0, 2.0, 2.0, 2.0])
    kept = np.array([1, 2, 4])
    assert not _coarsening_preserves_frame_bound(current, np.array([35.0, 4.0, 6.0]), kept, 5.0, closed=True)
    assert _coarsening_preserves_frame_bound(current, np.array([35.0, 4.0, 4.0]), kept, 5.0, closed=True)


def test_existing_bad_crease_comparison_noise_does_not_veto_safe_mesh_repair():
    current, kept = np.array([2.0, 23.4832255]), np.arange(3)
    assert _coarsening_preserves_frame_bound(current, np.array([2.0, 23.4832581]), kept, 5.0, closed=False)
    assert not _coarsening_preserves_frame_bound(current, np.array([5.000002, 23.4832581]), kept, 5.0, closed=False)
    assert not _coarsening_preserves_frame_bound(current, np.array([2.0, 23.5]), kept, 5.0, closed=False)


def test_triangle_quality_repair_does_not_reapproximate_spatial_curve():
    points = np.array([[-0.001, 0, 0], [0, 0.00001, 0], [0.001, 0, 0]])
    config = CoilGenConfig(sweep_max_chord_error=5e-5)
    options = dict(closed=False, minimum_point_count=2, excluded=set(), maximum_removals=1)
    for geometry in (WirePathGeometry.FREE_3D, WirePathGeometry.OFFSET_SURFACE):
        assert not _safe_fold_coarsening_removals(points, [geometry] * 2, np.array([0]), config, **options)
    # Projectable surface spans retain the ordinary chord-error reduction.
    assert _safe_fold_coarsening_removals(points, [WirePathGeometry.SURFACE] * 2, np.array([0]), config, **options) == {
        1
    }
