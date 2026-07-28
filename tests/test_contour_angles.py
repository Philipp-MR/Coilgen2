from __future__ import annotations

import numpy as np

from coilgen import (
    CoilPart,
    CoilGenConfig,
    ContourLine,
    Mesh,
    analyze_contour_angle_stability,
    intrinsic_contour_angle_trace,
    regularize_contour_paths,
    validate_intrinsic_topology_inputs,
)
from coilgen.contours import extract_contours
from coilgen.geometry import build_cylinder_mesh, normalize_mesh


def test_regular_circle_has_constant_intrinsic_curvature_and_no_high_frequency_ripple() -> None:
    mesh, contour = _fan_contour(np.full(16, 0.6, dtype=float))

    trace = intrinsic_contour_angle_trace(mesh, contour)

    assert trace is not None
    assert trace.closed
    np.testing.assert_allclose(
        trace.geodesic_curvature_per_m,
        np.full(16, trace.geodesic_curvature_per_m[0]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(trace.curvature_variation_per_m2, 0.0, atol=1.0e-11)
    np.testing.assert_allclose(trace.curvature_second_derivative_per_m3, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(trace.high_frequency_curvature_per_m, 0.0, atol=1.0e-12)


def test_alternating_radial_wiggle_is_visible_in_angle_stability_metrics() -> None:
    smooth_mesh, smooth_contour = _fan_contour(np.full(16, 0.6, dtype=float))
    wavy_radii = 0.6 * (1.0 + 0.08 * np.power(-1.0, np.arange(16)))
    wavy_mesh, wavy_contour = _fan_contour(wavy_radii)
    smooth_part = analyze_contour_angle_stability([CoilPart(mesh=smooth_mesh, contour_lines=[smooth_contour])])[0]
    wavy_part = analyze_contour_angle_stability([CoilPart(mesh=wavy_mesh, contour_lines=[wavy_contour])])[0]

    smooth = smooth_part.contour_level_diagnostics
    wavy = wavy_part.contour_level_diagnostics
    assert smooth["contour_angle_analyzed_count"] == 1
    assert wavy["contour_angle_analyzed_count"] == 1
    assert smooth["contour_high_frequency_curvature_rms_per_m"] < 1.0e-11
    assert wavy["contour_high_frequency_curvature_rms_per_m"] > 1.0
    assert wavy["contour_curvature_variation_rms_per_m2"] > smooth["contour_curvature_variation_rms_per_m2"]
    assert (
        wavy["contour_curvature_second_derivative_rms_per_m3"]
        > smooth["contour_curvature_second_derivative_rms_per_m3"]
    )


def test_intrinsic_turn_is_rotation_invariant_and_removes_triangle_fold() -> None:
    fold_angle = np.deg2rad(55.0)
    cosine = np.cos(fold_angle)
    sine = np.sin(fold_angle)
    vertices = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -cosine, -sine],
        ],
        dtype=float,
    )
    mesh = Mesh(vertices=vertices, faces=np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64))
    crossing = np.zeros(3, dtype=float)
    points = np.array(
        [
            [0.0, 0.25, 0.0],
            crossing,
            [0.0, -0.25 * cosine, -0.25 * sine],
        ],
        dtype=float,
    )
    contour = ContourLine(v=points, potential=0.0, segment_faces=np.array([0, 1], dtype=np.int64))
    trace = intrinsic_contour_angle_trace(mesh, contour)

    rotation = _rotation_matrix(0.37, -0.61)
    rotated_mesh = Mesh(vertices=mesh.vertices @ rotation.T, faces=mesh.faces.copy())
    rotated_contour = ContourLine(
        v=contour.v @ rotation.T,
        potential=0.0,
        segment_faces=contour.segment_faces.copy(),
    )
    rotated_trace = intrinsic_contour_angle_trace(rotated_mesh, rotated_contour)

    assert trace is not None
    assert rotated_trace is not None
    np.testing.assert_allclose(trace.turn_angle_rad, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(rotated_trace.turn_angle_rad, trace.turn_angle_rad, atol=1.0e-12)
    np.testing.assert_allclose(
        rotated_trace.geodesic_curvature_per_m,
        trace.geodesic_curvature_per_m,
        atol=1.0e-12,
    )


def test_parameterized_bspline_smoothing_is_regular_and_exactly_on_surface() -> None:
    radii = 0.6 * (1.0 + 0.08 * np.power(-1.0, np.arange(16)))
    mesh, contour = _fan_contour(radii)
    part = CoilPart(mesh=mesh, contour_lines=[contour])
    before = intrinsic_contour_angle_trace(mesh, contour)

    regularize_contour_paths(
        [part],
        CoilGenConfig(
            contour_path_regularization_mode="parameterized_bspline",
            contour_smoothing_factor=0.25,
            contour_smoothing_max_displacement=0.01,
            contour_smoothing_preserve_min_turn_distance=False,
            min_point_loop_number=3,
        ),
    )

    smoothed = part.contour_lines[0]
    after = intrinsic_contour_angle_trace(mesh, smoothed)
    assert before is not None
    assert after is not None
    assert part.wire_path is None
    assert part.contour_level_diagnostics["contour_path_smoothing_scope"] == "unconnected_contours"
    assert part.contour_level_diagnostics["contour_path_smoothing_applied"]
    assert part.contour_level_diagnostics["contour_path_smoothing_method"] == (
        "harmonic_parameterized_periodic_bspline"
    )
    assert part.contour_level_diagnostics["contour_path_smoothing_max_applied_displacement_m"] <= 0.01
    assert part.contour_level_diagnostics["contour_path_smoothing_segment_length_cv_after"] < 0.01
    assert _weighted_rms(after.high_frequency_curvature_per_m, after.sample_weight_m) < _weighted_rms(
        before.high_frequency_curvature_per_m,
        before.sample_weight_m,
    )
    assert len(smoothed.v) > len(contour.v)
    assert smoothed.surface_edge_vertices is None
    assert smoothed.surface_edge_t is None
    for face, barycentric, point in zip(smoothed.surface_faces, smoothed.surface_barycentric, smoothed.v):
        reconstructed = barycentric @ mesh.vertices[mesh.faces[int(face)]]
        np.testing.assert_allclose(point, reconstructed, atol=1.0e-12)
    for segment_index, face in enumerate(smoothed.segment_faces):
        triangle = mesh.vertices[mesh.faces[int(face)]]
        reconstructed = smoothed.segment_barycentric[segment_index] @ triangle
        np.testing.assert_allclose(smoothed.v[segment_index : segment_index + 2], reconstructed, atol=1.0e-12)
    assert validate_intrinsic_topology_inputs(part, check_separating=True).ok


def test_parameterized_bspline_smoothing_handles_curved_annular_mesh_patch() -> None:
    mesh = normalize_mesh(build_cylinder_mesh(0.25, 0.065, 32, 20, 1, 0, 0, 0))
    part = CoilPart(mesh=mesh)
    azimuth = np.arctan2(mesh.vertices[:, 1], mesh.vertices[:, 0])
    part.stream_function = mesh.vertices[:, 2] - 0.025 * np.sin(2.0 * azimuth)
    part.potential_level_list = np.array([0.0], dtype=float)
    part.contour_step = 1.0
    extract_contours([part], CoilGenConfig(min_point_loop_number=8))
    assert len(part.contour_lines) == 1

    regularize_contour_paths(
        [part],
        CoilGenConfig(
            contour_path_regularization_mode="parameterized_bspline",
            contour_resample_target_segment_length=0.003,
            contour_smoothing_factor=0.1,
            contour_smoothing_patch_rings=4,
            contour_smoothing_max_displacement=0.002,
            contour_smoothing_preserve_min_turn_distance=False,
            min_point_loop_number=8,
        ),
    )

    diagnostics = part.contour_level_diagnostics
    contour = part.contour_lines[0]
    assert diagnostics["contour_path_smoothing_applied"]
    assert diagnostics["contour_path_smoothing_max_applied_displacement_m"] <= 0.002
    assert (
        diagnostics["contour_path_smoothing_segment_length_cv_after"]
        < diagnostics["contour_path_smoothing_segment_length_cv_before"]
    )
    assert validate_intrinsic_topology_inputs(part).ok
    for face, barycentric, point in zip(contour.surface_faces, contour.surface_barycentric, contour.v):
        np.testing.assert_allclose(
            point,
            barycentric @ mesh.vertices[mesh.faces[int(face)]],
            atol=1.0e-12,
        )


def test_mesh_coupled_smoothing_scales_with_cylinder_mesh_and_allows_disabled_displacement_cap() -> None:
    coarse_part = _cylindrical_test_contour(32, 20)
    fine_part = _cylindrical_test_contour(64, 40)

    for part in (coarse_part, fine_part):
        regularize_contour_paths(
            [part],
            CoilGenConfig(
                contour_path_regularization_mode="parameterized_bspline",
                contour_smoothing_scale_mode="mesh_coupled",
                contour_smoothing_mesh_gain=0.1,
                contour_smoothing_work_samples_per_mesh_length=6,
                contour_smoothing_output_samples_per_mesh_length=8,
                contour_resample_target_segment_length=0.01,
                contour_smoothing_max_displacement=0.0,
                contour_smoothing_preserve_min_turn_distance=False,
                min_point_loop_number=8,
            ),
        )

    coarse = coarse_part.contour_level_diagnostics
    fine = fine_part.contour_level_diagnostics
    assert coarse["contour_path_smoothing_applied"]
    assert fine["contour_path_smoothing_applied"]
    assert coarse["contour_path_smoothing_scale_mode"] == "mesh_coupled"
    assert not coarse["contour_path_smoothing_displacement_cap_enabled"]
    assert coarse["contour_path_smoothing_mesh_gain_requested"] == 0.1
    assert np.isclose(coarse["contour_path_smoothing_mesh_gain_applied"], 0.1)
    assert coarse["contour_path_smoothing_max_applied_displacement_m"] > 0.0
    assert fine["contour_path_smoothing_mesh_scale_m"] < 0.6 * coarse["contour_path_smoothing_mesh_scale_m"]
    assert (
        fine["contour_path_smoothing_output_target_segment_length_m"]
        < 0.6 * coarse["contour_path_smoothing_output_target_segment_length_m"]
    )
    assert validate_intrinsic_topology_inputs(coarse_part).ok
    assert validate_intrinsic_topology_inputs(fine_part).ok


def _cylindrical_test_contour(angular_divisions: int, length_divisions: int) -> CoilPart:
    mesh = normalize_mesh(
        build_cylinder_mesh(
            0.25,
            0.065,
            angular_divisions,
            length_divisions,
            1,
            0,
            0,
            0,
        )
    )
    part = CoilPart(mesh=mesh)
    azimuth = np.arctan2(mesh.vertices[:, 1], mesh.vertices[:, 0])
    mesh_wiggle = 0.0015 * np.power(-1.0, np.arange(len(mesh.vertices)) % angular_divisions)
    part.stream_function = mesh.vertices[:, 2] - 0.025 * np.sin(2.0 * azimuth) + mesh_wiggle
    part.potential_level_list = np.array([0.0], dtype=float)
    part.contour_step = 1.0
    extract_contours([part], CoilGenConfig(min_point_loop_number=8))
    assert len(part.contour_lines) == 1
    return part


def _fan_contour(radii: np.ndarray) -> tuple[Mesh, ContourLine]:
    radii = np.asarray(radii, dtype=float)
    count = len(radii)
    angles = 2.0 * np.pi * np.arange(count, dtype=float) / count
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count, dtype=float)))
    mesh = Mesh(
        vertices=np.vstack((np.zeros((1, 3), dtype=float), directions)),
        faces=np.asarray([[0, index + 1, (index + 1) % count + 1] for index in range(count)], dtype=np.int64),
    )
    unique_points = radii[:, None] * directions
    edge_vertices = np.asarray([[0, index + 1] for index in range(count)] + [[0, 1]], dtype=np.int64)
    edge_t = np.concatenate((radii, radii[:1]))
    segment_barycentric = np.asarray(
        [
            [
                [1.0 - radii[index], radii[index], 0.0],
                [1.0 - radii[(index + 1) % count], 0.0, radii[(index + 1) % count]],
            ]
            for index in range(count)
        ],
        dtype=float,
    )
    surface_barycentric = np.asarray(
        [[1.0 - radii[index], radii[index], 0.0] for index in range(count)] + [[1.0 - radii[0], radii[0], 0.0]],
        dtype=float,
    )
    contour = ContourLine(
        v=np.vstack((unique_points, unique_points[:1])),
        potential=0.0,
        surface_faces=np.concatenate((np.arange(count, dtype=np.int64), np.asarray([0], dtype=np.int64))),
        surface_barycentric=surface_barycentric,
        surface_edge_vertices=edge_vertices,
        surface_edge_t=edge_t,
        segment_faces=np.arange(count, dtype=np.int64),
        segment_barycentric=segment_barycentric,
    )
    return mesh, contour


def _rotation_matrix(x_angle: float, z_angle: float) -> np.ndarray:
    cosine_x, sine_x = np.cos(x_angle), np.sin(x_angle)
    cosine_z, sine_z = np.cos(z_angle), np.sin(z_angle)
    rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, cosine_x, -sine_x], [0.0, sine_x, cosine_x]])
    rotate_z = np.array([[cosine_z, -sine_z, 0.0], [sine_z, cosine_z, 0.0], [0.0, 0.0, 1.0]])
    return rotate_z @ rotate_x


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(values)) / np.sum(weights)))
