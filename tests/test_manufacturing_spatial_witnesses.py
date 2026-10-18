"""New locatable diagnostics refer to the actual, unchanged conductor."""

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import (
    _final_path_bend_diagnostics,
    _polygon_centroid,
    _support_conformity_diagnostics,
    _sweep_wire_path_with_diagnostics,
)
from coilgen.models import CoilPart, Mesh, WirePath


def plane():
    return normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )


@pytest.mark.parametrize("closed", [False, True])
def test_declared_ring_and_cap_order_reconstructs_exact_generated_vertices(closed):
    if closed:
        angle = np.linspace(0.0, 2 * np.pi, 25)
        points = np.column_stack((0.04 * np.cos(angle), 0.04 * np.sin(angle), np.zeros_like(angle)))
        points[-1] = points[0]
    else:
        points = np.array([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]])
    profile = np.array([[0.0001, -0.0002], [0.0006, -0.0002], [0.0005, 0.0007], [0.0001, 0.0007]])
    config = CoilGenConfig(
        cross_sectional_points=profile,
        conductor_mounting_mode="surface_outer",
        sweep_min_bend_radius=0.0,
        sweep_max_segment_length=0.01,
    )
    part = CoilPart(mesh=plane(), wire_path=WirePath(points))
    mesh, diagnostics = _sweep_wire_path_with_diagnostics(part, profile, config)
    frames = part.fabrication_path
    offsets = frames.cross_section_m - _polygon_centroid(frames.cross_section_m)
    rings = (
        frames.points_m[:, None, :]
        + offsets[None, :, :1] * frames.transverse[:, None, :]
        + offsets[None, :, 1:] * frames.normals[:, None, :]
    )
    expected = rings.reshape(-1, 3)
    if not closed:
        expected = np.vstack((expected, rings[0].mean(axis=0), rings[-1].mean(axis=0)))
    np.testing.assert_allclose(expected, mesh.vertices, atol=2e-17, rtol=0)
    assert diagnostics["sweep_support_vertex_order"] == "ring_major_profile_then_open_cap_centers"
    assert diagnostics["sweep_support_penetration_vertex_count"] == len(expected)


def test_support_penetration_witness_is_actual_worst_vertex():
    vertices = np.array([[0.0, 0.0, -0.0002], [0.01, 0.0, -0.0007], [0.0, 0.01, 0.0001]])
    body = Mesh(vertices, np.array([[0, 1, 2]]))
    part = CoilPart(mesh=plane())
    config = CoilGenConfig(conductor_mounting_mode="surface_outer", sweep_manufacturing_tolerance=0.00005)
    diagnostics = _support_conformity_diagnostics(part, body, vertices, config)
    assert diagnostics["sweep_support_penetration_violation"]
    assert diagnostics["sweep_support_penetration_vertex_index"] == 1
    assert diagnostics["sweep_support_max_penetration_m"] == pytest.approx(0.0007)


@pytest.mark.parametrize("closed", [False, True])
def test_bend_witness_tracks_original_point_indices_including_reversal(closed):
    profile = np.array([[-0.0001, -0.0001], [0.0001, -0.0001], [0.0001, 0.0001], [-0.0001, 0.0001]])
    config = CoilGenConfig(sweep_min_bend_radius=0.005)
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])
    diagnostics = _final_path_bend_diagnostics(points, profile, config, closed=closed)
    assert diagnostics["sweep_final_min_bend_point_index"] == 1
    assert diagnostics["sweep_final_bend_point_count"] == 4
    assert diagnostics["sweep_final_min_bend_radius_m"] == 0.0
    assert diagnostics["sweep_final_min_bend_radius_violation"]
