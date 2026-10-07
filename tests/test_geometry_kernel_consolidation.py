"""Analytic segment cases protect routing clearance across physical scales."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry_kernels import (
    NUMBA_AVAILABLE,
    segment_closest_parameters,
    segment_distance,
    segment_distances,
    segment_pair_distances,
)
from coilgen.layout import (
    _segment_segment_distance_3d,
    _wire_segment_pair_index_distances,
    find_min_wire_segment_distance_3d,
)
from coilgen.manufacturing import _segment_distance


@pytest.mark.parametrize("scale", [1e-9, 1e-6, 1e-4, 1.0, 1e3])
@pytest.mark.parametrize("reverse", [False, True])
def test_analytic_distances_across_scalar_batched_and_compiled_paths(scale, reverse):
    first = np.repeat(np.array([[[0.0, 0, 0], [1, 0, 0]]]), 8, axis=0)
    second = np.array(
        [
            [[0.5, -0.5, 0], [0.5, 0.5, 0]],  # Interior crossing.
            [[0.5, -0.5, 0.25], [0.5, 0.5, 0.25]],  # Skew with known gap.
            [[0.25, 0, 0], [0.75, 0, 0]],  # Collinear overlap.
            [[2, 0, 0], [3, 0, 0]],  # Clamped endpoints.
            [[0, 0.5, 0], [1, 0.5, 0]],  # Parallel separated segments.
            [[0, -1e-10, 0], [1, 1e-10, 0]],  # Nearly parallel interior crossing.
            [[0.25, 0.5, 0], [0.25, 0.5, 0]],  # Point to segment.
            [[0, 0.5, 0], [0, 0.5, 0]],  # Point to point.
        ]
    )
    first[-1, 1] = first[-1, 0]
    first, second = scale * first, scale * second
    if reverse:
        first, second = second[:, ::-1], first[:, ::-1]
    expected = scale * np.array([0, 0.25, 0, 1, 0.5, 0, 0.5, 0.5])
    for scalar in (segment_distance, _segment_segment_distance_3d, _segment_distance):
        measured = [scalar(*a, *b) for a, b in zip(first, second)]
        np.testing.assert_allclose(measured, expected, atol=scale * 1e-14, rtol=1e-12)
    measured = segment_distances(first[:, 0], first[:, 1] - first[:, 0], second[:, 0], second[:, 1] - second[:, 0])
    np.testing.assert_allclose(measured, expected, atol=scale * 1e-14, rtol=1e-12)
    starts = np.concatenate((first[:, 0], second[:, 0]))
    directions = np.concatenate((first[:, 1] - first[:, 0], second[:, 1] - second[:, 0]))
    pairs = np.column_stack((np.arange(len(first)), np.arange(len(first), 2 * len(first))))
    np.testing.assert_allclose(
        _wire_segment_pair_index_distances(starts, directions, pairs), expected, atol=scale * 1e-14, rtol=1e-12
    )
    for backend in ("reference", "numpy", *(("numba",) if NUMBA_AVAILABLE else ())):
        np.testing.assert_allclose(
            segment_pair_distances(first, second, backend=backend).values_m, expected, atol=scale * 1e-14, rtol=1e-12
        )


@pytest.mark.parametrize("method", ["spatial", "all_pairs", "auto"])
@pytest.mark.parametrize("gap", [0.0, 0.1])
def test_fine_wire_routing_uses_the_interior_closest_points(method, gap):
    # The former m^4 cutoff returned 50 um clearance for this 100 um crossing.
    points = 1e-4 * np.array([[0.0, 0, 0], [1, 0, 0], [0.5, -0.5, gap], [0.5, 0.5, gap]])
    config = CoilGenConfig(wire_distance_method=method, wire_distance_chunk_size=1)
    first, second, distance = find_min_wire_segment_distance_3d(points, config)
    assert (first, second) == (0, 2)
    assert distance == pytest.approx(1e-4 * gap, abs=1e-17)


def test_default_core_geometry_does_not_import_research_or_numba():
    code = """
import sys
import numpy as np
from coilgen.manufacturing import _swept_mesh_self_intersection_diagnostics
from coilgen.models import Mesh
mesh = Mesh(vertices=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), faces=np.array([[0, 1, 2]]))
assert not _swept_mesh_self_intersection_diagnostics(mesh)['sweep_mesh_self_intersection']
assert 'numba' not in sys.modules
assert 'coilgen.research' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_closest_parameters_broadcast_and_reconstruct_interior_and_degenerate_contacts():
    starts = np.array([[0.0, -1, 0], [2, -1, 0], [0.5, 0.5, 0]])
    directions = np.array([[0.0, 2, 0], [0, 2, 0], [0, 0, 0]])
    other_start, other_direction = np.zeros(3), np.array([1.0, 0, 0])
    first, second = segment_closest_parameters(starts, directions, other_start, other_direction)
    np.testing.assert_allclose(first, [0.5, 0.5, 0])
    np.testing.assert_allclose(second, [0, 1, 0.5])
    expected = np.array([0.0, 1, 0.5])
    np.testing.assert_allclose(segment_distances(starts, directions, other_start, other_direction), expected)
    np.testing.assert_allclose(
        np.linalg.norm(starts + first[:, None] * directions - second[:, None] * other_direction, axis=1), expected
    )
