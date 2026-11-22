"""Shared fixtures for axial ring cut symmetry."""

from __future__ import annotations
import numpy as np
from coilgen.layout import (
    calculate_group_centers,
)
from coilgen.models import CoilPart, ContourLine, LoopGroup, Mesh


_SAMPLES = 48


_RADIUS = 0.1


def _rotation(tilted: bool) -> np.ndarray:
    if not tilted:
        return np.eye(3)
    axis = np.asarray([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = 0.63
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _ring(z: float, seam: int, reverse: bool, *, subdivide: bool = False) -> np.ndarray:
    angles = np.arange(_SAMPLES) * (2.0 * np.pi / _SAMPLES)
    points = np.column_stack((_RADIUS * np.cos(angles), _RADIUS * np.sin(angles), np.full(_SAMPLES, z)))
    if subdivide:
        # Add collinear samples, changing neither the polygon nor its enclosed area.
        points = np.insert(points, 6, points[5] + np.arange(1, 8)[:, None] / 8.0 * (points[6] - points[5]), axis=0)
    points = np.roll(points, -seam, axis=0)
    if reverse:
        points = points[::-1]
    return np.vstack((points, points[:1]))


def _fixture(*, seam: int = 0, reverse: bool = False, subdivide: bool = False, tilted: bool = False):
    rotation = _rotation(tilted)
    origin = np.asarray([0.31, -0.22, 0.4]) if tilted else np.zeros(3)

    def transform(points):
        return np.asarray(points) @ rotation.T + origin

    vertices = np.vstack((_ring(-0.15, 0, False)[:-1], _ring(0.15, 0, False)[:-1]))
    faces = []
    for first in range(_SAMPLES):
        following = (first + 1) % _SAMPLES
        faces.extend(((first, following, first + _SAMPLES), (following, following + _SAMPLES, first + _SAMPLES)))
    mesh = Mesh(
        vertices=transform(vertices),
        faces=np.asarray(faces),
        boundaries=[np.arange(_SAMPLES), np.arange(_SAMPLES, 2 * _SAMPLES)],
    )
    groups = []
    contours = []
    for group_index, levels in enumerate(((0.04, 0.065, 0.09), (-0.04, -0.065, -0.09))):
        loops = [
            ContourLine(
                transform(
                    _ring(z, seam + 7 * group_index + 3 * index, reverse != bool(group_index), subdivide=subdivide)
                ),
                float(index + 1) * (-1 if group_index else 1),
            )
            for index, z in enumerate(levels)
        ]
        first = len(contours)
        groups.append(
            LoopGroup(loops=loops, loop_indices=list(range(first, first + len(loops))), outer_loop_indices=(first,))
        )
        contours.extend(loops)
    part = CoilPart(mesh=mesh, contour_lines=contours, groups=groups, group_levels=[[0, 1]], level_positions=[[]])
    calculate_group_centers([part])
    axis = rotation[:, 2]
    radial = np.asarray([1.0, 0.0, 0.0]) - axis * axis[0]
    radial /= np.linalg.norm(radial)
    return part, origin, axis, radial
