"""Shared fixtures for geodesic cache."""

from __future__ import annotations
import numpy as np
from coilgen.models import Mesh


def _grid_mesh(size: int = 6) -> Mesh:
    vertices = np.asarray([[x, y, 0.0] for y in range(size) for x in range(size)])
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            faces.extend(((a, a + 1, a + size + 1), (a, a + size + 1, a + size)))
    return Mesh(vertices=vertices, faces=np.asarray(faces))
