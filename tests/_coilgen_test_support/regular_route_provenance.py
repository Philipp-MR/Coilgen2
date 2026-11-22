"""Shared fixtures for regular route provenance."""

from types import SimpleNamespace
import numpy as np
from coilgen.models import CoilPart, ConnectedGroup, LoopGroup, Mesh, WirePath


def _part():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    mesh = Mesh(vertices=points[:4], faces=np.array([[0, 1, 2], [0, 2, 3]]))
    part = CoilPart(mesh)
    part.groups = [LoopGroup(loops=[], loop_indices=[7], opened_loops=[SimpleNamespace(v=points[:2])])]
    part.connected_groups = [ConnectedGroup(points, points[2:], points[:3], points[:3])]
    part.wire_path = WirePath(points.copy())
    return part
