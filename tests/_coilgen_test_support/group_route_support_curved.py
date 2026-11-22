"""Shared fixtures for group route support curved."""

from coilgen.geometry import build_conical_frustum_mesh, normalize_mesh


def _frustum(bottom=0.02, top=0.02, count=16):
    return normalize_mesh(build_conical_frustum_mesh(0.06, bottom, top, count, 2, 1, 0, 0, 0, 0, 0, 0))
