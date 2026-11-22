"""Shared fixtures for group route support cap."""

import numpy as np
from coilgen.geometry import build_spherical_cap_mesh, normalize_mesh


def cap():
    return normalize_mesh(build_spherical_cap_mesh(0.08, np.pi / 2, 24, 8, 0, 0, 1, 0, 0, 0, 0))
