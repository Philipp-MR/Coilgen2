"""Shared fixtures for view clarity."""

import numpy as np


def cylinder(radius, *, window=False, nphi=36, nz=12, height=0.2, offset=0):
    """Open cylinder about z; ``window`` removes a 60° × middle-third patch at φ = 0."""
    phi = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
    z = np.linspace(-height / 2, height / 2, nz)
    vertices = np.array([[radius * np.cos(p), radius * np.sin(p), h] for h in z for p in phi])
    faces = []
    for k in range(nz - 1):
        for j in range(nphi):
            a, b = k * nphi + j, k * nphi + (j + 1) % nphi
            c, d = a + nphi, b + nphi
            middle = (phi[j] + np.pi / nphi) % (2 * np.pi)
            centre = min(middle, 2 * np.pi - middle)
            if window and centre < np.radians(30) and nz // 3 <= k < 2 * nz // 3:
                continue
            faces += [[a, b, d], [a, d, c]]
    return vertices, np.asarray(faces) + offset


def pair(window=True):
    inner, inner_faces = cylinder(0.08, window=window)
    outer, outer_faces = cylinder(0.1, window=window, offset=len(inner))
    return {"vertices_m": np.vstack([inner, outer]), "faces": np.vstack([inner_faces, outer_faces])}
