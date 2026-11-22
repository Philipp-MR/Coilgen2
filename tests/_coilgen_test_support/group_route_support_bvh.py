"""Shared fixtures for group route support bvh."""

import numpy as np
from coilgen.geometry import normalize_mesh
from coilgen.models import Mesh


def plane():
    axis = np.linspace(-0.03, 0.03, 9)
    vertices = np.array([[x, y, 0] for y in axis for x in axis])
    faces = []
    for y in range(8):
        for x in range(8):
            a = y * 9 + x
            faces.extend([[a, a + 1, a + 10], [a, a + 10, a + 9]])
    return normalize_mesh(Mesh(vertices, np.array(faces)))


def cylinder():
    theta = np.arange(48) * 2 * np.pi / 48
    ring = 0.02 * np.c_[np.cos(theta), np.sin(theta), np.zeros(48)]
    vertices = np.vstack((ring + [0, 0, -0.03], ring + [0, 0, 0.03]))
    faces = np.array(
        [[a, (a + 1) % 48, (a + 1) % 48 + 48] for a in range(48)] + [[a, (a + 1) % 48 + 48, a + 48] for a in range(48)]
    )
    return normalize_mesh(Mesh(vertices, faces))
