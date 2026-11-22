"""Shared fixtures for group route support."""

import numpy as np
from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import create_swept_coil_bodies
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry


def _plane():
    return normalize_mesh(
        Mesh(
            np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0], [-0.02, 0.02, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )


def _swept_circle(*, mode="centered", height=0.0, lift=0.002, radius=0.006, x_offset=0.0):
    theta = np.linspace(0, 2 * np.pi, 181)
    points = np.column_stack((x_offset + radius * np.cos(theta), radius * np.sin(theta), np.full(len(theta), height)))
    config = CoilGenConfig(
        conductor_cross_section_width=0.0006,
        conductor_cross_section_height=0.0004,
        conductor_mounting_mode=mode,
        normal_shift_length=lift,
        sweep_min_bend_radius=0.0012,
        sweep_bend_mode="fillet",
        sweep_max_frame_rotation_deg=5.0,
        sweep_max_chord_error=5e-6,
    )
    part = CoilPart(
        mesh=_plane(),
        wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * (len(points) - 1)),
    )
    create_swept_coil_bodies([part], config)
    return part, config
