"""Shared fixtures for regular ring traversal."""

import numpy as np
from coilgen.config import CoilGenConfig
from coilgen.contours import extract_contours
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.models import CoilPart
from coilgen.regular_rings import closed_curve_centroid


def _config(policy="regular"):
    config = CoilGenConfig(interconnection_cut_width=0.003, topological_grouping_method="auto")
    config.joint_group_routing_selection_policy = policy
    return config


def _part(samples=24, axial_samples=8):
    mesh = normalize_mesh(build_cylinder_mesh(0.4, 0.2, samples, axial_samples, 1, 0, 0, 0))
    part = CoilPart(mesh=mesh, stream_function=mesh.vertices[:, 2].copy())
    part.potential_level_list = np.linspace(-0.13, 0.13, 6)
    part.contour_step = float(np.diff(part.potential_level_list)[0])
    part = extract_contours([part], CoilGenConfig(min_point_loop_number=8))[0]
    # Reproduce extraction/potential order unrelated to axial adjacency.
    part.contour_lines = [part.contour_lines[index] for index in [0, 5, 1, 4, 2, 3]]
    for index, loop in enumerate(part.contour_lines):
        loop.potential = float(index)
    return part


def _positions(part):
    return [float(closed_curve_centroid(loop.v)[2]) for loop in part.groups[0].loops]
