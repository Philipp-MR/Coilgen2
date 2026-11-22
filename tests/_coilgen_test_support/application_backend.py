"""Shared fixtures for application backend."""

from __future__ import annotations
import numpy as np
from coilgen import (
    CoilGenConfig,
    Mesh,
)
from coilgen.models import TargetField
from coilgen.object_registry import (
    register_in_memory_mesh,
    register_in_memory_target_field,
)


def _mesh(scale=1.0):
    return Mesh(
        vertices=scale * np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        faces=np.asarray([[0, 1, 2]]),
        metadata={"units": "SI", "name": "support"},
    )


def _target():
    return TargetField(
        coords=np.asarray([[0.0, 0.0, 0.02], [0.0, 0.01, 0.03]]),
        b=np.asarray([[1e-5, 2e-5, 3e-5], [0.0, 0.0, 0.0]]),
        weights=np.asarray([1.0, 0.25]),
        suppressed=np.asarray([False, True]),
        target_field_group_inds=np.asarray([1, 7]),
        target_gradient_dbdxyz=None,
        suppression_normals=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        suppression_area_weights=np.asarray([0.0, 0.004]),
        target_diagnostics={"source": "engineer", "nested": {"frame": "world"}},
    )


def _config():
    return CoilGenConfig(
        coil_mesh_file=register_in_memory_mesh(_mesh()),
        target_field_definition_file=register_in_memory_target_field(_target()),
        imported_mesh_length_scale_to_m=0.001,
        research_physics={"objective": "power", "objective_weights": {"power": 1.0, "minimax": 2.0}},
    )
