"""Shared fixtures for viewport system meshes."""

from dataclasses import replace
import numpy as np
from coilgen.application import ProjectAssembly
from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config


def _project(name, radius, *, rotated=False, missing=False):
    config = replace(
        preset_config(),
        coil_mesh_file="none" if missing else "create cylinder mesh",
        field_shape_function="none",
        cylinder_mesh_parameter_list=(0.2125, radius, 12, 6, 1, 0, 0, 0),
        generated_mesh_rotation_axis=(1, 0, 0),
        generated_mesh_rotation_angle=np.pi / 2 if rotated else 0,
    )
    return ProjectDocument.from_config(config, name=name)


def _assembly(*, missing=False):
    assembly = ProjectAssembly.new(
        "Nested gradient supports",
        (_project("Gx", 0.08), _project("Gy", 0.09, rotated=True), _project("Gz", 0.10, missing=missing)),
    )
    return assembly.update_channel(assembly.channels[2].id, enabled=False)
