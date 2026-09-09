"""Independent world-frame checks for generated ROI centring."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.models import CoilPart, Mesh
from coilgen.target import define_target_field
from coilgen.research.validation import validate_on_independent_grid


def planar_mesh():
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 0]], float) * 0.05
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return Mesh(vertices, faces, boundaries=[np.array([0, 1, 2, 3, 0])])


def test_generated_roi_translates_with_mesh_and_expression_uses_world_coordinates():
    mesh = planar_mesh()
    delta = np.array([0.1, -0.07, 0.03])
    config = CoilGenConfig(
        set_roi_into_mesh_center=True,
        target_region_resolution=5,
        field_shape_function="x + 2*y",
        target_gradient_normalization="none",
    )
    initial = define_target_field([mesh], None, None, config)
    shifted = define_target_field([replace(mesh, vertices=mesh.vertices + delta)], None, None, config)
    assert shifted.coords == pytest.approx(initial.coords + delta)
    assert shifted.coords.mean(axis=0) == pytest.approx(delta, abs=1e-15)
    assert shifted.b[:, 2] == pytest.approx(shifted.coords[:, 0] + 2 * shifted.coords[:, 1])
    assert shifted.b[:, 2] - initial.b[:, 2] == pytest.approx(np.full(len(initial.coords), delta[0] + 2 * delta[1]))
    imported_roi = Mesh(np.array([[0.4, 0.2, 0.3], [0.5, 0.2, 0.3], [0.4, 0.3, 0.3]]), np.array([[0, 1, 2]]))
    # Explicit target meshes retain their own world coordinates.
    explicit = define_target_field([mesh], imported_roi, None, config)
    explicit_without_centering = define_target_field(
        [mesh], imported_roi, None, replace(config, set_roi_into_mesh_center=False)
    )
    assert explicit.coords == pytest.approx(explicit_without_centering.coords)


def test_independent_grid_and_sheet_field_are_translation_covariant():
    delta = np.array([0.1, -0.07, 0.03])
    config = CoilGenConfig(
        set_roi_into_mesh_center=True,
        target_region_resolution=5,
        target_region_radius=0.02,
        field_shape_function="1e-6",
        target_gradient_normalization="none",
        gauss_order=3,
        progress_flag=False,
    )
    validations = []
    for displacement in (np.zeros(3), delta):
        mesh = planar_mesh()
        mesh.vertices += displacement
        part = CoilPart(mesh, stream_function=np.array([0, 0, 0, 0, 1.0]))
        result = SimpleNamespace(
            input_data=config,
            combined_mesh=mesh,
            coil_parts=[part],
            target_field=define_target_field([mesh], None, None, config),
            regularization_diagnostics={},
        )
        validations.append(validate_on_independent_grid(result, resolution=4))
    base, moved = validations
    assert moved["points_m"].mean(axis=0) == pytest.approx(delta, abs=1e-15)
    assert moved["points_m"] == pytest.approx(base["points_m"] + delta)
    assert moved["field_T"] == pytest.approx(base["field_T"], rel=1e-12, abs=1e-18)
    assert moved["target_T"] == pytest.approx(base["target_T"])
