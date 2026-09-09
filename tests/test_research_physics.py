from dataclasses import replace

import numpy as np
import pytest

from coilgen.research.physics import (
    EnergyQuadrature,
    SurfaceOperators,
    force_torque_operators,
    surface_magnetic_energy,
    triangle_potential,
    validate_stream_function_topology,
)


def square_surface(scale=1.0):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float) * scale
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    tri = vertices[faces]
    return SurfaceOperators(
        tri, np.ones(2) * scale**2 / 2, tri.mean(axis=1), np.array([[[1.0], [0], [0]]] * 2) / scale, (2,)
    )


@pytest.mark.parametrize("values", [{"initial_order": 2.5}, {"maximum_order": True}, {"maximum_faces": 1.5}])
def test_energy_quadrature_requires_integer_work_limits(values):
    with pytest.raises(ValueError, match="integer"):
        EnergyQuadrature(**values)


def test_triangle_potential_analytic_corner_and_equilateral_center():
    right = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert triangle_potential(right[:1], right)[0] == pytest.approx(np.sqrt(2) * np.arcsinh(1), rel=1e-13)
    equilateral = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    assert triangle_potential(equilateral.mean(axis=0)[None], equilateral)[0] == pytest.approx(
        np.sqrt(3) * np.arcsinh(np.sqrt(3)), rel=1e-13
    )


def test_off_plane_triangle_potential_matches_independent_integral():
    from scipy.integrate import dblquad

    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    point = np.array([[0.2, 0.3, 0.4]])
    expected, _error = dblquad(
        lambda y, x: 1 / np.sqrt((x - 0.2) ** 2 + (y - 0.3) ** 2 + 0.4**2), 0, 1, lambda x: 0, lambda x: 1 - x
    )
    assert triangle_potential(point, triangle)[0] == pytest.approx(expected, rel=1e-10)


def test_energy_matches_analytic_square_integral_including_self_and_mutual():
    surface = square_surface()
    energy = surface_magnetic_energy(surface, EnergyQuadrature(relative_tolerance=1e-5, maximum_order=64))
    # 4 integral_0^1 integral_0^1 (1-x)(1-y)/sqrt(x*x+y*y) dxdy.
    integral = 4 * (np.arcsinh(1) + (1 - np.sqrt(2)) / 3)
    expected = 0.5e-7 * integral
    assert energy.energy_J(np.ones(1)) == pytest.approx(expected, rel=2e-5)
    assert energy.inductance_H(np.ones(1), 2.0) == pytest.approx(expected / 2, rel=2e-5)
    diagonal_only = sum(
        surface_magnetic_energy(
            SurfaceOperators(
                surface.triangles_m[i : i + 1],
                surface.areas_m2[i : i + 1],
                surface.centroids_m[i : i + 1],
                surface.current_density_per_A_m[i : i + 1],
                (1,),
            )
        ).energy_J(np.ones(1))
        for i in range(2)
    )
    assert energy.energy_J(np.ones(1)) > diagonal_only * 1.2


def test_energy_coordinate_scaling_translation_and_current_quadratic_law():
    base = surface_magnetic_energy(square_surface())
    scaled = surface_magnetic_energy(square_surface(0.01))
    surface = square_surface()
    moved = replace(surface, triangles_m=surface.triangles_m + [3, -2, 4], centroids_m=surface.centroids_m + [3, -2, 4])
    shifted = surface_magnetic_energy(moved)
    assert scaled.energy_J(np.ones(1)) == pytest.approx(base.energy_J(np.ones(1)) * 0.01, rel=1e-10)
    assert shifted.energy_J(np.ones(1)) == pytest.approx(base.energy_J(np.ones(1)), rel=1e-10)
    assert base.energy_J(np.array([3.0])) == pytest.approx(9 * base.energy_J(np.ones(1)))


def test_force_and_torque_are_exact_and_transform_with_explicit_origin():
    surface = square_surface()
    b0 = np.array([0, 0, 3.0])
    force, torque = force_torque_operators(surface, b0, np.zeros(3))
    assert force[:, 0] == pytest.approx([0, -3, 0])
    assert torque[:, 0] == pytest.approx([0, 0, -1.5])
    displacement = np.array([0.7, -0.2, 1.0])
    shifted = replace(
        surface, triangles_m=surface.triangles_m + displacement, centroids_m=surface.centroids_m + displacement
    )
    shifted_force, shifted_torque = force_torque_operators(shifted, b0, displacement)
    assert shifted_force == pytest.approx(force)
    assert shifted_torque == pytest.approx(torque)
    _, other_origin = force_torque_operators(surface, b0, displacement)
    assert other_origin[:, 0] == pytest.approx(torque[:, 0] - np.cross(displacement, force[:, 0]))


def test_nonuniform_prescribed_background_integrates_each_face():
    surface = square_surface()
    background = np.array([[0, 0, 2.0], [0, 0, 4.0]])
    force, torque = force_torque_operators(surface, background, np.zeros(3))
    expected_forces = np.cross(np.array([[1.0, 0, 0]] * 2), background) * 0.5
    assert force[:, 0] == pytest.approx(expected_forces.sum(axis=0))
    assert torque[:, 0] == pytest.approx(np.cross(surface.centroids_m, expected_forces).sum(axis=0))


def test_energy_force_and_torque_rotate_as_scalar_and_vectors():
    surface = square_surface()
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    rotated = replace(
        surface,
        triangles_m=surface.triangles_m @ rotation.T,
        centroids_m=surface.centroids_m @ rotation.T,
        current_density_per_A_m=np.einsum("ab,fbn->fan", rotation, surface.current_density_per_A_m),
    )
    b0, origin = np.array([0, 0, 3.0]), np.array([0.1, 0.2, 0.3])
    force, torque = force_torque_operators(surface, b0, origin)
    other_force, other_torque = force_torque_operators(rotated, rotation @ b0, rotation @ origin)
    assert other_force == pytest.approx(rotation @ force)
    assert other_torque == pytest.approx(rotation @ torque)
    assert surface_magnetic_energy(rotated).matrix_J_per_A2 == pytest.approx(
        surface_magnetic_energy(surface).matrix_J_per_A2
    )


def test_energy_rejects_nonconvergence_and_explicit_work_budget():
    with pytest.raises(ValueError, match="did not converge"):
        surface_magnetic_energy(
            square_surface(), EnergyQuadrature(initial_order=2, maximum_order=4, relative_tolerance=1e-12)
        )
    with pytest.raises(ValueError, match="maximum_faces"):
        surface_magnetic_energy(square_surface(), EnergyQuadrature(maximum_faces=1))


def test_torus_positive_genus_fails_instead_of_dropping_harmonic_currents():
    count = 6
    vertices = []
    faces = []
    for i in range(count):
        u = 2 * np.pi * i / count
        for j in range(count):
            v = 2 * np.pi * j / count
            vertices.append([(2 + np.cos(v)) * np.cos(u), (2 + np.cos(v)) * np.sin(u), np.sin(v)])
            a, b = i * count + j, ((i + 1) % count) * count + j
            c, d = ((i + 1) % count) * count + (j + 1) % count, i * count + (j + 1) % count
            faces.extend([[a, b, c], [a, c, d]])
    with pytest.raises(ValueError, match="Positive-genus.*harmonic"):
        validate_stream_function_topology(np.array(vertices), np.array(faces))
