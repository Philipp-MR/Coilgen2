"""Sweeping about the profile origin, and concave (omega) holder profiles."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen import CoilGenConfig, CoilPart, Mesh, WirePath, create_swept_coil_meshes
from coilgen.application.conductor_profiles import (
    omega_holder_profile,
    round_wire_profile,
    validate_conductor_profile,
)
from coilgen.config_validation import validate_config
from coilgen.manufacturing import _ear_clip_triangles, _is_convex_profile, _validated_cross_section


def _support():
    return Mesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]), faces=np.array([[0, 1, 2], [0, 2, 3]])
    )


def _config(**kwargs):
    return CoilGenConfig(
        conductor_cross_section_width=0.001, conductor_cross_section_height=0.001, sweep_max_chord_error=0, **kwargs
    )


def _open_part():
    return CoilPart(mesh=_support(), wire_path=WirePath(v=np.array([[-0.01, 0, 0], [0.01, 0, 0]])), contour_step=1.0)


def _closed_part():
    angle = np.linspace(0, 2 * np.pi, 65)
    points = np.column_stack((0.03 * np.cos(angle), 0.03 * np.sin(angle), np.zeros_like(angle)))
    points[-1] = points[0]
    return CoilPart(mesh=_support(), wire_path=WirePath(v=points), contour_step=1.0)


def _edge_counts(faces):
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return counts


def test_omega_and_round_presets_are_valid_simple_profiles():
    omega = omega_holder_profile(0.0015)
    summary = validate_conductor_profile(omega)
    assert not _is_convex_profile(_validated_cross_section(np.asarray(omega)))
    # The wire channel is centred on the origin, inside the holder's extent.
    assert summary.bounds_min_m[1] < -0.00075 < 0.00075 < summary.bounds_max_m[1] + 0.0008
    round_profile = round_wire_profile(0.0015)
    assert _is_convex_profile(_validated_cross_section(np.asarray(round_profile)))
    np.testing.assert_allclose(validate_conductor_profile(round_profile).centroid_m, (0, 0), atol=1e-15)


def test_ear_clipping_covers_a_concave_polygon_exactly():
    polygon = _validated_cross_section(np.asarray(omega_holder_profile(0.002)))
    triangles = _ear_clip_triangles(polygon)
    assert len(triangles) == len(polygon) - 2

    def triangle_area(a, b, c):
        (x1, y1), (x2, y2) = polygon[b] - polygon[a], polygon[c] - polygon[a]
        return 0.5 * abs(x1 * y2 - y1 * x2)

    area = sum(triangle_area(*triangle) for triangle in triangles)
    shoelace = 0.5 * abs(
        np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1])
    )
    assert area == pytest.approx(shoelace, rel=1e-12)


def test_path_reference_validation():
    errors = validate_config(_config(conductor_path_reference="profile_origin")).errors
    assert not [e for e in errors if e.field == "conductor_path_reference"]
    assert any(
        e.field == "conductor_path_reference" for e in validate_config(_config(conductor_path_reference="x")).errors
    )


@pytest.mark.parametrize("make_part", [_open_part, _closed_part], ids=["open", "closed"])
def test_omega_holder_sweeps_watertight(make_part):
    part = make_part()
    config = _config(cross_sectional_points=omega_holder_profile(0.0015), conductor_path_reference="profile_origin")
    create_swept_coil_meshes([part], config)
    faces = np.asarray(part.swept_body.faces)
    assert len(faces)
    assert np.all(_edge_counts(faces) == 2), "swept omega holder must be a closed 2-manifold"


def test_profile_origin_reference_exports_the_wire_path_itself():
    offset_square = ((0.0, 0.0), (0.001, 0.0), (0.001, 0.002), (0.0, 0.002))
    centroid_part, origin_part = _open_part(), _open_part()
    create_swept_coil_meshes([centroid_part], _config(cross_sectional_points=offset_square))
    create_swept_coil_meshes(
        [origin_part], _config(cross_sectional_points=offset_square, conductor_path_reference="profile_origin")
    )
    wire = np.asarray(origin_part.wire_path.v)
    origin_path = np.asarray(origin_part.fabrication_path.points_m)
    np.testing.assert_allclose(origin_path[:, 2], 0.0, atol=1e-12)
    assert np.max(np.abs(origin_path[:, 1] - wire[0, 1])) < 1e-12
    # The default still follows the profile centroid, 1 mm above the support.
    assert np.asarray(centroid_part.fabrication_path.points_m)[:, 2] == pytest.approx(0.001, abs=1e-9)
    assert origin_part.manufacturing_diagnostics["sweep_physics_geometry"] == "mounted_profile_origin"
    assert centroid_part.manufacturing_diagnostics["sweep_physics_geometry"] == "mounted_profile_centroid"
    # The solid itself is identical: only the reference path changes.
    np.testing.assert_allclose(origin_part.swept_body.vertices, centroid_part.swept_body.vertices, atol=1e-15)


def test_default_reference_keeps_previous_numbers():
    part = _closed_part()
    create_swept_coil_meshes([part], _config())
    baseline = np.asarray(part.fabrication_path.points_m).copy()
    again = _closed_part()
    create_swept_coil_meshes([again], replace(_config(), conductor_path_reference="centroid"))
    np.testing.assert_array_equal(np.asarray(again.fabrication_path.points_m), baseline)
