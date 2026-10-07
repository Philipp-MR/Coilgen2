"""Fast support bounds retain the actual finite faceted geometry."""

import numpy as np
import pytest

import coilgen.connector_bridge_validation as validation
from coilgen.faceted_support import faceted_cylinder_support
from coilgen.geometry import build_cylinder_mesh
from coilgen.joint_cylinder_curves import _cylinder_chart
from coilgen.manufacturing import _SurfaceProjector, _closest_point_and_barycentric


def _projector():
    return _SurfaceProjector(build_cylinder_mesh(0.25, 0.066, 8, 3, 1, 0, 0, 0))


def _segments(points):
    return np.stack((points[:-1], points[1:]), axis=1)


def _check(projector, points, lift=0.003):
    normal = projector.project(points[:1])[1][0]
    diagnostics = {}
    reasons = validation._validate_support(projector, points[0], _segments(points), normal, lift, 1e-8, diagnostics)
    return reasons, diagnostics


@pytest.mark.parametrize("rotate", [False, True])
def test_faceted_distances_match_all_actual_triangles_including_inside_and_rims(rotate):
    projector = _projector()
    if rotate:
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_rotvec([0.51, -0.22, 0.83]).as_matrix()
        translation = np.array([0.016, -0.019, 0.031])
        projector.vertices = projector.vertices @ rotation.T + translation
        projector.triangles = projector.vertices[projector.faces]
        projector.face_normals = projector.face_normals @ rotation.T
    support = faceted_cylinder_support(projector)
    assert support is not None
    rng = np.random.default_rng(78135)
    points = rng.uniform([-0.12, -0.12, -0.22], [0.12, 0.12, 0.22], (32, 3))
    if rotate:
        points = points @ rotation.T + translation
    projected = support.project(points)[0]
    for point, closest in zip(points, projected):
        reference = min(
            (_closest_point_and_barycentric(point, triangle)[0] for triangle in projector.triangles),
            key=lambda candidate: np.linalg.norm(point - candidate),
        )
        np.testing.assert_allclose(closest, reference, rtol=0, atol=2e-15)


def test_whole_faceted_surface_path_has_same_decision_as_adaptive_certificate(monkeypatch):
    projector = _projector()
    chart = _cylinder_chart(projector)
    developed = np.column_stack((np.r_[chart.starts, chart.perimeter], np.zeros(9), np.zeros(9)))
    points = chart.map(developed)
    accelerated, diagnostics = _check(projector, points)
    assert not accelerated
    assert diagnostics["support_certificate"] == "exact_convex_faceted_extrusion"
    monkeypatch.setattr(validation, "faceted_cylinder_support", lambda _projector: None)
    reference, _ = _check(projector, points)
    assert accelerated == reference


def test_lifted_faceted_chords_keep_distance_bound_and_adaptive_acceptance(monkeypatch):
    projector = _projector()
    chart = _cylinder_chart(projector)
    developed = np.array(
        [[0.2 * chart.lengths[0], 0, 0], [0.5 * chart.lengths[0], 0, 0], [0.8 * chart.lengths[0], 0, 0]]
    )
    points = chart.map(developed)
    points[1] += 0.0029 * np.cross(chart.directions[0], chart.axis)
    accelerated, diagnostics = _check(projector, points)
    assert not accelerated
    assert diagnostics["support_certificate"] == "exact_convex_faceted_extrusion"
    assert diagnostics["maximum_support_distance_m"] == pytest.approx(0.0029)
    monkeypatch.setattr(validation, "faceted_cylinder_support", lambda _projector: None)
    reference, _ = _check(projector, points)
    assert accelerated == reference


@pytest.mark.parametrize("failure", ["over_lift", "piercing", "rim", "inward"])
def test_ambiguous_or_invalid_chords_use_existing_adaptive_checks(monkeypatch, failure):
    projector = _projector()
    chart = _cylinder_chart(projector)
    points = chart.map(
        np.array([[0.2 * chart.lengths[0], 0, 0], [0.5 * chart.lengths[0], 0, 0], [0.8 * chart.lengths[0], 0, 0]])
    )
    lift = 0.003
    if failure == "over_lift":
        points[1] += 0.0031 * np.cross(chart.directions[0], chart.axis)
    elif failure == "piercing":
        points[1] -= 0.002 * np.cross(chart.directions[0], chart.axis)
    elif failure == "rim":
        points += (chart.upper + 0.002) * chart.axis
    else:
        points[1] -= 0.002 * np.cross(chart.directions[0], chart.axis)
        lift = -0.003
    accelerated, diagnostics = _check(projector, points, lift)
    assert diagnostics["support_certificate"] != "exact_convex_faceted_extrusion"
    monkeypatch.setattr(validation, "faceted_cylinder_support", lambda _projector: None)
    reference, _ = _check(projector, points, lift)
    assert accelerated == reference


def test_holes_and_changed_mesh_or_normals_never_reuse_a_cylinder_certificate():
    projector = _projector()
    first = faceted_cylinder_support(projector)
    assert first is not None and faceted_cylinder_support(projector) is first
    projector.face_normals[0] *= -1
    assert faceted_cylinder_support(projector) is None
    projector = _projector()
    projector.faces = projector.faces[1:]
    projector.triangles = projector.vertices[projector.faces]
    projector.face_normals = projector.face_normals[1:]
    assert faceted_cylinder_support(projector) is None
    projector = _projector()
    assert faceted_cylinder_support(projector) is not None
    projector.vertices[0, 0] += 0.001
    assert faceted_cylinder_support(projector) is None


def test_near_rim_raised_path_preserves_support_domain_tolerance(monkeypatch):
    projector = _projector()
    chart = _cylinder_chart(projector)
    developed = np.array(
        [
            [0.2 * chart.lengths[0], chart.upper - 1e-7, 0],
            [0.5 * chart.lengths[0], chart.upper - 1e-7, 0],
            [0.8 * chart.lengths[0], chart.upper - 1e-7, 0],
        ]
    )
    points = chart.map(developed)
    points[1] += 0.0029 * np.cross(chart.directions[0], chart.axis)
    accelerated, diagnostics = _check(projector, points)
    assert "support_domain_violation" in accelerated
    assert diagnostics["support_certificate"] != "exact_convex_faceted_extrusion"
    monkeypatch.setattr(validation, "faceted_cylinder_support", lambda _projector: None)
    reference, _ = _check(projector, points)
    assert accelerated == reference


def test_axial_vertex_edits_cannot_extend_stale_triangle_coverage():
    projector = _projector()
    assert faceted_cylinder_support(projector) is not None
    projector.vertices[:, 2] *= 2
    assert faceted_cylinder_support(projector) is None
    projector = _projector()
    projector.triangles[:, :, 2] *= 2
    assert faceted_cylinder_support(projector) is None


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_facet_plane_broadphase_preserves_original_triangle_intersections(monkeypatch, sign):
    projector = _projector()
    support = faceted_cylinder_support(projector)
    rng = np.random.default_rng(713581)
    segments = rng.uniform([-0.1, -0.1, -0.17], [0.1, 0.1, 0.17], (256, 2, 3))
    # Include nearly coplanar contacts, support corners and sub-trim chords.
    segments = np.concatenate((segments, projector.triangles[:, :2], projector.triangles[:, :2] + 1e-13))
    for segment in segments:
        accelerated = support.crosses_surface(segment[None], sign, 1e-8)
        with monkeypatch.context() as context:
            context.setattr(validation, "faceted_cylinder_support", lambda _projector: None)
            reference = validation._SpatialSupport(projector, 0.003, sign, 1e-8).crosses_surface(segment[None])
        assert accelerated == reference


def test_tangent_contamination_or_scaled_normals_disable_fast_certificates():
    projector = _projector()
    support = faceted_cylinder_support(projector)
    assert support is not None
    sides = np.argmax(projector.face_normals @ support.normals.T, axis=1)
    projector.face_normals += 10 * support.directions[sides]
    assert faceted_cylinder_support(projector) is None
    projector = _projector()
    projector.face_normals *= 2
    assert faceted_cylinder_support(projector) is None


def test_original_midpoint_distance_rejects_excess_lift_with_exact_negative_certificate(monkeypatch):
    projector = _projector()
    chart = _cylinder_chart(projector)
    developed = np.array(
        [[0.2 * chart.lengths[0], 0, 0], [0.5 * chart.lengths[0], 0, 0], [0.8 * chart.lengths[0], 0, 0]]
    )
    points = chart.map(developed)
    points[1] += 0.006 * np.cross(chart.directions[0], chart.axis)
    accelerated, diagnostics = _check(projector, points)
    assert accelerated == ["lift_limit_violation"]
    assert diagnostics["support_certificate"] == "exact_faceted_distance_violation"
    monkeypatch.setattr(validation, "faceted_cylinder_support", lambda _projector: None)
    reference, _ = _check(projector, points)
    assert accelerated == reference


def test_exact_intersections_keep_priority_over_negative_distance_rejection():
    projector = _projector()
    chart = _cylinder_chart(projector)
    developed = np.array(
        [[0.2 * chart.lengths[0], 0, 0], [0.5 * chart.lengths[0], 0, 0], [0.8 * chart.lengths[0], 0, 0]]
    )
    points = chart.map(developed)
    points = np.vstack(
        (
            points[0],
            points[1] + 0.01 * np.cross(chart.directions[0], chart.axis),
            points[1] - 0.01 * np.cross(chart.directions[0], chart.axis),
            points[2],
        )
    )
    reasons, _ = _check(projector, points)
    assert reasons == ["support_surface_intersection"]
