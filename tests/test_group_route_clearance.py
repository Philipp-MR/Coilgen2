"""Finished inter-part conductor surfaces require gap and volume separation."""

from dataclasses import replace
import builtins

import numpy as np
import pytest

from coilgen import group_route_clearance as audit
from coilgen.group_route_validation import GroupRouteValidationLimits
from coilgen.models import Mesh


LIMITS = GroupRouteValidationLimits()


def cube(half=0.001, center=(0.0, 0.0, 0.0)):
    vertices = (
        half
        * np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=float,
        )
        + center
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return Mesh(vertices, faces)


def combine(first, second):
    return Mesh(
        np.vstack((first.vertices, second.vertices)), np.vstack((first.faces, second.faces + len(first.vertices)))
    )


@pytest.mark.parametrize(
    "gap,required,passed",
    [
        (0.0001, 0.0002, False),
        (0.0002, 0.0002, True),
        (0.0002000005, 0.0002, True),
        (0.000199995, 0.0002, False),
        (1e-8, 0.0, True),
        (0.0, 0.0, False),
    ],
)
def test_parallel_nonintersecting_bodies_need_configured_surface_clearance(gap, required, passed):
    bodies = [cube(), cube(center=(0.0, 0.0, 0.002 + gap))]
    snapshots = [(body.vertices.copy(), body.faces.copy()) for body in bodies]
    diagnostics, reason = audit.audit_interpart_clearance(bodies, required, LIMITS)
    assert (reason == "") is passed, diagnostics
    assert diagnostics["interpart_clearance_completed"] is passed
    if not passed:
        assert reason == "interpart_surface_clearance_violation"
        assert diagnostics["interpart_witness_body_indices"] == [0, 1]
    for body, (vertices, faces) in zip(bodies, snapshots):
        np.testing.assert_array_equal(body.vertices, vertices)
        np.testing.assert_array_equal(body.faces, faces)


def test_crossing_bodies_have_no_shared_vertex_or_face_exemption():
    for second in (cube(center=(0.001, 0.0, 0.0)), cube()):
        diagnostics, reason = audit.audit_interpart_clearance([cube(), second], 0, LIMITS)
        assert reason == "interpart_surface_clearance_violation"
        assert diagnostics["interpart_witness_surface_distance_m"] == 0


def test_nested_solids_reject_even_without_surface_intersection():
    diagnostics, reason = audit.audit_interpart_clearance([cube(), cube(0.0003)], 0.0001, LIMITS)
    assert reason == "interpart_solid_containment", diagnostics
    assert diagnostics["interpart_containment_witness_body_indices"] == [1, 0]
    assert diagnostics["interpart_triangle_pairs_tested"] == 0
    assert not diagnostics["interpart_clearance_completed"]


def test_every_disconnected_component_is_tested_for_containment():
    outer = cube()
    mixed = combine(cube(0.0002, center=(-0.004, 0.0, 0.0)), cube(0.0002))
    diagnostics, reason = audit.audit_interpart_clearance([outer, mixed], 0, LIMITS)
    assert reason == "interpart_solid_containment", diagnostics


@pytest.mark.parametrize("reverse_orientation", [False, True])
def test_nested_target_components_use_material_union_not_parity_xor(reverse_orientation):
    target = combine(cube(0.003), cube(0.002))
    source = cube(0.001)
    if reverse_orientation:
        target.faces = target.faces[:, ::-1]
        source.faces = source.faces[:, ::-1]
    diagnostics, reason = audit.audit_interpart_clearance([target, source], 0, LIMITS)
    assert reason == "interpart_solid_containment", diagnostics
    assert diagnostics["interpart_containment_witness_body_indices"] == [1, 0]
    assert not diagnostics["interpart_clearance_completed"]


@pytest.mark.parametrize("reverse_orientation", [False, True])
def test_vertex_welded_nested_shells_do_not_cancel_containment_parity(reverse_orientation):
    outer = cube(0.003)
    tetra_vertices = 0.001 * np.array([[3, 3, 3], [1, 1, 2], [1, 2, 1], [2, 1, 1]], dtype=float)
    tetra_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    for face in tetra_faces:
        triangle = tetra_vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if normal @ (triangle.mean(axis=0) - tetra_vertices.mean(axis=0)) < 0:
            face[:] = face[::-1]
    target = combine(outer, Mesh(tetra_vertices, tetra_faces))
    # Both positive-volume shells share the actual outer-corner vertex index;
    # vertex connectivity must not XOR their material regions into a cavity.
    target.vertices, inverse = np.unique(target.vertices, axis=0, return_inverse=True)
    target.faces = inverse[target.faces]
    source = cube(0.00005, center=(0.00175, 0.00175, 0.00175))
    if reverse_orientation:
        target.faces = target.faces[:, ::-1]
        source.faces = source.faces[:, ::-1]
    diagnostics, reason = audit.audit_interpart_clearance([target, source], 0, LIMITS)
    assert reason == "interpart_solid_containment", diagnostics
    assert diagnostics["interpart_containment_witness_body_indices"] == [1, 0]
    assert diagnostics["interpart_minimum_checked_surface_distance_m"] > 0.0001
    assert not diagnostics["interpart_clearance_completed"]


def test_cavity_orientation_semantics_are_explicitly_unsupported_not_claimed_overlap():
    inner = cube(0.002)
    inner.faces = inner.faces[:, ::-1]
    target = combine(cube(0.003), inner)
    diagnostics, reason = audit.audit_interpart_clearance([target, cube(0.001)], 0, LIMITS)
    assert reason == "interpart_cavity_shell_semantics_unsupported", diagnostics
    assert not diagnostics["interpart_clearance_completed"]
    assert "interpart_containment_witness_point_m" not in diagnostics


def test_disjoint_bounds_certify_gap_without_assuming_cavity_material_semantics():
    inner = cube(0.002)
    inner.faces = inner.faces[:, ::-1]
    target = combine(cube(0.003), inner)
    diagnostics, reason = audit.audit_interpart_clearance([target, cube(0.001, center=(0.01, 0.0, 0.0))], 0.001, LIMITS)
    assert not reason, diagnostics
    assert diagnostics["interpart_clearance_completed"]


def test_overlapping_body_bounds_are_not_mistaken_for_solid_containment():
    first = combine(cube(0.0002, center=(-0.002, 0.0, 0.0)), cube(0.0002, center=(0.002, 0.0, 0.0)))
    second = combine(cube(0.0002, center=(0.0, -0.002, 0.0)), cube(0.0002, center=(0.0, 0.002, 0.0)))
    diagnostics, reason = audit.audit_interpart_clearance([first, second], 0.0001, LIMITS)
    assert not reason, diagnostics
    assert diagnostics["interpart_clearance_completed"] and diagnostics["interpart_containment_completed"]


def test_face_and_triangle_budgets_fail_closed_before_distance_work(monkeypatch):
    monkeypatch.setattr(
        audit, "triangle_pair_distances", lambda *_args: pytest.fail("Over-budget exact pair allocation")
    )
    bodies = [cube(), cube(center=(0.0, 0.0, 0.0021))]
    diagnostics, reason = audit.audit_interpart_clearance(bodies, 0.0002, replace(LIMITS, maximum_sweep_faces=23))
    assert reason == "interpart_face_budget_exceeded" and not diagnostics["interpart_clearance_completed"]
    diagnostics, reason = audit.audit_interpart_clearance(bodies, 0.0002, replace(LIMITS, maximum_triangle_pairs=10))
    assert reason == "interpart_triangle_pair_budget_exceeded" and not diagnostics["interpart_clearance_completed"]


def test_body_pair_budget_is_explicit():
    diagnostics, reason = audit.audit_interpart_clearance(
        [cube(), cube(), cube()], 0, replace(LIMITS, maximum_triangle_pairs=1)
    )
    assert reason == "interpart_body_pair_budget_exceeded"
    assert not diagnostics["interpart_clearance_completed"]


def test_containment_ray_budget_fails_closed_after_separated_surfaces():
    outer = cube()
    # Refine only to make the AABB tree separate the inner cube from each
    # outer surface before any leaf triangle-pair comparison is necessary.
    for _ in range(3):
        triangles = outer.vertices[outer.faces]
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        refined = np.concatenate(
            (
                np.stack((a, ab, ca), axis=1),
                np.stack((ab, b, bc), axis=1),
                np.stack((ca, bc, c), axis=1),
                np.stack((ab, bc, ca), axis=1),
            )
        )
        vertices, faces = np.unique(refined.reshape(-1, 3), axis=0, return_inverse=True)
        outer = Mesh(vertices, faces.reshape(-1, 3))
    diagnostics, reason = audit.audit_interpart_clearance(
        [outer, cube(0.0001)], 0, replace(LIMITS, maximum_triangle_pairs=10)
    )
    assert reason == "interpart_containment_pair_budget_exceeded", diagnostics
    assert diagnostics["interpart_triangle_candidates"] == 0
    assert not diagnostics["interpart_clearance_completed"]


def test_vertex_hit_ray_retries_instead_of_double_counting_shared_triangles():
    outer = cube()
    outer.vertices *= [1.0, np.sqrt(2), np.sqrt(3)]
    # The first tested inner vertex is the origin; the first parity ray passes
    # exactly through an outer corner. A second direction must resolve it.
    inner = cube(0.0001, center=(0.0001, 0.0001, 0.0001))
    diagnostics, reason = audit.audit_interpart_clearance([outer, inner], 0, LIMITS)
    assert reason == "interpart_solid_containment", diagnostics
    assert diagnostics["interpart_containment_triangles_tested"] >= 24


def test_numerically_unresolved_containment_cannot_report_clearance(monkeypatch):
    def unresolved(*_args, **_kwargs):
        raise audit._AuditIncomplete("interpart_containment_numerically_ambiguous")

    monkeypatch.setattr(audit, "_inside_body", unresolved)
    diagnostics, reason = audit.audit_interpart_clearance([cube(), cube(0.0001)], 0, LIMITS)
    assert reason == "interpart_containment_numerically_ambiguous"
    assert not diagnostics["interpart_clearance_completed"]


def test_no_scipy_dependency_and_distance_is_rotation_invariant(monkeypatch):
    real_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("Optional SciPy unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    kernel = audit.triangle_pair_distances
    monkeypatch.setattr(
        audit, "triangle_pair_distances", lambda first, second: kernel(first, second, backend="reference")
    )
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [2, -1, 4], [3, 4, -1]]))
    bodies = [cube(), cube(center=(0.0, 0.0, 0.0021))]
    for body in bodies:
        body.vertices = body.vertices @ rotation.T + [0.081, -0.043, 0.009]
    diagnostics, reason = audit.audit_interpart_clearance(bodies, 0.0002, LIMITS)
    assert reason == "interpart_surface_clearance_violation"
    assert diagnostics["interpart_minimum_checked_surface_distance_m"] == pytest.approx(0.0001, abs=1e-15)


def test_open_or_nonfinite_body_cannot_certify_a_solid_gap():
    open_body = cube()
    open_body.faces = open_body.faces[:-1]
    _, reason = audit.audit_interpart_clearance([cube(), open_body], 0, LIMITS)
    assert reason == "interpart_containment_requires_closed_manifold_bodies"
    bad = cube()
    bad.vertices[0, 0] = np.nan
    _, reason = audit.audit_interpart_clearance([cube(), bad], 0, LIMITS)
    assert reason == "invalid_interpart_body"


@pytest.mark.parametrize("clearance", [-1.0, np.nan, np.inf])
def test_invalid_clearance_never_authorizes_bodies(clearance):
    diagnostics, reason = audit.audit_interpart_clearance([cube(), cube()], clearance, LIMITS)
    assert reason == "invalid_interpart_clearance" and not diagnostics["interpart_clearance_completed"]
