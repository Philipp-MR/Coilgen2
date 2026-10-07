"""Independent strict BVH broad-phase, predicate, and budget regressions."""

import numpy as np
import pytest

import coilgen.group_route_support as NEW
from coilgen.connector_bridge_validation import _SpatialSupport
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import _SurfaceProjector
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


@pytest.mark.parametrize("mesh_factory", [plane, cylinder])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_bvh_inclusive_candidates_and_crossing_boolean_equal_full_flat_reference(mesh_factory, sign):
    mesh = mesh_factory()
    before_vertices, before_faces = mesh.vertices.copy(), mesh.faces.copy()
    projector = _SurfaceProjector(mesh)
    old = _SpatialSupport(projector, 0.003, sign, 1e-8)
    new = NEW._CenteredSpatialSupport(projector, 0.003, sign, 1e-8)
    rng = np.random.default_rng(24851)
    segments = rng.uniform(-0.045, 0.045, (160, 2, 3))
    special = np.array(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1e-8, 0, 0]],
            [[-0.01, 0, 0.001], [0.01, 0, -0.001]],
            [[0.02, -0.01, 0], [0.02, 0.01, 0]],
            [[-0.04, 0.03, 0.001], [0.04, 0.03, -0.001]],
            [[0, 0, 1e-8], [0, 0, -1e-8]],
        ]
    )
    for start, end in np.concatenate((segments, special)):
        length = np.linalg.norm(end - start)
        if length > 2 * old.tolerance:
            first, last = start + old.tolerance * (end - start) / length, end - old.tolerance * (end - start) / length
            mask = np.all(old.low <= np.maximum(first, last), axis=1) & np.all(
                old.high >= np.minimum(first, last), axis=1
            )
            np.testing.assert_array_equal(new._candidate_faces(first, last), np.flatnonzero(mask))
        batch = np.stack((start, end))[None]
        assert new.crosses_surface(batch) == old.crosses_surface(batch)
    np.testing.assert_array_equal(mesh.vertices, before_vertices)
    np.testing.assert_array_equal(mesh.faces, before_faces)
    assert new.box_work <= new.maximum_box_work


@pytest.mark.parametrize(
    "kind",
    [
        "legal_lift",
        "contact",
        "within_contact_tolerance",
        "wrong_side",
        "crossing",
        "boundary_overhang",
        "raw_degenerate_wrong_side",
    ],
)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_strict_default_support_height_boundary_and_raw_source_semantics_are_unchanged(kind, sign):
    projector = _SurfaceProjector(plane())
    old = _SpatialSupport(projector, 0.003, sign, 1e-8)
    new = NEW._CenteredSpatialSupport(projector, 0.003, sign, 1e-8)
    segments = np.array([[[-0.005, 0, 0.001], [0.005, 0, 0.001]]])
    if kind == "contact":
        segments[:, :, 2] = 0
    elif kind == "within_contact_tolerance":
        segments[:, :, 2] = -5e-9
    elif kind == "wrong_side":
        segments[:, :, 2] = -2e-8
    elif kind == "crossing":
        segments[0, :, 2] = [-0.001, 0.001]
    elif kind == "boundary_overhang":
        segments[0, :, 0] = [0.027, 0.032]
    elif kind == "raw_degenerate_wrong_side":
        segments = np.array([[[0, 0, -2e-8], [0, 0, -2e-8]]])
    segments[:, :, 2] *= sign
    old_reasons = old.validate(segments, maximum_evaluations=4096)
    new_reasons = new.validate(segments, maximum_evaluations=4096)
    assert new_reasons == old_reasons
    assert new.evaluations == old.evaluations
    assert (new.minimum_height, new.maximum_height, new.maximum_distance) == (
        old.minimum_height,
        old.maximum_height,
        old.maximum_distance,
    )
    assert bool(new_reasons) == (kind in {"wrong_side", "crossing", "boundary_overhang", "raw_degenerate_wrong_side"})
    assert NEW._CenteredSpatialSupport.sample is _SpatialSupport.sample


def test_tree_construction_refuses_insufficient_budget_before_a_false_certificate():
    with pytest.raises(NEW._SupportScanBudgetExceeded):
        NEW._CenteredSpatialSupport(_SurfaceProjector(plane()), 0.003, 1.0, 1e-8, maximum_box_work=1)


def test_traversal_budget_exhaustion_is_explicit_and_partial_scan_is_not_cached_as_no_hit():
    checker = NEW._CenteredSpatialSupport(_SurfaceProjector(plane()), 0.003, 1.0, 1e-8)
    construction = checker.box_work
    checker.maximum_box_work = construction + 1
    crossing = np.array([[[-0.01, 0, 0.001], [0.01, 0, -0.001]]])
    with pytest.raises(NEW._SupportScanBudgetExceeded):
        checker.crosses_surface(crossing)
    assert checker.box_work == construction + 1
    assert checker.attempted_box_work > checker.maximum_box_work
    assert not checker._crossing_cache
    checker.maximum_box_work = 50_000_000
    assert checker.crosses_surface(crossing)


def test_only_identical_completed_batches_reuse_crossing_certificate():
    checker = NEW._CenteredSpatialSupport(_SurfaceProjector(plane()), 0.003, 1.0, 1e-8)
    segments = np.array([[[-0.005, 0, 0.001], [0.005, 0, 0.001]]])
    assert not checker.crosses_surface(segments)
    completed = checker.box_work
    assert not checker.crosses_surface(segments.copy())
    assert checker.box_work == completed
    segments[0, 0, 2] = -0.001
    assert checker.crosses_surface(segments)
    assert checker.box_work > completed
