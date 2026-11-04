"""Analytic, witness, and exhaustive differential checks for the STL audit."""

from itertools import combinations

import numpy as np
from coilgen.geometry_kernels import triangle_pair_distances

from surface_gap import ACCEPTANCE_EPS, Body, closest_features, compare
from skin_contact import classify, inspect_skin
from assembly_context import audit_context


def test_nearly_parallel_disjoint_faces_with_overlapping_boxes():
    # The faces are only picometres from the opposing infinite planes, but
    # their actual 3D triangles end 0.359 mm apart. These are rounded STL
    # vertices from the Gz fixture, including the tiny plane angle.
    a = np.array(
        [
            [-0.05000000001, 0.02817989430, -0.008304683305],
            [-0.05000000001, 0.02839635955, -0.007979385211],
            [-0.04999999999, 0.02984234852, -0.009416549291],
        ]
    )
    b = np.array(
        [
            [-0.05000000001, 0.02857269738, -0.007666886756],
            [-0.05000000001, 0.02873311318, -0.007326773518],
            [-0.04999999999, 0.03055658255, -0.008148332693],
        ]
    )
    distance = closest_features(a[None], b[None])[0][0]
    assert 0.000358 < distance < 0.000360
    assert not classify(a[None], b[None])[0][0]


def test_nearly_parallel_facets_that_really_pierce():
    a = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    b = np.array([[0.002, 0.002, -1e-11], [0.006, 0.002, 1e-11], [0.002, 0.006, 1e-11]])
    hit, pierces, patch, point, _ = classify(a[None], b[None])
    assert hit[0] and pierces[0] and not patch[0]
    assert abs(point[0, 2]) < 1e-13
    assert np.isclose(point[0, 0] + point[0, 1], 0.006, atol=1e-12)


def box(name, z, height=0.001, length=0.02, x=0.0):
    profile = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * height / 2
    points = np.array([[x, 0, z], [x + length, 0, z]])
    rings = np.stack([np.column_stack((np.full(4, p[0]), profile[:, 0], profile[:, 1] + z)) for p in points])
    triangles = []
    for i in range(4):
        j = (i + 1) % 4
        triangles.extend([[rings[0, i], rings[0, j], rings[1, j]], [rings[0, i], rings[1, j], rings[1, i]]])
    for i in range(4):
        j = (i + 1) % 4
        triangles.extend([[points[0], rings[0, j], rings[0, i]], [points[1], rings[1, i], rings[1, j]]])
    return Body(name, np.array(triangles), profile, points, np.array(["straight"]), False, 0.0001, 0.00005, 0.0)


def test_required_actual_gap_and_witnesses():
    a = box("lower", 0)
    for gap, fail in [
        (0, True),
        (0.0002, True),
        (0.00025, False),
        (0.00025 - ACCEPTANCE_EPS / 2, False),
        (0.0003, False),
    ]:
        r = compare(a, box("upper", 0.001 + gap))
        w = r["witnesses"]["minimum"]
        assert np.isclose(w["distance_m"], gap, atol=1e-15)
        assert np.isclose(
            np.linalg.norm(np.subtract(w["first"]["world_xyz_m"], w["second"]["world_xyz_m"])), gap, atol=1e-15
        )
        assert r["failed"] is fail
    assert compare(a, box("inside", 0, height=0.0005, length=0.01, x=0.005))["contained"]
    assert compare(a, a)["status"] == "no_nonlocal_intervals"


def test_mixed_assembly_certifies_remote_slabs_and_measures_close_pairs():
    a, b, c = box("a", 0.0), box("b", 0.00120), box("c", 0.006)
    rows = audit_context({"A": [a], "B": [b], "C": [c]})
    assert len(rows) == 3
    close = next(r for r in rows if r["failed"])
    assert close["method"] == "finite_stl_facets"
    assert np.isclose(close["witnesses"]["minimum"]["distance_m"], 0.00020)
    assert all(r["method"] == "disjoint_world_boxes" for r in rows if not r["failed"])
    replacement = box("b-new", 0.00130)
    rows = audit_context({"A": [a], "B": [replacement], "C": [c]}, changed={"B"})
    assert len(rows) == 2 and not any(r["failed"] for r in rows)


def test_closest_features():
    rng = np.random.default_rng(1253)
    for planar in (False, True):
        a, b = rng.normal(size=(2, 2400, 3, 3))
        if planar:
            a[:, :, 2] = b[:, :, 2] = 0
        d, p, q = closest_features(a, b)
        reference = triangle_pair_distances(a, b, backend="numpy").values_m
        np.testing.assert_allclose(d, reference, atol=1e-12)
        np.testing.assert_allclose(d, np.linalg.norm(p - q, axis=1), atol=1e-14)
        for tri, pt in ((a, p), (b, q)):
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            np.testing.assert_allclose(np.einsum("ij,ij->i", pt - tri[:, 0], normal), 0.0, atol=1e-12)
            for i in range(3):
                side = np.cross(tri[:, (i + 1) % 3] - tri[:, i], pt - tri[:, i])
                assert np.min(np.einsum("ij,ij->i", side, normal)) > -1e-12


def test_nonlocal_and_bvh():
    # A synthetic closed polygon with deliberately colliding opposite visits.
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    profile = np.array([[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01]])
    v = (
        points[:-1, None]
        + profile[None, :, 0, None] * np.array([0.0, 1.0, 0.0])
        + profile[None, :, 1, None] * np.array([0.0, 0.0, 1.0])
    )
    triangles = []
    for s in range(4):
        for i in range(4):
            k, j = (s + 1) % 4, (i + 1) % 4
            triangles.extend([[v[s, i], v[s, j], v[k, j]], [v[s, i], v[k, j], v[k, i]]])
    body = Body(
        "closed bow",
        np.array(triangles),
        profile,
        points,
        np.array(["one", "two", "three", "four"]),
        True,
        0.01,
        0.02,
        0.0,
    )
    row = compare(body, body)
    pairs = np.array(list(combinations(range(len(body.triangles)), 2)))
    a, b = pairs[body.separation(*pairs.T) > 4 * body.diameter].T
    d = triangle_pair_distances(body.triangles[a], body.triangles[b], backend="numpy").values_m
    assert np.isclose(row["witnesses"]["minimum"]["distance_m"], d.min())
    assert row["facet_pair_counts"]["contact"] == np.count_nonzero(d <= 1e-12)
    assert row["facet_pair_counts"]["positive_subminimum"] == np.count_nonzero(
        (d > 1e-12) & (d < 0.04 - ACCEPTANCE_EPS)
    )


def test_short_hairpin_and_incident_skin():
    p = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.021, 0.0, 0.0012], [0.0005, 0.0, 0.0012]])
    profile = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.0005
    v = (
        p[:, None]
        + profile[None, :, 0, None] * np.array([0.0, 1.0, 0.0])
        + profile[None, :, 1, None] * np.array([0.0, 0.0, 1.0])
    )
    tri = []
    for s in range(3):
        for i in range(4):
            j = (i + 1) % 4
            tri.extend([[v[s, i], v[s, j], v[s + 1, j]], [v[s, i], v[s + 1, j], v[s + 1, i]]])
    b = Body(
        "short hairpin",
        np.array(tri),
        profile,
        p,
        np.array(["outbound", "bend", "return"]),
        False,
        0.0001,
        0.00005,
        0.0,
    )
    row = compare(b, b, "short_fold")
    assert row["failed"] and abs(row["witnesses"]["minimum"]["distance_m"] - 0.0002) < 1e-12
    assert not inspect_skin(box("valid", 0))["failed"]
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    shared_vertex = np.array([[0, 0, 0], [0.4, 0.1, -0.5], [0.1, 0.4, 0.5]])
    coplanar_fold = np.array([[0, 0, 0], [1, 0, 0], [0.1, 0.5, 0]])
    ordinary_edge = np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]])
    hit, pierce, overlap, point, _ = classify(
        np.repeat(a[None], 3, axis=0), np.stack((shared_vertex, coplanar_fold, ordinary_edge))
    )
    np.testing.assert_array_equal(hit, [True, True, True])
    np.testing.assert_array_equal(pierce, [True, False, False])
    np.testing.assert_array_equal(overlap, [False, True, False])
    assert point[0, 0] > 0 and point[0, 1] > 0 and point[0, 2] == 0
    tangent = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, -1.0, 0.2]])
    coplanar_tangent = tangent.copy()
    coplanar_tangent[2, 2] = 0.0
    for tri in (tangent, coplanar_tangent):
        hit, pierce, patch, _, extended = classify(a[None], tri[None])
        assert hit[0] and extended[0] and not (pierce[0] or patch[0])
    rng = np.random.default_rng(243)
    a, b = rng.normal(size=(2, 700, 3, 3))
    for planar in (False, True):
        if planar:
            a[:, :, 2] = b[:, :, 2] = 0
        hits, *_ = classify(a, b)
        np.testing.assert_array_equal(hits, triangle_pair_distances(a, b, backend="numpy").values_m <= 1e-12)


if __name__ == "__main__":
    test_required_actual_gap_and_witnesses()
    test_closest_features()
    test_nonlocal_and_bvh()
    test_short_hairpin_and_incident_skin()
    print("All surface clearance and witness checks passed.")
