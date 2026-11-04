"""Finite rectangular copper can miss the required gap without intersecting."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.group_route_validation import GroupRouteValidationLimits, _exact_collision_audit
from coilgen.models import Mesh
from coilgen.swept_clearance import loft_pair_batches, required_copper_gap


def test_loft_tree_matches_exhaustive_aabbs_with_odd_leaf_counts():
    rng = np.random.default_rng(304)
    lo = rng.random((87, 3))
    hi = lo + rng.random((87, 3)) / 4
    gap = 0.12
    actual = np.concatenate(list(loft_pair_batches(lo, hi, gap, GroupRouteValidationLimits(), {})))
    i, j = np.triu_indices(len(lo), 1)
    delta = np.maximum(0, np.maximum(lo[i] - hi[j], lo[j] - hi[i]))
    expected = set(zip(i[np.linalg.norm(delta, axis=1) <= gap], j[np.linalg.norm(delta, axis=1) <= gap]))
    assert len(actual) == len(expected)
    assert set(map(tuple, actual)) == expected


def _hairpin(gap):
    # Exact annular half-loft with two straight, parallel legs. Copper is
    # 1 mm wide and thick; straight opposing inner faces are 'gap' apart.
    radius = (0.001 + gap) / 2
    theta = np.linspace(0, np.pi, 33)
    centres = np.stack([0.020 + radius * np.sin(theta), radius * np.cos(theta), np.zeros_like(theta)], axis=-1)
    centres = np.vstack(([0, radius, 0], centres, [0, -radius, 0]))
    outward = np.stack([np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=-1)
    outward = np.vstack((outward[0], outward, outward[-1]))
    profile = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.0005
    rings = centres[:, None, :] + outward[:, None, :] * profile[:, 0, None]
    rings[:, :, 2] += profile[:, 1]
    faces = []
    for k in range(len(centres) - 1):
        for p in range(4):
            a, b = 4 * k + p, 4 * k + (p + 1) % 4
            faces.extend([[a, b, a + 4], [b, b + 4, a + 4]])
    return Mesh(rings.reshape(-1, 3), np.asarray(faces)), SimpleNamespace(
        closed=False, wire_points_m=centres, cross_section_m=profile
    )


@pytest.mark.parametrize("gap, failure", [(0.00018, True), (0.00031, True), (0.00050, False)])
def test_folded_stretches_need_insulation_on_both_faces(gap, failure):
    mesh, path = _hairpin(gap)
    target = required_copper_gap(
        CoilGenConfig(sweep_insulation_thickness=0.0001, sweep_manufacturing_tolerance=0.00005)
    )
    assert target == pytest.approx(0.00025)
    diag, reason = _exact_collision_audit(
        mesh, GroupRouteValidationLimits(), fabrication_path=path, required_gap_m=target
    )
    assert diag["exact_collision_completed"] and diag["exact_intersection_count"] == 0
    assert (reason == "insufficient_copper_gap") == failure
    assert (diag["copper_gap_violating_facet_pairs"] > 0) == failure


def test_gui_uses_the_copper_measurement_and_keeps_incomplete_work_unverified():
    from coilgen.gui.acceptance import _body_rows

    d = dict(
        sweep_capsule_clearance_violation=True,
        sweep_self_intersection_skipped=True,
        exact_collision_completed=True,
        copper_clearance_failure="",
        required_copper_gap_m=0.00025,
        copper_gap_minimum_lower_bound_m=0.00025,
        exact_intersection_count=0,
    )
    metadata = {"fabrication": {"paths": [{"body_id": "foil", "diagnostics": d}]}}
    row = _body_rows(metadata)[0]
    assert row.requirement == "Copper surface separation" and row.status == "Passed"
    assert row.observed == row.limit == 0.00025
    d.update(exact_collision_completed=False, copper_clearance_failure="sweep_face_budget_exceeded")
    row = _body_rows(metadata)[0]
    assert row.status == "Unverified" and row.observed is None


def test_ascii_stl_round_trips_the_verified_facet_coordinates(tmp_path):
    from coilgen.manufacturing import save_mesh_as_stl

    mesh, _ = _hairpin(0.00031)
    mesh.vertices += np.array([0.123456789012345, -0.09876543210987, 0.03040506070809])
    path = save_mesh_as_stl(mesh, tmp_path / "copper.stl")
    vertices = np.asarray(
        [
            [float(c) for c in line.split()[1:]]
            for line in path.read_text().splitlines()
            if line.lstrip().startswith("vertex ")
        ]
    )
    np.testing.assert_array_equal(vertices.reshape(-1, 3, 3), mesh.vertices[mesh.faces])
