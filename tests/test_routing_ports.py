"""Attachment search retains directed source arcs and immutable baseline evidence."""

from copy import deepcopy
from dataclasses import replace
import pickle
import numpy as np
import pytest

from _group_routing_cases import make_group_routing_case
from coilgen.config import CoilGenConfig
from coilgen.config_validation import validate_config
from coilgen.joint_cut_variants import _cut_signature, _opened_contour_follows_source, propose_joint_cut_variants
from coilgen.layout import interconnect_within_groups
from coilgen.models import ContourLine
from coilgen.routing_ports import (
    ContourArc,
    _source_orientation,
    propose_routing_port_plans,
    audit_shared_corridor_reservations,
)


@pytest.mark.parametrize("center", [0.0, 0.007, 0.015, 0.039])
def test_opening_is_a_forward_source_arc_with_exact_retained_vertices(center):
    points = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0], [0, 0, 0]])
    source = ContourLine(points, 2.4, -1.0)
    arc = ContourArc(source)
    before = pickle.dumps(source)
    opened, cut = arc.open(center, 0.004)
    assert _opened_contour_follows_source(source, opened)
    assert opened.current_orientation == -1 and opened.potential == 2.4
    assert np.linalg.norm(np.diff(opened.v, axis=0), axis=1).sum() == pytest.approx(0.036)
    assert np.array_equal(cut, opened.v[[0, -1]])
    for point in opened.v[1:-1]:
        assert any(np.array_equal(point, vertex) for vertex in points)
    assert pickle.dumps(source) == before


def test_opening_keeps_exact_affine_face_provenance_at_interpolated_ports():
    points = np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0]])
    bary = np.eye(3)
    source = ContourLine(
        points,
        1.0,
        segment_faces=np.zeros(3, dtype=int),
        segment_barycentric=np.stack([bary[[0, 1]], bary[[1, 2]], bary[[2, 0]]]),
    )
    opened, _ = ContourArc(source).open(0.001, 0.003)
    triangles = points[:3]
    np.testing.assert_allclose(
        opened.segment_barycentric @ triangles, np.stack((opened.v[:-1], opened.v[1:]), axis=1), atol=1e-15
    )
    assert np.all(opened.segment_barycentric >= -1e-12)
    assert _opened_contour_follows_source(source, opened)


def test_wrapped_seam_interval_keeps_its_forward_source_face_and_barycentrics():
    points = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0], [0, 0, 0]])
    triangles = np.stack((points[:-1], points[1:], points[:-1] + [0, 0, 0.02]), axis=1)
    source = ContourLine(
        points, 1.0, segment_faces=np.arange(4), segment_barycentric=np.tile([[1.0, 0, 0], [0.0, 1, 0]], (4, 1, 1))
    )
    opened, _ = ContourArc(source).open(0.005239379946012895, 0.00399932910999027)
    assert opened.segment_faces.tolist() == [0, 1, 2, 3, 0]
    recovered = np.einsum("nij,njk->nik", opened.segment_barycentric, triangles[opened.segment_faces])
    np.testing.assert_allclose(recovered, np.stack((opened.v[:-1], opened.v[1:]), axis=1), atol=2e-17)
    assert opened.segment_barycentric.min() >= 0 and opened.segment_barycentric.max() <= 1


@pytest.mark.parametrize("span", [0, np.nan, np.inf, 0.03])
def test_unbounded_or_degenerate_ports_are_rejected(span):
    source = ContourLine(np.array([[0.0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0], [0, 0.01, 0], [0, 0, 0]]), 1.0)
    with pytest.raises(ValueError):
        ContourArc(source).open(0.01, span)


def _case():
    case = make_group_routing_case("opposed_direction")
    case.config = replace(case.config, joint_group_routing_selection_policy="regular")
    interconnect_within_groups([case.part], case.config)
    return case


def test_shared_ports_search_varies_real_arclength_positions_and_keeps_baseline():
    case = _case()
    before = pickle.dumps(case.part)
    plans = propose_routing_port_plans(case.part, case.config, maximum_candidates=8)
    assert len(plans) == 8
    assert any(abs(row["seam_shift_m"]) > 0 for plan in plans for row in plan.diagnostics["ports"])
    assert all(not plan.diagnostics["physical_constraints_relaxed"] for plan in plans)
    result = propose_joint_cut_variants(case.part, case.config, maximum_candidates=7)
    assert 2 < len(result.variants) <= 7
    assert any("port_plan" in row for row in result.diagnostics["variants"])
    for context in result.variants:
        for source, group in zip(case.part.groups, context.groups):
            assert group.loop_indices == source.loop_indices
            for full, opened in zip(group.loops, group.opened_loops):
                assert _opened_contour_follows_source(full, opened)
        assert pickle.dumps(context.wire_path) == pickle.dumps(case.part.wire_path)
    assert pickle.dumps(case.part) == before


def test_equal_contour_ports_with_distinct_return_lanes_are_not_duplicates():
    case = _case()
    alternate = deepcopy(case.part)
    alternate.connected_groups[0].return_path_v[1, 2] += 0.001
    assert all(
        np.array_equal(a.v, b.v)
        for ga, gb in zip(case.part.groups, alternate.groups)
        for a, b in zip(ga.opened_loops, gb.opened_loops)
    )
    assert _cut_signature(case.part) != _cut_signature(alternate)
    assert _cut_signature(case.part) == _cut_signature(deepcopy(case.part))


def test_cross_group_corridor_intersection_is_measured_in_3d_and_budget_is_explicit():
    case = _case()
    # Independent group joins cross inside their intervals, with no attached
    # contour endpoint in common. Projecting separated z lanes must not clash.
    case.part.connected_groups[0].spiral_in_v = np.array([[-0.003, 0, 0], [0.003, 0, 0]])
    case.part.connected_groups[1].spiral_in_v = np.array([[0, -0.003, 0], [0, 0.003, 0]])
    for group in case.part.connected_groups:
        group.return_path_v = np.zeros((0, 3))
    report = audit_shared_corridor_reservations(case.part, case.config)
    assert report["completed"] and report["intersection_pair_count"] == 1
    assert report["minimum_distance_m"] < 1e-12
    assert not audit_shared_corridor_reservations(case.part, case.config, maximum_pairs=0)["completed"]
    case.part.connected_groups[1].spiral_in_v[:, 2] = 0.01
    report = audit_shared_corridor_reservations(case.part, case.config)
    assert report["completed"] and report["intersection_pair_count"] == 0
    report = audit_shared_corridor_reservations(case.part, case.config, maximum_broad_phase_pairs=0)
    assert not report["completed"] and report["reason"] == "corridor_broad_phase_budget_exhausted"
    assert report["evaluated_pairs"] == 0


def test_forced_cut_is_exactly_preserved_in_extended_context_search():
    case = _case()
    case.config.force_cut_selection = ("high", "auto")
    old = pickle.dumps(case.part.groups[0])
    result = propose_joint_cut_variants(case.part, case.config, maximum_candidates=7)
    assert result.variants
    for context in result.variants:
        assert pickle.dumps(context.groups[0]) == old


def test_port_plan_is_equivariant_to_rigid_motion_and_independent_of_cyclic_storage():
    case = _case()
    first = propose_routing_port_plans(case.part, case.config, maximum_candidates=8)
    moved = deepcopy(case.part)
    angle = 0.6
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    shift = np.array([0.3, -0.2, 0.5])
    moved.mesh.vertices = moved.mesh.vertices @ rotation.T + shift
    for group in moved.groups:
        for loop in group.loops + group.opened_loops:
            loop.v = loop.v @ rotation.T + shift
    second = propose_routing_port_plans(moved, case.config, maximum_candidates=8)
    for a, b in zip(first, second):
        for gi in a.groups:
            for source, target, pa, pb in zip(
                case.part.groups[gi].loops, moved.groups[gi].loops, a.groups[gi].ports, b.groups[gi].ports
            ):
                va = ContourArc(source).open(pa.center_m, pa.span_m)[0].v
                vb = ContourArc(target).open(pb.center_m, pb.span_m)[0].v
                np.testing.assert_allclose(va @ rotation.T + shift, vb, atol=1e-13)
    rolled = deepcopy(case.part)
    for group in rolled.groups:
        for loop in group.loops:
            vertices = np.roll(loop.v[:-1], 7, axis=0)
            loop.v = np.vstack((vertices, vertices[0]))
    third = propose_routing_port_plans(rolled, case.config, maximum_candidates=8)
    for a, b in zip(first, third):
        for gi in a.groups:
            for source, target, pa, pb in zip(
                case.part.groups[gi].loops, rolled.groups[gi].loops, a.groups[gi].ports, b.groups[gi].ports
            ):
                np.testing.assert_allclose(
                    ContourArc(source).open(pa.center_m, pa.span_m)[0].v,
                    ContourArc(target).open(pb.center_m, pb.span_m)[0].v,
                    atol=1e-13,
                )


@pytest.mark.parametrize("tilt", [-1e-11, 0.0, 1e-11])
@pytest.mark.parametrize("direction", [-1, 1])
def test_nearly_axial_attachment_uses_explicit_axial_basis(tilt, direction):
    points = np.array([[1.0, 0, 0], [1.0, tilt, 1], [1.0, 1, 1], [1.0, 1, 0], [1.0, 0, 0]])
    if direction < 0:
        points = points[::-1]
    arc = ContourArc(ContourLine(points, 1.0))
    center = 0.5 if direction > 0 else arc.perimeter - 0.5
    assert _source_orientation(arc, center, 0.2, np.array([0.0, 0, 1]), np.zeros(3)) == (direction, "local_axial")


def test_unresolved_attachment_and_planar_area_do_not_choose_a_roundoff_sign():
    points = np.array([[1.0, 0, 0], [2.0, 0, 0], [2.0, 0, 1], [1.0, 0, 1], [1.0, 0, 0]])
    arc = ContourArc(ContourLine(points, 1.0))
    assert _source_orientation(arc, 0.5, 0.2, np.array([0.0, 0, 1]), np.zeros(3)) == (
        None,
        "unresolved_attachment_direction",
    )
    planar = ContourArc(ContourLine(np.array([[0.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0], [1.0, 0, 0], [0.0, 0, 0]]), 1.0))
    assert _source_orientation(planar, 0.5, 0.2, np.array([0.0, 0, 1])) == (None, "unresolved_planar_orientation")


@pytest.mark.parametrize(
    "field,value",
    [
        ("joint_group_routing_port_search_count", True),
        ("joint_group_routing_port_search_count", 33),
        ("joint_group_routing_cut_span_max_factor", 0.9),
        ("joint_group_routing_cut_span_max_factor", np.nan),
        ("joint_group_routing_seam_search_extent_rel", 0.3),
        ("joint_group_routing_seam_search_extent_rel", False),
    ],
)
def test_port_controls_reject_invalid_limits(field, value):
    config = CoilGenConfig(**{field: value})
    assert any(error.field == field for error in validate_config(config).errors)
