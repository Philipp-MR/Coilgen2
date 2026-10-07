"""Axial ring seams stay coherent through small perturbations and cut search."""

from copy import deepcopy

import numpy as np
import pytest

from test_axial_ring_cut_symmetry import _fixture
from coilgen import layout
from coilgen.config import CoilGenConfig
from coilgen.joint_cut_variants import propose_joint_cut_variants
from coilgen.layout import (
    GroupConnectionCandidate,
    InterGroupCutAlignmentState,
    _axial_ring_cut_frame,
    _sort_group_connection_candidates,
    calculate_group_centers,
    interconnect_within_groups,
)


def _config(**kwargs):
    return CoilGenConfig(interconnection_cut_width=0.002, **kwargs)


def test_submicrometre_contour_noise_does_not_flip_only_one_return():
    part, _, axis, _ = _fixture(seam=13)
    rng = np.random.default_rng(2)
    for group in part.groups:
        for loop in group.loops:
            loop.v[:, :2] += rng.normal(size=loop.v[:, :2].shape) * 1e-7
            loop.v[-1] = loop.v[0]
    calculate_group_centers([part])
    originals = [loop.v.copy() for loop in part.contour_lines]
    assert all(_axial_ring_cut_frame(group, part.mesh, axis) is not None for group in part.groups)

    interconnect_within_groups([part], _config())

    assert len({group.within_group_cut_side for group in part.groups}) == 1
    radials = [connection.return_path_v[:, :2].mean(axis=0) for connection in part.connected_groups]
    assert float(radials[0] @ radials[1]) > 0
    for loop, original in zip(part.contour_lines, originals):
        np.testing.assert_array_equal(loop.v, original)


def test_alternate_cut_contexts_flip_the_entire_axial_cohort_once():
    part, *_ = _fixture(seam=13)
    config = _config()
    interconnect_within_groups([part], config)
    originals = deepcopy(part.groups)

    result = propose_joint_cut_variants(part, config)

    assert result.reasons == ()
    assert len(result.variants) == 1
    assert result.diagnostics["coupled_axial_group_indices"] == [0, 1]
    assert result.diagnostics["attempts"][0]["changed_group_indices"] == [0, 1]
    assert not result.diagnostics["coupled_axial_side_pinned"]
    selected = {group.within_group_cut_side for group in result.variants[0].groups}
    assert len(selected) == 1
    assert selected != {group.within_group_cut_side for group in part.groups}
    for before, after in zip(originals, part.groups):
        assert before.within_group_cut_side == after.within_group_cut_side
        for original, loop in zip(before.opened_loops, after.opened_loops):
            np.testing.assert_array_equal(original.v, loop.v)


@pytest.mark.parametrize("forced", [("low", "auto"), ("auto", "high"), ("high", "low")])
def test_forced_axial_side_is_not_undone_by_alternative_cut_search(forced):
    part, *_ = _fixture(seam=13)
    config = _config(force_cut_selection=forced)
    interconnect_within_groups([part], config)
    before = [group.within_group_cut_side for group in part.groups]
    for requested, actual in zip(forced, before):
        if requested != "auto":
            assert actual == requested
    if "auto" in forced:
        assert len(set(before)) == 1

    result = propose_joint_cut_variants(part, config)

    assert result.variants == []
    assert result.diagnostics["coupled_axial_side_pinned"]
    assert before == [group.within_group_cut_side for group in part.groups]


@pytest.mark.parametrize("forced", [(), ("high", "auto")])
def test_unavailable_common_side_fails_instead_of_splitting_the_axial_cohort(monkeypatch, forced):
    part, *_ = _fixture(seam=13)
    build = layout._build_within_group_cut_rail

    def unavailable(group, cut_infos, side, config, **kwargs):
        candidate = build(group, cut_infos, side, config, **kwargs)
        if (group is part.groups[0] and side == "low") or (group is part.groups[1] and side == "high"):
            candidate.score = np.inf
        return candidate

    monkeypatch.setattr(layout, "_build_within_group_cut_rail", unavailable)
    with pytest.raises(ValueError, match="(?i)common.*feasible|feasible.*common"):
        interconnect_within_groups([part], _config(force_cut_selection=forced))


def test_explicit_alignment_opt_out_keeps_independent_cut_search():
    part, *_ = _fixture(seam=13)
    config = _config(inter_group_cut_alignment="off")
    interconnect_within_groups([part], config)
    before = [group.within_group_cut_side for group in part.groups]

    result = propose_joint_cut_variants(part, config)

    assert len(result.variants) == 2
    assert result.diagnostics["coupled_axial_group_indices"] == []
    assert sum(a != b.within_group_cut_side for a, b in zip(before, result.variants[1].groups)) == 1


def test_old_mixed_side_axial_baseline_is_not_used_for_new_incoherent_variants():
    part, *_ = _fixture(seam=13)
    interconnect_within_groups([part], _config(force_cut_selection=("high", "low")))

    result = propose_joint_cut_variants(part, _config())

    assert result.variants == []
    assert result.reasons == ("incoherent_axial_cut_baseline",)


def test_opposite_contour_tilts_cannot_cancel_in_axial_ring_gate():
    part, *_ = _fixture()
    group = part.groups[0]
    group.loops = group.loops[:2]
    for index, loop in enumerate(group.loops):
        loop.v[:, 2] += (1 if index == 0 else -1) * 0.3 * loop.v[:, 0]

    assert _axial_ring_cut_frame(group, part.mesh, [0, 0, 1]) is None


def test_tiny_endpoint_mismatch_cannot_move_bridge_to_opposite_cylinder_side():
    state = InterGroupCutAlignmentState(
        origin_v=np.zeros(3),
        axis_v=np.asarray([0.0, 0.0, 1.0]),
        preferred_radial_v=np.asarray([1.0, 0.0, 0.0]),
        distance_rel_tol=0.05,
        candidate_factor=4,
        return_corridor_aligned=True,
        axial_ring_alignment=True,
    )

    def candidate(first_angle, second_angle):
        angles = np.deg2rad([first_angle, second_angle])
        points = np.column_stack((0.1 * np.cos(angles), 0.1 * np.sin(angles), [0.04, -0.04]))
        distance = float(np.linalg.norm(points[1] - points[0]))
        return GroupConnectionCandidate(
            local_a=0,
            local_b=1,
            point_a_v=points[0],
            point_b_v=points[1],
            direct_distance=distance,
            geodesic_distance=np.inf,
            selection_distance=distance,
            used_geodesic_selection=False,
        )

    front = candidate(0.0, 0.01)
    back = candidate(180.0, 180.0)
    assert _sort_group_connection_candidates([back, front], state)[0] is front
