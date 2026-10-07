"""Late return interactions must preserve compatible low-jump complete routes."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import joint_route_shapes as shapes


def choices_with_late_return(extra_jump_detour=False):
    critical = [
        SimpleNamespace(
            points=np.array([[-0.04, 0, 0], [-0.035, gap, 0], [0.035, gap, 0], [0.04, 0, 0]]),
            profile_family="cylinder_surface",
        )
        for gap in (0.0, 0.0001, 0.0002, 0.01)
    ]
    choices = [critical]
    # 64 nearly equal cheap prefixes occupy more than the unchanged beam48.
    for offset in np.arange(1, 7) * 0.5:
        choices.append(
            [
                SimpleNamespace(
                    points=np.array([[offset, 0, 0], [offset + 0.02, deviation, 0], [offset + 0.04, 0, 0]]),
                    profile_family="cylinder_surface",
                )
                for deviation in (0.0, 1e-6)
            ]
        )
    choices.append(
        [
            SimpleNamespace(
                points=np.array([[0, 0, -0.02], [0, 0, 0.07]]), profile_family="cylinder_compact_lifted_return"
            )
        ]
    )
    jumps = [[0] * len(options) for options in choices[:-1]] + [[1]]
    if extra_jump_detour:
        choices[-1].append(
            SimpleNamespace(
                points=np.array([[0, 0.02, -0.02], [0, 0.02, 0.07]]), profile_family="cylinder_compact_lifted_return"
            )
        )
        jumps[-1].append(2)
    return choices, jumps


@pytest.mark.parametrize("extra_jump_detour", [False, True])
def test_future_domains_keep_slightly_longer_h0_join_and_one_jump_completion(extra_jump_detour):
    choices, jumps = choices_with_late_return(extra_jump_detour)
    options = dict(jump_counts=jumps, return_blocks=(False,) * 7 + (True,))
    diagnostics = {}
    current = shapes._rank_curve_combinations(
        choices, np.empty((0, 2, 3)), 0.005, 0, 3, diagnostics=diagnostics, **options
    )
    assert all(selected[0] == 3 and selected[-1] == 0 and gap == 0.005 for selected, gap, _cost in current)
    assert diagnostics["forward_domain_checks"] > 0
    assert diagnostics["forward_domain_scope"] == "necessary_nominal_completion_conditions_only"
    assert diagnostics["spatial_candidate_pairs"] <= shapes.JointShapeLimits().maximum_coupling_spatial_pairs


def test_no_feasible_completion_keeps_honest_infeasible_fallbacks():
    choices, jumps = choices_with_late_return()
    choices[0] = choices[0][:3]
    jumps[0] = jumps[0][:3]
    diagnostics = {}
    result = shapes._rank_curve_combinations(
        choices, np.empty((0, 2, 3)), 0.005, 0, 3, diagnostics=diagnostics, jump_counts=jumps
    )
    assert result and all(gap < 0.005 for _selected, gap, _cost in result)
    assert diagnostics["forward_domain_failed_prefixes"] > 0


def test_forward_scoring_respects_the_same_hard_pair_work_budget():
    choices, jumps = choices_with_late_return()
    work = shapes._ShapeWork(shapes.JointShapeLimits(maximum_spatial_pairs=1))
    with pytest.raises(shapes._DistanceBudgetExceeded, match="joint_spatial_pair_budget_exceeded"):
        shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.005, 0, 3, work=work, jump_counts=jumps)


def test_short_compact_joins_do_not_mask_the_compatible_actual_compact_return():
    choices = []
    for z in (0.0, 0.025, 0.05):
        choices.append(
            [
                SimpleNamespace(
                    points=np.array([[-0.04, 0, z], [0.04, 0, z]]), profile_family="cylinder_compact_return"
                ),
                SimpleNamespace(
                    points=np.array([[-0.04, 0, z], [-0.035, 0.01, z], [0.035, 0.01, z], [0.04, 0, z]]),
                    profile_family="cylinder_surface",
                ),
            ]
        )
    choices.append(
        [
            SimpleNamespace(
                points=np.array([[0, 0, -0.02], [0, 0.02, -0.01], [0, 0.02, 0.06], [0, 0, 0.07]]),
                profile_family="planar_biarc_lifted_return",
            ),
            SimpleNamespace(
                points=np.array([[0, 0, -0.02], [-0.018, 0, -0.01], [-0.018, 0, 0.06], [0, 0, 0.07]]),
                profile_family="cylinder_compact_lifted_return",
            ),
        ]
    )
    options = dict(jump_counts=[[0, 0], [0, 0], [0, 0], [1, 1]])
    unscoped = shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.005, 0, 3, **options)
    assert all(state[0][3] == 0 for state in unscoped)
    scoped = shapes._rank_curve_combinations(
        choices, np.empty((0, 2, 3)), 0.005, 0, 3, return_blocks=(False, False, False, True), **options
    )
    assert scoped[0][0] == (0, 0, 0, 0)
    assert any(state[0] == (1, 1, 1, 1) for state in scoped)
    assert all(state[1] == 0.005 for state in scoped)
    assert shapes._return_family_counts(choices, (0, 0, 0, 0), (False, False, False, True))["intrinsic_compact"] == 0
    assert shapes._return_family_counts(choices, (1, 1, 1, 1), (False, False, False, True))["intrinsic_compact"] == 1
    # More jumps do not gain a slot through the return-family reservation.
    extra_jump = shapes._rank_curve_combinations(
        choices,
        np.empty((0, 2, 3)),
        0.005,
        0,
        3,
        return_blocks=(False, False, False, True),
        jump_counts=[[0, 0], [0, 0], [0, 0], [1, 2]],
    )
    assert all(state[0][3] == 0 for state in extra_jump)
