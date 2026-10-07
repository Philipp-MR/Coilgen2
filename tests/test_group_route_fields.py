"""Engineering regressions for the fixed-current group-route field gate."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.group_route_fields import assess_group_route_field
from coilgen.models import TargetField


def target(count=2, *, suppressed=None, weights=None, values=None):
    return TargetField(
        coords=np.column_stack((np.arange(count), np.zeros((count, 2)))),
        b=np.zeros((count, 3)) if values is None else values,
        weights=np.ones(count) if weights is None else weights,
        suppressed=np.zeros(count, dtype=bool) if suppressed is None else suppressed,
        target_field_group_inds=np.ones(count, dtype=int),
    )


def assess(before, after, field=None, config=None, current=1.0, budget=1e-6):
    return assess_group_route_field(
        before,
        after,
        target(len(before)) if field is None else field,
        CoilGenConfig() if config is None else config,
        current,
        budget,
    )


def test_opposing_point_changes_cannot_cancel_the_explicit_budget():
    before = np.zeros((2, 3))
    after = np.array([[2e-6, 0, 0], [-2e-6, 0, 0]])
    result = assess(before, after)
    assert not result.accepted
    assert result.reasons == ("pointwise_field_delta_budget",)
    assert result.metrics["maximum_pointwise_vector_delta_t_per_a"] == 2e-6


def test_vector_budget_is_not_a_per_component_budget():
    result = assess(np.zeros((1, 3)), np.full((1, 3), 0.8e-6))
    assert not result.accepted
    assert result.metrics["maximum_pointwise_vector_delta_t_per_a"] == pytest.approx(np.sqrt(3) * 0.8e-6)


def test_budget_is_per_ampere_and_independent_of_operating_current():
    for current in (1e-6, 1.0, -100.0):
        result = assess(np.zeros((1, 3)), np.array([[2e-6, 0, 0]]), current=current)
        assert not result.accepted
        assert "pointwise_field_delta_budget" in result.reasons


def test_constant_error_is_not_hidden_by_zero_standard_deviation():
    config = CoilGenConfig(layout_target_error_std_limit_T=1e-9, layout_target_error_rms_limit_T=1e-7)
    result = assess(np.zeros((2, 3)), np.tile([0, 0, 2e-7], (2, 1)), config=config)
    assert result.reasons == ("layout_target_rms_limit",)
    assert result.metrics["candidate_dsv_residual_std_T"] == 0.0


def test_target_fit_setting_cannot_refit_the_candidate_current():
    before = np.array([[0, 0, -1e-6], [0, 0, 1e-6]])
    config = CoilGenConfig(field_evaluation_current_source="target_fit", layout_target_error_rms_limit_T=1e-8)
    result = assess(before, 2 * before, target(values=before), config, budget=2e-6)
    assert not result.accepted
    assert "layout_target_rms_limit" in result.reasons
    assert result.metrics["current_a"] == 1.0
    assert result.metrics["current_refitted"] is False
    assert result.metrics["candidate_dsv_residual_rms_T"] == 1e-6


def test_explicit_layout_acceptance_tolerance_is_honored():
    config = CoilGenConfig(layout_target_error_rms_limit_T=1e-7, layout_target_error_acceptance_absolute_T=1e-8)
    assert assess(np.zeros((1, 3)), np.array([[0, 0, 1.05e-7]]), config=config).accepted


def test_legacy_slotted_config_missing_field_policy_keeps_absolute_acceptance():
    config = CoilGenConfig(layout_target_error_rms_limit_T=1e-7)
    del config.joint_group_routing_field_acceptance
    baseline = np.array([[0.0, 0, 2e-7]])
    result = assess(baseline, baseline, config=config)
    assert result.metrics["field_acceptance_policy"] == "absolute"
    assert result.reasons == ("layout_target_rms_limit",)


def test_optimal_stream_function_percentage_is_not_a_routing_budget():
    config = CoilGenConfig(target_error_tolerance_percent=100, suppression_error_tolerance_percent=100)
    result = assess(np.zeros((1, 3)), np.array([[1e-6, 0, 0]]), config=config, budget=0)
    assert result.reasons == ("pointwise_field_delta_budget",)


def _gradient_case():
    coords = 0.01 * np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])
    values = np.column_stack((np.zeros((len(coords), 2)), 0.01 * coords[:, 0]))
    field = TargetField(coords, values, np.ones(len(coords)), np.zeros(len(coords), bool), np.ones(len(coords), int))
    config = CoilGenConfig(
        target_gradient_strength=0.01,
        gradient_strength_tolerance_percent=1.0,
        linearity_rms_tolerance_percent=1.0,
        linearity_peak_tolerance_percent=2.0,
        linearity_cross_gradient_tolerance_percent=1.0,
    )
    return field, config


@pytest.mark.parametrize("defect", ["strength", "cross_gradient", "nonlinear"])
def test_absolute_gradient_contract_cannot_be_bypassed_by_generous_routing_delta(defect):
    field, config = _gradient_case()
    current = 2.0
    before = field.b / current
    after = before.copy()
    if defect == "strength":
        after *= 1.02
    elif defect == "cross_gradient":
        after[:, 2] += 0.0002 * field.coords[:, 1] / current
    else:
        after[:, 2] += field.coords[:, 0] ** 2 * 0.2 / current
    result = assess(before, after, field, config, current=current, budget=1.0)
    assert result.reasons == ("gradient_linearity_contract",)
    assert result.metrics["candidate_linearity_status"] == "FAIL"
    assert result.metrics["current_refitted"] is False
    assert assess(before, before, field, config, current=current, budget=0).accepted


def test_gradient_contract_rejects_unobservable_cross_gradient_on_planar_samples():
    field, config = _gradient_case()
    field.coords[:, 2] = 0
    result = assess(field.b, field.b, field, config, budget=0)
    assert result.reasons == ("linearity_domain_rank_deficient",)


def test_inactive_samples_cannot_authorize_the_gradient_contract():
    field, config = _gradient_case()
    field.weights[:] = 0
    field.weights[:3] = 1
    result = assess(field.b, field.b, field, config, budget=0)
    assert result.reasons == ("linearity_domain_not_certified",)


@pytest.mark.parametrize("region", ["dsv", "target", "roi", "fov", "primary", ""])
def test_dsv_aliases_never_use_suppression_samples_for_linearity(region):
    field, config = _gradient_case()
    field.suppressed[:] = True
    config.linearity_region = region
    result = assess(field.b, field.b, field, config, budget=0)
    assert result.reasons == ("linearity_domain_not_certified",)


def test_suppression_points_are_included_in_full_vector_delta_gate():
    field = target(suppressed=[False, True])
    before, after = np.zeros((2, 3)), np.zeros((2, 3))
    after[1, 0] = 2e-6  # Scalar Bz suppression alone would miss this change.
    result = assess(before, after, field)
    assert "pointwise_field_delta_budget" in result.reasons


@pytest.mark.parametrize("component", ["Bx", "vector", "Bmag", "components"])
def test_suppression_residual_regression_is_rejected(component):
    field = target(suppressed=[False, True])
    after = np.array([[0, 0, 0], [1e-7, 0, 0]])
    result = assess(np.zeros((2, 3)), after, field, CoilGenConfig(suppression_field_component=component))
    assert {"suppression_rms_regression", "suppression_peak_regression"} <= set(result.reasons)


def test_bmag_suppression_does_not_cancel_opposite_equal_magnitudes():
    before = np.array([[0, 0, 0], [1e-7, 0, 0]])
    after = -before
    field = target(suppressed=[False, True], values=before.copy())
    result = assess(before, after, field, CoilGenConfig(suppression_field_component="Bmag"))
    assert not result.accepted
    assert result.metrics["candidate_suppression_residual_peak_T"] == 2e-7


def test_vector_and_componentwise_peak_follow_constraint_semantics():
    before = np.array([[1e-7, 1e-7, 1e-7]])
    field = target(1, suppressed=[True])
    for component, peak in (("vector", np.sqrt(3) * 1e-7), ("components", 1e-7)):
        result = assess(before, before, field, CoilGenConfig(suppression_field_component=component))
        assert result.accepted
        assert result.metrics["candidate_suppression_residual_rms_T"] == pytest.approx(np.sqrt(3) * 1e-7)
        assert result.metrics["candidate_suppression_residual_peak_T"] == pytest.approx(peak)


def test_area_weighted_suppression_uses_surface_area_not_objective_weights():
    field = target(suppressed=[True, True], weights=[1, 100])
    field.suppression_area_weights = np.array([100.0, 1.0])
    before, after = np.array([[0, 0, 1e-7], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 1e-7]])
    config = CoilGenConfig(suppression_surface_weighting="area")
    result = assess(before, after, field, config)
    assert result.accepted
    assert result.metrics["candidate_suppression_residual_rms_T"] == pytest.approx(1e-7 / np.sqrt(101))


def test_declared_normal_suppression_allows_tangential_changes_only_within_budget():
    field = target(1, suppressed=[True])
    field.suppression_normals = np.array([[2.0, 0, 0]])
    config = CoilGenConfig(suppression_field_component="Bnormal")
    before = np.zeros((1, 3))
    assert assess(before, np.array([[0, 1e-7, 0]]), field, config).accepted
    assert not assess(before, np.array([[1e-7, 0, 0]]), field, config).accepted


def test_missing_declared_normal_or_area_data_is_not_implicitly_accepted():
    field, before = target(1, suppressed=[True]), np.zeros((1, 3))
    for config in (
        CoilGenConfig(suppression_field_component="Bnormal"),
        CoilGenConfig(suppression_surface_weighting="area"),
    ):
        result = assess(before, before, field, config)
        assert not result.accepted
        assert "invalid_field_context" in result.reasons
        assert result.metrics["checked"] is False


def test_explicit_suppression_limit_cannot_be_bypassed_by_unchanged_bad_baseline():
    field = target(1, suppressed=[True])
    before = np.array([[0, 0, 1e-7]])
    result = assess(before, before, field, CoilGenConfig(suppression_field_peak_limit_T=5e-8))
    assert result.reasons == ("suppression_peak_limit",)


def test_roundoff_only_suppression_regression_is_permitted():
    field = target(1, suppressed=[True])
    before = np.array([[0, 0, 1e-7]])
    after = before.copy()
    after[0, 2] = np.nextafter(before[0, 2], np.inf)
    assert assess(before, after, field).accepted
    after[0, 2] += 1e-15
    assert not assess(before, after, field).accepted


@pytest.mark.parametrize(
    "before_values,after_values,reason",
    [([1, 1], [1.2, 0], "suppression_peak_regression"), ([2, 0, 0], [1.9, 1.9, 0], "suppression_rms_regression")],
)
def test_suppression_rms_and_peak_must_each_be_preserved(before_values, after_values, reason):
    before, after = np.zeros((len(before_values), 3)), np.zeros((len(before_values), 3))
    before[:, 2], after[:, 2] = np.asarray(before_values) * 1e-7, np.asarray(after_values) * 1e-7
    field = target(len(before), suppressed=np.ones(len(before), dtype=bool))
    assert assess(before, after, field).reasons == (reason,)


def test_directional_target_is_evaluated_in_the_configured_b0_frame():
    before = np.zeros((2, 3))
    after = np.array([[1e-7, 0, 0], [-1e-7, 0, 0]])
    config = CoilGenConfig(
        target_field_component="BdotB0", b_0_direction=(2, 0, 0), layout_target_error_std_limit_T=5e-8
    )
    assert assess(before, after, config=config).reasons == ("layout_target_std_limit",)


@pytest.mark.parametrize("current", [0.0, np.nan, np.inf])
def test_zero_or_nonfinite_current_is_an_unchecked_context(current):
    result = assess(np.zeros((1, 3)), np.zeros((1, 3)), current=current)
    assert result.reasons == ("invalid_fixed_current",)
    assert result.metrics["checked"] is False


@pytest.mark.parametrize("budget", [-1.0, np.nan, np.inf])
def test_invalid_explicit_budget_rejects(budget):
    assert assess(np.zeros((1, 3)), np.zeros((1, 3)), budget=budget).reasons == ("invalid_field_delta_budget",)


def test_inactive_bad_samples_do_not_poison_active_checks_or_mutate_inputs():
    field = target(weights=[1.0, 0.0])
    field.coords[1] = np.nan
    field.b[1] = np.nan
    before = np.array([[0, 0, 0], [np.nan, np.inf, -np.inf]])
    after = before.copy()
    result = assess(before, after, field)
    assert result.accepted
    assert result.metrics["active_sample_count"] == 1
    np.testing.assert_equal(before, after)
    assert np.isnan(field.coords[1]).all()


def test_active_nonfinite_sample_and_empty_domain_reject():
    before = np.zeros((1, 3))
    assert assess(before, np.full((1, 3), np.nan)).reasons == ("nonfinite_active_field_context",)
    assert assess(before, before, target(1, weights=[0])).reasons == ("no_active_field_samples",)


def test_missing_context_and_mismatched_shapes_reject():
    before = np.zeros((1, 3))
    result = assess_group_route_field(before, before, None, CoilGenConfig(), 1.0, 1e-6)
    assert result.reasons == ("invalid_field_context",)
    assert not assess(before, np.zeros((2, 3))).accepted


def test_vector_target_uses_weighted_component_residuals_at_signed_current():
    before = np.array([[1e-7, 2e-7, 0], [-1e-7, 0, 3e-7]])
    field = target(values=-2 * before)
    config = CoilGenConfig(target_field_component="vector", layout_target_error_rms_limit_T=1e-10)
    result = assess(before, before.copy(), field, config, current=-2)
    assert result.accepted
    assert result.metrics["candidate_dsv_residual_rms_T"] == 0.0
    assert not assess(before, before.copy(), field, replace(config, layout_target_error_rms_limit_T=np.nan)).accepted


@pytest.mark.parametrize("metric", ["rms", "std"])
def test_explicit_nonregression_allows_only_improving_inherited_layout_violations(metric):
    config = CoilGenConfig(
        joint_group_routing_field_acceptance="non_regression",
        **{f"layout_target_error_{metric}_limit_T": 1e-7},
    )
    before = np.array([[0, 0, -2e-7], [0, 0, 2e-7]])
    after = 0.75 * before
    result = assess(before, after, config=config)
    assert result.accepted, result.reasons
    assert result.metrics["absolute_field_contract_met"] is False
    reason = f"layout_target_{metric}_limit"
    assert result.metrics["absolute_field_contract_reasons"] == reason
    assert result.metrics["inherited_field_contract_violations"] == reason
    assert assess(before, before, config=config, budget=0).accepted
    assert assess(before, 1.001 * before, config=config).reasons == (reason,)
    assert assess(np.zeros_like(before), after, config=config).reasons == (reason,)
    # The default contract remains absolute even for a useful partial repair.
    assert assess(before, after, config=replace(config, joint_group_routing_field_acceptance="absolute")).reasons == (
        reason,
    )


def test_inherited_layout_roundoff_is_permitted_but_meaningful_regression_is_rejected():
    config = CoilGenConfig(joint_group_routing_field_acceptance="non_regression", layout_target_error_rms_limit_T=1e-7)
    before = np.array([[0, 0, 2e-7]])
    after = before.copy()
    after[0, 2] = np.nextafter(before[0, 2], np.inf)
    assert assess(before, after, config=config).accepted
    after[0, 2] += 1e-14
    assert not assess(before, after, config=config).accepted


@pytest.mark.parametrize("defect", ["strength", "cross_gradient", "nonlinear"])
def test_nonregression_preserves_each_gradient_contract_and_reports_inherited_failures(defect):
    field, config = _gradient_case()
    config.joint_group_routing_field_acceptance = "non_regression"

    def defective(amount):
        values = field.b.copy()
        if defect == "strength":
            values *= 1 + 0.02 * amount
        elif defect == "cross_gradient":
            values[:, 2] += 0.0002 * field.coords[:, 1] * amount
        else:
            values[:, 2] += field.coords[:, 0] ** 2 * 0.2 * amount
        return values

    before, after = defective(1), defective(0.75)
    result = assess(before, after, field, config, budget=1)
    assert result.accepted, result.reasons
    assert result.metrics["candidate_linearity_status"] == "FAIL"
    assert not result.metrics["absolute_field_contract_met"]
    assert result.metrics["inherited_field_contract_violations"]
    assert result.metrics["current_refitted"] is False
    assert assess(before, before, field, config, budget=0).accepted
    assert assess(before, defective(1.001), field, config, budget=1).reasons == ("gradient_linearity_contract",)
    assert assess(field.b, after, field, config, budget=1).reasons == ("gradient_linearity_contract",)


def test_improving_one_inherited_gradient_failure_cannot_introduce_another():
    field, config = _gradient_case()
    config.joint_group_routing_field_acceptance = "non_regression"
    before, after = 1.03 * field.b, 1.02 * field.b
    after[:, 2] += 0.0002 * field.coords[:, 1]
    result = assess(before, after, field, config, budget=1)
    assert result.reasons == ("gradient_linearity_contract",)
    assert result.metrics["inherited_field_contract_violations"] == "gradient_strength_contract"
    assert "linearity_cross_gradient_contract" in result.metrics["absolute_field_contract_reasons"]


def test_nonregression_cannot_waive_vector_budget_or_suppression_regression():
    config = CoilGenConfig(joint_group_routing_field_acceptance="non_regression", suppression_field_peak_limit_T=5e-8)
    before = np.array([[0, 0, 1e-7]])
    field = target(1, suppressed=[True])
    retained = assess(before, before, field, config, budget=0)
    assert retained.accepted
    assert retained.metrics["absolute_field_contract_met"] is False
    assert retained.metrics["inherited_field_contract_violations"] == "suppression_peak_limit"
    increased = assess(before, 1.01 * before, field, config)
    assert "suppression_peak_regression" in increased.reasons
    repaired_but_over_budget = assess(before, 0.5 * before, field, config, budget=1e-8)
    assert repaired_but_over_budget.reasons == ("pointwise_field_delta_budget",)


def test_unknown_field_acceptance_policy_is_unchecked_context():
    config = CoilGenConfig(joint_group_routing_field_acceptance="assume_valid")
    result = assess(np.zeros((1, 3)), np.zeros((1, 3)), config=config)
    assert result.reasons == ("invalid_field_context",)
    assert not result.metrics["checked"]
    assert not result.metrics["absolute_field_contract_met"]


@pytest.mark.parametrize("metric", ["std", "rms"])
def test_budget_only_reports_new_and_worsened_inherited_layout_failures(metric):
    config = CoilGenConfig(
        joint_group_routing_field_acceptance="budget_only",
        **{f"layout_target_error_{metric}_limit_T": 1e-7},
    )
    failed = np.array([[0, 0, -2e-7], [0, 0, 2e-7]])
    reason = f"layout_target_{metric}_limit"
    new = assess(np.zeros_like(failed), failed, config=config)
    assert new.accepted, new.reasons
    assert not new.metrics["absolute_field_contract_met"]
    assert new.metrics["absolute_field_contract_reasons"] == reason
    assert new.metrics["newly_failed_field_contract_violations"] == reason
    assert new.metrics["inherited_field_contract_violations"] == ""
    assert new.metrics["budget_waived_field_contract_violations"] == reason
    worsened = assess(failed, 1.1 * failed, config=config)
    assert worsened.accepted, worsened.reasons
    assert worsened.metrics["inherited_field_contract_violations"] == reason
    assert worsened.metrics["regressed_inherited_field_contract_violations"] == reason
    assert worsened.metrics["newly_failed_field_contract_violations"] == ""
    improved = assess(failed, 0.75 * failed, config=config)
    assert improved.accepted
    assert improved.metrics["inherited_field_contract_violations"] == reason
    assert improved.metrics["regressed_inherited_field_contract_violations"] == ""


@pytest.mark.parametrize("defect", ["strength", "cross_gradient", "nonlinear"])
def test_budget_only_reports_new_and_inherited_gradient_failures_without_refitting(defect):
    field, config = _gradient_case()
    config.joint_group_routing_field_acceptance = "budget_only"
    after = field.b.copy()
    if defect == "strength":
        after *= 1.03
    elif defect == "cross_gradient":
        after[:, 2] += 0.0002 * field.coords[:, 1]
    else:
        after[:, 2] += field.coords[:, 0] ** 2 * 0.2
    new = assess(field.b, after, field, config, budget=1)
    assert new.accepted, new.reasons
    assert not new.metrics["absolute_field_contract_met"]
    assert new.metrics["newly_failed_field_contract_violations"]
    assert new.metrics["inherited_field_contract_violations"] == ""
    assert new.metrics["candidate_linearity_status"] == "FAIL"
    assert new.metrics["current_refitted"] is False
    inherited = assess(after, after.copy(), field, config, budget=0)
    assert inherited.accepted
    assert inherited.metrics["inherited_field_contract_violations"]
    assert inherited.metrics["newly_failed_field_contract_violations"] == ""


@pytest.mark.parametrize("policy", ["absolute", "non_regression", "budget_only"])
def test_longitudinal_bound_uses_normalized_b0_and_preserves_vector_bound(policy):
    field = target(1)
    config = CoilGenConfig(joint_group_routing_field_acceptance=policy, b_0_direction=(3, 4, 0))
    before = np.zeros((1, 3))
    parallel = np.array([[6e-7, 8e-7, 0]])

    def check(after, vector=2e-6, longitudinal=0.5e-6):
        return assess_group_route_field(
            before,
            after,
            field,
            config,
            -2,
            vector,
            maximum_longitudinal_field_delta_t_per_a=longitudinal,
        )

    result = check(parallel)
    assert result.reasons == ("pointwise_longitudinal_field_delta_budget",)
    assert result.metrics["maximum_pointwise_longitudinal_delta_t_per_a"] == pytest.approx(1e-6)
    assert check(parallel, longitudinal=1e-6).accepted
    transverse = np.array([[8e-7, -6e-7, 0]])
    assert check(transverse).accepted
    assert check(transverse, vector=0.5e-6).reasons == ("pointwise_field_delta_budget",)


def test_longitudinal_bound_includes_active_suppression_samples_and_ignores_inactive_data():
    field = target(3, suppressed=[False, True, False], weights=[1, 1, 0])
    before = np.zeros((3, 3))
    after = np.array([[0, 0, 0], [0, 0, 2e-7], [np.nan, np.nan, np.nan]])
    config = CoilGenConfig(joint_group_routing_field_acceptance="budget_only", suppression_field_component="Bx")
    result = assess_group_route_field(
        before,
        after,
        field,
        config,
        1,
        1e-6,
        maximum_longitudinal_field_delta_t_per_a=1e-7,
    )
    assert result.reasons == ("pointwise_longitudinal_field_delta_budget",)
    assert result.metrics["active_sample_count"] == 2


@pytest.mark.parametrize("budget", [-1.0, np.nan, np.inf, True])
def test_longitudinal_bound_rejects_invalid_absolute_budget(budget):
    result = assess_group_route_field(
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        target(1),
        CoilGenConfig(),
        1,
        1e-6,
        maximum_longitudinal_field_delta_t_per_a=budget,
    )
    assert result.reasons == ("invalid_longitudinal_field_delta_budget",)
    assert not result.metrics["checked"]


def test_declared_longitudinal_budget_cannot_be_silently_omitted_by_a_direct_field_call():
    config = CoilGenConfig(joint_group_routing_max_longitudinal_field_change_rel=0.05)
    result = assess(np.zeros((1, 3)), np.zeros((1, 3)), config=config)
    assert result.reasons == ("missing_longitudinal_field_delta_budget",)


def test_budget_only_keeps_suppression_nonregression_and_original_absolute_limits():
    config = CoilGenConfig(joint_group_routing_field_acceptance="budget_only", suppression_field_peak_limit_T=5e-8)
    before = np.array([[0, 0, 1e-7]])
    field = target(1, suppressed=[True])
    unchanged = assess(before, before, field, config, budget=0)
    assert unchanged.accepted
    assert not unchanged.metrics["absolute_field_contract_met"]
    assert unchanged.metrics["inherited_field_contract_violations"] == "suppression_peak_limit"
    assert unchanged.metrics["budget_waived_field_contract_violations"] == ""
    increased = assess(before, 1.1 * before, field, config)
    assert "suppression_peak_regression" in increased.reasons
    assert "suppression_peak_limit" in increased.reasons
    assert increased.metrics["regressed_inherited_field_contract_violations"] == "suppression_peak_limit"
    assert increased.metrics["budget_waived_field_contract_violations"] == ""
    newly_bad = assess(np.zeros_like(before), before, field, config)
    assert "suppression_peak_regression" in newly_bad.reasons
    assert "suppression_peak_limit" in newly_bad.reasons
    assert newly_bad.metrics["newly_failed_field_contract_violations"] == "suppression_peak_limit"
    assert newly_bad.metrics["inherited_field_contract_violations"] == ""


def test_budget_only_keeps_linearity_domain_and_finite_context_hard():
    field, config = _gradient_case()
    config.joint_group_routing_field_acceptance = "budget_only"
    field.coords[:, 2] = 0
    assert assess(field.b, field.b, field, config, budget=0).reasons == ("linearity_domain_rank_deficient",)
    bad = field.b.copy()
    bad[0] = np.nan
    assert assess(field.b, bad, field, config).reasons == ("nonfinite_active_field_context",)


def test_legacy_config_missing_optional_longitudinal_slot_retains_old_policy_behavior():
    config = CoilGenConfig(joint_group_routing_field_acceptance="non_regression")
    del config.joint_group_routing_max_longitudinal_field_change_rel
    result = assess(np.zeros((1, 3)), np.zeros((1, 3)), config=config, budget=0)
    assert result.accepted
    assert "maximum_longitudinal_field_delta_t_per_a" not in result.metrics


@pytest.mark.parametrize("direction", [(0, 0, 1e300), (3e300, 4e300, 0)])
def test_optional_longitudinal_bound_cannot_collapse_for_large_finite_b0_directions(direction):
    config = CoilGenConfig(joint_group_routing_field_acceptance="budget_only", b_0_direction=direction)
    before = np.zeros((1, 3))
    after = np.array([[0, 0, 1e-6]]) if direction[2] else np.array([[6e-7, 8e-7, 0]])
    result = assess_group_route_field(
        before,
        after,
        target(1),
        config,
        1,
        2e-6,
        maximum_longitudinal_field_delta_t_per_a=0.5e-6,
    )
    assert result.reasons == ("pointwise_longitudinal_field_delta_budget",)
    assert result.metrics["maximum_pointwise_longitudinal_delta_t_per_a"] == pytest.approx(1e-6)
