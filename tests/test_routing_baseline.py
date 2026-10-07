"""An infeasible incumbent must not veto a certified repair on objective value."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import routing_baseline
from coilgen.config import CoilGenConfig
from coilgen.group_route_validation import GroupRouteValidationLimits
from coilgen.models import TargetField


@pytest.fixture
def baseline_case(monkeypatch):
    parts = [
        SimpleNamespace(manufacturing_diagnostics={"manufacturing_ok": True}, swept_body=object()) for _ in range(2)
    ]
    fields = [np.tile([0, 0, value], (2, 1)) for value in (1e-6, 2e-6)]
    context = TargetField(
        np.array([[0, 0, 0], [0.01, 0.02, 0.03]]),
        (fields[0] + fields[1]) * 2,
        np.ones(2),
        np.zeros(2, bool),
        np.ones(2, int),
    )
    prepared = SimpleNamespace(
        reasons=(),
        diagnostics={"manufacturing_ok": True, "exact_collision_completed": True},
        route_field_per_amp=fields[0].copy(),
        physical_field_per_amp=fields[0].copy(),
    )
    calls = []
    monkeypatch.setattr(
        routing_baseline,
        "certify_group_route_support",
        lambda *_: SimpleNamespace(
            accepted=True,
            reasons=(),
            diagnostics={"support_checked": True},
        ),
    )

    def audit(bodies, gap, limits):
        calls.append((bodies, gap, limits))
        return {"interpart_clearance_completed": True}, ""

    monkeypatch.setattr(routing_baseline, "audit_interpart_clearance", audit)
    kwargs = dict(
        part=parts[0],
        parts=parts,
        index=0,
        config=CoilGenConfig(layout_target_error_rms_limit_T=1e-9, routing_circuit_scope="per_part"),
        target_field=context,
        current_a=2.0,
        physical_fields=deepcopy(fields),
        route_fields=fields,
        prepared_baseline=prepared,
        limits=GroupRouteValidationLimits(),
    )
    return kwargs, calls


def test_feasible_incumbent_uses_audited_own_fields_and_both_unchanged_current_backgrounds(baseline_case):
    kwargs, calls = baseline_case
    # The content-audited baseline owns its own field; only other entries supply backgrounds.
    kwargs["physical_fields"][0][:] = 99
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert result["accepted"], result
    for stage in ("route", "physical"):
        assert result[f"{stage}_field"]["candidate_dsv_residual_rms_T"] == 0
        assert result[f"{stage}_field"]["maximum_pointwise_vector_delta_t_per_a"] == 0
        assert result[f"{stage}_field"]["current_a"] == 2
        assert not result[f"{stage}_field"]["current_refitted"]
    assert len(calls) == 1
    assert calls[0][0] == [part.swept_body for part in kwargs["parts"]]


@pytest.mark.parametrize("stage", ["route", "physical"])
def test_manufacturing_ok_does_not_waive_a_fixed_current_field_violation(baseline_case, stage):
    kwargs, _ = baseline_case
    getattr(kwargs["prepared_baseline"], f"{stage}_field_per_amp")[:, 2] += 1e-6
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"]
    assert f"{stage}_layout_target_rms_limit" in result["reasons"]
    assert kwargs["part"].manufacturing_diagnostics["manufacturing_ok"]


def test_baseline_is_not_required_to_match_original_contours_without_a_declared_field_limit(baseline_case):
    kwargs, _ = baseline_case
    kwargs["config"].layout_target_error_rms_limit_T = 0
    kwargs["target_field"].b[:] = 100
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert result["accepted"], result
    assert result["physical_field"]["candidate_dsv_residual_rms_T"] > 1


def test_existing_bad_suppression_cannot_pass_a_declared_limit_by_comparing_to_itself(baseline_case):
    kwargs, _ = baseline_case
    kwargs["target_field"].suppressed[-1] = True
    kwargs["target_field"].b[-1] = 0
    kwargs["config"].suppression_field_peak_limit_T = 1e-9
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"]
    assert "physical_suppression_peak_limit" in result["reasons"]


def test_support_failure_prevents_a_manufacturable_incumbent_from_vetoing_repairs(baseline_case, monkeypatch):
    kwargs, _ = baseline_case
    monkeypatch.setattr(
        routing_baseline,
        "certify_group_route_support",
        lambda *_: SimpleNamespace(
            accepted=False,
            reasons=("support_penetration",),
            diagnostics={"support_checked": True},
        ),
    )
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"]
    assert "support_penetration" in result["reasons"]


def test_prepared_exact_audit_failure_is_not_overruled_by_legacy_manufacturing_flag(baseline_case):
    kwargs, _ = baseline_case
    kwargs["prepared_baseline"].reasons = ("triangle_pair_budget_exceeded",)
    kwargs["prepared_baseline"].diagnostics["exact_collision_completed"] = False
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"]
    assert "triangle_pair_budget_exceeded" in result["reasons"]
    assert "baseline_exact_collision_unchecked" in result["reasons"]


def test_absent_manufacturing_certificate_is_unchecked_even_without_a_reason_string(baseline_case):
    kwargs, _ = baseline_case
    kwargs["prepared_baseline"].diagnostics.pop("manufacturing_ok")
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"]
    assert "baseline_manufacturing_not_certified" in result["reasons"]


def test_other_invalid_body_does_not_supply_a_clearance_certificate(baseline_case):
    kwargs, calls = baseline_case
    kwargs["parts"][1].manufacturing_diagnostics["manufacturing_ok"] = False
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"]
    assert "other_part_manufacturing_not_certified" in result["reasons"]
    assert not calls


@pytest.mark.parametrize("reason", ["interpart_surface_clearance_violation", "interpart_solid_containment"])
def test_interpart_obstructions_are_incumbent_failures_too(baseline_case, monkeypatch, reason):
    kwargs, _ = baseline_case
    monkeypatch.setattr(
        routing_baseline,
        "audit_interpart_clearance",
        lambda *_: (
            {"interpart_clearance_completed": False},
            reason,
        ),
    )
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"] and reason in result["reasons"]


def test_interpart_gap_matches_conductor_insulation_and_tolerance_contract(baseline_case):
    kwargs, calls = baseline_case
    kwargs["config"].sweep_insulation_thickness = 0.001
    kwargs["config"].sweep_manufacturing_tolerance = 0.0002
    kwargs["config"].sweep_bend_clearance_margin = 0.0003
    assert routing_baseline.assess_routing_baseline(**kwargs)["accepted"]
    assert calls[0][1] == pytest.approx(0.0025)


@pytest.mark.parametrize("defect", ["no_dsv", "current", "other_field", "missing_physical", "cache_length"])
def test_unchecked_field_context_is_not_a_feasible_incumbent(baseline_case, defect):
    kwargs, _ = baseline_case
    if defect == "no_dsv":
        kwargs["target_field"].weights[:] = 0
    elif defect == "current":
        kwargs["current_a"] = 0
    elif defect == "other_field":
        kwargs["route_fields"][1] = np.zeros((1, 3))
    elif defect == "missing_physical":
        kwargs["prepared_baseline"].physical_field_per_amp = None
    else:
        kwargs["physical_fields"] = []
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert not result["accepted"] and result["reasons"]


def test_certification_preserves_source_diagnostics_and_fields(baseline_case):
    kwargs, _ = baseline_case
    original = deepcopy((kwargs["part"].manufacturing_diagnostics, kwargs["prepared_baseline"], kwargs["route_fields"]))
    assert routing_baseline.assess_routing_baseline(**kwargs)["accepted"]
    assert kwargs["part"].manufacturing_diagnostics == original[0]
    assert kwargs["prepared_baseline"].diagnostics == original[1].diagnostics
    np.testing.assert_array_equal(
        kwargs["prepared_baseline"].physical_field_per_amp, original[1].physical_field_per_amp
    )
    np.testing.assert_array_equal(kwargs["route_fields"], original[2])


def test_inactive_nonfinite_field_sums_do_not_invalidate_a_feasible_incumbent(baseline_case):
    kwargs, _ = baseline_case
    kwargs["target_field"].weights[-1] = 0
    for stage in ("route", "physical"):
        getattr(kwargs["prepared_baseline"], f"{stage}_field_per_amp")[-1] = np.inf
        kwargs[f"{stage}_fields"][1][-1] = -np.inf
    result = routing_baseline.assess_routing_baseline(**kwargs)
    assert result["accepted"], result
