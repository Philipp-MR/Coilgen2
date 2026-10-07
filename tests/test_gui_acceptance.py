"""Saved evidence must not broaden budgets, sampling, ownership or acceptance."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from coilgen.gui.acceptance import acceptance_report


def _metadata():
    return {
        "project_id": "owner",
        "project_fingerprint": "saved",
        "pipeline_stage": "evaluated",
        "config": {"target_error_tolerance_percent": 3},
        "summary": {"scope": "fabrication_review", "sheet_accepted": True, "fabrication_review_performed": True},
        "constraints": [
            {
                "name": "standard:dsv",
                "scope": "continuous_sheet",
                "value": 2e-6,
                "limit": 2e-6,
                "effective_limit": 2.1e-6,
                "unit": "T",
                "passed": True,
            },
            {
                "name": "Independent grid field error",
                "scope": "continuous_sheet",
                "value": 2.5e-6,
                "limit": 4e-6,
                "unit": "T",
                "passed": True,
            },
        ],
        "validation": {"available": True, "verified": True, "point_count": 3112},
        "wire_fields": {
            "contour": {
                "available": True,
                "relative_rms_error": 0.09,
                "rms_error_T": 3e-6,
                "current_A": 18,
                "current_refitted": False,
                "sample_count": 257,
                "component": "Bz",
                "geometry_sources": ["oriented_unconnected_contours"],
            },
            "routed": {
                "available": True,
                "relative_rms_error": 0.1,
                "rms_error_T": 4e-6,
                "current_A": 20,
                "current_refitted": True,
                "sample_count": 257,
                "component": "Bz",
                "geometry_sources": ["fabricated_centroid"],
            },
        },
        "pipeline_sweeps": {"checks_passed": False, "body_count": 1},
        "fabrication": {"ready": False},
    }


def _check(report, key):
    return next(row for row in report.checks if row.key == key)


def _stage(report, key):
    return next(row for row in report.stages if row.key == key)


def test_no_record_and_sparse_record_do_not_establish_passes():
    assert all(stage.status == "Not run" for stage in acceptance_report().stages)
    report = acceptance_report({})
    assert all(stage.status == "Unverified" for stage in report.stages)
    assert not any(check.status == "Passed" for check in report.checks)


def test_saved_effective_limit_governs_margin_even_after_editing_requirements():
    metadata = _metadata()
    before = deepcopy(metadata)
    report = acceptance_report(
        metadata,
        project=SimpleNamespace(id="owner", fingerprint="edited", values={"target_error_tolerance_percent": 1}),
    )
    row = _check(report, "constraint:0")
    assert row.limit == 2e-6 and row.effective_limit == 2.1e-6
    assert row.margin == pytest.approx(1e-7)
    assert row.status == "Passed"
    assert "Earlier model/study" in report.identity
    assert metadata == before


def test_contour_and_routed_metrics_do_not_inherit_sheet_acceptance():
    report = acceptance_report(_metadata())
    assert _stage(report, "sheet").status == "Passed"
    assert _stage(report, "physical").status == "Failed"
    for key, expected in (("contours:relative", 9), ("routed:relative", 10)):
        row = _check(report, key)
        assert row.observed == pytest.approx(expected)
        assert row.status == "Evaluated"
        assert row.limit is row.effective_limit is row.margin is None
        assert "257 active samples" in row.scope
        assert "descriptive" in row.scope
    assert "current refitted" in _check(report, "routed:rms").scope
    assert "3112 samples" in _check(report, "constraint:1").scope


def test_false_verified_flag_cannot_be_promoted_by_a_passed_boolean():
    metadata = _metadata()
    metadata["validation"]["verified"] = False
    report = acceptance_report(metadata)
    assert _check(report, "constraint:1").status == "Unverified"
    assert _check(report, "constraint:1").margin is None
    assert _stage(report, "sheet").status == "Unverified"


@pytest.mark.parametrize(
    "stage, expected",
    (("prepared", "Not run"), ("sheet", "Not run"), ("contours", "Not run"), ("routed", "Unverified")),
)
def test_stage_boundary_distinguishes_missing_evidence_from_not_run(stage, expected):
    report = acceptance_report({"pipeline_stage": stage, "summary": {"fabrication_review_performed": False}})
    assert _stage(report, "routed").status == expected
    assert _check(report, "physical:review").status == "Not run"


def test_independent_turns_report_connection_stage_not_run():
    metadata = _metadata()
    metadata["wire_fields"].pop("routed")
    metadata["config"]["contour_only_sweep_mode"] = True
    assert _stage(acceptance_report(metadata), "routed").status == "Not run"


@pytest.mark.parametrize(
    "owner,fingerprint,pending,expected",
    (
        ("other", "saved", False, "Another channel"),
        ("owner", "saved", True, "Pending edits"),
        ("owner", "saved", False, "Current applied"),
        (None, "saved", False, "Run identity unverified"),
    ),
)
def test_identity_is_separate_from_saved_verdict(owner, fingerprint, pending, expected):
    metadata = _metadata()
    metadata["project_id"] = owner
    report = acceptance_report(metadata, project=SimpleNamespace(id="owner", fingerprint=fingerprint), pending=pending)
    assert expected in report.identity
    assert _stage(report, "sheet").status == "Passed"


def test_run_identity_includes_saved_name_and_revision_with_explicit_missing_revision():
    metadata = _metadata()
    metadata.update(name="Cylindrical Y gradient", project_revision=4)
    assert acceptance_report(metadata).identity.startswith("Cylindrical Y gradient · revision 4")
    metadata.pop("project_revision")
    assert "revision unrecorded" in acceptance_report(metadata).identity


def test_lower_geometry_margin_keeps_pipeline_and_review_realizations_separate():
    metadata = _metadata()
    metadata["pipeline_sweeps"]["bodies"] = [
        {
            "body_id": "body1",
            "diagnostics": {
                "sweep_final_min_bend_radius_m": 0.004,
                "sweep_required_bend_radius_m": 0.006,
                "sweep_final_min_bend_radius_violation": True,
            },
        }
    ]
    metadata["fabrication"]["paths"] = [
        {
            "body_id": "body1",
            "diagnostics": {
                "sweep_final_min_bend_radius_m": 0.007,
                "sweep_required_bend_radius_m": 0.006,
                "sweep_final_min_bend_radius_violation": False,
            },
        }
    ]
    report = acceptance_report(metadata)
    pipeline = _check(report, "body:Pipeline sweep:body1:bend")
    review = _check(report, "body:Strict review:body1:bend")
    assert pipeline.status == "Failed" and pipeline.margin == pytest.approx(-0.002)
    assert review.status == "Passed" and review.margin == pytest.approx(0.001)
    assert pipeline.comparison == review.comparison == "lower"
    assert pipeline.scope != review.scope


def test_closed_geometry_does_not_claim_drive_feed_design_or_construction_release():
    metadata = _metadata()
    metadata["winding_steps"] = [{"circuit_complete": True, "drive_feed_designed": False}]
    report = acceptance_report(metadata)
    assert _check(report, "physical:feeds").status == "Not designed"
    assert "Construction release is not established" in report.message


def test_skipped_clearance_check_is_not_certified_by_a_false_violation_flag():
    metadata = _metadata()
    metadata["fabrication"]["paths"] = [
        {
            "body_id": "body1",
            "diagnostics": {
                "sweep_capsule_min_centerline_distance_m": 0.004,
                "sweep_required_centerline_clearance_m": 0.003,
                "sweep_capsule_clearance_violation": False,
                "sweep_self_intersection_skipped": True,
            },
        }
    ]
    row = _check(acceptance_report(metadata), "body:Strict review:body1:clearance")
    assert row.status == "Not run" and row.margin is None


def test_saved_run_config_supplies_frame_rotation_limit_without_current_editor_values():
    metadata = _metadata()
    metadata["config"]["sweep_max_frame_rotation_deg"] = 10
    metadata["fabrication"]["paths"] = [
        {
            "body_id": "body1",
            "diagnostics": {
                "sweep_final_max_frame_rotation_deg": 12,
                "sweep_final_frame_rotation_violation": True,
            },
        }
    ]
    row = _check(acceptance_report(metadata), "body:Strict review:body1:frame")
    assert row.status == "Failed" and row.limit == 10 and row.margin == -2


def test_individual_geometry_failure_cannot_be_hidden_by_aggregate_passes():
    metadata = _metadata()
    metadata["pipeline_sweeps"]["checks_passed"] = True
    metadata["fabrication"]["ready"] = True
    metadata["fabrication"]["paths"] = [
        {
            "body_id": "body1",
            "diagnostics": {
                "sweep_final_min_bend_radius_m": 0.004,
                "sweep_required_bend_radius_m": 0.006,
                "sweep_final_min_bend_radius_violation": True,
            },
        }
    ]
    assert _stage(acceptance_report(metadata), "physical").status == "Failed"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "3"))
def test_malformed_numeric_values_never_generate_margins(value):
    metadata = _metadata()
    metadata["constraints"][0]["value"] = value
    row = _check(acceptance_report(metadata), "constraint:0")
    assert row.observed is None and row.margin is None
