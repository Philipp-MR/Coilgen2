"""A completed numerical solve must not imply a usable physical winding."""

from types import SimpleNamespace

import pytest
import test_native_workbench as native
from PySide6.QtWidgets import QLabel

from coilgen.application.diagnostics import run_problems, run_verdicts

application = native.application
window = native.window


def _metadata():
    return {
        "pipeline_stage": "evaluated",
        "summary": {
            "accepted": True,
            "sheet_accepted": True,
            "scope": "continuous_sheet",
            "construction_released": False,
        },
        "fabrication": None,
        "pipeline_sweeps": {
            "scope": "pipeline_sweep_geometry",
            "body_count": 1,
            "checks_passed": False,
            "bodies": [
                {
                    "body_id": "pipeline_part001_body0001",
                    "manufacturing_ok": False,
                    "mesh_solid_ok": True,
                    "diagnostics": {
                        "manufacturing_failure_reasons": "sweep_clearance_violation;folded_sweep_quad",
                        "sweep_capsule_min_centerline_distance_m": 0.0001,
                        "sweep_required_centerline_clearance_m": 0.0028,
                    },
                }
            ],
        },
    }


def test_sheet_pass_retains_failed_sweep_and_absent_fabrication_review():
    metadata = _metadata()
    assert run_verdicts(metadata) == {"sheet": "Passed", "sweeps": "Failed", "fabrication": "Not run"}
    assert metadata["summary"]["accepted"] is True


@pytest.mark.parametrize("value", [None, 1, 0, "Passed"])
def test_unrecorded_or_non_boolean_evidence_cannot_become_a_pass(value):
    metadata = {
        "summary": {"accepted": True, "scope": "fabrication_review"},
        "pipeline_sweeps": {"checks_passed": value},
    }
    assert run_verdicts(metadata) == {"sheet": "Unverified", "sweeps": "Unverified", "fabrication": "Unverified"}


def test_review_pass_does_not_imply_pipeline_sweep_checks_were_run():
    metadata = {"summary": {"sheet_accepted": True}, "fabrication": {"ready": True}}
    assert run_verdicts(metadata) == {"sheet": "Passed", "sweeps": "Unverified", "fabrication": "Passed"}


def test_explicitly_disabled_sweep_is_not_run_but_missing_evidence_is_unknown():
    metadata = {"config": {"skip_sweep": True}, "pipeline_sweeps": {"body_count": 0, "checks_passed": None}}
    assert run_verdicts(metadata)["sweeps"] == "Not run"
    metadata["config"]["skip_sweep"] = False
    assert run_verdicts(metadata)["sweeps"] == "Unverified"


def test_pipeline_failures_are_actionable_without_invented_spatial_witnesses(tmp_path):
    record = SimpleNamespace(directory=tmp_path, metadata=_metadata(), arrays={})
    issues = run_problems(record)
    assert len(issues) == 2
    assert all(issue.scope == "pipeline_sweep_geometry" and issue.dataset == "swept" for issue in issues)
    assert all(not issue.locatable and issue.settings_node == "winding" for issue in issues)
    clearance = issues[0]
    assert clearance.value == 0.0001 and clearance.limit == 0.0028


def test_failed_sweep_without_reason_codes_is_visible(tmp_path):
    metadata = _metadata()
    metadata["pipeline_sweeps"]["bodies"][0]["diagnostics"] = {}
    issues = run_problems(SimpleNamespace(directory=tmp_path, metadata=metadata, arrays={}))
    assert len(issues) == 1 and issues[0].severity == "error"
    assert not issues[0].locatable


def test_full_run_reviews_distinct_verdicts_and_visible_pipeline_problems(window, tmp_path):
    metadata = _metadata()
    window.service.record = native.FakeRun(tmp_path / "completed", metadata, {})
    window.load_run(window.service.record.directory)
    window.navigate_workflow("review")
    banner = window.result_state.text()
    assert "Sheet checks: Passed" in banner and "Sweep geometry: Failed" in banner
    assert "Fabrication review: Not run" in banner
    assert "engineering acceptance: Passed" not in banner
    texts = [label.text() for label in window.settings.widget().findChildren(QLabel)]
    assert any("Construction release: Not released" in text for text in texts)
    assert any("Pipeline sweep geometry: Failed" in text for text in texts)
    assert window.runs_table.item(0, 1).text() == "Passed"
    assert window.runs_table.item(0, 2).text() == "Failed"
    assert window.runs_table.item(0, 3).text() == "Not run"
    assert "2 errors" in window.problems_panel.summary.text()
    assert "Pipeline sweep" in window.problems_panel.table.item(0, 1).text()
    assert not window.problems_panel.locate.isEnabled()
    assert not window.problems_panel.navigate.isEnabled()
    assert "original request.json" in window.problems_panel.navigate.toolTip()
    assert window.view_results_button.isEnabled()


def test_wire_field_sampling_and_original_current_are_visible(window, tmp_path):
    metadata = _metadata()
    metadata["wire_fields"] = {
        "routed": {"available": True, "current_A": 276.49, "current_refitted": False, "sample_count": 32}
    }
    window.service.record = native.FakeRun(tmp_path / "completed", metadata, {})
    window.load_run(window.service.record.directory)
    assert window.wire_fields_table.item(0, 1).text() == "276.49"
    assert window.wire_fields_table.item(0, 6).text() == "32"
    assert "Design grid" in window.wire_fields_table.item(0, 7).text()
    assert "original current" in window.wire_fields_table.item(0, 7).text()
