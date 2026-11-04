"""Engineering measurements and individual saved failures shown in the browser."""

import pytest
import numpy as np

pytest.importorskip("trame")

from coilgen.application.service import RunRecord
from coilgen.application.scene_spec import DATASET_LABELS, PREFERRED_RUN_PLOTS
from coilgen.web.app import create_app
from coilgen.web.result_display import result_metrics, result_overview


def test_default_plot_order_covers_all_saved_datasets_without_using_editing_geometry():
    assert PREFERRED_RUN_PLOTS[0] == "review_conductor"
    assert set(PREFERRED_RUN_PLOTS) == DATASET_LABELS.keys() - {"support", "preview_target"}


def test_browser_report_shows_percent_errors_units_and_separate_geometry_checks(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-readable-results")
    browser.load_template("cylinder")
    project = browser.session.project
    diagnostics = {
        "manufacturing_failure_reasons": "route_clearance_violation;minimum_bend_radius_violation",
        "sweep_route_clearance_margin_m": -0.00025,
        "sweep_final_min_bend_radius_m": 0.0014468,
        "sweep_required_bend_radius_m": 0.006,
    }
    report = {
        "name": project.name,
        "project_id": project.id,
        "project_fingerprint": project.fingerprint,
        "project_revision": project.revision,
        "pipeline_stage": "evaluated",
        "summary": {
            "accepted": False,
            "sheet_accepted": True,
            "relative_field_error": 0.03,
            "validated_relative_field_error": 0.03109,
            "resistance_ohm": 0.0123,
            "inductance_h": 4.2e-6,
            "minimum_spacing_m": 0.0032,
        },
        "fabrication": {
            "ready": False,
            "field_metrics": {
                "dsv_relative_residual_percent": 5.019,
                "current_source": "original_contour_step_no_refit",
            },
            "paths": [{"body_id": "conductor", "diagnostics": diagnostics}],
            "failure_reasons": ["conductor:route_clearance_violation;minimum_bend_radius_violation"],
        },
        "pipeline_sweeps": {
            "checks_passed": False,
            "bodies": [{"body_id": "pipeline", "diagnostics": diagnostics}],
        },
        "constraints": [
            {
                "name": "Fabricated path field error (original current)",
                "passed": False,
                "value": 3e-5,
                "effective_limit": 2e-5,
                "unit": "T",
            }
        ],
    }
    record = RunRecord(tmp_path, report, {})
    monkeypatch.setattr(browser.session, "load_run", lambda _id: record)
    monkeypatch.setattr(browser.session, "export_options", lambda _: ())
    browser._open_run("saved-run")
    state = server.state
    rows = {row["name"]: row["value"] for row in state.metrics}
    assert state.summary["reviewed_wire_metric"] == {
        "name": "Reviewed wire relative RMS field error (design DSV, original current)",
        "value": "5.019 %",
    }
    assert rows["Stream function relative field error (design samples)"] == "3 %"
    assert rows["Stream function relative field error (independent samples)"] == "3.109 %"
    assert rows["Fabricated circuit resistance"] == "12.3 mΩ"
    assert rows["Fabricated circuit inductance"] == "4.2 µH"
    assert rows["Minimum unconnected contour spacing (surface distance)"] == "3.2 mm"
    assert state.summary["label"] == "Engineering checks failed"
    # Metadata-only archives must never borrow a support from the editing model.
    assert state.dataset == state.result_plot == "saved_unavailable"
    assert "no viewable plot data" in state.scene_caption
    assert state.summary["overview"]["field_cards"] == []
    groups = {group["title"]: group for group in state.summary["problems"]}
    for title, body in (("Strict fabrication review", "conductor"), ("Pipeline conductor sweep", "pipeline")):
        assert groups[title]["counts"] == "2 failed"
        issue = next(item for item in groups[title]["items"] if item["title"] == "Conductor bend radius is too small")
        assert (
            body in issue["details"] and "Measured 1.4468 mm" in issue["details"] and "Limit 6 mm" in issue["details"]
        )
    (field,) = groups["Numerical and engineering constraints"]["items"]
    assert field["status"] == "Failed" and field["title"] == "Fabricated path field error (original current)"
    assert "3e-05 T" in field["details"] and "2e-05 T" in field["details"]
    assert "route_clearance_violation" not in state.draft_notice


def test_prepared_model_does_not_invent_an_independent_field_error(tmp_path):
    rows = result_metrics({"pipeline_stage": "prepared", "summary": {"vertices": 12}})
    assert [row["value"] for row in rows] == ["Not run", "Not run", "12"]
    record = RunRecord(
        tmp_path,
        {"pipeline_stage": "prepared", "summary": {"vertices": 12}},
        {key: np.zeros((12, 3)) for key in ("target_points_m", "sheet_field_T")},
    )
    board = result_overview(record)
    assert not board["field_cards"] and not board["coil_cards"]
    assert board["partial_note"] == "Prepare model stops before stream function optimization."


def test_failed_independent_winding_check_cannot_be_hidden_by_passing_geometry(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-independent-verdict")
    record = RunRecord(
        tmp_path,
        {
            "pipeline_stage": "evaluated",
            "summary": {"accepted": True, "sheet_accepted": True},
            "fabrication": {"ready": True},
            "pipeline_sweeps": {"checks_passed": True},
            "winding_validation": {"available": True, "passed": False},
        },
        {},
    )
    monkeypatch.setattr(browser.session, "load_run", lambda _id: record)
    monkeypatch.setattr(browser.session, "export_options", lambda _: ())
    browser._open_run("saved-run")
    assert server.state.summary["label"] == "Engineering checks failed"
    assert {"name": "Independent winding field", "value": "Failed"} in server.state.summary["check_rows"]


def test_browser_result_cards_use_signed_original_current_and_report_separate_bodies(tmp_path):
    record = RunRecord(
        tmp_path,
        {
            "config": {"target_field_component": "By"},
            "wire_fields": {
                "routed": {
                    "available": True,
                    "current_refitted": True,
                    "current_A": 8.8,
                    "rms_error_T": 0,
                    "at_stream_current": {
                        "available": True,
                        "current_A": -1.25,
                        "rms_error_T": 3.1e-6,
                        "peak_error_T": 4.2e-6,
                        "relative_rms_error": 0.031,
                        "target_rms_T": 1e-4,
                        "sample_count": 12,
                    },
                }
            },
            "fabrication": {
                "electrical_metrics": {
                    "bodies": [
                        {"body_id": "a", "circuit_complete": True, "resistance_ohm": 0.1},
                        {"body_id": "b", "circuit_complete": True, "resistance_ohm": 0.2},
                    ]
                }
            },
        },
        {key: np.zeros((12, 3)) for key in ("target_points_m", "routed_field_T", "routed_field_at_stream_current_T")},
    )
    board = result_overview(record)
    assert {card["name"]: card["value"] for card in board["field_cards"]} == {
        "RMS error": "3.1 µT",
        "Peak error": "4.2 µT",
        "Relative RMS error": "3.1 %",
        "Original current": "-1.25 A",
    }
    assert "original contour current · By error" in board["field_context"]
    assert "n=12" in board["field_context"]
    assert board["field_plot"] == ""  # the browser's routed-field plot uses a refit
    assert board["coil_cards"][0]["value"] == "Per body"
    resistances = {row["body"]: row["value"] for row in board["body_rows"] if row["name"] == "DC resistance"}
    assert resistances == {"a": "0.1 Ω", "b": "0.2 Ω"}
