"""Scientific icon integration retains saved scope and labelled actions."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QPushButton

from coilgen.application.cad import GeometryOption
from coilgen.application.design_studies import DesignStudyState, RunReference
from coilgen.application.diagnostics import ProblemIssue
from coilgen.application.history import HistoryEntry
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.service import RunRecord
from coilgen.gui.acceptance_panel import AcceptancePanel
from coilgen.gui.design_brief import DesignBriefPanel, design_brief
from coilgen.gui.export_panel import GeometryExportPanel
from coilgen.gui.history import HistoryPanel
from coilgen.gui.problems import ProblemsPanel, _problem_icon
from coilgen.gui.readiness import ReadinessIssue, ReadinessPanel
from coilgen.gui.workflow import WORKFLOW_STEPS, WorkflowNavigation


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["scientific-icon-evidence-test"])


@pytest.fixture
def widgets(application):
    instances = []

    def make(cls, *args, **kwargs):
        widget = cls(*args, **kwargs)
        instances.append(widget)
        return widget

    yield make
    for widget in instances:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _assert_labelled(button):
    assert not button.icon().isNull()
    assert button.text()
    assert button.accessibleName()


def test_workflow_icons_select_sections_without_marking_them_accepted(widgets):
    panel = widgets(WorkflowNavigation)
    received = []
    panel.step_requested.connect(received.append)
    for step, label, _explanation in WORKFLOW_STEPS:
        button = panel.buttons[step]
        _assert_labelled(button)
        assert button.text() == label
        assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        assert "not accepted" in button.accessibleDescription()
        button.click()
        assert panel.current_step == step and received[-1] == step


def test_stage_badges_use_saved_evidence_and_do_not_promote_evaluated_fields(widgets):
    metadata = {
        "pipeline_stage": "evaluated",
        "summary": {"sheet_accepted": True, "construction_released": False},
        "wire_fields": {
            "contour": {"available": True, "rms_error_T": 1e-6},
            "routed": {"available": True, "rms_error_T": 2e-6},
        },
        "pipeline_sweeps": {"checks_passed": False, "body_count": 1},
        "fabrication": {"ready": False},
    }
    before = deepcopy(metadata)
    panel = widgets(AcceptancePanel)
    panel.set_record(SimpleNamespace(directory=Path("record"), metadata=metadata))
    assert panel.stage_buttons["sheet"].property("evidenceBadge") == "passed"
    assert panel.stage_buttons["contours"].property("evidenceBadge") == "evaluated"
    assert panel.stage_buttons["routed"].property("evidenceBadge") == "evaluated"
    assert panel.stage_buttons["physical"].property("evidenceBadge") == "failed"
    for stage in panel.report.stages:
        button = panel.stage_buttons[stage.key]
        _assert_labelled(button)
        assert stage.status in button.text()
        assert button.accessibleDescription() == stage.detail
    for row, check in enumerate(panel.report.checks):
        assert not panel.table.item(row, 0).icon().isNull()
        assert panel.table.item(row, 5).text() == check.scope
    assert metadata == before


def test_empty_evidence_has_absence_badges_and_disabled_stage_actions(widgets):
    panel = widgets(AcceptancePanel)
    for button in panel.stage_buttons.values():
        _assert_labelled(button)
        assert button.property("evidenceBadge") == "skipped"
        assert not button.isEnabled()
    for button in (panel.locate, panel.settings, panel.problems):
        _assert_labelled(button)
        assert not button.isEnabled()


@pytest.mark.parametrize(
    "kind,support_icon,target_icon,circuit_icon",
    (
        ("cylinder", "support-cylinder", "target-gradient", "connected-winding"),
        ("biplanar", "support-biplanar", "target-gradient", "separate-circuits"),
        ("independent_turns", "support-cylinder", "target-uniform", "independent-turns"),
    ),
)
def test_brief_pictograms_follow_physical_configuration(widgets, kind, support_icon, target_icon, circuit_icon):
    project = ProjectDocument.from_config(preset_config(kind))
    before = project.to_dict()
    brief = design_brief(project)
    rows = {item.key: item for item in brief.items}
    assert rows["support"].icon_name == support_icon
    assert rows["target"].icon_name == target_icon
    assert rows["circuit"].icon_name == circuit_icon
    panel = widgets(DesignBriefPanel)
    panel.set_brief(brief)
    received = []
    panel.section_requested.connect(received.append)
    for key in ("support", "target", "conductor", "circuit", "terminals", "manufacturing"):
        button = panel.section_buttons[key]
        _assert_labelled(button)
        assert button.accessibleDescription() == rows[key].scope
        button.click()
        assert received[-1] == rows[key].section
    assert project.to_dict() == before


def test_contours_only_brief_never_displays_connected_circuit_symbol(widgets):
    project = ProjectDocument.from_config(replace(preset_config(), skip_postprocessing=True))
    row = next(item for item in design_brief(project).items if item.key == "circuit")
    assert row.icon_name == "contours"


def test_history_icons_preserve_baseline_identity_and_owned_selection(widgets, tmp_path):
    metadata = {
        "name": "Saved reference",
        "project_id": "owner",
        "request_sha256": "0" * 64,
        "pipeline_stage": "sheet",
        "summary": {"scope": "continuous_sheet", "accepted": True},
    }
    record = RunRecord(tmp_path / "baseline", metadata, {})
    foreign = RunRecord(tmp_path / "foreign", dict(metadata, project_id="foreign", name="Other project"), {})
    entries = (
        HistoryEntry("baseline", "owner", None, "Reference", record.directory, "0" * 64, record),
        HistoryEntry("foreign", "foreign", None, "Other project", foreign.directory, "0" * 64, foreign),
    )
    panel = widgets(HistoryPanel)
    panel.set_records(entries)
    panel.set_study_state(DesignStudyState(baseline=RunReference.from_metadata(metadata)), project_id="owner")
    assert panel.runs_table.item(0, 0).text() == "Baseline · Reference"
    assert not panel.runs_table.item(0, 0).icon().isNull()
    assert panel.select_record("foreign")
    assert panel.owned_selection is None
    assert not panel.create_variant_button.isEnabled()
    for button in (
        panel.rename_button,
        panel.compare_button,
        panel.pin_baseline_button,
        panel.clear_baseline_button,
        panel.compare_baseline_button,
        panel.create_variant_button,
        panel.save_notes_button,
    ):
        _assert_labelled(button)
    assert all(not panel.comparison_tabs.tabIcon(index).isNull() for index in range(3))


def test_export_icons_leave_saved_geometry_scope_units_and_release_intact(widgets):
    metadata = {
        "name": "Saved conductor",
        "project_revision": 2,
        "summary": {"sheet_accepted": False, "construction_released": False},
        "fabrication": {"ready": True},
    }
    option = GeometryOption(
        name="conductor.stl",
        body_id="body1",
        format="stl",
        scope="saved_geometry_only",
        check_status="passed",
        source_unit="m",
        bounds_m=((0, 0, 0), (0.1, 0.2, 0.3)),
        inspection_only=False,
        source_sha256="0" * 64,
    )
    panel = widgets(GeometryExportPanel, SimpleNamespace(directory=Path("record"), metadata=metadata), (option,))
    for button in (panel.export_button, panel.review_button, panel.problems_button, panel.include_project):
        _assert_labelled(button)
    assert not panel.table.item(0, 0).icon().isNull()
    assert not panel.table.item(0, 2).icon().isNull()
    assert panel.table.item(0, 2).toolTip() == "saved_geometry_only"
    assert "Not released" in panel.findChild(type(panel.scope), "geometry_export_release").text()
    requests = []
    panel.export_requested.connect(lambda *args: requests.append(args))
    panel.units.setCurrentIndex(1)
    panel.export_button.click()
    assert requests == [(("conductor.stl",), "m", True)]
    assert panel.export_button.accessibleName() == panel.export_button.text()


def test_problem_icons_retain_original_evidence_and_missing_witness_guard(widgets, monkeypatch):
    from coilgen.gui import problems

    issues = (
        ProblemIssue(
            "body1:sweep_clearance_violation",
            "Conductor envelopes are too close",
            "Saved capsule screen fails.",
            body_id="body1",
            scope="fabrication_review",
            value=0.001,
            limit=0.003,
            unit="m",
        ),
        ProblemIssue(
            "constraint:2",
            "Independent field RMS error",
            "This sample contract is unverified.",
            severity="warning",
            scope="continuous_sheet",
            settings_node="optimization",
        ),
    )
    monkeypatch.setattr(problems, "run_problems", lambda _record: issues)
    panel = widgets(ProblemsPanel)
    panel.set_run(SimpleNamespace(metadata={"artifacts": [{"name": "request.json", "sha256": "a" * 64}]}))
    received = []
    panel.navigate_requested.connect(received.append)
    for row, issue in enumerate(issues):
        panel.table.selectRow(row)
        assert not panel.table.item(row, 0).icon().isNull()
        assert issue.severity.capitalize() in panel.table.item(row, 0).text()
        assert issue.scope.replace("_", " ") in panel.table.item(row, 1).text().lower()
        assert not panel.locate.isEnabled()
        assert "No spatial witness" in panel.details.text()
        panel.navigate.click()
        assert received[-1] == issue.settings_node
    for button in (panel.locate, panel.navigate):
        _assert_labelled(button)
    assert _problem_icon(issues[0]) == "wire-spacing"
    assert _problem_icon(issues[1]) == "field-error"
    # A field-dependent force limit is a different quantity from field error.
    assert (
        _problem_icon(ProblemIssue("constraint:3", "Force in background field", "Saved force bound.")) == "constraint"
    )
    assert _problem_icon(ProblemIssue("constraint:4", "Wire current limit", "Saved current bound.")) == "constraint"
    panel.set_run(None)
    assert not panel.locate.isEnabled() and not panel.navigate.isEnabled()


def test_readiness_icons_distinguish_blocking_inputs_from_nonblocking_notes(widgets):
    issues = (
        ReadinessIssue("Bend radius must be nonnegative", "winding", "sweep_min_bend_radius", "Set a valid radius."),
        ReadinessIssue(
            "Wire field budget remains unspecified",
            "winding",
            "layout_target_error_rms_limit_T",
            "Choose a budget if required.",
            severity="warning",
            title="Modeling note",
        ),
    )
    panel = widgets(ReadinessPanel)
    panel.set_issues(issues)
    received = []
    panel.issue_requested.connect(received.append)
    assert "Cannot compute" in panel.heading.text()
    for index, (issue, symbol) in enumerate(zip(issues, panel._issue_icons)):
        assert symbol.pixmap() is not None and not symbol.pixmap().isNull()
        assert symbol.property("evidenceBadge") == ("failed" if issue.blocking else "unverified")
        assert symbol.accessibleName() == issue.title
        assert issue.key in symbol.toolTip()
        row = panel.rows.itemAt(index).widget()
        button = row.findChild(QPushButton)
        _assert_labelled(button)
        button.click()
        assert received[-1] is issue
    panel.set_issues(issues[1:])
    assert "Compute is available" in panel.heading.text()
    assert panel._issue_icons[0].property("evidenceBadge") == "unverified"
    panel.set_issues(())
    assert panel.isHidden() and panel._issue_icons == []
