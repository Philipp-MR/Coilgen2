"""Decision views retain recorded scopes, dates and ownership through filtering."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QDialog

from coilgen.application.system_presets import system_assembly
from coilgen.gui.acceptance_panel import AcceptancePanel
from coilgen.gui.channels_panel import ChannelOverview
from coilgen.gui.export_panel import GeometryExportPanel
from coilgen.gui.history import HistoryPanel
from coilgen.gui.run_context import run_compatibility, run_timestamp
from test_gui_acceptance_panel import _record as acceptance_record
from test_gui_export_panel import _options


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["result-decision-test"])


@pytest.fixture
def panels(application):
    widgets = []

    def make(kind, *args, **kwargs):
        panel = kind(*args, **kwargs)
        widgets.append(panel)
        return panel

    yield make
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def native_font(application):
    font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        pytest.skip("Native font is unavailable.")
    yield QFont(families[0], 9)
    QFontDatabase.removeApplicationFont(font_id)


def _run(project, *, fingerprint=None, revision=None, date="2026-10-04T16:32:10+00:00", sheet=True, fabrication=False):
    return SimpleNamespace(
        directory=Path("owned_result"),
        metadata={
            "name": "Saved engineering design",
            "project_id": project.id,
            "project_fingerprint": project.fingerprint if fingerprint is None else fingerprint,
            "project_revision": project.revision if revision is None else revision,
            "pipeline_stage": "evaluated",
            "job": {"created_at": date},
            "summary": {"sheet_accepted": sheet},
            "fabrication": {"ready": fabrication},
            "pipeline_sweeps": {"checks_passed": True, "body_count": 1},
        },
    )


def _entry(id_, record):
    return SimpleNamespace(
        id=id_, run=record, diagnostic="", label=id_, alias=id_, directory=record.directory, verified=True
    )


def test_acceptance_default_attention_view_and_stage_filter_preserve_checks(panels):
    panel = panels(AcceptancePanel)
    record = acceptance_record()
    before = copy.deepcopy(record.metadata)
    panel.set_record(record)
    for row, check in enumerate(panel.report.checks):
        assert panel.table.isRowHidden(row) == (check.status not in {"Failed", "Unverified", "Not designed", "Not run"})
    assert "Failures and evidence gaps" in panel.filter_summary.text()
    panel.show_all.setChecked(True)
    assert all(not panel.table.isRowHidden(row) for row in range(panel.table.rowCount()))
    panel.stage_buttons["physical"].click()
    for row, check in enumerate(panel.report.checks):
        assert panel.table.isRowHidden(row) == (check.stage != "physical")
    assert panel.stage_buttons["physical"].isChecked()
    panel.stage_buttons["physical"].click()
    assert not panel.stage_buttons["physical"].isChecked()
    assert all(not panel.table.isRowHidden(row) for row in range(panel.table.rowCount()))
    assert record.metadata == before


def test_standalone_results_identity_uses_only_original_recorded_submission_date(panels):
    panel = panels(AcceptancePanel)
    record = acceptance_record("Imported on 2020-01-01")
    record.metadata["name"] = "Display alias dated 2025-02-03"
    record.metadata["job"] = {"created_at": "2026-10-04T16:32:10+00:00"}
    before = copy.deepcopy(record.metadata)
    panel.set_record(record)
    original_checks = panel.report.checks
    date = run_timestamp(record.metadata)
    assert "Submitted: " + date.text in panel.identity.text()
    assert date.tooltip in panel.identity.toolTip()
    assert "UTC:" in panel.identity.toolTip() and "Local:" in panel.identity.toolTip()
    record.metadata.pop("job")
    panel.set_record(record)
    assert "Submitted: date unrecorded" in panel.identity.text()
    assert "Submitted: 2020" not in panel.identity.text() and "Submitted: 2025" not in panel.identity.text()
    assert "No valid execution timestamp" in panel.identity.toolTip()
    assert panel.report.checks == original_checks
    assert record.metadata == {key: value for key, value in before.items() if key != "job"}


@pytest.mark.parametrize("identity", ("project_id", "request_sha256"))
def test_explicitly_mismatched_live_job_never_supplies_result_timestamp(identity):
    metadata = {
        "project_id": "owner",
        "request_sha256": "request",
        "job": {"project_id": "owner", "request_sha256": "request", "created_at": "2026-10-04T16:32:10Z"},
    }
    before = copy.deepcopy(metadata)
    assert run_timestamp(metadata).timestamp is not None
    metadata["job"][identity] = "another-result"
    assert run_timestamp(metadata).timestamp is None
    record = SimpleNamespace(metadata=metadata, execution_times={"created_at": "2026-10-03T15:22:00Z"})
    assert run_timestamp(record).timestamp == run_timestamp({"job": {"created_at": "2026-10-03T15:22:00Z"}}).timestamp
    metadata["job"].pop(identity)
    assert run_timestamp(metadata).timestamp is not None
    assert metadata == {**before, "job": {key: value for key, value in before["job"].items() if key != identity}}


def test_current_and_electrical_summary_never_merges_refit_with_original(panels):
    record = acceptance_record()
    record.metadata["wire_fields"] = {
        "routed": {
            "available": True,
            "current_refitted": True,
            "current_A": 7.5,
            "current_source": "least-squares refit",
            "at_stream_current": {"available": True, "current_A": 8.25, "current_refitted": False},
        }
    }
    record.metadata["summary"].update(
        current_a=8.25, resistance_ohm=0.41, inductance_h=12e-6, power_w=9.7, fabricated_power_w=27.906
    )
    panel = panels(AcceptancePanel)
    panel.set_record(record)
    assert "Connected conductor: original wire current 8.25 A" in panel.metrics.text()
    assert "Connected conductor: saved refitted current 7.5 A" in panel.metrics.text()
    assert "Reviewed circuit: I 8.25 A · R 0.41 Ω · L 12 µH" in panel.metrics.text()
    assert "Sheet power 9.7 W" in panel.metrics.text()
    assert "Reviewed conductor Joule power 27.906 W" in panel.metrics.text()
    record.metadata["summary"]["scope"] = "continuous_sheet"
    panel.set_record(record)
    assert "Reviewed circuit" not in panel.metrics.text()
    record.metadata["wire_fields"] = {}
    record.metadata["summary"]["power_w"] = float("nan")
    panel.set_record(record)
    assert not panel.metrics.text()


def test_history_dates_are_recorded_submission_dates_and_filters_use_full_identity(panels):
    project = system_assembly().channels[0].project
    current = _run(project, revision=project.revision + 20)  # Same values can recur at another revision.
    earlier = _run(project, fingerprint="earlier", revision=project.revision, date=None)
    unknown = _run(project)
    unknown.metadata.pop("project_fingerprint")
    entries = (_entry("current", current), _entry("earlier", earlier), _entry("unknown", unknown))
    panel = panels(HistoryPanel)
    panel.set_records(entries, project=project)
    assert panel.runs_table.item(0, 7).text() == run_timestamp(current.metadata).text
    assert "UTC:" in panel.runs_table.item(0, 7).toolTip()
    assert panel.runs_table.item(1, 7).text() == "Not recorded"
    assert panel.runs_table.item(0, 8).text() == "Matches model/study"
    assert panel.runs_table.item(1, 8).text() == "Earlier model/study"
    assert panel.runs_table.item(2, 8).text() == "Identity unverified"
    assert panel.runs_table.horizontalHeader().visualIndex(7) == 1
    assert panel.runs_table.horizontalHeader().visualIndex(8) == 2
    panel.compatibility_filter.setCurrentIndex(panel.compatibility_filter.findData("current"))
    assert not panel.runs_table.isRowHidden(0)
    assert panel.runs_table.isRowHidden(1) and panel.runs_table.isRowHidden(2)
    assert panel.filter_count.text() == "1 / 3"
    assert panel.select_record("earlier")  # An explicit request clears the presentation filter.
    assert panel.compatibility_filter.currentData() == "all"
    assert panel.selected_entries == (entries[1],)
    assert run_compatibility(current.metadata, project)[0] == "current"


def test_history_filter_cannot_compare_selected_hidden_runs(panels):
    project = system_assembly().channels[0].project
    earlier = _entry("old", _run(project, fingerprint="older"))
    panel = panels(HistoryPanel)
    panel.set_records((earlier,), project=project)
    panel.select_record("old")
    panel.compatibility_filter.setCurrentIndex(panel.compatibility_filter.findData("current"))
    assert panel.selected_entries == ()
    assert not panel.compare_button.isEnabled() and not panel.pin_baseline_button.isEnabled()
    assert not panel.create_variant_button.isEnabled()


def test_history_decisions_open_on_demand_and_preserve_unsaved_notes(panels):
    panel = panels(HistoryPanel)
    assert panel.details_window.isHidden()
    assert panel.comparison_tabs.parentWidget() is panel.details_window
    panel.decisions_button.click()
    assert panel.details_window.isVisible()
    assert panel.comparison_tabs.currentIndex() == 2
    panel.hypothesis.setPlainText("Test a 2 mm conductor on unchanged target samples")
    panel.details_window.close()
    panel.decisions_button.click()
    assert panel.hypothesis.toPlainText() == "Test a 2 mm conductor on unchanged target samples"


def test_channel_overview_prefers_matching_owned_evidence_and_separates_execution(panels):
    assembly = system_assembly()
    channel = assembly.channels[0]
    matching = _run(channel.project)
    earlier = _run(channel.project, fingerprint="earlier", date="2026-10-05T16:32:10+00:00")
    foreign = _run(assembly.channels[1].project, date="2026-10-06T16:32:10+00:00")
    states = [
        {
            "project_id": channel.project.id,
            "project_fingerprint": "earlier",
            "status": "running",
            "project_revision": 2,
            "stage": "Current sheet",
            "created_at": "2026-10-05T17:00:00+00:00",
        }
    ]
    panel = panels(
        ChannelOverview, assembly, channel.id, states=states, records={channel.id: (matching, earlier, foreign)}
    )
    assert panel.table.item(0, 6).text() == "Running · rev 2"
    assert panel.table.item(0, 7).text() == f"Matching · rev {channel.project.revision}"
    assert "2026-10-04" in panel.table.item(0, 7).toolTip()
    assert panel.table.item(0, 8).text() == "Passed"
    assert panel.table.item(0, 9).text() == "Sweep Passed · Review Failed"
    assert "Wait for the active study" in panel.next_action.text()
    assert all(panel.table.isColumnHidden(column) for column in range(2, 6))
    panel.refresh_evidence(records={channel.id: (earlier,)})
    assert panel.table.item(0, 7).text() == f"Earlier · rev {channel.project.revision}"
    assert "Compute this channel's current" in panel.next_action.text()
    panel.configuration_button.click()
    window = panel.findChild(QDialog, "channel_configuration_window")
    assert window is not None and window.isVisible()
    assert window.windowTitle().startswith(channel.name)
    window.close()


def test_export_context_distinguishes_older_cad_and_current_owned_assembly(panels):
    assembly = system_assembly()
    owner = assembly.channels[0].project
    current_other_channel = assembly.channels[1].project
    record = _run(owner, fingerprint="older", revision=4)
    before = copy.deepcopy(record.metadata)
    panel = panels(
        GeometryExportPanel, record, _options(), project=current_other_channel, assembly=assembly, pending=True
    )
    text = panel.companion_context.text()
    assert "CAD source: saved revision 4" in text
    assert f"Current assembly (9 channels) companion: applied revision {owner.revision}" in text
    assert "earlier model/study" in text
    assert "Pending edits will be applied" in text
    assert "CAD remains the selected run's geometry" in text
    panel.include_project.setChecked(False)
    assert panel.companion_context.isHidden()
    assert panel.selected_names() == (_options()[0].name,)
    assert record.metadata == before


def test_compact_results_preserve_status_scope_margin_and_full_saved_checks(panels, application, native_font):
    record = acceptance_record()
    panel = panels(AcceptancePanel)
    panel.setFont(native_font)
    panel.set_record(record)
    panel.resize(778, 620)  # Content width of an 800 px focused window.
    panel.show()
    for _ in range(4):
        application.processEvents()
    assert panel.table.horizontalScrollBar().maximum() == 0
    assert all(not panel.table.isColumnHidden(column) for column in range(6))
    row = next(
        row for row, check in enumerate(panel.report.checks) if check.key == "body:Strict review:body1:clearance"
    )
    panel.table.selectRow(row)
    assert "Recorded margin: -2 mm" in panel.details.text()
    assert panel.report.checks[row].scope in panel.details.text()
    assert panel.table.item(row, 3).text() == "-2 mm"
    assert panel.table.item(row, 5).text() == panel.report.checks[row].scope
    panel.resize(1200, 620)
    application.processEvents()
    assert not panel.table.isColumnHidden(3) and not panel.table.isColumnHidden(5)


def test_compact_history_keeps_dates_compatibility_and_error_visible_with_all_details(panels, application, native_font):
    project = system_assembly().channels[0].project
    record = _run(project)
    record.metadata["summary"].update(scope="continuous_sheet", accepted=False, validated_relative_field_error=0.043)
    entry = _entry("A deliberately descriptive run name " * 3, record)
    panel = panels(HistoryPanel)
    panel.setFont(native_font)
    panel.set_records((entry,), project=project)
    panel.resize(778, 620)
    panel.show()
    for _ in range(4):
        application.processEvents()
    assert panel.runs_table.horizontalScrollBar().maximum() == 0
    assert all(not panel.runs_table.isColumnHidden(column) for column in (0, 3, 5, 7, 8))
    assert all(panel.runs_table.isColumnHidden(column) for column in (1, 2, 4, 6))
    panel.select_record(entry.id)
    assert "saved scope: continuous_sheet" in panel.selected_summary.text()
    panel.run_details_button.click()
    assert panel.run_details_window.isVisible()
    assert panel.run_details_rows[2].text() == "continuous_sheet"
    assert panel.run_details_rows[3].text() == "Failed"
    assert panel.run_details_rows[5].text() == "4.3"
    assert panel.run_details_rows[0].text() == entry.label
    assert panel.run_details_rows[4].text() == str(project.revision)
    assert "UTC:" in panel.run_details_rows[7].toolTip()


def test_channel_snapshot_refresh_preserves_selection_and_refreshes_physical_inputs(panels):
    assembly = system_assembly()
    channel = assembly.channels[1]
    old_record = _run(channel.project)
    panel = panels(ChannelOverview, assembly, channel.id, records=(old_record,))
    panel.configuration_button.click()
    initial_configuration = panel._configuration_windows[channel.id]
    panel.configuration_button.click()
    assert panel._configuration_windows[channel.id] is initial_configuration
    assert "applied revision" in initial_configuration.windowTitle()
    states = [
        {
            "project_id": channel.project.id,
            "project_fingerprint": channel.project.fingerprint,
            "status": "running",
            "project_revision": channel.project.revision,
        }
    ]
    panel.set_snapshot(assembly, states=states, records=(old_record,))
    assert panel.table.currentRow() == 1
    assert initial_configuration.isVisible()  # Job-only updates do not invalidate applied-input snapshots.
    assert panel.table.item(1, 6).text().startswith("Running")
    modified = channel.project.update_config(
        {"target_gradient_strength": 0.031, "conductor_cross_section_width": 0.003}
    )
    changed = assembly.update_channel(channel.id, project=modified, enabled=False)
    before = changed.to_dict()
    panel.set_snapshot(changed, records=(old_record,))
    assert panel.table.currentRow() == 1
    assert panel.assembly is changed
    assert initial_configuration.isHidden()
    assert channel.id not in panel._configuration_windows
    assert "31 mT/m" in panel.table.item(1, 2).text()
    assert panel.table.item(1, 4).text() == "3 × 2 mm"
    assert panel.table.item(1, 1).text() == "Disabled"
    assert "8 enabled" in panel.heading.text()
    assert panel.table.item(1, 7).text().startswith("Earlier")
    panel.configuration_button.click()
    assert panel._configuration_windows[channel.id] is not initial_configuration
    assert f"applied revision {modified.revision}" in panel._configuration_windows[channel.id].windowTitle()
    assert changed.to_dict() == before


def test_channel_snapshot_removal_and_empty_assembly_disable_actions(panels):
    from coilgen.application.assembly import ProjectAssembly

    assembly = system_assembly()
    panel = panels(ChannelOverview, assembly, assembly.channels[0].id)
    panel.configuration_button.click()
    configuration = panel._configuration_windows[assembly.channels[0].id]
    panel.set_snapshot(ProjectAssembly.new())
    assert panel.table.rowCount() == 0
    assert not panel.compute_button.isEnabled() and not panel.open_button.isEnabled()
    assert not panel.configuration_button.isEnabled()
    assert configuration.isHidden()
    assert not panel._configuration_windows
