"""Run-bound interconnection access leaves the active model and desktop alone."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QDialog

from coilgen.application.routing_analysis import analyze_routes
from coilgen.gui.results_overview import ResultsOverviewPanel
from coilgen.gui.routing_analysis import RoutingAnalysisDialog
from test_routing_analysis import record
from test_workbench_usability import application as application, windows as windows


def test_result_board_exposes_geometry_without_field_samples(application, tmp_path):
    panel = ResultsOverviewPanel()
    saved = record(tmp_path)
    panel.set_record(saved)
    assert not panel.isHidden() and panel.routing_button.isEnabled()
    panel.set_analysis_available(False)
    assert panel.routing_button.isEnabled()
    panel.set_record(saved, source_error="Scientific array checksum changed")
    assert not panel.routing_button.isEnabled()
    panel.close()


def test_dialog_selects_each_support_and_physical_geometry_with_witness(application, tmp_path):
    analysis = analyze_routes(record(tmp_path, two_planes=True, frames=True))
    dialog = RoutingAnalysisDialog(analysis)
    assert dialog.support.count() == 2
    assert dialog.selected_source.key == "review_conductor_0"
    assert dialog.envelopes.isEnabled()
    assert dialog.problems.rowCount() == 1
    dialog.problems.setCurrentCell(0, 0)
    assert "Saved minimum-bend triplet" in dialog.witness.text()
    assert dialog.view.currentIndex() == 1 and dialog.radius.isEnabled()
    assert dialog.envelopes.isChecked()
    assert dialog.plot.figure.axes[0].get_xlabel() == "Local distance [mm]"
    dialog.radius.setValue(4.0)
    assert dialog.plot.figure.axes[0].get_xlim() == (-4.0, 4.0)
    dialog.view.setCurrentIndex(0)
    assert not dialog.radius.isEnabled()
    assert dialog.plot.figure.axes[0].get_xlabel() == "Developed distance [mm]"
    dialog.support.setCurrentIndex(1)
    assert dialog.selected_source.part_index == 1
    assert dialog.selected_source.key == "route_1"
    assert not dialog.envelopes.isEnabled()
    assert "Projection crossings are not physical collision evidence" in dialog.witness.text()
    dialog.close()


def test_routing_access_preserves_pending_model_draft_and_pins_saved_run(windows, tmp_path, monkeypatch):
    import coilgen.application.evidence as evidence
    import coilgen.gui.routing_analysis as gui

    class Inspector(QDialog):
        def __init__(self, analysis, parent=None):
            super().__init__(parent)
            self.analysis = analysis

    window = windows()
    original = window.project
    saved = record(tmp_path)
    window.runs["source"] = saved
    window._selected_run = "source"
    window._update_execution_feedback()
    assert window.routing_analysis_action.isEnabled()
    assert window.results_overview.routing_button.isEnabled()
    window._draft_controls["radius_mm"].setValue(174)
    requested = []
    monkeypatch.setattr(gui, "RoutingAnalysisDialog", Inspector)
    monkeypatch.setattr(evidence, "verify_record", lambda selected: selected)
    monkeypatch.setattr(
        window,
        "_gradient_task",
        lambda operation, callback, label, **options: requested.append((operation, callback, options)),
    )
    window._results_action("routing")
    operation, callback, options = requested.pop()
    assert options["retain_receipt"]
    window._selected_run = None
    callback(operation(original))
    inspector = next(iter(window._routing_analysis_windows.values()))
    assert inspector.analysis.run_directory == str(saved.directory)
    assert window.project is original and window._draft_dirty
    assert window._draft_controls["radius_mm"].value() == 174
    assert window.undo_stack.count() == 0
    inspector.close()


def test_routing_actions_disable_for_failed_source_integrity(windows, tmp_path):
    window = windows()
    saved = record(tmp_path)
    window.runs["source"] = saved
    window._selected_run = "source"
    window._update_execution_feedback()
    window._source_health.mark_failed(saved.directory, "Recorded artifact changed")
    window._update_execution_feedback()
    assert not window.routing_analysis_action.isEnabled()
    assert not window.results_overview.routing_button.isEnabled()
