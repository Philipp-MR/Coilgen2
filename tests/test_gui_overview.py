"""Guidance remains actionable, compact and readable without native interaction."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from coilgen.gui.colors import _contrast, apply_tone, item_foreground, tone_color
from coilgen.gui.overview import DesignOverview


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["coilgen-overview-test"])


def state(*, action="compute", tone="info"):
    stages = tuple(
        SimpleNamespace(key=key, label=label, status=status, detail=detail, tone=stage_tone)
        for key, label, status, detail, stage_tone in (
            ("target", "Target", "defined", "Absolute Bz in tesla; coordinates in metres.", "neutral"),
            ("support", "Support", "defined", "Open cylindrical support surface.", "neutral"),
            ("conductor", "Conductor", "defined", "Copper with 0.1 mm insulation.", "neutral"),
            ("solve", "Solve", "not_computed", "No saved sheet solution.", "neutral"),
            ("winding", "Winding", "not_computed", "Contour generation and connected routes requested.", "neutral"),
            ("check", "Check", "unverified", "Independent field and conductor checks have not run.", "neutral"),
            ("export", "Export", "not_run", "No computed conductor solid is available.", "neutral"),
        )
    )
    return SimpleNamespace(
        title="Ready to compute the current model",
        detail="Compute generates the sheet, contours, connected routes and conductor checks.",
        action_label="Compute full design",
        action_key=action,
        tone=tone,
        stages=stages,
    )


@pytest.fixture
def native_font(application):
    """Keep native glyph metrics available for the whole widget lifetime."""
    original_font = QFont(application.font())
    font_id = -1
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else ()
    font = QFont(families[0], 9) if families else original_font
    # Styled QToolButtons resolve their font from QApplication while they are
    # created. Setting only the parent font later leaves fallback glyph metrics
    # cached on Windows' offscreen platform after another test removes its font.
    application.setFont(font)
    application.processEvents()
    yield font
    application.setFont(original_font)
    application.processEvents()
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)


@pytest.fixture
def overview(application, native_font):
    widget = DesignOverview()
    widget.setFont(native_font)
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_guidance_is_collapsed_and_updates_without_emitting_actions(overview):
    actions, sections = [], []
    overview.action_requested.connect(actions.append)
    overview.section_requested.connect(sections.append)
    overview.set_state(state(), {"target": "Gy: 20 mT/m", "roi": "20 mm sphere", "support": "80 mm radius"})
    assert not overview.expand_button.isChecked() and overview.details.isHidden()
    assert overview.summary_labels["target"].text() == "Gy: 20 mT/m · 20 mm sphere"
    assert overview.summary_labels["support"].text() == "80 mm radius"
    assert overview.status_labels["check"].text().endswith("Unverified")
    assert not actions and not sections
    overview.set_state(state(action="geometry", tone="error"))
    overview.action_button.click()
    assert actions == ["geometry"]
    overview.expand_button.click()
    overview.stage_buttons["support"].click()
    assert sections == ["support"]
    assert overview.expand_button.isChecked() and not overview.details.isHidden()


def test_keyboard_can_expand_navigate_and_request_action(overview, application):
    actions, sections = [], []
    overview.action_requested.connect(actions.append)
    overview.section_requested.connect(sections.append)
    overview.set_state(state())
    overview.show()
    overview.expand_button.setFocus()
    QTest.keyClick(overview.expand_button, Qt.Key.Key_Space)
    application.processEvents()
    assert not overview.details.isHidden()
    overview.stage_buttons["conductor"].setFocus()
    QTest.keyClick(overview.stage_buttons["conductor"], Qt.Key.Key_Space)
    overview.action_button.setFocus()
    QTest.keyClick(overview.action_button, Qt.Key.Key_Space)
    assert sections == ["conductor"] and actions == ["compute"]
    assert all(button.focusPolicy() == Qt.FocusPolicy.StrongFocus for button in overview.stage_buttons.values())


def test_compact_strip_preserves_plot_space_and_actions(overview, application):
    overview.set_state(state())
    overview.resize(420, overview.sizeHint().height())
    overview.show()
    application.processEvents()
    assert overview.minimumSizeHint().width() <= 420
    assert overview.height() <= 130
    for control in (overview.title, overview.detail, overview.action_button):
        top_left = control.mapTo(overview, QPoint(0, 0))
        bottom_right = control.mapTo(overview, QPoint(control.width() - 1, control.height() - 1))
        assert overview.contentsRect().contains(top_left)
        assert overview.contentsRect().contains(bottom_right)
    overview.expand_button.click()
    overview.resize(420, overview.sizeHint().height())
    application.processEvents()
    assert overview.details.height() <= 260
    assert overview.details.verticalScrollBar().maximum() > 0


def test_disabled_action_and_empty_overview_do_not_emit_or_leave_blank_panel(overview):
    actions = []
    overview.action_requested.connect(actions.append)
    current = state(action="")
    overview.set_state(current)
    overview.action_button.click()
    assert not actions and not overview.action_button.isEnabled()
    overview.expand_button.click()
    current.stages = ()
    overview.set_state(current)
    assert overview.details.isHidden() and overview.expand_button.isHidden()


@pytest.mark.parametrize("background,text", [("#ffffff", "#222222"), ("#202020", "#eeeeee")])
def test_semantic_colors_keep_readable_theme_contrast(background, text):
    palette = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
        palette.setColor(role, QColor(background))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
    for tone in ("info", "success", "warning", "error"):
        color = tone_color(tone, palette)
        assert _contrast(color, QColor(background)) >= 4.5
        assert item_foreground(tone, palette).color() == color


def test_labels_keep_text_and_reset_to_neutral_after_error(application):
    parent = QWidget()
    palette = parent.palette()
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#123456"))
    parent.setPalette(palette)
    label = QLabel("Missing support mesh", parent)
    apply_tone(label, "error")
    assert label.text() == "Missing support mesh" and label.property("semanticTone") == "error"
    apply_tone(label, "neutral")
    assert label.palette().color(QPalette.ColorRole.WindowText) == QColor("#123456")
    parent.deleteLater()


def test_scoped_status_retains_supplied_failure_and_plain_summary_text(overview):
    current = state()
    current.stages[-2].status = "failed"
    current.stages[-2].tone = "error"
    current.stages[-2].detail = "Independent sheet passes; conductor clearance fails."
    overview.set_state(current, {"target": "Bz < 100 µT & external map"})
    assert overview.status_labels["check"].text().endswith("Failed")
    assert overview.status_labels["check"].property("semanticTone") == "error"
    assert "clearance fails" in overview.summary_labels["check"].text()
    assert overview.summary_labels["target"].textFormat() == Qt.TextFormat.PlainText
    assert overview.summary_labels["target"].text() == "Bz < 100 µT & external map"


def test_actual_guidance_state_contract(overview):
    from coilgen.application import ProjectDocument
    from coilgen.application.presets import preset_config
    from coilgen.gui.guidance import assess_guidance, resolved_model_summary

    project = ProjectDocument.from_config(replace(preset_config(), sf_opt_method="tikhonov", research_physics=None))
    current = assess_guidance(project)
    overview.set_state(current, resolved_model_summary(project))
    assert tuple(overview.stage_buttons) == tuple(stage.key for stage in current.stages)
    assert overview.action_button.text() == current.action_label


def test_actual_guidance_details_use_full_strip_width(overview, application):
    from coilgen.application import ProjectDocument
    from coilgen.application.presets import preset_config
    from coilgen.gui.guidance import assess_guidance, resolved_model_summary

    project = ProjectDocument.from_config(replace(preset_config(), sf_opt_method="tikhonov", research_physics=None))
    overview.set_state(assess_guidance(project), resolved_model_summary(project))
    overview.resize(420, overview.heightForWidth(420))
    application.processEvents()
    assert overview.height() <= 150
    assert overview.detail.width() > overview.title.width()
    assert overview.detail.width() > 380
    assert overview.detail.height() >= overview.detail.heightForWidth(overview.detail.width())


def test_theme_change_updates_existing_guidance_status(overview, application):
    current = state(tone="warning")
    current.stages[0].tone = "info"
    current.stages[-2].tone = "success"
    current.stages[-2].status = "passed"
    overview.set_state(current)
    palette = overview.palette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
        palette.setColor(role, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
    overview.setPalette(palette)
    application.processEvents()
    foreground = overview.title.palette().color(QPalette.ColorRole.WindowText)
    assert _contrast(foreground, QColor("#202020")) >= 4.5
    assert foreground == tone_color("warning", palette)
    for key, tone in (("target", "info"), ("check", "success"), ("export", "neutral")):
        foreground = overview.status_labels[key].palette().color(QPalette.ColorRole.WindowText)
        assert foreground == tone_color(tone, palette)
        assert _contrast(foreground, QColor("#202020")) >= 4.5


def test_export_navigation_retains_state_and_readable_colors_after_theme_change(application, native_font):
    from coilgen.gui.workflow import WorkflowNavigation

    navigation = WorkflowNavigation()
    navigation.setFont(native_font)
    try:
        navigation.set_current("export")
        assert "solid" in navigation.buttons["export"].toolTip()
        assert "units" in navigation.buttons["export"].toolTip()
        before = navigation.styleSheet()
        palette = navigation.palette()
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
            palette.setColor(role, QColor("#202020"))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
            palette.setColor(role, QColor("#eeeeee"))
        navigation.setPalette(palette)
        application.processEvents()
        accent = tone_color("info", palette)
        assert accent.name() in navigation.styleSheet() and navigation.styleSheet() != before
        assert "#202020" in navigation.styleSheet()
        assert _contrast(accent, QColor("#202020")) >= 4.5
        assert navigation.current_step == "export" and navigation.buttons["export"].isChecked()
        navigation.resize(560, 40)
        navigation.show()
        application.processEvents()
        assert navigation.minimumSizeHint().width() <= 560
        assert all(navigation.rect().contains(button.geometry()) for button in navigation.buttons.values())
    finally:
        navigation.close()
        navigation.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _panel_palette(panel, background, text):
    palette = panel.palette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(role, QColor(background))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(role, QColor(text))
    panel.setPalette(palette)
    return palette


def test_recorded_problem_and_readiness_colors_follow_theme_without_text_changes(application, monkeypatch):
    from coilgen.application.diagnostics import ProblemIssue
    from coilgen.gui import problems
    from coilgen.gui.problems import ProblemsPanel
    from coilgen.gui.readiness import ReadinessIssue, ReadinessPanel

    issues = (
        ProblemIssue("clearance", "Clearance failure", "Measured clearance is below the recorded limit."),
        ProblemIssue(
            "unverified", "Unverified field scope", "No independent wire test was recorded.", severity="warning"
        ),
    )
    monkeypatch.setattr(problems, "run_problems", lambda _record: issues)
    panel, readiness = ProblemsPanel(), ReadinessPanel()
    try:
        panel.set_run(SimpleNamespace(metadata={}))
        readiness.set_issues((ReadinessIssue("Missing support mesh", "geometry", "coil_mesh_file", "Import a mesh."),))
        for background, text in (("#ffffff", "#222222"), ("#202020", "#eeeeee")):
            _panel_palette(panel, background, text)
            _panel_palette(readiness, background, text)
            application.processEvents()
            for row, issue in enumerate(issues):
                color = panel.table.item(row, 0).foreground().color()
                assert color == tone_color(issue.severity, panel.table.palette())
                assert _contrast(color, QColor(background)) >= 4.5
            assert "Clearance failure" in panel.table.item(0, 0).text()
            assert readiness.heading.property("semanticTone") == "error"
            assert "Cannot compute" in readiness.heading.text()
            assert (
                _contrast(readiness.heading.palette().color(QPalette.ColorRole.WindowText), QColor(background)) >= 4.5
            )
            assert "Missing support mesh" in readiness._issue_labels[0].text()
        readiness.set_issues(
            (ReadinessIssue("Mesh is coarse", "geometry", "mesh", "Refine if needed.", severity="warning"),)
        )
        assert readiness.heading.property("semanticTone") == "warning"
        assert "Compute is available" in readiness.heading.text()
        panel.set_run(None)
        assert panel.summary.property("semanticTone") == "neutral"
    finally:
        for widget in (panel, readiness):
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_history_colors_distinguish_saved_completion_from_scoped_pass(application, tmp_path):
    from coilgen.application.history import RunHistoryStore
    from coilgen.gui.history import HistoryPanel
    from test_run_history import write_run

    store = RunHistoryStore(tmp_path / "history.json")
    for index, summary in enumerate(
        (
            {"scope": "continuous_sheet", "accepted": True},
            {"scope": "fabrication_review", "accepted": False},
            {"scope": "prepared", "accepted": False},
        )
    ):
        store.add_run("p", write_run(tmp_path / f"run{index}", summary=summary))
    panel = HistoryPanel()
    try:
        panel.set_records(store.restore("p"))
        for background, text in (("#ffffff", "#222222"), ("#202020", "#eeeeee")):
            _panel_palette(panel, background, text)
            application.processEvents()
            for row, verdict, tone in ((0, "Passed", "success"), (1, "Failed", "error"), (2, "Unverified", "neutral")):
                item = panel.runs_table.item(row, 3)
                assert item.text() == verdict
                assert item.foreground().color() == tone_color(tone, panel.runs_table.palette())
                assert _contrast(item.foreground().color(), QColor(background)) >= 4.5
                completion = panel.runs_table.item(row, 1)
                assert completion.foreground().color() == tone_color("info", panel.runs_table.palette())
                assert completion.foreground().color() != tone_color("success", panel.runs_table.palette())
            assert "does not release" in panel.runs_table.item(0, 3).toolTip()
    finally:
        panel.close()
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_export_theme_repaint_preserves_failed_scope_and_selection(application):
    from coilgen.gui.export_panel import GeometryExportPanel
    from test_gui_export_panel import _options, _record

    panel = GeometryExportPanel(_record(ready=False), _options(status="failed", inspection=True))
    try:
        original = panel.status.text()
        selected = panel.selected_names()
        assert panel.include_project.text() == "Include project JSON and results"
        assert "self-contained" in panel.include_project.toolTip()
        for background, text in (("#ffffff", "#222222"), ("#202020", "#eeeeee")):
            _panel_palette(panel, background, text)
            application.processEvents()
            foreground = panel.table.item(0, 2).foreground().color()
            assert foreground == tone_color("error", panel.table.palette())
            assert _contrast(foreground, QColor(background)) >= 4.5
            assert panel.status.property("semanticTone") == "error"
            assert _contrast(panel.status.palette().color(QPalette.ColorRole.WindowText), QColor(background)) >= 4.5
            assert panel.status.text() == original and panel.selected_names() == selected
            assert panel.export_button.isEnabled()
    finally:
        panel.close()
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
