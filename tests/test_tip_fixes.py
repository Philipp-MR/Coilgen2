"""Terminal handling control and one-click fixes for review tips, in both clients."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from coilgen.application import ProjectDocument
from coilgen.application.presets import EXAMPLE_TITLES, preset_config, preset_study
from coilgen.application.session import TERMINAL_STRATEGIES, WorkbenchSession
from coilgen.application.submitted_design import choice_title
from coilgen.config import CoilGenConfig
from coilgen.config_validation import ConfigValidationMessage, validate_config


def _issues(session):
    return session.preflight(session.project.study["pipeline_stage"]).issues


def _terminal_tips(issues):
    return [issue for issue in issues if issue.key == "layout_terminal_strategy"]


def test_message_fix_is_optional_and_does_not_change_equality():
    plain = ConfigValidationMessage("warning", "levels", "Text", "Hint")
    fixed = ConfigValidationMessage("warning", "levels", "Text", "Hint", fix={"levels": 4}, fix_label="Use 4")
    assert plain.fix is None and plain.fix_label == ""
    assert plain == fixed and hash(plain) == hash(fixed)
    assert fixed.format() == plain.format() == "[WARNING] levels: Text Suggestion: Hint"


def test_terminal_tips_offer_a_valid_fix_and_mounting_tip_offers_none():
    config = replace(preset_config(), skip_postprocessing=False, skip_sweep=False, contour_only_sweep_mode=False)
    report = validate_config(config)
    terminal = [message for message in report.warnings if message.field == "layout_terminal_strategy"]
    assert len(terminal) == 2
    for message in terminal:
        assert message.fix == {"layout_terminal_strategy": "waived"}
        assert message.fix_label and "wire_stubs" not in message.suggestion
        assert "Terminal handling" in message.suggestion
    fixed = replace(config, **terminal[0].fix)
    assert not [m for m in validate_config(fixed).messages if m.field == "layout_terminal_strategy"]
    mounting = next(message for message in report.warnings if message.field == "conductor_mounting_mode")
    assert config.conductor_mounting_mode == "surface_outer"
    assert mounting.fix is None


def test_terminal_strategy_values_match_the_engine():
    assert set(TERMINAL_STRATEGIES) == {"undefined", "waived", "pads", "wire_stubs"}
    for value in TERMINAL_STRATEGIES:
        report = validate_config(CoilGenConfig(layout_terminal_strategy=value))
        assert not [m for m in report.errors if m.field == "layout_terminal_strategy"]
    assert [choice_title("layout_terminal_strategy", value) for value in TERMINAL_STRATEGIES] == [
        "Not decided",
        "Terminal pads",
        "Wire stubs",
        "Handled outside CoilGen",
    ]


def test_session_applies_fix_as_one_undoable_revision_and_tips_disappear(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    tips = _terminal_tips(_issues(session))
    assert len(tips) == 2 and all(tip.fix == {"layout_terminal_strategy": "waived"} for tip in tips)
    assert all(tip.node == "winding" and not tip.blocking for tip in tips)
    mounting = next(issue for issue in _issues(session) if issue.key == "conductor_mounting_mode")
    assert mounting.fix is None
    revision = session.project.revision
    session.apply_fix("layout_terminal_strategy")
    assert session.project.config.layout_terminal_strategy == "waived"
    assert session.project.revision == revision + 1
    assert not _terminal_tips(_issues(session))
    assert any(issue.key == "conductor_mounting_mode" for issue in _issues(session))
    with pytest.raises(ValueError):
        session.apply_fix("layout_terminal_strategy")
    with pytest.raises(ValueError):
        session.apply_fix("conductor_mounting_mode")
    session.undo()
    assert session.project.config.layout_terminal_strategy == "undefined"
    assert len(_terminal_tips(_issues(session))) == 2


def test_session_winding_section_has_terminal_handling(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    field = next(item for item in session.settings_fields("winding") if item.key == "layout_terminal_strategy")
    assert field.label == "Terminal handling" and field.kind == "choice"
    assert field.choices == TERMINAL_STRATEGIES
    assert session.settings_values("winding")["layout_terminal_strategy"] == "undefined"
    session.apply_settings({"layout_terminal_strategy": "pads"})
    assert session.project.config.layout_terminal_strategy == "pads"
    assert not _terminal_tips(_issues(session))


def test_browser_shows_apply_button_and_applies_fix(tmp_path):
    pytest.importorskip("trame")
    pytest.importorskip("aiohttp")
    from coilgen.web.app import create_app

    server, browser = create_app(tmp_path, user="tester", server_name="test-tip-fixes")
    state = server.state
    browser.load_template("cylinder")
    assert "cg-apply-fix" in state.trame__template_main
    terminal = [item for item in state.preflight if item["fix_key"] == "layout_terminal_strategy"]
    assert len(terminal) == 2 and all(item["node"] == "winding" for item in terminal)
    assert terminal[0]["fix_label"] == "Mark as handled outside CoilGen"
    mounting = next(item for item in state.preflight if item["node"] == "conductor")
    assert mounting["fix_key"] == ""
    state.node = "winding"
    browser._refresh_settings()
    entry = next(field for field in state.settings_fields if field["key"] == "layout_terminal_strategy")
    assert entry["label"] == "Terminal handling"
    assert {"title": "Handled outside CoilGen", "value": "waived"} in entry["choices"]
    browser.apply_fix("layout_terminal_strategy")
    assert browser.session.project.config.layout_terminal_strategy == "waived"
    assert not [item for item in state.preflight if item["fix_key"]]
    assert state.settings_values["layout_terminal_strategy"] == "waived"
    browser.undo()
    assert browser.session.project.config.layout_terminal_strategy == "undefined"
    browser.apply_fix("levels")  # no such fix: reported, nothing changed
    assert state.message_color == "error"
    assert browser.session.project.config.layout_terminal_strategy == "undefined"


@pytest.fixture
def qt_window(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    import test_native_workbench as native
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(["coilgen-gui-test"])
    from coilgen.gui.workbench import CoilGenWorkbench

    project = ProjectDocument.from_config(preset_config(), name=EXAMPLE_TITLES["cylinder"]).update_study(preset_study())
    window = CoilGenWorkbench(
        project,
        jobs=native.FakeJobs(tmp_path / "jobs"),
        service=native.FakeService(),
        viewport_factory=native.FakeViewport,
    )
    window.timer.stop()
    yield window
    window.undo_stack.setClean()
    window.close()
    window.undo_stack.clear()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_qt_modeling_note_apply_button_is_undoable(qt_window):
    from PySide6.QtWidgets import QPushButton

    window = qt_window
    window._refresh_guidance()
    panel = window.modeling_notes_panel
    assert len(_terminal_tips(panel.issues)) == 2
    buttons = panel.findChildren(QPushButton, "apply_issue_fix")
    assert len(buttons) == 2
    assert buttons[0].text() == "Mark as handled outside CoilGen"
    buttons[0].click()
    assert window.project.config.layout_terminal_strategy == "waived"
    window._refresh_guidance()
    assert not _terminal_tips(panel.issues)
    assert any(issue.key == "conductor_mounting_mode" for issue in panel.issues)
    window.undo_stack.undo()
    assert window.project.config.layout_terminal_strategy == "undefined"


def test_qt_winding_settings_edit_terminal_handling(qt_window):
    window = qt_window
    window.select_node("winding")
    control = window._draft_controls["layout_terminal_strategy"]
    assert control.currentText() == "undefined"
    control.setCurrentText("wire_stubs")
    assert window.apply_settings()
    assert window.project.config.layout_terminal_strategy == "wire_stubs"


def test_qt_fix_commits_pending_draft_first(qt_window):
    window = qt_window
    window.select_node("winding")
    window._draft_controls["levels"].setValue(13)
    window._refresh_guidance()
    issue = _terminal_tips(window.modeling_notes_panel.issues)[0]
    assert window._apply_issue_fix(issue)
    assert window.project.config.levels == 13
    assert window.project.config.layout_terminal_strategy == "waived"


def test_expert_settings_offers_terminal_handling():
    pytest.importorskip("PySide6")
    from coilgen.gui.expert_settings import FIELDS

    field = next(item for item in FIELDS if item.key == "layout_terminal_strategy")
    assert field.kind == "enum" and set(field.choices) == set(TERMINAL_STRATEGIES)
