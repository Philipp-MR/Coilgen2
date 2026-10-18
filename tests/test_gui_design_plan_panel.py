"""The vertical plan preserves evidence semantics and passive inspection."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont, QFontDatabase, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.colors import tone_color
from coilgen.gui.design_plan import build_design_plan
from coilgen.gui.design_plan_panel import DesignPlanPanel


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    widget = DesignPlanPanel()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _project(name="Gx"):
    return ProjectDocument.from_config(preset_config(), name=name)


def _assembly(*names):
    return ProjectAssembly.new("Research gradient system", tuple(_project(name) for name in names))


def _state(channel, **values):
    return {
        "id": "job-" + channel.name.lower(),
        "project_id": channel.project.id,
        "project_fingerprint": channel.project.fingerprint,
        "project_revision": channel.project.revision,
        "pipeline_stage": "evaluated",
        **values,
    }


def _show(panel, app, width=320, height=720):
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            panel.setFont(QFont(families[0], 9))
    panel.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    panel.resize(width, height)
    panel.show()
    app.processEvents()


def test_single_channel_initial_stages_are_vertical_expanded_grey_and_passive(panel, app):
    project = _project()
    panel.set_plan(build_design_plan(project, requested_stage="evaluated"))
    _show(panel, app)
    channel = panel.channel_items[project.id]
    assert channel.isExpanded() and channel.childCount() == 5
    assert panel.tree.topLevelItemCount() == 1
    assert panel.editing_label.text().startswith("Editing: Gx")
    assert panel.compute_channels_button.isHidden()
    muted = panel.tree.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
    for row in panel.stage_items.values():
        assert row.text(1) == "Pending"
        assert row.foreground(0).color() == muted
        assert not row.font(0).bold()
    actions = []
    panel.channel_requested.connect(lambda value: actions.append(("channel", value)))
    panel.compute_channels_requested.connect(lambda: actions.append(("compute", None)))
    panel.tree.itemActivated.emit(channel.child(0), 0)
    assert actions == []
    panel.tree.itemActivated.emit(channel, 0)
    assert actions == [("channel", project.id)]


def test_completed_current_and_reusable_rows_remain_visually_distinct(panel):
    assembly = _assembly("Gx")
    channel = assembly.channels[0]
    reuse = SimpleNamespace(available=(SimpleNamespace(stage="contours"),), stale=())
    plan = build_design_plan(
        assembly,
        states=(
            _state(channel, status="running", progress=0.45, completed_stages=("prepared",), active_stage="sheet"),
        ),
        reuse_plans={channel.id: reuse},
    )
    panel.set_plan(plan)
    completed = panel.stage_items[(channel.id, "prepared")]
    current = panel.stage_items[(channel.id, "sheet")]
    cached = panel.stage_items[(channel.id, "contours")]
    assert completed.text(1) == "Completed"
    assert completed.foreground(0).color() == panel.tree.palette().color(QPalette.ColorRole.WindowText)
    assert "engineering acceptance is separate" in completed.toolTip(0)
    assert current.foreground(0).color() == tone_color("info", panel.tree.palette())
    assert current.font(0).bold()
    assert cached.text(1) == "Cached" and not cached.font(0).bold()
    assert "rechecks it before reuse" in cached.toolTip(0)
    assert panel.progress_bar.maximum() == 100 and panel.progress_bar.value() == 45


def test_multichannel_reports_real_worker_without_averaging_and_excludes_disabled_channel(panel):
    assembly = _assembly("Gx", "Gy", "Gz")
    gx, gy, gz = assembly.channels
    assembly = assembly.update_channel(gz.id, enabled=False)
    plan = build_design_plan(
        assembly,
        active_channel_id=gz.id,
        states=(
            _state(gx, status="queued", progress=0),
            _state(gy, status="running", progress=0.67, active_stage="sheet", stage="Solving current sheet"),
        ),
    )
    panel.set_plan(plan)
    panel.set_following(False)
    assert panel.tree.topLevelItemCount() == 3
    assert not panel.compute_channels_button.isHidden() and panel.compute_channels_button.isEnabled()
    assert panel.channel_items[gz.id].text(1) == "Excluded"
    assert "0/10 steps recorded" in panel.summary_label.text()
    assert "1 running" in panel.summary_label.text() and "1 queued" in panel.summary_label.text()
    assert "%" not in panel.summary_label.text()
    assert "Worker: Gy" in panel.activity_label.text()
    assert panel.worker_channel.channel_id == gy.id
    assert panel.progress_bar.value() == 67
    assert not panel.follow_button.isChecked() and panel.follow_button.isEnabled()


def test_running_worker_takes_activity_priority_over_selected_queued_channel(panel):
    assembly = _assembly("Gx", "Gy")
    gx, gy = assembly.channels
    plan = build_design_plan(
        assembly,
        active_channel_id=gx.id,
        states=(_state(gx, status="queued"), _state(gy, status="running", progress=0.35)),
    )
    panel.set_plan(plan)
    assert panel.worker_channel.channel_id == gy.id
    assert panel.progress_bar.value() == 35
    assert panel.channel_items[gx.id].text(1) == "Queued"


def test_updates_preserve_collapsed_channels_row_identity_selection_and_scroll(panel, app):
    assembly = _assembly(*(f"Research {index}" for index in range(12)))
    panel.set_plan(build_design_plan(assembly))
    _show(panel, app, height=540)
    first = assembly.channels[0]
    header = panel.channel_items[first.id]
    row = panel.stage_items[(first.id, "sheet")]
    header.setExpanded(False)
    panel.tree.setCurrentItem(panel.channel_items[assembly.channels[3].id])
    panel.tree.verticalScrollBar().setValue(100)
    scroll = panel.tree.verticalScrollBar().value()
    selected = panel.tree.currentItem()
    panel.set_plan(build_design_plan(assembly, states=(_state(first, status="running", progress=0.1),)))
    assert panel.channel_items[first.id] is header
    assert panel.stage_items[(first.id, "sheet")] is row
    assert not header.isExpanded()
    assert panel.tree.currentItem() is selected
    assert panel.tree.verticalScrollBar().value() == scroll
    shortened = replace(assembly, channels=assembly.channels[1:])
    panel.set_plan(build_design_plan(shortened))
    assert first.id not in panel.channel_items
    assert not any(identity[0] == first.id for identity in panel.stage_items)


def test_explicit_actions_and_follow_requests_are_not_emitted_by_refresh(panel, app):
    assembly = _assembly("Gx", "Gy")
    gx = assembly.channels[0]
    plan = build_design_plan(assembly, states=(_state(gx, status="running", progress=0.25),))
    events = []
    panel.action_requested.connect(lambda value: events.append(("action", value)))
    panel.compute_channels_requested.connect(lambda: events.append(("compute", None)))
    panel.job_requested.connect(lambda value: events.append(("job", value)))
    panel.follow_requested.connect(lambda value: events.append(("follow", value)))
    panel.set_plan(plan)
    panel.set_state("evaluated", {"status": "running"}, following=False)
    panel.set_following(True, job_id="different-job")
    assert not panel.follow_button.isChecked()
    panel.set_following(False, job_id="job-gx")
    assert events == []
    _show(panel, app)
    QTest.mouseClick(panel.action_buttons["plan"], Qt.MouseButton.LeftButton)
    QTest.mouseClick(panel.action_buttons["jobs"], Qt.MouseButton.LeftButton)
    QTest.mouseClick(panel.job_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(panel.compute_channels_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(panel.follow_button, Qt.MouseButton.LeftButton)
    assert events == [("action", "plan"), ("action", "jobs"), ("job", "job-gx"), ("compute", None), ("follow", True)]


def test_runtime_unavailable_hides_current_percentage_and_keeps_disabled_follow(panel):
    assembly = _assembly("Gx", "Gy")
    gx = assembly.channels[0]
    plan = build_design_plan(
        assembly,
        states=(_state(gx, status="running", progress=0.8, completed_stages=("prepared",)),),
        runtime_unavailable="Cannot refresh worker status",
    )
    panel.set_plan(plan)
    panel.set_following(True, job_id="job-gx")
    assert panel.progress_bar.isHidden() and panel.progress_bar.maximum() == 100
    assert not panel.follow_button.isHidden() and not panel.follow_button.isEnabled()
    assert "unavailable" in panel.follow_button.toolTip()
    assert "Runtime unavailable" in panel.activity_label.text()
    assert panel.stage_items[(gx.id, "prepared")].text(1) == "Completed"
    assert panel.stage_items[(gx.id, "sheet")].text(1) == "Pending"
    assert not panel.compute_channels_button.isEnabled()
    events = []
    panel.follow_requested.connect(events.append)
    recovered = build_design_plan(assembly, states=(_state(gx, status="running", progress=0.82),))
    panel.set_plan(recovered)
    panel.set_following(True, job_id="job-gx")
    assert not panel.progress_bar.isHidden() and panel.progress_bar.value() == 82
    assert panel.follow_button.isEnabled() and panel.follow_button.isChecked()
    assert panel.compute_channels_button.isEnabled()
    assert "Runtime unavailable" not in panel.activity_label.text()
    assert events == []


def test_verifying_results_remain_distinct_from_finished_computation(panel):
    assembly = _assembly("Gx")
    gx = assembly.channels[0]
    plan = build_design_plan(assembly, states=(_state(gx, status="completed", progress=1),))
    panel.set_plan(plan)
    assert panel.channel_items[gx.id].text(1) == "Verifying"
    assert panel.progress_bar.maximum() == 0
    assert not panel.progress_bar.isHidden()
    assert not any(row.text(1) == "Completed" for row in panel.stage_items.values())
    assert "verifying saved evidence" in panel.activity_label.text()


def test_short_study_stale_failure_and_cancelled_states_use_explicit_text(panel):
    assembly = _assembly("Gx")
    gx = assembly.channels[0]
    short = build_design_plan(assembly, requested_stage="sheet")
    panel.set_plan(short)
    assert panel.stage_items[(gx.id, "contours")].text(1) == "Outside plan"
    stale = replace(
        short,
        channels=(
            replace(
                short.channels[0],
                status="stale",
                stages=(replace(short.channels[0].stages[0], status="stale"), *short.channels[0].stages[1:]),
            ),
        ),
    )
    panel.set_plan(stale)
    row = panel.stage_items[(gx.id, "prepared")]
    assert row.text(1) == "Inputs changed"
    assert row.foreground(1).color() == tone_color("warning", panel.tree.palette())
    for status, expected, tone in (("failed", "Failed", "error"), ("cancelled", "Stopped", "warning")):
        plan = build_design_plan(
            assembly,
            states=(_state(gx, status=status, completed_stages=("prepared",), active_stage="sheet"),),
        )
        panel.set_plan(plan)
        assert panel.stage_items[(gx.id, "prepared")].text(1) == "Completed"
        row = panel.stage_items[(gx.id, "sheet")]
        assert row.text(1) == expected
        assert row.foreground(0).color() == tone_color(tone, panel.tree.palette())
        assert panel.progress_bar.isHidden()


def test_known_work_and_indeterminate_activity_do_not_guess_completion(panel):
    assembly = _assembly("Gx")
    gx = assembly.channels[0]
    for progress, maximum in ((None, 0), (0.43, 100)):
        panel.set_plan(build_design_plan(assembly, states=(_state(gx, status="running", progress=progress),)))
        assert panel.progress_bar.maximum() == maximum
        assert not panel.progress_bar.isHidden()
        assert not any(row.text(1) == "Completed" for row in panel.stage_items.values())
    assert panel.progress_bar.value() == 43
    panel.set_plan(
        build_design_plan(assembly, states=(_state(gx, status="running", progress=0.43, progress_indeterminate=True),))
    )
    assert panel.progress_bar.maximum() == 0
    assert panel.progress_bar.format() == "Working"


@pytest.mark.parametrize("width", (280, 320))
def test_narrow_native_font_layout_and_long_names_fit_without_horizontal_overflow(panel, app, width):
    assembly = _assembly("An unusual research gradient coil with a deliberately long descriptive channel name", "Gy")
    channel = assembly.channels[0]
    plan = build_design_plan(assembly, states=(_state(channel, status="running", progress=0.32),))
    panel.set_plan(plan)
    _show(panel, app, width=width, height=740)
    assert panel.minimumSizeHint().width() <= width
    assert panel.tree.header().length() <= panel.tree.viewport().width()
    for button in (
        *panel.action_buttons.values(),
        panel.follow_button,
        panel.job_button,
        panel.compute_channels_button,
    ):
        assert button.geometry().right() < width
    assert channel.name in panel.channel_items[channel.id].toolTip(0)
    assert not panel.grab().isNull()


def test_legacy_snapshot_is_retained_but_does_not_override_multichannel_rows(panel):
    assembly = _assembly("Gx", "Gy")
    gy = assembly.channels[1]
    panel.set_plan(build_design_plan(assembly, states=(_state(gy, status="running", progress=0.62),)))
    panel.set_state("sheet", completed=("prepared",), following=False)
    assert panel.snapshot.stages[0].status == "complete"
    assert panel.snapshot.stages[2].status == "skipped"
    assert panel.progress_bar.value() == 62
    assert "Worker: Gy" in panel.activity_label.text()
    panel.set_plan(None)
    assert panel.tree.topLevelItemCount() == 0
    assert panel.activity_label.text() == ""
    assert panel.progress_bar.isHidden() and panel.worker_channel is None
