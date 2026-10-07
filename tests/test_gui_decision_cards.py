"""Decision cards and popup navigation keep one authoritative scientific draft."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QVBoxLayout, QWidget

from coilgen.gui.controls import ScientificSpinBox
from coilgen.gui.colors import apply_tone, tone_color
from coilgen.gui.choice_dialogs import GraphicalChoiceCombo
from coilgen.gui.region_selectors import TargetRegionGallery
from coilgen.gui.focused_editors import FocusedFormSection, control_form, group_form_controls, open_focused_control


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["decision-card-test"])


@pytest.fixture
def decision(application):
    page = QWidget()
    page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    form = QFormLayout(page)
    control = ScientificSpinBox()
    control.setDecimals(15)
    control.setRange(-1e6, 1e6)
    control.setKeyboardTracking(False)
    control.setValue(1.234567891234567)
    wrapper = QWidget()
    body = QVBoxLayout(wrapper)
    body.addWidget(control)
    error = QLabel("Correct this coordinate", wrapper)
    error.hide()
    body.addWidget(error)
    label = QLabel("Center x", page)
    label.setBuddy(control)
    form.addRow(label, wrapper)
    window = SimpleNamespace(_draft_controls={"center": control}, _focused_sections=[])
    section = group_form_controls(window, form, "Placement", ["center"], summary=lambda: control.text())
    section.dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window._focused_sections.append(section)
    yield page, form, control, error, section, window
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_card_is_opt_in_and_keeps_the_same_controls(decision):
    _page, _form, control, error, section, _window = decision
    value = control.value()
    assert section.button.text() == "Placement…"
    assert not section.property("decisionCard")
    section.set_step(2, 4)
    assert section.button.text() == "Edit…"
    assert section.step_label.text() == "2"
    assert section.title_label.text() == "Placement"
    assert section.accessibleName() == "Decision 2 of 4: Placement"
    assert section.button.accessibleName() == "Edit placement"
    assert section.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
    assert not section.icon_label.pixmap().isNull()
    assert section.controls == (control,)
    assert error.parentWidget() is control.parentWidget()
    assert control.value() == value
    assert not section.dialog.isModal()
    # Updating a sequence number must not rebuild or clone its row.
    section.set_step(3, 5)
    assert section.step_label.text() == "3" and section.form.rowCount() == 1
    assert control.value() == value


@pytest.mark.parametrize("number,total", [(0, None), (-1, None), (True, None), (1.5, None), (2, 1), (1, True)])
def test_card_rejects_invalid_sequence_numbers(decision, number, total):
    section = decision[4]
    with pytest.raises(ValueError):
        section.set_step(number, total)
    assert not section.property("decisionCard")


def test_opened_signal_and_keyboard_actions_do_not_apply_a_draft(decision, application):
    _page, _form, control, _error, section, _window = decision
    calls = []
    section.set_step(1, 2)
    section.opened.connect(lambda: calls.append("opened"))
    section.configure_navigation(
        back_callback=lambda: calls.append("back"),
        next_callback=lambda: calls.append("next"),
        next_label="Choose sampling",
    )
    section.open()
    application.processEvents()
    control.lineEdit().selectAll()
    QTest.keyClicks(control.lineEdit(), "1e")
    assert not control.hasAcceptableInput()
    section.next_button.setFocus()
    QTest.keyClick(section.next_button, Qt.Key.Key_Space)
    section.back_button.setFocus()
    QTest.keyClick(section.back_button, Qt.Key.Key_Space)
    assert calls == ["opened", "next", "back"]
    assert section.dialog.isVisible()
    assert not control.hasAcceptableInput() and control.lineEdit().text().startswith("1e")
    section.dialog.close()
    section.open()
    assert calls == ["opened", "next", "back", "opened"]


def test_navigation_reconfiguration_replaces_callbacks_and_hides_unused_actions(decision, application):
    section = decision[4]
    calls = []
    section.configure_navigation(next_callback=lambda: calls.append("old"))
    original = section.next_button
    section.configure_navigation(next_callback=lambda: calls.append("new"), next_label="Field limits")
    section.open()
    application.processEvents()
    original.click()
    assert section.next_button is original
    assert calls == ["new"]
    assert section.next_button.text() == "Field limits →"
    assert not section.back_button.isVisible()
    section.configure_navigation()
    assert not section.next_button.isVisible()
    assert not section.back_button.isVisible()
    assert not section._navigation_row.isVisible()


def test_revert_is_one_owned_callback_and_does_not_close_or_rebuild_the_editor(decision, application):
    _page, _form, control, error, section, _window = decision
    value = control.value()
    calls = []
    section.set_revert_callback(lambda: calls.append("old"))
    button = section.revert_button
    section.set_revert_callback(lambda: calls.append("new"))
    section.open()
    application.processEvents()
    error.show()
    QTest.keyClick(button, Qt.Key.Key_Space)
    assert calls == ["new"]
    assert section.revert_button is button and section.dialog.isVisible()
    assert error.isVisible() and control.value() == value
    section.set_revert_callback(None)
    assert not button.isVisible()


def test_control_form_returns_moved_row_owner_and_supports_label_and_visibility(decision):
    _page, original, control, _error, section, window = decision
    assert original.labelForField(control.parentWidget()) is None
    owner = control_form(window, control, original)
    assert owner is section.form
    owner.labelForField(control.parentWidget()).setText("World center x")
    owner.setRowVisible(control.parentWidget(), False)
    assert section.form.labelForField(control.parentWidget()).text() == "World center x"
    assert not section.form.isRowVisible(control.parentWidget())
    assert section.open_for(control)
    assert section.form.isRowVisible(control.parentWidget())
    # An unrelated field retains its explicitly supplied main form.
    unrelated = QLabel("Unrelated")
    try:
        assert control_form(window, unrelated, original) is original
    finally:
        unrelated.deleteLater()


def test_control_form_includes_target_region_sections_and_finds_main_form(decision):
    page, original, control, _error, section, _window = decision
    page.form = original
    window = SimpleNamespace(
        _focused_sections=[],
        _target_region_editor=SimpleNamespace(focused_sections=[None, section]),
        settings=SimpleNamespace(widget=lambda: page),
    )
    assert control_form(window, control) is section.form
    assert control_form(window, section.button) is original


def test_simple_form_controls_can_reveal_their_actual_row(application):
    page = QWidget()
    form = QFormLayout(page)
    control = ScientificSpinBox(page)
    form.addRow("Amplitude", control)
    form.setRowVisible(control, False)
    section = FocusedFormSection(form, [control], title="Amplitude")
    try:
        assert not section.form.isRowVisible(control)
        assert section.open_for(control)
        assert section.form.isRowVisible(control)
        window = SimpleNamespace(_focused_sections=[section])
        assert control_form(window, control, form) is section.form
    finally:
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_card_adapts_to_a_narrow_inspector_and_palette_changes(decision, application):
    _page, _form, control, _error, section, _window = decision
    section.set_step(1, 3)
    section.resize(480, 90)
    section.show()
    application.processEvents()
    section._adapt_card()
    assert not section.icon_label.isHidden()
    section.resize(280, 110)
    section._adapt_card()
    assert section.icon_label.isHidden()
    assert not section.step_label.isHidden() and not section.button.isHidden()
    palette = QPalette(section.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#242424"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#ededed"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#ededed"))
    section.setPalette(palette)
    assert "#242424" in section.styleSheet()
    assert section.summary_label.text() == control.text()


def test_navigation_and_footer_stay_outside_the_scroll_body(decision, application):
    _page, _form, _control, _error, section, _window = decision
    section.configure_navigation(back_callback=lambda: None, next_callback=lambda: None)
    section.set_revert_callback(lambda: None)
    for index in range(30):
        section.form.addRow("Extra setting", QLabel(str(index)))
    section.dialog.resize(500, 300)
    section.open()
    application.processEvents()
    assert section.scroll_area.verticalScrollBar().maximum() > 0
    assert section.back_button.isVisible() and section.next_button.isVisible()
    assert section.footer.isVisible() and section.revert_button.isVisible()
    assert not section.scroll_area.isAncestorOf(section.next_button)
    assert not section.scroll_area.isAncestorOf(section.revert_button)
    assert section.footer.geometry().bottom() < section.dialog.height()


def test_card_styling_does_not_override_owned_editor_error_tones(decision, application):
    _page, _form, _control, error, section, _window = decision
    apply_tone(error, "error")
    expected = error.palette().color(QPalette.ColorRole.WindowText)
    section.set_step(1, 2)
    section.open()
    error.show()
    application.processEvents()
    assert error.palette().color(QPalette.ColorRole.WindowText) == expected
    assert error.property("semanticTone") == "error"


def test_direct_graphical_choice_embeds_original_tiles_and_preserves_draft_row(application):
    page = QWidget()
    page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    form = QFormLayout(page)
    combo = GraphicalChoiceCombo(
        title="Choose field",
        choices={
            "Gradient": ("target-gradient", "Gradient", "Linear spatial variation of Bz."),
            "Shim": ("target-harmonic", "Shim", "A normalized real harmonic target."),
        },
    )
    combo.addItems(["Gradient", "Shim"])
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.addWidget(combo)
    error = QLabel("Review this target", wrapper)
    error.hide()
    layout.addWidget(error)
    form.addRow("Target definition", wrapper)
    section = FocusedFormSection(form, [combo], title="Choose field", summary=combo.currentText)
    section.dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    combo.dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    changes = []
    combo.currentTextChanged.connect(changes.append)
    original_tiles, original_model, original_scroll = tuple(combo.buttons), combo.model(), combo.scroll
    try:
        section.set_step(1, 2)
        section.configure_navigation(next_callback=lambda: None)
        section.set_direct_choice(combo)
        assert section.button.text() == "Choose…"
        assert combo.isHidden() and combo.model() is original_model
        assert combo.scroll is original_scroll and tuple(combo.buttons) == original_tiles
        assert section.form.getWidgetPosition(combo.scroll)[0] >= 0
        section.open()
        application.processEvents()
        assert section.dialog.isVisible() and not combo.dialog.isVisible()
        assert combo.scroll.isVisible() and section.next_button.isVisible()
        assert combo.currentText() == "Gradient" and changes == []
        section.dialog.close()
        combo.showPopup()
        application.processEvents()
        assert section.dialog.isVisible() and not combo.dialog.isVisible()
        QTest.keyClick(combo.buttons[1], Qt.Key.Key_Return)
        assert combo.currentText() == "Shim" and changes == ["Shim"]
        assert section.summary_label.text() == "Shim"
        assert section.dialog.isVisible() and not combo.dialog.isVisible()
        assert section.dialog.focusWidget() is combo.buttons[1]
        section.dialog.close()
        section.open()
        application.processEvents()
        assert combo.currentText() == "Shim" and changes == ["Shim"]
        error.show()
        assert section.open_for(combo)
        application.processEvents()
        assert error.isVisible() and combo.isHidden()
        assert not combo.dialog.isVisible()
        # Repeated configuration does not duplicate the gallery row.
        section.set_direct_choice(combo)
        assert section.form.rowCount() == 2
        combo.setEnabled(False)
        application.processEvents()
        assert section.dialog.isVisible()
        assert all(not tile.isEnabled() for tile in combo.buttons)
    finally:
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_direct_graphical_choice_requires_one_authoritative_control(decision):
    section = decision[4]
    with pytest.raises(ValueError, match="one owned graphical"):
        section.set_direct_choice(decision[2])


@pytest.mark.parametrize("surface", ("title_label", "summary_label", "icon_label", "step_label", "whitespace"))
def test_numbered_card_opens_from_its_visible_graphical_surface(decision, application, surface):
    page, _form, control, _error, section, _window = decision
    value = control.value()
    section.set_step(1, 3)
    page.resize(620, 120)
    page.show()
    application.processEvents()
    calls = []
    section.opened.connect(lambda: calls.append("opened"))
    target = section if surface == "whitespace" else getattr(section, surface)
    position = QPoint(3, 3) if surface == "whitespace" else target.rect().center()
    QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=position)
    application.processEvents()
    assert calls == ["opened"] and section.dialog.isVisible()
    assert control.value() == value
    assert section.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert section.dialog.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_default_section_surface_click_does_not_change_original_launcher_behavior(decision, application):
    page, _form, _control, _error, section, _window = decision
    page.resize(620, 120)
    page.show()
    application.processEvents()
    calls = []
    section.opened.connect(lambda: calls.append("opened"))
    QTest.mouseClick(section.summary_label, Qt.MouseButton.LeftButton)
    QTest.mouseClick(section, Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
    assert not section.dialog.isVisible() and calls == []
    QTest.mouseClick(section.button, Qt.MouseButton.LeftButton)
    assert section.dialog.isVisible() and calls == ["opened"]


def test_numbered_card_button_opens_once_and_retains_keyboard_access(decision, application):
    page, _form, _control, _error, section, _window = decision
    section.set_step(1, 2)
    page.resize(620, 120)
    page.show()
    application.processEvents()
    calls = []
    section.opened.connect(lambda: calls.append("opened"))
    QTest.mouseClick(section.button, Qt.MouseButton.LeftButton)
    assert calls == ["opened"]
    section.dialog.close()
    section.button.setFocus()
    QTest.keyClick(section.button, Qt.Key.Key_Space)
    assert calls == ["opened", "opened"]
    assert section.button.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_numbered_card_right_click_keeps_editor_closed(decision):
    section = decision[4]
    section.set_step(1, 2)
    QTest.mouseClick(section.title_label, Qt.MouseButton.RightButton)
    assert not section.dialog.isVisible()


def test_numbered_card_draft_border_and_accessibility_keep_input_states_distinct(decision):
    section = decision[4]
    section.set_step(2, 3)
    original = section.styleSheet()
    blue = tone_color("info", section.palette()).name()
    warning = tone_color("warning", section.palette()).name()
    error = tone_color("error", section.palette()).name()
    section.set_draft_state(pending=True)
    assert section.property("draftState") == "pending"
    assert f"border: 1px solid {warning}" in section.styleSheet()
    assert f"QLabel#decision_number {{ color: {blue};" in section.styleSheet()
    assert "Pending input edits" in section.accessibleDescription()
    section.set_draft_state(pending=True, error=True)
    assert section.property("draftState") == "error"
    assert f"border: 1px solid {error}" in section.styleSheet()
    assert f"border-color: {error}" in section.styleSheet()
    assert "Invalid input draft" in section.accessibleDescription()
    assert "Pending input edits" not in section.accessibleDescription()
    assert "accepted" not in section.accessibleDescription().lower()
    section.set_draft_state()
    assert section.styleSheet() == original and section.property("draftState") == "normal"
    assert "Invalid input draft" not in section.accessibleDescription()


def test_default_section_draft_state_does_not_add_a_graphical_card(decision):
    section = decision[4]
    original = section.styleSheet()
    section.set_draft_state(pending=True, error=True)
    assert section.styleSheet() == original and not section.property("decisionCard")
    section.set_step(1)
    assert section.property("draftState") == "error"


@pytest.mark.parametrize("loaded", ("imported_map", "legacy_mesh"))
def test_direct_region_gallery_keeps_loaded_readonly_state_and_original_tiles(application, loaded):
    page = QWidget()
    page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    form = QFormLayout(page)
    gallery = TargetRegionGallery(compact=True)
    if loaded == "legacy_mesh":
        gallery.add_legacy_mesh()
    gallery.setCurrentIndex(gallery.findData(loaded))
    form.addRow("Field region", gallery)
    section = FocusedFormSection(form, [gallery], title="Choose field region", summary=gallery.currentText)
    section.dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    gallery.dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    model, buttons = gallery.model(), tuple(gallery.buttons)
    changes = []
    gallery.currentIndexChanged.connect(changes.append)
    try:
        section.set_step(1, 2)
        section.set_direct_choice(gallery)
        section.open()
        application.processEvents()
        selected = gallery.findData(loaded)
        assert gallery.model() is model and tuple(gallery.buttons) == buttons
        assert buttons[selected].isVisible() and not buttons[selected].isEnabled()
        assert gallery.currentData() == loaded and changes == []
        assert section.dialog.isVisible() and not gallery.dialog.isVisible()
        assert gallery.scroll.isVisible() and gallery.isHidden()
        section.dialog.close()
        gallery.showPopup()
        application.processEvents()
        assert section.dialog.isVisible() and not gallery.dialog.isVisible()
        assert gallery.currentData() == loaded and changes == []
        gallery.setEnabled(False)
        application.processEvents()
        assert all(not tile.isEnabled() for tile in buttons)
        assert section.dialog.isVisible() and gallery.currentData() == loaded
    finally:
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_direct_region_gallery_rejects_expanded_inline_gallery(application):
    page = QWidget()
    form = QFormLayout(page)
    gallery = TargetRegionGallery(compact=False)
    form.addRow("Field region", gallery)
    section = FocusedFormSection(form, [gallery], title="Choose field region")
    try:
        with pytest.raises(ValueError, match="one owned graphical"):
            section.set_direct_choice(gallery)
    finally:
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_error_reveal_enables_only_an_explicitly_edited_invalid_inactive_numeric(decision, application):
    _page, _form, control, _error, section, window = decision
    original = control.value()
    section.open()
    application.processEvents()
    control.lineEdit().selectAll()
    QTest.keyClicks(control.lineEdit(), "1e")
    control.setEnabled(False)
    window._changed_controls = {"center"}
    assert open_focused_control(window, control)
    assert control.isEnabled() and control.property("repairInactiveDraft")
    assert control.value() == original and not control.hasAcceptableInput()
    assert "does not change the selected mode" in control.toolTip()
    tooltip = control.toolTip()
    assert open_focused_control(window, control)
    assert control.toolTip() == tooltip


def test_error_reveal_preserves_untouched_disabled_numeric_preferences(decision):
    _page, _form, control, _error, _section, window = decision
    control.setEnabled(False)
    window._changed_controls = set()
    assert open_focused_control(window, control)
    assert not control.isEnabled() and not control.property("repairInactiveDraft")
