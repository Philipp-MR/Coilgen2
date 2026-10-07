"""Focused modeless editors over the workbench's existing draft widgets.

Sections move complete form rows, including their error labels. Opening or
closing an editor neither copies values nor commits a model revision.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .colors import tone_color
from .icons import icon


class FocusedFormSection(QWidget):
    """A compact launcher for one owned editor, preserving widget identity."""

    opened = Signal()

    def __init__(self, form, controls, *, title, icon_name="settings", summary=None, parent=None, apply_callback=None):
        super().__init__(parent or form.parentWidget())
        self.controls = tuple(controls)
        self._summary = summary
        self._title, self._icon_name = title, icon_name
        self._step = None
        self._styling_card = False
        self._navigation_callbacks = (None, None)
        self._revert_callback = None
        self._direct_choice = None
        self._card_click_targets = ()
        self._card_pressed = None
        self._draft_state = False, False
        self.back_button = self.next_button = self.revert_button = None
        self._navigation_row = None
        self.setObjectName("focused_" + title.lower().replace(" ", "_"))
        row = self._row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        self.button = QToolButton(self)
        self.button.setText(title + "…")
        self.button.setIcon(icon(icon_name))
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setAccessibleName("Edit " + title.lower())
        self.button.setToolTip("Open the focused editor. Closing it keeps pending edits.")
        row.addWidget(self.button)
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        row.addWidget(self.summary_label, 1)

        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(title)
        self.dialog.setObjectName(self.objectName() + "_window")
        self.dialog.setModal(False)
        self.dialog.setMinimumWidth(420)
        dialog_layout = QVBoxLayout(self.dialog)
        scroll = QScrollArea(self.dialog)
        self.scroll_area = scroll
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.form = QFormLayout(body)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        scroll.setWidget(body)
        dialog_layout.addWidget(scroll, 1)
        visible_rows = 0
        for control in self.controls:
            field = control.parentWidget()
            # Registered controls live in a field wrapper. For simple forms a
            # caller may pass the field itself instead.
            if form.getWidgetPosition(field)[0] < 0:
                field = control
            position = form.getWidgetPosition(field)[0]
            if position < 0:
                continue
            visible = form.isRowVisible(field)
            visible_rows += int(visible)
            result = form.takeRow(field)
            label = result.labelItem.widget() if result.labelItem is not None else None
            wrapper = result.fieldItem.widget()
            if label is None:
                self.form.addRow(wrapper)
            else:
                self.form.addRow(label, wrapper)
            self.form.setRowVisible(wrapper, visible)
            self._observe(control)
        footer = self.footer = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self.dialog)
        footer.rejected.connect(self.dialog.hide)
        if apply_callback is not None:
            apply = QPushButton("Apply & preview", self.dialog)
            apply.setIcon(icon("model-tree"))
            apply.clicked.connect(apply_callback)
            footer.addButton(apply, QDialogButtonBox.ButtonRole.ApplyRole)
        dialog_layout.addWidget(footer)
        # Small editors should feel focused. The scroll area still owns all
        # rows and expands error labels safely when a draft is invalid.
        screen = self.dialog.screen()
        available_height = screen.availableGeometry().height() - 64 if screen is not None else 720
        self.dialog.resize(520, min(max(120, available_height), 120 + 50 * visible_rows))
        self.button.clicked.connect(self.open)
        self.refresh_summary()

    def set_step(self, number, total=None):
        """Opt into a numbered decision card; numbering never gates navigation."""
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError("Decision number must be a positive integer.")
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < number):
            raise ValueError("Decision total must be an integer at least as large as its number.")
        if self._step is None:
            self._row.removeWidget(self.button)
            self._row.removeWidget(self.summary_label)
            self._row.setContentsMargins(9, 8, 9, 8)
            self._row.setSpacing(8)
            self.step_label = QLabel(self)
            self.step_label.setObjectName("decision_number")
            self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.step_label.setFixedWidth(28)
            self.icon_label = QLabel(self)
            self.icon_label.setObjectName("decision_icon")
            self.icon_label.setFixedSize(38, 38)
            self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._row.addWidget(self.step_label)
            self._row.addWidget(self.icon_label)
            text = QWidget(self)
            text.setMinimumWidth(0)
            text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            text_layout = QVBoxLayout(text)
            text_layout.setContentsMargins(0, 0, 0, 0)
            text_layout.setSpacing(3)
            self.title_label = QLabel(self._title, text)
            self.title_label.setObjectName("decision_title")
            self.title_label.setWordWrap(True)
            self.title_label.setTextFormat(Qt.TextFormat.PlainText)
            text_layout.addWidget(self.title_label)
            self.summary_label.setMinimumWidth(0)
            self.summary_label.setObjectName("decision_summary")
            text_layout.addWidget(self.summary_label)
            self._row.addWidget(text, 1)
            self._card_click_targets = (text, self.step_label, self.icon_label, self.title_label, self.summary_label)
            for target in self._card_click_targets:
                target.installEventFilter(self)
                target.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.dialog.setCursor(Qt.CursorShape.ArrowCursor)
            self.button.setText("Choose…" if self._direct_choice is not None else "Edit…")
            self.button.setIcon(icon("support-gallery" if self._direct_choice is not None else "settings"))
            self.button.setIconSize(QSize(16, 16))
            self.button.setObjectName("decision_editor_action")
            self.button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._row.addWidget(self.button)
            self.setMinimumHeight(74)
            self.setProperty("decisionCard", True)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._step = number, total
        self.step_label.setText(str(number))
        self.step_label.setToolTip(f"Decision {number}" + (f" of {total}" if total is not None else ""))
        self.setAccessibleName(self.step_label.toolTip() + ": " + self._title)
        self._describe_card()
        self._update_card_style()
        self._adapt_card()
        self.updateGeometry()

    def set_draft_state(self, pending=False, error=False):
        """Mark input edits on a numbered card, without making a result verdict."""
        self._draft_state = bool(pending), bool(error)
        if self._step is not None:
            self._update_card_style()
            self._describe_card()

    def _describe_card(self):
        pending, error = self._draft_state
        status = (
            " Invalid input draft; correct the highlighted settings."
            if error
            else " Pending input edits."
            if pending
            else ""
        )
        self.setAccessibleDescription("Open this decision independently. Closing keeps pending edits." + status)
        self.setProperty("draftState", "error" if error else "pending" if pending else "normal")

    def set_direct_choice(self, control):
        """Show one combo's existing graphical choices directly in this editor.

        The hidden compact combo retains its model, signals and registered
        draft row. Its original tiles move here without cloning selection.
        """
        from .choice_dialogs import GraphicalChoiceCombo
        from .region_selectors import TargetRegionGallery

        supported = isinstance(control, GraphicalChoiceCombo) or (
            isinstance(control, TargetRegionGallery) and control.compact
        )
        if not supported or self.controls != (control,):
            raise ValueError("Direct choices require one owned graphical choice control.")
        if self._direct_choice is not None:
            if self._direct_choice is not control:
                raise ValueError("This decision already owns a different direct choice.")
            return
        self._direct_choice = control
        control.dialog.hide()
        control.dialog.layout().removeWidget(control.scroll)
        self.form.addRow(control.scroll)
        control.set_embedded_host(self.dialog, open_callback=self.open)
        control.scroll.setMinimumHeight(240)
        # The outer form owns the available height. The embedded gallery's
        # preferred dialog size must not create a second scrollbar around
        # otherwise visible tiles; longer galleries scroll inside this view.
        control.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._hide_choice_summary()
        self.button.setText("Choose…")
        self.button.setIcon(icon("support-gallery"))
        self.button.setAccessibleName(self._title)
        self.button.setToolTip("Open the graphical choices. Closing keeps pending edits.")
        bounds = self.dialog.screen().availableGeometry()
        self.dialog.resize(min(650, bounds.width() - 24), min(530, bounds.height() - 64))

    def _hide_choice_summary(self):
        control = self._direct_choice
        if control is not None:
            label = self.form.labelForField(_control_field(self.form, control))
            if label is not None:
                label.hide()
            control.hide()

    def _focus_direct_choice(self):
        control = self._direct_choice
        if control is not None:
            selected = control.currentIndex()
            if not 0 <= selected < len(control.buttons) or not control.buttons[selected].isEnabled():
                selected = next((index for index, tile in enumerate(control.buttons) if tile.isEnabled()), -1)
            if selected >= 0:
                control.scroll.ensureWidgetVisible(control.buttons[selected])
                control.buttons[selected].setFocus(Qt.FocusReason.OtherFocusReason)

    def configure_navigation(self, back_callback=None, next_callback=None, next_label="Next decision"):
        """Add optional navigation actions; the owner validates and chooses destinations."""
        for callback in (back_callback, next_callback):
            if callback is not None and not callable(callback):
                raise TypeError("Decision navigation callbacks must be callable or None.")
        self._navigation_callbacks = back_callback, next_callback
        if self._navigation_row is None:
            self._navigation_row = QWidget(self.dialog)
            navigation = QHBoxLayout(self._navigation_row)
            navigation.setContentsMargins(0, 0, 0, 0)
            self.back_button = QPushButton("← Back", self._navigation_row)
            self.back_button.setObjectName("previous_decision")
            self.back_button.setAutoDefault(False)
            self.back_button.clicked.connect(lambda: self._navigate(0))
            navigation.addWidget(self.back_button)
            navigation.addStretch(1)
            self.next_button = QPushButton(self._navigation_row)
            self.next_button.setObjectName("next_decision")
            self.next_button.setAutoDefault(False)
            self.next_button.clicked.connect(lambda: self._navigate(1))
            navigation.addWidget(self.next_button)
            self.dialog.layout().insertWidget(self.dialog.layout().indexOf(self.footer), self._navigation_row)
        self.back_button.setVisible(back_callback is not None)
        self.back_button.setAccessibleName("Open previous decision")
        self.back_button.setToolTip("Go to the previous decision. Pending edits are checked by the workbench.")
        self.next_button.setText(str(next_label) + " →")
        self.next_button.setAccessibleName(str(next_label))
        self.next_button.setToolTip("Go to the next decision. Pending edits are checked by the workbench.")
        self.next_button.setVisible(next_callback is not None)
        self._navigation_row.setVisible(back_callback is not None or next_callback is not None)
        self._fit_footer()

    def set_revert_callback(self, callback):
        """Keep one explicit Revert action; its owner restores the scientific draft."""
        if callback is not None and not callable(callback):
            raise TypeError("Decision revert callback must be callable or None.")
        self._revert_callback = callback
        if self.revert_button is None and callback is not None:
            self.revert_button = QPushButton("Revert draft", self.dialog)
            self.revert_button.setObjectName("revert_decision_draft")
            self.revert_button.setIcon(icon("undo"))
            self.revert_button.setAutoDefault(False)
            self.revert_button.clicked.connect(self._revert)
            self.footer.addButton(self.revert_button, QDialogButtonBox.ButtonRole.ResetRole)
        if self.revert_button is not None:
            self.revert_button.setVisible(callback is not None)
            self.revert_button.setToolTip("Revert pending edits using the workbench's draft controls.")
        self._fit_footer()

    def _navigate(self, index):
        callback = self._navigation_callbacks[index]
        if callback is not None:
            callback()

    def _revert(self):
        if self._revert_callback is not None:
            self._revert_callback()

    def _fit_footer(self):
        # Navigation lives outside the scrolling body. Avoid adding enough
        # footer buttons in one line to force a wide scientific editor.
        navigation_height = (
            self._navigation_row.sizeHint().height()
            if self._navigation_row is not None and not self._navigation_row.isHidden()
            else 0
        )
        self.dialog.setMinimumHeight(80 + self.footer.sizeHint().height() + navigation_height)

    def _update_card_style(self):
        if getattr(self, "_step", None) is None or self._styling_card:
            return
        self._styling_card = True
        try:
            palette = self.palette()
            accent = tone_color("info", palette).name()
            pending, error = self._draft_state
            border = (
                tone_color("error" if error else "warning", palette).name()
                if pending or error
                else palette.color(QPalette.ColorRole.Mid).name()
            )
            hover_border = border if pending or error else accent
            background = palette.color(QPalette.ColorRole.Base).name()
            foreground = palette.color(QPalette.ColorRole.Text).name()
            self.setStyleSheet(
                f'QWidget[decisionCard="true"] {{ background: {background}; border: 1px solid {border};'
                " border-radius: 6px; }"
                f'QWidget[decisionCard="true"]:hover {{ border-color: {hover_border}; }}'
                f"QLabel#decision_number {{ color: {accent}; font-weight: 600; font-size: 16px; }}"
                f"QLabel#decision_summary {{ color: {foreground}; }}"
                f"QLabel#decision_title {{ color: {foreground}; font-weight: 600; }}"
                f"QToolButton#decision_editor_action {{ border: 1px solid {border}; border-radius: 4px; padding: 5px; }}"
                f"QToolButton#decision_editor_action:hover {{ border-color: {accent}; }}"
                f"QToolButton#decision_editor_action:focus {{ border: 2px solid {accent}; padding: 4px; }}"
            )
            self.icon_label.setPixmap(icon(self._icon_name).pixmap(QSize(36, 36)))
        finally:
            self._styling_card = False

    def _adapt_card(self):
        if getattr(self, "_step", None) is not None:
            # The number, title and action stay visible on narrow inspectors.
            self.icon_label.setVisible(self.width() >= 340)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adapt_card()

    def _card_mouse_event(self, surface, event):
        if getattr(self, "_step", None) is None or event.button() != Qt.MouseButton.LeftButton:
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            self._card_pressed = surface
        elif event.type() == QEvent.Type.MouseButtonRelease:
            clicked = self._card_pressed is surface and surface.rect().contains(event.position().toPoint())
            self._card_pressed = None
            if clicked:
                self.open()
        else:
            return False
        event.accept()
        return True

    def eventFilter(self, watched, event):
        if watched in self._card_click_targets and event.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            if self._card_mouse_event(watched, event):
                return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):
        if not self._card_mouse_event(self, event):
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._card_mouse_event(self, event):
            super().mouseReleaseEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            self._update_card_style()

    def _observe(self, control):
        if isinstance(control, QAbstractSpinBox):
            control.valueChanged.connect(self.refresh_summary)
            control.lineEdit().textEdited.connect(self.refresh_summary)
        elif isinstance(control, QComboBox):
            control.currentTextChanged.connect(self.refresh_summary)
        elif isinstance(control, QLineEdit):
            control.textEdited.connect(self.refresh_summary)
        elif hasattr(control, "toggled"):
            control.toggled.connect(self.refresh_summary)
        elif hasattr(control, "changed"):
            control.changed.connect(self.refresh_summary)

    def refresh_summary(self, *_args):
        if self._summary is None:
            text = ""
        else:
            text = self._summary() if callable(self._summary) else str(self._summary)
        self.summary_label.setText(text)
        self.summary_label.setToolTip(text)

    def open(self, *_args):
        self._hide_choice_summary()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        self._focus_direct_choice()
        self.opened.emit()

    def open_for(self, control):
        if control not in self.controls:
            return False
        field = _control_field(self.form, control)
        if field is not None:
            self.form.setRowVisible(field, True)
        self.open()
        if control is self._direct_choice:
            return True
        control.setFocus(Qt.FocusReason.OtherFocusReason)
        self.scroll_area.widget().layout().activate()
        self.scroll_area.ensureWidgetVisible(control)
        return True

    def dispose(self):
        self.dialog.hide()
        self.dialog.deleteLater()
        self.deleteLater()


def group_form_controls(window, form, title, keys, *, icon_name="settings", summary=None, apply_callback=None):
    """Move registered controls into one focused section and add its launcher."""
    controls = [window._draft_controls[key] for key in keys if key in window._draft_controls]
    if not controls:
        return None
    section = FocusedFormSection(
        form,
        controls,
        title=title,
        icon_name=icon_name,
        summary=summary,
        apply_callback=apply_callback,
    )
    form.addRow(section)
    return section


def open_focused_control(window, control):
    """Reveal a failed control in the inspector or its owned focused window."""
    changed = getattr(window, "_changed_controls", ())
    key = next(
        (name for name, candidate in getattr(window, "_draft_controls", {}).items() if candidate is control), None
    )
    if (
        isinstance(control, QAbstractSpinBox)
        and not control.isEnabled()
        and not control.hasAcceptableInput()
        and key in changed
    ):
        # A policy switch can make a user-entered invalid preference inactive.
        # Repair that exact retained entry without switching the scientific mode.
        control.setEnabled(True)
        if not control.property("repairInactiveDraft"):
            help_text = "Repair this retained draft entry. Correcting its text does not change the selected mode."
            control.setToolTip((control.toolTip() + "\n\n" + help_text).strip())
            control.setAccessibleDescription((control.accessibleDescription() + " " + help_text).strip())
            control.setProperty("repairInactiveDraft", True)
    sections = list(getattr(window, "_focused_sections", ()))
    editor = getattr(window, "_target_region_editor", None)
    sections.extend(getattr(editor, "focused_sections", ()))
    for section in sections:
        if section is not None and section.open_for(control):
            return True
    return False


def _control_field(form, control):
    """Registered controls use a row wrapper; simple forms can own controls directly."""
    for field in (control.parentWidget(), control):
        if field is not None and form.getWidgetPosition(field)[0] >= 0:
            return field
    return None


def control_form(window, control, fallback=None):
    """Find a moved control's current form for labels and conditional visibility."""
    sections = list(getattr(window, "_focused_sections", ()))
    editor = getattr(window, "_target_region_editor", None)
    sections.extend(getattr(editor, "focused_sections", ()))
    for section in sections:
        if section is not None and _control_field(section.form, control) is not None:
            return section.form
    if fallback is not None:
        return fallback
    settings = getattr(window, "settings", None)
    page = settings.widget() if settings is not None else None
    return getattr(page, "form", None)
