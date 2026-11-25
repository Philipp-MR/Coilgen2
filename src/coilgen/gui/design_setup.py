"""Persistent setup sidebar. Common controls and inline details share one draft."""

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..application.design_setup import DesignSetupDraft, SetupInputError


class DesignSetupPanel(QWidget):
    changed = Signal()

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.draft = None
        self.controls, self.cards, self.details = {}, {}, {}
        self.expanded = set()
        self._rebuilding = False
        self.advanced_host = QWidget(self)
        self.advanced_host.setLayout(QVBoxLayout())
        self.advanced_host.layout().setContentsMargins(0, 0, 0, 0)
        self.advanced_host.hide()
        self.setObjectName("design_setup")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(self.scroll, 1)
        footer = QWidget(self)
        actions = QVBoxLayout(footer)
        actions.addWidget(window.readiness_banner)
        self.status = QLabel("Edit the setup, then preview or compute.")
        self.status.setWordWrap(True)
        actions.addWidget(self.status)
        row = QHBoxLayout()
        self.preview = QPushButton("Preview")
        self.preview.clicked.connect(window.apply_settings)
        self.revert = QPushButton("Revert edits")
        self.revert.clicked.connect(self.reset)
        self.revert.setEnabled(False)
        row.addWidget(self.preview)
        row.addWidget(self.revert)
        actions.addLayout(row)
        layout.addWidget(footer)

    @property
    def dirty(self):
        return self.draft is not None and self.draft.dirty

    def set_project(self, project):
        if self.draft is not None and self.draft.project is project:
            return
        if self.draft is not None and self.draft.project.id != project.id:
            self.expanded.clear()
        self.draft = DesignSetupDraft(project)
        self._render()
        self._changed()

    def reset(self):
        if self.draft is not None:
            self.draft = DesignSetupDraft(self.window.project)
            self._render()
            self._changed()

    def _render(self):
        if self.draft is None:
            return
        self._rebuilding = True
        position = self.scroll.verticalScrollBar().value()
        body = QWidget()
        body.setObjectName("setup_sections")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(12)
        self.controls, self.cards, self.details = {}, {}, {}
        for section in self.draft.sections():
            key = section["key"]
            card = QFrame(body)
            card.setObjectName("setup_" + key)
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            title = QLabel(section["title"])
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            card_layout.addWidget(title)
            common = QFormLayout()
            common.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            common.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            card_layout.addLayout(common)
            toggle = QToolButton(card)
            toggle.setText("Details")
            toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            toggle.setCheckable(True)
            toggle.setObjectName("setup_details_" + key)
            toggle.setAccessibleName(section["title"] + " details")
            card_layout.addWidget(toggle)
            details = QWidget(card)
            detail_layout = QVBoxLayout(details)
            detail_layout.setContentsMargins(0, 0, 0, 0)
            help_ = QLabel(section["help"])
            help_.setWordWrap(True)
            detail_layout.addWidget(help_)
            if section["summary"]:
                summary = QLabel(section["summary"])
                summary.setWordWrap(True)
                detail_layout.addWidget(summary)
            detail_form = QFormLayout()
            detail_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            detail_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            detail_layout.addLayout(detail_form)
            for entry in section["fields"]:
                control = self._control(entry)
                form = common if entry["common"] else detail_form
                label = QLabel(entry["label"])
                label.setWordWrap(True)
                label.setBuddy(control)
                field = QWidget()
                field_layout = QVBoxLayout(field)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.setSpacing(4)
                field_layout.addWidget(label)
                field_layout.addWidget(control)
                form.addRow(field)
            card_layout.addWidget(details)
            self.cards[key], self.details[key] = card, (toggle, details)
            toggle.toggled.connect(lambda expanded, section_key=key: self._expand(section_key, expanded))
            toggle.setChecked(key in self.expanded)
            self._expand(key, key in self.expanded)
            layout.addWidget(card)
        # Specialist operations remain reachable, without occupying the main view.
        tools = QToolButton(body)
        tools.setText("More tools")
        tools.setCheckable(True)
        tools.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tools.setArrowType(Qt.ArrowType.RightArrow)
        layout.addWidget(tools)
        extra = QWidget(body)
        extra_layout = QVBoxLayout(extra)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        for title, callback in (
            ("Import support mesh…", self.window._import_mesh_dialog),
            ("Import target map…", self.window._import_target_dialog),
            ("Groups and circuits…", self.window._open_groups_editor),
            ("Advanced parameters…", self.window._advanced_dialog),
        ):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, operation=callback: self._tool(operation))
            extra_layout.addWidget(button)
        extra.hide()
        tools.toggled.connect(extra.setVisible)
        tools.toggled.connect(
            lambda opened: tools.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        )
        layout.addWidget(extra)
        layout.addWidget(self.advanced_host)
        layout.addStretch(1)
        previous = self.scroll.takeWidget()
        self.scroll.setWidget(body)
        if previous is not None:
            previous.deleteLater()
        self.scroll.verticalScrollBar().setValue(position)
        self._rebuilding = False

    def _control(self, entry):
        key, kind = entry["key"], entry["kind"]
        value = self.draft.values.get(key, "")
        if kind == "bool":
            control = QCheckBox("Enabled")
            control.setChecked(bool(value))
            control.toggled.connect(lambda checked: self._edit(key, checked))
        elif kind == "choice":
            control = QComboBox()
            control.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            control.setMinimumContentsLength(10)
            for choice in entry["choices"]:
                control.addItem(str(choice["title"]), choice["value"])
            control.setCurrentIndex(control.findData(value))
            control.currentIndexChanged.connect(lambda _index: self._edit(key, control.currentData(), rebuild=True))
        else:
            control = QLineEdit(str(value))
            control.textEdited.connect(lambda text: self._edit(key, text))
        control.setObjectName("setup_input_" + key)
        control.setAccessibleName(entry["label"])
        control.setToolTip(entry["help"])
        control.setMinimumWidth(0)
        self.controls[key] = control
        return control

    def _edit(self, key, value, *, rebuild=False):
        if self._rebuilding:
            return
        self.draft.update({key: value})
        if key in self.controls:
            self.controls[key].setStyleSheet("")
        self._changed()
        if rebuild:
            QTimer.singleShot(0, self, self._render)

    def _changed(self):
        self.status.setText(
            "Edits pending · Preview or Compute applies the whole setup."
            if self.dirty
            else "Edit the setup, then preview or compute."
        )
        self.revert.setEnabled(self.dirty)
        self.changed.emit()

    def _expand(self, key, expanded):
        toggle, details = self.details[key]
        details.setVisible(expanded)
        toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if expanded:
            self.expanded.add(key)
        else:
            self.expanded.discard(key)

    def focus_section(self, key, field=""):
        key = "conductor" if key == "material" else key
        if key not in self.cards:
            return
        self.details[key][0].setChecked(True)
        target = self.controls.get(field, self.cards[key])
        self.scroll.ensureWidgetVisible(target, 0, 24)
        if field in self.controls:
            target.setFocus(Qt.FocusReason.OtherFocusReason)

    def _tool(self, callback):
        if self.window.apply_settings():
            callback()

    def show_advanced(self, editor):
        editor.setParent(self.advanced_host, Qt.WindowType.Widget)
        editor.setMinimumWidth(0)
        editor.setMinimumHeight(580)
        editor.layout().itemAt(1).layout().setDirection(QHBoxLayout.Direction.TopToBottom)
        editor.configure_physics.setText("Configure physical objective…")
        for field in editor.fields.values():
            field.layout().itemAt(1).layout().setDirection(QHBoxLayout.Direction.TopToBottom)
        self.advanced_host.layout().addWidget(editor)
        self.advanced_host.show()
        editor.show()
        QTimer.singleShot(0, self, lambda: self.scroll.ensureWidgetVisible(self.advanced_host, 0, 24))

    def candidate(self, **kwargs):
        try:
            return self.draft.build(**kwargs)
        except SetupInputError as exc:
            self.status.setText(str(exc))
            self.focus_section(exc.section, exc.key)
            if exc.key in self.controls:
                self.controls[exc.key].setStyleSheet("border: 1px solid #b02020;")
            return None
