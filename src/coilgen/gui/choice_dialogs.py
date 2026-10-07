"""Compact, modeless graphical choices over the standard combo-box model.

The combo remains the sole source of selection and item data. The summary and
popup are views of that model, so viewing choices never alters an input and
legacy or imported values remain available without inventing a replacement.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import icon


class _ChoiceTile(QAbstractButton):
    """A keyboard-accessible card with independently wrapping labels."""

    def __init__(self, text, symbol, label, description, parent):
        super().__init__(parent)
        self.setText(label)
        self.setIcon(icon(symbol))
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(text)
        self.setAccessibleDescription(description)
        self.setToolTip(description)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        self.symbol_label = QLabel(self)
        self.symbol_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.symbol_label.setFixedHeight(42)
        self.title_label = QLabel(label, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.title_label.setWordWrap(True)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        self.description_label = QLabel(description, self)
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        for child in (self.symbol_label, self.title_label, self.description_label):
            if child is not self.symbol_label:
                child.setTextFormat(Qt.TextFormat.PlainText)
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            child.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(child)
        layout.addStretch()
        self._update_icon()

    def _update_icon(self):
        mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
        self.symbol_label.setPixmap(self.icon().pixmap(QSize(40, 40), mode))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.EnabledChange):
            self._update_icon()
            self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        group = QPalette.ColorGroup.Active if self.isEnabled() else QPalette.ColorGroup.Disabled
        border = QPalette.ColorRole.Highlight if self.isChecked() or self.hasFocus() else QPalette.ColorRole.Mid
        background = QPalette.ColorRole.AlternateBase if self.isChecked() or self.underMouse() else QPalette.ColorRole.Base
        painter.setPen(palette.color(group, border))
        painter.setBrush(palette.color(group, background))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 7, 7)

    def sizeHint(self):
        return QSize(190, max(150, self.layout().heightForWidth(max(170, self.width()))))

    def minimumSizeHint(self):
        return QSize(130, 150)


class GraphicalChoiceCombo(QComboBox):
    """One selected icon/label; the complete graphical menu opens on demand.

    ``choices`` maps exact combo text to ``(icon_name, title, description)``.
    Item data, model flags, current values and signals retain QComboBox semantics.
    Unknown items are explicitly presented as imported/custom choices.
    """

    def __init__(self, parent=None, *, choices=None, title="Choose an option", columns=3):
        super().__init__(parent)
        self._choices = dict(choices or {})
        self._maximum_columns = max(1, int(columns))
        self._columns = self._maximum_columns
        self._connected_model = None
        self._embedded_host = None
        self._embedded_open_callback = None
        self._selection_label = "Choose…"
        self.buttons: list[_ChoiceTile] = []
        self.setFixedHeight(38)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(title)

        summary = QHBoxLayout(self)
        summary.setContentsMargins(0, 0, 0, 0)
        summary.setSpacing(4)
        self.selected_button = QToolButton(self)
        self.selected_button.setIconSize(QSize(26, 26))
        self.selected_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.selected_button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.selected_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.selected_button.clicked.connect(self.showPopup)
        self.change_button = QToolButton(self)
        self.change_button.setText("Change…")
        self.change_button.setIcon(icon("support-gallery"))
        self.change_button.setIconSize(QSize(19, 19))
        self.change_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.change_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.change_button.setAccessibleName(f"{title}: change selection")
        self.change_button.setToolTip("Open the graphical choices. Closing keeps the current selection.")
        self.change_button.clicked.connect(self.showPopup)
        summary.addWidget(self.selected_button, 1)
        summary.addWidget(self.change_button)
        self.setStyleSheet(
            "QToolButton { border: 1px solid palette(mid); border-radius: 5px; padding: 5px; }"
            "QToolButton:hover { background-color: palette(alternate-base); }"
            "QToolButton:focus { border: 2px solid palette(highlight); }"
        )

        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(title)
        self.dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.dialog.setModal(False)
        self.dialog.setObjectName("graphical_choice_dialog")
        self.dialog.setMinimumSize(260, 260)
        layout = QVBoxLayout(self.dialog)
        heading = QLabel(title, self.dialog)
        heading.setTextFormat(Qt.TextFormat.PlainText)
        heading.setWordWrap(True)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        self.scroll = QScrollArea(self.dialog)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setSpacing(10)
        self.scroll.setWidget(self.content)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)
        footer = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self.dialog)
        footer.rejected.connect(self.dialog.reject)
        layout.addWidget(footer)
        self.dialog.finished.connect(lambda _result: self.setFocus(Qt.FocusReason.PopupFocusReason))
        self.currentIndexChanged.connect(self._sync_selection)
        self._connect_model()
        self._rebuild()

    def _connect_model(self):
        if self._connected_model is not None:
            for name in ("rowsInserted", "rowsRemoved", "modelReset", "dataChanged", "layoutChanged"):
                try:
                    getattr(self._connected_model, name).disconnect(self._rebuild)
                except RuntimeError:
                    # QComboBox destroys its former default model on replacement.
                    break
        self._connected_model = self.model()
        for name in ("rowsInserted", "rowsRemoved", "modelReset", "dataChanged", "layoutChanged"):
            getattr(self._connected_model, name).connect(self._rebuild)

    def setModel(self, model):
        super().setModel(model)
        if hasattr(self, "_connected_model"):
            self._connect_model()
            self._rebuild()

    def setModelColumn(self, column):
        super().setModelColumn(column)
        if hasattr(self, "buttons"):
            self._rebuild()

    def _choice(self, index):
        text = self.itemText(index)
        return self._choices.get(
            text,
            (
                "support-import",
                text,
                "Imported / custom value. Retained until you choose another option.",
            ),
        )

    def _rebuild(self, *_args):
        while self.grid.count():
            self.grid.takeAt(0)
        for button in self.buttons:
            button.hide()
            button.deleteLater()
        self.buttons = []
        for index in range(self.count()):
            symbol, label, description = self._choice(index)
            button = _ChoiceTile(self.itemText(index), symbol, label, description, self.content)
            button.setObjectName(f"graphical_choice_{index}")
            button.installEventFilter(self)
            button.clicked.connect(lambda _checked=False, selected=index: self._choose(selected))
            self.buttons.append(button)
        self._arrange()
        self._sync_selection()

    def _arrange(self):
        self._columns = min(
            self._maximum_columns,
            max(1, len(self.buttons)),
            max(1, self.scroll.viewport().width() // 180),
        )
        for row in range(self.grid.rowCount()):
            self.grid.setRowStretch(row, 0)
        while self.grid.count():
            self.grid.takeAt(0)
        for index, button in enumerate(self.buttons):
            self.grid.addWidget(button, index // self._columns, index % self._columns)
        for column in range(self._maximum_columns):
            self.grid.setColumnStretch(column, int(column < self._columns))
        self.grid.setRowStretch(max(1, (len(self.buttons) + self._columns - 1) // self._columns), 1)

    def _item_enabled(self, index):
        return bool(self.model().flags(self.model().index(index, self.modelColumn(), self.rootModelIndex())) & Qt.ItemFlag.ItemIsEnabled)

    def _sync_selection(self, *_args):
        for index, button in enumerate(self.buttons):
            button.setChecked(index == self.currentIndex())
            button.setEnabled(self.isEnabled() and self._item_enabled(index))
        current = self.currentIndex()
        if 0 <= current < self.count():
            symbol, label, description = self._choice(current)
            self._selection_label = label.replace("\n", " ")
            self.selected_button.setIcon(icon(symbol))
            self.selected_button.setToolTip(f"{self.currentText()}\n{description}")
            self.selected_button.setAccessibleName(self.currentText())
            self.selected_button.setAccessibleDescription(description)
        else:
            self._selection_label = "Choose…"
            self.selected_button.setIcon(icon("support-gallery"))
            self.selected_button.setToolTip("No value is selected.")
            self.selected_button.setAccessibleName("No value selected")
        self.change_button.setEnabled(self.count() > 0)
        self._elide_selection()

    def _elide_selection(self):
        available = max(30, self.selected_button.width() - self.selected_button.iconSize().width() - 22)
        self.selected_button.setText(self.fontMetrics().elidedText(self._selection_label, Qt.TextElideMode.ElideRight, available))

    def setCurrentIndex(self, index):
        super().setCurrentIndex(index)
        if hasattr(self, "buttons"):
            self._sync_selection()

    def setCurrentText(self, text):
        super().setCurrentText(text)
        if hasattr(self, "buttons"):
            self._sync_selection()

    def _choose(self, index):
        if not self.isEnabled() or not 0 <= index < self.count() or not self._item_enabled(index):
            return
        self.setCurrentIndex(index)
        self.activated.emit(index)
        self.textActivated.emit(self.itemText(index))
        self.hidePopup()

    def set_embedded_host(self, dialog, open_callback=None):
        """Redirect this view to an existing decision after its scroll is moved.

        The owner reparents ``scroll`` and retains the same registered combo.
        This changes presentation only: items, values and signals stay intact.
        Selecting a tile keeps that decision open for its Apply/navigation.
        """
        if dialog is not None and not isinstance(dialog, QDialog):
            raise TypeError("An embedded choice host must be a QDialog or None.")
        if open_callback is not None and not callable(open_callback):
            raise TypeError("The embedded open callback must be callable or None.")
        self.dialog.hide()
        self._embedded_host = dialog
        self._embedded_open_callback = open_callback
        self._sync_selection()

    def _focus_choice(self):
        selected = self.currentIndex()
        if not 0 <= selected < len(self.buttons) or not self.buttons[selected].isEnabled():
            selected = next((i for i, button in enumerate(self.buttons) if button.isEnabled()), -1)
        if selected >= 0:
            self.scroll.ensureWidgetVisible(self.buttons[selected])
            self.buttons[selected].setFocus(Qt.FocusReason.PopupFocusReason)

    def showPopup(self):
        if not self.isEnabled() or not self.count():
            return
        self._sync_selection()
        if self._embedded_host is not None:
            if self._embedded_open_callback is not None:
                self._embedded_open_callback()
            else:
                self._embedded_host.show()
                self._embedded_host.raise_()
            self._arrange()
            self._focus_choice()
            return
        if not self.dialog.isVisible():
            screen = self.screen().availableGeometry()
            if not self.dialog.property("initial_size_set"):
                columns = min(self._maximum_columns, max(1, self.count()))
                width = min(max(340, columns * 220 + 24), max(260, screen.width() - 40))
                columns = min(columns, max(1, (width - 24) // 180))
                rows = (self.count() + columns - 1) // columns
                height = min(max(260, 82 + 150 * rows + 10 * (rows - 1)), 580, max(260, screen.height() - 60))
                self.dialog.resize(width, height)
                self.dialog.setProperty("initial_size_set", True)
            self.dialog.show()
        self.dialog.raise_()
        self._arrange()
        self._focus_choice()

    def hidePopup(self):
        self.dialog.hide()
        if self._embedded_host is not None:
            if self._embedded_host.isVisible():
                self._focus_choice()
        else:
            self.setFocus(Qt.FocusReason.PopupFocusReason)

    def eventFilter(self, watched, event):
        if watched is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._arrange()
        elif watched in self.buttons and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._choose(self.buttons.index(watched))
                return True
            deltas = {
                Qt.Key.Key_Left: -1,
                Qt.Key.Key_Right: 1,
                Qt.Key.Key_Up: -self._columns,
                Qt.Key.Key_Down: self._columns,
            }
            if key in deltas:
                current = self.buttons.index(watched)
                delta = deltas[key]
                candidate = current + delta
                while 0 <= candidate < len(self.buttons):
                    if self.buttons[candidate].isEnabled():
                        self.buttons[candidate].setFocus(Qt.FocusReason.OtherFocusReason)
                        self.scroll.ensureWidgetVisible(self.buttons[candidate])
                        break
                    candidate += delta
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F4, Qt.Key.Key_Down):
            self.showPopup()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.hidePopup()
            event.accept()
        else:
            # Viewing or navigating the compact summary is never a selection.
            event.ignore()

    def wheelEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.showPopup()
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide_selection()

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "selected_button") and event.type() == QEvent.Type.FontChange:
            self._elide_selection()
        if hasattr(self, "dialog") and event.type() == QEvent.Type.EnabledChange:
            self._sync_selection()
            if not self.isEnabled():
                self.dialog.hide()

    def sizeHint(self):
        return QSize(300, 38)

    def minimumSizeHint(self):
        return QSize(190, 38)

    def paintEvent(self, _event):
        # The two summary buttons replace the native combo frame and arrow.
        pass
