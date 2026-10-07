"""Compact scientific choice tiles with the existing combo-box contract.

The combo owns the model, current value and signals; its labelled buttons are
only a second view of those choices. Existing configuration handlers therefore
keep their exact semantics, including signal blocking during model loading.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import QComboBox, QGridLayout, QLayout, QMenu, QSizePolicy, QToolButton

from .editors import SUPPORT_EDITORS
from .icons import icon
from .winding_intent import WINDING_INTENTS


class IconChoiceCombo(QComboBox):
    """A combo model presented as clickable tiles rather than a popup.

    Standard ``addItem``, ``addItems``, item data and selection methods remain
    available. ``choices`` maps model labels to (icon identifier, tile label,
    tooltip). Unknown values receive a distinct imported/custom tile.
    """

    def __init__(self, parent=None, *, choices=None, columns=3):
        super().__init__(parent)
        self._choices = choices or {}
        self._maximum_columns = max(1, columns)
        self._columns = self._maximum_columns
        self._tile_width = 96
        self._tile_height = 78
        self.buttons: list[QToolButton] = []
        self._grid = QGridLayout(self)
        self._grid.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.currentIndexChanged.connect(self._sync_selection)
        model = self.model()
        model.rowsInserted.connect(self._rebuild)
        model.rowsRemoved.connect(self._rebuild)
        model.modelReset.connect(self._rebuild)
        model.dataChanged.connect(self._rebuild)

    def _rebuild(self, *_args):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() and item.widget() is not getattr(self, "more_button", None):
                item.widget().hide()
                item.widget().deleteLater()
        self.buttons = []
        for index in range(self.count()):
            text = self.itemText(index)
            symbol, label, tooltip = self._choices.get(
                text,
                ("support-import", "Imported /\ncustom", f"{text}. Geometry and source units remain in the model."),
            )
            button = QToolButton(self)
            button.setObjectName(f"choice_tile_{index}")
            button.setText(label)
            button.setAccessibleName(text)
            button.setAccessibleDescription(tooltip)
            button.setToolTip(tooltip)
            button.setIcon(icon(symbol))
            button.setIconSize(QSize(36, 36))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            button.setMinimumHeight(self._tile_height)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setStyleSheet(
                "QToolButton { border: 1px solid palette(mid); border-radius: 5px; padding: 4px; }"
                "QToolButton:checked { border: 2px solid palette(highlight);"
                " background-color: palette(alternate-base); font-weight: 600; }"
                "QToolButton:focus { border: 2px dashed palette(highlight); }"
                "QToolButton:hover { background-color: palette(alternate-base); }"
            )
            button.installEventFilter(self)
            button.clicked.connect(lambda _checked=False, selected=index: self._choose(selected))
            self.buttons.append(button)
        self._arrange()
        self._sync_selection()

    def _choose(self, index):
        self.setCurrentIndex(index)
        # A click on the already selected tile must leave it visibly selected.
        self._sync_selection()
        self.activated.emit(index)

    def _sync_selection(self, *_args):
        for index, button in enumerate(self.buttons):
            button.setChecked(index == self.currentIndex())
            button.setEnabled(
                bool(self.model().flags(self.model().index(index, self.modelColumn())) & Qt.ItemIsEnabled)
            )

    def setCurrentIndex(self, index):
        super().setCurrentIndex(index)
        self._sync_selection()

    def setCurrentText(self, text):
        super().setCurrentText(text)
        self._sync_selection()

    def _arrange(self):
        self._columns = min(self._maximum_columns, max(1, (self.width() + 6) // (self._tile_width + 6)))
        while self._grid.count():
            self._grid.takeAt(0)
        shown = self._shown_indices()
        for index, button in enumerate(self.buttons):
            button.setVisible(index in shown)
        for position, index in enumerate(shown):
            self._grid.addWidget(self.buttons[index], position // self._columns, position % self._columns)
        for column in range(self._maximum_columns):
            self._grid.setColumnStretch(column, int(column < self._columns))
        rows = max(1, math.ceil(len(shown) / self._columns))
        more_button = getattr(self, "more_button", None)
        extra_height = 0
        if more_button is not None and not more_button.isHidden():
            self._grid.addWidget(more_button, rows, 0, 1, self._columns)
            extra_height = more_button.sizeHint().height() + 6
        self.setFixedHeight(rows * self._tile_height + (rows - 1) * 6 + 4 + extra_height)
        self.updateGeometry()

    def _shown_indices(self):
        return list(range(len(self.buttons)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange()

    def sizeHint(self):
        columns = min(self._maximum_columns, max(1, len(self._shown_indices())))
        rows = max(1, math.ceil(len(self._shown_indices()) / columns))
        more_button = getattr(self, "more_button", None)
        extra_height = (
            more_button.sizeHint().height() + 6 if more_button is not None and not more_button.isHidden() else 0
        )
        return QSize(
            columns * self._tile_width + (columns - 1) * 6 + 4,
            rows * self._tile_height + (rows - 1) * 6 + 4 + extra_height,
        )

    def minimumSizeHint(self):
        return QSize(self._tile_width + 4, self.height())

    def paintEvent(self, _event):
        # The child tiles paint the control; no hidden native arrow or frame.
        pass

    def mousePressEvent(self, _event):
        # Empty space between tiles must not summon an unrelated popup.
        self.setFocus(Qt.FocusReason.MouseFocusReason)

    def showPopup(self):
        # Space/F4 keeps the graphical selector in view and focuses its choice.
        if 0 <= self.currentIndex() < len(self.buttons):
            self.buttons[self.currentIndex()].setFocus(Qt.FocusReason.PopupFocusReason)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if 0 <= self.currentIndex() < len(self.buttons):
            self.buttons[self.currentIndex()].setFocus(event.reason())

    def eventFilter(self, watched, event):
        if watched in self.buttons and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                delta = {
                    Qt.Key.Key_Left: -1,
                    Qt.Key.Key_Right: 1,
                    Qt.Key.Key_Up: -self._columns,
                    Qt.Key.Key_Down: self._columns,
                }[key]
                candidate = max(0, min(self.count() - 1, self.currentIndex() + delta))
                if self.model().flags(self.model().index(candidate, self.modelColumn())) & Qt.ItemIsEnabled:
                    self._choose(candidate)
                    self.buttons[candidate].setFocus(Qt.FocusReason.OtherFocusReason)
                return True
        return super().eventFilter(watched, event)


class SurfaceGallery(IconChoiceCombo):
    """Three common/active tiles, with the full surface family list on demand."""

    def __init__(self, parent=None):
        self._expanded = False
        self.more_button = None
        self.more_menu = None
        self.surface_actions = {}
        super().__init__(
            parent,
            choices={
                editor.label: (editor.icon_name, editor.label.replace(" ", "\n"), editor.description)
                for editor in SUPPORT_EDITORS
            },
        )
        self.setAccessibleName("Winding support surface")
        self.more_button = QToolButton(self)
        self.more_button.setObjectName("more_support_surfaces")
        self.more_button.setText("More surfaces…")
        self.more_button.setIcon(icon("support-gallery"))
        self.more_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.more_button.setAccessibleName("Choose another support surface primitive")
        self.more_button.setToolTip("Planar surfaces, bore profiles, paired surfaces, caps and toroidal segments.")
        self.more_menu = QMenu(self.more_button)
        self.more_button.setMenu(self.more_menu)
        for editor in SUPPORT_EDITORS:
            self.addItem(editor.label, editor.source)

    def _shown_indices(self):
        if self._expanded:
            return list(range(len(self.buttons)))
        shown = list(range(min(3, len(self.buttons))))
        current = self.currentIndex()
        if current >= 3 and current < len(self.buttons):
            shown[-1] = current
        return shown

    def set_expanded(self, expanded=True):
        """Browse every primitive graphically inside an on-demand editor."""
        self._expanded = bool(expanded)
        self.more_button.setVisible(not self._expanded)
        self._arrange()

    def _rebuild(self, *_args):
        super()._rebuild(*_args)
        if self.more_menu is not None:
            self.more_menu.clear()
            self.surface_actions = {}
            editors = {editor.source: editor for editor in SUPPORT_EDITORS}
            categories = dict.fromkeys(editor.category for editor in SUPPORT_EDITORS)
            for category in (*categories, "Imported / custom"):
                entries = [
                    index
                    for index in range(self.count())
                    if (
                        editors[self.itemData(index)].category
                        if self.itemData(index) in editors
                        else "Imported / custom"
                    )
                    == category
                ]
                if not entries:
                    continue
                self.more_menu.addSection(category)
                for index in entries:
                    source = self.itemData(index)
                    editor = editors.get(source)
                    action = self.more_menu.addAction(
                        icon(editor.icon_name if editor else "support-import"), self.itemText(index)
                    )
                    action.setCheckable(True)
                    action.setToolTip(editor.description if editor else self.itemText(index))
                    action.triggered.connect(lambda _checked=False, selected=index: self._choose(selected))
                    self.surface_actions[source] = action
            self._sync_selection()

    def _sync_selection(self, *_args):
        super()._sync_selection(*_args)
        self._arrange()
        for source, action in self.surface_actions.items():
            action.setChecked(source == self.currentData())
            index = self.findData(source)
            action.setEnabled(
                bool(self.model().flags(self.model().index(index, self.modelColumn())) & Qt.ItemIsEnabled)
            )

    def eventFilter(self, watched, event):
        if self._expanded:
            return super().eventFilter(watched, event)
        if watched in self.buttons and event.type() == QEvent.Type.KeyPress:
            shown = self._shown_indices()
            key = event.key()
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                delta = -1 if key in (Qt.Key.Key_Left, Qt.Key.Key_Up) else 1
                current = shown.index(self.currentIndex()) if self.currentIndex() in shown else 0
                target = max(0, min(len(shown) - 1, current + delta))
                index = shown[target]
                if self.buttons[index].isEnabled():
                    self._choose(index)
                    self.buttons[index].setFocus(Qt.FocusReason.OtherFocusReason)
                return True
            if (
                not self._expanded
                and key == Qt.Key.Key_Tab
                and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            ):
                self.more_button.setFocus(Qt.FocusReason.TabFocusReason)
                return True
        return super().eventFilter(watched, event)


class CircuitSelector(IconChoiceCombo):
    """Existing winding intention modes shown as physical circuit diagrams."""

    def __init__(self, parent=None):
        super().__init__(
            parent,
            columns=2,
            choices={
                "Connected circuit": (
                    "connected-winding",
                    "Connected circuit",
                    "Connect contours into one circuit. External feeds still require review.",
                ),
                "Separate circuits per support": (
                    "separate-circuits",
                    "Circuit per support",
                    "Connect contours separately on each support. External feeds still require review.",
                ),
                "Independent closed turns": (
                    "independent-turns",
                    "Independent turns",
                    "Keep turns electrically separate; no interconnection routing.",
                ),
                "Contours only": (
                    "contours",
                    "Contours only",
                    "Generate unconnected contours without routing or physical conductor solids.",
                ),
            },
        )
        self.setAccessibleName("Winding circuit intention")
        self.addItems(list(WINDING_INTENTS))
