"""Graphical target-region and sampling choices with the existing combo API."""

from PySide6.QtCore import QEvent, QSignalBlocker, QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
)

from .icons import icon
from .visual_selectors import IconChoiceCombo


_REGIONS = (
    ("sphere", "Sphere", "roi-sphere", "Sphere", "Spherical target region with a radius and world centre."),
    (
        "ellipsoid",
        "Ellipsoid",
        "roi-ellipsoid",
        "Ellipsoid",
        "Three semi-axes define the region; position and orientation refer to the world frame.",
    ),
    (
        "box",
        "Box",
        "roi-box",
        "Box",
        "A rectangular region defined by full side lengths, world centre and orientation.",
    ),
    (
        "cylinder",
        "Cylinder",
        "roi-cylinder",
        "Cylinder",
        "A closed cylindrical target region; radius, full length, world centre and orientation are explicit.",
    ),
    (
        "enclosing_mesh",
        "Enclosing mesh",
        "roi-enclosing-mesh",
        "Enclosing\nmesh",
        "Owned SI world mesh. Surface mode samples triangles and permits open surfaces; "
        "volume mode requires a closed watertight enclosure.",
    ),
    (
        "imported_map",
        "Imported target map",
        "target-map",
        "Imported\nmap",
        "Imported map coordinates and field values define the samples. This tile only describes an already loaded map.",
    ),
)


class TargetRegionGallery(IconChoiceCombo):
    """Primitive and owned-mesh regions; imported maps are an existing-input state."""

    def __init__(self, parent=None, *, compact=False):
        self.compact = compact
        choices = {label: (symbol, tile, description) for _value, label, symbol, tile, description in _REGIONS}
        if compact:
            QComboBox.__init__(self, parent)
            self._choices = choices
            self._embedded_host = None
            self._embedded_open_callback = None
            self.buttons = []
            self.dialog = QDialog(self)
            self.dialog.setWindowTitle("Choose target region")
            self.dialog.setObjectName("target_region_picker_window")
            self.dialog.setModal(False)
            self.dialog.setMinimumSize(260, 230)
            self.dialog.resize(430, 300)
            picker_layout = QVBoxLayout(self.dialog)
            self.scroll = QScrollArea(self.dialog)
            self.scroll.setWidgetResizable(True)
            self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self.scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
            self._picker = _CurrentRegionTiles(choices=choices, columns=3)
            self._picker.setModel(self.model())
            # Arrange against the intended window width before model loading;
            # a transient one-column minimum would otherwise enlarge the
            # dialog permanently while the items are inserted.
            self._picker.resize(self.dialog.width() - 22, self._picker.height())
            self.scroll.setWidget(self._picker)
            picker_layout.addWidget(self.scroll, 1)
            close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            close.rejected.connect(self.dialog.hide)
            picker_layout.addWidget(close)
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            self.selected_icon = QLabel(self)
            self.selected_label = QLabel(self)
            self.selected_label.setWordWrap(True)
            row.addWidget(self.selected_icon)
            row.addWidget(self.selected_label, 1)
            self.change_button = QToolButton(self)
            self.change_button.setText("Change…")
            self.change_button.setIcon(icon("support-gallery"))
            self.change_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.change_button.setAccessibleName("Choose target region geometry")
            self.change_button.clicked.connect(self.showPopup)
            row.addWidget(self.change_button)
            self.setFixedHeight(38)
            self.currentIndexChanged.connect(self._sync_selection)
            self._picker.currentIndexChanged.connect(self._picker_changed)
            self._picker.activated.connect(self._picker_activated)
            for signal in (
                self.model().rowsInserted,
                self.model().rowsRemoved,
                self.model().modelReset,
                self.model().dataChanged,
            ):
                signal.connect(self._rebuild)
        else:
            super().__init__(parent, choices=choices, columns=3)
        self.setObjectName("target_region_gallery")
        self.setAccessibleName("Target region geometry")
        for value, label, _symbol, _tile, _description in _REGIONS:
            self.addItem(label, value)
        # Map importing belongs to the parent workflow. Preserve programmatic
        # selection for loaded maps without offering a tile that creates one.
        self.model().item(self.findData("imported_map")).setEnabled(False)

    def _rebuild(self, *_args):
        if not self.compact:
            return super()._rebuild(*_args)
        self._picker._choices = self._choices
        self._picker._rebuild()
        self.buttons = self._picker.buttons
        self._sync_selection()

    def _sync_selection(self, *_args):
        if not self.compact:
            return super()._sync_selection(*_args)
        with QSignalBlocker(self._picker):
            self._picker.setCurrentIndex(self.currentIndex())
        self._picker.setEnabled(self.isEnabled())
        self._picker._arrange()
        text = self.currentText() or "Choose a valid region"
        symbol, _tile, description = self._choices.get(text, ("roi-enclosing-mesh", "", text))
        self.selected_label.setText(text)
        self.selected_icon.setPixmap(icon(symbol).pixmap(QSize(28, 28)))
        self.setToolTip(description)

    def _picker_changed(self, index):
        self.setCurrentIndex(index)

    def _picker_activated(self, index):
        if not self.isEnabled():
            return
        self.activated.emit(index)
        self.textActivated.emit(self.itemText(index))
        self.hidePopup()

    def set_embedded_host(self, dialog, open_callback=None):
        """Redirect the compact picker's existing scroll view to one decision.

        The owner moves ``scroll`` before calling this method. The main combo
        and the picker continue sharing their original item model.
        """
        if not self.compact:
            raise ValueError("Only compact target-region galleries can be embedded.")
        if dialog is not None and not isinstance(dialog, QDialog):
            raise TypeError("An embedded region host must be a QDialog or None.")
        if open_callback is not None and not callable(open_callback):
            raise TypeError("The embedded open callback must be callable or None.")
        self.dialog.hide()
        self._embedded_host = dialog
        self._embedded_open_callback = open_callback
        self._sync_selection()

    def _focus_choice(self):
        shown = self._picker._shown_indices()
        selected = self.currentIndex()
        if selected not in shown:
            selected = next((index for index in shown if self.buttons[index].isEnabled()), -1)
        if selected >= 0:
            self.scroll.ensureWidgetVisible(self.buttons[selected])
            if self.buttons[selected].isEnabled():
                self.buttons[selected].setFocus(Qt.FocusReason.PopupFocusReason)

    def hidePopup(self):
        if not self.compact:
            return super().hidePopup()
        self.dialog.hide()
        if self._embedded_host is not None:
            if self._embedded_host.isVisible():
                self._focus_choice()
        else:
            self.setFocus(Qt.FocusReason.PopupFocusReason)

    def _arrange(self):
        if not self.compact:
            return super()._arrange()

    def sizeHint(self):
        return QSize(250, 38) if self.compact else super().sizeHint()

    def minimumSizeHint(self):
        return QSize(180, 38) if self.compact else super().minimumSizeHint()

    def showPopup(self):
        if not self.compact:
            return super().showPopup()
        if not self.isEnabled():
            return
        self._sync_selection()
        if self._embedded_host is not None:
            if self._embedded_open_callback is not None:
                self._embedded_open_callback()
            else:
                self._embedded_host.show()
                self._embedded_host.raise_()
        else:
            self.dialog.show()
            self.dialog.raise_()
            self.dialog.activateWindow()
        self._focus_choice()

    def changeEvent(self, event):
        super().changeEvent(event)
        if self.compact and hasattr(self, "_picker") and event.type() == QEvent.Type.EnabledChange:
            self._sync_selection()
            if not self.isEnabled():
                self.dialog.hide()

    def focusInEvent(self, event):
        if self.compact:
            QComboBox.focusInEvent(self, event)
        else:
            super().focusInEvent(event)

    def add_legacy_mesh(self):
        """Represent a loaded legacy target mesh without reinterpreting samples."""
        existing = self.findData("legacy_mesh")
        if existing >= 0:
            return existing
        self._choices["Legacy mesh"] = (
            "roi-enclosing-mesh",
            "Legacy\nmesh",
            "Preserve original target-mesh vertex/bounding-box sampling. "
            "This is not an enclosing-volume sampling request.",
        )
        self.addItem("Legacy mesh", "legacy_mesh")
        index = self.findData("legacy_mesh")
        self.model().item(index).setEnabled(False)
        return index


class _CurrentRegionTiles(IconChoiceCombo):
    """Ignored map/legacy states remain visible only when they are selected."""

    def _choose(self, index):
        if (
            self.isEnabled()
            and 0 <= index < self.count()
            and self.model().flags(self.model().index(index, self.modelColumn())) & Qt.ItemFlag.ItemIsEnabled
        ):
            super()._choose(index)

    def _shown_indices(self):
        return [
            index
            for index in range(len(self.buttons))
            if self.itemData(index) not in {"imported_map", "legacy_mesh"} or index == self.currentIndex()
        ]

    def eventFilter(self, watched, event):
        if (
            watched in self.buttons
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            self._choose(self.buttons.index(watched))
            return True
        return super().eventFilter(watched, event)


class RegionSamplingSelector(IconChoiceCombo):
    """Sampling scope is separate from region shape and acceptance evidence."""

    def __init__(self, parent=None):
        super().__init__(
            parent,
            columns=2,
            choices={
                "Volume samples": (
                    "roi-volume",
                    "Volume\nsamples",
                    "Sample throughout the region interior. Sampled results still need their stated acceptance checks.",
                ),
                "Surface samples": (
                    "roi-surface",
                    "Surface\nsamples",
                    "Sample only the region boundary. Surface results do not establish field accuracy throughout its volume.",
                ),
            },
        )
        self.setObjectName("target_region_sampling")
        self.setAccessibleName("Target region sampling scope")
        self.addItem("Volume samples", "volume")
        self.addItem("Surface samples", "surface")

    def _rebuild(self, *_args):
        super()._rebuild(*_args)
        self._tile_height = 36
        for button in self.buttons:
            button.setText(button.text().split("\n", 1)[0])
            button.setMinimumHeight(36)
            button.setIconSize(QSize(22, 22))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._arrange()

    def _arrange(self):
        # The two scopes are a segmented control, never a tall tile gallery.
        self._columns = 2
        while self._grid.count():
            self._grid.takeAt(0)
        for index, button in enumerate(self.buttons):
            self._grid.addWidget(button, 0, index)
            self._grid.setColumnStretch(index, 1)
        self.setFixedHeight(40)
        self.updateGeometry()

    def sizeHint(self):
        return QSize(250, 40)

    def minimumSizeHint(self):
        return QSize(180, 40)
