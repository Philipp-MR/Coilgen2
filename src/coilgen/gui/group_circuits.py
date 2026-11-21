"""Integrated contour-group plot and circuit assignment table."""

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor

from ..application.group_editor import draw_group_map, group_hit_positions
from .field_analysis import _FigurePanel


class GroupsCircuitsEditor(QWidget):
    prepare_requested = Signal()
    load_requested = Signal()
    apply_requested = Signal()
    close_requested = Signal()
    advanced_requested = Signal()
    draft_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor = None
        self.invalid_fields = {}
        self.rejected = set()
        self.expected_fingerprint = ""
        self._project = None
        self.stale = False
        self.setObjectName("groups_circuits_editor")
        layout = QVBoxLayout(self)
        toolbar = QGridLayout()
        toolbar_index = 0
        for label, name, signal in (
            ("Prepare groups", "prepare", self.prepare_requested),
            ("Load groups from selected run", "load", self.load_requested),
            ("Advanced: Routing…", "advanced", self.advanced_requested),
            ("Back to coil view", "back", self.close_requested),
        ):
            button = QPushButton(label)
            if name == "load":
                self.load_button = button
            if name == "prepare":
                self.prepare_button = button
            button.setObjectName("groups_" + name)
            button.clicked.connect(signal.emit)
            toolbar.addWidget(button, toolbar_index // 2, toolbar_index % 2)
            toolbar_index += 1
        layout.addLayout(toolbar)
        self.source = QLabel()
        self.source.setWordWrap(True)
        layout.addWidget(self.source)
        self.note = QLabel(
            "Prepare groups to inspect the contour topology. Existing calculations are reused when compatible."
        )
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.plot = _FigurePanel(self)
        self.plot.setMinimumHeight(180)
        layout.addWidget(self.plot, 1)
        actions = QGridLayout()
        self.view = QComboBox()
        self.view.addItems(("3D", "XY", "XZ", "YZ"))
        self.view.setAccessibleName("Group plot view")
        self.view.currentTextChanged.connect(self._draw)
        actions.addWidget(self.view, 0, 0, 1, 2)
        for index, (label, mode) in enumerate(
            (
                ("Join selected", "join"),
                ("Separate selected", "separate"),
                ("Connect all", "all"),
                ("Separate per group", "per_group"),
            )
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, selected=mode: self._assign(selected))
            actions.addWidget(button, 1 + index // 2, index % 2)
        layout.addLayout(actions)
        self.table = QTableWidget(0, 6)
        self.table.setObjectName("groups_circuits_table")
        self.table.setHorizontalHeaderLabels(
            ("Select", "Group / hierarchy", "Contours", "Circuit", "Spiral order", "Cut side")
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(230)
        self.table.itemSelectionChanged.connect(self._focus_row)
        layout.addWidget(self.table)
        bottom = QHBoxLayout()
        self.apply_button = QPushButton("Apply routing plan")
        self.apply_button.setObjectName("groups_apply")
        self.apply_button.clicked.connect(self.apply_requested.emit)
        self.revert_button = QPushButton("Revert group edits")
        self.revert_button.clicked.connect(self.revert)
        bottom.addWidget(self.apply_button)
        bottom.addWidget(self.revert_button)
        self.status = QLabel("Common evaluation current; external leads require separate review.")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        layout.addLayout(bottom)
        self.plot.canvas.mpl_connect("button_press_event", self._pick)
        self._positions = []
        self._last_view = ""
        self._refresh()

    @property
    def dirty(self):
        return bool(self.invalid_fields) or self.editor is not None and self.editor.dirty

    def set_editor(self, editor, fingerprint, project=None):
        self.editor = editor
        self.invalid_fields.clear()
        self.expected_fingerprint = fingerprint
        self._project = project if project is not None else self._project
        if editor is not None and self._project is not None:
            editor.bind(self._project)
        self.stale = False
        self._last_view = ""
        self._refresh()

    def synchronize(self, project):
        self._project = project
        if self.editor is None:
            return
        if (
            project is not None
            and project.fingerprint == self.expected_fingerprint
            and (self.editor.binding is None or self.editor.matches_inputs(project))
        ):
            was_stale = self.stale
            self.stale = False
            if was_stale:
                self.note.setText("Draft retained for this model revision. Apply or revert the group edits.")
                self._refresh()
            return
        if project is not None and not self.invalid_fields and self.editor.synchronize(project):
            self.expected_fingerprint = project.fingerprint
            self.stale = False
            self.note.setText("Group choices follow the current project revision.")
        elif self.dirty:
            self.stale = True
            self.note.setText("The model changed. Revert this draft, then prepare its current groups.")
        else:
            self.editor = None
            self.stale = False
            self.expected_fingerprint = ""
            self.note.setText("The model changed. Prepare its current contour groups.")
        self._refresh()

    def _focus_row(self):
        if self.editor is not None and self.table.currentRow() >= 0:
            self.editor.focused = self.editor.rows()[self.table.currentRow()]["key"]
            self._draw()

    def _pick(self, event):
        if self.editor is None or event.x is None or not self._positions:
            return
        height = self.plot.figure.bbox.height
        positions = group_hit_positions(self.plot.figure.axes[0], self.editor, self.view.currentText())
        key, x, y = min(positions, key=lambda p: (p[1] - event.x) ** 2 + (p[2] - (height - event.y)) ** 2)
        if (x - event.x) ** 2 + (y - (height - event.y)) ** 2 > 30**2:
            return
        self.editor.focused = key
        row = next(index for index, item in enumerate(self.editor.rows()) if item["key"] == key)
        self.table.selectRow(row)
        self._draw()

    def _draw(self, *_args):
        if self.editor is not None:
            view = self.view.currentText()
            camera = None
            if self._last_view == view and self.plot.figure.axes:
                axis = self.plot.figure.axes[0]
                camera = {"xlim": axis.get_xlim(), "ylim": axis.get_ylim()}
                if view == "3D":
                    camera.update(elev=axis.elev, azim=axis.azim, zlim=axis.get_zlim())
            self._positions = draw_group_map(self.plot.figure, self.editor, view, camera)
            self._last_view = view
            self.plot.canvas.draw_idle()

    def _select(self, key, checked):
        if checked:
            self.editor.selected.add(key)
        else:
            self.editor.selected.discard(key)

    def _change(self, key, field, value):
        try:
            self.editor.update(key, **{field: value})
            self.invalid_fields.pop((key, field), None)
            self.note.setText(
                "Spiral order preserves current circulation. Apply the plan, then recompute routing and review its checks."
            )
        except ValueError as exc:
            self.invalid_fields[(key, field)] = value
            self.note.setText(str(exc))
            self.apply_button.setEnabled(False)
            self.revert_button.setEnabled(True)
            self.draft_changed.emit(True)
            return
        self.apply_button.setEnabled(self.dirty and not self.invalid_fields and not self.stale)
        self.revert_button.setEnabled(self.dirty)
        if self._project:
            self.status.setText(self.editor.summary(self._project.config.levels))
        self.draft_changed.emit(self.dirty)

    def _assign(self, mode):
        if self.editor is None:
            return
        try:
            self.editor.assign(mode)
        except ValueError as exc:
            self.note.setText(str(exc))
            return
        self._refresh()

    def revert(self):
        if self.editor is not None:
            self.editor.discard()
            self.invalid_fields.clear()
            if self._project is not None or self.stale:
                self.synchronize(self._project)
            self._refresh()

    def mark_applied(self, fingerprint):
        self.editor.original = deepcopy(self.editor.plan)
        self.editor.initial_plan = deepcopy(self.editor.plan)
        self.expected_fingerprint = fingerprint
        self.stale = False
        if self._project is not None:
            self.editor.bind(self._project)
        self._refresh()

    def _refresh(self):
        self.table.blockSignals(True)
        rows = self.editor.rows() if self.editor is not None else []
        for row in rows:
            for (key, name), value in self.invalid_fields.items():
                if row["key"] == key:
                    row[name] = value
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            key = row["key"]
            check = QCheckBox()
            check.setAccessibleName("Select " + row["label"])
            check.setChecked(row["selected"])
            check.toggled.connect(lambda value, key=key: self._select(key, value))
            self.table.setCellWidget(index, 0, check)
            label_item = QTableWidgetItem(row["label"] + " · " + row["hierarchy"])
            label_item.setForeground(QColor(row["color"]))
            label_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(index, 1, label_item)
            count_item = QTableWidgetItem(str(row["contour_count"]))
            count_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(index, 2, count_item)
            for column, name, choices in (
                (3, "circuit", None),
                (
                    4,
                    "spiral",
                    (("auto", "Auto"), ("spiral_in", "Outside → inside"), ("spiral_out", "Inside → outside")),
                ),
                (5, "cut", (("auto", "Auto"), ("high", "High"), ("low", "Low"))),
            ):
                if choices is None:
                    control = QLineEdit(row[name])
                    control.setMaxLength(40)
                    control.editingFinished.connect(
                        lambda control=control, key=key: self._change(key, "circuit", control.text())
                    )
                else:
                    control = QComboBox()
                    for value, label in choices:
                        control.addItem(label, value)
                    control.setCurrentIndex(control.findData(row[name]))
                    control.currentIndexChanged.connect(
                        lambda _index, control=control, key=key, name=name: self._change(
                            key, name, control.currentData()
                        )
                    )
                control.setAccessibleName(row["label"] + " " + name)
                self.table.setCellWidget(index, column, control)
        self.table.blockSignals(False)
        self.table.setMaximumHeight(min(230, 50 + 35 * len(rows)))
        self.apply_button.setEnabled(self.dirty and not self.invalid_fields and not self.stale)
        self.revert_button.setEnabled(self.dirty)
        if rows:
            if self._project:
                self.status.setText(self.editor.summary(self._project.config.levels))
        else:
            self._positions = []
            self.plot.figure.clear()
            self.plot.canvas.draw_idle()
        self._draw()
        self.draft_changed.emit(self.dirty)
