"""A retained cross-section draft: coordinates and directly movable vertices."""

from __future__ import annotations

import math

import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractItemDelegate,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..application.conductor_profiles import (
    canonical_profile_points,
    is_rectangular_profile,
    validate_conductor_profile,
)
from .icons import icon


class ConductorProfileEditor(QWidget):
    """One draft control storing physical SI points, with millimetre UI values.

    Coordinates are not recentered or reordered. Closing the owning window
    keeps both valid and incomplete table edits. The parent commits ``profile``
    through its normal Apply/undo/save boundary.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("conductor_profile_editor")
        self.setAccessibleName("Conductor cross-section editor")
        self._loading = False
        self._polygon_initialized = False
        self._loaded_error = ""
        self._width_mm = self._height_mm = 2.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Cross-section", self))
        self.mode_combo = QComboBox(self)
        self.mode_combo.setObjectName("conductor_profile_mode")
        self.mode_combo.addItem(icon("conductor-profile"), "Rectangle from width / height", "rectangle")
        self.mode_combo.addItem(icon("mesh"), "Polygon from straight segments", "polygon")
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)
        self.instructions = QLabel(
            "Edit coordinates or drag numbered points. The last point closes to the first.", self
        )
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        table_panel = QWidget(splitter)
        self._table_panel = table_panel
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 2, table_panel)
        self.table.setItemDelegate(_PointDelegate(self.table))
        self.table.setObjectName("conductor_profile_points")
        self.table.setAccessibleName("Ordered profile point coordinates in millimetres")
        self.table.setHorizontalHeaderLabels(("u [mm]", "n [mm]"))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumWidth(180)
        self.table.setToolTip(
            "Point order defines the boundary segments. u is transverse to the wire; n follows the support normal."
        )
        table_layout.addWidget(self.table, 1)
        point_actions = QHBoxLayout()
        self.add_button = QPushButton("Split segment", table_panel)
        self.add_button.setIcon(icon("edit"))
        self.add_button.setToolTip("Insert a midpoint after the selected point, then move it to form a convex corner.")
        self.remove_button = QPushButton("Remove", table_panel)
        self.remove_button.setIcon(icon("remove"))
        self.remove_button.setToolTip("Remove the selected point; a polygon needs at least three points.")
        point_actions.addWidget(self.add_button)
        point_actions.addWidget(self.remove_button)
        table_layout.addLayout(point_actions)
        self.preview = _ProfilePreview(splitter)
        splitter.addWidget(table_panel)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((240, 420))
        layout.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        self.rectangle_button = QPushButton("Start from rectangle", self)
        self.rectangle_button.setIcon(icon("conductor-profile"))
        self.rectangle_button.setToolTip(
            "Replace polygon vertices with a centered rectangle using the current width and height."
        )
        self.fit_button = QPushButton("Fit view", self)
        self.fit_button.setIcon(icon("fit-view"))
        bottom.addWidget(self.rectangle_button)
        bottom.addStretch(1)
        bottom.addWidget(self.fit_button)
        layout.addLayout(bottom)
        self.status_label = QLabel(self)
        self.status_label.setObjectName("conductor_profile_validation")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        axes = QLabel("u: in-surface, across wire · n: support normal · origin: modeled path reference", self)
        axes.setWordWrap(True)
        axes.setToolTip(
            "Conductor mounting may subsequently offset this reference. Editing the profile does not move or recenter it."
        )
        layout.addWidget(axes)
        self.table.itemChanged.connect(self._table_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.add_button.clicked.connect(self.insert_point)
        self.remove_button.clicked.connect(self.remove_point)
        self.rectangle_button.clicked.connect(self.start_rectangle)
        self.fit_button.clicked.connect(self.preview.fit_view)
        self.preview.point_selected.connect(self.select_point)
        self.preview.point_moved.connect(self._move_point)
        self.set_profile(((0.0, 0.0),))

    def set_profile(self, profile_metres, width_mm=2.0, height_mm=2.0):
        """Load an applied model without generating a draft change signal."""
        self._loading = True
        try:
            self._width_mm, self._height_mm = width_mm, height_mm
            rectangle = is_rectangular_profile(profile_metres)
            self._polygon_initialized = not rectangle
            self._loaded_error = ""
            try:
                points = self._rectangle_points() if rectangle else canonical_profile_points(profile_metres)
            except (ValueError, TypeError, OverflowError) as error:
                # Invalid imported data still needs a usable correction window.
                # Retain representable cells; do not reinterpret bad data as
                # the default rectangle or modify the applied project.
                self._loaded_error = str(error)
                try:
                    raw = np.asarray(profile_metres, dtype=object)
                    if raw.ndim == 2 and raw.shape[0] == 2 and raw.shape[1] != 2:
                        raw = raw.T
                    points = tuple(tuple(row) for row in raw) if raw.ndim == 2 and raw.shape[1:] == (2,) else ()
                except (ValueError, TypeError):
                    points = ()
            self._set_points(points)
            self.mode_combo.setCurrentIndex(0 if rectangle else 1)
            self._update_state()
            self.preview.fit_view()
        finally:
            self._loading = False

    def sizeHint(self):
        return QSize(640, 390)

    def minimumSizeHint(self):
        return QSize(460, 375)

    def set_rectangle_dimensions(self, width_mm, height_mm):
        self._width_mm, self._height_mm = width_mm, height_mm
        if self.mode_combo.currentData() == "rectangle":
            self._update_state()

    def draft_value(self):
        """Stable raw draft identity, including invalid or incomplete input."""
        mode = self.mode_combo.currentData()
        rows = tuple(
            tuple(self.table.item(row, col).text() for col in range(2)) for row in range(self.table.rowCount())
        )
        return (mode, rows if mode == "polygon" else ())

    def value(self):
        return self.profile()

    def summary_text(self):
        error = self.validation_message()
        if error:
            return "Profile incomplete"
        if self.mode_combo.currentData() == "rectangle":
            return f"Rectangle {float(self._width_mm):g} × {float(self._height_mm):g} mm"
        summary = validate_conductor_profile(self.profile())
        return f"Polygon · {len(summary.points_m)} points · {summary.area_m2 * 1e6:g} mm²"

    def focus_error(self):
        for row in range(self.table.rowCount()):
            for col in range(2):
                try:
                    self._coordinate_metres(row, col)
                except (ValueError, TypeError, OverflowError):
                    self.table.setCurrentCell(row, col)
                    self.table.setFocus()
                    self.table.editItem(self.table.item(row, col))
                    return
        self.table.setFocus()

    def profile(self):
        """Return the rectangle sentinel or validated, ordered SI vertices."""
        if self.mode_combo.currentData() == "rectangle":
            try:
                width, height = float(self._width_mm) * 1e-3, float(self._height_mm) * 1e-3
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError("Enter positive rectangle width and height in Conductor settings.") from error
            validate_conductor_profile(((0.0, 0.0),), width_m=width, height_m=height)
            return ((0.0, 0.0),)
        if self._loaded_error and self.table.rowCount() == 0:
            raise ValueError(self._loaded_error + " Start from rectangle to replace this malformed profile.")
        points = tuple(
            tuple(self._coordinate_metres(row, col) for col in range(2)) for row in range(self.table.rowCount())
        )
        summary = validate_conductor_profile(points)
        return tuple(tuple(point) for point in summary.points_m)

    def validation_message(self):
        try:
            self.profile()
        except (ValueError, TypeError, OverflowError) as error:
            return str(error) or "Enter finite cross-section coordinates."
        return ""

    def _coordinate_metres(self, row, col):
        item = self.table.item(row, col)
        text = item.text().strip()
        try:
            mm = float(text)
        except ValueError as error:
            raise ValueError(f"Point {row + 1}, {'u' if col == 0 else 'n'}: enter a number in mm.") from error
        if not math.isfinite(mm):
            raise ValueError(f"Point {row + 1}: coordinates must be finite.")
        source = item.data(Qt.ItemDataRole.UserRole)
        # Loading/showing/saving a profile must not round unchanged SI values.
        return float(source[1]) if source is not None and source[0] == text else mm * 1e-3

    def _rectangle_points(self):
        try:
            width, height = float(self._width_mm) * 1e-3, float(self._height_mm) * 1e-3
            if not all(math.isfinite(number) and number > 0 for number in (width, height)):
                return ()
        except (TypeError, ValueError, OverflowError):
            return ()
        return ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2))

    def _set_points(self, points):
        self._close_cell_editor()
        previous = self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(points))
            for row, point in enumerate(points):
                for col, coordinate in enumerate(point):
                    try:
                        source = float(coordinate)
                        text = repr(source * 1e3)
                    except (ValueError, TypeError, OverflowError):
                        source = None
                        text = str(coordinate)
                    item = QTableWidgetItem(text)
                    if source is not None:
                        item.setData(Qt.ItemDataRole.UserRole, (text, source))
                    self.table.setItem(row, col, item)
            self.table.setVerticalHeaderLabels(tuple(str(index + 1) for index in range(len(points))))
            if len(points):
                self.table.selectRow(0)
        finally:
            self.table.blockSignals(previous)

    def _mode_changed(self):
        if self.mode_combo.currentData() == "polygon" and not self._polygon_initialized:
            self._set_points(self._rectangle_points())
            self._polygon_initialized = True
        self._update_state()
        self.preview.fit_view()
        self._emit_changed()

    def start_rectangle(self):
        if not self._rectangle_points():
            self.status_label.setText("Complete positive width and height before starting a rectangle.")
            return
        loading = self._loading
        self._loading = True
        self._set_points(self._rectangle_points())
        self._polygon_initialized = True
        self._loaded_error = ""
        self.mode_combo.setCurrentIndex(1)
        self._loading = loading
        self._update_state()
        self.preview.fit_view()
        self._emit_changed()

    def insert_point(self):
        if self.mode_combo.currentData() != "polygon":
            return
        try:
            points = [
                tuple(self._coordinate_metres(row, col) for col in range(2)) for row in range(self.table.rowCount())
            ]
        except (ValueError, TypeError, OverflowError):
            self.status_label.setText("Complete coordinate values before splitting a segment.")
            return
        if len(points) < 3:
            return
        row = max(0, self.table.currentRow())
        following = (row + 1) % len(points)
        midpoint = tuple((points[row][col] + points[following][col]) / 2 for col in range(2))
        # Rebuilding from exact SI values preserves every untouched coordinate.
        points.insert(row + 1, midpoint)
        self._set_points(points)
        self.select_point(row + 1)
        self._update_state()
        self._emit_changed()

    def remove_point(self):
        row = self.table.currentRow()
        if self.mode_combo.currentData() != "polygon" or self.table.rowCount() <= 3 or row < 0:
            return
        self._close_cell_editor()
        self.table.removeRow(row)
        self.table.setVerticalHeaderLabels(tuple(str(index + 1) for index in range(self.table.rowCount())))
        self.select_point(min(row, self.table.rowCount() - 1))
        self._update_state()
        self._emit_changed()

    def select_point(self, row):
        if 0 <= row < self.table.rowCount():
            self.table.selectRow(row)
            self.table.setCurrentCell(row, 0)
        self._selection_changed()

    def _selection_changed(self):
        self.preview.set_selected(self.table.currentRow())
        self.remove_button.setEnabled(
            self.mode_combo.currentData() == "polygon" and self.table.rowCount() > 3 and self.table.currentRow() >= 0
        )

    def _move_point(self, row, u_mm, n_mm):
        if self.mode_combo.currentData() != "polygon" or not 0 <= row < self.table.rowCount():
            return
        self._close_cell_editor()
        previous = self.table.blockSignals(True)
        for col, number in enumerate((u_mm, n_mm)):
            self.table.item(row, col).setText(repr(float(number)))
        self.table.blockSignals(previous)
        self._table_changed()

    def _close_cell_editor(self):
        delegate = self.table.itemDelegate()
        if delegate.editor is not None:
            delegate.closeEditor.emit(delegate.editor, QAbstractItemDelegate.EndEditHint.NoHint)
            delegate.editor = None

    def _table_changed(self):
        self._update_state()
        self._emit_changed()

    def _emit_changed(self):
        if not self._loading:
            self.changed.emit()

    def _update_state(self):
        polygon = self.mode_combo.currentData() == "polygon"
        self._table_panel.setVisible(polygon)
        self.rectangle_button.setText("Reset to rectangle" if polygon else "Edit as polygon")
        self.table.setEnabled(polygon)
        self.add_button.setEnabled(polygon)
        self.instructions.setText(
            "Edit coordinates or drag numbered points. The last point closes to the first."
            if polygon
            else "The rectangle uses width and height from Conductor settings. Choose polygon to edit its segments."
        )
        self._selection_changed()
        error = self.validation_message()
        try:
            points = (
                tuple(
                    tuple(self._coordinate_metres(row, col) * 1e3 for col in range(2))
                    for row in range(self.table.rowCount())
                )
                if polygon
                else tuple(tuple(value * 1e3 for value in point) for point in self._rectangle_points())
            )
        except (ValueError, TypeError, OverflowError):
            points = ()
        if points:
            us, ns = zip(*points)
            if not all(math.isfinite(number) for number in (max(us) - min(us), max(ns) - min(ns))):
                points = ()
        self.preview.set_profile(points, editable=polygon, valid=not error)
        if error:
            self.status_label.setText("Cannot apply · " + error)
            self.status_label.setStyleSheet("color: #ba3c32;")
        else:
            summary = (
                validate_conductor_profile(
                    self.profile(), width_m=float(self._width_mm) * 1e-3, height_m=float(self._height_mm) * 1e-3
                )
                if not polygon
                else validate_conductor_profile(self.profile())
            )
            self.status_label.setText(
                (f"{len(summary.points_m)} points" if polygon else "Rectangle")
                + f" · area {summary.area_m2 * 1e6:g} mm² · "
                f"extent {summary.width_m * 1e3:g} × {summary.height_m * 1e3:g} mm · convex"
            )
            self.status_label.setStyleSheet("")


class _PointDelegate(QStyledItemDelegate):
    """Keep the model current even when Apply uses a keyboard shortcut.

    Text remains a string, so incomplete numeric input is retained and rejected
    by the scientific validator rather than converted by a spin-box delegate.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor = None

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        self.editor = editor
        editor.destroyed.connect(lambda: self._clear_editor(editor))
        if isinstance(editor, QLineEdit):
            editor.textEdited.connect(lambda _text: self._commit_text(editor))
        return editor

    def _clear_editor(self, editor):
        if self.editor is editor:
            self.editor = None

    def _commit_text(self, editor):
        cursor = editor.cursorPosition()
        start, selected = editor.selectionStart(), editor.selectedText()
        self.commitData.emit(editor)
        # Qt refreshes an open editor after dataChanged. Preserve the user's
        # insertion point rather than moving it to the end on each keystroke.
        editor.setCursorPosition(cursor)
        if start >= 0:
            editor.setSelection(start, len(selected))


class _ProfilePreview(QWidget):
    """Equal-scale physical axes with a fixed view transform during dragging."""

    point_selected = Signal(int)
    point_moved = Signal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("conductor_profile_drag_preview")
        self.setAccessibleName("Drag cross-section points in the local u and n plane")
        self.setMinimumSize(250, 250)
        self.setMouseTracking(True)
        self._points = ()
        self._editable = False
        self._valid = True
        self._selected = -1
        self._drag = -1
        self._bounds = (-1.4, 1.4, -1.4, 1.4)
        self._offset = QPointF()
        self.setToolTip(
            "Drag a numbered vertex. Both axes use the same physical scale. Fit view includes the coordinate origin."
        )

    def sizeHint(self):
        return QSize(420, 260)

    def set_profile(self, points, *, editable, valid):
        first = not self._points
        self._points = tuple(points)
        self._editable, self._valid = editable, valid
        if first and self._points and self._drag < 0:
            self.fit_view()
        self.update()

    def set_selected(self, selected):
        self._selected = selected
        self.update()

    def fit_view(self):
        if self._points:
            us, ns = zip(*self._points)
            left, right = min(0.0, *us), max(0.0, *us)
            bottom, top = min(0.0, *ns), max(0.0, *ns)
            span = max(right - left, top - bottom, 1e-6)
            padding = 0.18 * span
            self._bounds = (left - padding, right + padding, bottom - padding, top + padding)
        self.update()

    def _transform(self):
        rect = QRectF(48, 30, max(1, self.width() - 75), max(1, self.height() - 85))
        left, right, bottom, top = self._bounds
        scale = min(rect.width() / (right - left), rect.height() / (top - bottom))
        return rect.center(), scale, ((left + right) / 2, (bottom + top) / 2)

    def point_position(self, point):
        """Map millimetres to preview pixels (also used by offscreen tests)."""
        center, scale, middle = self._transform()
        return QPointF(center.x() + (point[0] - middle[0]) * scale, center.y() - (point[1] - middle[1]) * scale)

    def _coordinates(self, pixel):
        center, scale, middle = self._transform()
        return ((pixel.x() - center.x()) / scale + middle[0], -(pixel.y() - center.y()) / scale + middle[1])

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        text = palette.color(palette.ColorRole.Text)
        painter.fillRect(self.rect(), palette.color(palette.ColorRole.Base))
        if not self._points:
            painter.setPen(text)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Enter finite coordinates to preview")
            return
        origin = self.point_position((0, 0))
        painter.setPen(QPen(palette.color(palette.ColorRole.Mid), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(44, origin.y()), QPointF(self.width() - 20, origin.y()))
        painter.drawLine(QPointF(origin.x(), 20), QPointF(origin.x(), self.height() - 40))
        painter.setPen(text)
        painter.drawText(origin + QPointF(6, 17), "0,0")
        painter.drawText(
            QRectF(0, self.height() - 25, self.width(), 22), Qt.AlignmentFlag.AlignCenter, "u [mm] · transverse →"
        )
        painter.save()
        painter.translate(18, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-self.height() / 2, -12, self.height(), 22), Qt.AlignmentFlag.AlignCenter, "n [mm] · normal →"
        )
        painter.restore()
        color = QColor(204, 119, 50) if self._valid else QColor(186, 60, 50)
        painter.setPen(QPen(color.darker(140), 2))
        painter.setBrush(
            QColor(color.red(), color.green(), color.blue(), 115) if self._valid else Qt.BrushStyle.NoBrush
        )
        polygon = QPolygonF([self.point_position(point) for point in self._points])
        painter.drawPolygon(polygon)
        if self._editable:
            for index, pixel in enumerate(polygon):
                selected = index == self._selected
                painter.setBrush(QColor(31, 109, 181) if selected else palette.color(palette.ColorRole.Base))
                painter.setPen(QPen(QColor(31, 109, 181) if selected else color.darker(140), 2))
                painter.drawEllipse(pixel, 6 if selected else 5, 6 if selected else 5)
                painter.setPen(text)
                painter.drawText(pixel + QPointF(9, -9), str(index + 1))
        # Show numerical extents without inventing dimensions for an invalid polygon.
        us, ns = zip(*self._points)
        painter.setPen(text)
        painter.drawText(
            QRectF(0, 3, self.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            f"u {min(us):g} … {max(us):g} mm · n {min(ns):g} … {max(ns):g} mm",
        )

    def mousePressEvent(self, event):
        if not self._editable or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        position = event.position()
        for index, point in enumerate(self._points):
            pixel = self.point_position(point)
            if math.hypot(pixel.x() - position.x(), pixel.y() - position.y()) <= 12:
                self._drag = index
                self._offset = pixel - position
                self.point_selected.emit(index)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag >= 0:
            self.point_moved.emit(self._drag, *self._coordinates(event.position() + self._offset))
            event.accept()
            return
        if self._editable:
            hovering = any(
                math.hypot(
                    self.point_position(point).x() - event.position().x(),
                    self.point_position(point).y() - event.position().y(),
                )
                <= 12
                for point in self._points
            )
            self.setCursor(Qt.CursorShape.OpenHandCursor if hovering else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self._drag = -1
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
