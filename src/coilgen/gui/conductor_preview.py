"""Dimensioned conductor schematic; a view of supplied draft values only."""

from __future__ import annotations

import math

import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..application.conductor_profiles import validate_conductor_profile


class ConductorPreview(QWidget):
    """Show supplied copper and insulation in the support's local frame.

    Dimensions are millimetres. This schematic does not infer a physical shape
    for an unspecified profile and never writes to configuration or draft widgets.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("conductor_cross_section_preview")
        self.setAccessibleName("Conductor cross-section and insulation preview")
        self.setToolTip(
            "Width lies in the support tangent plane, transverse to the wire. Height follows the support normal. "
            "The schematic preserves aspect ratio and shows uniform insulation on each side."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.drawing = _CrossSection(self)
        layout.addWidget(self.drawing)
        self.summary_label = QLabel(self)
        self.summary_label.setObjectName("conductor_cross_section_summary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary_label)
        self.dimensions = None
        self.custom = False
        self.set_dimensions(None, None, None)

    def set_dimensions(self, width_mm, height_mm, insulation_mm, custom=False):
        self.custom = bool(custom)
        if self.custom:
            self.dimensions = None
            self.summary_label.setText("Custom profile · inspect the saved profile for shape and dimensions.")
            self.drawing.set_state(None, "Custom profile\nOpen the profile viewer")
            return
        try:
            if any(isinstance(value, bool) for value in (width_mm, height_mm, insulation_mm)):
                raise ValueError
            width, height, insulation = map(float, (width_mm, height_mm, insulation_mm))
            if not all(math.isfinite(value) for value in (width, height, insulation)):
                raise ValueError
            if width <= 0 or height <= 0 or insulation < 0:
                raise ValueError
            outside_width, outside_height = width + 2 * insulation, height + 2 * insulation
            if not all(math.isfinite(value) for value in (outside_width, outside_height, width * height)):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            self.dimensions = None
            self.summary_label.setText("Complete valid conductor dimensions to preview.")
            self.drawing.set_state(None, "Dimensions incomplete")
            return
        self.dimensions = (width, height, insulation)
        self.summary_label.setText(
            f"Cu {width:g} × {height:g} mm · area {width * height:g} mm²\n"
            f"Outside {outside_width:g} × {outside_height:g} mm · insulation {insulation:g} mm per side"
        )
        self.drawing.set_state(self.dimensions, "")

    def set_profile(self, points_m, insulation_mm):
        """Draw a validated polygon at its actual local coordinates."""
        try:
            profile = validate_conductor_profile(points_m)
            insulation = float(insulation_mm)
            if isinstance(insulation_mm, bool) or not math.isfinite(insulation) or insulation < 0:
                raise ValueError("Insulation must be finite and nonnegative.")
        except (ValueError, TypeError, OverflowError) as exc:
            self.set_invalid_profile(str(exc))
            return
        self.custom = True
        self.dimensions = None
        lo = np.asarray(profile.bounds_min_m) * 1000
        hi = np.asarray(profile.bounds_max_m) * 1000
        self.summary_label.setText(
            f"Polygon · {len(profile.points_m)} points · Cu area {profile.area_m2 * 1e6:g} mm²\n"
            f"u {lo[0]:g}…{hi[0]:g} mm · n {lo[1]:g}…{hi[1]:g} mm · insulation {insulation:g} mm per side"
        )
        self.drawing.set_profile(np.asarray(profile.points_m) * 1000, insulation)

    def set_invalid_profile(self, message):
        self.custom = True
        self.dimensions = None
        self.summary_label.setText("Invalid profile draft · " + message)
        self.drawing.set_state(None, "Correct the cross-section profile")


class _CrossSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._dimensions = None
        self._profile = None
        self._insulation = 0.0
        self._message = ""

    def sizeHint(self):
        return QSize(300, 190)

    def set_state(self, dimensions, message):
        self._dimensions, self._message = dimensions, message
        self._profile = None
        self.update()

    def set_profile(self, points_mm, insulation_mm):
        self._dimensions, self._message = None, ""
        self._profile = np.array(points_mm, copy=True)
        self._insulation = insulation_mm
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        text = palette.color(palette.ColorRole.Text)
        painter.setPen(text)
        if self._profile is not None:
            self._paint_profile(painter, text)
            return
        if self._dimensions is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return
        width, height, insulation = self._dimensions
        outside_width, outside_height = width + 2 * insulation, height + 2 * insulation
        available = QRectF(32, 24, max(40, self.width() - 98), max(40, self.height() - 84))
        scale = min(available.width() / outside_width, available.height() / outside_height)
        envelope = QRectF(0, 0, outside_width * scale, outside_height * scale)
        envelope.moveCenter(available.center())
        copper = QRectF(
            envelope.left() + insulation * scale,
            envelope.top() + insulation * scale,
            width * scale,
            height * scale,
        )
        painter.setBrush(QColor(245, 206, 112, 170))
        painter.setPen(QPen(palette.color(palette.ColorRole.Mid), 1.2))
        painter.drawRect(envelope)
        painter.setBrush(QColor(204, 119, 50))
        painter.setPen(QPen(QColor(129, 65, 20), 1.2))
        painter.drawRect(copper)
        painter.setPen(QColor(30, 24, 17))
        if copper.width() > 26 and copper.height() > 18:
            painter.drawText(copper, Qt.AlignmentFlag.AlignCenter, "Cu")
        painter.setPen(QPen(text, 1.0))
        normal_x, normal_y = 16.0, available.center().y()
        painter.drawLine(normal_x, normal_y + 18, normal_x, normal_y - 18)
        painter.drawLine(normal_x, normal_y - 18, normal_x - 4, normal_y - 12)
        painter.drawLine(normal_x, normal_y - 18, normal_x + 4, normal_y - 12)
        painter.drawText(QRectF(5, normal_y - 40, 22, 18), Qt.AlignmentFlag.AlignCenter, "n")
        x = envelope.right() + 12
        y = envelope.bottom() + 12
        painter.drawLine(copper.left(), y, copper.right(), y)
        painter.drawLine(copper.left(), y - 4, copper.left(), y + 4)
        painter.drawLine(copper.right(), y - 4, copper.right(), y + 4)
        painter.drawLine(x, copper.top(), x, copper.bottom())
        painter.drawLine(x - 4, copper.top(), x + 4, copper.top())
        painter.drawLine(x - 4, copper.bottom(), x + 4, copper.bottom())
        painter.drawText(
            QRectF(envelope.left() - 10, y + 3, envelope.width() + 20, 18),
            Qt.AlignmentFlag.AlignCenter,
            f"{width:g} mm",
        )
        painter.drawText(
            QRectF(x + 6, copper.center().y() - 10, 56, 24), Qt.AlignmentFlag.AlignVCenter, f"{height:g} mm"
        )
        painter.drawText(
            QRectF(0, self.height() - 20, self.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            "Width: surface plane · height: support normal",
        )

    def _paint_profile(self, painter, text):
        points, insulation = self._profile, self._insulation
        lower = np.minimum(points.min(axis=0) - insulation, 0)
        upper = np.maximum(points.max(axis=0) + insulation, 0)
        span = upper - lower
        available = QRectF(35, 22, max(40, self.width() - 70), max(40, self.height() - 65))
        scale = min(available.width() / max(span[0], 1e-12), available.height() / max(span[1], 1e-12))
        center = (lower + upper) / 2

        def point(x, y):
            return QPointF(
                available.center().x() + (x - center[0]) * scale, available.center().y() - (y - center[1]) * scale
            )

        origin = point(0, 0)
        painter.setPen(QPen(self.palette().color(self.palette().ColorRole.Mid), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(available.left(), origin.y()), QPointF(available.right(), origin.y()))
        painter.drawLine(QPointF(origin.x(), available.top()), QPointF(origin.x(), available.bottom()))
        path = QPainterPath()
        path.moveTo(point(*points[0]))
        for vertex in points[1:]:
            path.lineTo(point(*vertex))
        path.closeSubpath()
        if insulation > 0:
            stroker = QPainterPathStroker()
            stroker.setWidth(2 * insulation * scale)
            stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.fillPath(stroker.createStroke(path), QColor(100, 163, 205, 120))
        painter.fillPath(path, QColor(204, 119, 50))
        painter.setPen(QPen(QColor(129, 65, 20), 1.2))
        painter.drawPath(path)
        painter.setPen(QPen(text, 1))
        painter.setBrush(text)
        painter.drawEllipse(origin, 2, 2)
        painter.drawText(origin + QPointF(4, -4), "0")
        painter.drawText(QPointF(available.right() + 4, origin.y() + 4), "u")
        painter.drawText(QPointF(origin.x() + 4, available.top() - 6), "n")
        painter.drawText(
            QRectF(0, self.height() - 22, self.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            "u: support transverse · n: surface normal [mm]",
        )
