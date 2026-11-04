"""Repository-owned scientific pictograms rendered against the current Qt palette.

Icons describe model objects and operations. A status badge describes recorded
evidence; the conductor and export pictograms alone make no readiness claim.
The shared SVG paths have no font or package-data dependencies.
Labels and accessible names belong to the controls.
"""

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from .colors import tone_color
from .pictograms import (
    _DRAWINGS,
    svg_bytes,
    ICON_NAMES as ICON_NAMES,
    WORKFLOW_ICONS as WORKFLOW_ICONS,
    INSPECTION_ICONS as INSPECTION_ICONS,
    WINDING_ICONS as WINDING_ICONS,
    SUPPORT_ICONS as SUPPORT_ICONS,
    REGION_ICONS as REGION_ICONS,
    TARGET_MODE_ICONS as TARGET_MODE_ICONS,
    PANEL_ICONS as PANEL_ICONS,
)


# The badge is a separate assertion, never an intrinsic property of a shape.
_BADGES = {
    "passed": ("success", '<path d="m16 18.5 1.6 1.6 3.4-3.6"/>'),
    "failed": ("error", '<path d="m16.5 16.5 4 4m0-4-4 4"/>'),
    "unverified": ("warning", '<path d="M17 17a1.5 1.5 0 0 1 3 0c0 1.3-1.5 1-1.5 2"/><path d="M18.5 20.7v.1"/>'),
    "pending": ("info", '<path d="M18.5 15.7v2.8l1.8 1"/>'),
    "stale": ("warning", '<path d="M16.1 18.5a2.5 2.5 0 1 1 1.1 2M16 16.5v2h2"/>'),
    "baseline": ("info", '<path d="M17 15.7h3l-.5 2 1 1h-4l1-1zM18.5 18.7v2.6"/>'),
    "evaluated": ("info", '<path d="M16 18.5h5M18.5 16v5"/>'),
    "skipped": ("neutral", '<path d="M16 18.5h5"/>'),
}
STATUS_BADGES = tuple(_BADGES)


def svg_data(name: str, *, badge: str | None = None, mode=QIcon.Mode.Normal, foreground="default") -> bytes:
    """Return the current theme's SVG, useful for previews and render validation."""
    if name not in _DRAWINGS:
        raise ValueError(f"Unknown CoilGen icon: {name!r}")
    if badge is not None and badge not in _BADGES:
        raise ValueError(f"Unknown CoilGen icon badge: {badge!r}")
    if foreground not in ("default", "primary"):
        raise ValueError(f"Unknown CoilGen icon foreground: {foreground!r}")
    palette = QApplication.palette()
    disabled = mode == QIcon.Mode.Disabled
    selected = mode == QIcon.Mode.Selected
    group = QPalette.ColorGroup.Disabled if disabled else QPalette.ColorGroup.Active
    role = QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.ButtonText
    # The workbench primary-action stylesheet has a fixed #075aaa background.
    # Its explicit light foreground is independent of the OS highlight palette.
    primary = foreground == "primary" and not disabled and not selected
    color = "#ffffff" if primary else palette.color(group, role).name()
    drawing = _DRAWINGS[name]
    if badge is not None:
        tone, glyph = _BADGES[badge]
        badge_color = color if disabled or selected or primary else tone_color(tone, palette).name()
        backing_role = QPalette.ColorRole.Highlight if selected else QPalette.ColorRole.Window
        backing = "#075aaa" if primary else palette.color(group, backing_role).name()
        drawing += (
            f'<g stroke="{backing}" stroke-width="2.8" fill="{backing}">'
            '<circle cx="18.5" cy="18.5" r="4.2"/></g>'
            f'<g stroke="{badge_color}" stroke-width="1.2" fill="none">'
            f'<circle cx="18.5" cy="18.5" r="4.2"/>{glyph}</g>'
        )
    return svg_bytes(drawing, color)


class _ScientificIconEngine(QIconEngine):
    """Render vectors on demand so existing QIcons follow palette changes."""

    def __init__(self, name: str, badge: str | None = None, foreground="default"):
        super().__init__()
        self.name = name
        self.badge = badge
        self.foreground = foreground
        self._svg = None
        self._renderer = None

    def clone(self):
        return _ScientificIconEngine(self.name, self.badge, self.foreground)

    def key(self):
        return "CoilGenScientificIcon"

    def paint(self, painter, rect, mode, state):
        data = svg_data(self.name, badge=self.badge, mode=mode, foreground=self.foreground)
        if data != self._svg:
            self._svg = data
            self._renderer = QSvgRenderer(QByteArray(data))
        # Keep the drawing square even when the requested button rectangle is not.
        bounds = QRectF(rect)
        side = min(bounds.width(), bounds.height())
        square = QRectF(0, 0, side, side)
        square.moveCenter(bounds.center())
        self._renderer.render(painter, square)

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size, mode, state, scale):
        if size.isEmpty() or scale <= 0:
            return QPixmap()
        physical = QSize(round(size.width() * scale), round(size.height() * scale))
        result = QPixmap(physical)
        result.fill(Qt.GlobalColor.transparent)
        result.setDevicePixelRatio(scale)
        painter = QPainter(result)
        self.paint(painter, QRectF(0, 0, size.width(), size.height()), mode, state)
        painter.end()
        return result


def icon(name: str, *, badge: str | None = None, foreground="default") -> QIcon:
    """Create a theme-aware vector icon; unknown names/states fail explicitly."""
    if name not in _DRAWINGS:
        raise ValueError(f"Unknown CoilGen icon: {name!r}")
    if badge is not None and badge not in _BADGES:
        raise ValueError(f"Unknown CoilGen icon badge: {badge!r}")
    if foreground not in ("default", "primary"):
        raise ValueError(f"Unknown CoilGen icon foreground: {foreground!r}")
    return QIcon(_ScientificIconEngine(name, badge, foreground))
