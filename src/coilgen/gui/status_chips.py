"""Plain-language readiness for the model tree: status chips, banner and legend.

The chips say what the user should do: "Ready", "2 tips" (worth a look, never
blocking), "Needs input" (fix before computing) and "Add" (an optional feature
that is switched off). Every state also has its own symbol, so colour is never
the only cue.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .colors import tone_color

# 24-unit outline symbols per section state.
_SYMBOLS = {
    "set": '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>',
    "warning": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="7.5" r=".8"/>',
    "blocking": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6m0-6-6 6"/>',
    "off": '<circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/>',
    "run": '<circle cx="12" cy="12" r="9"/><path d="m10 8.5 5 3.5-5 3.5z"/>',
}
STATE_TONES = {"set": "success", "warning": "warning", "blocking": "error", "off": "info", "run": "neutral"}
LEGEND = (
    ("set", "Ready", "nothing missing"),
    ("warning", "Tip", "worth a look, never blocks"),
    ("blocking", "Needs input", "blocks computing"),
)


def status_icon(state, palette, size=16):
    """Coloured state symbol drawn against the current palette."""
    color = tone_color(STATE_TONES.get(state, "neutral"), palette).name()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
        f'<g fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{_SYMBOLS.get(state, _SYMBOLS['run'])}</g></svg>"
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("ascii")))
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2.0)
    return QIcon(pixmap)


def chip_background(state, palette):
    """Very light tint behind a status cell; the symbol and text carry the meaning."""
    if state not in {"set", "warning", "blocking"}:
        return None
    color = QColor(tone_color(STATE_TONES[state], palette))
    color.setAlpha(28)
    return color


class ReadinessBanner(QFrame):
    """One line above the model tree: what blocks computing, or that it's ready."""

    action_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("readiness_banner")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        self.symbol = QLabel(self)
        self.symbol.setFixedSize(22, 22)
        layout.addWidget(self.symbol, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.title = QLabel(self)
        self.title.setObjectName("readiness_banner_title")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600")
        self.detail = QLabel(self)
        self.detail.setObjectName("readiness_banner_detail")
        self.detail.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.detail)
        layout.addLayout(text, 1)
        self.button = QPushButton(self)
        self.button.setObjectName("readiness_banner_action")
        self.button.clicked.connect(self.action_requested.emit)
        layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.state = ""
        self.hide()

    def set_readiness(self, overall, labels, *, pending_edits=False, palette=None):
        """Show ``OverallReadiness``; ``labels`` maps section nodes to tree names.

        ``palette`` should be the main window's palette: colours are chosen for
        contrast against it, like the tree's status cells.
        """
        if pending_edits:
            state, title = "warning", "Edits not applied yet"
            detail, action = "Apply them to check the inputs again.", "Apply"
        else:
            state = {"blocking": "blocking", "tips": "set", "ready": "set"}[overall.state]
            title, detail, action = overall.title(), overall.detail(labels), overall.action
        self.state = "pending" if pending_edits else overall.state
        tone = STATE_TONES[state]
        palette = palette or QApplication.palette()
        accent = QColor(tone_color(tone, palette))
        tint = QColor(accent)
        tint.setAlpha(30)
        self.setStyleSheet(
            "#readiness_banner { background: rgba(%d, %d, %d, %d); border: 1px solid %s; border-radius: 6px; }"
            % (tint.red(), tint.green(), tint.blue(), tint.alpha(), accent.name())
        )
        self.symbol.setPixmap(status_icon(state, palette, 20).pixmap(20, 20))
        self.title.setText(title)
        self.detail.setText(detail)
        self.title.setStyleSheet(f"font-weight: 600; color: {accent.name()}")
        self.button.setText(action)
        self.button.setVisible(bool(action))
        self.setToolTip(
            "Computing needs every section to be Ready or to have only tips. "
            "Tips never block computing. Engineering checks run after computation."
        )
        self.show()


def legend_widget(parent=None):
    """Compact key under the tree explaining the three status chips."""
    widget = QWidget(parent)
    widget.setObjectName("readiness_legend")
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(6, 2, 6, 4)
    layout.setSpacing(10)
    palette = QApplication.palette()
    for state, name, meaning in LEGEND:
        item = QLabel(widget)
        item.setPixmap(status_icon(state, palette, 14).pixmap(14, 14))
        layout.addWidget(item)
        label = QLabel(name, widget)
        label.setToolTip(f"{name}: {meaning}")
        layout.addWidget(label)
    layout.addStretch(1)
    widget.setToolTip(" · ".join(f"{name}: {meaning}" for _state, name, meaning in LEGEND))
    return widget
