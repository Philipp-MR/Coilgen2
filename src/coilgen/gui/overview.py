"""Compact, nonmodal guidance and an optional physical model overview."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .colors import apply_tone, tone_color


_STATUS_TEXT = {
    "defined": "Defined",
    "needs_input": "Needs input",
    "pending": "Pending edits",
    "not_computed": "Not computed",
    "computed": "Computed",
    "running": "Running",
    "stale": "Stale",
    "failed": "Failed",
    "passed": "Passed in scope",
    "unverified": "Unverified",
    "available": "Available",
    "not_run": "Not run",
}
_MARKERS = {"neutral": "○", "info": "●", "success": "✓", "warning": "△", "error": "!"}
_SUMMARY_KEYS = {
    "target": ("target", "roi"),
    "support": ("support",),
    "conductor": ("conductor",),
    "solve": ("study",),
}


def _plain_label(parent, name):
    label = QLabel(parent)
    label.setObjectName(name)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class DesignOverview(QWidget):
    """Display supplied evidence only; navigation never changes model settings."""

    action_requested = Signal(str)
    section_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("design_overview")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._state = None
        self._summaries = {}
        self._restyling = False
        self.stage_buttons, self.status_labels, self.summary_labels = {}, {}, {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.strip = QFrame(self)
        self.strip.setObjectName("design_guidance_strip")
        strip_layout = QVBoxLayout(self.strip)
        strip_layout.setContentsMargins(10, 6, 10, 6)
        strip_layout.setSpacing(3)
        header = QHBoxLayout()
        header.setSpacing(10)
        self.title = _plain_label(self.strip, "design_guidance_title")
        title_font = self.title.font()
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.detail = _plain_label(self.strip, "design_guidance_detail")
        header.addWidget(self.title, 1)
        self.action_button = QPushButton(self.strip)
        self.action_button.setObjectName("design_guidance_action")
        self.action_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.action_button.setMinimumHeight(30)
        self.action_button.clicked.connect(self._request_action)
        header.addWidget(self.action_button, 0, Qt.AlignmentFlag.AlignVCenter)
        strip_layout.addLayout(header)
        strip_layout.addWidget(self.detail)
        layout.addWidget(self.strip)

        self.expand_button = QToolButton(self)
        self.expand_button.setObjectName("design_overview_expand")
        self.expand_button.setText("Model overview")
        self.expand_button.setAccessibleName("Show physical model overview")
        self.expand_button.setToolTip("Inspect the target, support, conductor, solve, winding, checks and exports.")
        self.expand_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.expand_button.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_button.setCheckable(True)
        self.expand_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.expand_button.toggled.connect(self._set_expanded)
        layout.addWidget(self.expand_button, 0, Qt.AlignmentFlag.AlignLeft)

        self.details = QScrollArea(self)
        self.details.setObjectName("design_overview_details")
        self.details.setFrameShape(QFrame.Shape.NoFrame)
        self.details.setWidgetResizable(True)
        self.details.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.details.setMaximumHeight(260)
        self.details.setMinimumHeight(100)
        self.details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._sections = QWidget(self.details)
        self._sections_layout = QVBoxLayout(self._sections)
        self._sections_layout.setContentsMargins(6, 0, 8, 4)
        self._sections_layout.setSpacing(8)
        self.details.setWidget(self._sections)
        self.details.hide()
        layout.addWidget(self.details)
        self.hide()

    def set_state(self, state, summaries=None):
        """Synchronize guidance without emitting navigation or compute requests."""
        self._state = state
        self._summaries = dict(summaries or {})
        marker = _MARKERS.get(state.tone, _MARKERS["neutral"])
        self.title.setText(f"{marker} {state.title}")
        self.title.setAccessibleName(state.title)
        self.detail.setText(state.detail)
        self.detail.setToolTip(state.detail)
        self.detail.setVisible(bool(state.detail))
        self.action_button.setText(state.action_label)
        self.action_button.setAccessibleName(state.action_label)
        self.action_button.setAccessibleDescription(state.detail)
        self.action_button.setToolTip(state.detail)
        self.action_button.setEnabled(bool(state.action_key))
        self.action_button.setVisible(bool(state.action_label))

        keys = tuple(stage.key for stage in state.stages)
        if keys != tuple(self.stage_buttons):
            self._rebuild_sections(state.stages)
        for stage in state.stages:
            tone = getattr(stage, "tone", "neutral")
            self.stage_buttons[stage.key].setText(stage.label)
            self.stage_buttons[stage.key].setAccessibleName(f"Open {stage.label}")
            self.stage_buttons[stage.key].setToolTip(stage.detail)
            text = _STATUS_TEXT.get(stage.status, str(stage.status).replace("_", " ").capitalize())
            marker = _MARKERS.get(tone, _MARKERS["neutral"])
            status = self.status_labels[stage.key]
            status.setText(f"{marker} {text}")
            status.setAccessibleName(f"{stage.label}: {text}")
            status.setToolTip(stage.detail)
            summary_parts = tuple(self._summaries.get(key, "") for key in _SUMMARY_KEYS.get(stage.key, ()))
            summary = " · ".join(str(part) for part in summary_parts if part) or stage.detail
            self.summary_labels[stage.key].setText(summary)
            self.summary_labels[stage.key].setVisible(bool(summary))
        self.expand_button.setVisible(bool(state.stages))
        if not state.stages:
            self.expand_button.setChecked(False)
            self.details.hide()
        self._apply_colors()
        self.show()

    def _rebuild_sections(self, stages):
        while self._sections_layout.count():
            item = self._sections_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.stage_buttons.clear()
        self.status_labels.clear()
        self.summary_labels.clear()
        for stage in stages:
            section = QWidget(self._sections)
            layout = QVBoxLayout(section)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            header = QHBoxLayout()
            button = QToolButton(section)
            button.setObjectName(f"overview_section_{stage.key}")
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            button.clicked.connect(lambda _checked=False, key=stage.key: self.section_requested.emit(key))
            header.addWidget(button)
            header.addStretch(1)
            status = _plain_label(section, f"overview_status_{stage.key}")
            status.setWordWrap(False)
            header.addWidget(status)
            layout.addLayout(header)
            summary = _plain_label(section, f"overview_summary_{stage.key}")
            layout.addWidget(summary)
            self.stage_buttons[stage.key] = button
            self.status_labels[stage.key] = status
            self.summary_labels[stage.key] = summary
            self._sections_layout.addWidget(section)
        self._sections_layout.addStretch(1)

    def _request_action(self):
        if self._state is not None and self._state.action_key:
            self.action_requested.emit(self._state.action_key)

    def _set_expanded(self, expanded):
        self.details.setVisible(expanded)
        self.expand_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.expand_button.setAccessibleName(
            "Hide physical model overview" if expanded else "Show physical model overview"
        )

    def _apply_colors(self):
        if self._restyling or self._state is None:
            return
        self._restyling = True
        try:
            palette = self.palette()
            border = tone_color(self._state.tone, palette).name()
            background = palette.color(QPalette.ColorRole.Window).name()
            self.strip.setStyleSheet(
                f"QFrame#design_guidance_strip {{ background: {background}; border: 1px solid {border};"
                f"border-left: 4px solid {border}; border-radius: 4px; }}"
            )
            apply_tone(self.title, self._state.tone)
            for stage in self._state.stages:
                apply_tone(self.status_labels[stage.key], getattr(stage, "tone", "neutral"))
        finally:
            self._restyling = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "_state") and not self._restyling:
                QTimer.singleShot(0, self, self._apply_colors)
