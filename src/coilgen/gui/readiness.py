"""Qt rendering of the shared scientific preflight report."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ..application.preflight import PreflightIssue as ReadinessIssue
from ..application.preflight import PreflightReport as ReadinessReport
from ..application.preflight import check_preflight, settings_node
from ..application.service import DesignService
from .colors import apply_tone, tone_color
from .icons import icon


__all__ = ["ReadinessIssue", "ReadinessReport", "ReadinessPanel", "check_readiness", "settings_node"]


def check_readiness(project, *, service=DesignService, pipeline_stage=None) -> ReadinessReport:
    """Keep the desktop adapter's injectable validator and public imports."""
    return check_preflight(
        project,
        validator=getattr(service, "validate", DesignService.validate),
        pipeline_stage=pipeline_stage,
    )


def _readiness_icon(issue):
    """Use the actual setting key; ambiguous conditions retain a generic bound."""
    key = issue.key.lower()
    if key == "cross_sectional_points":
        return "conductor-profile"
    if "bend_radius" in key:
        return "bend-radius"
    if "clearance" in key or key in {"min_wire_distance_m", "min_turn_distance"}:
        return "wire-spacing"
    if "penetration" in key or "intersection" in key:
        return "collision"
    if key.startswith(("target_error_", "layout_target_error_")):
        return "field-error"
    if "current_density" in key:
        return "current-density"
    if key.startswith("layout_terminal"):
        return "terminals"
    return "constraint"


class ReadinessPanel(QWidget):
    """Nonmodal action list; warnings remain visible without blocking Compute."""

    issue_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("compute_readiness")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.issues = ()
        self._issue_labels = []
        self._issue_icons = []
        self._updating_colors = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.heading = QLabel()
        self.heading.setObjectName("compute_readiness_heading")
        self.heading.setWordWrap(True)
        font = self.heading.font()
        font.setBold(True)
        self.heading.setFont(font)
        layout.addWidget(self.heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        content = QWidget()
        self.rows = QVBoxLayout(content)
        self.rows.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        layout.addStretch(1)
        self.hide()

    def set_issues(self, issues):
        self.issues = tuple(issues)
        self._issue_labels.clear()
        self._issue_icons.clear()
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        blocking = sum(issue.blocking for issue in self.issues)
        self.heading.setText(
            "Cannot compute yet — resolve the item below."
            if blocking == 1
            else f"Cannot compute yet — resolve these {blocking} items."
            if blocking
            else "Modeling notes — Compute is available."
        )
        for issue in self.issues:
            row = QWidget()
            layout = QHBoxLayout(row)
            symbol = QLabel()
            symbol.setFixedSize(24, 24)
            symbol.setAccessibleName(issue.title)
            symbol.setAccessibleDescription("Blocking input error" if issue.blocking else "Unverified modeling note")
            symbol.setToolTip(f"{issue.severity.capitalize()}: {issue.key}")
            self._issue_icons.append(symbol)
            layout.addWidget(symbol, 0, Qt.AlignmentFlag.AlignTop)
            text = QLabel(f"{issue.title}: {issue.message}\n{issue.action}")
            text.setTextFormat(Qt.TextFormat.PlainText)
            text.setWordWrap(True)
            self._issue_labels.append(text)
            layout.addWidget(text, 1)
            button = QPushButton("Open settings")
            button.setIcon(icon("settings"))
            button.setIconSize(QSize(18, 18))
            button.setAccessibleName(f"Open settings: {issue.title}")
            button.setAccessibleDescription(issue.action)
            button.clicked.connect(lambda _checked=False, item=issue: self.issue_requested.emit(item))
            layout.addWidget(button)
            self.rows.addWidget(row)
        self._refresh_colors()
        self.setVisible(bool(self.issues))

    def _refresh_colors(self):
        if self._updating_colors:
            return
        self._updating_colors = True
        try:
            tone = "error" if any(issue.blocking for issue in self.issues) else "warning" if self.issues else "neutral"
            palette = self.palette()
            background = palette.color(QPalette.ColorRole.Window).name()
            border = tone_color(tone, palette).name()
            self.setStyleSheet(
                f"QWidget#compute_readiness {{ background: {background}; border: 1px solid {border}; border-radius: 5px; }}"
            )
            apply_tone(self.heading, tone)
            for issue, label in zip(self.issues, self._issue_labels):
                apply_tone(label, issue.severity)
            for issue, symbol in zip(self.issues, self._issue_icons):
                badge = "failed" if issue.blocking else "unverified"
                symbol.setProperty("evidenceBadge", badge)
                symbol.setProperty("scientificIcon", _readiness_icon(issue))
                symbol.setPixmap(
                    icon(_readiness_icon(issue), badge=badge).pixmap(QSize(24, 24), symbol.devicePixelRatioF())
                )
        finally:
            self._updating_colors = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "heading") and not self._updating_colors:
                QTimer.singleShot(0, self, self._refresh_colors)
