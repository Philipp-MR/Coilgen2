"""Prominent saved field errors with explicit current and sampling scope.

This panel reads immutable scalar summaries and array names. It neither loads
field arrays nor calculates residuals, current fits or engineering acceptance.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics, QPalette
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ..application.cad import PRINT_FORMER_EXPORT_HINT, has_saved_print_former, print_former_status
from ..application.routing_analysis import routing_analysis_available
from .colors import apply_tone, tone_color
from .coil_statistics import build_coil_statistics
from .icons import icon
from .run_metadata import read_run_metadata
from .result_summary import (
    FieldResultSummary,
    coil_metric_cards,
    drive_metric_cards,
    field_metric_cards,
    summarize_field_result,
)


class _Heading(QLabel):
    def set_heading(self, text):
        self.full_text = text
        self.setToolTip(text)
        self._elide()

    def _elide(self):
        self.setText(
            QFontMetrics(self.font()).elidedText(
                getattr(self, "full_text", ""), Qt.TextElideMode.ElideRight, max(0, self.width())
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


class _Value(QLabel):
    def set_value(self, text, *, unavailable=False, point_size=16.0):
        self._preferred_size = 11.0 if unavailable else point_size
        self.setText(text)
        self._fit()

    def _fit(self):
        font = self.font()
        size = getattr(self, "_preferred_size", 16.0)
        font.setPointSizeF(size)
        while size > 10.0 and QFontMetrics(font).horizontalAdvance(self.text()) > self.width():
            size -= 0.5
            font.setPointSizeF(size)
        self.setFont(font)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class ResultsOverviewPanel(QFrame):
    """Compact run-bound error board, with explicit analysis navigation."""

    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is not None:
            parent.installEventFilter(self)
        self.setObjectName("results_overview_panel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.record, self.summary, self._cache_key = None, FieldResultSummary(), None
        self.coil_report = build_coil_statistics()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(4)
        header = QHBoxLayout()
        self.heading = _Heading(self)
        self.heading.setTextFormat(Qt.TextFormat.PlainText)
        self.heading.setMinimumWidth(0)
        self.heading.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        font = self.heading.font()
        font.setBold(True)
        self.heading.setFont(font)
        self.heading.setAccessibleName("Saved field result identity")
        header.addWidget(self.heading, 1)
        self.badge = QLabel(self)
        self.badge.setTextFormat(Qt.TextFormat.PlainText)
        self.badge.setAccessibleName("Saved result acceptance and freshness")
        font = self.badge.font()
        font.setBold(True)
        self.badge.setFont(font)
        header.addWidget(self.badge)
        self.routing_button = QPushButton(icon("connected-winding"), "Interconnections", self)
        self.routing_button.setObjectName("results_overview_routing")
        self.routing_button.setAccessibleName("Inspect saved interconnections")
        self.routing_button.setMinimumHeight(28)
        self.routing_button.clicked.connect(lambda _checked=False: self.action_requested.emit("routing"))
        header.addWidget(self.routing_button)
        self.metrics_button = QPushButton("Hide metrics", self)
        self.metrics_button.setObjectName("results_overview_toggle_metrics")
        self.metrics_button.setMinimumHeight(28)
        self.metrics_button.setToolTip("Collapse or expand the recorded field and coil metrics.")
        self.metrics_button.setCheckable(True)
        self.metrics_button.setChecked(True)
        self.metrics_button.toggled.connect(self._show_metrics)
        header.addWidget(self.metrics_button)
        layout.addLayout(header)
        self.metric_details = QWidget(self)
        metrics = QVBoxLayout(self.metric_details)
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setSpacing(4)
        layout.addWidget(self.metric_details)
        # Headline: the current that produces the requested field, and the
        # field-shape error that remains at that current.
        self.drive_cards = QFrame(self)
        drive_row = QHBoxLayout(self.drive_cards)
        drive_row.setContentsMargins(0, 0, 0, 0)
        drive_row.setSpacing(7)
        self.drive_labels, self.drive_values = {}, {}
        for key, text in (
            ("drive", "Drive current"),
            ("shape", "Shape error"),
            ("gain", "Field at design current"),
        ):
            card = QFrame(self)
            card.setObjectName("drive_result_" + key)
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setMinimumWidth(0)
            card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 4, 8, 4)
            card_layout.setSpacing(0)
            caption, value = QLabel(text, card), _Value(card)
            for widget in (caption, value):
                widget.setTextFormat(Qt.TextFormat.PlainText)
                widget.setMinimumWidth(0)
            font = value.font()
            font.setPointSizeF(max(16.0, font.pointSizeF() + 4))
            font.setBold(True)
            value.setFont(font)
            value.setAccessibleName(text)
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            drive_row.addWidget(card, 1)
            self.drive_labels[key], self.drive_values[key] = caption, value
        metrics.addWidget(self.drive_cards)
        self.field_cards = QFrame(self)
        cards = QHBoxLayout(self.field_cards)
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(7)
        self.metric_labels, self.metric_values = {}, {}
        for key, text in (
            ("rms", "RMS error"),
            ("peak", "Peak error"),
            ("relative", "Relative RMS error"),
            ("current", "Original current"),
        ):
            card = QFrame(self)
            card.setObjectName("field_result_" + key)
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setMinimumWidth(0)
            card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 4, 8, 4)
            card_layout.setSpacing(0)
            caption, value = QLabel(text, card), _Value(card)
            for widget in (caption, value):
                widget.setTextFormat(Qt.TextFormat.PlainText)
                widget.setMinimumWidth(0)
            font = value.font()
            font.setPointSizeF(max(16.0, font.pointSizeF() + 4))
            font.setBold(True)
            value.setFont(font)
            value.setAccessibleName(text)
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            cards.addWidget(card, 1)
            self.metric_labels[key], self.metric_values[key] = caption, value
        metrics.addWidget(self.field_cards)
        self.context = QLabel(self)
        self.context.setTextFormat(Qt.TextFormat.PlainText)
        self.context.setWordWrap(True)
        self.context.setMinimumWidth(0)
        self.context.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.context.setAccessibleName("Field source, current and sampling scope")
        metrics.addWidget(self.context)
        properties = QHBoxLayout()
        properties.setSpacing(7)
        self.coil_metric_labels, self.coil_metric_values = {}, {}
        for key, text in (
            ("resistance", "Resistance"),
            ("inductance", "Self inductance"),
            ("mass", "Conductor mass"),
            ("volume", "Conductor volume"),
        ):
            card = QFrame(self)
            card.setObjectName("coil_result_" + key)
            card.setMinimumWidth(0)
            card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 3, 8, 3)
            card_layout.setSpacing(0)
            caption, value = QLabel(text, card), _Value(card)
            for widget in (caption, value):
                widget.setTextFormat(Qt.TextFormat.PlainText)
                widget.setMinimumWidth(0)
            font = value.font()
            font.setBold(True)
            value.setFont(font)
            value.setAccessibleName(text)
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            properties.addWidget(card, 1)
            self.coil_metric_labels[key], self.coil_metric_values[key] = caption, value
        metrics.addLayout(properties)
        self.coil_scope = QLabel(self)
        self.coil_scope.setTextFormat(Qt.TextFormat.PlainText)
        self.coil_scope.setWordWrap(True)
        self.coil_scope.setMinimumWidth(0)
        self.coil_scope.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.coil_scope.setAccessibleName("Saved electrical, conductor material and volume scope")
        metrics.addWidget(self.coil_scope)
        self.former_scope = QLabel(self)
        self.former_scope.setObjectName("results_overview_former_status")
        self.former_scope.setTextFormat(Qt.TextFormat.PlainText)
        self.former_scope.setWordWrap(True)
        self.former_scope.setMinimumWidth(0)
        actions = QHBoxLayout()
        actions.setSpacing(5)
        self.buttons = {}
        for key, text, drawing in (
            ("field", "Field map", "field-error"),
            ("coil", "Coil stats", "conductor-profile"),
            ("checks", "Engineering checks", "check"),
        ):
            button = QPushButton(icon(drawing), text, self)
            button.setObjectName("results_overview_" + key)
            button.setMinimumHeight(32)
            button.setAccessibleName("Open " + text.lower() + " for the selected run")
            button.clicked.connect(lambda _checked=False, action=key: self.action_requested.emit(action))
            actions.addWidget(button, 1)
            self.buttons[key] = button
        layout.addLayout(actions)
        self.former_button = QPushButton(icon("export-conductor"), "Export print former…", self)
        self.former_button.setObjectName("results_overview_export_former")
        self.former_button.setMinimumHeight(32)
        self.former_button.setToolTip("Export the selected run's saved former STL for inspection.")
        self.former_button.clicked.connect(lambda _checked=False: self.action_requested.emit("export_former"))
        self.former_plot_button = QPushButton(icon("mesh"), "Show print former", self)
        self.former_plot_button.setObjectName("results_overview_show_former")
        self.former_plot_button.setMinimumHeight(32)
        self.former_plot_button.setToolTip("Show the selected run's saved grooved former geometry.")
        self.former_plot_button.clicked.connect(lambda _checked=False: self.action_requested.emit("show_former"))
        former_actions = QHBoxLayout()
        former_actions.addWidget(self.former_scope, 1)
        former_actions.addWidget(self.former_plot_button)
        former_actions.addWidget(self.former_button)
        layout.addLayout(former_actions)
        self._style_panel()
        self.hide()

    def _show_metrics(self, expanded):
        self.metrics_button.setText("Hide metrics" if expanded else "Show metrics")
        self.metric_details.setVisible(expanded)

    def _style_panel(self):
        parent = self.parentWidget()
        palette = parent.palette() if parent is not None else self.palette()
        outline = palette.color(QPalette.ColorRole.Mid).name()
        background = palette.color(QPalette.ColorRole.Base).name()
        foreground = palette.color(QPalette.ColorRole.WindowText).name()
        accent = tone_color("info", palette).name()
        self.setStyleSheet(
            f"QFrame#results_overview_panel {{ border: 1px solid {outline}; border-radius: 7px; }}"
            f"QFrame#field_result_rms, QFrame#field_result_peak, QFrame#field_result_relative, "
            f"QFrame#field_result_current, QFrame#coil_result_resistance, QFrame#coil_result_inductance, "
            f"QFrame#coil_result_mass, QFrame#coil_result_volume, QFrame#drive_result_drive, "
            f"QFrame#drive_result_shape, QFrame#drive_result_gain {{ background: {background}; border: 1px solid {outline}; "
            "border-radius: 5px; }"
            f"QFrame#field_result_rms, QFrame#drive_result_drive {{ border-left: 3px solid {accent}; }}"
            f"QFrame#results_overview_panel QLabel {{ color: {foreground}; }}"
        )
        if self.summary.title:
            color = tone_color(self.summary.tone, palette).name()
            self.badge.setStyleSheet(
                f"color: {color}; border: 1px solid {color}; border-radius: 5px; padding: 2px 7px;"
            )
            badge_palette = QPalette(self.badge.palette())
            badge_palette.setColor(QPalette.ColorRole.WindowText, tone_color(self.summary.tone, palette))
            self.badge.setPalette(badge_palette)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Type.PaletteChange and hasattr(self, "buttons"):
            self._style_panel()
        return super().eventFilter(watched, event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "buttons"):
            self._style_panel()

    @property
    def source_key(self):
        return self.summary.source_key

    @property
    def component(self):
        return self.summary.component

    @property
    def has_coil_statistics(self):
        return self.coil_report.has_values

    def _set_coil_statistics(self):
        cards, scope, tooltip = coil_metric_cards(self.coil_report)
        for card in cards:
            key = card["key"]
            value = self.coil_metric_values[key]
            value.set_value(card["value"], unavailable=card["unavailable"], point_size=13.0)
            value.setToolTip(card["tooltip"])
            self.coil_metric_labels[key].setToolTip(card["tooltip"])
        self.coil_scope.setText(scope)
        self.coil_scope.setToolTip(tooltip)

    def set_record(self, record, *, source_error="", stale=False, former_plot_available=False):
        key = (id(record), str(source_error), bool(stale), bool(former_plot_available))
        if key == self._cache_key:
            return
        self._cache_key, self.record = key, record
        self.summary = summary = summarize_field_result(record, source_error=source_error, stale=stale)
        self.coil_report = build_coil_statistics(record)
        routing_available = routing_analysis_available(record)
        metadata = read_run_metadata(record)
        former_status = print_former_status(metadata)
        former_available = has_saved_print_former(metadata)
        visible = record is not None and (
            summary.visible
            or self.has_coil_statistics
            or routing_available
            or bool(former_status)
            or former_plot_available
        )
        self.setVisible(visible)
        if not visible:
            return
        self.field_cards.setVisible(summary.visible)
        self.heading.set_heading(summary.title)
        self.badge.setText(summary.badge)
        self.badge.setToolTip(summary.tooltip)
        apply_tone(self.badge, summary.tone)
        self._style_panel()
        self.context.setText(summary.context)
        self.context.setToolTip(summary.tooltip)
        self.setToolTip(summary.tooltip)
        self._set_coil_statistics()
        self.former_scope.setVisible(bool(former_status or former_available or former_plot_available))
        self.former_scope.setText(
            (former_status or "Print former") + (f". {PRINT_FORMER_EXPORT_HINT}" if not former_available else "")
        )
        self.former_button.setVisible(former_available)
        self.former_button.setEnabled(former_available and not source_error)
        self.former_plot_button.setVisible(former_plot_available)
        self.former_plot_button.setEnabled(former_plot_available and not source_error)
        drive = drive_metric_cards(summary)
        self.drive_cards.setVisible(summary.visible and any(not card["unavailable"] for card in drive))
        for card in drive:
            value = self.drive_values[card["key"]]
            value.set_value(card["value"], unavailable=card["unavailable"])
            value.setToolTip(card["tooltip"])
            self.drive_labels[card["key"]].setText(card["name"])
            self.drive_labels[card["key"]].setToolTip(card["tooltip"])
        for card in field_metric_cards(summary):
            value = self.metric_values[card["key"]]
            value.set_value(card["value"], unavailable=card["unavailable"])
            value.setToolTip(summary.tooltip)
        self.metric_labels["current"].setText(summary.current_label)
        self.metric_values["current"].setAccessibleName(summary.current_label)
        for key, button in self.buttons.items():
            button.setEnabled(
                not source_error
                and (key == "checks" or (key == "coil" and self.has_coil_statistics) or summary.analysis_available)
            )
            button.setToolTip(
                "Inspect saved checks and source diagnostics."
                if key == "checks"
                else self.coil_report.message
                if key == "coil"
                else "Source verification failed; reopen and verify the run before field analysis."
                if not summary.analysis_available
                else f"Open {button.text().lower()} for {summary.source_label} ({summary.component or 'recorded component'})."
            )
        self.routing_button.setEnabled(bool(routing_available and not source_error))
        self.routing_button.setToolTip(
            "Inspect exact saved interconnections, separate supports, developed distances, conductor heights and recorded body witnesses."
            if not source_error
            else "Source verification failed; reopen and verify the run before inspecting geometry."
        )

    def set_analysis_available(self, available):
        """Apply the controller's cached finite field/target capability check."""
        for key, button in self.buttons.items():
            if key not in {"checks", "coil"}:
                button.setEnabled(bool(available) and self.summary.visible and self.summary.analysis_available)
            elif key == "coil":
                button.setEnabled(self.buttons["checks"].isEnabled() and (bool(available) or self.has_coil_statistics))
