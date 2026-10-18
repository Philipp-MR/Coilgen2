"""Prominent saved field errors with explicit current and sampling scope.

This panel reads immutable scalar summaries and array names. It neither loads
field arrays nor calculates residuals, current fits or engineering acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics, QPalette
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from ..application.diagnostics import run_verdicts
from ..application.routing_analysis import routing_analysis_available
from ..field_components import field_component_label, normalize_field_component
from .colors import apply_tone, tone_color
from .coil_statistics import build_coil_statistics
from .icons import icon
from .run_metadata import read_run_metadata


@dataclass(frozen=True, slots=True)
class FieldResultSummary:
    visible: bool = False
    title: str = ""
    badge: str = ""
    tone: str = "neutral"
    source_key: str = ""
    source_label: str = ""
    component: str = ""
    rms_uT: float | None = None
    peak_uT: float | None = None
    relative_percent: float | None = None
    current_A: float | None = None
    current_label: str = "Original current"
    current_display: str = ""
    context: str = ""
    tooltip: str = ""
    analysis_available: bool = False


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _number(value, scale=1.0, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value) * scale
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and (not nonnegative or result >= 0) else None


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _component(evidence, parent, metadata):
    value = (
        evidence.get("component")
        or parent.get("component")
        or _mapping(metadata.get("config")).get("target_field_component")
        or metadata.get("target_field_component")
    )
    if not value:
        return ""
    try:
        return normalize_field_component(value, allow_bmag=True)
    except ValueError:
        return ""


def _candidate(record, metadata):
    names = set(getattr(record, "array_names", ()))
    wires = _mapping(metadata.get("wire_fields"))

    def wire(kind):
        parent = _mapping(wires.get(kind))
        original = _mapping(parent.get("at_stream_current"))
        label = "Routed winding" if kind == "routed" else "Unconnected contours"
        if original.get("available") is True and {"target_points_m", kind + "_field_at_stream_current_T"} <= names:
            return kind + "_at_stream_current", label, original, parent, "original contour current", "design"
        if (
            parent.get("available") is True
            and parent.get("current_refitted") is False
            and {"target_points_m", kind + "_field_T"} <= names
        ):
            return kind, label, parent, parent, "original recorded current", "design"
        return None

    physics = _mapping(metadata.get("fabricated_physics"))
    winding = _mapping(metadata.get("winding_validation"))
    if (
        winding.get("available") is True
        and winding.get("current_refitted") is False
        and winding.get("current_source") == "original_contour_current_no_refit"
        and {"winding_validation_points_m", "winding_validation_field_T"} <= names
    ):
        return "winding_validation", "Reviewed winding", winding, winding, "original contour current", "independent"
    if (
        "fabricated_field_T" in names
        and {"fabricated_points_m", "target_points_m"} & names
        and physics.get("current_source") == "original_contour_current_no_refit"
    ):
        rows = physics.get("checks", ())
        for row in rows if isinstance(rows, (list, tuple)) else ():
            if (
                isinstance(row, Mapping)
                and row.get("name") == "Fabricated path field error (original current)"
                and row.get("verified") is True
                and _number(row.get("rms_error_T"), nonnegative=True) is not None
            ):
                return "fabricated", "Reviewed conductor", row, physics, "original contour current", "design"
    if candidate := wire("routed"):
        return candidate
    if candidate := wire("contour"):
        return candidate
    validation = _mapping(metadata.get("validation"))
    if validation.get("available") is True and {"validation_points_m", "validation_field_T"} <= names:
        return "validation", "Continuous sheet", validation, validation, "sheet solution", "independent"
    if {"target_points_m", "sheet_field_T"} <= names:
        return "sheet", "Continuous sheet", {}, {}, "sheet solution", "design"
    # Legacy results may retain only a refitted wire field. Identify that fact
    # rather than presenting it as the physical original-current result.
    for kind in ("routed", "contour"):
        parent = _mapping(wires.get(kind))
        if parent.get("available") is True and {"target_points_m", kind + "_field_T"} <= names:
            return (
                kind,
                "Routed winding" if kind == "routed" else "Unconnected contours",
                parent,
                parent,
                (
                    "refitted recorded current"
                    if parent.get("current_refitted") is True
                    else "current policy unrecorded"
                ),
                "design",
            )
    return None


def _identity(metadata):
    name = _text(metadata.get("name")) or "Saved run"
    revision = metadata.get("project_revision", _mapping(metadata.get("job")).get("project_revision"))
    return f"Results · {name}" + (
        f" · revision {revision}" if isinstance(revision, int) and not isinstance(revision, bool) else ""
    )


def _badge(metadata, source_error, stale):
    verdicts = run_verdicts(metadata)
    summary = _mapping(metadata.get("summary"))
    independent_failed = any(
        _mapping(metadata.get(key)).get("available") is True and _mapping(metadata.get(key)).get("passed") is False
        for key in ("validation", "winding_validation")
    )
    checks = (
        "Checks failed"
        if "Failed" in verdicts.values() or summary.get("accepted") is False or independent_failed
        else (
            "Saved checks passed"
            if all(value == "Passed" for value in verdicts.values())
            else "Sheet passed"
            if verdicts["sheet"] == "Passed"
            else "Checks unverified"
        )
    )
    badge, tone = (
        ("Verification failed", "error")
        if source_error
        else ("Earlier model", "warning")
        if stale
        else (
            checks,
            "error"
            if checks == "Checks failed"
            else "success"
            if checks == "Saved checks passed"
            else "info"
            if checks == "Sheet passed"
            else "warning",
        )
    )
    return checks, badge, tone


def summarize_field_result(record, *, source_error="", stale=False):
    """Read saved error metrics without mistaking generic field RMS for error."""
    if record is None:
        return FieldResultSummary()
    metadata = read_run_metadata(record)
    title = _identity(metadata)
    checks, badge, tone = _badge(metadata, source_error, stale)
    candidate = _candidate(record, metadata)
    if candidate is None:
        context = "Field not calculated · saved coil properties" + (" · " + checks if stale else "")
        if source_error:
            context = "Retained snapshot · source verification failed · " + context
        return FieldResultSummary(
            title=title, badge=badge, tone=tone, context=context, tooltip=str(source_error) if source_error else context
        )
    key, label, evidence, parent, policy, sampling = candidate
    component = _component(evidence, parent, metadata)
    component_label = field_component_label(component) if component else "component unrecorded"
    count = _count(evidence.get("sample_count")) or _count(evidence.get("point_count"))
    rms = _number(evidence.get("rms_error_T"), 1e6, nonnegative=True)
    peak = _number(evidence.get("peak_error_T"), 1e6, nonnegative=True)
    target_rms = _number(evidence.get("target_rms_T"), nonnegative=True)
    relative = (
        _number(evidence.get("relative_rms_error", evidence.get("relative_field_error")), 100, nonnegative=True)
        if target_rms is not None and target_rms > 0
        else None
    )
    current = _number(evidence.get("current_A"))
    current_label = (
        "Refitted current"
        if policy == "refitted recorded current"
        else ("Recorded current" if policy == "current policy unrecorded" else "Original current")
    )
    current_display = ""
    if key == "winding_validation" and _mapping(evidence.get("currents_A")):
        current, current_label, current_display = None, "Original body currents", "Per body"
    summary = _mapping(metadata.get("summary"))
    if key == "fabricated" and summary.get("scope") == "fabrication_review":
        current = _number(summary.get("current_a"))
        if current is None:
            currents = _mapping(parent.get("currents_A"))
            if any(_number(value) is not None for value in currents.values()):
                current_label, current_display = "Original body currents", "Per body"
            else:
                electrical = _mapping(_mapping(metadata.get("fabrication")).get("electrical_metrics"))
                bodies = electrical.get("bodies", ())
                if isinstance(bodies, (list, tuple)) and any(
                    _number(_mapping(body).get("current_A")) is not None for body in bodies
                ):
                    current_label, current_display = "Original body currents", "Per body"
    if key in {"sheet", "validation"}:
        current, current_label, current_display = None, "Wire current", "Not applicable"
    sampled = "Independent samples" if sampling == "independent" else "Design samples"
    sample_text = sampled + (f" · n={count:,}" if count is not None else " · sample count unrecorded")
    detail = (
        "Independent quadrature unverified"
        if key == "validation" and evidence.get("independent_quadrature_check") is not True
        else "Independent thin-wire winding check"
        if key == "winding_validation"
        else (
            "Field diagnostic; wire acceptance is separate"
            if key not in {"sheet", "validation"}
            else "Sheet evidence; wire acceptance is separate"
        )
    )
    context = f"{label} · {policy} · {component_label} error\n{sample_text} · {detail}"
    if source_error:
        context = "Retained snapshot · source verification failed · " + context
    elif stale:
        context = "Earlier applied model · " + context + f" · {checks}"
    convention = _text(evidence.get("rms_convention")) or "unrecorded"
    tooltip = (
        f"{title}\n{context}\nResidual: calculated minus target. Recorded RMS convention: {convention}. "
        "Relative RMS uses the recorded global target RMS, never pointwise division at target zeros. "
        "These scalar summaries are retained from the saved run; no field is recalculated for this panel."
    )
    if key.startswith(("routed", "contour")) and parent.get("scope") == "design_grid_wire_field":
        tooltip += " Wire RMS uses active unsuppressed positive-weight design samples; thin-wire Biot–Savart model."
    if current_display == "Per body":
        tooltip += " Signed original body currents are recorded separately; no aggregate circuit current is inferred."
    if key in {"sheet", "validation"}:
        tooltip += " A continuous sheet solution does not define a single discrete wire current."
    if source_error:
        tooltip += "\n" + str(source_error)
    return FieldResultSummary(
        visible=True,
        title=title,
        badge=badge,
        tone=tone,
        source_key=key,
        source_label=label,
        component=component,
        rms_uT=rms,
        peak_uT=peak,
        relative_percent=relative,
        current_A=current,
        current_label=current_label,
        current_display=current_display,
        context=context,
        tooltip=tooltip,
        analysis_available=not bool(source_error),
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
        layout.addLayout(header)
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
        layout.addWidget(self.field_cards)
        self.context = QLabel(self)
        self.context.setTextFormat(Qt.TextFormat.PlainText)
        self.context.setWordWrap(True)
        self.context.setMinimumWidth(0)
        self.context.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.context.setAccessibleName("Field source, current and sampling scope")
        layout.addWidget(self.context)
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
        layout.addLayout(properties)
        self.coil_scope = QLabel(self)
        self.coil_scope.setTextFormat(Qt.TextFormat.PlainText)
        self.coil_scope.setWordWrap(True)
        self.coil_scope.setMinimumWidth(0)
        self.coil_scope.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.coil_scope.setAccessibleName("Saved electrical, conductor material and volume scope")
        layout.addWidget(self.coil_scope)
        actions = QHBoxLayout()
        actions.setSpacing(5)
        self.buttons = {}
        for key, text, drawing in (
            ("field", "Field map", "field-error"),
            ("slices", "Slices", "slice"),
            ("errors", "Errors", "chart"),
            ("harmonics", "Harmonics", "target-harmonic"),
            ("coil", "Coil stats", "conductor-profile"),
            ("checks", "Checks", "check"),
        ):
            button = QPushButton(icon(drawing), text, self)
            button.setObjectName("results_overview_" + key)
            button.setMinimumHeight(32)
            button.setAccessibleName("Open saved field " + text.lower())
            button.clicked.connect(lambda _checked=False, action=key: self.action_requested.emit(action))
            actions.addWidget(button, 1)
            self.buttons[key] = button
        layout.addLayout(actions)
        self._style_panel()
        self.hide()

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
            f"QFrame#coil_result_mass, QFrame#coil_result_volume {{ background: {background}; border: 1px solid {outline}; "
            "border-radius: 5px; }"
            f"QFrame#field_result_rms {{ border-left: 3px solid {accent}; }}"
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
        scopes = []
        for key in self.coil_metric_values:
            row = self.coil_report.row("coil:" + key)
            number = _number(row.value, nonnegative=True) if row is not None else None
            body_suffix = {
                "resistance": ":resistance_ohm",
                "inductance": ":inductance_H",
                "volume": ":mesh_volume",
            }.get(key)
            volume_prefix = (
                "reviewed:" if any(item.key.startswith("reviewed:") for item in self.coil_report.rows) else "pipeline:"
            )
            prefix = volume_prefix if key == "volume" else "body:"
            body_rows = tuple(
                item
                for item in self.coil_report.rows
                if body_suffix
                and item.key.startswith(prefix)
                and item.key.endswith(body_suffix)
                and _number(item.value, nonnegative=True) is not None
            )
            display = "Needs density" if key == "mass" and number is None else "Not recorded"
            if number is None and body_rows:
                display = "Per body"
            if number is not None:
                scale, unit = {
                    "resistance": (1.0, "Ω"),
                    "inductance": (1e6, "µH"),
                    "mass": (1e3, "g"),
                    "volume": (1e6, "cm³"),
                }[key]
                converted = _number(number, scale, nonnegative=True)
                display = f"{converted:.4g} {unit}" if converted is not None else "Not recorded"
            value = self.coil_metric_values[key]
            value.set_value(display, unavailable=number is None, point_size=13.0)
            scope = row.scope if row is not None else "No saved quantity or scope."
            source = row.source_key if row is not None else ""
            tooltip = f"{row.quantity}: {row.status}.\n{scope}\nSaved source: {source}" if row is not None else scope
            if number is None and body_rows:
                body_word = "body" if len(body_rows) == 1 else "bodies"
                tooltip += f"\nSaved values for {len(body_rows)} {body_word}; open Coil stats for values and individual scopes."
            if key == "mass" and number is None:
                tooltip += "\n" + self.coil_report.mass_unavailable_reason
            value.setToolTip(tooltip)
            self.coil_metric_labels[key].setToolTip(tooltip)
            if key in {"resistance", "volume"}:
                scopes.append(("R/L" if key == "resistance" else "Volume") + ": " + scope)
        scopes.append(
            "Mass: recorded conductor only"
            if _number(self.coil_report.row("coil:mass").value, nonnegative=True) is not None
            else "Mass: density required; insulation, feed and former excluded"
        )
        mass_scope = (
            "recorded conductor mass"
            if _number(self.coil_report.row("coil:mass").value, nonnegative=True) is not None
            else "mass needs density"
        )
        self.coil_scope.setText(self.coil_report.overview_scope + " · " + mass_scope)
        self.coil_scope.setToolTip("\n".join(scopes) + "\n" + self.coil_report.message)

    def set_record(self, record, *, source_error="", stale=False):
        key = (id(record), str(source_error), bool(stale))
        if key == self._cache_key:
            return
        self._cache_key, self.record = key, record
        self.summary = summary = summarize_field_result(record, source_error=source_error, stale=stale)
        self.coil_report = build_coil_statistics(record)
        routing_available = routing_analysis_available(record)
        visible = record is not None and (summary.visible or self.has_coil_statistics or routing_available)
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
        for key, value, unit in (
            ("rms", summary.rms_uT, "µT"),
            ("peak", summary.peak_uT, "µT"),
            ("relative", summary.relative_percent, "%"),
            ("current", summary.current_A, "A"),
        ):
            text = (
                (summary.current_display if key == "current" and summary.current_display else "Not recorded")
                if (value is None)
                else (f"{value:+.4g}" if key == "current" else f"{value:.4g}") + " " + unit
            )
            self.metric_values[key].set_value(text, unavailable=value is None)
            self.metric_values[key].setToolTip(summary.tooltip)
        self.metric_labels["current"].setText(summary.current_label)
        self.metric_values["current"].setAccessibleName(summary.current_label)
        for key, button in self.buttons.items():
            button.setEnabled((key in {"checks", "coil"} and not source_error) or summary.analysis_available)
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
