"""Semantic GUI colors with readable contrast against the active Qt theme."""

from PySide6.QtGui import QBrush, QColor, QPalette
from PySide6.QtWidgets import QApplication


_ACCENTS = {
    "info": ("#075aaa", "#87bcff"),
    "success": ("#08766c", "#76d1bb"),
    "warning": ("#8b5900", "#f3ca77"),
    "error": ("#b32e28", "#ffaaa4"),
}


def _luminance(color):
    channels = (color.redF(), color.greenF(), color.blueF())
    linear = tuple(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels)
    return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


def _contrast(first, second):
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def tone_color(tone, palette):
    """Choose a foreground that remains readable on Window and Base backgrounds."""
    window = palette.color(QPalette.ColorRole.Window)
    base = palette.color(QPalette.ColorRole.Base)
    if tone not in _ACCENTS:
        return palette.color(QPalette.ColorRole.WindowText)
    light, dark = _ACCENTS[tone]
    color = QColor(light if _luminance(window) > 0.3 else dark)
    if min(_contrast(color, window), _contrast(color, base)) >= 4.5:
        return color
    # Unusual custom themes can sit between the usual light/dark palettes.
    # Prefer their native text if the semantic hue would be hard to read.
    native = palette.color(QPalette.ColorRole.WindowText)
    candidates = (native, QColor("#000000"), QColor("#ffffff"))
    return QColor(max(candidates, key=lambda candidate: min(_contrast(candidate, window), _contrast(candidate, base))))


def item_foreground(tone, palette):
    """Foreground brush for tree/table status cells; text still conveys status."""
    return QBrush(tone_color(tone, palette))


def apply_tone(label, tone):
    """Color a status label without replacing its theme background or typography."""
    parent = label.parentWidget()
    base = parent.palette() if parent is not None else QApplication.palette()
    palette = QPalette(label.palette())
    foreground = tone_color(tone, base)
    palette.setColor(QPalette.ColorRole.WindowText, foreground)
    palette.setColor(QPalette.ColorRole.Text, foreground)
    label.setProperty("semanticTone", tone if tone in _ACCENTS else "neutral")
    label.setPalette(palette)
