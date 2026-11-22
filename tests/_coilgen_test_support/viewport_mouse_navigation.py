"""Shared fixtures for viewport mouse navigation."""

from PySide6.QtCore import QCoreApplication, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent


def mouse(
    viewport,
    kind,
    position=(30, 30),
    *,
    button=Qt.MouseButton.NoButton,
    buttons=Qt.MouseButton.NoButton,
    modifiers=Qt.KeyboardModifier.NoModifier,
):
    widget = viewport.plotter.interactor
    local = QPoint(*position)
    event = QMouseEvent(kind, QPointF(local), QPointF(widget.mapToGlobal(local)), button, buttons, modifiers)
    QCoreApplication.sendEvent(widget, event)
