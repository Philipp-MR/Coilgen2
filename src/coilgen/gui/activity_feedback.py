"""Mirror global computation activity in this workbench's owned dialogs."""

from __future__ import annotations

from collections.abc import Mapping
from weakref import WeakValueDictionary, ref

from PySide6.QtCore import QEvent, QObject, Slot
from PySide6.QtWidgets import QApplication, QBoxLayout, QDialog, QFileDialog, QFormLayout, QGridLayout, QMessageBox
from shiboken6 import getCppPointer, isValid

from .design_progress import ComputationIndicator


class ComputationFeedback(QObject):
    """Publish one runtime snapshot to the main window and passive dialog footers.

    The application event filter discovers owned dialogs as they open, including
    nested modal editors. It never polls jobs, opens windows or changes focus.
    Unsupported layouts and platform file/message dialogs remain untouched.
    """

    def __init__(self, root, main_indicator):
        super().__init__(root)
        self._root = ref(root)
        self._main_indicator = ref(main_indicator)
        self._app = ref(QApplication.instance())
        self.indicators = WeakValueDictionary()
        self._activity = ((), (), None)
        self._closed = False
        QApplication.instance().installEventFilter(self)
        root.destroyed.connect(self.close)

    def set_activity(self, states=(), *, working=(), unavailable=None):
        if self._closed:
            return
        # Worker dictionaries may be updated in place between publications.
        states = tuple(dict(state) for state in states)
        working = tuple(dict(item) if isinstance(item, Mapping) else item for item in working)
        self._activity = states, working, unavailable
        main = self._main_indicator()
        if main is not None and isValid(main):
            main.set_activity(states, working=working, unavailable=unavailable)
        for indicator in tuple(self.indicators.values()):
            if isValid(indicator):
                indicator.set_activity(states, working=working, unavailable=unavailable)

    def eventFilter(self, watched, event):
        if (
            not self._closed
            and event.type() == QEvent.Type.Show
            and isinstance(watched, QDialog)
            and not isinstance(watched, (QFileDialog, QMessageBox))
            and self._owns(watched)
        ):
            self._attach(watched)
        return False

    def _owns(self, dialog):
        root = self._root()
        if root is None or not isValid(root):
            return False
        owner = dialog.parentWidget()
        while owner is not None:
            if owner is root:
                return True
            owner = owner.parentWidget()
        return False

    def _attach(self, dialog):
        key = getCppPointer(dialog)[0]
        indicator = self.indicators.get(key)
        if indicator is not None and isValid(indicator):
            return
        layout = dialog.layout()
        vertical = isinstance(layout, QBoxLayout) and layout.direction() in (
            QBoxLayout.Direction.TopToBottom,
            QBoxLayout.Direction.BottomToTop,
        )
        if not vertical and not isinstance(layout, (QFormLayout, QGridLayout)):
            return
        indicator = ComputationIndicator(dialog)
        indicator.setObjectName("window_computation_indicator")
        self.indicators[key] = indicator
        indicator.destroyed.connect(lambda: self.indicators.pop(key, None))
        if vertical:
            layout.addWidget(indicator)
        elif isinstance(layout, QFormLayout):
            layout.addRow(indicator)
        else:
            layout.addWidget(indicator, layout.rowCount(), 0, 1, max(1, layout.columnCount()))
        states, working, unavailable = self._activity
        indicator.set_activity(states, working=working, unavailable=unavailable)

    @Slot()
    def close(self):
        """Stop discovery and remove stale activity when the workbench closes."""
        if self._closed:
            return
        self._closed = True
        app = self._app()
        if app is not None and isValid(app):
            app.removeEventFilter(self)
        for indicator in tuple(self.indicators.values()):
            if isValid(indicator):
                indicator.hide()
        self.indicators.clear()
