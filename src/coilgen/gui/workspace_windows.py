"""Owned, reusable windows for inspecting one scientific task at a time."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QTabWidget, QVBoxLayout, QWidget


class PanelWindow(QDialog):
    """Closing hides the original widget; it never destroys its draft or data."""

    closed = Signal()

    def __init__(self, title, content, parent, *, size=(960, 680)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.content = content
        layout = QVBoxLayout(self)
        layout.addWidget(content)
        screen = parent.screen() if parent is not None else self.screen()
        bounds = screen.availableGeometry() if screen else None
        self.resize(
            min(size[0], max(1, bounds.width() - 20)) if bounds else size[0],
            min(size[1], max(1, bounds.height() - 50)) if bounds else size[1],
        )

    def open_panel(self):
        if self.isMinimized():
            self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.show()
        self.raise_()

    def hideEvent(self, event):
        super().hideEvent(event)
        # Escape/reject/done bypass closeEvent. A non-spontaneous hide covers
        # all those paths; minimizing an OS window keeps it logically open.
        if not event.spontaneous():
            self.closed.emit()


class EvidenceWindows(QTabWidget):
    """Registry retaining the former page API and each original evidence widget.

    The tab widget is never displayed. Each registered page has an owned window;
    the registry's open flags are also the portable workspace preferences.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self._pages = []
        self._windows = {}
        self._contents = {}
        self.hide()

    def addTab(self, page, title):
        return self.insertTab(self.count(), page, title)

    def insertTab(self, index, page, title):
        if page in self._pages:
            raise ValueError("This evidence page is already registered.")
        index = min(max(index, 0), self.count())
        self._pages.insert(index, page)
        page.setParent(self)
        page.hide()
        index = super().insertTab(index, QWidget(self), title)
        super().setTabVisible(index, False)
        return index

    def widget(self, index):
        return self._pages[index] if 0 <= index < len(self._pages) else None

    def currentWidget(self):
        return self.widget(self.currentIndex())

    def indexOf(self, page):
        return self._pages.index(page) if page in self._pages else -1

    def setCurrentWidget(self, page):
        index = self.indexOf(page)
        if index >= 0:
            self.setCurrentIndex(index)

    def set_content(self, page, content):
        """Allow Jobs to include its progress and filtering controls."""
        if self.indexOf(page) < 0:
            raise ValueError("Unknown evidence window.")
        if page in self._windows:
            raise ValueError("Window content must be configured before opening the window.")
        self._contents[page] = content
        content.setParent(self)
        content.hide()

    def window_for(self, page):
        if page not in self._windows:
            index = self.indexOf(page)
            if index < 0:
                raise ValueError("Unknown evidence window.")
            window = PanelWindow(
                self.tabText(index) + " — CoilGen2",
                self._contents.get(page, page),
                self.parentWidget(),
                size=(1200, 740) if self.tabText(index) in {"Design acceptance", "Problems"} else (960, 680),
            )
            window.setWindowIcon(self.tabIcon(index))
            window.closed.connect(self._window_closed)
            self._windows[page] = window
            self._contents.get(page, page).show()
            page.show()
        return self._windows[page]

    def _window_closed(self):
        window = self.sender()
        page = next((page for page, owned in self._windows.items() if owned is window), None)
        if page is not None:
            self.close_page(page)

    def open_page(self, page):
        index = self.indexOf(page)
        if index < 0:
            raise ValueError("Unknown evidence window.")
        self.setTabVisible(index, True)
        self.setCurrentIndex(index)
        self.window_for(page).open_panel()

    def open_current(self):
        self.open_page(self.currentWidget() or self.widget(0))

    def close_page(self, page):
        index = self.indexOf(page)
        if index >= 0:
            self.setTabVisible(index, False)
        window = self._windows.get(page)
        if window is not None:
            window.hide()

    def close_all(self):
        for page in self._pages:
            self.close_page(page)

    def any_open(self):
        return any(self.isTabVisible(i) for i in range(self.count()))
