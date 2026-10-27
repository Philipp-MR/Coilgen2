"""Matplotlib plot-book infrastructure.

The plot book is the active plotting surface used by the Spyder examples. It
collects many normal Matplotlib figures and presents them as pages in one
window, so users can inspect dense coil-design outputs without dozens of
separate figure windows.

Extension model
---------------
Most new plots should not subclass the plot book. Add a normal Matplotlib
factory to ``build_result_plot_specs`` in ``matplotlib_visualization.py`` and
let ``MatplotlibPlotBookBuilder.add_specs`` register it in the book.

Use ``MatplotlibPlotBookBuilder.add_figure`` or ``add_figures`` for study
scripts that already create standalone figures, such as parameter sweeps.
Plot specifications are evaluated lazily by live Qt books so large project
books can open before every expensive 3D or slice-map page has rendered. Live
Qt books can then preload deferred pages during idle event-loop turns, keeping
Matplotlib and Qt canvas creation on the GUI thread while making later page
selection faster.

Subclass ``MatplotlibPlotBook`` only when changing the container/navigation
behavior itself. Current concrete implementations are:

- ``QtMatplotlibPlotBook`` for live Matplotlib canvases with rotation/zoom.
- ``ImageMatplotlibPlotBook`` as a non-Qt raster fallback.

Each page is represented either by a live ``PlotBookEntry`` or, in fallback
mode, by a rasterized ``PlotBookPage``. The metadata keys are intentionally
short and stable because they drive navigation labels and special page behavior
such as the scrollable performance table.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Sequence
import weakref

import numpy as np

from .attribution import AUTHOR_CREDIT, SOFTWARE_NAME, software_attribution


Figure = Any
PlotSpec = Any
ShortNameFactory = Callable[[PlotSpec, str, int, int], str]
_ACTIVE_PLOT_BOOKS: weakref.WeakSet["MatplotlibPlotBook"] = weakref.WeakSet()
_VISIBLE_QT_PLOT_BOOKS: set["QtMatplotlibPlotBook"] = set()
_PLOT_BOOK_RENDERING: ContextVar[bool] = ContextVar("coilgen_plot_book_rendering", default=False)


def is_plot_book_rendering() -> bool:
    """Whether a book/export owns the next draw of a plotting factory's figure."""
    return _PLOT_BOOK_RENDERING.get()


@contextmanager
def plot_book_rendering():
    """Let factories create artists without an unnecessary intermediate draw."""
    token = _PLOT_BOOK_RENDERING.set(True)
    try:
        yield
    finally:
        _PLOT_BOOK_RENDERING.reset(token)


SCROLLABLE_PLOT_BOOK_KEYS = frozenset(
    {
        "performance_table",
        "dsv_field_slices",
        "shim_homogeneity",
        "shim_mosaic",
        "residual_mosaic",
        "field_slice_mosaic",
        "gradient_slice_mosaic",
        "concomitant_slice_mosaic",
        "metric_bars",
        "parameter_table",
        "stream_contour_wire_residuals",
    }
)
THREE_D_VIEW_PRESETS = (
    ("Iso", "Isometric view", 22.0, 135.0),
    ("XY", "Look normal to the XY plane", 90.0, -90.0),
    ("XZ", "Look normal to the XZ plane", 0.0, -90.0),
    ("YZ", "Look normal to the YZ plane", 0.0, 0.0),
)
THREE_D_VIEW_ROTATIONS = (
    ("X+90", "Rotate current 3D view +90 deg about the world X axis.", "x", 1),
    ("X-90", "Rotate current 3D view -90 deg about the world X axis.", "x", -1),
    ("Y+90", "Rotate current 3D view +90 deg about the world Y axis.", "y", 1),
    ("Y-90", "Rotate current 3D view -90 deg about the world Y axis.", "y", -1),
    ("Z+90", "Rotate current 3D view +90 deg about the world Z axis.", "z", 1),
    ("Z-90", "Rotate current 3D view -90 deg about the world Z axis.", "z", -1),
)


@dataclass(frozen=True, slots=True)
class PlotBookPage:
    """Rasterized fallback page used when live Qt canvases are unavailable.

    ``PlotBookPage`` is deliberately data-only. It is used by
    ``ImageMatplotlibPlotBook`` after a source figure has been converted to an
    RGBA array. New fallback page metadata should be added here rather than
    stored in ad-hoc attributes on Matplotlib figures.
    """

    key: str
    title: str
    category: str
    image_rgba: np.ndarray
    channel: str = ""
    page_id: str = ""
    section: str = ""


@dataclass(frozen=True, slots=True)
class PlotBookEntry:
    """One live or deferred Matplotlib figure plus its navigation metadata.

    Parameters
    ----------
    key:
        Stable internal identifier. Keys can be used for special page handling,
        for example ``"performance_table"`` enables scrollbars.
    short_name:
        Compact label displayed in the right-side page selector.
    title:
        Full scientific title used for tooltips and window titles.
    category:
        Short grouping label such as ``"layout"``, ``"field"``, or
        ``"metrics"``.
    channel:
        Optional project channel label used by the page selector for
        multi-channel plot books.
    figure:
        The Matplotlib figure object that will be embedded or a private lazy
        page object created by ``MatplotlibPlotBookBuilder``.
    """

    key: str
    short_name: str
    title: str
    category: str
    figure: Figure
    channel: str = ""
    page_id: str = ""
    section: str = ""
    layout: str = "auto"


@dataclass(frozen=True, slots=True)
class _DeferredPlotBookFigures:
    """Plot specification wrapper evaluated when its page is first selected."""

    key: str
    title: str
    category: str
    factory: Callable[[], Figure | list[Figure]]
    short_name_factory: ShortNameFactory
    plt: Any
    timing_report: list[dict[str, float | int | str]] | None = None
    short_name: str = ""
    channel: str = ""
    page_id: str = ""
    section: str = ""
    layout: str = "auto"

    def placeholder_entry(self) -> PlotBookEntry:
        return PlotBookEntry(
            key=self.key,
            short_name=self.short_name_factory(self, self.title, 1, 1),
            title=self.title,
            category=self.category,
            figure=self,
            channel=self.channel,
            page_id=self.page_id,
            section=self.section,
            layout=self.layout,
        )

    def materialize(self) -> tuple[PlotBookEntry, ...]:
        start = time.perf_counter()
        existing_figures = set(getattr(self.plt, "get_fignums", lambda: [])())
        try:
            with pyplot_interactive_disabled(self.plt), plot_book_rendering():
                figure_or_figures = self.factory()
        except Exception:
            # A factory may allocate a figure before discovering invalid data.
            for number in set(getattr(self.plt, "get_fignums", lambda: [])()) - existing_figures:
                self.plt.close(number)
            raise
        figures = figure_or_figures if isinstance(figure_or_figures, (list, tuple)) else [figure_or_figures]
        entries: list[PlotBookEntry] = []
        total = len(figures)
        for index, figure in enumerate(figures, start=1):
            if figure is None:
                continue
            # The specification is authoritative; backend labels such as
            # "Figure 1" must not replace scientific page metadata.
            title = source_figure_title(figure, fallback=self.title)
            if total > 1:
                title = f"{title} ({index}/{total})"
            entries.append(
                PlotBookEntry(
                    key=self.key,
                    short_name=self.short_name_factory(self, title, index, total),
                    title=title,
                    category=self.category,
                    figure=figure,
                    channel=self.channel,
                    page_id=self.page_id if not entries else f"{self.page_id}/{index}",
                    section=self.section,
                    layout=self.layout,
                )
            )
        if self.timing_report is not None:
            self.timing_report.append(
                {
                    "plot": f"book:{self.key}",
                    "seconds": float(time.perf_counter() - start),
                    "figures": len(entries),
                }
            )
        return tuple(entries)


@dataclass(frozen=True, slots=True)
class StudyPlotPage:
    """Project-customizable plot-book page definition.

    This is the general, data-oriented page representation for study and
    project plot books. A page only describes what to draw and how to label it;
    the normal ``MatplotlibPlotBookBuilder`` still owns evaluation, embedding,
    and Qt/image fallback behavior.
    """

    key: str
    title: str
    category: str
    factory: Callable[[], Figure | list[Figure]]
    short_name: str = ""
    enabled: bool = True
    channel: str = ""
    page_id: str = ""
    section: str = ""
    layout: str = "auto"


@dataclass(frozen=True, slots=True)
class StudyPlotSection:
    """Named group of project plot-book pages."""

    title: str
    pages: tuple[StudyPlotPage, ...]
    category: str = ""
    enabled: bool = True

    def enabled_pages(self) -> tuple[StudyPlotPage, ...]:
        if not self.enabled:
            return ()
        return tuple(
            replace(page, section=page.section or self.title, category=page.category or self.category)
            for page in self.pages
            if page.enabled
        )


@dataclass(frozen=True, slots=True)
class StudyPlotBookDefinition:
    """General object representation for project-specific plot books.

    Projects can construct this object from arbitrary run results, tables, or
    figures. Core CoilGen then turns the page definitions into the same live
    plot-book window used by single-result inspection.
    """

    title: str
    pages: tuple[StudyPlotPage, ...] = ()
    sections: tuple[StudyPlotSection, ...] = ()

    def enabled_pages(self) -> tuple[StudyPlotPage, ...]:
        pages = [page for page in self.pages if page.enabled]
        for section in self.sections:
            pages.extend(section.enabled_pages())
        return tuple(pages)


@dataclass(slots=True)
class _PageState:
    status: str
    error: Exception | None = None
    seconds: float = 0.0
    original_size: tuple[float, float] | None = None


class _PageCollection:
    """Ordered page definitions and materialization state, independent of Qt."""

    def __init__(self, entries: Sequence[PlotBookEntry] = ()) -> None:
        self.entries: list[PlotBookEntry] = []
        self.states: dict[str, _PageState] = {}
        self._suffixes: dict[str, int] = {}
        self.add(entries)

    def add(self, entries: Sequence[PlotBookEntry], *, index: int | None = None) -> None:
        position = len(self.entries) if index is None else index
        for entry in entries:
            base = entry.page_id or entry.key or "page"
            page_id = base
            suffix = self._suffixes.get(base, 2)
            while page_id in self.states:
                page_id = f"{base}~{suffix}"
                suffix += 1
            self._suffixes[base] = suffix
            figure = entry.figure
            if isinstance(figure, _DeferredPlotBookFigures):
                figure = replace(figure, page_id=page_id)
            entry = replace(entry, page_id=page_id, figure=figure)
            self.entries.insert(position, entry)
            self.states[page_id] = _PageState(
                "deferred" if _entry_is_deferred(entry) else "ready",
                original_size=_figure_size(entry.figure) if not _entry_is_deferred(entry) else None,
            )
            position += 1

    def materialize(self, index: int, *, retry: bool = False) -> PlotBookEntry | None:
        entry = self.entries[index]
        state = self.states[entry.page_id]
        if state.error is not None and not retry:
            return None
        state.error = None
        if not _entry_is_deferred(entry):
            state.status = "ready"
            return entry
        state.status = "rendering"
        start = time.perf_counter()
        try:
            materialized = _materialize_deferred_entry(entry)
            if not materialized:
                raise ValueError("No figure was returned.")
            first = replace(materialized[0], page_id=entry.page_id)
            for item in materialized:
                unregister_pyplot_figure(item.figure)
            self.entries[index] = first
            self.add(materialized[1:], index=index + 1)
            state.status = "ready"
            state.original_size = _figure_size(first.figure)
            return first
        except Exception as exc:
            state.status = "unavailable" if isinstance(exc, ValueError) else "failed"
            state.error = exc
            return None
        finally:
            state.seconds = time.perf_counter() - start

    def dispose(self) -> None:
        for entry in self.entries:
            if not _entry_is_deferred(entry):
                _dispose_plot_book_figure(entry.figure)
        self.entries.clear()
        self.states.clear()
        self._suffixes.clear()


class MatplotlibPlotBook:
    """Base class for plot books shown as one user-facing window.

    Subclasses own the actual UI implementation. They must implement
    ``show()``, ``close()``, ``show_page()``, and ``page_count``. Keep plotting
    logic outside subclasses: they should receive ready-made Matplotlib figures
    or raster pages, not create CoilGen-specific plots.
    """

    def __init__(self, *, title: str) -> None:
        self.title = str(title)
        self.index = 0

    @property
    def page_count(self) -> int:
        raise NotImplementedError

    def show(self, *, block: bool = False) -> None:
        """Show the plot-book window.

        ``block=False`` is the default for Spyder so the IPython prompt remains
        usable while the plot book is open.
        """
        raise NotImplementedError

    def open(self, page: int | None = None, *, block: bool = False) -> None:
        """Show the plot book again from a retained script/workspace variable.

        ``open`` is a script-facing alias for ``show``. Passing ``page`` first
        selects the zero-based page index, then shows the plot-book window.
        """
        if page is not None:
            self.show_page(int(page))
        self.show(block=block)

    def reopen(self, page: int | None = None, *, block: bool = False) -> None:
        """Alias for ``open`` when a script wants explicit reopen wording."""
        self.open(page=page, block=block)

    def close(self) -> None:
        """Close the plot-book window and any owned Matplotlib figures."""
        raise NotImplementedError

    def show_page(self, index: int) -> None:
        """Show one page by zero-based index."""
        raise NotImplementedError

    def export_current_page(self, directory: str | Path) -> Path:
        """Export the currently visible page to an image file."""
        raise NotImplementedError

    def export_all_pages(
        self, directory: str | Path, *, format: str = "png", dpi: int = 180, scope: str = "all"
    ) -> tuple[Path, ...]:
        """Export pages and a JSON manifest; failed pages do not abort the report.

        ``scope='filtered'`` exports the current matching pages. PDF exports
        return one report path; PNG exports return one path per successful page.
        """
        if hasattr(self, "_export_item"):
            return _export_plot_book(self, directory, format=format, dpi=dpi, scope=scope)
        return tuple(self._export_page(index, directory) for index in range(self.page_count))

    def export_filtered_pages(self, directory: str | Path, *, format: str = "png", dpi: int = 180) -> tuple[Path, ...]:
        return self.export_all_pages(directory, format=format, dpi=dpi, scope="filtered")

    def _export_page(self, index: int, directory: str | Path) -> Path:
        raise NotImplementedError


class QtMatplotlibPlotBook(MatplotlibPlotBook):
    """Qt-backed plot book with live, interactive Matplotlib canvases.

    This is the preferred implementation. It keeps every page as a real
    Matplotlib canvas, so 3D rotation, zoom, pan, and toolbar interactions still
    work. It also unregisters embedded figures from pyplot so Spyder does not
    show extra empty figure windows.

    Extension notes
    ---------------
    - Use ``add_figure`` or ``add_entry`` to append pages after construction.
    - Override ``_page_widget`` only for new container-level behavior.
    - Keep CoilGen-specific decisions, such as which plots to include, in the
      calling code or in ``MatplotlibPlotBookBuilder``.
    """

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[PlotBookEntry],
        *,
        title: str,
        preload_pages: str | bool = "idle",
        max_live_canvases: int = 5,
    ) -> "QtMatplotlibPlotBook":
        """Create a live Qt plot book from prebuilt page entries."""
        QtCore, QtWidgets, FigureCanvas, NavigationToolbar = _import_qt_plot_book_widgets()
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        window = QtWidgets.QMainWindow()
        _ensure_readable_qt_font(window)
        window.setWindowTitle(title)
        window.resize(1500, 950)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        stack = QtWidgets.QStackedWidget()
        side_panel = QtWidgets.QWidget()
        side_panel.setMinimumWidth(320)
        side_panel.setSizePolicy(
            _qt_size_policy(QtWidgets, "Preferred"),
            _qt_size_policy(QtWidgets, "Expanding"),
        )
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(4, 4, 4, 4)
        side_layout.setSpacing(5)
        search_edit = QtWidgets.QLineEdit()
        search_edit.setPlaceholderText("Search plots, channels, sections")
        search_edit.setAccessibleName("Search plot pages")
        filter_combo = QtWidgets.QComboBox()
        filter_combo.setToolTip("Filter plot pages by kind.")
        channel_filter_combo = QtWidgets.QComboBox()
        channel_filter_combo.setToolTip("Filter plot pages by project channel.")
        table = QtWidgets.QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(("Plot", "Kind", "Channel", "Status"))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumWidth(300)
        table.setSizePolicy(
            _qt_size_policy(QtWidgets, "Expanding"),
            _qt_size_policy(QtWidgets, "Expanding"),
        )
        table.setHorizontalScrollBarPolicy(_qt_scrollbar_policy(QtCore, "ScrollBarAsNeeded"))
        table.setVerticalScrollBarPolicy(_qt_scrollbar_policy(QtCore, "ScrollBarAsNeeded"))
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
        _set_qt_header_resize_mode(QtWidgets, table.horizontalHeader(), 0, "Interactive")
        _set_qt_header_resize_mode(QtWidgets, table.horizontalHeader(), 1, "ResizeToContents")
        _set_qt_header_resize_mode(QtWidgets, table.horizontalHeader(), 2, "ResizeToContents")
        _set_qt_header_resize_mode(QtWidgets, table.horizontalHeader(), 3, "ResizeToContents")
        export_current_button = QtWidgets.QPushButton("Export Page")
        export_current_button.setToolTip("Export the selected plot page as a PNG file.")
        export_all_button = QtWidgets.QPushButton("Export All")
        export_all_button.setToolTip("Export every plot-book page as a PNG file.")
        side_layout.addWidget(search_edit)
        side_layout.addWidget(QtWidgets.QLabel("Plot type"))
        side_layout.addWidget(filter_combo)
        side_layout.addWidget(QtWidgets.QLabel("Channel"))
        side_layout.addWidget(channel_filter_combo)
        side_layout.addWidget(table, 1)
        status_label = QtWidgets.QLabel()
        status_label.setWordWrap(True)
        side_layout.addWidget(status_label)
        navigation = QtWidgets.QHBoxLayout()
        previous_button = QtWidgets.QPushButton("Previous")
        next_button = QtWidgets.QPushButton("Next")
        navigation.addWidget(previous_button)
        navigation.addWidget(next_button)
        side_layout.addLayout(navigation)
        retry_button = QtWidgets.QPushButton("Retry page")
        side_layout.addWidget(retry_button)
        preload_check = QtWidgets.QCheckBox("Preload nearby pages")
        preload_check.setChecked(_normalize_plot_book_preload_mode(preload_pages) != "none")
        side_layout.addWidget(preload_check)
        side_layout.addWidget(export_current_button)
        export_filtered_button = QtWidgets.QPushButton("Export Filtered")
        side_layout.addWidget(export_filtered_button)
        side_layout.addWidget(export_all_button)
        export_format_combo = QtWidgets.QComboBox()
        export_format_combo.addItems(("PNG", "PDF"))
        export_format_combo.setAccessibleName("Export format")
        side_layout.addWidget(export_format_combo)

        splitter.addWidget(side_panel)
        splitter.addWidget(stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes((440, 1040))
        window.setCentralWidget(splitter)

        book = cls(
            app=app,
            window=window,
            table=table,
            filter_combo=filter_combo,
            channel_filter_combo=channel_filter_combo,
            export_current_button=export_current_button,
            export_all_button=export_all_button,
            stack=stack,
            QtCore=QtCore,
            QtWidgets=QtWidgets,
            FigureCanvas=FigureCanvas,
            NavigationToolbar=NavigationToolbar,
            title=title,
            preload_pages=preload_pages,
            max_live_canvases=max_live_canvases,
        )
        book.search_edit = search_edit
        book.status_label = status_label
        book.previous_button = previous_button
        book.next_button = next_button
        book.retry_button = retry_button
        book.preload_check = preload_check
        book.export_format_combo = export_format_combo
        book.export_filtered_button = export_filtered_button
        search_edit.textChanged.connect(book._set_search_query)
        previous_button.clicked.connect(book.previous)
        next_button.clicked.connect(book.next)
        retry_button.clicked.connect(book.retry_current_page)
        preload_check.toggled.connect(book._toggle_preloading)
        export_filtered_button.clicked.connect(lambda: book._export_dialog("filtered"))
        book._install_shortcuts()
        book.add_entries(entries)
        if book.page_count == 0:
            raise ValueError("A plot book needs at least one page.")
        setattr(window, "_coilgen_plot_book", book)
        _ACTIVE_PLOT_BOOKS.add(book)
        return book

    def __init__(
        self,
        *,
        app: Any,
        window: Any,
        table: Any,
        filter_combo: Any,
        channel_filter_combo: Any,
        export_current_button: Any,
        export_all_button: Any,
        stack: Any,
        QtCore: Any,
        QtWidgets: Any,
        FigureCanvas: Any,
        NavigationToolbar: Any,
        title: str,
        preload_pages: str | bool = "idle",
        max_live_canvases: int = 5,
    ) -> None:
        super().__init__(title=title)
        self.app = app
        self.window = window
        self.table = table
        self.filter_combo = filter_combo
        self.channel_filter_combo = channel_filter_combo
        self.export_current_button = export_current_button
        self.export_all_button = export_all_button
        self.stack = stack
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.FigureCanvas = FigureCanvas
        self.NavigationToolbar = NavigationToolbar
        self._collection = _PageCollection()
        self.entries = self._collection.entries
        self.page_widgets: list[Any | None] = []
        self._row_ids: list[str] = []
        self._last_used: dict[str, int] = {}
        self._use_counter = 0
        self.max_live_canvases = max(1, int(max_live_canvases))
        self.search_query = ""
        self._visible_indices: list[int] = []
        self._has_shown = False
        self._disposed = False
        self.preload_paused = False
        self.category_filter = "All"
        self.channel_filter = "All"
        self.preload_pages = _normalize_plot_book_preload_mode(preload_pages)
        self._preload_active = False
        self._preload_visited_indices: set[int] = set()
        self._preload_visited_ids: set[str] = set()
        self._preload_timer = QtCore.QTimer(window)
        self._preload_timer.setSingleShot(True)
        self._preload_timer.timeout.connect(self._preload_next_idle_page)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        self.filter_combo.currentTextChanged.connect(self._set_category_filter)
        self.channel_filter_combo.currentTextChanged.connect(self._set_channel_filter)
        self.export_current_button.clicked.connect(self._export_current_page_dialog)
        self.export_all_button.clicked.connect(self._export_all_pages_dialog)
        # Native close and programmatic close have the same retained-book
        # semantics. Explicit dispose releases all figures and widgets.
        owner = weakref.ref(self)

        class CloseFilter(QtCore.QObject):
            def eventFilter(self, _watched: Any, event: Any) -> bool:
                if event.type() == QtCore.QEvent.Close:
                    book = owner()
                    if book is not None:
                        book._preload_timer.stop()
                        book._preload_active = False
                        _ACTIVE_PLOT_BOOKS.discard(book)
                        _VISIBLE_QT_PLOT_BOOKS.discard(book)
                return False

        self._close_filter = CloseFilter(window)
        window.installEventFilter(self._close_filter)

    @property
    def page_count(self) -> int:
        return len(self.entries)

    def add_figure(
        self,
        figure: Figure,
        *,
        key: str | None = None,
        short_name: str | None = None,
        title: str | None = None,
        category: str = "Study",
        channel: str = "",
    ) -> PlotBookEntry:
        """Append one already-created Matplotlib figure as a new live page."""
        entry = external_plot_book_entry(
            figure,
            index=self.page_count + 1,
            key=key,
            name=short_name,
            category=category,
            channel=channel,
            title=title,
        )
        self.add_entry(entry)
        return entry

    def add_entry(self, entry: PlotBookEntry) -> None:
        self.add_entries((entry,))

    def add_entries(self, entries: Sequence[PlotBookEntry]) -> None:
        """Register a batch without evaluating factories or drawing canvases."""
        self._collection.add(entries)
        self._sync_collection()

    def _sync_collection(self) -> None:
        initial = not self._row_ids
        selected_id = self._row_ids[self.index] if 0 <= self.index < len(self._row_ids) else None
        old_widgets = {page_id: self.stack.widget(i) for i, page_id in enumerate(self._row_ids)}
        live_widgets = {page_id: self.page_widgets[i] for i, page_id in enumerate(self._row_ids)}
        for widget in old_widgets.values():
            self.stack.removeWidget(widget)
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(self.page_count)
            self.page_widgets = []
            for index, entry in enumerate(self.entries):
                self.stack.addWidget(old_widgets.pop(entry.page_id, None) or self._placeholder_page_widget(entry))
                self.page_widgets.append(live_widgets.get(entry.page_id))
                for column, value in enumerate((entry.short_name, entry.category, entry.channel, "")):
                    item = self.QtWidgets.QTableWidgetItem(value)
                    item.setToolTip(entry.title)
                    self.table.setItem(index, column, item)
                self._update_row(index)
            self._row_ids = [entry.page_id for entry in self.entries]
            self.index = (
                self._row_ids.index(selected_id) if selected_id in self._row_ids else (0 if self.entries else -1)
            )
            if initial:
                _resize_plot_book_table_columns(self.table)
        finally:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)
        for widget in old_widgets.values():
            widget.deleteLater()
        self._refresh_filters()

    def _update_row(self, index: int) -> None:
        entry = self.entries[index]
        state = self._collection.states[entry.page_id]
        for column, value in enumerate((entry.short_name, entry.category, entry.channel, state.status)):
            item = self.table.item(index, column)
            if item is not None:
                item.setText(value)
                detail = f"{entry.title}\n{entry.section}" if entry.section else entry.title
                if state.error is not None:
                    detail += f"\n{type(state.error).__name__}: {state.error}"
                item.setToolTip(detail)

    def show(self, *, block: bool = False) -> None:
        if self._disposed:
            raise RuntimeError("This plot book was disposed; build a new book to inspect it.")
        _ACTIVE_PLOT_BOOKS.add(self)
        _VISIBLE_QT_PLOT_BOOKS.add(self)
        self._has_shown = True
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        # Paint the shell before any expensive factory can run.
        self.app.processEvents()
        if self.index >= 0:
            self.QtCore.QTimer.singleShot(0, self._show_selected_if_visible)
        if block:
            self.app.exec()

    def _show_selected_if_visible(self) -> None:
        if not self._disposed and self.window.isVisible() and self.index >= 0:
            self.show_page(self.index)

    def close(self) -> None:
        """Hide the retained book; ``open()`` preserves its pages and view state."""
        self._preload_timer.stop()
        self._preload_active = False
        _ACTIVE_PLOT_BOOKS.discard(self)
        _VISIBLE_QT_PLOT_BOOKS.discard(self)
        self.window.close()

    def dispose(self) -> None:
        """Permanently release the book's figures, canvases and Qt window."""
        if self._disposed:
            return
        self.close()
        self._disposed = True
        self._collection.dispose()
        self.page_widgets.clear()
        self._row_ids.clear()
        self.window.deleteLater()

    def show_page(self, index: int) -> None:
        if not self.entries:
            return
        bounded = max(0, min(int(index), self.page_count - 1))
        if not self._matches_filters(self.entries[bounded]):
            # An explicit programmatic selection takes precedence, with the UI
            # filters reset so they cannot contradict the displayed page.
            self.category_filter = self.channel_filter = "All"
            self.search_query = ""
            self.search_edit.blockSignals(True)
            self.search_edit.clear()
            self.search_edit.blockSignals(False)
            for combo in (self.filter_combo, self.channel_filter_combo):
                combo.blockSignals(True)
                combo.setCurrentText("All")
                combo.blockSignals(False)
            self._apply_table_filters(render=False)
        self.index = bounded
        self._ensure_page_widget(self.index)
        entry = self.entries[self.index]
        self.stack.setCurrentIndex(self.index)
        self.window.setWindowTitle(f"{self.title}: {entry.title}")
        self.table.blockSignals(True)
        self.table.selectRow(self.index)
        self.table.blockSignals(False)
        self._touch(entry.page_id)
        self._trim_canvas_cache()
        self._update_status()
        self._start_idle_preload()

    def _touch(self, page_id: str) -> None:
        self._use_counter += 1
        self._last_used[page_id] = self._use_counter

    def _desired_preload_indices(self) -> tuple[int, ...]:
        visible = self._visible_indices
        if self.index not in visible:
            return ()
        order = _plot_book_preload_order(visible.index(self.index), len(visible))
        limit = len(order) if self.preload_pages == "all" else self.max_live_canvases
        return tuple(visible[position] for position in order[:limit])

    def _start_idle_preload(self) -> None:
        if self.preload_pages == "none" or self.preload_paused or self._preload_active or not self.entries:
            return
        if not self.window.isVisible() or self._next_preload_index() is None:
            return
        self._preload_active = True
        self._schedule_idle_preload(delay_ms=0 if self.preload_pages == "all" else 120)

    def _schedule_idle_preload(self, *, delay_ms: int) -> None:
        self._preload_timer.start(max(0, int(delay_ms)))

    def pause_preloading(self, paused: bool = True) -> None:
        self.preload_paused = bool(paused)
        if paused:
            self._preload_timer.stop()
            self._preload_active = False
        else:
            self._start_idle_preload()

    def _toggle_preloading(self, enabled: bool) -> None:
        if enabled and self.preload_pages == "none":
            self.preload_pages = "idle"
        self.pause_preloading(not enabled)

    def _preload_next_idle_page(self) -> None:
        self._preload_active = False
        if self.preload_paused or self.preload_pages == "none" or not self.window.isVisible():
            return
        index = self._next_preload_index()
        if index is not None:
            self._preload_visited_ids.add(self.entries[index].page_id)
            self._ensure_page_widget(index)
            self._update_status()
        self._start_idle_preload()

    def _next_preload_index(self) -> int | None:
        for index in self._desired_preload_indices():
            entry = self.entries[index]
            if self.preload_pages == "all" and entry.page_id in self._preload_visited_ids:
                continue
            state = self._collection.states[entry.page_id]
            if self.page_widgets[index] is None and state.error is None:
                return index
        return None

    def _trim_canvas_cache(self) -> None:
        live = [
            i
            for i, widget in enumerate(self.page_widgets)
            if widget is not None and self._collection.states[self.entries[i].page_id].status == "ready"
        ]
        preferred = set(self._desired_preload_indices()) if self.preload_pages != "all" else {self.index}
        candidates = sorted(
            (i for i in live if i != self.index),
            key=lambda i: (i in preferred, self._last_used.get(self.entries[i].page_id, 0)),
        )
        for index in candidates[: max(0, len(live) - self.max_live_canvases)]:
            # Keep scientific artists/data and camera state, release the costly
            # Qt canvas/renderer. Revisit embeds this same Figure without
            # rerunning a potentially stateful user factory.
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            _replace_figure_canvas(self.entries[index].figure, FigureCanvasAgg)
            self._replace_page_widget(index, self._placeholder_page_widget(self.entries[index]))
            self.page_widgets[index] = None

    def export_current_page(self, directory: str | Path, *, format: str = "png", dpi: int = 180) -> Path:
        if self.index < 0 or self.index not in self._visible_indices:
            raise ValueError("No plot page is selected.")
        paths = _export_plot_book(self, directory, format=format, dpi=dpi, scope="current")
        if not paths:
            raise ValueError(f"The selected page could not be exported; see {self.last_export_manifest}.")
        return paths[0]

    def _export_page(self, index: int, directory: str | Path) -> Path:
        paths = _export_plot_book(self, directory, format="png", dpi=180, scope="current", current_index=index)
        if not paths:
            raise ValueError("The requested page is unavailable.")
        return paths[0]

    def _export_item(self, index: int) -> tuple[PlotBookEntry, _PageState]:
        # Exports create artists only: they do not instantiate a toolbar,
        # scroll area, or book canvas for every report page.
        before_ids = tuple(self._row_ids)
        self._collection.materialize(index)
        if before_ids != tuple(entry.page_id for entry in self.entries):
            self._sync_collection()
        else:
            self._update_row(index)
        entry = self.entries[index]
        self._update_status()
        return entry, self._collection.states[entry.page_id]

    def _ensure_page_widget(self, index: int) -> None:
        index = max(0, min(int(index), self.page_count - 1))
        if self.page_widgets[index] is not None:
            return
        entry, state = self._export_item(index)
        if state.error is not None:
            page_widget = self._unavailable_page_widget(entry, state.error)
        else:
            try:
                page_widget = self._page_widget(entry)
            except Exception as exc:
                state.status = "failed"
                state.error = exc
                page_widget = self._unavailable_page_widget(entry, exc)
                self._update_row(index)
        self._replace_page_widget(index, page_widget)
        self._touch(entry.page_id)
        self._trim_canvas_cache()

    def _replace_entry(self, index: int, entry: PlotBookEntry) -> None:
        self.entries[index] = entry
        self._update_row(index)

    def _replace_page_widget(self, index: int, page_widget: Any) -> None:
        old_widget = self.stack.widget(index)
        current_widget = self.stack.currentWidget()
        self.stack.removeWidget(old_widget)
        self.stack.insertWidget(index, page_widget)
        self.page_widgets[index] = page_widget
        if index == self.index:
            self.stack.setCurrentWidget(page_widget)
        elif current_widget is not old_widget:
            self.stack.setCurrentWidget(current_widget)
        old_widget.deleteLater()

    def _unavailable_page_widget(self, entry: PlotBookEntry, exc: Exception) -> Any:
        page = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QVBoxLayout(page)
        label = self.QtWidgets.QLabel(
            f"{entry.title}\n\n{type(exc).__name__}: {exc}\n\nUse Retry page after correcting the cause."
        )
        label.setAlignment(self.QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        label.setTextInteractionFlags(self.QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(label, 1)
        return page

    def _placeholder_page_widget(self, entry: PlotBookEntry) -> Any:
        # Bare placeholders avoid allocating a label/layout for every deferred
        # page. Status/progress information is shared in the side panel.
        return self.QtWidgets.QWidget()

    def retry_current_page(self) -> None:
        if self.index < 0:
            return
        entry = self.entries[self.index]
        state = self._collection.states[entry.page_id]
        state.error = None
        state.status = "deferred" if _entry_is_deferred(entry) else "ready"
        self._replace_page_widget(self.index, self._placeholder_page_widget(entry))
        self.page_widgets[self.index] = None
        self.show_page(self.index)

    def _refresh_filters(self) -> None:
        for combo, field, current in (
            (self.filter_combo, "category", self.category_filter),
            (self.channel_filter_combo, "channel", self.channel_filter),
        ):
            values = ["All"] + sorted({getattr(entry, field) for entry in self.entries if getattr(entry, field)})
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(values)
            combo.setCurrentText(current if current in values else "All")
            combo.blockSignals(False)
        self.category_filter = self.filter_combo.currentText() or "All"
        self.channel_filter = self.channel_filter_combo.currentText() or "All"
        self._apply_table_filters(render=False)

    def _refresh_category_filter(self) -> None:
        self._refresh_filters()

    def _refresh_channel_filter(self) -> None:
        self._refresh_filters()

    def _set_category_filter(self, category: str) -> None:
        self.category_filter = str(category) or "All"
        self._apply_table_filters()

    def _set_channel_filter(self, channel: str) -> None:
        self.channel_filter = str(channel) or "All"
        self._apply_table_filters()

    def _set_search_query(self, text: str) -> None:
        self.search_query = str(text).strip().casefold()
        self._apply_table_filters()

    def _matches_filters(self, entry: PlotBookEntry) -> bool:
        category_matches = self.category_filter == "All" or entry.category == self.category_filter
        channel_matches = self.channel_filter == "All" or entry.channel == self.channel_filter
        haystack = " ".join((entry.title, entry.short_name, entry.key, entry.channel, entry.section)).casefold()
        return category_matches and channel_matches and all(word in haystack for word in self.search_query.split())

    def _apply_table_filters(self, *, render: bool = True) -> None:
        self._visible_indices = [i for i, entry in enumerate(self.entries) if self._matches_filters(entry)]
        visible = set(self._visible_indices)
        for row in range(self.page_count):
            self.table.setRowHidden(row, row not in visible)
        if not visible:
            self.index = -1
            self.table.blockSignals(True)
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
            self.table.blockSignals(False)
            self.stack.hide()
        else:
            self.stack.show()
            if self.index not in visible:
                self.index = self._visible_indices[0]
            self.table.blockSignals(True)
            self.table.selectRow(self.index)
            self.table.blockSignals(False)
            self.stack.setCurrentIndex(self.index)
            if render and self._has_shown:
                self.show_page(self.index)
        self._update_status()

    def _update_status(self) -> None:
        selected = 0 <= self.index < self.page_count and self.index in self._visible_indices
        state = self._collection.states[self.entries[self.index].page_id] if selected else None
        ready = sum(state.status == "ready" for state in self._collection.states.values())
        failed = sum(state.error is not None for state in self._collection.states.values())
        text = f"{len(self._visible_indices)}/{self.page_count} pages · {ready} ready · {failed} unavailable/failed"
        if not self._visible_indices:
            text = "No pages match these filters. Clear the search or select All."
        self.status_label.setText(text)
        self.export_current_button.setEnabled(selected and state.error is None)
        self.retry_button.setEnabled(selected and state.error is not None)
        self.previous_button.setEnabled(bool(self._visible_indices))
        self.next_button.setEnabled(bool(self._visible_indices))

    def previous(self) -> None:
        self._navigate_visible(-1)

    def next(self) -> None:
        self._navigate_visible(1)

    def _navigate_visible(self, delta: int) -> None:
        if self._visible_indices:
            position = self._visible_indices.index(self.index) if self.index in self._visible_indices else 0
            self.show_page(self._visible_indices[(position + delta) % len(self._visible_indices)])

    def _install_shortcuts(self) -> None:
        from matplotlib.backends.qt_compat import QtGui

        shortcut_class = getattr(QtGui, "QShortcut", None) or self.QtWidgets.QShortcut
        self._shortcuts = []
        for key, callback in (
            ("Ctrl+F", self.search_edit.setFocus),
            ("Alt+Left", self.previous),
            ("Alt+Right", self.next),
        ):
            shortcut = shortcut_class(QtGui.QKeySequence(key), self.window)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _export_dialog(self, scope: str) -> None:
        directory = self.QtWidgets.QFileDialog.getExistingDirectory(self.window, "Export CoilGen plot pages")
        if not directory:
            return
        try:
            format = self.export_format_combo.currentText().lower()
            paths = _export_plot_book(self, directory, format=format, dpi=180, scope=scope)
            self.status_label.setText(f"Exported {len(paths)} file(s). Manifest: {self.last_export_manifest.name}")
        except Exception as exc:
            self.QtWidgets.QMessageBox.warning(self.window, "Plot export failed", str(exc))

    def _export_current_page_dialog(self) -> None:
        self._export_dialog("current")

    def _export_all_pages_dialog(self) -> None:
        self._export_dialog("all")

    def _page_widget(self, entry: PlotBookEntry) -> Any:
        """Build the QWidget that contains one page toolbar and canvas.

        Special cases belong here only when they are UI-container concerns. The
        scrollable pages keep the original canvas size and place it in a scroll
        area so long tables and tall metric summaries remain readable.
        """
        page = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        figure = entry.figure
        canvas = figure.canvas
        if not isinstance(canvas, self.QtWidgets.QWidget):
            _retire_figure_manager(figure)
            canvas = _replace_figure_canvas(figure, self.FigureCanvas)
        canvas.setParent(page)
        toolbar = self.NavigationToolbar(canvas, page)
        layout.addWidget(toolbar)
        view_controls = self._view_controls_widget(figure, canvas, page)
        if view_controls is not None:
            layout.addWidget(view_controls)
        if entry.layout == "scroll" or (entry.layout == "auto" and _plot_book_page_is_scrollable(entry.key)):
            scroll_area = self.QtWidgets.QScrollArea(page)
            scroll_area.setWidgetResizable(False)
            scroll_area.setVerticalScrollBarPolicy(_qt_scrollbar_policy(self.QtCore, "ScrollBarAsNeeded"))
            scroll_area.setHorizontalScrollBarPolicy(_qt_scrollbar_policy(self.QtCore, "ScrollBarAsNeeded"))
            # ``canvas.get_width_height()`` can already reflect a Qt layout
            # compression at this point.  Recover the intended native canvas
            # dimensions from the Matplotlib figure so a tall diagnostic page
            # scrolls instead of being scaled down to one viewport.
            width, height = _native_figure_canvas_size(figure, canvas)
            canvas.setMinimumSize(width, height)
            canvas.resize(width, height)
            try:
                canvas.setSizePolicy(self.QtWidgets.QSizePolicy.Fixed, self.QtWidgets.QSizePolicy.Fixed)
                canvas.setFixedSize(width, height)
            except Exception:
                pass
            scroll_area.setWidget(canvas)
            layout.addWidget(scroll_area, 1)
        else:
            layout.addWidget(canvas, 1)
        unregister_pyplot_figure(figure)
        _retire_figure_manager(figure)
        canvas.draw_idle()
        return page

    def _view_controls_widget(self, figure: Figure, canvas: Figure, parent: Any) -> Any | None:
        axes = self._three_d_axes(figure)
        if not axes:
            return None
        widget = self.QtWidgets.QWidget(parent)
        layout = self.QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = self.QtWidgets.QLabel("3D view:")
        label.setToolTip("Set the orientation for all 3D axes on this plot-book page.")
        layout.addWidget(label)
        for name, tooltip, elev, azim in THREE_D_VIEW_PRESETS:
            button = self.QtWidgets.QPushButton(name)
            button.setToolTip(tooltip)
            button.setMaximumWidth(54)
            button.clicked.connect(
                lambda _checked=False, elev=elev, azim=azim: self._apply_3d_view_preset(
                    figure,
                    canvas,
                    elev=elev,
                    azim=azim,
                )
            )
            layout.addWidget(button)
        rotate_label = self.QtWidgets.QLabel("Rotate:")
        rotate_label.setToolTip("Rotate the current 3D camera view in 90 degree steps.")
        layout.addWidget(rotate_label)
        for name, tooltip, axis, quarter_turns in THREE_D_VIEW_ROTATIONS:
            button = self.QtWidgets.QPushButton(name)
            button.setToolTip(tooltip)
            button.setMaximumWidth(64)
            button.clicked.connect(
                lambda _checked=False, axis=axis, quarter_turns=quarter_turns: self._rotate_3d_view(
                    figure,
                    canvas,
                    axis=axis,
                    quarter_turns=quarter_turns,
                )
            )
            layout.addWidget(button)
        layout.addStretch(1)
        return widget

    def _three_d_axes(self, figure: Figure) -> list[Any]:
        axes = []
        for ax in getattr(figure, "axes", ()):
            if hasattr(ax, "view_init") and (hasattr(ax, "get_zlim3d") or getattr(ax, "name", "") == "3d"):
                axes.append(ax)
        return axes

    def _apply_3d_view_preset(self, figure: Figure, canvas: Figure, *, elev: float, azim: float) -> None:
        for ax in self._three_d_axes(figure):
            _set_3d_axes_view(ax, elev=float(elev), azim=float(azim), roll=0.0)
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass
        self._sync_camera_controls(figure)
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def _rotate_3d_view(self, figure: Figure, canvas: Figure, *, axis: str, quarter_turns: int) -> None:
        for ax in self._three_d_axes(figure):
            elev = float(getattr(ax, "elev", 30.0))
            azim = float(getattr(ax, "azim", -60.0))
            roll = float(getattr(ax, "roll", 0.0))
            new_elev, new_azim, new_roll = _rotate_3d_view_angles(
                elev=elev,
                azim=azim,
                roll=roll,
                axis=axis,
                quarter_turns=quarter_turns,
            )
            _set_3d_axes_view(ax, elev=new_elev, azim=new_azim, roll=new_roll)
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass
        self._sync_camera_controls(figure)
        try:
            canvas.draw_idle()
        except Exception:
            pass

    @staticmethod
    def _sync_camera_controls(figure: Figure) -> None:
        controller = getattr(figure, "_coilgen_swept_mesh_inspector", None)
        if controller is not None:
            controller._sync_camera_axis_labels()

    def _on_current_cell_changed(
        self,
        current_row: int,
        _current_col: int,
        _previous_row: int,
        _previous_col: int,
    ) -> None:
        if current_row >= 0:
            self.show_page(current_row)


class ImageMatplotlibPlotBook(MatplotlibPlotBook):
    """Lazy raster viewer for Matplotlib installations without usable Qt."""

    @classmethod
    def from_entries(cls, entries: Sequence[PlotBookEntry], *, title: str, plt: Any) -> "ImageMatplotlibPlotBook":
        if not entries:
            raise ValueError("A plot book needs at least one page.")
        fig = plt.figure(figsize=(12.5, 8.5))
        fig.set_label(title)
        return cls(fig, (), title=title, collection=_PageCollection(entries))

    def __init__(
        self,
        fig: Figure,
        pages: Sequence[PlotBookPage],
        *,
        title: str,
        collection: _PageCollection | None = None,
    ) -> None:
        super().__init__(title=title)
        if not pages and (collection is None or not collection.entries):
            raise ValueError("A plot book needs at least one page.")
        self.fig = fig
        self._provided_pages = list(pages)
        self._collection = collection
        self._raster_cache: dict[str, PlotBookPage] = {}
        self.max_cached_pages = 5
        self._closed = False
        self._disposed = False
        self._initialize_navigation()
        _ACTIVE_PLOT_BOOKS.add(self)

    def _initialize_navigation(self) -> None:
        fig = self.fig
        fig.clf()
        setattr(fig, "_coilgen_plot_book", self)
        self.image_ax = fig.add_axes([0.025, 0.145, 0.95, 0.80])
        self.image_ax.axis("off")
        self.status_text = fig.text(0.5, 0.095, "Select a page to render it.", ha="center", va="center", fontsize=9)
        fig.text(
            0.5,
            0.065,
            "Previous/Next or Left/Right arrow keys. Raster preview; export supports PNG/PDF.",
            ha="center",
            va="center",
            fontsize=8,
            color="#475569",
        )
        Button = _import_matplotlib_button()
        self.prev_button = Button(fig.add_axes([0.36, 0.02, 0.10, 0.045]), "Previous")
        self.next_button = Button(fig.add_axes([0.54, 0.02, 0.10, 0.045]), "Next")
        self.prev_button.on_clicked(lambda _event: self.previous())
        self.next_button.on_clicked(lambda _event: self.next())
        self.key_callback_id = fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        fig.canvas.mpl_connect("close_event", self._on_close)

    @property
    def page_count(self) -> int:
        return len(self._collection.entries) if self._collection is not None else len(self._provided_pages)

    @property
    def entries(self) -> list[PlotBookEntry]:
        return self._collection.entries if self._collection is not None else []

    @property
    def pages(self) -> list[PlotBookPage]:
        """Materialized raster pages; explicitly accessing this evaluates all pages."""
        if self._collection is None:
            return self._provided_pages
        pages = []
        index = 0
        while index < self.page_count:
            page = self._raster_page(index)
            if page is not None:
                pages.append(page)
            index += 1
        return pages

    def _raster_page(self, index: int) -> PlotBookPage | None:
        if self._collection is None:
            return self._provided_pages[index]
        entry = self.entries[index]
        if entry.page_id in self._raster_cache:
            page = self._raster_cache.pop(entry.page_id)
            self._raster_cache[entry.page_id] = page
            return page
        entry = self._collection.materialize(index)
        if entry is None:
            return None
        try:
            page = plot_book_page_from_entry(entry)
        except Exception as exc:
            state = self._collection.states[entry.page_id]
            state.status, state.error = "failed", exc
            return None
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        # The RGBA cache owns this rendering; do not also retain its renderer.
        _retire_figure_manager(entry.figure)
        _replace_figure_canvas(entry.figure, FigureCanvasAgg)
        self._raster_cache[entry.page_id] = page
        while len(self._raster_cache) > self.max_cached_pages:
            self._raster_cache.pop(next(iter(self._raster_cache)))
        return page

    def add_page(self, page: PlotBookPage) -> None:
        if self._collection is not None:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            figure = Figure()
            FigureCanvasAgg(figure)
            ax = figure.add_axes((0, 0, 1, 1))
            ax.imshow(page.image_rgba)
            ax.axis("off")
            self.add_figure(figure, key=page.key, title=page.title, category=page.category, channel=page.channel)
        else:
            self._provided_pages.append(page)

    def add_figure(
        self,
        figure: Figure,
        *,
        key: str | None = None,
        short_name: str | None = None,
        title: str | None = None,
        category: str = "Study",
        channel: str = "",
    ) -> PlotBookEntry:
        entry = external_plot_book_entry(
            figure, index=self.page_count + 1, key=key, name=short_name, category=category, channel=channel, title=title
        )
        if self._collection is not None:
            self._collection.add((entry,))
        else:
            self._provided_pages.append(plot_book_page_from_entry(entry))
        return entry

    def show(self, *, block: bool = False) -> None:
        if self._disposed:
            raise RuntimeError("This plot book was disposed; build a new book to inspect it.")
        plt = _import_pyplot()
        if self._closed:
            self.fig = plt.figure(figsize=(12.5, 8.5))
            self.fig.set_label(self.title)
            self._initialize_navigation()
            self._closed = False
        _ACTIVE_PLOT_BOOKS.add(self)
        self.show_page(self.index)
        plt.show(block=block)

    def _on_close(self, _event: Any = None) -> None:
        self._closed = True
        _ACTIVE_PLOT_BOOKS.discard(self)

    def close(self) -> None:
        self._on_close()
        _import_pyplot().close(self.fig)

    def dispose(self) -> None:
        if self._disposed:
            return
        self.close()
        self._disposed = True
        if self._collection is not None:
            self._collection.dispose()
        self._provided_pages.clear()
        self._raster_cache.clear()
        self.fig.clear()

    def show_page(self, index: int) -> None:
        if not self.page_count:
            return
        self.index = int(index) % self.page_count
        page = self._raster_page(self.index)
        self.image_ax.clear()
        self.image_ax.axis("off")
        if page is None:
            entry = self.entries[self.index]
            state = self._collection.states[entry.page_id]
            self.image_ax.text(
                0.5,
                0.5,
                f"{entry.title}\n\n{type(state.error).__name__}: {state.error}",
                ha="center",
                va="center",
                transform=self.image_ax.transAxes,
                wrap=True,
            )
            title = entry.title
        else:
            self.image_ax.imshow(page.image_rgba)
            title = page.title
        self.fig.suptitle(textwrap.fill(f"{self.title}: {title}", width=96), fontsize=12)
        self.status_text.set_text(f"Page {self.index + 1}/{self.page_count}")
        self.fig.canvas.draw_idle()

    def retry_current_page(self) -> None:
        if self._collection is not None:
            entry = self.entries[self.index]
            self._collection.materialize(self.index, retry=True)
            self._raster_cache.pop(entry.page_id, None)
            self.show_page(self.index)

    def _export_item(self, index: int) -> tuple[PlotBookEntry | PlotBookPage, _PageState]:
        if self._collection is None:
            return self._provided_pages[index], _PageState("ready")
        self._collection.materialize(index)
        entry = self.entries[index]
        return entry, self._collection.states[entry.page_id]

    def _matches_filters(self, _entry: PlotBookEntry | PlotBookPage) -> bool:
        return True

    def export_current_page(self, directory: str | Path, *, format: str = "png", dpi: int = 180) -> Path:
        paths = _export_plot_book(self, directory, format=format, dpi=dpi, scope="current")
        if not paths:
            raise ValueError(f"The selected page could not be exported; see {self.last_export_manifest}.")
        return paths[0]

    def _export_page(self, index: int, directory: str | Path) -> Path:
        paths = _export_plot_book(self, directory, format="png", dpi=180, scope="current", current_index=index)
        if not paths:
            raise ValueError("The requested page is unavailable.")
        return paths[0]

    def next(self) -> None:
        self.show_page(self.index + 1)

    def previous(self) -> None:
        self.show_page(self.index - 1)

    def _on_key_press(self, event: Any) -> None:
        if event.key in {"right", "down", "pagedown", "n"}:
            self.next()
        elif event.key in {"left", "up", "pageup", "p"}:
            self.previous()


class MatplotlibPlotBookBuilder:
    """Collect Matplotlib figures and build the best available plot book.

    The builder is the preferred extension point. It separates plot creation
    from plot-book UI construction:

    - ``add_specs`` registers plot specifications whose ``factory`` functions
      return Matplotlib figures when a page is first selected.
    - ``add_figures`` adds existing figures from scripts or parameter studies.
    - ``build`` first tries the live Qt implementation and falls back to the
      raster image implementation if Qt is unavailable.

    A plot specification passed to ``add_spec`` must provide ``key``, ``title``,
    ``category``, and ``factory`` attributes. This loose protocol keeps the
    builder independent of CoilGen-specific plot registries.
    """

    def __init__(
        self,
        *,
        title: str,
        plt: Any,
        timing_report: list[dict[str, float | int | str]] | None = None,
        preload_pages: str | bool = "idle",
        max_live_canvases: int = 5,
    ) -> None:
        self.title = str(title)
        self.plt = plt
        self.timing_report = timing_report
        self.preload_pages = _normalize_plot_book_preload_mode(preload_pages)
        self.max_live_canvases = max(1, int(max_live_canvases))
        self.entries: list[PlotBookEntry] = []

    def add_specs(
        self,
        specs: Sequence[PlotSpec],
        *,
        short_name_factory: ShortNameFactory,
    ) -> "MatplotlibPlotBookBuilder":
        """Evaluate and add multiple plot specifications."""
        for spec in specs:
            self.add_spec(spec, short_name_factory=short_name_factory)
        return self

    def add_spec(
        self,
        spec: PlotSpec,
        *,
        short_name_factory: ShortNameFactory,
    ) -> "MatplotlibPlotBookBuilder":
        """Register one plot specification for lazy plot-book evaluation.

        ``ValueError`` from a factory means that the plot is not applicable for
        the current result, for example a wire-path plot before routing exists;
        live books show that page as unavailable if selected. Raster fallback
        skips unavailable pages while materializing.
        """
        deferred = _DeferredPlotBookFigures(
            key=str(getattr(spec, "key", "")),
            title=str(getattr(spec, "title", "") or getattr(spec, "key", "") or "Plot"),
            category=str(getattr(spec, "category", "") or "Study"),
            factory=getattr(spec, "factory"),
            short_name_factory=short_name_factory,
            plt=self.plt,
            timing_report=self.timing_report,
            short_name=str(getattr(spec, "short_name", "") or ""),
            channel=str(getattr(spec, "channel", "") or ""),
            page_id=str(getattr(spec, "page_id", "") or ""),
            section=str(getattr(spec, "section", "") or ""),
            layout=str(getattr(spec, "layout", "auto") or "auto"),
        )
        self.add_entry(deferred.placeholder_entry())
        return self

    def add_figures(
        self,
        figures: Sequence[Figure] | None,
        *,
        names: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        channels: Sequence[str] | None = None,
    ) -> "MatplotlibPlotBookBuilder":
        """Add existing figures, optionally with page names, categories, and channels."""
        if not figures:
            return self
        name_list = [str(name).strip() for name in (names or ())]
        category_list = [str(category).strip() for category in (categories or ())]
        channel_list = [str(channel).strip() for channel in (channels or ())]
        for index, figure in enumerate(figures, start=1):
            if figure is None:
                continue
            name = name_list[index - 1] if index - 1 < len(name_list) else ""
            category = category_list[index - 1] if index - 1 < len(category_list) else "Study"
            channel = channel_list[index - 1] if index - 1 < len(channel_list) else ""
            self.add_figure(figure, name=name, category=category, channel=channel)
        return self

    def add_definition(self, definition: StudyPlotBookDefinition) -> "MatplotlibPlotBookBuilder":
        """Add every enabled page from a generic study plot-book definition."""
        for page in definition.enabled_pages():
            self.add_spec(page, short_name_factory=_study_plot_page_short_name)
        return self

    def add_figure(
        self,
        figure: Figure,
        *,
        key: str | None = None,
        name: str | None = None,
        category: str = "Study",
        channel: str = "",
        title: str | None = None,
    ) -> "MatplotlibPlotBookBuilder":
        """Add one existing figure and infer missing page metadata."""
        self.add_entry(
            external_plot_book_entry(
                figure,
                index=len(self.entries) + 1,
                key=key,
                name=name,
                category=category,
                channel=channel,
                title=title,
            )
        )
        return self

    def add_entry(self, entry: PlotBookEntry) -> "MatplotlibPlotBookBuilder":
        """Add a fully specified live entry without evaluating a factory."""
        if not _entry_is_deferred(entry):
            hide_pyplot_manager_window(entry.figure)
        self.entries.append(entry)
        return self

    def build(self) -> MatplotlibPlotBook:
        """Build a Qt plot book when possible, otherwise build image fallback."""
        if not self.entries:
            raise ValueError("A plot book needs at least one page.")
        try:
            return QtMatplotlibPlotBook.from_entries(
                self.entries,
                title=self.title,
                preload_pages=self.preload_pages,
                max_live_canvases=self.max_live_canvases,
            )
        except Exception as exc:
            if self.timing_report is not None:
                self.timing_report.append(
                    {
                        "plot": f"book:live_qt_unavailable ({type(exc).__name__})",
                        "seconds": 0.0,
                        "figures": 0,
                    }
                )
            return ImageMatplotlibPlotBook.from_entries(self.entries, title=self.title, plt=self.plt)


def plot_study_plot_book(
    definition: StudyPlotBookDefinition,
    *,
    show: bool = True,
    block: bool = False,
    timing_report: list[dict[str, float | int | str]] | None = None,
    preload_pages: str | bool = "idle",
) -> MatplotlibPlotBook:
    """Build and optionally show a plot book from a generic study definition."""

    if show:
        _activate_plot_book_backend()
    plt = _import_pyplot()
    builder = MatplotlibPlotBookBuilder(
        title=definition.title,
        plt=plt,
        timing_report=timing_report,
        preload_pages=preload_pages,
    )
    builder.add_definition(definition)
    book = builder.build()
    if show:
        show_start = time.perf_counter()
        book.show(block=block)
        if timing_report is not None:
            timing_report.append({"plot": "show", "seconds": float(time.perf_counter() - show_start), "figures": 1})
    return book


def _activate_plot_book_backend() -> str:
    try:
        from .matplotlib_backend import activate_matplotlib_interactive_backend

        return activate_matplotlib_interactive_backend()
    except Exception:
        return ""


def close_active_plot_books() -> None:
    """Close live plot-book windows tracked by the CoilGen plotting API."""

    while _ACTIVE_PLOT_BOOKS:
        book = _ACTIVE_PLOT_BOOKS.pop()
        try:
            book.close()
        except Exception:
            pass


def _figure_size(figure: Figure) -> tuple[float, float] | None:
    try:
        width, height = figure.get_size_inches()
        return float(width), float(height)
    except (AttributeError, TypeError, ValueError):
        return None


def _replace_figure_canvas(figure: Figure, canvas_class: Any) -> Any:
    """Rebind legacy Matplotlib widgets when changing a retained canvas.

    Current AxesWidget.canvas is a property; supported older Matplotlib
    versions store it directly. Figure-level callbacks survive canvas changes.
    """
    old_canvas = figure.canvas
    canvas = canvas_class(figure)
    candidates = list(vars(figure).values())
    for name in ("_coilgen_slice_inspector", "_coilgen_gradient_slice_inspector", "_coilgen_swept_mesh_inspector"):
        controller = getattr(figure, name, None)
        if controller is not None and hasattr(controller, "__dict__"):
            candidates.extend(vars(controller).values())
    for callbacks in canvas.callbacks.callbacks.values():
        for reference in callbacks.values():
            callback = reference()
            owner = getattr(callback, "__self__", None)
            if owner is not None:
                candidates.append(owner)
    for candidate in candidates:
        if hasattr(candidate, "__dict__") and vars(candidate).get("canvas") is old_canvas:
            candidate.canvas = canvas
    return canvas


def _retire_figure_manager(figure: Figure) -> None:
    """Release the old pyplot window without closing retained figure widgets."""
    canvas = getattr(figure, "canvas", None)
    manager = getattr(canvas, "manager", None)
    unregister_pyplot_figure(figure)
    if manager is not None:
        try:
            with canvas.callbacks.blocked(signal="close_event"):
                manager.destroy()
        except (AttributeError, RuntimeError):
            pass
        canvas.manager = None


def _dispose_plot_book_figure(figure: Figure) -> None:
    for name in ("_coilgen_slice_inspector", "_coilgen_gradient_slice_inspector", "_coilgen_swept_mesh_inspector"):
        controller = getattr(figure, name, None)
        if controller is not None:
            controller.disconnect()
    _retire_figure_manager(figure)
    try:
        figure.clear()
    except (AttributeError, RuntimeError):
        pass


def _export_plot_book(
    book: MatplotlibPlotBook,
    directory: str | Path,
    *,
    format: str,
    dpi: int,
    scope: str,
    current_index: int | None = None,
) -> tuple[Path, ...]:
    """Expand siblings in source order and record every success/failure."""
    format = str(format).lower()
    if format not in {"png", "pdf", "svg"}:
        raise ValueError("Export format must be 'png', 'pdf', or 'svg'.")
    if scope not in {"all", "filtered", "current"}:
        raise ValueError("Export scope must be 'all', 'filtered', or 'current'.")
    if int(dpi) <= 0:
        raise ValueError("Export DPI must be positive.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    selected_index = book.index if current_index is None else int(current_index)
    if scope == "current" and not 0 <= selected_index < book.page_count:
        raise ValueError("No plot page is selected.")
    pdf_path = directory / f"{_safe_file_stem(book.title)}_{scope}.pdf"
    cancelled = False
    software_credit = f"{SOFTWARE_NAME} — {AUTHOR_CREDIT}"
    was_paused = getattr(book, "preload_paused", None)
    if was_paused is not None:
        book.pause_preloading(True)
    try:
        with ExitStack() as cleanup:
            pdf = None
            if format == "pdf":
                from matplotlib.backends.backend_pdf import PdfPages

                pdf = cleanup.enter_context(
                    PdfPages(pdf_path, metadata={"Title": book.title, "Creator": software_credit})
                )
            index = selected_index if scope == "current" else 0
            while index < book.page_count:
                raw_entry = book.entries[index] if book.entries else book._provided_pages[index]
                if scope == "filtered" and not book._matches_filters(raw_entry):
                    index += 1
                    continue
                entry, state = book._export_item(index)
                row: dict[str, Any] = {
                    "page_id": entry.page_id or f"page-{index + 1}",
                    "index": index + 1,
                    "title": entry.title,
                    "key": entry.key,
                    "category": entry.category,
                    "channel": entry.channel,
                    "section": entry.section,
                    "status": state.status,
                }
                if state.error is not None:
                    row["error"] = f"{type(state.error).__name__}: {state.error}"
                else:
                    temporary_figure = None
                    figure = None
                    saved_size = None
                    live_canvas = isinstance(book, QtMatplotlibPlotBook) and book.page_widgets[index] is not None
                    try:
                        if isinstance(entry, PlotBookPage):
                            if format == "png":
                                from matplotlib.image import imsave

                                path = directory / f"{index + 1:03d}_{_safe_file_stem(row['page_id'])}.png"
                                imsave(path, entry.image_rgba, metadata={"Software": software_credit})
                            else:
                                from matplotlib.figure import Figure
                                from matplotlib.backends.backend_agg import FigureCanvasAgg

                                height, width = entry.image_rgba.shape[:2]
                                temporary_figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                                FigureCanvasAgg(temporary_figure)
                                ax = temporary_figure.add_axes((0, 0, 1, 1))
                                ax.imshow(entry.image_rgba)
                                ax.axis("off")
                                figure = temporary_figure
                        else:
                            figure = entry.figure
                            saved_size = _figure_size(figure)
                            if state.original_size is not None:
                                figure.set_size_inches(state.original_size, forward=False)
                            if not live_canvas:
                                from matplotlib.backends.backend_agg import FigureCanvasAgg

                                _retire_figure_manager(figure)
                                _replace_figure_canvas(figure, FigureCanvasAgg)
                        if figure is not None:
                            if pdf is not None:
                                pdf.savefig(figure, dpi=dpi)
                                path = pdf_path
                                row["pdf_page"] = pdf.get_pagecount()
                            else:
                                path = directory / f"{index + 1:03d}_{_safe_file_stem(row['page_id'])}.{format}"
                                figure.savefig(
                                    path,
                                    format=format,
                                    dpi=dpi,
                                    metadata={"Creator" if format == "svg" else "Software": software_credit},
                                )
                        row["status"] = "exported"
                        row["file"] = path.name
                        if path not in paths:
                            paths.append(path)
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        try:
                            if figure is not None:
                                if saved_size is not None:
                                    figure.set_size_inches(saved_size, forward=False)
                                if not live_canvas:
                                    from matplotlib.backends.backend_agg import FigureCanvasAgg

                                    _replace_figure_canvas(figure, FigureCanvasAgg)
                            if temporary_figure is not None:
                                temporary_figure.clear()
                        except (AttributeError, RuntimeError, TypeError):
                            # A malformed factory return or damaged canvas is
                            # already recorded above; cleanup must not prevent
                            # the remaining report pages being exported.
                            pass
                rows.append(row)
                if scope == "current":
                    break
                index += 1
    finally:
        manifest = directory / f"plot_book_{scope}_{format}.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "software": software_attribution(),
                    "title": book.title,
                    "scope": scope,
                    "format": format,
                    "dpi": int(dpi),
                    "cancelled": cancelled,
                    "pages": rows,
                    "metadata": getattr(book, "export_metadata", {}),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        book.last_export_manifest = manifest
        if was_paused is not None:
            book.pause_preloading(was_paused)
    return tuple(paths)


def external_plot_book_entry(
    figure: Figure,
    *,
    index: int,
    key: str | None = None,
    name: str | None = None,
    category: str = "Study",
    channel: str = "",
    title: str | None = None,
) -> PlotBookEntry:
    fallback = str(name or "").strip() or f"Plot {index}"
    resolved_title = str(title).strip() if title else source_figure_title(figure, fallback=fallback)
    short_name = compact_plot_book_label(fallback or resolved_title)
    return PlotBookEntry(
        key=str(key or external_plot_book_key(fallback, index=index)),
        short_name=short_name,
        title=resolved_title,
        category=str(category or "Study"),
        figure=figure,
        channel=str(channel or ""),
    )


def _entry_is_deferred(entry: PlotBookEntry) -> bool:
    return isinstance(entry.figure, _DeferredPlotBookFigures)


def _normalize_plot_book_preload_mode(value: str | bool) -> str:
    if isinstance(value, bool):
        return "idle" if value else "none"
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "none", "off", "false", "0", "disabled", "disable", "no"}:
        return "none"
    if text in {"idle", "background", "queued", "queue", "true", "1", "on", "yes"}:
        return "idle"
    if text in {"all", "eager"}:
        return "all"
    raise ValueError("preload_pages must be 'none', 'idle', 'all', True, or False.")


def _plot_book_preload_order(current_index: int, page_count: int) -> tuple[int, ...]:
    count = max(0, int(page_count))
    if count <= 0:
        return ()
    current = max(0, min(int(current_index), count - 1))
    order = [current]
    for distance in range(1, count):
        forward = current + distance
        backward = current - distance
        if forward < count:
            order.append(forward)
        if backward >= 0:
            order.append(backward)
    return tuple(order)


def _materialize_deferred_entry(entry: PlotBookEntry) -> tuple[PlotBookEntry, ...]:
    if _entry_is_deferred(entry):
        return entry.figure.materialize()
    return (entry,)


def plot_book_page_from_entry(entry: PlotBookEntry) -> PlotBookPage:
    if not hasattr(entry.figure.canvas, "buffer_rgba"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        _retire_figure_manager(entry.figure)
        _replace_figure_canvas(entry.figure, FigureCanvasAgg)
    entry.figure.canvas.draw()
    width, height = entry.figure.canvas.get_width_height()
    image = np.asarray(entry.figure.canvas.buffer_rgba()).reshape((height, width, 4)).copy()
    return PlotBookPage(
        key=entry.key,
        title=entry.title,
        category=entry.category,
        image_rgba=image,
        channel=entry.channel,
        page_id=entry.page_id,
        section=entry.section,
    )


def source_figure_title(fig: Figure, *, fallback: str) -> str:
    label = str(getattr(fig, "get_label", lambda: "")()).strip()
    if label:
        return label
    try:
        title = str(fig.canvas.manager.get_window_title()).strip()
        if title and not (title.startswith("Figure ") and title.removeprefix("Figure ").isdigit()):
            return title
    except Exception:
        pass
    return str(fallback)


def compact_plot_book_label(label: str, *, max_length: int = 32) -> str:
    text = " ".join(str(label).split())
    return text[:max_length] if text else "plot"


def _study_plot_page_short_name(spec: PlotSpec, title: str, index: int, total: int) -> str:
    short_name = str(getattr(spec, "short_name", "") or "").strip()
    label = short_name or str(title or getattr(spec, "title", "") or getattr(spec, "key", "")).strip()
    if total > 1:
        label = f"{label} {index}/{total}"
    return compact_plot_book_label(label)


def _safe_file_stem(value: str) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value).strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:80] or "plot"


def external_plot_book_key(name: str, *, index: int) -> str:
    text = str(name or "").strip().lower()
    key = "".join(char if char.isalnum() else "_" for char in text)
    key = "_".join(part for part in key.split("_") if part)
    return key or f"external_{index}"


def _set_3d_axes_view(ax: Any, *, elev: float, azim: float, roll: float) -> None:
    try:
        ax.view_init(elev=float(elev), azim=float(azim), roll=float(roll))
    except TypeError:
        ax.view_init(elev=float(elev), azim=float(azim))


def _rotate_3d_view_angles(
    *,
    elev: float,
    azim: float,
    roll: float,
    axis: str,
    quarter_turns: int,
) -> tuple[float, float, float]:
    view_direction, up_direction = _view_direction_and_up(elev=elev, azim=azim, roll=roll)
    rotation = _world_axis_quarter_turn_matrix(axis, quarter_turns)
    rotated_view = rotation @ view_direction
    rotated_up = rotation @ up_direction
    return _angles_from_view_direction_and_up(rotated_view, rotated_up)


def _view_direction_and_up(*, elev: float, azim: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    roll_rad = np.deg2rad(float(roll))
    cos_elev = float(np.cos(elev_rad))
    view = _unit_vector(
        np.array(
            [
                cos_elev * float(np.cos(azim_rad)),
                cos_elev * float(np.sin(azim_rad)),
                float(np.sin(elev_rad)),
            ],
            dtype=float,
        )
    )
    up_without_roll = _baseline_up_vector(view, elev=float(elev), azim=float(azim))
    up = _rotate_vector_about_axis(up_without_roll, view, -roll_rad)
    return view, _unit_vector(up)


def _angles_from_view_direction_and_up(view: np.ndarray, up: np.ndarray) -> tuple[float, float, float]:
    view = _unit_vector(view)
    elev = float(np.rad2deg(np.arcsin(np.clip(view[2], -1.0, 1.0))))
    azim = _wrap_degrees(float(np.rad2deg(np.arctan2(view[1], view[0]))))
    baseline_up = _baseline_up_vector(view, elev=elev, azim=azim)
    projected_up = np.asarray(up, dtype=float) - view * float(np.dot(up, view))
    if float(np.linalg.norm(projected_up)) <= 1e-14:
        projected_up = baseline_up
    projected_up = _unit_vector(projected_up)
    roll = float(
        np.rad2deg(
            np.arctan2(
                float(np.dot(np.cross(baseline_up, projected_up), view)),
                float(np.dot(baseline_up, projected_up)),
            )
        )
    )
    return elev, azim, _wrap_degrees(-roll)


def _baseline_up_vector(view: np.ndarray, *, elev: float, azim: float) -> np.ndarray:
    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    up = np.array(
        [
            -float(np.sin(elev_rad)) * float(np.cos(azim_rad)),
            -float(np.sin(elev_rad)) * float(np.sin(azim_rad)),
            float(np.cos(elev_rad)),
        ],
        dtype=float,
    )
    up -= np.asarray(view, dtype=float) * float(np.dot(up, view))
    return _unit_vector(up)


def _world_axis_quarter_turn_matrix(axis: str, quarter_turns: int) -> np.ndarray:
    key = str(axis).strip().lower()
    if key not in {"x", "y", "z"}:
        raise ValueError("3D view rotation axis must be 'x', 'y', or 'z'.")
    angle = 0.5 * np.pi * int(quarter_turns)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    if key == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]], dtype=float)
    if key == "y":
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=float)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotate_vector_about_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit_vector(axis)
    vector = np.asarray(vector, dtype=float)
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return vector * cosine + np.cross(axis, vector) * sine + axis * float(np.dot(axis, vector)) * (1.0 - cosine)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return array / norm


def _wrap_degrees(angle: float) -> float:
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    if abs(wrapped + 180.0) <= 1e-12:
        return 180.0
    return wrapped


def _plot_book_page_is_scrollable(key: str) -> bool:
    normalized = external_plot_book_key(str(key), index=0)
    return any(
        normalized == scrollable_key or normalized.endswith(f"_{scrollable_key}")
        for scrollable_key in SCROLLABLE_PLOT_BOOK_KEYS
    )


def _native_figure_canvas_size(figure: Figure, canvas: Any) -> tuple[int, int]:
    """Return the Matplotlib figure's intended pixel size before Qt scaling."""

    try:
        size_inches = np.asarray(figure.get_size_inches(), dtype=float).reshape(-1)
        dpi = float(figure.get_dpi())
        if size_inches.size >= 2 and np.all(np.isfinite(size_inches[:2])) and np.isfinite(dpi) and dpi > 0.0:
            width = int(round(float(size_inches[0]) * dpi))
            height = int(round(float(size_inches[1]) * dpi))
            if width > 0 and height > 0:
                return width, height
    except Exception:
        pass
    width, height = canvas.get_width_height()
    return max(int(width), 1), max(int(height), 1)


def hide_pyplot_manager_window(figure: Figure) -> None:
    try:
        manager = figure.canvas.manager
        window = getattr(manager, "window", None)
        if window is not None:
            window.hide()
    except Exception:
        pass


@contextmanager
def pyplot_interactive_disabled(plt: Any):
    """Temporarily disable pyplot interactive mode while source pages render."""

    try:
        was_interactive = bool(plt.isinteractive())
    except Exception:
        was_interactive = False
    if was_interactive:
        try:
            plt.ioff()
        except Exception:
            pass
    try:
        yield
    finally:
        if was_interactive:
            try:
                plt.ion()
            except Exception:
                pass


def unregister_pyplot_figure(figure: Figure) -> None:
    """Keep an embedded plot-book canvas out of pyplot's figure registry."""

    hide_pyplot_manager_window(figure)
    try:
        from matplotlib import _pylab_helpers

        manager = figure.canvas.manager
        if manager is not None:
            if hasattr(manager, "_cidgcf"):
                manager.canvas.mpl_disconnect(manager._cidgcf)
            if _pylab_helpers.Gcf.figs.get(manager.num) is manager:
                _pylab_helpers.Gcf.figs.pop(manager.num)
    except Exception:
        pass


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Matplotlib is required for active Spyder plots. Install it in the "
            "Python environment used by Spyder, or use save_plot_gallery for "
            "dependency-free SVG files."
        ) from exc
    return plt


def _import_matplotlib_button() -> Any:
    try:
        from matplotlib.widgets import Button
    except ModuleNotFoundError as exc:
        raise ImportError("Matplotlib widgets are required for plot-book navigation.") from exc
    return Button


def _import_qt_plot_book_widgets() -> tuple[Any, Any, Any, Any]:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    except Exception:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.backends.qt_compat import QtCore, QtWidgets

    return QtCore, QtWidgets, FigureCanvas, NavigationToolbar


def _ensure_readable_qt_font(window: Any) -> None:
    """Supply a font only on runtimes whose default cannot render basic text."""
    from matplotlib import get_data_path
    from matplotlib.backends.qt_compat import QtGui

    if QtGui.QFontMetrics(window.font()).inFontUcs4(ord("A")):
        return
    font_path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
    families = QtGui.QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if families:
        window.setFont(QtGui.QFont(families[0], max(9, window.font().pointSize())))


def _qt_scrollbar_policy(QtCore: Any, name: str) -> Any:
    direct = getattr(QtCore.Qt, name, None)
    if direct is not None:
        return direct
    enum = getattr(QtCore.Qt, "ScrollBarPolicy", None)
    if enum is not None:
        value = getattr(enum, name, None)
        if value is not None:
            return value
    return name


def _qt_size_policy(QtWidgets: Any, name: str) -> Any:
    size_policy = getattr(QtWidgets, "QSizePolicy", None)
    if size_policy is None:
        return name
    direct = getattr(size_policy, name, None)
    if direct is not None:
        return direct
    enum = getattr(size_policy, "Policy", None)
    if enum is not None:
        value = getattr(enum, name, None)
        if value is not None:
            return value
    return name


def _set_qt_header_resize_mode(QtWidgets: Any, header: Any, section: int, mode_name: str) -> None:
    header_view = getattr(QtWidgets, "QHeaderView", None)
    if header_view is None:
        return
    mode = getattr(header_view, mode_name, None)
    enum = getattr(header_view, "ResizeMode", None)
    if mode is None and enum is not None:
        mode = getattr(enum, mode_name, None)
    if mode is None:
        return
    try:
        header.setSectionResizeMode(int(section), mode)
    except Exception:
        try:
            header.setResizeMode(int(section), mode)
        except Exception:
            pass


def _resize_plot_book_table_columns(table: Any) -> None:
    try:
        table.resizeColumnsToContents()
    except Exception:
        pass
    for section, minimum_width in enumerate((160, 70, 70)):
        try:
            table.setColumnWidth(section, max(int(table.columnWidth(section)), minimum_width))
        except Exception:
            pass
