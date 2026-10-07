"""Regression coverage for page state, reporting, and real Qt canvas ownership."""

import json
import os
import gc
from types import SimpleNamespace
from unittest.mock import patch
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from coilgen.plot_book import (
    ImageMatplotlibPlotBook,
    MatplotlibPlotBookBuilder,
    QtMatplotlibPlotBook,
    StudyPlotBookDefinition,
    StudyPlotPage,
    StudyPlotSection,
    _ACTIVE_PLOT_BOOKS,
    _VISIBLE_QT_PLOT_BOOKS,
    _replace_figure_canvas,
    _rotate_3d_view_angles,
    _world_axis_quarter_turn_matrix,
    is_plot_book_rendering,
    plot_book_rendering,
    unregister_pyplot_figure,
)


def figure(label=""):
    fig = plt.figure(figsize=(3, 2))
    fig.set_label(label)
    fig.add_subplot().plot([0, 1], [0, 1])
    return fig


def entries(pages):
    builder = MatplotlibPlotBookBuilder(title="Runtime tests", plt=plt, preload_pages=False)
    builder.add_definition(StudyPlotBookDefinition(title="Runtime tests", pages=tuple(pages)))
    return builder.entries


@pytest.fixture
def books():
    created = []
    yield created
    for book in created:
        book.dispose()
    plt.close("all")


def qt_book(books, pages, **kwargs):
    pytest.importorskip("matplotlib.backends.qt_compat")
    book = QtMatplotlibPlotBook.from_entries(entries(pages), title="Runtime tests", preload_pages=False, **kwargs)
    books.append(book)
    return book


def test_visible_qt_book_stays_alive_until_closed():
    pytest.importorskip("matplotlib.backends.qt_compat")

    def open_without_retaining_locally():
        book = QtMatplotlibPlotBook.from_entries(
            entries([StudyPlotPage("page", "Page", "Field", lambda: figure("Page"))]),
            title="Lifetime test",
            preload_pages=False,
        )
        book.open(block=False)
        book.app.processEvents()
        return weakref.ref(book)

    book_ref = open_without_retaining_locally()
    try:
        gc.collect()
        book = book_ref()
        assert book is not None
        assert book in _VISIBLE_QT_PLOT_BOOKS
        assert book.window.isVisible()
        book.window.close()
        book.app.processEvents()
        assert book not in _VISIBLE_QT_PLOT_BOOKS
    finally:
        if book_ref() is not None:
            book_ref().dispose()


def test_bulk_registration_is_linear_and_never_runs_factories(books):
    calls = []
    pages = [StudyPlotPage(f"page-{i}", f"Page {i}", "Field", lambda: calls.append("unexpected")) for i in range(1000)]
    with patch("coilgen.plot_book._resize_plot_book_table_columns") as resize:
        book = qt_book(books, pages)
    assert calls == []
    assert book.page_count == 1000
    assert resize.call_count == 1
    assert all(widget is None for widget in book.page_widgets)
    assert book.table.rowCount() == 1000


def test_expansion_stays_adjacent_and_leaf_titles_survive(books):
    book = qt_book(
        books,
        [
            StudyPlotPage("a", "A", "Field", lambda: figure("A")),
            StudyPlotPage("parts", "Coil parts", "Layout", lambda: [figure("Part 1"), figure("Part 2")]),
            StudyPlotPage("z", "Z", "Field", lambda: figure("Z")),
        ],
    )
    book.show_page(1)
    assert [entry.page_id for entry in book.entries] == ["a", "parts", "parts/2", "z"]
    assert [entry.title for entry in book.entries][1:3] == ["Part 1 (1/2)", "Part 2 (2/2)"]
    assert book.index == 1
    assert book.stack.currentIndex() == 1


def test_explicit_spec_title_is_not_replaced_by_generic_backend_title():
    entry = entries([StudyPlotPage("residual", "DSV Bz residual at 1 A", "Field", lambda: figure())])[0]
    materialized = entry.figure.materialize()[0]
    assert materialized.title == "DSV Bz residual at 1 A"
    plt.close(materialized.figure)


def test_section_and_explicit_identity_propagate(books):
    builder = MatplotlibPlotBookBuilder(title="Study", plt=plt, preload_pages=False)
    builder.add_definition(
        StudyPlotBookDefinition(
            "Study",
            sections=(
                StudyPlotSection(
                    "Run A",
                    (StudyPlotPage("wire", "Wire", "", lambda: figure(), page_id="run-a/wire"),),
                    category="Layout",
                ),
            ),
        )
    )
    book = QtMatplotlibPlotBook.from_entries(builder.entries, title="Study", preload_pages=False)
    books.append(book)
    entry = book.entries[0]
    assert (entry.page_id, entry.section, entry.category) == ("run-a/wire", "Run A", "Layout")


def test_exports_expand_all_pages_and_record_failures_without_widgets(books, tmp_path):
    def unavailable():
        raise ValueError("No wire path")

    book = qt_book(
        books,
        [
            StudyPlotPage("parts", "Parts", "Layout", lambda: [figure("Part 1"), figure("Part 2")]),
            StudyPlotPage("missing", "Missing", "Layout", unavailable),
            StudyPlotPage("last", "Last", "Field", lambda: figure("Last")),
        ],
    )
    paths = book.export_all_pages(tmp_path)
    report = json.loads(book.last_export_manifest.read_text(encoding="utf-8"))
    assert len(paths) == 3
    assert all(path.is_file() for path in paths)
    assert [row["page_id"] for row in report["pages"]] == ["parts", "parts/2", "missing", "last"]
    assert [row["status"] for row in report["pages"]] == ["exported", "exported", "unavailable", "exported"]
    assert all(widget is None for widget in book.page_widgets)
    assert [entry.page_id for entry in book.entries] == ["parts", "parts/2", "missing", "last"]
    assert book.preload_pages == "none"
    assert not book.preload_paused
    assert not book.preload_check.isChecked()
    pdfs = book.export_all_pages(tmp_path, format="pdf")
    assert len(pdfs) == 1 and pdfs[0].read_bytes().startswith(b"%PDF")
    pdf_report = json.loads(book.last_export_manifest.read_text(encoding="utf-8"))
    assert [row["pdf_page"] for row in pdf_report["pages"] if row["status"] == "exported"] == [1, 2, 3]


def test_filtered_exports_do_not_materialize_hidden_pages(books, tmp_path):
    calls = []

    def draw(channel):
        calls.append(channel)
        return figure(channel)

    book = qt_book(
        books, [StudyPlotPage(ch, ch, "Field", lambda ch=ch: draw(ch), channel=ch) for ch in ("Gx", "Gy", "Gz")]
    )
    book._set_channel_filter("Gy")
    paths = book.export_filtered_pages(tmp_path)
    assert calls == ["Gy"]
    assert len(paths) == 1
    assert json.loads(book.last_export_manifest.read_text())["pages"][0]["channel"] == "Gy"


def test_invalid_factory_return_does_not_abort_remaining_exports(books, tmp_path):
    book = qt_book(
        books,
        [
            StudyPlotPage("invalid", "Invalid", "Field", object),
            StudyPlotPage("valid", "Valid", "Field", lambda: figure("Valid")),
        ],
    )
    paths = book.export_all_pages(tmp_path)
    assert len(paths) == 1
    report = json.loads(book.last_export_manifest.read_text())
    assert [row["status"] for row in report["pages"]] == ["failed", "exported"]


def test_empty_filters_clear_selection_and_disable_export(books):
    book = qt_book(
        books,
        [
            StudyPlotPage("a", "A", "Field", lambda: figure(), channel="Gx"),
            StudyPlotPage("b", "B", "Hardware", lambda: figure(), channel="Gy"),
        ],
    )
    book.show_page(0)
    book._set_category_filter("Field")
    book._set_channel_filter("Gy")
    assert book.index == -1
    assert not book.export_current_button.isEnabled()
    assert book.table.currentRow() == -1
    assert book.stack.isHidden()
    assert "No pages match" in book.status_label.text()


def test_search_navigation_preloading_and_column_widths(books):
    book = qt_book(
        books,
        [
            StudyPlotPage(
                str(i),
                f"Map {i}",
                "Field",
                lambda i=i: figure(str(i)),
                channel="Gx" if i % 2 == 0 else "Gy",
                section="Reference",
            )
            for i in range(8)
        ],
        max_live_canvases=3,
    )
    book._set_search_query("reference map")
    book._set_channel_filter("Gx")
    book.preload_pages = "idle"
    book.show_page(0)
    assert book._next_preload_index() == 2
    assert book._desired_preload_indices() == (0, 2, 4)
    book.next()
    assert book.index == 2
    book.previous()
    assert book.index == 0
    book.table.setColumnWidth(0, 431)
    book.show_page(4)
    assert book.table.columnWidth(0) == 431
    book.pause_preloading(True)
    assert not book._preload_timer.isActive()


def test_retry_records_failure_and_cleans_up_partial_figures(books):
    calls = []

    def flaky():
        calls.append(1)
        fig = figure()
        if len(calls) == 1:
            raise RuntimeError("Temporary data error")
        return fig

    before = set(plt.get_fignums())
    book = qt_book(books, [StudyPlotPage("flaky", "Flaky", "Field", flaky)])
    book.show_page(0)
    assert set(plt.get_fignums()) == before
    assert book._collection.states["flaky"].status == "failed"
    assert book.retry_button.isEnabled()
    book.retry_current_page()
    assert calls == [1, 1]
    assert book._collection.states["flaky"].status == "ready"
    assert book.export_current_button.isEnabled()


def test_qt_eviction_preserves_widgets_callbacks_and_figure_state(books):
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.widgets import Button, Slider

    clicked, closed, disconnected, calls = [], [], [], []

    def interactive():
        calls.append(1)
        fig = figure("Interactive")
        slider = Slider(fig.add_axes((0.2, 0.05, 0.5, 0.05)), "Value", 0, 1, valinit=0.2)
        button = Button(fig.add_axes((0.7, 0.05, 0.2, 0.1)), "Apply")
        button.on_clicked(lambda event: clicked.append(1))
        fig.canvas.mpl_connect("close_event", lambda event: closed.append(1))
        fig._coilgen_slice_inspector = SimpleNamespace(
            slider=slider, button=button, disconnect=lambda: disconnected.append(1)
        )
        return fig

    book = qt_book(
        books,
        [
            StudyPlotPage("interactive", "Interactive", "Field", interactive),
            StudyPlotPage("other", "Other", "Field", lambda: figure("Other")),
        ],
        max_live_canvases=1,
    )
    book.show_page(0)
    fig = book.entries[0].figure
    controller = fig._coilgen_slice_inspector
    controller.slider.set_val(0.75)
    first_canvas = fig.canvas
    book.show_page(1)
    assert book.page_widgets[0] is None
    assert fig.canvas is not first_canvas
    assert closed == disconnected == []
    book.show_page(0)
    assert calls == [1]
    assert controller.slider.val == 0.75
    assert controller.slider.canvas is fig.canvas
    fig.canvas.draw()
    x, y = controller.button.ax.transAxes.transform((0.5, 0.5))
    for event_name in ("button_press_event", "button_release_event"):
        event = MouseEvent(event_name, fig.canvas, x, y, button=1)
        fig.canvas.callbacks.process(event_name, event)
    assert clicked == [1]
    assert sum(widget is not None for widget in book.page_widgets) == 1
    book.dispose()
    assert disconnected == [1]


def test_legacy_widget_canvas_is_rebound():
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = figure()
    client = SimpleNamespace(canvas=fig.canvas)
    fig._coilgen_slice_inspector = SimpleNamespace(slider=client)
    new_canvas = _replace_figure_canvas(fig, FigureCanvasAgg)
    assert client.canvas is new_canvas
    plt.close(fig)


def test_adopting_qtagg_source_does_not_close_its_inspector(books):
    pytest.importorskip("matplotlib.backends.qt_compat")
    closed = []
    plt.switch_backend("QtAgg")
    try:
        source = figure("Qt source")
        source.canvas.mpl_connect("close_event", lambda event: closed.append(1))
        builder = MatplotlibPlotBookBuilder(title="Adoption", plt=plt, preload_pages=False)
        builder.add_figure(source)
        book = builder.build()
        books.append(book)
        assert isinstance(book, QtMatplotlibPlotBook)
        book.show_page(0)
        assert closed == []
        assert source.canvas.manager is None
        assert plt.get_fignums() == []
        book.dispose()
    finally:
        plt.switch_backend("Agg")


def test_export_uses_source_dimensions_and_keeps_live_view_size(books, tmp_path):
    from matplotlib.image import imread

    book = qt_book(books, [StudyPlotPage("one", "One", "Field", lambda: figure())])
    book.show_page(0)
    source = book.entries[0].figure
    source.set_size_inches((9, 6), forward=False)
    output = book.export_current_page(tmp_path, dpi=80)
    assert imread(output).shape[:2] == (160, 240)
    np.testing.assert_allclose(source.get_size_inches(), (9, 6))


def test_book_camera_controls_refresh_swept_inspector_axis_labels(books):
    from coilgen.models import CoilPart, Mesh
    from coilgen.swept_mesh_inspector import SweptMeshInspector

    def swept():
        fig = plt.figure()
        mesh = Mesh(np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.04, 0]], dtype=float), np.array([[0, 1, 2]]))
        SweptMeshInspector(fig, [CoilPart(mesh=mesh, swept_body=mesh)])
        return fig

    book = qt_book(books, [StudyPlotPage("swept", "Swept", "Manufacturing", swept)])
    book.show_page(0)
    fig = book.entries[0].figure
    inspector = fig._coilgen_swept_mesh_inspector
    inspector.set_view("XY", draw=False)
    assert not inspector.ax.zaxis.label.get_visible()
    book._apply_3d_view_preset(fig, fig.canvas, elev=22, azim=135)
    assert all(axis.label.get_visible() for axis in (inspector.ax.xaxis, inspector.ax.yaxis, inspector.ax.zaxis))
    inspector.set_view("XY", draw=False)
    book._rotate_3d_view(fig, fig.canvas, axis="x", quarter_turns=1)
    assert inspector.ax.zaxis.label.get_visible()
    assert not inspector.ax.yaxis.label.get_visible()


def test_native_close_reopens_retained_qt_book(books):
    book = qt_book(books, [StudyPlotPage("one", "One", "Field", lambda: figure("One"))])
    book.show()
    book.app.processEvents()
    fig = book.entries[0].figure
    book.window.close()
    assert book not in _ACTIVE_PLOT_BOOKS
    assert not book._preload_timer.isActive()
    book.reopen()
    book.app.processEvents()
    assert book.entries[0].figure is fig
    assert book in _ACTIVE_PLOT_BOOKS


def test_fallback_same_title_is_independent_lazy_and_reopenable(books):
    calls = []

    def make():
        calls.append(1)
        return figure()

    specification = [StudyPlotPage("one", "One", "Field", make)]
    first = ImageMatplotlibPlotBook.from_entries(entries(specification), title="Same", plt=plt)
    second = ImageMatplotlibPlotBook.from_entries(entries(specification), title="Same", plt=plt)
    books.extend((first, second))
    assert calls == []
    assert first.fig is not second.fig
    first.show_page(0)
    second.show_page(0)
    assert first.image_ax in first.fig.axes
    original = first.fig
    first.close()
    with patch.object(plt, "show"):
        first.reopen()
    assert first.fig is not original
    assert first.fig.number in plt.get_fignums()
    assert calls == [1, 1]


@pytest.mark.parametrize("axis", ("x", "y", "z"))
@pytest.mark.parametrize("angles", ((22, 135, 17), (90, -90, 0), (-89.9, 40, -67)))
def test_world_rotations_match_actual_matplotlib_camera(axis, angles):
    fig = figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=angles[0], azim=angles[1], roll=angles[2])
    ax.get_proj()
    original = (ax._view_u.copy(), ax._view_v.copy(), ax._view_w.copy())
    rotated = _rotate_3d_view_angles(elev=angles[0], azim=angles[1], roll=angles[2], axis=axis, quarter_turns=1)
    ax.view_init(elev=rotated[0], azim=rotated[1], roll=rotated[2])
    ax.get_proj()
    rotation = _world_axis_quarter_turn_matrix(axis, 1)
    for before, after in zip(original, (ax._view_u, ax._view_v, ax._view_w)):
        np.testing.assert_allclose(after, rotation @ before, atol=1e-12)
    plt.close(fig)


def test_unregister_cannot_remove_another_manager_reusing_number():
    first = figure()
    original_number = first.number
    unregister_pyplot_figure(first)
    second = plt.figure(num=original_number)
    unregister_pyplot_figure(first)
    assert plt.fignum_exists(second.number)
    plt.close(second)


def test_rendering_context_restores_state_after_failure():
    assert not is_plot_book_rendering()
    with pytest.raises(RuntimeError):
        with plot_book_rendering():
            assert is_plot_book_rendering()
            raise RuntimeError("expected")
    assert not is_plot_book_rendering()
