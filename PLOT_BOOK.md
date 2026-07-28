# Plot Book Extension Guide

The plot book is CoilGen's active Matplotlib inspection window. It collects many
2D/3D Matplotlib figures into one navigable window so Spyder users can inspect a
coil design without dozens of separate figure windows.

The implementation lives in:

```text
src/coilgen/plot_book.py
```

The CoilGen-specific plot registry lives in:

```text
src/coilgen/matplotlib_visualization.py
```

## Class Structure

`MatplotlibPlotBook` is the abstract base class. It defines the common window
interface:

- `page_count`
- `show(block=False)`
- `close()`
- `show_page(index)`
- `export_current_page(directory)`
- `export_all_pages(directory)`

`QtMatplotlibPlotBook` is the preferred implementation. It embeds live
Matplotlib canvases in a Qt window. This preserves 3D rotation, zoom, pan, and
the Matplotlib toolbar.

`ImageMatplotlibPlotBook` is the fallback implementation. It rasterizes each
figure into an image page if Qt is unavailable. This keeps the one-window
workflow but does not preserve live 3D interaction.

`MatplotlibPlotBookBuilder` is the normal extension point. It collects figures,
assigns page metadata, and builds the best available plot book.

`StudyPlotBookDefinition`, `StudyPlotSection`, and `StudyPlotPage` are the
general object representation for project-specific plot books. They let a
project describe pages and figure factories without subclassing the plot-book
window or changing CoilGen's result-plot registry.

The Qt implementation also provides a compact category filter in the right-side
selector, a project-channel filter for multi-channel books, and export buttons
for the selected page or all pages. The export
methods are available programmatically on both Qt and raster fallback books.
For pages with Matplotlib 3D axes, the Qt plot book also adds `Iso`, `XY`,
`XZ`, and `YZ` orientation buttons below the Matplotlib toolbar. The selected
view is applied to every 3D axes on that page.

## Page Data

`PlotBookEntry` describes one live Matplotlib figure:

- `key`: stable internal identifier
- `short_name`: compact selector label
- `title`: full title used for tooltips/window title
- `category`: group label such as `layout`, `field`, `metrics`, or `Study`
- `channel`: optional project channel label such as `Gx`, `Gy`, or `B0`
- `figure`: the Matplotlib figure

`PlotBookPage` describes one raster fallback page. It is used only by
`ImageMatplotlibPlotBook`.

The `key` should be stable because it can drive UI behavior. For example,
`performance_table` is displayed inside a scroll area so long scientific tables
remain readable.

Common CoilGen result plot keys include:

- `layout_field_overview`: coil layout together with transparent DSV and
  suppression target-point samples colored by connected-wire Bz per ampere;
- `wire`: connected wire path on the coil support mesh;
- `wire_sequence`: final ordered conductor path colored by cumulative route
  distance, with current arrows, numbered loop visits, and connector styles;
- `contour_current_orientation`: unconnected contours with actual-current
  arrowheads and signed chirality where the local surface normal is coherent;
- `contour_groups`: topologically grouped stream-function contours;
- `field_pointclouds`: target, connected-wire, stream-function, and error Bz
  point clouds;
- `gradient_maps` and `error_maps`: DSV gradient and field-error maps;
- `performance_table`: compact scientific result summary.

Projects that request a geometry-oriented page (`wire`, `raw_contours`,
`contour_groups`, `interconnect_debug`, or `swept_body`) automatically receive
both `wire_sequence` and `contour_current_orientation` as companion pages. This
keeps project-specific include lists concise while ensuring that Halbach,
cylindrical, planar, and arbitrary-mesh designs expose current direction and
interconnection order. Include lists containing only non-geometry diagnostics
are left unchanged.

## Adding a New CoilGen Result Plot

Most extensions should be normal Matplotlib plotting functions, not plot-book
subclasses.

1. Add a plotting function in `matplotlib_visualization.py`.

```python
def plot_my_quantity_3d(result, *, show=True, block=False):
    plt = _import_pyplot()
    fig = plt.figure(num="My Quantity", figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    # draw scientific content here
    return _finish_figure(fig, plt, show=show, block=block)
```

2. Register it in `build_result_plot_specs(...)`.

```python
add(
    "my_quantity",
    "DSV Target Points: My Quantity",
    "field",
    lambda: plot_my_quantity_3d(result, show=False),
)
```

3. Add a short label in `_short_plot_book_name(...)` if the default key label is
too long.

```python
"my_quantity": "my qty",
```

The public `plot_result_matplotlib_book(...)` function will then place the new
figure into the plot book automatically.

## Adding Parameter-Study or Script Figures

If a script already creates normal Matplotlib figures, do not open them with
`plt.show()`. Add them to a plot book:

```python
from coilgen import MatplotlibPlotBookBuilder

builder = MatplotlibPlotBookBuilder(title="Study Plot Book", plt=plt)
builder.add_figures(
    figures,
    names=("Efficiency sweep", "Shielding sweep"),
    categories=("Sweep", "Sweep"),
)
book = builder.build()
book.show(block=False)
```

The example helper `plot_figures_book_if_requested(...)` wraps this pattern for
Spyder scripts.

## Adding Project-Specific Plot Books

For project workflows with several results, create a study plot-book definition
instead of forcing everything through one `CoilGenResult`:

```python
from coilgen import StudyPlotBookDefinition, StudyPlotPage, StudyPlotSection, plot_study_plot_book

definition = StudyPlotBookDefinition(
    title="My Project Study",
    sections=(
        StudyPlotSection(
            title="Summary",
            pages=(
                StudyPlotPage(
                    key="summary_table",
                    short_name="summary",
                    title="Study Summary Table",
                    category="Metrics",
                    channel="Gx",
                    factory=lambda: make_summary_figure(rows),
                ),
            ),
        ),
    ),
)
book = plot_study_plot_book(definition, show=True, block=False)
```

This is the preferred pattern for multi-channel systems, parameter sweeps, and
project-specific comparisons. Each page factory returns a normal Matplotlib
figure or a list of figures. The same Qt/image fallback, category selector, and
export buttons are used automatically.

## Adding Figures to an Existing Book

Both concrete plot-book classes support appending pages:

```python
book.add_figure(fig, key="study_metric", short_name="metric", category="Study")
```

Use this only after a book has already been built. For normal scripts, collecting
all figures first and building once is easier and avoids transient empty windows.

## Subclassing

Subclass `MatplotlibPlotBook` only when changing container behavior, navigation,
or backend implementation. Do not subclass it just to add a new scientific plot.

When subclassing, implement:

- `page_count`
- `show(block=False)`
- `close()`
- `show_page(index)`
- `_export_page(index, directory)`

If the subclass owns Matplotlib figures, its `close()` method must close them.
If it embeds live canvases, it should hide or unregister pyplot managers so
Spyder does not open additional empty figure windows.

## Current UI-Specific Behavior

`QtMatplotlibPlotBook._page_widget(...)` handles page-container details:

- normal pages are embedded as live Matplotlib canvases;
- `performance_table` pages are embedded in a scroll area;
- each page has its own Matplotlib toolbar;
- pages with 3D axes get `Iso`, `XY`, `XZ`, and `YZ` orientation buttons;
- embedded figures are unregistered from pyplot to avoid extra windows.

Keep new UI-specific behavior in `_page_widget(...)` only when it is about page
presentation. Keep scientific plotting logic in normal plotting functions.

Category filtering, page export, and selector behavior belong to the plot-book
container class. New scientific plot types should not implement their own
navigation or export controls.

## Testing

The bundled test runtime may not have Matplotlib/Qt installed, so active GUI
tests can be skipped. Always run:

```powershell
python -m compileall -q src projects tests
python -m unittest discover -s tests -v
```

For GUI behavior, run one Spyder example and verify:

- only one plot-book window is opened;
- selector rows switch pages;
- the category filter hides and shows the expected pages;
- Export Page and Export All write PNG files;
- 3D pages rotate and zoom;
- the performance table page scrolls vertically when there are many rows.
