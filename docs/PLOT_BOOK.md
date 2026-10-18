# Plot Book Extension Guide

The plot book is CoilGen's active Matplotlib inspection window. It collects many
2D/3D Matplotlib figures into one navigable window so Spyder users can inspect a
coil design without dozens of separate figure windows.

Install the live Qt interface with `python -m pip install -e ".[plot-qt]"`.
The smaller `plot` extra provides Matplotlib and the raster fallback. Building a
project book with `show=False` preserves the current Matplotlib backend.

## Numerical and display conventions

The default `plot_pointcloud_max_points=2000` limits displayed samples per
point-cloud panel. Set it to `0` for the complete cloud. Sampling is deterministic,
retains field and coordinate extrema and represents target groups; all numerical
diagnostics and color/coordinate limits still use the complete source data.
The sample count is shown on the figure. This display budget does not change
coil optimization, currents, winding paths, or computed field arrays.

Related target/layout/stream panels share color limits. Signed residual maps
use a zero-centered diverging scale, and constant fields use a range relative
to their physical magnitude. Relative-error labels state the denominator;
their existing arithmetic is preserved. Invalid or unidentifiable local
gradient components remain missing instead of appearing as zero error.

Project SH comparisons fit every channel with the same center, radius, and
display unit, chosen from the first available channel. Each channel retains
its own DSV samples and weights, so its signal RMS and term RMS contributions
can differ even for the same physical field. Unavailable channels are reported
on the comparison page. General and shim SH pages reuse one multi-source
least-squares solve for their plots and coefficient tables.

Diagnostic caches check source contents and request parameters, including
in-place changes to equal-sized arrays. Custom slice-map requests do not replace
the configured default request. Slice interpolation reuses geometry across
fields, and finite differences evaluate only requested derivative directions.
The concomitant maps remain the existing global affine-gradient Maxwell
approximation; their labels make that scope explicit.

## Inspecting swept conductor meshes

The `swept_body` page opens an interactive mesh inspector. When geometry pages
are requested and a swept mesh is available, the registry includes this page
automatically. It also remains available directly through
`plot_swept_body_3d(part)`.

The inspector uses the actual, complete triangle mesh. It prefers independent
`swept_bodies` so contour identity, turn multiplicity, fabrication paths, and
stored manufacturing diagnostics remain available. Older results with only a
combined `swept_body` are supported.

Select one body to examine overlapping windings, then fit the view to that
body. Switch between surfaces and mesh edges, adjust opacity, and show the
support mesh or fabrication centroid path as needed. The default orthographic
projection avoids perspective distortion. Plotbook orientation controls and
the Matplotlib rotation, pan, and zoom tools remain available. Display controls
never alter the fabrication mesh or rerun sweep/collision calculations.

Stored manufacturing failure reasons appear above the mesh and follow the
body selection. The all-bodies view counts how many bodies report each reason.
Missing check results remain unknown. `sweep_self_intersection` reports the
conservative capsule-envelope check; `swept_mesh_self_intersection` reports
triangle-mesh collision. A watertight mesh can still fail manufacturing checks.

**Inspect issue** cycles through stored folded-quad, profile-Jacobian, and
frame-rotation failure locations, marking the location and fitting nearby
actual sweep rings. Start from All bodies to visit issues across bodies, or
select a body first to inspect only its issues. Fit selected/All bodies returns
to the broader view. Older results without stored locations show an explicit
message; no failure locations or collision checks are recomputed.

## Inspecting field slices

Use the `gradient_slice_inspector` page, or
`from coilgen import plot_gradient_slice_inspector`, when slice maps are enabled.
Its slice slider, component selector, engineering/SI unit selector, and
common/individual scale toggle reuse the evaluated field arrays. Hover over
the image to read physical coordinates and the pixel value. The full static
mosaics remain available for reports.

The implementation lives in:

```text
src/coilgen/plot_book.py
```

The CoilGen-specific plot registry lives in:

```text
src/coilgen/matplotlib_visualization.py
```

## Navigation, loading, and export

Search matches page titles, channels, and study sections. Combine search with
the plot-type and channel filters; Previous/Next navigate only matching pages.
`Ctrl+F` focuses search, and `Alt+Left` / `Alt+Right` navigate. An empty filter
result clears the displayed page and disables current-page export. Failed or
unavailable pages retain their place and provide a retry action.

Registration does not run page factories. Showing the book materializes the
selected page, then the default idle mode prepares nearby matching pages.
Preloading can be paused. Live Qt canvases are bounded to five by default;
eviction retains figures, artists, controls, and camera state so returning to a
page does not rerun a user factory. This bounds canvas/renderer memory, not all
scientific arrays and artists. Factories still run on the GUI thread, so an
individual expensive page can temporarily occupy the interface.

`close()` hides a book for later `open()` / `reopen()`; `dispose()` releases its
owned figures and controls permanently. Keep the returned book in a workspace
variable when you want to reopen it.

Both live and fallback books support report export:

```python
book.export_current_page("plots")
book.export_filtered_pages("plots", format="pdf")
book.export_all_pages("plots", format="png", dpi=180)
```

Exports include lazy multipart pages in source order. One unavailable or failed
page does not prevent the others from exporting. PNG/SVG export returns one
path per successful page; PDF returns a multipage report path. A JSON manifest
records page identity, title, channel, section, result, and failures;
`book.last_export_manifest` points to it. Project books also include project,
output-directory, and channel identity. Report exports use the original figure
size instead of the current window size. The fallback displays raster pages;
when source figures are retained it can still export them directly.

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

If the subclass owns Matplotlib figures, its `dispose()` method must release them.
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

Install the optional Qt runtime to exercise the real widgets. The quality CI
runs the regression suite with Qt installed and the offscreen platform:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:MPLBACKEND = "Agg"
python -m pytest -q
python -m ruff check src tests projects
python -m ruff format --check src tests projects
```

For GUI behavior, run one Spyder example and verify:

- only one plot-book window is opened;
- selector rows switch pages;
- the category filter hides and shows the expected pages;
- Export Page and Export All write PNG files;
- 3D pages rotate and zoom;
- the swept-mesh selector isolates a body and edge/opacity controls update it;
- slice/component/unit controls remain usable after navigating away and back;
- the performance table page scrolls vertically when there are many rows.
