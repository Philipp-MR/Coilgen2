from __future__ import annotations

from dataclasses import dataclass
import textwrap
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .basis import current_density_from_basis
from .design_context import EvaluationFeature, normalize_design_context
from .field_maps import (
    AXIS_NAMES,
    display_field_unit_1a,
    display_field_values_1a,
    central_slice_selection,
    coil_spherical_harmonic_coefficient_rows,
    coil_spherical_harmonic_fits,
    field_error_scalar_maps,
    field_relative_error_scalar_maps,
    field_scalar_maps,
    format_label_with_unit,
    gradient_component_scalar_maps,
    gradient_relative_error_scalar_maps,
    interpolated_slice_mosaic,
    layout_bz_1a,
    result_uses_sparsified_contour_layout,
    shim_field_scalar_maps,
    shim_homogeneity_stats,
    shim_spherical_harmonic_comparison,
    shim_spherical_harmonic_fits,
    target_domain_mask,
)
from .gradient_slice_maps import gradient_slice_map_result, gradient_slice_maps_enabled
from .matplotlib_backend import activate_matplotlib_interactive_backend as activate_matplotlib_interactive_backend
from .models import CoilGenResult, CoilPart, ContourLine, WirePathGeometry, WireSegmentOwner
from .metrics import dsv_target_mask, suppression_target_mask
from .plot_book import (
    MatplotlibPlotBookBuilder,
    close_active_plot_books,
)
from .regularization_diagnostics import (
    normalized_linear_trace,
    normalized_log_trace,
    regularization_scan_arrays,
)
from .reporting import format_result_summary_value, result_performance_table_rows


Figure = Any
COORDINATE_UNIT = "m"
STREAM_FUNCTION_UNIT = "A"
CURRENT_DENSITY_UNIT = "A/m"
RELATIVE_ERROR_UNIT = "%"
COLORBAR_TICK_COUNT = 5
GRADIENT_SLICE_TILE_SIZE_IN = 1.55
GRADIENT_SLICE_COLORBAR_WIDTH_IN = 0.08
GRADIENT_SLICE_COLORBAR_GAP_IN = 0.025
GRADIENT_SLICE_COLORBAR_TICK_GUTTER_IN = 0.62
GRADIENT_SLICE_COLUMN_GAP_IN = 0.16
GRADIENT_SLICE_ROW_GAP_IN = 0.18
GRADIENT_SLICE_LEFT_MARGIN_IN = 0.28
GRADIENT_SLICE_RIGHT_MARGIN_IN = 0.12
GRADIENT_SLICE_BOTTOM_MARGIN_IN = 0.28
GRADIENT_SLICE_TOP_MARGIN_IN = 0.52
MESH_FACE_COLOR = "#f8fafc"
MESH_EDGE_COLOR = "#64748b"
MESH_EDGE_LINEWIDTH = 0.38
MESH_FACE_ALPHA = 0.34
REFERENCE_MESH_EDGE_COLOR = "#94a3b8"
REFERENCE_MESH_EDGE_LINEWIDTH = 0.42
REFERENCE_MESH_EDGE_ALPHA = 0.62
SCALAR_MESH_EDGE_COLOR = "#475569"
SCALAR_MESH_EDGE_LINEWIDTH = 0.24
RAW_CONTOUR_LINEWIDTH = 2.45
GROUP_CONTOUR_LINEWIDTH = 2.65
WIRE_PATH_LINEWIDTH = 2.55
CURRENT_DIRECTION_POSITIVE_COLOR = "#2563eb"
CURRENT_DIRECTION_NEGATIVE_COLOR = "#d97706"
CURRENT_DIRECTION_UNDEFINED_COLOR = "#6b7280"
FIELD_POINT_ALPHA = 0.72
FACE_ISOMETRIC_VIEW = "face_isometric"
FACE_ISOMETRIC_ELEVATION_DEG = 22.0
FACE_ISOMETRIC_AZIMUTH_DEG = 135.0
DEFAULT_RESULT_PLOT_KEYS = (
    "wire",
    "wire_sequence",
    "problem_setup_geometry",
    "layout_field_overview",
    "raw_contours",
    "contour_current_orientation",
    "contour_groups",
    "stream_function",
    "current_density",
    "interconnect_debug",
    "swept_body",
    "field_comparison",
    "field_pointclouds",
    "shim_pointclouds",
    "shim_histograms",
    "shim_spherical_harmonics",
    "gradient_maps",
    "error_maps",
    "spherical_harmonics",
    "shim_homogeneity",
    "field_cross_sections",
    "gradient_cross_sections",
    "field_slice_mosaic",
    "gradient_slice_mosaic",
    "concomitant_slice_mosaic",
    "error_cross_sections",
    "error_metrics",
    "performance_table",
    "regularization",
)
MULTI_PART_RESULT_PLOT_KEYS = {
    "wire",
    "raw_contours",
    "contour_groups",
    "stream_function",
    "current_density",
    "interconnect_debug",
    "swept_body",
}
CURRENT_FLOW_COMPANION_ANCHOR_KEYS = {
    "wire",
    "raw_contours",
    "contour_groups",
    "interconnect_debug",
    "swept_body",
}
CURRENT_FLOW_COMPANION_KEYS = {
    "wire_sequence",
    "contour_current_orientation",
}


@dataclass(frozen=True, slots=True)
class MatplotlibPlotSpec:
    """Definition of one result plot type.

    The factory intentionally returns normal Matplotlib figures. This keeps new
    plot types easy to add and lets the same registry drive individual windows
    and compact plot-book pages.
    """

    key: str
    title: str
    category: str
    factory: Callable[[], Figure | list[Figure]]
    channel: str = ""


@dataclass(frozen=True, slots=True)
class ProblemSetupFieldOverlay:
    """Explicit field point cloud for the problem-setup geometry page."""

    coords: np.ndarray
    values: np.ndarray
    dsv_mask: np.ndarray
    suppression_mask: np.ndarray
    label_with_unit: str
    dsv_label: str = "DSV target field samples"
    suppression_label: str = "suppression target field samples"
    cmap: str = "coolwarm"


def pointcloud_plot_indices(
    config: Any,
    point_count: int,
    *,
    eligible_mask: np.ndarray | None = None,
    seed_offset: int = 0,
) -> np.ndarray:
    """Return a reproducible random display subset without changing source data."""

    count = int(point_count)
    if count < 0:
        raise ValueError("Point count must be non-negative.")
    if eligible_mask is None:
        eligible = np.arange(count, dtype=int)
    else:
        mask = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if len(mask) != count:
            raise ValueError("Point-cloud eligibility mask must match the point count.")
        eligible = np.flatnonzero(mask)

    max_points = int(getattr(config, "plot_pointcloud_max_points", 0) or 0)
    if max_points <= 0 or len(eligible) <= max_points:
        return eligible

    seed = int(getattr(config, "plot_pointcloud_random_seed", 1729))
    if seed < 0:
        raise ValueError("Point-cloud plot random seed must be non-negative.")
    rng = np.random.default_rng((seed + int(seed_offset)) % (2**64))
    return np.sort(rng.choice(eligible, size=max_points, replace=False))


def build_result_plot_specs(
    result: CoilGenResult,
    *,
    include: Sequence[str] | None = None,
    max_parts: int | None = None,
) -> list[MatplotlibPlotSpec]:
    """Build the plot registry entries requested for a CoilGen result."""

    include_set = _normalise_result_plot_keys(include)
    design_context = normalize_design_context(getattr(result.input_data, "design_context", "mr_gradient"))
    specs: list[MatplotlibPlotSpec] = []
    parts = result.coil_parts[:max_parts] if max_parts is not None else result.coil_parts
    sparsified_layout = result_uses_sparsified_contour_layout(result)

    def add(
        key: str,
        title: str,
        category: str,
        factory: Callable[[], Figure | list[Figure]],
        *,
        include_set_for_key: set[str] | None = None,
    ) -> None:
        active_keys = include_set_for_key if include_set_for_key is not None else include_set
        if key in active_keys:
            specs.append(MatplotlibPlotSpec(key=key, title=title, category=category, factory=factory))

    if len(parts) > 1:
        if sparsified_layout:
            add(
                "wire",
                "All Coil Parts: Final Sparsified Weighted-Contour Layout",
                "layout",
                lambda: plot_raw_contours_all_parts_3d(
                    parts,
                    title="All Coil Parts: Final Sparsified Weighted-Contour Layout",
                    show_source_contours=False,
                    show=False,
                ),
            )
        else:
            add(
                "wire",
                "All Coil Parts: Connected Wire Paths",
                "layout",
                lambda: plot_wire_paths_3d(parts, title="All Coil Parts: Connected Wire Paths", show=False),
            )
        add(
            "raw_contours",
            (
                "All Coil Parts: Final Sparsified Contours (Original Contours Dashed)"
                if sparsified_layout
                else "All Coil Parts: Raw Stream-Function Contours"
            ),
            "contours",
            lambda: plot_raw_contours_all_parts_3d(
                parts,
                title=(
                    "All Coil Parts: Final Sparsified Contours (Original Contours Dashed)"
                    if sparsified_layout
                    else "All Coil Parts: Raw Stream-Function Contours"
                ),
                show=False,
            ),
        )
        add(
            "contour_groups",
            "All Coil Parts: Topological Loop Groups",
            "topology",
            lambda: plot_contour_groups_all_parts_3d(
                parts, title="All Coil Parts: Topological Loop Groups", show=False
            ),
        )
        add(
            "stream_function",
            (
                "All Coil Parts: Continuous Stream-Function Reference (Pre-Sparsification)"
                if sparsified_layout
                else "All Coil Parts: Optimized Stream Function"
            ),
            "optimization",
            lambda: plot_stream_function_all_parts_3d(
                parts,
                title=(
                    "All Coil Parts: Continuous Stream-Function Reference (Pre-Sparsification)"
                    if sparsified_layout
                    else "All Coil Parts: Optimized Stream Function"
                ),
                show=False,
            ),
        )
        add(
            "current_density",
            (
                "All Coil Parts: Continuous Current-Density Reference (Pre-Sparsification)"
                if sparsified_layout
                else "All Coil Parts: Surface Current-Density Magnitude"
            ),
            "current density",
            lambda: plot_current_density_all_parts_3d(
                parts,
                title=(
                    "All Coil Parts: Continuous Current-Density Reference (Pre-Sparsification)"
                    if sparsified_layout
                    else "All Coil Parts: Surface Current-Density Magnitude"
                ),
                show=False,
            ),
        )
        add(
            "interconnect_debug",
            "All Coil Parts: Contour Opening Cuts and Group Connectors",
            "routing",
            lambda: plot_interconnect_debug_all_parts_3d(
                parts, title="All Coil Parts: Contour Opening Cuts and Group Connectors", show=False
            ),
        )
        add(
            "swept_body",
            "All Coil Parts: Swept Conductor Body Meshes",
            "manufacturing",
            lambda: plot_swept_bodies_all_parts_3d(
                parts, title="All Coil Parts: Swept Conductor Body Meshes", show=False
            ),
        )

    part_include_set = include_set - MULTI_PART_RESULT_PLOT_KEYS if len(parts) > 1 else include_set
    for part_ind, part in enumerate(parts, start=1):
        if sparsified_layout:
            add(
                "wire",
                f"Coil Part {part_ind}: Final Sparsified Weighted-Contour Layout",
                "layout",
                lambda part=part, part_ind=part_ind: plot_raw_contours_3d(
                    part,
                    title=f"Coil Part {part_ind}: Final Sparsified Weighted-Contour Layout",
                    show_source_contours=False,
                    show=False,
                ),
                include_set_for_key=part_include_set,
            )
        else:
            add(
                "wire",
                f"Coil Part {part_ind}: Connected Wire Path on Coil Surface",
                "layout",
                lambda part=part, part_ind=part_ind: plot_wire_path_3d(
                    part, title=f"Coil Part {part_ind}: Connected Wire Path on Coil Surface", show=False
                ),
                include_set_for_key=part_include_set,
            )
        add(
            "wire_sequence",
            f"Coil Part {part_ind}: Final Conductor Traversal Sequence",
            "routing",
            lambda part=part, part_ind=part_ind: plot_wire_path_sequence_3d(
                part,
                title=f"Coil Part {part_ind}: Final Conductor Traversal Sequence",
                show=False,
            ),
            include_set_for_key=part_include_set,
        )
        add(
            "raw_contours",
            (
                f"Coil Part {part_ind}: Final Sparsified Contours (Original Contours Dashed)"
                if sparsified_layout
                else f"Coil Part {part_ind}: Raw Stream-Function Contours"
            ),
            "contours",
            lambda part=part, part_ind=part_ind: plot_raw_contours_3d(
                part,
                title=(
                    f"Coil Part {part_ind}: Final Sparsified Contours (Original Contours Dashed)"
                    if sparsified_layout
                    else f"Coil Part {part_ind}: Raw Stream-Function Contours"
                ),
                show=False,
            ),
            include_set_for_key=part_include_set,
        )
        add(
            "contour_current_orientation",
            f"Coil Part {part_ind}: Contour Current Direction and Surface Chirality",
            "contours",
            lambda part=part, part_ind=part_ind: plot_contour_current_orientation_3d(
                part,
                title=f"Coil Part {part_ind}: Contour Current Direction and Surface Chirality",
                show=False,
            ),
            include_set_for_key=part_include_set,
        )
        add(
            "contour_groups",
            f"Coil Part {part_ind}: Topological Loop Groups",
            "topology",
            lambda part=part, part_ind=part_ind: plot_contour_groups_3d(
                part, title=f"Coil Part {part_ind}: Topological Loop Groups", show=False
            ),
            include_set_for_key=part_include_set,
        )
        add(
            "stream_function",
            (
                f"Coil Part {part_ind}: Continuous Stream-Function Reference (Pre-Sparsification)"
                if sparsified_layout
                else f"Coil Part {part_ind}: Optimized Stream Function on Mesh"
            ),
            "optimization",
            lambda part=part, part_ind=part_ind: plot_stream_function_3d(
                part,
                title=(
                    f"Coil Part {part_ind}: Continuous Stream-Function Reference (Pre-Sparsification)"
                    if sparsified_layout
                    else f"Coil Part {part_ind}: Optimized Stream Function on Mesh"
                ),
                show=False,
            ),
            include_set_for_key=part_include_set,
        )
        add(
            "current_density",
            (
                f"Coil Part {part_ind}: Continuous Current-Density Reference (Pre-Sparsification)"
                if sparsified_layout
                else f"Coil Part {part_ind}: Surface Current-Density Magnitude"
            ),
            "current density",
            lambda part=part, part_ind=part_ind: plot_current_density_3d(
                part,
                title=(
                    f"Coil Part {part_ind}: Continuous Current-Density Reference (Pre-Sparsification)"
                    if sparsified_layout
                    else f"Coil Part {part_ind}: Surface Current-Density Magnitude"
                ),
                show=False,
            ),
            include_set_for_key=part_include_set,
        )
        add(
            "interconnect_debug",
            f"Coil Part {part_ind}: Contour Opening Cuts and Group Connectors",
            "routing",
            lambda part=part, part_ind=part_ind: plot_interconnect_debug_3d(
                part, title=f"Coil Part {part_ind}: Contour Opening Cuts and Group Connectors", show=False
            ),
            include_set_for_key=part_include_set,
        )
        add(
            "swept_body",
            f"Coil Part {part_ind}: Swept Conductor Body Mesh",
            "manufacturing",
            lambda part=part, part_ind=part_ind: plot_swept_body_3d(
                part, title=f"Coil Part {part_ind}: Swept Conductor Body Mesh", show=False
            ),
            include_set_for_key=part_include_set,
        )

    add(
        "problem_setup_geometry",
        "Problem Setup Geometry: Current Surface and Target Field",
        "layout",
        lambda: plot_problem_setup_geometry_3d(result, show=False),
    )
    add(
        "layout_field_overview",
        "Coil Layout with DSV and Suppression Field Points",
        "layout",
        lambda: plot_layout_field_overview_3d(result, show=False),
    )
    add(
        "field_comparison",
        "Target Points: Bz Field Comparison per Ampere",
        "field",
        lambda: plot_field_comparison_matplotlib(result, show=False),
    )
    add(
        "field_pointclouds",
        "Target Points: Bz Field Maps per Ampere",
        "field",
        lambda: plot_field_pointclouds_3d(result, show=False),
    )
    add(
        "target_field_maps",
        "Target Points: Bz Field Maps per Ampere",
        "field",
        lambda: plot_target_field_maps_3d(result, show=False),
    )
    if design_context.enabled(EvaluationFeature.SHIM_HOMOGENEITY):
        add(
            "shim_pointclouds",
            "Shim Target Points: Before and After B0 Inhomogeneity",
            "field",
            lambda: plot_shim_pointclouds_3d(result, show=False),
        )
        add(
            "shim_histograms",
            "Shim Target Points: Field Histograms Before and After",
            "field",
            lambda: plot_shim_field_histograms(result, show=False),
        )
        add(
            "shim_homogeneity",
            "B0 Inhomogeneity: Before, After, and Improvement",
            "field",
            lambda: plot_shim_before_after_mosaics(result, show=False),
        )
    if design_context.enabled(EvaluationFeature.SHIM_SPHERICAL_HARMONICS):
        add(
            "shim_spherical_harmonics",
            "Shim Target Points: Spherical Harmonic Coefficients",
            "field",
            lambda: plot_shim_spherical_harmonics(result, show=False),
        )
    if design_context.enabled(EvaluationFeature.FIELD_ERROR):
        add(
            "spherical_harmonics",
            "DSV Field Spherical Harmonic Coefficients",
            "field",
            lambda: plot_field_spherical_harmonics(result, show=False),
        )
    if design_context.enabled(EvaluationFeature.GRADIENT_FIT):
        add(
            "gradient_maps",
            "DSV Target Points: Bz Gradient Maps",
            "field",
            lambda: plot_gradient_maps_3d(result, show=False),
        )
        add(
            "gradient_cross_sections",
            "DSV Target Points: Bz Gradient Central Cross Sections",
            "field",
            lambda: plot_gradient_cross_section_slices(result, show=False),
        )
        if gradient_slice_maps_enabled(result.input_data):
            add(
                "field_slice_mosaic",
                "MR Gradient Pixel Slices: Field Components",
                "field",
                lambda: plot_gradient_field_slice_mosaic(result, show=False),
            )
            add(
                "gradient_slice_mosaic",
                "MR Gradient Pixel Slices: Bz Gradients",
                "field",
                lambda: plot_gradient_derivative_slice_mosaic(result, show=False),
            )
            add(
                "concomitant_slice_mosaic",
                "MR Gradient Pixel Slices: Concomitant Field",
                "field",
                lambda: plot_concomitant_slice_mosaic(result, show=False),
            )
    add("error_maps", "DSV Target Points: Bz Error Maps", "field", lambda: plot_error_maps_3d(result, show=False))
    add(
        "field_cross_sections",
        "Target Points: Bz Field Central Cross Sections",
        "field",
        lambda: plot_field_cross_section_slices(result, show=False),
    )
    add(
        "error_cross_sections",
        "DSV Target Points: Bz Error Central Cross Sections",
        "field",
        lambda: plot_error_cross_section_slices(result, show=False),
    )
    add(
        "error_metrics",
        "Coil Design DSV Field-Error Metrics",
        "metrics",
        lambda: plot_error_metrics_matplotlib(result, show=False),
    )
    add(
        "performance_table",
        "Coil Design Performance Summary Table",
        "metrics",
        lambda: plot_performance_table_matplotlib(result, show=False),
    )
    add(
        "regularization",
        (
            "Continuous Stream-Function Regularization Reference (Pre-Sparsification)"
            if sparsified_layout
            else "Stream-Function Regularization L-Curve Scan"
        ),
        "optimization",
        lambda: plot_l_curve_regularization_matplotlib(result, show=False),
    )
    return specs


def plot_result_matplotlib(
    result: CoilGenResult,
    *,
    include: Sequence[str] | None = None,
    max_parts: int | None = None,
    show: bool = True,
    block: bool = False,
    timing_report: list[dict[str, float | int | str]] | None = None,
    print_timing: bool = False,
) -> list[Figure]:
    """Create active matplotlib 2D/3D inspection figures for a CoilGen result."""

    plt = _import_pyplot()
    figures: list[Figure] = []
    for spec in build_result_plot_specs(result, include=include, max_parts=max_parts):
        _append_plot_spec_figures(figures, spec, timing_report=timing_report)
    if show:
        show_start = time.perf_counter()
        plt.show(block=block)
        if timing_report is not None:
            timing_report.append(
                {"plot": "show", "seconds": float(time.perf_counter() - show_start), "figures": len(figures)}
            )
    if print_timing and timing_report is not None:
        print("Matplotlib plot timings:")
        for row in timing_report:
            print(f"  {row['plot']}: {row['seconds']:.3f} s ({int(row['figures'])} figure(s))")
    return figures


def plot_result_matplotlib_book(
    result: CoilGenResult,
    *,
    include: Sequence[str] | None = None,
    max_parts: int | None = None,
    extra_figures: Sequence[Figure] | None = None,
    extra_names: Sequence[str] | None = None,
    extra_categories: Sequence[str] | None = None,
    show: bool = True,
    block: bool = False,
    timing_report: list[dict[str, float | int | str]] | None = None,
    print_timing: bool = False,
    preload_pages: str | bool = "idle",
) -> Any:
    """Create one active plot-book window with interactable CoilGen plots."""

    plt = _import_pyplot()
    specs = build_result_plot_specs(result, include=include, max_parts=max_parts)
    title = "CoilGen Result Plot Book"
    builder = MatplotlibPlotBookBuilder(
        title=title,
        plt=plt,
        timing_report=timing_report,
        preload_pages=preload_pages,
    )
    builder.add_specs(specs, short_name_factory=_short_plot_book_name)
    builder.add_figures(extra_figures, names=extra_names, categories=extra_categories)
    book = builder.build()

    if show:
        show_start = time.perf_counter()
        book.show(block=block)
        if timing_report is not None:
            timing_report.append({"plot": "show", "seconds": float(time.perf_counter() - show_start), "figures": 1})
    if print_timing and timing_report is not None:
        print("Matplotlib plot-book timings:")
        for row in timing_report:
            print(f"  {row['plot']}: {row['seconds']:.3f} s ({int(row['figures'])} page(s))")
    return book


def plot_matplotlib_figures_book(
    figures: Sequence[Figure],
    *,
    title: str = "Matplotlib Plot Book",
    names: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    show: bool = True,
    block: bool = False,
    preload_pages: str | bool = "idle",
) -> Any:
    """Embed already-created Matplotlib figures in one live plot-book window."""

    plt = _import_pyplot()
    builder = MatplotlibPlotBookBuilder(title=title, plt=plt, preload_pages=preload_pages)
    builder.add_figures(figures, names=names, categories=categories)
    book = builder.build()
    if show:
        book.show(block=block)
    return book


def close_matplotlib_plot_books() -> None:
    """Close live Qt plot-book windows created by plot_result_matplotlib_book."""

    close_active_plot_books()


def plot_wire_path_3d(
    part: CoilPart, *, title: str = "Connected Coil Wire Path on Coil Surface", show: bool = True, block: bool = False
) -> Figure:
    """Plot the connected wire path and coil mesh on a matplotlib 3D axis."""

    if part.wire_path is None:
        raise ValueError("Part has no wire path.")
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_mesh_3d(ax, part)
    _plot_path_3d(ax, part.wire_path.v, color="#111827", linewidth=WIRE_PATH_LINEWIDTH, label="connected wire path")
    if part.shift_array is not None and np.any(np.asarray(part.shift_array) > 0):
        _draw_shifted_segments_3d(ax, part.wire_path.v, part.shift_array)
    _finish_3d_axis(ax, _stack_points([part.mesh.vertices, part.wire_path.v]), view=FACE_ISOMETRIC_VIEW)
    _deduplicate_legend(ax)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_wire_path_sequence_3d(
    part: CoilPart,
    *,
    title: str = "Final Conductor Traversal Sequence",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot final route order, current direction, ownership transitions, and connector geometry."""

    if part.wire_path is None or len(part.wire_path.v) < 2:
        raise ValueError("Part has no wire path.")
    points = np.asarray(part.wire_path.v, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(lengths))
    if total_length <= 1e-15:
        raise ValueError("Part wire path has zero length.")

    plt = _import_pyplot()
    colors = _import_matplotlib_colors()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_mesh_3d(ax, part)

    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    route_position = (cumulative[:-1] + 0.5 * lengths) / total_length
    segments = np.stack((points[:-1], points[1:]), axis=1)
    valid = lengths > 1e-15
    style_keys = _wire_path_style_keys(part)
    style_specs = {
        "winding": ("-", WIRE_PATH_LINEWIDTH + 0.35, "winding segment"),
        "surface_connection": ("--", WIRE_PATH_LINEWIDTH, "surface connection"),
        "offset_surface": (":", WIRE_PATH_LINEWIDTH + 0.15, "normal-offset segment"),
        "free_3d": ("-.", WIRE_PATH_LINEWIDTH + 0.15, "free-3D connector"),
    }
    used_styles: list[str] = []
    for style_key, (linestyle, linewidth, _label) in style_specs.items():
        indices = np.flatnonzero(valid & (style_keys == style_key))
        if len(indices) == 0:
            continue
        used_styles.append(style_key)
        collection = _new_line3d_collection(
            segments[indices],
            colors=cmap(norm(route_position[indices])),
            linewidths=linewidth,
            linestyles=linestyle,
            alpha=0.98,
        )
        ax.add_collection3d(collection)

    scale = _plot_geometry_scale(_stack_points([part.mesh.vertices, points]))
    arrow_count = _adaptive_arrow_count(total_length, scale, minimum=4, maximum=16, spacing_scale=0.65)
    arrow_length = 0.045 * scale
    for position, tangent, fraction in _path_samples_at_fractions(points, np.linspace(0.06, 0.94, arrow_count)):
        ax.quiver(
            *position,
            *tangent,
            length=arrow_length,
            normalize=True,
            color=cmap(norm(fraction)),
            arrow_length_ratio=0.42,
            linewidth=1.1,
        )

    loop_visit_count, connector_event_count = _annotate_wire_owner_runs(ax, part)
    closed = np.linalg.norm(points[0] - points[-1]) <= max(1e-10, 1e-8 * scale)
    if closed:
        ax.scatter(*points[0], color="#111827", marker="D", s=46, depthshade=False, label="reference start/end")
        ax.text(*points[0], " REF", color="#111827", fontsize=8)
        terminal_note = "Closed route: s=0 is the stored reference point; s/L=1 returns to the same point."
    else:
        ax.scatter(*points[0], color="#16a34a", marker="^", s=52, depthshade=False, label="IN terminal")
        ax.scatter(*points[-1], color="#dc2626", marker="s", s=45, depthshade=False, label="OUT terminal")
        ax.text(*points[0], " IN", color="#166534", fontsize=8)
        ax.text(*points[-1], " OUT", color="#991b1b", fontsize=8)
        terminal_note = "Open route: color and arrows run from IN (s=0) to OUT (s/L=1)."

    for style_key in used_styles:
        linestyle, linewidth, label = style_specs[style_key]
        ax.plot([], [], [], color="#334155", linewidth=linewidth, linestyle=linestyle, label=label)
    ax.plot([], [], [], color="#334155", marker=">", linestyle="None", label="current-direction arrows")
    sequence_note = (
        f" Numbered loop visits: {loop_visit_count}; labeled inter-group connectors: {connector_event_count}."
    )
    ax.text2D(
        0.02,
        0.02,
        terminal_note + sequence_note,
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    _add_colorbar(fig, ax, cmap, norm, "normalized traversal distance, s / L")
    _finish_3d_axis(ax, _stack_points([part.mesh.vertices, points]), view=FACE_ISOMETRIC_VIEW)
    _deduplicate_legend(ax)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_wire_paths_3d(
    parts: Sequence[CoilPart],
    *,
    title: str = "Connected Coil Wire Paths on All Coil Parts",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot connected wire paths for every coil part in one 3D assembly view."""

    valid_parts = [part for part in parts if part.wire_path is not None and len(part.wire_path.v) >= 2]
    if not valid_parts:
        raise ValueError("No coil part has a wire path.")
    plt = _import_pyplot()
    cmap = plt.get_cmap("tab10")
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_meshes_3d(ax, parts)
    all_points: list[np.ndarray] = [part.mesh.vertices for part in parts]
    for part_ind, part in enumerate(valid_parts, start=1):
        color = cmap((part_ind - 1) % 10)
        _plot_path_3d(
            ax, part.wire_path.v, color=color, linewidth=WIRE_PATH_LINEWIDTH, label=f"part {part_ind} wire path"
        )
        all_points.append(part.wire_path.v)
        if part.shift_array is not None and np.any(np.asarray(part.shift_array) > 0):
            _draw_shifted_segments_3d(ax, part.wire_path.v, part.shift_array)
    _finish_3d_axis(ax, _stack_points(all_points), view=FACE_ISOMETRIC_VIEW)
    _deduplicate_legend(ax)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_layout_field_overview_3d(
    result: CoilGenResult,
    *,
    title: str = "Coil Layout with DSV and Suppression Field Points",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot coil layout together with DSV and suppression target-point fields."""

    plt = _import_pyplot()
    coords = np.asarray(result.target_field.coords, dtype=float)
    layout_bz = layout_bz_1a(result)
    layout_bz_display = display_field_values_1a(result, layout_bz)
    layout_map = field_scalar_maps(result)["layout"]
    if len(coords) == 0 or len(layout_bz) != len(coords):
        raise ValueError("Result has no compatible target-point layout field.")

    dsv_mask = dsv_target_mask(result.target_field)
    suppression_mask = suppression_target_mask(result.target_field)
    finite = np.isfinite(layout_bz)
    dsv_mask = dsv_mask & finite
    suppression_mask = suppression_mask & finite
    if not np.any(dsv_mask) and not np.any(suppression_mask):
        raise ValueError("Result has no finite DSV or suppression field samples.")

    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_meshes_3d(ax, result.coil_parts)

    all_points: list[np.ndarray] = [part.mesh.vertices for part in result.coil_parts]
    _draw_layout_paths_or_contours_3d(ax, result.coil_parts, all_points)

    eligible_mask = dsv_mask | suppression_mask
    display_indices = pointcloud_plot_indices(
        result.input_data,
        len(coords),
        eligible_mask=eligible_mask,
        seed_offset=11,
    )
    display_mask = np.zeros(len(coords), dtype=bool)
    display_mask[display_indices] = True
    display_dsv_mask = dsv_mask & display_mask
    display_suppression_mask = suppression_mask & display_mask
    all_points.append(coords[eligible_mask])

    color_values = layout_bz_display[eligible_mask]
    norm = _symmetric_color_norm(color_values)
    scatter = None
    if np.any(display_dsv_mask):
        dsv_coords = coords[display_dsv_mask]
        scatter = ax.scatter(
            dsv_coords[:, 0],
            dsv_coords[:, 1],
            dsv_coords[:, 2],
            c=layout_bz_display[display_dsv_mask],
            cmap="coolwarm",
            norm=norm,
            s=20,
            alpha=FIELD_POINT_ALPHA,
            marker="o",
            depthshade=False,
            label="DSV layout Bz samples",
        )
        all_points.append(dsv_coords)
    if np.any(display_suppression_mask):
        suppression_coords = coords[display_suppression_mask]
        scatter = ax.scatter(
            suppression_coords[:, 0],
            suppression_coords[:, 1],
            suppression_coords[:, 2],
            c=layout_bz_display[display_suppression_mask],
            cmap="coolwarm",
            norm=norm,
            s=18,
            alpha=FIELD_POINT_ALPHA,
            marker="^",
            depthshade=False,
            label="suppression layout Bz samples",
        )
        all_points.append(suppression_coords)
    if scatter is not None:
        _add_artist_colorbar(
            fig,
            ax,
            scatter,
            format_label_with_unit(layout_map.label, display_field_unit_1a(result)),
            shrink=0.72,
            pad=0.08,
        )
    _finish_3d_axis(ax, _stack_points(all_points), view=FACE_ISOMETRIC_VIEW)
    _deduplicate_legend(ax)
    fig.tight_layout()
    return _finish_figure(fig, plt, show=show, block=block)


def plot_problem_setup_geometry_3d(
    result: CoilGenResult,
    *,
    title: str = "Problem Setup Geometry: Current Surface and Target Field",
    field_overlay: ProblemSetupFieldOverlay | None = None,
    pyplot_managed: bool = True,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot current-carrying surfaces with a requested or explicit field point cloud."""

    plt = _import_pyplot()
    collections = _import_mplot3d_collections()
    if field_overlay is None:
        coords = np.asarray(result.target_field.coords, dtype=float)
        target_map = shim_field_scalar_maps(result)["target"]
        target_bz = np.asarray(target_map.values, dtype=float).reshape(-1)
        dsv_mask = dsv_target_mask(result.target_field)
        suppression_mask = suppression_target_mask(result.target_field)
        colorbar_label = target_map.label_with_unit
        dsv_label = "DSV target field samples"
        suppression_label = "suppression target field samples"
        cmap_name = "coolwarm"
    else:
        coords = np.asarray(field_overlay.coords, dtype=float)
        target_bz = np.asarray(field_overlay.values, dtype=float).reshape(-1)
        dsv_mask = np.asarray(field_overlay.dsv_mask, dtype=bool).reshape(-1)
        suppression_mask = np.asarray(field_overlay.suppression_mask, dtype=bool).reshape(-1)
        colorbar_label = str(field_overlay.label_with_unit)
        dsv_label = str(field_overlay.dsv_label)
        suppression_label = str(field_overlay.suppression_label)
        cmap_name = str(field_overlay.cmap)
    if len(coords) == 0 or len(target_bz) != len(coords):
        raise ValueError("Result has no compatible target field point cloud.")
    if len(dsv_mask) != len(coords) or len(suppression_mask) != len(coords):
        raise ValueError("Problem-setup field masks must match the field point count.")
    finite = _finite_point_scalar_mask(coords, target_bz)
    if not np.any(finite):
        raise ValueError("Target field point cloud has no finite samples.")

    fig, ax = _new_3d_figure(plt, title, pyplot_managed=pyplot_managed)
    all_points: list[np.ndarray] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        mesh = part.mesh
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        collection = collections.Poly3DCollection(
            vertices[faces],
            facecolors="#bae6fd",
            edgecolors="#64748b",
            linewidths=0.12,
            alpha=0.26,
        )
        collection.set_label(f"part {part_ind} current surface")
        ax.add_collection3d(collection)
        all_points.append(vertices)

    dsv_mask = dsv_mask & finite
    suppression_mask = suppression_mask & finite
    if not np.any(dsv_mask) and not np.any(suppression_mask):
        dsv_mask = finite
    eligible_mask = dsv_mask | suppression_mask
    display_indices = pointcloud_plot_indices(
        result.input_data,
        len(coords),
        eligible_mask=eligible_mask,
        seed_offset=17,
    )
    display_mask = np.zeros(len(coords), dtype=bool)
    display_mask[display_indices] = True
    display_dsv_mask = dsv_mask & display_mask
    display_suppression_mask = suppression_mask & display_mask
    all_points.append(coords[eligible_mask])
    norm = _symmetric_color_norm(target_bz[finite])
    cmap = plt.get_cmap(cmap_name)
    scatter = None
    if np.any(display_dsv_mask):
        dsv_coords = coords[display_dsv_mask]
        scatter = ax.scatter(
            dsv_coords[:, 0],
            dsv_coords[:, 1],
            dsv_coords[:, 2],
            c=target_bz[display_dsv_mask],
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=FIELD_POINT_ALPHA,
            marker="o",
            depthshade=False,
            label=dsv_label,
        )
        all_points.append(dsv_coords)
    if np.any(display_suppression_mask):
        suppression_coords = coords[display_suppression_mask]
        scatter = ax.scatter(
            suppression_coords[:, 0],
            suppression_coords[:, 1],
            suppression_coords[:, 2],
            c=target_bz[display_suppression_mask],
            cmap=cmap,
            norm=norm,
            s=18,
            alpha=FIELD_POINT_ALPHA,
            marker="^",
            depthshade=False,
            label=suppression_label,
        )
        all_points.append(suppression_coords)
    if scatter is not None:
        _add_artist_colorbar(fig, ax, scatter, colorbar_label, shrink=0.72, pad=0.08)
    _finish_3d_axis(ax, _stack_points(all_points), view=FACE_ISOMETRIC_VIEW)
    _deduplicate_legend(ax)
    fig.tight_layout()
    return _finish_figure(fig, plt, show=show, block=block)


def plot_raw_contours_3d(
    part: CoilPart,
    *,
    title: str = "Raw Stream-Function Contours Before Topology Grouping",
    show_source_contours: bool = True,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot active contours, optionally with pre-sparsification contours dashed."""

    if not part.contour_lines:
        raise ValueError("Part has no contours.")
    plt = _import_pyplot()
    colors = _import_matplotlib_colors()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_mesh_3d(ax, part)
    if show_source_contours and part.contour_sparsification is not None:
        for source in part.contour_sparsification.source_contours:
            _plot_path_3d(
                ax,
                _closed_path(source.v),
                color="#6b7280",
                linewidth=1.0,
                alpha=0.35,
                linestyle="--",
            )
    potentials = np.asarray([line.potential for line in part.contour_lines], dtype=float)
    norm = colors.Normalize(vmin=float(np.min(potentials)), vmax=float(np.max(potentials)))
    cmap = plt.get_cmap("coolwarm")
    for line in part.contour_lines:
        _plot_path_3d(
            ax,
            _closed_path(line.v),
            color=cmap(norm(line.potential)),
            linewidth=_contour_display_linewidth(line),
        )
        if line.turn_multiplicity > 1:
            center = np.mean(_closed_path(line.v), axis=0)
            ax.text(*center, f"x{line.turn_multiplicity}", color=cmap(norm(line.potential)), fontsize=8)
    ax.text2D(
        0.02,
        0.02,
        f"{part.nominal_contour_count} geometries / {part.physical_turn_count} turns",
        transform=ax.transAxes,
        fontsize=9,
    )
    _add_colorbar(fig, ax, cmap, norm, format_label_with_unit("stream function", STREAM_FUNCTION_UNIT))
    _finish_3d_axis(
        ax, _stack_points([part.mesh.vertices, *(line.v for line in part.contour_lines)]), view=FACE_ISOMETRIC_VIEW
    )
    return _finish_figure(fig, plt, show=show, block=block)


def plot_contour_current_orientation_3d(
    part: CoilPart,
    *,
    title: str = "Contour Current Direction and Surface Chirality",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot current-oriented contours with arrows and chirality relative to local mesh normals."""

    if not part.contour_lines:
        raise ValueError("Part has no contours.")
    from .layout import contour_orientation_diagnostics

    diagnostics = contour_orientation_diagnostics(part)
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_mesh_3d(ax, part)
    all_points = _stack_points([part.mesh.vertices, *(line.v for line in part.contour_lines)])
    scale = _plot_geometry_scale(all_points)
    arrow_length = 0.045 * scale
    seen_chiralities: set[int] = set()

    for contour_ind, (line, row) in enumerate(zip(part.contour_lines, diagnostics)):
        orientation_sign = int(row["orientation_sign"])
        current_sign = int(row["current_orientation_sign"])
        chirality = (
            orientation_sign * current_sign
            if bool(row["chirality_defined"]) and orientation_sign and current_sign
            else 0
        )
        color = _contour_chirality_color(chirality)
        points = _closed_path(line.v)
        _plot_path_3d(
            ax,
            points,
            color=color,
            linewidth=_contour_display_linewidth(line),
            label=_contour_chirality_label(chirality) if chirality not in seen_chiralities else None,
        )
        seen_chiralities.add(chirality)
        if current_sign == 0:
            label_position = np.mean(points, axis=0)
        else:
            contour_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            arrow_count = _adaptive_arrow_count(
                contour_length,
                scale,
                minimum=1,
                maximum=3,
                spacing_scale=1.25,
            )
            first_fraction = (0.13 + 0.173 * contour_ind) % 1.0
            fractions = (first_fraction + np.arange(arrow_count, dtype=float) / arrow_count) % 1.0
            samples = _path_samples_at_fractions(points, fractions)
            for position, tangent, _fraction in samples:
                ax.quiver(
                    *position,
                    *(current_sign * tangent),
                    length=arrow_length,
                    normalize=True,
                    color=color,
                    arrow_length_ratio=0.44,
                    linewidth=1.15,
                )
            label_fraction = (0.31 + 0.38196601125 * contour_ind) % 1.0
            label_sample = _path_samples_at_fractions(points, [label_fraction])
            label_position = label_sample[0][0] if label_sample else np.mean(points, axis=0)
        multiplier = f" x{line.turn_multiplicity}" if line.turn_multiplicity > 1 else ""
        chirality_text = "+" if chirality > 0 else "-" if chirality < 0 else "?"
        ax.text(
            *label_position,
            f" L{contour_ind}{multiplier} chi{chirality_text}",
            color=color,
            fontsize=8,
        )

    ax.plot([], [], [], color="#334155", marker=">", linestyle="None", label="arrowheads: actual current direction")
    ax.text2D(
        0.02,
        0.02,
        "chi = sign(I) sign(A dot n_bar); n_bar is the contour's mean oriented local mesh normal.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    _finish_3d_axis(ax, all_points, view=FACE_ISOMETRIC_VIEW)
    _deduplicate_legend(ax)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_raw_contours_all_parts_3d(
    parts: Sequence[CoilPart],
    *,
    title: str = "Raw Stream-Function Contours on All Coil Parts",
    show_source_contours: bool = True,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot active contours for all parts, optionally with sparse source contours."""

    contour_lines = [line for part in parts for line in part.contour_lines]
    if not contour_lines:
        raise ValueError("No coil part has contours.")
    plt = _import_pyplot()
    colors = _import_matplotlib_colors()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_meshes_3d(ax, parts)
    potentials = np.asarray([line.potential for line in contour_lines], dtype=float)
    norm = colors.Normalize(vmin=float(np.min(potentials)), vmax=float(np.max(potentials)))
    cmap = plt.get_cmap("coolwarm")
    all_points: list[np.ndarray] = [part.mesh.vertices for part in parts]
    for part in parts:
        if show_source_contours and part.contour_sparsification is not None:
            for source in part.contour_sparsification.source_contours:
                _plot_path_3d(
                    ax,
                    _closed_path(source.v),
                    color="#6b7280",
                    linewidth=1.0,
                    alpha=0.3,
                    linestyle="--",
                )
                all_points.append(source.v)
    for line in contour_lines:
        _plot_path_3d(
            ax,
            _closed_path(line.v),
            color=cmap(norm(line.potential)),
            linewidth=_contour_display_linewidth(line),
        )
        if line.turn_multiplicity > 1:
            center = np.mean(_closed_path(line.v), axis=0)
            ax.text(*center, f"x{line.turn_multiplicity}", color=cmap(norm(line.potential)), fontsize=8)
        all_points.append(line.v)
    geometry_count = sum(part.nominal_contour_count for part in parts)
    turn_count = sum(part.physical_turn_count for part in parts)
    ax.text2D(
        0.02,
        0.02,
        f"{geometry_count} geometries / {turn_count} turns",
        transform=ax.transAxes,
        fontsize=9,
    )
    _add_colorbar(fig, ax, cmap, norm, format_label_with_unit("stream function", STREAM_FUNCTION_UNIT))
    _finish_3d_axis(ax, _stack_points(all_points), view=FACE_ISOMETRIC_VIEW)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_contour_groups_3d(
    part: CoilPart,
    *,
    title: str = "Topological Loop Groups of Stream-Function Contours",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot contours colored by intrinsic topological loop group."""

    if not part.groups:
        raise ValueError("Part has no contour groups.")
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_mesh_3d(ax, part)
    cmap = plt.get_cmap("tab20")
    all_points: list[np.ndarray] = [part.mesh.vertices]
    for group_ind, group in enumerate(part.groups):
        color = cmap(group_ind % 20)
        for loop in group.loops:
            _plot_path_3d(
                ax,
                _closed_path(loop.v),
                color=color,
                linewidth=GROUP_CONTOUR_LINEWIDTH,
                label=f"G{group_ind}" if loop is group.loops[0] else None,
            )
            all_points.append(loop.v)
    _finish_3d_axis(ax, _stack_points(all_points), view=FACE_ISOMETRIC_VIEW)
    ax.legend(loc="upper left")
    return _finish_figure(fig, plt, show=show, block=block)


def plot_contour_groups_all_parts_3d(
    parts: Sequence[CoilPart],
    *,
    title: str = "Topological Loop Groups on All Coil Parts",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot topological contour groups for every coil part in one 3D assembly view."""

    if not any(part.groups for part in parts):
        raise ValueError("No coil part has contour groups.")
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_meshes_3d(ax, parts)
    cmap = plt.get_cmap("tab20")
    all_points: list[np.ndarray] = [part.mesh.vertices for part in parts]
    color_ind = 0
    for part_ind, part in enumerate(parts, start=1):
        for group_ind, group in enumerate(part.groups):
            color = cmap(color_ind % 20)
            color_ind += 1
            for loop in group.loops:
                label = f"P{part_ind} G{group_ind}" if loop is group.loops[0] else None
                _plot_path_3d(ax, _closed_path(loop.v), color=color, linewidth=GROUP_CONTOUR_LINEWIDTH, label=label)
                all_points.append(loop.v)
    _finish_3d_axis(ax, _stack_points(all_points), view=FACE_ISOMETRIC_VIEW)
    ax.legend(loc="upper left")
    return _finish_figure(fig, plt, show=show, block=block)


def plot_stream_function_3d(
    part: CoilPart, *, title: str = "Optimized Stream Function on Coil Mesh", show: bool = True, block: bool = False
) -> Figure:
    """Plot the optimized stream function as a colored 3D surface."""

    return _plot_face_scalar_3d(
        part,
        _face_average_stream_function(part),
        title=title,
        colorbar_label=format_label_with_unit("stream function, robust 99% display", STREAM_FUNCTION_UNIT),
        display_percentile=99.0,
        nonnegative=False,
        show=show,
        block=block,
    )


def plot_stream_function_all_parts_3d(
    parts: Sequence[CoilPart],
    *,
    title: str = "Optimized Stream Function on All Coil Parts",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot optimized stream function on all coil parts with one shared color scale."""

    return _plot_face_scalar_all_parts_3d(
        parts,
        [_face_average_stream_function(part) for part in parts],
        title=title,
        colorbar_label=format_label_with_unit("stream function, robust 99% display", STREAM_FUNCTION_UNIT),
        display_percentile=99.0,
        nonnegative=False,
        show=show,
        block=block,
    )


def plot_current_density_3d(
    part: CoilPart, *, title: str = "Surface Current-Density Magnitude |J|", show: bool = True, block: bool = False
) -> Figure:
    """Plot the current-density magnitude as a colored 3D surface."""

    return _plot_face_scalar_3d(
        part,
        _current_density_magnitude(part),
        title=title,
        colorbar_label=format_label_with_unit("|J|, robust 99.5% display", CURRENT_DENSITY_UNIT),
        display_percentile=99.5,
        nonnegative=True,
        show=show,
        block=block,
    )


def plot_current_density_all_parts_3d(
    parts: Sequence[CoilPart],
    *,
    title: str = "Surface Current-Density Magnitude on All Coil Parts",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot current-density magnitude on all coil parts with one shared color scale."""

    return _plot_face_scalar_all_parts_3d(
        parts,
        [_current_density_magnitude(part) for part in parts],
        title=title,
        colorbar_label=format_label_with_unit("|J|, robust 99.5% display", CURRENT_DENSITY_UNIT),
        display_percentile=99.5,
        nonnegative=True,
        show=show,
        block=block,
    )


def plot_interconnect_debug_3d(
    part: CoilPart, *, title: str = "Contour Opening Cuts and Group Connectors", show: bool = True, block: bool = False
) -> Figure:
    """Plot contours, within-group cuts, return paths, and inter-group cuts."""

    if not part.connected_groups:
        raise ValueError("Part has no connected groups.")
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_mesh_3d(ax, part)
    all_points: list[np.ndarray] = [part.mesh.vertices]
    for group in part.groups:
        for loop in group.loops:
            _plot_path_3d(ax, _closed_path(loop.v), color="#1f77b4", linewidth=RAW_CONTOUR_LINEWIDTH)
            all_points.append(loop.v)
        for cut_v in group.opening_cuts_v:
            _plot_path_3d(ax, cut_v, color="#f97316", linewidth=2.0, label="within-group opening cuts")
            all_points.append(cut_v)
    for connected in part.connected_groups:
        _plot_path_3d(
            ax, connected.return_path_v, color="#111827", linewidth=1.8, linestyle="--", label="group return path"
        )
        all_points.append(connected.return_path_v)
    for cut in part.opening_cuts_among_groups:
        for cut_v in (cut.cut1_v, cut.cut2_v):
            _plot_path_3d(ax, cut_v, color="#a855f7", linewidth=2.0, label="inter-group opening cuts")
            all_points.append(cut_v)
        _plot_path_3d(ax, cut.connection_path_v, color="#7e22ce", linewidth=1.9, label="inter-group connector")
        all_points.append(cut.connection_path_v)
        _plot_path_3d(
            ax,
            cut.closing_connection_path_v,
            color="#c026d3",
            linewidth=1.9,
            label="inter-group closing connector",
        )
        all_points.append(cut.closing_connection_path_v)
    _finish_3d_axis(ax, _stack_points(all_points))
    _deduplicate_legend(ax)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_interconnect_debug_all_parts_3d(
    parts: Sequence[CoilPart],
    *,
    title: str = "Contour Opening Cuts and Group Connectors on All Coil Parts",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot contour openings and connectors for every coil part in one 3D view."""

    if not any(part.connected_groups for part in parts):
        raise ValueError("No coil part has connected groups.")
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    _draw_reference_meshes_3d(ax, parts)
    all_points: list[np.ndarray] = [part.mesh.vertices for part in parts]
    for part in parts:
        for group in part.groups:
            for loop in group.loops:
                _plot_path_3d(ax, _closed_path(loop.v), color="#1f77b4", linewidth=RAW_CONTOUR_LINEWIDTH)
                all_points.append(loop.v)
            for cut_v in group.opening_cuts_v:
                _plot_path_3d(ax, cut_v, color="#f97316", linewidth=2.0, label="within-group opening cuts")
                all_points.append(cut_v)
        for connected in part.connected_groups:
            _plot_path_3d(
                ax, connected.return_path_v, color="#111827", linewidth=1.8, linestyle="--", label="group return path"
            )
            all_points.append(connected.return_path_v)
        for cut in part.opening_cuts_among_groups:
            for cut_v in (cut.cut1_v, cut.cut2_v):
                _plot_path_3d(ax, cut_v, color="#a855f7", linewidth=2.0, label="inter-group opening cuts")
                all_points.append(cut_v)
            _plot_path_3d(ax, cut.connection_path_v, color="#7e22ce", linewidth=1.9, label="inter-group connector")
            all_points.append(cut.connection_path_v)
            _plot_path_3d(
                ax,
                cut.closing_connection_path_v,
                color="#c026d3",
                linewidth=1.9,
                label="inter-group closing connector",
            )
            all_points.append(cut.closing_connection_path_v)
    _finish_3d_axis(ax, _stack_points(all_points))
    _deduplicate_legend(ax)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_swept_body_3d(
    part: CoilPart, *, title: str = "Swept Conductor Body Mesh", show: bool = True, block: bool = False
) -> Figure:
    """Plot the swept conductor body mesh if it exists."""

    if part.swept_body is None:
        raise ValueError("Part has no swept conductor body.")
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    ax.plot_trisurf(
        part.swept_body.vertices[:, 0],
        part.swept_body.vertices[:, 1],
        part.swept_body.vertices[:, 2],
        triangles=part.swept_body.faces,
        color="#c7d2fe",
        edgecolor="#64748b",
        linewidth=0.15,
        alpha=0.75,
        shade=False,
    )
    _finish_3d_axis(ax, part.swept_body.vertices)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_swept_bodies_all_parts_3d(
    parts: Sequence[CoilPart],
    *,
    title: str = "Swept Conductor Body Meshes on All Coil Parts",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot swept conductor body meshes for every coil part in one 3D view."""

    valid_parts = [part for part in parts if part.swept_body is not None]
    if not valid_parts:
        raise ValueError("No coil part has a swept conductor body.")
    plt = _import_pyplot()
    fig, ax = _new_3d_figure(plt, title)
    cmap = plt.get_cmap("tab10")
    all_points: list[np.ndarray] = []
    for part_ind, part in enumerate(valid_parts, start=1):
        body = part.swept_body
        color = cmap((part_ind - 1) % 10)
        ax.plot_trisurf(
            body.vertices[:, 0],
            body.vertices[:, 1],
            body.vertices[:, 2],
            triangles=body.faces,
            color=color,
            edgecolor="#64748b",
            linewidth=0.12,
            alpha=0.65,
            shade=False,
        )
        all_points.append(body.vertices)
    _finish_3d_axis(ax, _stack_points(all_points))
    return _finish_figure(fig, plt, show=show, block=block)


def plot_field_comparison_matplotlib(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot target, layout, and stream-function Bz values over target point index."""

    plt = _import_pyplot()
    fields = field_scalar_maps(result)
    target = fields["target"].values
    layout = fields["layout"].values
    stream = fields["stream"].values
    x = np.arange(len(target))
    figure_title = "Target Points: Bz Field Comparison per Ampere"
    fig = plt.figure(num=figure_title)
    fig.clf()
    ax = fig.subplots()
    ax.plot(x, target, color="#111827", linewidth=1.8, label=f"{fields['target'].label} per 1 A")
    ax.plot(x, layout, color="#d73a49", linewidth=1.4, label=f"{fields['layout'].label} per 1 A")
    ax.plot(x, stream, color="#1f6feb", linewidth=1.4, label=f"{fields['stream'].label} per 1 A")
    ax.set_xlabel("target point index [-]")
    ax.set_ylabel(format_label_with_unit("Bz", display_field_unit_1a(result)))
    _set_axis_title(
        ax,
        f"Bz at target points: {fields['target'].label}, {fields['layout'].label}, and {fields['stream'].label}",
        width=72,
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _finish_figure(fig, plt, show=show, block=block)


def plot_target_field_maps_3d(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Backward-compatible alias for field point-cloud maps."""

    return plot_field_pointclouds_3d(result, show=show, block=block)


def plot_field_pointclouds_3d(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot target, layout, stream, and layout-error Bz values as 3D point clouds."""

    plt = _import_pyplot()
    coords = np.asarray(result.target_field.coords, dtype=float)
    scalar_maps = list(field_scalar_maps(result).values())
    coordinate_finite = np.all(np.isfinite(coords), axis=1)
    display_indices = pointcloud_plot_indices(
        result.input_data,
        len(coords),
        eligible_mask=coordinate_finite,
        seed_offset=23,
    )
    display_mask = np.zeros(len(coords), dtype=bool)
    display_mask[display_indices] = True
    fig = plt.figure(num="Target Points: Bz Field Maps per Ampere", figsize=(11, 8))
    fig.clf()
    for index, scalar_map in enumerate(scalar_maps, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        map_values = np.asarray(scalar_map.values, dtype=float).reshape(-1)
        finite = _finite_point_scalar_mask(coords, map_values)
        plotted = finite & display_mask
        if np.any(plotted):
            scatter = ax.scatter(
                coords[plotted, 0],
                coords[plotted, 1],
                coords[plotted, 2],
                c=map_values[plotted],
                cmap="coolwarm",
                norm=_linear_color_norm(map_values[finite]),
                s=24,
                alpha=FIELD_POINT_ALPHA,
                depthshade=False,
            )
            _add_artist_colorbar(fig, ax, scatter, scalar_map.label_with_unit, shrink=0.65, pad=0.08)
        else:
            ax.text2D(0.08, 0.5, "No finite samples in this domain", transform=ax.transAxes, color="#64748b")
        _set_axis_title(ax, _scalar_subplot_title(scalar_map))
        _finish_3d_axis(ax, coords[coordinate_finite], view=FACE_ISOMETRIC_VIEW)
    fig.tight_layout()
    return _finish_figure(fig, plt, show=show, block=block)


def plot_gradient_maps_3d(
    result: CoilGenResult,
    *,
    components: Sequence[str] = ("magnitude", "dbz_dx", "dbz_dy", "dbz_dz"),
    neighbor_count: int = 12,
    show: bool = True,
    block: bool = False,
) -> list[Figure]:
    """Plot local Bz gradient maps as 3D point clouds."""

    plt = _import_pyplot()
    figures = [
        _plot_scalar_pointcloud_grid(
            result,
            [
                *gradient_component_scalar_maps(result, component=component, neighbor_count=neighbor_count).values(),
                *gradient_relative_error_scalar_maps(
                    result, component=component, neighbor_count=neighbor_count
                ).values(),
            ],
            figure_title=f"DSV Target Points: Bz Gradient Maps - {_gradient_component_title(component)}",
            show=False,
            block=False,
        )
        for component in components
    ]
    if show:
        plt.show(block=block)
    return figures


def plot_error_maps_3d(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot absolute and relative-percent layout/stream Bz error maps as 3D point clouds."""

    return _plot_scalar_pointcloud_grid(
        result,
        [*field_error_scalar_maps(result).values(), *field_relative_error_scalar_maps(result).values()],
        figure_title="DSV Target Points: Bz Error Maps - Absolute and Relative",
        show=show,
        block=block,
    )


def plot_shim_pointclouds_3d(
    result: CoilGenResult,
    *,
    source: str = "layout",
    domain: str = "dsv",
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot before- and after-shimming B0 inhomogeneity on the target point cloud."""

    plt = _import_pyplot()
    coords = np.asarray(result.target_field.coords, dtype=float)
    maps = shim_field_scalar_maps(result)
    after_source = _after_shim_source(source)
    entries = (
        ("before", maps["before_inhomogeneity"]),
        (after_source, maps[_shim_source_key(after_source)]),
    )
    domain_mask = target_domain_mask(result, domain)
    finite_masks = [
        domain_mask & _finite_point_scalar_mask(coords, np.asarray(scalar_map.values, dtype=float))
        for _, scalar_map in entries
    ]
    if not any(np.any(mask) for mask in finite_masks):
        raise ValueError("Shim point-cloud plot has no finite samples.")
    eligible_mask = np.logical_or.reduce(finite_masks)
    display_indices = pointcloud_plot_indices(
        result.input_data,
        len(coords),
        eligible_mask=eligible_mask,
        seed_offset=29,
    )
    display_mask = np.zeros(len(coords), dtype=bool)
    display_mask[display_indices] = True
    color_limit = _shim_reference_color_limit(result, source=after_source, domain=domain)
    if color_limit is None:
        color_limit = max(
            float(np.nanmax(np.abs(np.asarray(scalar_map.values, dtype=float).reshape(-1)[mask])))
            for (_, scalar_map), mask in zip(entries, finite_masks)
            if np.any(mask)
        )
    color_limit = max(float(color_limit), 1e-30)

    colors = _import_matplotlib_colors()
    norm = colors.Normalize(vmin=-color_limit, vmax=color_limit)
    cmap = plt.get_cmap("RdBu_r")
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        cmap.set_bad("#f8fafc")
    except Exception:
        pass

    figure_title = "Shim Target Points: Before and After B0 Inhomogeneity"
    fig = plt.figure(num=figure_title, figsize=(12, 5.4))
    fig.clf()
    axes = []
    for index, ((source_name, scalar_map), finite) in enumerate(zip(entries, finite_masks), start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        axes.append(ax)
        plotted = finite & display_mask
        if np.any(plotted):
            plot_coords = coords[plotted]
            values = np.asarray(scalar_map.values, dtype=float).reshape(-1)[plotted]
            ax.scatter(
                plot_coords[:, 0],
                plot_coords[:, 1],
                plot_coords[:, 2],
                c=values,
                cmap=cmap,
                norm=norm,
                s=24,
                alpha=FIELD_POINT_ALPHA,
                depthshade=False,
            )
            stats = shim_homogeneity_stats(result, source=source_name, domain=domain)
            subtitle = (
                f"{scalar_map.label_with_unit}\n"
                f"std {float(stats['std']):.4g}, RMS {float(stats['rms']):.4g}, "
                f"max |{_shim_abs_metric_label(scalar_map.key)}| {float(stats['max_abs']):.4g} {stats['unit']}"
            )
        else:
            plot_coords = coords[eligible_mask]
            ax.text2D(0.08, 0.5, "No finite samples in this domain", transform=ax.transAxes, color="#64748b")
            subtitle = scalar_map.label_with_unit
        _set_axis_title(ax, subtitle, width=46)
        _finish_3d_axis(ax, coords[eligible_mask], view=FACE_ISOMETRIC_VIEW)

    scalar_mappable = _import_scalar_mappable()(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=axes, shrink=0.72, pad=0.04)
    _set_scientific_colorbar_ticks(colorbar, -color_limit, color_limit)
    colorbar.set_label(format_label_with_unit("B0 inhomogeneity / residual", entries[0][1].unit))
    # ``tight_layout`` does not support this shared 3-D colorbar and emits a
    # warning on every plot-book build. Explicit margins are deterministic.
    fig.subplots_adjust(left=0.02, right=0.9, bottom=0.04, top=0.91, wspace=0.04)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_shim_field_histograms(
    result: CoilGenResult,
    *,
    source: str = "layout",
    domain: str = "dsv",
    bins: int = 54,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot before- and after-shimming field histograms over target points."""

    plt = _import_pyplot()
    maps = shim_field_scalar_maps(result)
    after_source = _after_shim_source(source)
    before_map = maps["before_inhomogeneity"]
    after_map = maps[_shim_source_key(after_source)]
    before_values = _shim_histogram_domain_values(result, before_map, domain=domain)
    after_values = _shim_histogram_domain_values(result, after_map, domain=domain)
    if len(before_values) == 0 or len(after_values) == 0:
        raise ValueError("Shim histogram plot has no finite samples.")

    bin_count = max(int(bins), 8)
    signed_limit = max(float(np.max(np.abs(before_values))), float(np.max(np.abs(after_values))), 1e-30)
    signed_bins = np.linspace(-signed_limit, signed_limit, bin_count + 1)
    abs_limit = max(float(np.max(np.abs(before_values))), float(np.max(np.abs(after_values))), 1e-30)
    abs_bins = np.linspace(0.0, abs_limit, bin_count + 1)
    unit = before_map.unit

    figure_title = "Shim Target Points: Field Histograms Before and After"
    fig = plt.figure(num=figure_title, figsize=(12.4, 5.6))
    fig.clf()
    axes = fig.subplots(1, 2)
    signed_ax, abs_ax = axes

    before_color = "#64748b"
    after_color = "#0f766e"
    signed_ax.hist(before_values, bins=signed_bins, color=before_color, alpha=0.45, label="before shimming")
    signed_ax.hist(after_values, bins=signed_bins, color=after_color, alpha=0.55, label="after shimming")
    signed_ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.8)
    signed_ax.set_xlabel(format_label_with_unit("B0 inhomogeneity / residual", unit))
    signed_ax.set_ylabel("target point count")
    _set_axis_title(signed_ax, "Signed field distribution", width=42)
    signed_ax.grid(True, alpha=0.22)
    signed_ax.legend(loc="best")

    abs_ax.hist(
        np.abs(before_values),
        bins=abs_bins,
        histtype="step",
        linewidth=2.3,
        color=before_color,
        label="before shimming",
    )
    abs_ax.hist(
        np.abs(after_values), bins=abs_bins, histtype="step", linewidth=2.3, color=after_color, label="after shimming"
    )
    abs_ax.set_xlabel(format_label_with_unit("abs(B0 inhomogeneity / residual)", unit))
    abs_ax.set_ylabel("target point count")
    _set_axis_title(abs_ax, "Absolute field distribution", width=42)
    abs_ax.grid(True, alpha=0.22)
    abs_ax.legend(loc="best")

    stats_text = _shim_histogram_stats_text(result, after_source=after_source, domain=domain)
    fig.text(0.5, 0.015, stats_text, ha="center", va="bottom", fontsize=9, color="#1f2937")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return _finish_figure(fig, plt, show=show, block=block)


def plot_shim_spherical_harmonics(
    result: CoilGenResult,
    *,
    source: str = "layout",
    domain: str = "dsv",
    max_order: int = 4,
    min_contribution_fraction: float = 0.01,
    max_terms: int = 18,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot significant spherical-harmonic terms before and after shimming."""

    plt = _import_pyplot()
    fits = shim_spherical_harmonic_fits(result, source=source, domain=domain, max_order=max_order)
    rows = shim_spherical_harmonic_comparison(
        result,
        source=source,
        domain=domain,
        max_order=max_order,
        min_contribution_fraction=0.0,
        max_terms=None,
    )
    if not rows:
        raise ValueError("No spherical harmonic coefficient rows were available.")
    rows = _engineer_sh_plot_rows(
        rows,
        max_order=max_order,
        min_contribution_fraction=min_contribution_fraction,
        max_terms=max_terms,
    )
    term_names = [_mr_sh_term_name(row) for row in rows]
    term_indices = [f"l={int(row['degree'])}, m={int(row['order']):+d}" for row in rows]
    y = np.arange(len(rows), dtype=float)
    before_coefficients = np.asarray([float(row["before_coefficient"]) for row in rows], dtype=float)
    after_coefficients = np.asarray([float(row["after_coefficient"]) for row in rows], dtype=float)
    before_contributions = np.asarray([float(row["before_contribution_rms"]) for row in rows], dtype=float)
    after_contributions = np.asarray([float(row["after_contribution_rms"]) for row in rows], dtype=float)
    reductions = before_contributions - after_contributions
    unit = str(rows[-1].get("unit", "") or fits["before"].unit)
    unit_suffix = f" [{unit}]" if unit else ""

    figure_title = "Shim Target Points: Spherical Harmonic Coefficients"
    fig = plt.figure(num=figure_title, figsize=(17.4, max(7.4, 0.48 * len(rows) + 3.2)))
    fig.clf()
    grid = fig.add_gridspec(
        2,
        5,
        height_ratios=[0.25, 1.0],
        width_ratios=[0.60, 2.78, 2.40, 2.30, 2.30],
        left=0.035,
        right=0.985,
        top=0.96,
        bottom=0.10,
        wspace=0.12,
        hspace=0.14,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    before = fits["before"]
    after = fits["after"]
    before_stats = shim_homogeneity_stats(result, source="before", domain=domain)
    after_stats = shim_homogeneity_stats(result, source=_after_shim_source(source), domain=domain)
    title_ax.text(
        0.0,
        0.72,
        f"MR shim-order decomposition of B0 inhomogeneity ({domain} domain, SH order <= {max_order})",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    title_ax.text(
        0.0,
        0.43,
        (
            f"Before std {_format_performance_table_value(before_stats.get('std'))}{unit_suffix}, "
            f"after std {_format_performance_table_value(after_stats.get('std'))}{unit_suffix}; "
            f"basis RMS before/after {_format_performance_table_value(before.field_rms)} / "
            f"{_format_performance_table_value(after.field_rms)}{unit_suffix}; "
            f"SH fit residuals {_format_performance_table_value(before.residual_rms)} / "
            f"{_format_performance_table_value(after.residual_rms)}{unit_suffix}; "
            f"n={before.sample_count}, center=({before.center[0]:.4g}, {before.center[1]:.4g}, {before.center[2]:.4g}) m, "
            f"radius={before.radius:.4g} m"
        ),
        ha="left",
        va="center",
        fontsize=9,
        color="#475569",
    )
    title_ax.text(
        0.0,
        0.12,
        "Color bands group terms by MR shim order. Positive reduction means the connected target-specific shim wire reduced that field component.",
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
        clip_on=False,
    )
    before_color = "#64748b"
    after_color = "#0f766e"
    try:
        from matplotlib.patches import Patch

        title_ax.legend(
            handles=[
                Patch(facecolor=before_color, alpha=0.72, label="before"),
                Patch(facecolor=after_color, alpha=0.78, label="after"),
                Patch(facecolor="#15803d", alpha=0.82, label="reduced"),
                Patch(facecolor="#b91c1c", alpha=0.82, label="worse"),
            ],
            loc="lower right",
            ncol=4,
            frameon=True,
            fontsize=9,
        )
    except Exception:
        pass

    order_ax = fig.add_subplot(grid[1, 0])
    label_ax = fig.add_subplot(grid[1, 1], sharey=order_ax)
    coefficient_ax = fig.add_subplot(grid[1, 2], sharey=order_ax)
    contribution_ax = fig.add_subplot(grid[1, 3], sharey=order_ax)
    reduction_ax = fig.add_subplot(grid[1, 4], sharey=order_ax)
    reduction_colors = np.where(reductions >= 0.0, "#15803d", "#b91c1c")
    data_axes = (label_ax, coefficient_ax, contribution_ax, reduction_ax)
    _shade_sh_degree_groups(data_axes, rows)
    _draw_sh_order_strip(order_ax, rows)
    _draw_sh_term_label_column(label_ax, term_names, term_indices, rows)

    coefficient_ax.barh(y - 0.18, before_coefficients, height=0.34, color=before_color, alpha=0.72, label="before")
    coefficient_ax.barh(y + 0.18, after_coefficients, height=0.34, color=after_color, alpha=0.78, label="after")
    coefficient_ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.65)
    _prepare_sh_data_axis(coefficient_ax, rows)
    coefficient_ax.set_xlabel(f"signed normalized SH coefficient{unit_suffix}")
    _set_axis_title(coefficient_ax, "Signed coefficient\nfield-pattern polarity", width=30)
    coefficient_ax.grid(True, axis="x", alpha=0.22)

    contribution_ax.barh(y - 0.18, before_contributions, height=0.34, color=before_color, alpha=0.72, label="before")
    contribution_ax.barh(y + 0.18, after_contributions, height=0.34, color=after_color, alpha=0.78, label="after")
    _prepare_sh_data_axis(contribution_ax, rows)
    contribution_ax.set_xlabel(f"RMS contribution{unit_suffix}")
    _set_axis_title(contribution_ax, "RMS contribution\non brain target cloud", width=30)
    contribution_ax.grid(True, axis="x", alpha=0.22)

    reduction_ax.barh(y, reductions, height=0.50, color=reduction_colors, alpha=0.82)
    reduction_ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.72)
    _prepare_sh_data_axis(reduction_ax, rows)
    reduction_ax.set_xlabel(f"before - after RMS contribution{unit_suffix}")
    _set_axis_title(reduction_ax, "Reduction by shim wire\npositive is better", width=30)
    reduction_ax.grid(True, axis="x", alpha=0.22)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_field_spherical_harmonics(
    result: CoilGenResult,
    *,
    domain: str = "dsv",
    max_order: int = 4,
    sources: Sequence[str] = ("target", "layout", "stream", "layout_error"),
    min_contribution_fraction: float = 0.01,
    max_terms: int = 22,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot general coil-field spherical-harmonic coefficients over the DSV."""

    plt = _import_pyplot()
    fits = coil_spherical_harmonic_fits(result, domain=domain, max_order=max_order, sources=sources)
    rows = coil_spherical_harmonic_coefficient_rows(
        result,
        domain=domain,
        max_order=max_order,
        sources=sources,
        min_contribution_fraction=0.0,
        max_terms=None,
    )
    if not rows:
        raise ValueError("No coil spherical-harmonic coefficient rows were available.")
    rows = _engineer_sh_plot_rows(
        rows,
        max_order=max_order,
        min_contribution_fraction=min_contribution_fraction,
        max_terms=max_terms,
    )
    source_keys = tuple(fits)
    if not source_keys:
        raise ValueError("No requested coil spherical-harmonic sources were available.")
    unit = str(next(iter(fits.values())).unit or "")
    unit_suffix = f" [{unit}]" if unit else ""
    y = np.arange(len(rows), dtype=float)
    term_names = [_mr_sh_term_name(row) for row in rows]
    term_indices = [f"l={int(row['degree'])}, m={int(row['order']):+d}" for row in rows]
    colors = _coil_sh_source_colors(source_keys)
    bar_height = min(0.72 / max(len(source_keys), 1), 0.24)
    offsets = (
        np.linspace(-0.36 + bar_height / 2.0, 0.36 - bar_height / 2.0, len(source_keys))
        if len(source_keys) > 1
        else np.asarray([0.0])
    )

    figure_title = "DSV Field Spherical Harmonic Coefficients"
    fig = plt.figure(num=figure_title, figsize=(16.8, max(7.0, 0.46 * len(rows) + 3.1)))
    fig.clf()
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[0.25, 1.0],
        width_ratios=[0.62, 2.85, 2.75, 2.55],
        left=0.035,
        right=0.985,
        top=0.96,
        bottom=0.10,
        wspace=0.13,
        hspace=0.15,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    reference = fits[source_keys[0]]
    title_ax.text(
        0.0,
        0.70,
        f"DSV spherical-harmonic decomposition of coil fields ({domain} domain, SH order <= {max_order})",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    title_ax.text(
        0.0,
        0.39,
        (
            f"n={reference.sample_count}, center=({reference.center[0]:.4g}, {reference.center[1]:.4g}, "
            f"{reference.center[2]:.4g}) m, radius={reference.radius:.4g} m; "
            + "; ".join(
                f"{_coil_sh_source_label(source)} fit RMS {_format_performance_table_value(fits[source].field_rms)}"
                for source in source_keys
            )
            + unit_suffix
        ),
        ha="left",
        va="center",
        fontsize=9,
        color="#475569",
    )
    title_ax.text(
        0.0,
        0.10,
        "Coefficients use the normalized real solid-SH basis on the selected DSV. RMS contribution weights each coefficient by the sampled basis RMS.",
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
        clip_on=False,
    )
    try:
        from matplotlib.patches import Patch

        title_ax.legend(
            handles=[
                Patch(facecolor=colors[source], alpha=0.78, label=_coil_sh_source_label(source))
                for source in source_keys
            ],
            loc="lower right",
            ncol=min(len(source_keys), 4),
            frameon=True,
            fontsize=8.6,
        )
    except Exception:
        pass

    order_ax = fig.add_subplot(grid[1, 0])
    label_ax = fig.add_subplot(grid[1, 1], sharey=order_ax)
    coefficient_ax = fig.add_subplot(grid[1, 2], sharey=order_ax)
    contribution_ax = fig.add_subplot(grid[1, 3], sharey=order_ax)
    data_axes = (label_ax, coefficient_ax, contribution_ax)
    _shade_sh_degree_groups(data_axes, rows)
    _draw_sh_order_strip(order_ax, rows)
    _draw_sh_term_label_column(label_ax, term_names, term_indices, rows)

    for offset, source in zip(offsets, source_keys):
        coefficients = np.asarray([float(row.get(f"{source}_coefficient", np.nan)) for row in rows], dtype=float)
        contributions = np.asarray([float(row.get(f"{source}_contribution_rms", np.nan)) for row in rows], dtype=float)
        coefficient_ax.barh(
            y + offset,
            coefficients,
            height=bar_height,
            color=colors[source],
            alpha=0.78,
            label=_coil_sh_source_label(source),
        )
        contribution_ax.barh(
            y + offset,
            contributions,
            height=bar_height,
            color=colors[source],
            alpha=0.78,
            label=_coil_sh_source_label(source),
        )

    coefficient_ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.65)
    _prepare_sh_data_axis(coefficient_ax, rows)
    coefficient_ax.set_xlabel(f"signed normalized SH coefficient{unit_suffix}")
    _set_axis_title(coefficient_ax, "Signed coefficient\nfield-pattern polarity", width=30)
    coefficient_ax.grid(True, axis="x", alpha=0.22)

    _prepare_sh_data_axis(contribution_ax, rows)
    contribution_ax.set_xlabel(f"RMS contribution{unit_suffix}")
    _set_axis_title(contribution_ax, "RMS contribution\non DSV target cloud", width=30)
    contribution_ax.grid(True, axis="x", alpha=0.22)
    return _finish_figure(fig, plt, show=show, block=block)


def _engineer_sh_plot_rows(
    rows: Sequence[dict[str, Any]],
    *,
    max_order: int,
    min_contribution_fraction: float,
    max_terms: int | None,
) -> list[dict[str, Any]]:
    """Keep low MR shim orders visible and group rows by physical order."""

    full_rows = list(rows)
    low_order_limit = min(int(max_order), 3)
    threshold = max(float(min_contribution_fraction), 0.0)
    low_order_rows = [row for row in full_rows if int(row["degree"]) <= low_order_limit]
    high_order_rows = [
        row
        for row in full_rows
        if int(row["degree"]) > low_order_limit and float(row.get("max_contribution_fraction", 0.0)) >= threshold
    ]
    high_order_rows.sort(key=_sh_row_strength, reverse=True)
    if max_terms is not None:
        remaining = max(0, int(max_terms) - len(low_order_rows))
        selected = [*low_order_rows, *high_order_rows[:remaining]]
        if not selected:
            selected = full_rows[: max(1, int(max_terms))]
    else:
        selected = [*low_order_rows, *high_order_rows]
    selected.sort(key=lambda row: (int(row["degree"]), -_sh_row_strength(row), int(row["order"])))
    return selected


def _sh_row_strength(row: dict[str, Any]) -> float:
    return float(
        row.get(
            "max_contribution_rms",
            max(float(row.get("before_contribution_rms", 0.0)), float(row.get("after_contribution_rms", 0.0))),
        )
    )


def _mr_sh_term_name(row: dict[str, Any]) -> str:
    degree = int(row["degree"])
    order = int(row["order"])
    names = {
        (0, 0): "B0 offset / static",
        (1, -1): "Gy linear gradient",
        (1, 0): "Gz linear gradient",
        (1, 1): "Gx linear gradient",
        (2, -2): "XY second-order",
        (2, -1): "YZ second-order",
        (2, 0): "Z2 second-order",
        (2, 1): "XZ second-order",
        (2, 2): "X2-Y2 second-order",
        (3, -3): "Y3-like third-order",
        (3, -2): "XYZ-like third-order",
        (3, -1): "YZ2-like third-order",
        (3, 0): "Z3 third-order",
        (3, 1): "XZ2-like third-order",
        (3, 2): "Z(X2-Y2)-like third-order",
        (3, 3): "X3-like third-order",
    }
    return names.get((degree, order), f"{_ordinal_sh_degree(degree)}-order m={order:+d}")


def _coil_sh_source_label(source: str) -> str:
    labels = {
        "target": "target",
        "layout": "connected layout",
        "stream": "stream function",
        "layout_error": "layout - target",
        "stream_error": "stream - target",
    }
    return labels.get(str(source), str(source).replace("_", " "))


def _coil_sh_source_colors(sources: Sequence[str]) -> dict[str, str]:
    palette = {
        "target": "#111827",
        "layout": "#d73a49",
        "stream": "#1f6feb",
        "layout_error": "#f59e0b",
        "stream_error": "#7c3aed",
    }
    fallback = ("#0f766e", "#db2777", "#0891b2", "#9333ea", "#ca8a04", "#475569")
    colors: dict[str, str] = {}
    for index, source in enumerate(sources):
        colors[str(source)] = palette.get(str(source), fallback[index % len(fallback)])
    return colors


def _shade_sh_degree_groups(axes: Sequence[Any], rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    group_start = 0
    while group_start < len(rows):
        degree = int(rows[group_start]["degree"])
        group_stop = group_start + 1
        while group_stop < len(rows) and int(rows[group_stop]["degree"]) == degree:
            group_stop += 1
        color = _sh_degree_color(degree)
        for ax in axes:
            ax.axhspan(group_start - 0.5, group_stop - 0.5, color=color, alpha=0.075, zorder=0)
        group_start = group_stop


def _draw_sh_order_strip(ax: Any, rows: Sequence[dict[str, Any]]) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("shim\norder", fontsize=9, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    group_start = 0
    while group_start < len(rows):
        degree = int(rows[group_start]["degree"])
        group_stop = group_start + 1
        while group_stop < len(rows) and int(rows[group_stop]["degree"]) == degree:
            group_stop += 1
        ax.axhspan(group_start - 0.48, group_stop - 0.52, color=_sh_degree_color(degree), alpha=0.65, linewidth=0)
        ax.text(
            0.5,
            (group_start + group_stop - 1) / 2.0,
            _sh_degree_group_label(degree),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=_sh_degree_text_color(degree),
            linespacing=0.95,
        )
        group_start = group_stop


def _draw_sh_term_label_column(
    ax: Any,
    term_names: Sequence[str],
    term_indices: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("term", fontsize=9, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row_ind, (name, index_text, row) in enumerate(zip(term_names, term_indices, rows)):
        degree = int(row["degree"])
        ax.text(
            0.02,
            row_ind,
            name,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold" if degree <= 1 else "normal",
            color="#111827",
        )
        ax.text(
            0.98,
            row_ind,
            index_text,
            ha="right",
            va="center",
            fontsize=8,
            color="#64748b",
        )


def _prepare_sh_data_axis(ax: Any, rows: Sequence[dict[str, Any]]) -> None:
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)


def _sh_degree_group_label(degree: int) -> str:
    labels = {
        0: "B0\nstatic",
        1: "1st\nlinear",
        2: "2nd\norder",
        3: "3rd\norder",
    }
    return labels.get(int(degree), f"{_ordinal_sh_degree(degree)}\nhigher")


def _sh_degree_color(degree: int) -> str:
    colors = {
        0: "#e2e8f0",
        1: "#dbeafe",
        2: "#dcfce7",
        3: "#fef3c7",
        4: "#fae8ff",
        5: "#fee2e2",
    }
    return colors.get(int(degree), "#f1f5f9")


def _sh_degree_text_color(degree: int) -> str:
    colors = {
        0: "#334155",
        1: "#1d4ed8",
        2: "#166534",
        3: "#92400e",
        4: "#86198f",
        5: "#991b1b",
    }
    return colors.get(int(degree), "#475569")


def _ordinal_sh_degree(degree: int) -> str:
    degree = int(degree)
    if 10 <= degree % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(degree % 10, "th")
    return f"{degree}{suffix}"


def plot_shim_homogeneity_mosaic(
    result: CoilGenResult,
    *,
    source: str = "after",
    axis: int = 2,
    slice_count: int = 12,
    domain: str = "dsv",
    grid_size: int = 96,
    min_points_per_slice: int = 24,
    slab_thickness_m: float | None = None,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot filled before/after shimming inhomogeneity slices."""

    plt = _import_pyplot()
    payload = _shim_mosaic_payload(
        result,
        source=source,
        axis=axis,
        slice_count=slice_count,
        domain=domain,
        grid_size=grid_size,
        min_points_per_slice=min_points_per_slice,
        slab_thickness_m=slab_thickness_m,
    )
    color_limit = _shim_reference_color_limit(result, source=source, domain=domain)
    if color_limit is not None:
        payload = _shim_payload_with_color_limit(payload, color_limit)

    columns = int(np.ceil(np.sqrt(len(payload["slices"]))))
    rows = int(np.ceil(len(payload["slices"]) / columns))
    figure_title = f"{payload['scalar_map'].label}: Filled Z-Slice Mosaic"
    fig = plt.figure(
        num=figure_title,
        figsize=(3.2 * columns + 0.7, 2.9 * rows + 0.6),
        constrained_layout=True,
    )
    outer = fig.add_gridspec(1, 1)
    _add_shim_mosaic_section(fig, outer[0], payload, plt=plt, title_fontsize=12)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_shim_before_after_mosaics(
    result: CoilGenResult,
    *,
    source: str = "layout",
    axis: int = 2,
    slice_count: int = 12,
    domain: str = "dsv",
    grid_size: int = 96,
    min_points_per_slice: int = 24,
    slab_thickness_m: float | None = None,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot before, after, and before-minus-after shim mosaics in one figure."""

    plt = _import_pyplot()
    before = _shim_mosaic_payload(
        result,
        source="before",
        axis=axis,
        slice_count=slice_count,
        domain=domain,
        grid_size=grid_size,
        min_points_per_slice=min_points_per_slice,
        slab_thickness_m=slab_thickness_m,
    )
    after = _shim_mosaic_payload(
        result,
        source=_after_shim_source(source),
        axis=axis,
        slice_count=slice_count,
        domain=domain,
        grid_size=grid_size,
        min_points_per_slice=min_points_per_slice,
        slab_thickness_m=slab_thickness_m,
    )
    improvement = _shim_mosaic_payload(
        result,
        source=_improvement_shim_source(source),
        axis=axis,
        slice_count=slice_count,
        domain=domain,
        grid_size=grid_size,
        min_points_per_slice=min_points_per_slice,
        slab_thickness_m=slab_thickness_m,
    )
    field_limit = max(float(before["limit"]), float(after["limit"]))
    before = _shim_payload_with_color_limit(before, field_limit)
    after = _shim_payload_with_color_limit(after, field_limit)
    row_count = max(len(before["slices"]), len(after["slices"]), len(improvement["slices"]))
    figure_title = "B0 Inhomogeneity: Before, After, and Improvement"
    fig = plt.figure(
        num=figure_title,
        figsize=(13.8, max(7.5, 2.25 * row_count + 2.4)),
        constrained_layout=False,
    )
    _add_shim_before_after_slice_rows(fig, before, after, improvement, plt=plt)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_field_cross_section_slices(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot central x/y/z cross-section slices for field maps."""

    return _plot_cross_section_grid(
        result,
        list(field_scalar_maps(result).values()),
        figure_title="Target Points: Bz Field Central Cross Sections",
        show=show,
        block=block,
    )


def plot_gradient_cross_section_slices(
    result: CoilGenResult,
    *,
    component: str = "magnitude",
    neighbor_count: int = 12,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """Plot central x/y/z cross-section slices for local gradient maps."""

    return _plot_cross_section_grid(
        result,
        [
            *gradient_component_scalar_maps(result, component=component, neighbor_count=neighbor_count).values(),
            *gradient_relative_error_scalar_maps(result, component=component, neighbor_count=neighbor_count).values(),
        ],
        figure_title=f"DSV Target Points: Bz Gradient Central Cross Sections - {_gradient_component_title(component)}",
        show=show,
        block=block,
    )


def plot_error_cross_section_slices(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot central x/y/z cross-section slices for absolute and relative Bz error maps."""

    return _plot_cross_section_grid(
        result,
        [*field_error_scalar_maps(result).values(), *field_relative_error_scalar_maps(result).values()],
        figure_title="DSV Target Points: Bz Error Central Cross Sections",
        show=show,
        block=block,
    )


def plot_gradient_field_slice_mosaic(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot pixel-center MR-gradient field-component slice mosaics."""

    evaluated = gradient_slice_map_result(result)
    return _plot_gradient_slice_mosaic(
        result,
        evaluated.available_components(evaluated.spec.field_components),
        figure_title="MR Gradient Pixel Slices: Field Components",
        show=show,
        block=block,
    )


def plot_gradient_derivative_slice_mosaic(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot pixel-center finite-difference Bz-gradient slice mosaics."""

    evaluated = gradient_slice_map_result(result)
    return _plot_gradient_slice_mosaic(
        result,
        evaluated.available_components(evaluated.spec.gradient_components),
        figure_title="MR Gradient Pixel Slices: Bz Gradients",
        show=show,
        block=block,
    )


def plot_concomitant_slice_mosaic(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot pixel-center concomitant-field slice mosaics."""

    evaluated = gradient_slice_map_result(result)
    return _plot_gradient_slice_mosaic(
        result,
        evaluated.available_components(evaluated.spec.concomitant_components),
        figure_title="MR Gradient Pixel Slices: Concomitant Field",
        show=show,
        block=block,
    )


def plot_error_metrics_matplotlib(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot field-error metrics as a horizontal bar chart."""

    if not result.error_vals:
        raise ValueError("Result has no error metrics.")
    plt = _import_pyplot()
    items = list(result.error_vals.items())
    sparsified_layout = result_uses_sparsified_contour_layout(result)
    if sparsified_layout:
        # With ideal sparse-contour realization, ``layout`` and
        # ``unconnected_contours`` are the same final field. Keep one copy.
        items = [(key, value) for key, value in items if "unconnected_contours" not in key]
    labels = [
        key.replace("layout", "final_sparsified_weighted_contours").replace("_", " ")
        if sparsified_layout
        else key.replace("_", " ")
        for key, _value in items
    ]
    values = [float(value) for _key, value in items]
    fig = plt.figure(num="Coil Design DSV Field-Error Metrics", figsize=(9, max(4, 0.35 * len(labels))))
    fig.clf()
    ax = fig.subplots()
    y = np.arange(len(labels))
    ax.barh(y, values, color="#1f77b4", alpha=0.85)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel(format_label_with_unit("relative error", RELATIVE_ERROR_UNIT))
    _set_axis_title(
        ax,
        (
            "DSV relative Bz error metrics for the final sparsified weighted-contour layout"
            if sparsified_layout
            else "DSV relative Bz error metrics for connected wire and unconnected contours"
        ),
        width=72,
    )
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return _finish_figure(fig, plt, show=show, block=block)


def plot_performance_table_matplotlib(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot the most relevant coil-design performance parameters as a table."""

    rows = _performance_table_rows(result)
    if not rows:
        raise ValueError("Result has no finite performance parameters.")
    plt = _import_pyplot()
    fig = plt.figure(
        num="Coil Design Performance Summary Table",
        figsize=(11.5, max(5.2, 0.34 * len(rows) + 1.8)),
    )
    fig.clf()
    ax = fig.subplots()
    ax.axis("off")
    _set_axis_title(ax, "Coil Design Performance Summary", width=80)
    table = ax.table(
        cellText=rows,
        colLabels=("Group", "Quantity", "Value", "Unit"),
        cellLoc="left",
        colLoc="left",
        colWidths=(0.19, 0.43, 0.22, 0.16),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_ind, col_ind), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.6)
        if row_ind == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#111827")
        elif col_ind == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold", color="#334155")
        else:
            cell.set_facecolor("#ffffff")
    fig.tight_layout(pad=1.2)
    return _finish_figure(fig, plt, show=show, block=block)


def plot_l_curve_regularization_matplotlib(result: CoilGenResult, *, show: bool = True, block: bool = False) -> Figure:
    """Plot the L-curve regularization scan and the selected knee point."""

    scan = _regularization_scan_arrays(result)
    plt = _import_pyplot()
    figure_title = (
        "Continuous Stream-Function Regularization Reference (Pre-Sparsification)"
        if result_uses_sparsified_contour_layout(result)
        else "Stream-Function Regularization L-Curve Scan"
    )
    fig = plt.figure(num=figure_title, figsize=(12, 4.8))
    fig.clf()
    curve_ax, scan_ax = fig.subplots(1, 2)

    valid = scan["valid"]
    selected_index = scan["selected_index"]
    diagnostics = result.regularization_diagnostics or {}
    selected_word = "fallback" if bool(diagnostics.get("fallback", False)) else "selected"
    selected_label = f"{selected_word} factor {scan['factors'][selected_index]:.4g}"

    curve_ax.loglog(
        scan["residuals"][valid],
        scan["regularization_norms"][valid],
        color="#1f6feb",
        linewidth=1.8,
        marker="o",
        markersize=4,
    )
    curve_ax.scatter(
        scan["residuals"][selected_index],
        scan["regularization_norms"][selected_index],
        s=90,
        color="#d73a49",
        edgecolor="#111827",
        zorder=5,
        label=selected_label,
    )
    curve_ax.set_xlabel("field residual norm ||A x - b||2 [T/A]")
    curve_ax.set_ylabel("resistance seminorm ||R x||2 [Ohm A]")
    _set_axis_title(curve_ax, "L-curve: field residual vs resistance seminorm", width=44)
    curve_ax.grid(True, which="both", alpha=0.25)
    curve_ax.legend(loc="best")

    scan_ax.semilogx(
        scan["factors"][valid],
        normalized_log_trace(scan["residuals"][valid]),
        color="#1f6feb",
        linewidth=1.8,
        label="log residual",
    )
    scan_ax.semilogx(
        scan["factors"][valid],
        normalized_log_trace(scan["regularization_norms"][valid]),
        color="#f97316",
        linewidth=1.8,
        label="log resistance seminorm",
    )
    finite_curvature = np.isfinite(scan["curvature"]) & (scan["curvature"] >= 0.0)
    if np.any(finite_curvature):
        scan_ax.semilogx(
            scan["factors"][finite_curvature],
            normalized_linear_trace(scan["curvature"][finite_curvature]),
            color="#16a34a",
            linewidth=1.8,
            label="normalized curvature",
        )
    scan_ax.axvline(
        scan["factors"][selected_index], color="#d73a49", linewidth=1.4, linestyle="--", label=selected_label
    )
    scan_ax.set_xlabel("Tikhonov regularization factor [-]")
    scan_ax.set_ylabel("normalized diagnostic value [-]")
    _set_axis_title(scan_ax, "1D scan over regularization factor", width=44)
    scan_ax.grid(True, which="both", alpha=0.25)
    scan_ax.legend(loc="best")

    fig.tight_layout(pad=1.4)
    return _finish_figure(fig, plt, show=show, block=block)


def _plot_scalar_pointcloud_grid(
    result: CoilGenResult,
    maps: Sequence[Any],
    *,
    figure_title: str,
    show: bool,
    block: bool,
) -> Figure:
    plt = _import_pyplot()
    coords = result.target_field.coords
    rows, cols = _subplot_grid(len(maps))
    fig = plt.figure(num=figure_title, figsize=(5.2 * cols, 4.4 * rows))
    fig.clf()
    for index, scalar_map in enumerate(maps, start=1):
        ax = fig.add_subplot(rows, cols, index, projection="3d")
        values = np.asarray(scalar_map.values, dtype=float).reshape(-1)
        finite = _finite_point_scalar_mask(coords, values)
        display_indices = pointcloud_plot_indices(
            result.input_data,
            len(coords),
            eligible_mask=finite,
            seed_offset=31,
        )
        plotted = np.zeros(len(coords), dtype=bool)
        plotted[display_indices] = True
        if np.any(plotted):
            plot_coords = coords[plotted]
            scatter = ax.scatter(
                plot_coords[:, 0],
                plot_coords[:, 1],
                plot_coords[:, 2],
                c=values[plotted],
                cmap="coolwarm",
                norm=_linear_color_norm(values[finite]),
                s=24,
                alpha=FIELD_POINT_ALPHA,
                depthshade=False,
            )
            _add_artist_colorbar(fig, ax, scatter, scalar_map.label_with_unit, shrink=0.65, pad=0.08)
        else:
            plot_coords = coords
            ax.text2D(0.08, 0.5, "No finite samples in this domain", transform=ax.transAxes, color="#64748b")
        _set_axis_title(ax, _scalar_subplot_title(scalar_map))
        _finish_3d_axis(ax, coords[finite] if np.any(finite) else plot_coords, view=FACE_ISOMETRIC_VIEW)
    fig.tight_layout()
    return _finish_figure(fig, plt, show=show, block=block)


def _plot_cross_section_grid(
    result: CoilGenResult,
    maps: Sequence[Any],
    *,
    figure_title: str,
    show: bool,
    block: bool,
) -> Figure:
    plt = _import_pyplot()
    coords = result.target_field.coords
    slice_axes = (0, 1, 2)
    rows = len(maps)
    cols = len(slice_axes)
    fig = plt.figure(num=figure_title, figsize=(4.8 * cols, max(3.2, 2.8 * rows)))
    fig.clf()
    for row_ind, scalar_map in enumerate(maps):
        values = np.asarray(scalar_map.values, dtype=float)
        for col_ind, axis in enumerate(slice_axes):
            selection = central_slice_selection(coords, axis)
            ax = fig.add_subplot(rows, cols, row_ind * cols + col_ind + 1)
            plot_x, plot_y = selection.plot_axes
            mask = selection.mask & _finite_point_scalar_mask(coords, values)
            if np.any(mask):
                scatter = ax.scatter(coords[mask, plot_x], coords[mask, plot_y], c=values[mask], cmap="coolwarm", s=28)
                _add_artist_colorbar(fig, ax, scatter, scalar_map.label_with_unit, shrink=0.78, pad=0.03)
            else:
                ax.text(
                    0.5, 0.5, "No finite samples", transform=ax.transAxes, ha="center", va="center", color="#64748b"
                )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(_coordinate_axis_label(plot_x))
            ax.set_ylabel(_coordinate_axis_label(plot_y))
            _set_axis_title(
                ax,
                f"{_wrap_plot_title(_scalar_subplot_title(scalar_map), width=32)}\nCentral {AXIS_NAMES[axis]} slice at {selection.value:.4g} m",
                width=34,
            )
            ax.grid(True, alpha=0.18)
    fig.tight_layout()
    return _finish_figure(fig, plt, show=show, block=block)


def _plot_gradient_slice_mosaic(
    result: CoilGenResult,
    components: Sequence[str],
    *,
    figure_title: str,
    show: bool,
    block: bool,
) -> Figure:
    evaluated = gradient_slice_map_result(result)
    components = tuple(component for component in components if component in evaluated.component_units)
    if not components:
        raise ValueError("No requested gradient slice-map components are available.")
    if not evaluated.slices:
        raise ValueError("Gradient slice maps have no evaluated slices.")

    plt = _import_pyplot()
    colors = _import_matplotlib_colors()
    rows = len(components)
    cols = len(evaluated.slices)
    layout = _gradient_slice_mosaic_layout(rows, cols)
    fig = plt.figure(num=figure_title, figsize=layout["figsize"])
    fig.clf()
    for row, component in enumerate(components):
        arrays = [np.asarray(slice_data.maps.get(component), dtype=float) for slice_data in evaluated.slices]
        norm = _gradient_slice_component_norm(colors, arrays, component)
        cmap = _gradient_slice_component_cmap(plt, component)
        for col, slice_data in enumerate(evaluated.slices):
            ax = fig.add_axes(_gradient_slice_axes_rect(layout, row, col, rows))
            cax = fig.add_axes(_gradient_slice_colorbar_rect(layout, row, col, rows))
            values = np.ma.masked_invalid(np.asarray(slice_data.maps.get(component), dtype=float))
            ax.imshow(
                values,
                origin="lower",
                extent=_gradient_slice_pixel_extent(slice_data),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="auto",
            )
            _draw_gradient_slice_dsv_outline(ax, slice_data)
            _style_gradient_slice_tile_axis(ax, slice_data)
            _draw_gradient_slice_tile_label(
                ax,
                _gradient_slice_tile_label(component, evaluated.axis_name, slice_data.value),
            )
            scalar_mappable = _import_scalar_mappable()(norm=norm, cmap=cmap)
            scalar_mappable.set_array([])
            colorbar = fig.colorbar(scalar_mappable, cax=cax)
            _style_gradient_slice_colorbar(
                colorbar,
                norm.vmin,
                norm.vmax,
                unit_label=evaluated.component_units.get(component, ""),
            )
            try:
                colorbar.outline.set_linewidth(0.35)
            except Exception:
                pass
    fig.suptitle(
        (
            f"{figure_title}; source={evaluated.spec.source}, current={evaluated.spec.current}, "
            f"domain={evaluated.spec.domain}, h={evaluated.gradient_step_m:.3g} m"
        ),
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=9.5,
    )
    return _finish_figure(fig, plt, show=show, block=block)


def _gradient_slice_mosaic_layout(rows: int, cols: int) -> dict[str, float | tuple[float, float]]:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    tile = GRADIENT_SLICE_TILE_SIZE_IN
    cbar = GRADIENT_SLICE_COLORBAR_WIDTH_IN
    cbar_gap = GRADIENT_SLICE_COLORBAR_GAP_IN
    cbar_tick_gutter = GRADIENT_SLICE_COLORBAR_TICK_GUTTER_IN
    col_gap = GRADIENT_SLICE_COLUMN_GAP_IN
    row_gap = GRADIENT_SLICE_ROW_GAP_IN
    left = GRADIENT_SLICE_LEFT_MARGIN_IN
    right = GRADIENT_SLICE_RIGHT_MARGIN_IN
    bottom = GRADIENT_SLICE_BOTTOM_MARGIN_IN
    top = GRADIENT_SLICE_TOP_MARGIN_IN
    figure_width = left + right + cols * (tile + cbar_gap + cbar + cbar_tick_gutter) + (cols - 1) * col_gap
    figure_height = bottom + top + rows * tile + (rows - 1) * row_gap
    return {
        "figsize": (figure_width, figure_height),
        "figure_width": figure_width,
        "figure_height": figure_height,
        "tile": tile,
        "colorbar": cbar,
        "colorbar_gap": cbar_gap,
        "colorbar_tick_gutter": cbar_tick_gutter,
        "column_gap": col_gap,
        "row_gap": row_gap,
        "left": left,
        "bottom": bottom,
    }


def _gradient_slice_axes_rect(
    layout: Mapping[str, float | tuple[float, float]], row: int, col: int, rows: int
) -> list[float]:
    x, y = _gradient_slice_tile_origin(layout, row, col, rows)
    return _rect_in_figure_fraction(layout, x, y, float(layout["tile"]), float(layout["tile"]))


def _gradient_slice_colorbar_rect(
    layout: Mapping[str, float | tuple[float, float]], row: int, col: int, rows: int
) -> list[float]:
    x, y = _gradient_slice_tile_origin(layout, row, col, rows)
    cbar_x = x + float(layout["tile"]) + float(layout["colorbar_gap"])
    return _rect_in_figure_fraction(layout, cbar_x, y, float(layout["colorbar"]), float(layout["tile"]))


def _gradient_slice_tile_origin(
    layout: Mapping[str, float | tuple[float, float]], row: int, col: int, rows: int
) -> tuple[float, float]:
    tile = float(layout["tile"])
    column_pitch = (
        tile
        + float(layout["colorbar_gap"])
        + float(layout["colorbar"])
        + float(layout["colorbar_tick_gutter"])
        + float(layout["column_gap"])
    )
    x = float(layout["left"]) + int(col) * column_pitch
    y = float(layout["bottom"]) + (int(rows) - 1 - int(row)) * (tile + float(layout["row_gap"]))
    return x, y


def _rect_in_figure_fraction(
    layout: Mapping[str, float | tuple[float, float]], x: float, y: float, width: float, height: float
) -> list[float]:
    return [
        float(x) / float(layout["figure_width"]),
        float(y) / float(layout["figure_height"]),
        float(width) / float(layout["figure_width"]),
        float(height) / float(layout["figure_height"]),
    ]


def _gradient_slice_pixel_extent(slice_data: Any) -> tuple[float, float, float, float]:
    x = np.asarray(slice_data.x, dtype=float).reshape(-1)
    y = np.asarray(slice_data.y, dtype=float).reshape(-1)
    if len(x) == 0 or len(y) == 0:
        return (-0.5, 0.5, -0.5, 0.5)
    dx = _regular_grid_spacing(x)
    dy = _regular_grid_spacing(y)
    return (float(x[0] - dx / 2.0), float(x[-1] + dx / 2.0), float(y[0] - dy / 2.0), float(y[-1] + dy / 2.0))


def _draw_gradient_slice_dsv_outline(ax: Any, slice_data: Any) -> None:
    radius = getattr(slice_data, "dsv_outline_radius_m", None)
    center = getattr(slice_data, "dsv_outline_center_m", None)
    if radius is None or center is None:
        return
    try:
        radius_value = float(radius)
        cx, cy = (float(center[0]), float(center[1]))
    except (TypeError, ValueError, IndexError):
        return
    if not np.isfinite(radius_value) or radius_value <= 0.0:
        return
    theta = np.linspace(0.0, 2.0 * np.pi, 241)
    x = cx + radius_value * np.cos(theta)
    y = cy + radius_value * np.sin(theta)
    ax.plot(x, y, color="#0f172a", linewidth=0.85, alpha=0.95, zorder=7)
    ax.plot(x, y, color="#ffffff", linewidth=0.35, alpha=0.95, zorder=8)


def _regular_grid_spacing(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if len(finite) >= 2:
        return abs(float(np.nanmedian(np.diff(finite))))
    return 1.0


def _gradient_slice_component_norm(colors: Any, arrays: Sequence[np.ndarray], component: str) -> Any:
    finite_arrays = [array[np.isfinite(array)] for array in arrays if np.any(np.isfinite(array))]
    values = np.concatenate(finite_arrays) if finite_arrays else np.zeros(0)
    if len(values) == 0:
        return colors.Normalize(vmin=-1.0, vmax=1.0)
    if _gradient_slice_component_nonnegative(component):
        upper = max(float(np.nanpercentile(values, 99.0)), float(np.nanmax(values)), 1.0e-30)
        return colors.Normalize(vmin=0.0, vmax=upper)
    limit = max(float(np.nanpercentile(np.abs(values), 99.0)), float(np.nanmax(np.abs(values))), 1.0e-30)
    return colors.Normalize(vmin=-limit, vmax=limit)


def _gradient_slice_component_cmap(plt: Any, component: str) -> Any:
    cmap = plt.get_cmap("viridis" if _gradient_slice_component_nonnegative(component) else "RdBu_r")
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        cmap.set_bad("#cbd5e1")
    except Exception:
        pass
    return cmap


def _style_gradient_slice_colorbar(colorbar: Any, vmin: float, vmax: float, *, unit_label: str = "") -> None:
    ticks = _colorbar_tick_values(vmin, vmax, count=3)
    if len(ticks):
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([_format_gradient_slice_colorbar_tick(value) for value in ticks])
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.ax.tick_params(labelsize=5.7, length=1.8, pad=1.0)
    colorbar.set_label(_gradient_slice_colorbar_unit_label(unit_label), fontsize=5.7, labelpad=2.0)


def _gradient_slice_colorbar_unit_label(unit_label: str) -> str:
    unit = str(unit_label or "").strip()
    return f"[{unit}]" if unit else ""


def _format_gradient_slice_colorbar_tick(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return ""
    if abs(value) < 1.0e-15:
        return "0"
    magnitude = abs(value)
    if magnitude < 1.0e-2 or magnitude >= 1.0e3:
        return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if magnitude < 10.0:
        return f"{value:.2g}"
    return f"{value:.3g}"


def _gradient_slice_component_nonnegative(component: str) -> bool:
    key = str(component).strip().lower()
    return key.endswith("_mag") or "percent" in key


def _gradient_slice_tile_label(component: str, axis_name: str, value: float) -> str:
    component_label = _gradient_slice_tile_component_label(component)
    return f"{component_label}, {axis_name}={_gradient_slice_tile_value(value)} m"


def _gradient_slice_tile_component_label(component: str) -> str:
    labels = {
        "dBz_dx": "dBz/dx",
        "dBz_dy": "dBz/dy",
        "dBz_dz": "dBz/dz",
        "gradBz_mag": "|grad Bz|",
        "linearity_residual": "linearity residual",
        "concomitant_Bz": "concomitant Bz",
        "concomitant_grad_mag": "concomitant |grad|",
    }
    return labels.get(str(component), str(component))


def _gradient_slice_tile_value(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 5.0e-13:
        value = 0.0
    return f"{value:.4g}"


def _style_gradient_slice_tile_axis(ax: Any, slice_data: Any) -> None:
    ax.set_aspect("auto")
    xticks = _gradient_slice_spatial_ticks(getattr(slice_data, "x", ()))
    yticks = _gradient_slice_spatial_ticks(getattr(slice_data, "y", ()))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([_format_gradient_slice_axis_tick(value) for value in xticks])
    ax.set_yticklabels([_format_gradient_slice_axis_tick(value) for value in yticks])
    ax.tick_params(
        axis="both",
        which="major",
        colors="#94a3b8",
        labelcolor="#64748b",
        labelsize=5.1,
        length=1.8,
        width=0.35,
        pad=1.0,
        direction="out",
        top=False,
        right=False,
    )
    ax.grid(False)
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.38)
        spine.set_edgecolor("#cbd5e1")


def _gradient_slice_spatial_ticks(values: Sequence[float], *, count: int = 3) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.unique(arr[np.isfinite(arr)])
    if len(finite) == 0:
        return []
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper <= lower:
        return [lower]
    if lower < 0.0 < upper:
        ticks = [lower, 0.0, upper]
    else:
        ticks = [float(value) for value in np.linspace(lower, upper, max(2, int(count)))]
    return _unique_tick_values(ticks)


def _unique_tick_values(values: Sequence[float]) -> list[float]:
    ticks: list[float] = []
    for value in values:
        number = float(value)
        if abs(number) < 5.0e-13:
            number = 0.0
        if not any(np.isclose(number, existing, rtol=0.0, atol=1.0e-12) for existing in ticks):
            ticks.append(number)
    return ticks


def _format_gradient_slice_axis_tick(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return ""
    if abs(value) < 5.0e-13:
        return "0"
    magnitude = abs(value)
    if magnitude < 1.0e-3 or magnitude >= 10.0:
        return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:.3g}"


def _draw_gradient_slice_tile_label(ax: Any, label: str) -> None:
    ax.text(
        0.022,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#111827",
        bbox={"facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.72, "boxstyle": "round,pad=0.16"},
        clip_on=True,
        zorder=10,
    )


def _finite_point_scalar_mask(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(coords):
        return np.zeros(len(coords), dtype=bool)
    return np.isfinite(values) & np.all(np.isfinite(coords), axis=1)


def _plot_face_scalar_3d(
    part: CoilPart,
    values: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    display_percentile: float = 100.0,
    nonnegative: bool = False,
    show: bool,
    block: bool,
) -> Figure:
    plt = _import_pyplot()
    colors = _import_matplotlib_colors()
    collections = _import_mplot3d_collections()
    values = np.asarray(values, dtype=float)
    if len(values) != len(part.mesh.faces):
        raise ValueError("Face values must match mesh face count.")
    fig, ax = _new_3d_figure(plt, title)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Face values contain no finite values.")
    norm = _face_scalar_display_norm(colors, finite, percentile=display_percentile, nonnegative=nonnegative)
    cmap = plt.get_cmap("coolwarm")
    polygons = part.mesh.vertices[part.mesh.faces]
    collection = collections.Poly3DCollection(
        polygons,
        linewidths=SCALAR_MESH_EDGE_LINEWIDTH,
        edgecolors=SCALAR_MESH_EDGE_COLOR,
        alpha=0.94,
    )
    collection.set_facecolor(cmap(norm(values)))
    ax.add_collection3d(collection)
    _add_colorbar(fig, ax, cmap, norm, colorbar_label)
    _finish_3d_axis(ax, part.mesh.vertices, view=FACE_ISOMETRIC_VIEW)
    return _finish_figure(fig, plt, show=show, block=block)


def _plot_face_scalar_all_parts_3d(
    parts: Sequence[CoilPart],
    values_by_part: Sequence[np.ndarray],
    *,
    title: str,
    colorbar_label: str,
    display_percentile: float = 100.0,
    nonnegative: bool = False,
    show: bool,
    block: bool,
) -> Figure:
    parts = list(parts)
    if len(parts) != len(values_by_part):
        raise ValueError("parts and values_by_part must have the same length.")
    plt = _import_pyplot()
    colors = _import_matplotlib_colors()
    collections = _import_mplot3d_collections()
    checked_values: list[np.ndarray] = []
    for part, values in zip(parts, values_by_part):
        values = np.asarray(values, dtype=float)
        if len(values) != len(part.mesh.faces):
            raise ValueError("Face values must match mesh face count.")
        checked_values.append(values)
    finite_parts = [values[np.isfinite(values)] for values in checked_values if np.any(np.isfinite(values))]
    if not finite_parts:
        raise ValueError("Face values contain no finite values.")
    finite = np.concatenate(finite_parts)
    norm = _face_scalar_display_norm(colors, finite, percentile=display_percentile, nonnegative=nonnegative)
    cmap = plt.get_cmap("coolwarm")
    fig, ax = _new_3d_figure(plt, title)
    all_points: list[np.ndarray] = []
    for part, values in zip(parts, checked_values):
        polygons = part.mesh.vertices[part.mesh.faces]
        collection = collections.Poly3DCollection(
            polygons,
            linewidths=SCALAR_MESH_EDGE_LINEWIDTH,
            edgecolors=SCALAR_MESH_EDGE_COLOR,
            alpha=0.94,
        )
        collection.set_facecolor(cmap(norm(values)))
        ax.add_collection3d(collection)
        all_points.append(part.mesh.vertices)
    _add_colorbar(fig, ax, cmap, norm, colorbar_label)
    _finish_3d_axis(ax, _stack_points(all_points), view=FACE_ISOMETRIC_VIEW)
    return _finish_figure(fig, plt, show=show, block=block)


def _face_scalar_display_norm(colors: Any, values: np.ndarray, *, percentile: float, nonnegative: bool) -> Any:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("Face values contain no finite values.")

    percentile = float(percentile)
    if percentile < 100.0:
        percentile = min(max(percentile, 1.0), 100.0)
        if nonnegative:
            low = 0.0 if float(np.nanmin(finite)) >= 0.0 else float(np.nanpercentile(finite, 100.0 - percentile))
            high = float(np.nanpercentile(finite, percentile))
        else:
            tail = 0.5 * (100.0 - percentile)
            low, high = (float(value) for value in np.nanpercentile(finite, [tail, 100.0 - tail]))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))
    else:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Face values contain no finite display range.")
    if np.isclose(low, high, rtol=1e-12, atol=1e-15):
        delta = max(abs(low), 1.0) * 0.05
        low -= delta
        high += delta
    return colors.Normalize(vmin=float(low), vmax=float(high), clip=True)


def _new_3d_figure(plt: Any, title: str, *, pyplot_managed: bool = True) -> tuple[Figure, Any]:
    if pyplot_managed:
        fig = plt.figure(num=title, figsize=(9, 7))
    else:
        # Lazy plot-book factories must not reuse a named pyplot/Qt manager
        # left in a long-lived Spyder kernel. Its window or toolbar QActions
        # may already have been deleted when an earlier book was replaced.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure as MatplotlibFigure

        fig = MatplotlibFigure(figsize=(9, 7))
        FigureCanvasAgg(fig)
        fig.set_label(title)
    fig.clf()
    ax = fig.add_subplot(111, projection="3d")
    _set_axis_title(ax, title, width=58)
    return fig, ax


def _draw_reference_meshes_3d(ax: Any, parts: Sequence[CoilPart]) -> None:
    for part in parts:
        _draw_reference_mesh_3d(ax, part)


def _draw_mesh_3d(ax: Any, part: CoilPart) -> None:
    vertices = part.mesh.vertices
    try:
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=part.mesh.faces,
            color=MESH_FACE_COLOR,
            edgecolor=MESH_EDGE_COLOR,
            linewidth=MESH_EDGE_LINEWIDTH,
            alpha=MESH_FACE_ALPHA,
            shade=False,
        )
    except Exception:
        _draw_mesh_edges_3d(ax, part)


def _draw_reference_mesh_3d(ax: Any, part: CoilPart) -> None:
    _draw_mesh_edges_3d(
        ax,
        part,
        color=REFERENCE_MESH_EDGE_COLOR,
        linewidth=REFERENCE_MESH_EDGE_LINEWIDTH,
        alpha=REFERENCE_MESH_EDGE_ALPHA,
    )


def _draw_mesh_edges_3d(
    ax: Any,
    part: CoilPart,
    *,
    color: Any = MESH_EDGE_COLOR,
    linewidth: float = MESH_EDGE_LINEWIDTH,
    alpha: float = 0.78,
) -> None:
    edges: set[tuple[int, int]] = set()
    for face in part.mesh.faces:
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            edges.add((min(a, b), max(a, b)))
    if not edges:
        return
    segments = np.asarray([part.mesh.vertices[[a, b]] for a, b in edges], dtype=float)
    try:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        collection = Line3DCollection(segments, colors=color, linewidths=linewidth, alpha=alpha)
        ax.add_collection3d(collection)
        return
    except Exception:
        pass
    for a, b in edges:
        points = part.mesh.vertices[[a, b]]
        _plot_path_3d(ax, points, color=color, linewidth=linewidth, alpha=alpha)


def _new_line3d_collection(segments: np.ndarray, **kwargs: Any) -> Any:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    return Line3DCollection(np.asarray(segments, dtype=float), **kwargs)


def _plot_geometry_scale(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    finite = points[np.all(np.isfinite(points), axis=1)] if points.ndim == 2 else np.zeros((0, 3), dtype=float)
    if len(finite) == 0:
        return 1.0
    return max(float(np.max(np.ptp(finite, axis=0))), 1e-12)


def _adaptive_arrow_count(
    path_length: float,
    geometry_scale: float,
    *,
    minimum: int,
    maximum: int,
    spacing_scale: float,
) -> int:
    spacing = max(float(spacing_scale) * max(float(geometry_scale), 1e-12), 1e-12)
    count = int(np.ceil(max(float(path_length), 0.0) / spacing))
    return int(np.clip(count, int(minimum), int(maximum)))


def _path_samples_at_fractions(
    points: np.ndarray,
    fractions: Iterable[float],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Sample positions and unit tangents by intrinsic polyline arclength."""

    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return []
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-15
    if not np.any(valid):
        return []
    starts = points[:-1][valid]
    vectors = vectors[valid]
    lengths = lengths[valid]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for raw_fraction in fractions:
        fraction = float(np.clip(raw_fraction, 0.0, 1.0))
        target = fraction * total
        segment_index = min(int(np.searchsorted(cumulative[1:], target, side="right")), len(lengths) - 1)
        local = (target - cumulative[segment_index]) / lengths[segment_index]
        position = starts[segment_index] + local * vectors[segment_index]
        tangent = vectors[segment_index] / lengths[segment_index]
        samples.append((position, tangent, fraction))
    return samples


def _contour_chirality_color(chirality: int) -> str:
    if chirality > 0:
        return CURRENT_DIRECTION_POSITIVE_COLOR
    if chirality < 0:
        return CURRENT_DIRECTION_NEGATIVE_COLOR
    return CURRENT_DIRECTION_UNDEFINED_COLOR


def _contour_chirality_label(chirality: int) -> str:
    if chirality > 0:
        return "positive chirality (chi=+1)"
    if chirality < 0:
        return "negative chirality (chi=-1)"
    return "undefined chirality"


def _wire_path_style_keys(part: CoilPart) -> np.ndarray:
    if part.wire_path is None:
        return np.zeros(0, dtype=object)
    segment_count = max(len(part.wire_path.v) - 1, 0)
    owners: list[WireSegmentOwner | None]
    if len(part.wire_path.segment_owners) == segment_count:
        owners = list(part.wire_path.segment_owners)
    else:
        owners = [None] * segment_count
    geometry = (
        list(part.wire_path.segment_geometry)
        if len(part.wire_path.segment_geometry) == segment_count
        else [WirePathGeometry.SURFACE] * segment_count
    )
    styles: list[str] = []
    for owner, segment_geometry in zip(owners, geometry):
        segment_geometry = WirePathGeometry(segment_geometry)
        if segment_geometry == WirePathGeometry.OFFSET_SURFACE:
            styles.append("offset_surface")
        elif segment_geometry == WirePathGeometry.FREE_3D:
            styles.append("free_3d")
        elif owner is None or owner.kind == "contour_loop":
            styles.append("winding")
        else:
            styles.append("surface_connection")
    return np.asarray(styles, dtype=object)


def _annotate_wire_owner_runs(
    ax: Any,
    part: CoilPart,
) -> tuple[int, int]:
    """Annotate loop visit order and inter-group connectors without labeling every rail segment."""

    if part.wire_path is None:
        return 0, 0
    points = np.asarray(part.wire_path.v, dtype=float)
    owners = list(part.wire_path.segment_owners)
    if len(owners) != max(len(points) - 1, 0) or not owners:
        return 0, 0

    run_starts = [0]
    run_owners = [owners[0]]
    previous = _wire_owner_identity(owners[0])
    for segment_index, owner in enumerate(owners[1:], start=1):
        identity = _wire_owner_identity(owner)
        if identity != previous:
            run_starts.append(segment_index)
            run_owners.append(owner)
            previous = identity
    run_stops = [*run_starts[1:], len(owners)]

    seen_loops: set[tuple[int, int]] = set()
    seen_connectors: set[tuple[int, str]] = set()
    loop_visit_count = 0
    connector_event_count = 0
    for run_start, run_stop, owner in zip(run_starts, run_stops, run_owners):
        midpoint_index = min((run_start + run_stop) // 2, len(points) - 1)
        position = points[midpoint_index]
        if owner.kind == "contour_loop":
            loop_identity = (int(owner.group_index), int(owner.loop_index))
            if loop_identity in seen_loops:
                continue
            seen_loops.add(loop_identity)
            loop_visit_count += 1
            multiplicity = (
                int(part.contour_lines[owner.loop_index].turn_multiplicity)
                if 0 <= owner.loop_index < len(part.contour_lines)
                else 1
            )
            multiplicity_suffix = f"x{multiplicity}" if multiplicity > 1 else ""
            ax.scatter(*position, color="#ffffff", edgecolor="#111827", marker="o", s=34, depthshade=False)
            ax.text(
                *position,
                f" {loop_visit_count}:L{owner.loop_index}{multiplicity_suffix}",
                color="#111827",
                fontsize=7,
            )
            continue
        if owner.kind != "inter_group_connector":
            continue
        connector_identity = (int(owner.connector_index), str(owner.connector_role))
        if connector_identity in seen_connectors:
            continue
        seen_connectors.add(connector_identity)
        connector_event_count += 1
        role_suffix = {
            "primary": "p",
            "closing": "c",
            "bridge": "b",
        }.get(str(owner.connector_role), "")
        ax.scatter(*position, color="#ffffff", edgecolor="#7e22ce", marker="s", s=32, depthshade=False)
        ax.text(
            *position,
            f" C{owner.connector_index}{role_suffix}",
            color="#7e22ce",
            fontsize=7,
        )
    return loop_visit_count, connector_event_count


def _wire_owner_identity(owner: WireSegmentOwner) -> tuple[str, int, int, int, int, int, str]:
    return (
        str(owner.kind),
        int(owner.group_index),
        int(owner.loop_index),
        int(owner.connector_index),
        int(owner.group1_index),
        int(owner.group2_index),
        str(owner.connector_role),
    )


def _plot_path_3d(
    ax: Any,
    points: np.ndarray,
    *,
    color: Any,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str = "-",
    label: str | None = None,
) -> None:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        label=label,
    )


def _contour_display_linewidth(contour: ContourLine) -> float:
    return float(RAW_CONTOUR_LINEWIDTH * np.sqrt(max(int(contour.turn_multiplicity), 1)))


def _draw_shifted_segments_3d(ax: Any, points: np.ndarray, shift_array: np.ndarray) -> None:
    shifted = np.asarray(shift_array, dtype=float) > 0
    start = None
    for index, value in enumerate(shifted):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(shifted) - 1):
            stop = index if value and index == len(shifted) - 1 else index - 1
            if stop > start:
                _plot_path_3d(
                    ax,
                    points[start : stop + 1],
                    color="#d73a49",
                    linewidth=2.9,
                    label="normal-offset return path section",
                )
            start = None


def _draw_layout_paths_or_contours_3d(ax: Any, parts: Sequence[CoilPart], all_points: list[np.ndarray]) -> None:
    cmap = _import_pyplot().get_cmap("tab10")
    for part_ind, part in enumerate(parts, start=1):
        color = cmap((part_ind - 1) % 10)
        if part.wire_path is not None and len(part.wire_path.v) >= 2:
            _plot_path_3d(
                ax,
                part.wire_path.v,
                color=color,
                linewidth=WIRE_PATH_LINEWIDTH,
                label=f"part {part_ind} connected wire path",
            )
            all_points.append(part.wire_path.v)
            if part.shift_array is not None and np.any(np.asarray(part.shift_array) > 0):
                _draw_shifted_segments_3d(ax, part.wire_path.v, part.shift_array)
            continue
        for line_ind, line in enumerate(part.contour_lines):
            _plot_path_3d(
                ax,
                _closed_path(line.v),
                color=color,
                linewidth=max(1.35, _contour_display_linewidth(line) - 0.55),
                alpha=0.95,
                label=(
                    f"part {part_ind} final sparsified weighted contours"
                    if line_ind == 0 and part.contour_sparsification is not None
                    else f"part {part_ind} unconnected contours"
                    if line_ind == 0
                    else None
                ),
            )
            if line.turn_multiplicity > 1:
                center = np.mean(_closed_path(line.v), axis=0)
                ax.text(*center, f"x{line.turn_multiplicity}", color=color, fontsize=8)
            all_points.append(line.v)


def _symmetric_color_norm(values: np.ndarray) -> Any:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return _import_matplotlib_colors().Normalize(vmin=-1.0, vmax=1.0)
    limit = max(float(np.max(np.abs(finite))), 1e-30)
    return _import_matplotlib_colors().Normalize(vmin=-limit, vmax=limit)


def _linear_color_norm(values: np.ndarray) -> Any:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return _import_matplotlib_colors().Normalize(vmin=-1.0, vmax=1.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high, rtol=1e-12, atol=1e-15):
        delta = max(abs(low), 1.0) * 0.05
        low -= delta
        high += delta
    return _import_matplotlib_colors().Normalize(vmin=low, vmax=high)


def _finish_3d_axis(ax: Any, points: np.ndarray, *, view: str | None = None) -> None:
    points = np.asarray(points, dtype=float)
    ax.set_xlabel(_coordinate_axis_label(0))
    ax.set_ylabel(_coordinate_axis_label(1))
    ax.set_zlabel(_coordinate_axis_label(2))
    _apply_3d_view(ax, view)
    if len(points) == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-12)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    _apply_3d_view(ax, view)


def _apply_3d_view(ax: Any, view: str | None) -> None:
    if view is None:
        return
    key = str(view).strip().lower().replace("-", "_")
    if key != FACE_ISOMETRIC_VIEW:
        raise ValueError(f"Unknown 3D view preset: {view!r}")
    ax.view_init(elev=FACE_ISOMETRIC_ELEVATION_DEG, azim=FACE_ISOMETRIC_AZIMUTH_DEG)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def _finish_figure(fig: Figure, plt: Any, *, show: bool, block: bool) -> Figure:
    fig.canvas.draw_idle()
    if show:
        plt.show(block=block)
    return fig


def _append_plot_spec_figures(
    figures: list[Figure],
    spec: MatplotlibPlotSpec,
    *,
    timing_report: list[dict[str, float | int | str]] | None = None,
) -> None:
    start = time.perf_counter()
    try:
        figure = spec.factory()
    except ValueError:
        return
    elapsed = float(time.perf_counter() - start)
    if isinstance(figure, list):
        figures.extend(figure)
        figure_count = len(figure)
    else:
        figures.append(figure)
        figure_count = 1
    if timing_report is not None:
        timing_report.append({"plot": spec.key, "seconds": elapsed, "figures": figure_count})


def _normalise_result_plot_keys(include: Sequence[str] | None) -> set[str]:
    keys = set(include or DEFAULT_RESULT_PLOT_KEYS)
    if keys & CURRENT_FLOW_COMPANION_ANCHOR_KEYS:
        keys.update(CURRENT_FLOW_COMPANION_KEYS)
    return keys


def _short_plot_book_name(spec: MatplotlibPlotSpec, title: str, index: int, total: int) -> str:
    base_by_key = {
        "wire": "wire",
        "wire_sequence": "wire order",
        "raw_contours": "raw contours",
        "contour_current_orientation": "current dirs",
        "contour_groups": "groups",
        "stream_function": "stream fn",
        "current_density": "current dens",
        "interconnect_debug": "routing cuts",
        "swept_body": "swept body",
        "problem_setup_geometry": "setup geometry",
        "layout_field_overview": "layout + fields",
        "field_comparison": "Bz lines",
        "field_pointclouds": "Bz cloud",
        "shim_pointclouds": "shim cloud",
        "shim_histograms": "shim hist",
        "shim_spherical_harmonics": "shim SH",
        "spherical_harmonics": "field SH",
        "target_field_maps": "Bz maps",
        "gradient_maps": "grad maps",
        "error_maps": "error maps",
        "shim_homogeneity": "shim mosaic",
        "field_cross_sections": "Bz slices",
        "gradient_cross_sections": "grad slices",
        "field_slice_mosaic": "field pixels",
        "gradient_slice_mosaic": "grad pixels",
        "concomitant_slice_mosaic": "concomitant",
        "error_cross_sections": "error slices",
        "error_metrics": "error bars",
        "performance_table": "summary",
        "regularization": "L-curve",
    }
    base = base_by_key.get(spec.key, spec.key.replace("_", " "))
    part = _plot_book_part_suffix(title)
    if part:
        base = f"{base} {part}"
    if total > 1:
        component = _plot_book_component_suffix(title)
        base = f"{base} {component or index}"
    return base[:32]


def _plot_book_part_suffix(title: str) -> str:
    import re

    match = re.search(r"(?:Coil )?Part\s+(\d+)", str(title), flags=re.IGNORECASE)
    return f"P{match.group(1)}" if match else ""


def _plot_book_component_suffix(title: str) -> str:
    title_lower = str(title).lower()
    for token, label in (
        ("before shimming", "before"),
        ("after shimming", "after"),
        ("|grad bz|", "|grad|"),
        ("dbz/dx", "dBz/dx"),
        ("dbz/dy", "dBz/dy"),
        ("dbz/dz", "dBz/dz"),
        ("magnitude", "mag"),
    ):
        if token in title_lower:
            return label
    return ""


def _performance_table_rows(result: CoilGenResult) -> list[list[str]]:
    return result_performance_table_rows(result)


def _format_performance_table_value(value: Any) -> str:
    return format_result_summary_value(value)


def _shim_histogram_domain_values(result: CoilGenResult, scalar_map: Any, *, domain: str) -> np.ndarray:
    coords = np.asarray(result.target_field.coords, dtype=float)
    values = np.asarray(scalar_map.values, dtype=float).reshape(-1)
    finite = target_domain_mask(result, domain) & _finite_point_scalar_mask(coords, values)
    return values[finite]


def _shim_histogram_stats_text(result: CoilGenResult, *, after_source: str, domain: str) -> str:
    before = shim_homogeneity_stats(result, source="before", domain=domain)
    after = shim_homogeneity_stats(result, source=after_source, domain=domain)

    def line(label: str, stats: dict[str, float | str]) -> str:
        unit = str(stats.get("unit", "")).strip()
        suffix = f" {unit}" if unit else ""
        return (
            f"{label}: std {_format_performance_table_value(stats.get('std'))}{suffix}, "
            f"RMS {_format_performance_table_value(stats.get('rms'))}{suffix}, "
            f"p95 abs {_format_performance_table_value(stats.get('p95_abs'))}{suffix}, "
            f"max abs {_format_performance_table_value(stats.get('max_abs'))}{suffix}, "
            f"n {int(float(stats.get('sample_count', 0.0)))}"
        )

    return f"{line('Before', before)}   |   {line('After', after)}"


def _shim_mosaic_payload(
    result: CoilGenResult,
    *,
    source: str,
    axis: int,
    slice_count: int,
    domain: str,
    grid_size: int,
    min_points_per_slice: int,
    slab_thickness_m: float | None,
) -> dict[str, Any]:
    coords = np.asarray(result.target_field.coords, dtype=float)
    scalar_map = shim_field_scalar_maps(result)[_shim_source_key(source)]
    values = np.asarray(scalar_map.values, dtype=float).reshape(-1)
    domain_mask = target_domain_mask(result, domain)
    finite = domain_mask & _finite_point_scalar_mask(coords, values)
    if not np.any(finite):
        raise ValueError("Shim homogeneity mosaic has no finite samples.")
    slices = interpolated_slice_mosaic(
        coords,
        values,
        axis=axis,
        slice_count=slice_count,
        sample_mask=finite,
        grid_size=grid_size,
        min_points=min_points_per_slice,
        slab_thickness=slab_thickness_m,
    )
    if not slices:
        raise ValueError("Shim homogeneity mosaic could not interpolate any slices.")
    return {
        "source": source,
        "axis": int(axis),
        "domain": domain,
        "scalar_map": scalar_map,
        "values": values,
        "finite": finite,
        "slices": slices,
        "limit": max(float(np.nanmax(np.abs(values[finite]))), 1e-30),
        "stats": shim_homogeneity_stats(result, source=source, domain=domain),
    }


def _shim_reference_color_limit(result: CoilGenResult, *, source: str, domain: str) -> float | None:
    source_key = _shim_source_key(source)
    if source_key not in {
        "layout_remaining_inhomogeneity",
        "stream_remaining_inhomogeneity",
        "layout_improvement_before_minus_after",
        "stream_improvement_before_minus_after",
    }:
        return None
    before_payload = _shim_mosaic_scalar_values(result, source="before", domain=domain)
    return max(float(np.nanmax(np.abs(before_payload["values"][before_payload["finite"]]))), 1e-30)


def _shim_mosaic_scalar_values(result: CoilGenResult, *, source: str, domain: str) -> dict[str, np.ndarray]:
    coords = np.asarray(result.target_field.coords, dtype=float)
    scalar_map = shim_field_scalar_maps(result)[_shim_source_key(source)]
    values = np.asarray(scalar_map.values, dtype=float).reshape(-1)
    finite = target_domain_mask(result, domain) & _finite_point_scalar_mask(coords, values)
    if not np.any(finite):
        raise ValueError("Shim homogeneity mosaic has no finite samples for color scaling.")
    return {"values": values, "finite": finite}


def _shim_payload_with_color_limit(payload: dict[str, Any], limit: float) -> dict[str, Any]:
    updated = dict(payload)
    updated["limit"] = max(float(limit), 1e-30)
    return updated


def _shim_abs_metric_label(source_key: str) -> str:
    key = str(source_key)
    if key in {"layout_improvement_before_minus_after", "stream_improvement_before_minus_after"}:
        return "improvement"
    if key == "before_inhomogeneity":
        return "inhomogeneity"
    if key in {"target", "layout", "stream"}:
        return "field"
    return "residual"


def _add_shim_before_after_slice_rows(
    fig: Figure,
    before: dict[str, Any],
    after: dict[str, Any],
    improvement: dict[str, Any],
    *,
    plt: Any,
) -> None:
    payloads = (before, after, improvement)
    row_count = max(len(payload["slices"]) for payload in payloads)
    axis = int(before["axis"])
    field_limit = max(float(before["limit"]), float(after["limit"]), 1e-30)
    improvement_limit = max(float(improvement["limit"]), 1e-30)
    before["limit"] = field_limit
    after["limit"] = field_limit
    improvement["limit"] = improvement_limit

    colors = _import_matplotlib_colors()
    field_norm = colors.Normalize(vmin=-field_limit, vmax=field_limit)
    improvement_norm = colors.Normalize(vmin=-improvement_limit, vmax=improvement_limit)
    field_cmap = _copy_matplotlib_cmap_with_bad(plt.get_cmap("jet"))
    improvement_cmap = _copy_matplotlib_cmap_with_bad(plt.get_cmap("RdYlGn"))

    fig.clf()
    grid = fig.add_gridspec(
        row_count + 2,
        7,
        height_ratios=[0.44, 0.48] + [1.0] * row_count,
        width_ratios=[0.62, 1.0, 0.050, 1.0, 0.050, 1.0, 0.050],
        left=0.055,
        right=0.955,
        top=0.985,
        bottom=0.035,
        wspace=0.14,
        hspace=0.20,
    )

    title_ax = fig.add_subplot(grid[0, :6])
    title_ax.set_axis_off()
    domain = before["domain"]
    unit = before["stats"]["unit"]
    title_ax.text(
        0.0,
        0.64,
        (
            f"B0 Inhomogeneity Z-Slice Review ({domain} domain; "
            f"before/after +/-{field_limit:.4g} {unit}; improvement +/-{improvement_limit:.4g} {unit})"
        ),
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    title_ax.text(
        0.0,
        0.18,
        (
            "Rows are z-slices; columns compare before shimming, "
            f"{after['scalar_map'].label.lower()}, and signed improvement."
        ),
        ha="left",
        va="center",
        fontsize=9,
        color="#475569",
    )

    header_labels = (
        _shim_slice_column_header(before, "Before"),
        _shim_slice_column_header(after, "After"),
        _shim_slice_column_header(improvement, "Improvement"),
    )
    for col, label in zip((1, 3, 5), header_labels):
        ax = fig.add_subplot(grid[1, col])
        ax.set_axis_off()
        ax.text(0.5, 0.55, label, ha="center", va="center", fontsize=10, fontweight="bold")
    for col, label in zip((2, 4, 6), ("Scale", "Scale", "Delta")):
        scale_header_ax = fig.add_subplot(grid[1, col])
        scale_header_ax.set_axis_off()
        scale_header_ax.text(
            0.5,
            0.55,
            format_label_with_unit(label, unit),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            rotation=90,
        )

    plot_x, plot_y = before["slices"][0].plot_axes
    field_mappable = _import_scalar_mappable()(norm=field_norm, cmap=field_cmap)
    field_mappable.set_array([])
    improvement_mappable = _import_scalar_mappable()(norm=improvement_norm, cmap=improvement_cmap)
    improvement_mappable.set_array([])
    column_specs = (
        (before, 1, 2, field_cmap, field_norm, field_mappable, field_limit),
        (after, 3, 4, field_cmap, field_norm, field_mappable, field_limit),
        (improvement, 5, 6, improvement_cmap, improvement_norm, improvement_mappable, improvement_limit),
    )
    for row in range(row_count):
        reference_slice = _payload_slice_at(before, row)
        label_ax = fig.add_subplot(grid[row + 2, 0])
        label_ax.set_axis_off()
        label_ax.text(
            0.98,
            0.50,
            _shim_slice_row_label(reference_slice, axis, before["finite"]),
            ha="right",
            va="center",
            fontsize=9,
            color="#334155",
        )
        for payload, plot_col, colorbar_col, cmap, norm, mappable, limit in column_specs:
            ax = fig.add_subplot(grid[row + 2, plot_col])
            cax = fig.add_subplot(grid[row + 2, colorbar_col])
            interpolated = _payload_slice_at(payload, row)
            if interpolated is None:
                ax.set_axis_off()
                cax.set_axis_off()
                continue
            _draw_shim_interpolated_slice(
                ax,
                interpolated,
                payload,
                cmap=cmap,
                norm=norm,
                show_x_label=row == row_count - 1,
                show_y_label=plot_col == 1,
                plot_x=plot_x,
                plot_y=plot_y,
            )
            colorbar = fig.colorbar(mappable, cax=cax)
            _set_scientific_colorbar_ticks(colorbar, -limit, limit, count=5)
            colorbar.ax.tick_params(labelsize=7)


def _copy_matplotlib_cmap_with_bad(cmap: Any, bad_color: str = "#f8fafc") -> Any:
    copied = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        copied.set_bad(bad_color)
    except Exception:
        pass
    return copied


def _shim_slice_column_header(payload: dict[str, Any], short_label: str) -> str:
    stats = payload["stats"]
    scalar_map = payload["scalar_map"]
    metric_label = _shim_abs_metric_label(scalar_map.key)
    return (
        f"{short_label}\n"
        f"std {float(stats['std']):.4g} {stats['unit']}; "
        f"RMS {float(stats['rms']):.4g}; "
        f"p95 |{metric_label}| {float(stats['p95_abs']):.4g}"
    )


def _payload_slice_at(payload: dict[str, Any], index: int) -> Any | None:
    slices = payload["slices"]
    return slices[index] if 0 <= int(index) < len(slices) else None


def _shim_slice_row_label(interpolated: Any | None, axis: int, finite: np.ndarray) -> str:
    if interpolated is None:
        return ""
    mask = interpolated.mask & finite
    return f"{AXIS_NAMES[int(axis)]} = {float(interpolated.value):.4g} m\nn = {int(np.count_nonzero(mask))}"


def _draw_shim_interpolated_slice(
    ax: Any,
    interpolated: Any,
    payload: dict[str, Any],
    *,
    cmap: Any,
    norm: Any,
    show_x_label: bool,
    show_y_label: bool,
    plot_x: int,
    plot_y: int,
) -> None:
    grid_values = np.ma.masked_invalid(interpolated.values)
    ax.imshow(
        grid_values,
        origin="lower",
        extent=(
            float(interpolated.x[0]),
            float(interpolated.x[-1]),
            float(interpolated.y[0]),
            float(interpolated.y[-1]),
        ),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_aspect("equal", adjustable="box")
    if show_x_label:
        ax.set_xlabel(_coordinate_axis_label(plot_x), fontsize=9)
    else:
        ax.set_xticklabels([])
    if show_y_label:
        ax.set_ylabel(_coordinate_axis_label(plot_y), fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(False)


def _add_shim_mosaic_section(
    fig: Figure,
    spec: Any,
    payload: dict[str, Any],
    *,
    plt: Any,
    columns: int | None = None,
    title_fontsize: int = 11,
) -> None:
    slices = payload["slices"]
    axis = int(payload["axis"])
    finite = payload["finite"]
    scalar_map = payload["scalar_map"]
    stats = payload["stats"]
    limit = float(payload["limit"])
    metric_label = _shim_abs_metric_label(scalar_map.key)
    if columns is None:
        columns = int(np.ceil(np.sqrt(len(slices))))
    rows = int(np.ceil(len(slices) / int(columns)))
    grid = spec.subgridspec(
        rows + 1,
        int(columns) + 1,
        height_ratios=[0.18] + [1.0] * rows,
        width_ratios=[1.0] * int(columns) + [0.07],
        wspace=0.18,
        hspace=0.34,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.55,
        (
            f"{scalar_map.label_with_unit}; interpolated {AXIS_NAMES[axis]}-slabs; {payload['domain']} domain; "
            f"std {float(stats['std']):.4g}, RMS {float(stats['rms']):.4g}, "
            f"p95 |{metric_label}| {float(stats['p95_abs']):.4g}, "
            f"max |{metric_label}| {float(stats['max_abs']):.4g} {stats['unit']}"
        ),
        ha="left",
        va="center",
        fontsize=title_fontsize,
    )

    colors = _import_matplotlib_colors()
    norm = colors.Normalize(vmin=-limit, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        cmap.set_bad("#f8fafc")
    except Exception:
        pass

    plot_x, plot_y = slices[0].plot_axes
    for slot in range(rows * int(columns)):
        row = slot // int(columns)
        col = slot % int(columns)
        ax = fig.add_subplot(grid[row + 1, col])
        if slot >= len(slices):
            ax.set_visible(False)
            continue
        interpolated = slices[slot]
        grid_values = np.ma.masked_invalid(interpolated.values)
        ax.imshow(
            grid_values,
            origin="lower",
            extent=(
                float(interpolated.x[0]),
                float(interpolated.x[-1]),
                float(interpolated.y[0]),
                float(interpolated.y[-1]),
            ),
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
            aspect="equal",
        )
        mask = interpolated.mask & finite
        ax.set_aspect("equal", adjustable="box")
        if row == rows - 1:
            ax.set_xlabel(_coordinate_axis_label(plot_x))
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(_coordinate_axis_label(plot_y))
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{AXIS_NAMES[axis]}={interpolated.value:.4g} m; n={int(np.count_nonzero(mask))}", fontsize=8)
        ax.grid(False)

    cax = fig.add_subplot(grid[1:, int(columns)])
    scalar_mappable = _import_scalar_mappable()(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, cax=cax)
    _set_scientific_colorbar_ticks(colorbar, -limit, limit)
    colorbar.set_label(scalar_map.label_with_unit)


def _scalar_subplot_title(scalar_map: Any) -> str:
    return str(getattr(scalar_map, "label_with_unit", getattr(scalar_map, "label", "")))


def _shim_source_key(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    if key in {
        "before",
        "pre",
        "pre_shim",
        "pre_shimming",
        "unshimmed",
        "fieldmap",
        "inhomogeneity",
        "before_inhomogeneity",
    }:
        return "before_inhomogeneity"
    if key in {
        "after",
        "post",
        "post_shim",
        "post_shimming",
        "remaining",
        "remaining_inhomogeneity",
        "layout",
        "connected",
        "connected_wire",
        "wire",
        "wire_layout",
        "coil",
        "coilfield",
        "layout_remaining",
        "layout_remaining_inhomogeneity",
    }:
        return "layout_remaining_inhomogeneity"
    if key in {"after_stream", "stream_remaining", "stream_remaining_inhomogeneity"}:
        return "stream_remaining_inhomogeneity"
    if key in {"layout_residual", "target_minus_layout"}:
        return "layout_residual"
    if key in {"stream", "stream_function", "sf"}:
        return "stream_remaining_inhomogeneity"
    if key in {"stream_residual", "target_minus_stream"}:
        return "stream_residual"
    if key in {
        "improvement",
        "delta",
        "delta_improvement",
        "before_minus_after",
        "layout_improvement",
        "wire_improvement",
        "connected_improvement",
        "layout_improvement_before_minus_after",
    }:
        return "layout_improvement_before_minus_after"
    if key in {"stream_improvement", "before_minus_after_stream", "stream_improvement_before_minus_after"}:
        return "stream_improvement_before_minus_after"
    raise ValueError("source must be 'before', 'after', 'layout'/'coil', 'stream', or 'improvement'.")


def _after_shim_source(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    if key in {
        "after",
        "post",
        "post_shim",
        "post_shimming",
        "remaining",
        "remaining_inhomogeneity",
        "layout",
        "connected",
        "connected_wire",
        "coil",
        "coilfield",
        "layout_remaining",
        "layout_remaining_inhomogeneity",
        "layout_residual",
        "target_minus_layout",
        "improvement",
        "delta",
        "delta_improvement",
        "before_minus_after",
        "layout_improvement",
        "wire_improvement",
        "connected_improvement",
        "layout_improvement_before_minus_after",
    }:
        return "after"
    if key in {
        "after_stream",
        "stream",
        "stream_function",
        "sf",
        "stream_remaining",
        "stream_remaining_inhomogeneity",
        "stream_residual",
        "target_minus_stream",
        "stream_improvement",
        "before_minus_after_stream",
        "stream_improvement_before_minus_after",
    }:
        return "after_stream"
    return source


def _improvement_shim_source(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    if key in {
        "after_stream",
        "stream",
        "stream_function",
        "sf",
        "stream_remaining",
        "stream_remaining_inhomogeneity",
        "stream_residual",
        "target_minus_stream",
        "stream_improvement",
        "before_minus_after_stream",
        "stream_improvement_before_minus_after",
    }:
        return "stream_improvement"
    return "improvement"


def _regularization_scan_arrays(result: CoilGenResult) -> dict[str, Any]:
    return regularization_scan_arrays(result)


def _set_axis_title(ax: Any, title: str, *, width: int = 34) -> None:
    ax.set_title(_wrap_plot_title(title, width=width), pad=10)


def _wrap_plot_title(title: str, *, width: int = 34) -> str:
    lines: list[str] = []
    for raw_line in str(title).splitlines():
        if len(raw_line) <= width:
            lines.append(raw_line)
        else:
            lines.extend(textwrap.wrap(raw_line, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines)


def _gradient_component_title(component: str) -> str:
    key = str(component).lower()
    labels = {
        "magnitude": "|grad Bz| magnitude",
        "mag": "|grad Bz| magnitude",
        "norm": "|grad Bz| magnitude",
        "dbz_dx": "dBz/dx component",
        "x": "dBz/dx component",
        "dx": "dBz/dx component",
        "dbz_dy": "dBz/dy component",
        "y": "dBz/dy component",
        "dy": "dBz/dy component",
        "dbz_dz": "dBz/dz component",
        "z": "dBz/dz component",
        "dz": "dBz/dz component",
    }
    return labels.get(key, key)


def _subplot_grid(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    return rows, cols


def _deduplicate_legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        filtered_handles.append(handle)
        filtered_labels.append(label)
    if filtered_handles:
        ax.legend(filtered_handles, filtered_labels, loc="upper left")


def _add_colorbar(fig: Figure, ax: Any, cmap: Any, norm: Any, label: str) -> None:
    scalar_mappable = _import_scalar_mappable()(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.7, pad=0.08)
    _set_scientific_colorbar_ticks(colorbar, norm.vmin, norm.vmax)
    colorbar.set_label(label)


def _add_artist_colorbar(fig: Figure, ax: Any, artist: Any, label: str, *, shrink: float, pad: float) -> None:
    colorbar = fig.colorbar(artist, ax=ax, shrink=shrink, pad=pad)
    vmin, vmax = _artist_color_range(artist)
    _set_scientific_colorbar_ticks(colorbar, vmin, vmax)
    colorbar.set_label(label)


def _artist_color_range(artist: Any) -> tuple[float, float]:
    if hasattr(artist, "get_clim"):
        clim = artist.get_clim()
        if clim is not None and len(clim) == 2:
            return float(clim[0]), float(clim[1])
    if hasattr(artist, "get_array"):
        values = np.asarray(artist.get_array(), dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if len(finite):
            return float(np.min(finite)), float(np.max(finite))
    return float("nan"), float("nan")


def _set_scientific_colorbar_ticks(
    colorbar: Any, vmin: float, vmax: float, *, count: int = COLORBAR_TICK_COUNT
) -> None:
    ticks = _colorbar_tick_values(vmin, vmax, count=count)
    if len(ticks) == 0:
        return
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([_format_colorbar_tick(value) for value in ticks])
    colorbar.ax.tick_params(labelsize=8)


def _colorbar_tick_values(vmin: float, vmax: float, *, count: int = COLORBAR_TICK_COUNT) -> np.ndarray:
    values = np.asarray([vmin, vmax], dtype=float)
    if not np.all(np.isfinite(values)):
        return np.zeros(0, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high, rtol=1e-12, atol=1e-15):
        return np.asarray([low], dtype=float)
    return np.linspace(low, high, max(2, int(count)))


def _format_colorbar_tick(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _coordinate_axis_label(axis: int) -> str:
    return format_label_with_unit(AXIS_NAMES[int(axis)], COORDINATE_UNIT)


def _face_average_stream_function(part: CoilPart) -> np.ndarray:
    if part.stream_function is None:
        raise ValueError("Part has no stream function.")
    stream = np.asarray(part.stream_function, dtype=float)
    return np.mean(stream[part.mesh.faces], axis=1)


def _current_density_magnitude(part: CoilPart) -> np.ndarray:
    current_density = part.current_density
    if current_density is None and part.current_density_mat is not None and part.stream_function is not None:
        current_density = np.tensordot(part.stream_function, part.current_density_mat, axes=(0, 0))
    if current_density is None:
        current_density = current_density_from_basis(part)
    if current_density is None:
        raise ValueError("Part has no current density.")
    current_density = np.asarray(current_density, dtype=float)
    if current_density.shape != (len(part.mesh.faces), 3):
        raise ValueError("Current density must have shape (n_faces, 3).")
    return np.linalg.norm(current_density, axis=1)


def _closed_path(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or np.linalg.norm(points[0] - points[-1]) <= 1e-10:
        return points
    return np.vstack((points, points[0]))


def _stack_points(arrays: Iterable[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(array, dtype=float) for array in arrays if array is not None and len(array)]
    if not valid:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(valid)


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Matplotlib is required for active Spyder plots. Install it in the Python environment used by Spyder, "
            "or use save_plot_gallery for dependency-free SVG files."
        ) from exc
    return plt


def _import_matplotlib_colors() -> Any:
    from matplotlib import colors

    return colors


def _import_matplotlib_button() -> Any:
    from matplotlib.widgets import Button

    return Button


def _import_scalar_mappable() -> Any:
    from matplotlib.cm import ScalarMappable

    return ScalarMappable


def _import_mplot3d_collections() -> Any:
    from mpl_toolkits.mplot3d import art3d

    return art3d
