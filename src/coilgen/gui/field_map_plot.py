"""Field sections and matched comparisons of saved or directly evaluated data."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampleSection:
    axes: tuple
    points_mm: np.ndarray
    values: np.ndarray
    normal_range_mm: tuple
    grid: tuple | None


def sample_section(points_m, values, plane):
    """Take the central saved plane, or a labelled thin slab for scattered data.

    Rectangular sections retain missing cells as NaN. Scattered samples are
    shown individually: never flatten an entire volume into a supposed slice.
    """
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[plane]
    points = np.asarray(points_m) * 1000
    values = np.asarray(values)
    normal = points[:, axes[2]]
    middle = (normal.min() + normal.max()) / 2
    tolerance = 128 * np.finfo(float).eps * max(1, float(np.abs(points).max()))
    nearest = normal[np.argmin(np.abs(normal - middle))]
    selected = np.abs(normal - nearest) <= tolerance
    # A repeated coordinate plane is a real section. For unstructured volumes
    # use a narrow, explicitly labelled slab rather than a fictitious surface.
    if selected.sum() < 3:
        half_width = max(float(np.ptp(normal)) * 0.05, float(np.min(np.abs(normal - middle))))
        selected = np.abs(normal - middle) <= half_width + tolerance
    section, scalar = points[selected], values[selected]
    u, ui = np.unique(section[:, axes[0]], return_inverse=True)
    v, vi = np.unique(section[:, axes[1]], return_inverse=True)
    grid = None
    planar = float(np.ptp(section[:, axes[2]])) <= tolerance
    # Only paint cells on a regular lattice; holes remain grey and duplicate
    # coordinates stay individual observations rather than being averaged.
    regular = all(len(a) > 1 and np.allclose(np.diff(a), np.diff(a)[0], rtol=1e-7, atol=tolerance) for a in (u, v))
    if planar and regular and len(u) * len(v) <= 4 * len(section):
        addresses = vi * len(u) + ui
        if len(np.unique(addresses)) == len(addresses):
            cells = np.full((len(v), len(u)), np.nan)
            cells[vi, ui] = scalar
            grid = (u, v, cells)
    return SampleSection(
        axes,
        section,
        scalar,
        (float(section[:, axes[2]].min()), float(section[:, axes[2]].max())),
        grid,
    )


def _evaluated_section(piece, values, plane):
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[plane]
    horizontal, vertical, _ = axes
    points = piece.coordinates_m * 1000
    return SampleSection(
        axes,
        points.reshape(-1, 3),
        values.ravel(),
        (piece.offset_m * 1000, piece.offset_m * 1000),
        (points[0, :, horizontal], points[:, 0, vertical], values),
    )


def _draw_section(axis, section, cmap, norm, *, piece=None, title=None):
    from matplotlib.ticker import MaxNLocator

    horizontal, vertical, normal = section.axes
    axis.set_facecolor("#e8edf2")
    if section.grid is not None:
        u, v, cells = section.grid
        masked = np.ma.masked_invalid(cells)
        if piece is not None and masked.count():
            axis.contourf(
                u, v, masked, levels=np.linspace(norm.vmin, norm.vmax, 81), cmap=cmap, norm=norm, corner_mask=False
            )
        else:
            axis.pcolormesh(u, v, masked, shading="nearest", cmap=cmap, norm=norm, rasterized=True)
        finite = cells[np.isfinite(cells)]
        if len(finite) and np.ptp(finite) > max(abs(norm.vmin), abs(norm.vmax)) * 1e-8:
            levels = np.linspace(norm.vmin, norm.vmax, 9)[1:-1]
            levels = levels[(levels > finite.min()) & (levels < finite.max())]
            if len(levels):
                axis.contour(
                    u, v, masked, levels=levels, colors="#243746", linewidths=0.55, alpha=0.65, corner_mask=False
                )
    else:
        axis.scatter(
            section.points_mm[:, horizontal],
            section.points_mm[:, vertical],
            c=section.values,
            cmap=cmap,
            norm=norm,
            s=36,
            edgecolors="white",
            linewidths=0.35,
            rasterized=True,
        )
    low, high = section.normal_range_mm
    position = f"{low:.4g}" if np.isclose(low, high, rtol=0, atol=1e-10) else f"{low:.4g} … {high:.4g}"
    kind = "saved plane" if low == high else "sample slab"
    if piece is not None:
        kind = f"{cells.shape[1]} × {cells.shape[0]} grid"
        for outline in piece.outlines_m:
            axis.plot(outline[:, horizontal] * 1000, outline[:, vertical] * 1000, color="#607080", linewidth=0.7)
        axis.set_xlim(float(u.min()), float(u.max()))
        axis.set_ylim(float(v.min()), float(v.max()))
    name = "xyz"[horizontal] + "xyz"[vertical]
    axis.set_title(title or f"{name.upper()} · {'xyz'[normal]} = {position} mm\n{kind}", fontsize=10, loc="left", pad=7)
    axis.set_xlabel(f"{'xyz'[horizontal]} [mm]", fontsize=10)
    axis.set_ylabel(f"{'xyz'[vertical]} [mm]", fontsize=10)
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(labelsize=9, length=3, colors="#405466")
    for spine in axis.spines.values():
        spine.set_color("#c4cfd9")


def draw_sample_sections(figure, points_m, values, label, unit, limits, signed, *, plane="all", evaluated=None):
    """Draw native observations or explicitly evaluated grids on one scale."""
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    figure.clear()
    figure.set_layout_engine("compressed")
    planes = ("xy", "xz", "yz") if plane == "all" else (plane,)
    width, height = figure.get_size_inches()
    compact = len(planes) == 3 and width / height < 1.8
    axes = figure.subplots(2, 2) if compact else figure.subplots(1, len(planes), squeeze=False)
    cmap = colormaps["RdBu_r" if signed else "viridis"].with_extremes(bad="#e8edf2")
    norm = Normalize(*limits)
    visible = []
    for axis, name in zip(axes.flat, planes):
        fine = (evaluated or {}).get(name)
        if fine is not None:
            piece, scalar, _ = fine
            section = _evaluated_section(piece, scalar, name)
        else:
            section = sample_section(points_m, values, name)
        _draw_section(axis, section, cmap, norm, piece=piece if fine is not None else None)
        visible.append(axis)
    if compact:
        distribution = axes[1, 1]
        counts, edges = np.histogram(np.asarray(values), bins=20, range=limits)
        distribution.bar(
            (edges[:-1] + edges[1:]) / 2,
            counts,
            width=np.diff(edges),
            color=cmap(norm((edges[:-1] + edges[1:]) / 2)),
            edgecolor="white",
            linewidth=0.4,
        )
        distribution.set_title("All saved samples\nValue distribution", fontsize=10, loc="left", pad=7)
        distribution.set_xlabel(f"Map [{unit}]", fontsize=10)
        distribution.set_ylabel("Samples", fontsize=10)
        distribution.set_box_aspect(1)
        distribution.set_xlim(*limits)
        distribution.xaxis.set_major_locator(MaxNLocator(4))
        distribution.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        distribution.tick_params(labelsize=9, colors="#405466")
        for spine in distribution.spines.values():
            spine.set_color("#c4cfd9")
    bar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=visible, orientation="horizontal", fraction=0.06, pad=0.08, aspect=45
    )
    bar.set_label(f"{label} [{unit}]", fontsize=10)
    bar.ax.tick_params(labelsize=9)
    bar.locator = MaxNLocator(5)
    bar.update_ticks()


def draw_field_comparison(figure, points_m, values, component, unit, *, plane, evaluated=None, scale_values=()):
    """Target | actual | residual with identical support and one shared field norm.

    ``values`` and each evaluated triple are ordered target, actual, residual.
    Additional scale values keep colour limits stable across orthogonal planes.
    """
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    from ..application.field_presentation import paired_color_limits

    figure.clear()
    figure.set_layout_engine("compressed")
    axes = figure.subplots(1, 3)
    triples = [values, *scale_values]
    if evaluated is not None:
        triples.append(evaluated[1])
    field_limits = paired_color_limits(
        np.concatenate([triple[0].ravel() for triple in triples]),
        np.concatenate([triple[1].ravel() for triple in triples]),
        signed=component != "magnitude",
    )
    residual_limits = paired_color_limits(np.concatenate([triple[2].ravel() for triple in triples]), None)
    field_norm, residual_norm = Normalize(*field_limits), Normalize(*residual_limits)
    field_cmap = colormaps["viridis" if component == "magnitude" else "RdBu_r"].with_extremes(bad="#e8edf2")
    residual_cmap = colormaps["RdBu_r"].with_extremes(bad="#e8edf2")
    label = "|B|" if component == "magnitude" else "B∥" if component == "Bparallel" else component
    titles = (f"Target {label}", f"Actual {label}", f"Residual Δ{label}")
    piece, shown = evaluated if evaluated is not None else (None, values)
    for index, (axis, scalar, title) in enumerate(zip(axes, shown, titles)):
        section = (
            _evaluated_section(piece, scalar, plane) if piece is not None else sample_section(points_m, scalar, plane)
        )
        _draw_section(
            axis,
            section,
            residual_cmap if index == 2 else field_cmap,
            residual_norm if index == 2 else field_norm,
            piece=piece,
            title=title,
        )
        if not np.any(np.isfinite(section.values)):
            axis.text(0.5, 0.5, "No paired values", transform=axis.transAxes, ha="center", va="center", fontsize=9)
    normal = section.axes[2]
    low, high = section.normal_range_mm
    position = f"{low:.4g}" if np.isclose(low, high, rtol=0, atol=1e-10) else f"{low:.4g} … {high:.4g}"
    kind = (
        f"{piece.coordinates_m.shape[1]} × {piece.coordinates_m.shape[0]} grid"
        if piece is not None
        else "saved samples"
    )
    figure.suptitle(f"{plane.upper()} · {'xyz'[normal]} = {position} mm · {kind}", fontsize=11, color="#243746")
    for selected_axes, norm, cmap, caption in (
        (list(axes[:2]), field_norm, field_cmap, f"Target / actual [{unit}] · shared scale"),
        ([axes[2]], residual_norm, residual_cmap, f"Actual − target [{unit}]"),
    ):
        bar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=selected_axes,
            orientation="horizontal",
            fraction=0.07,
            pad=0.12,
            aspect=35 if len(selected_axes) == 2 else 18,
        )
        bar.set_label(caption, fontsize=9)
        bar.ax.tick_params(labelsize=9)
        bar.locator = MaxNLocator(4)
        bar.update_ticks()
    return field_limits, residual_limits
