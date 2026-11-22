"""Orthogonal views of saved samples, without inventing unsampled field values."""

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


def draw_sample_sections(figure, points_m, values, label, unit, limits, signed, *, plane="all"):
    """Use one honest physical scale for every section and retain native holes."""
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
        section = sample_section(points_m, values, name)
        horizontal, vertical, normal = section.axes
        axis.set_facecolor("#e8edf2")
        if section.grid is not None:
            u, v, cells = section.grid
            masked = np.ma.masked_invalid(cells)
            axis.pcolormesh(u, v, masked, shading="nearest", cmap=cmap, norm=norm, rasterized=True)
            # Lines emphasize field structure but never bridge missing cells.
            finite = cells[np.isfinite(cells)]
            if len(finite) and np.ptp(finite) > max(abs(limits[0]), abs(limits[1])) * 1e-8:
                levels = np.linspace(*limits, 9)[1:-1]
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
        axis.set_title(f"{name.upper()} · {'xyz'[normal]} = {position} mm\n{kind}", fontsize=10, loc="left", pad=7)
        axis.set_xlabel(f"{'xyz'[horizontal]} [mm]", fontsize=10)
        axis.set_ylabel(f"{'xyz'[vertical]} [mm]", fontsize=10)
        axis.set_aspect("equal", adjustable="box")
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(labelsize=9, length=3, colors="#405466")
        for spine in axis.spines.values():
            spine.set_color("#c4cfd9")
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
