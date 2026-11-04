"""Compare paired, saved pre-routing contours on the positive-x biplanar Gy plate."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.spatial import ConvexHull


HERE = Path(__file__).resolve().parent
ORIGINAL, SMOOTH, INK = "#bd532c", "#00756c", "#203747"


def furthest_vertex(source, target):
    """Return a source vertex and its nearest point on any target segment, in mm."""
    start = target[:-1]
    delta = np.diff(target, axis=0)
    length2 = np.einsum("ij,ij->i", delta, delta)
    start, delta, length2 = start[length2 > 0], delta[length2 > 0], length2[length2 > 0]
    if not len(start):
        raise ValueError("A saved comparison path has no nonzero segments.")
    relative = source[:, None] - start
    fraction = np.clip(np.einsum("ijk,jk->ij", relative, delta) / length2, 0, 1)
    closest = start + fraction[:, :, None] * delta
    squared = ((source[:, None] - closest) ** 2).sum(axis=2)
    nearest = squared.argmin(axis=1)
    vertex = squared[np.arange(len(source)), nearest].argmax()
    return float(np.sqrt(squared[vertex, nearest[vertex]])), source[vertex], closest[vertex, nearest[vertex]]


def load_plate():
    with np.load(HERE / "centerlines/biplanar__0.npz", allow_pickle=False) as data:
        vertices = data["support_0_vertices_m"] * 1000
        keys = {
            int(match.group(1)) for key in data if (match := re.fullmatch(r"original_contour_0_(\d+)_points_m", key))
        }
        pairs = {}
        for index in sorted(keys):
            name = f"contour_0_{index}_points_m"
            old, new = data["original_" + name] * 1000, data[name] * 1000
            for path in (old, new):
                if not np.isfinite(path).all() or not np.array_equal(path[0], path[-1]):
                    raise ValueError(f"Loop {index} is not a finite closed polyline.")
                np.testing.assert_allclose(path[:, 0], 50, rtol=0, atol=1e-8)
            pairs[index] = old[:, 1:], new[:, 1:]
    np.testing.assert_allclose(vertices[:, 0], 50, rtol=0, atol=1e-8)
    return vertices[:, 1:], pairs


def draw_pair(ax, old, new):
    ax.plot(*new.T, color=SMOOTH, lw=2.1, solid_joinstyle="miter", zorder=4)
    ax.plot(*old.T, color=ORIGINAL, lw=1.55, dashes=(4, 2), dash_joinstyle="miter", zorder=5)


def main():
    plate, pairs = load_plate()
    # Ties go to the smaller archive loop index, then to the original vertex direction.
    witnesses = {
        index: max((furthest_vertex(old, new), furthest_vertex(new, old)), key=lambda sample: sample[0])
        for index, (old, new) in pairs.items()
    }
    index = max(witnesses, key=lambda key: witnesses[key][0])
    separation, a, b = witnesses[index]
    center = (a + b) / 2
    side = 10.0

    with plt.rc_context({"path.simplify": False, "font.family": "DejaVu Sans", "axes.unicode_minus": False}):
        fig, (overview, detail) = plt.subplots(1, 2, figsize=(9.5, 5.8))
        fig.subplots_adjust(left=0.085, right=0.97, bottom=0.225, top=0.80, wspace=0.27)
        fig.text(0.06, 0.94, "Biplanar Gy contour smoothing", fontsize=21, weight="bold", color=INK)
        fig.text(0.06, 0.888, "Before routing  ·  support at world x = +50 mm", fontsize=11, color="#476274")

        hull = plate[ConvexHull(plate).vertices]
        overview.fill(*hull.T, color="#f4f7f8", ec="#adbcc3", lw=0.9)
        for _, new in pairs.values():
            overview.plot(*new.T, color="#90a0a8", lw=0.62)
        old, new = pairs[index]
        draw_pair(overview, old, new)
        overview.set(xlim=(-105, 105), ylim=(-105, 105), xticks=(-100, -50, 0, 50, 100), yticks=(-100, -50, 0, 50, 100))
        overview.add_patch(Rectangle(center - side / 2, side, side, fill=False, ec=INK, lw=1.3, zorder=6))
        overview.annotate(
            "Detail",
            xy=center,
            xytext=(0.10, 0.88),
            textcoords="axes fraction",
            fontsize=9,
            weight="bold",
            color=INK,
            backgroundcolor="white",
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.0, "shrinkB": 7},
            zorder=7,
        )
        overview.set_title("Plate overview", loc="left", fontsize=13, weight="bold", color=INK, pad=10)
        draw_pair(detail, old, new)
        detail.plot([a[0], b[0]], [a[1], b[1]], "o-", color=INK, markersize=3.8, lw=1.2, zorder=6)
        detail.set(xlim=center[0] + np.array([-side / 2, side / 2]), ylim=center[1] + np.array([-side / 2, side / 2]))
        detail.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        detail.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        detail.set_title("10 × 10 mm detail", loc="left", fontsize=13, weight="bold", color=INK, pad=10)
        for ax in (overview, detail):
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("World y [mm]", color=INK)
            ax.set_ylabel("World z [mm]", color=INK)
            ax.tick_params(labelsize=8.5, colors=INK)
            for spine in ax.spines.values():
                spine.set_edgecolor("#a5b2b8")
        fig.legend(
            handles=[
                Line2D([], [], color=ORIGINAL, lw=1.55, dashes=(4, 2), label="Original contour"),
                Line2D([], [], color=SMOOTH, lw=2.1, label="Smoothed contour"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.163),
            frameon=False,
            ncol=2,
            fontsize=10,
        )
        fig.text(
            0.06,
            0.082,
            f"The marked paths differ by {separation:.3f} mm. Loop index {index} has the largest sampled separation on this plate.",
            fontsize=9,
            color=INK,
        )
        fig.text(
            0.06,
            0.044,
            "Distances compare saved vertices to opposing segments in both directions. Smoothing guards and fabrication are assessed separately.",
            fontsize=8.5,
            color="#476274",
        )
        fig.savefig(HERE / "smoothing-detail.png", dpi=170)
        plt.close(fig)
    print(
        f"Loop {index}, separation {separation:.8f} mm; detail center world (y, z) = ({center[0]:.5f}, {center[1]:.5f}) mm"
    )


if __name__ == "__main__":
    main()
