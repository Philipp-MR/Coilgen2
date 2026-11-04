"""Draw the saved, physical review centerlines; do not smooth or refit them for display."""

from pathlib import Path
import csv
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.spatial import ConvexHull


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Fixed oblique orthographic projection in the world frame for every channel.
CAMERA = np.array([0.70, -0.88, 0.56])
CAMERA /= np.linalg.norm(CAMERA)
RIGHT = np.cross([0.0, 0.0, 1.0], CAMERA)
RIGHT /= np.linalg.norm(RIGHT)
PROJECTION = np.stack([RIGHT, np.cross(CAMERA, RIGHT), CAMERA]).T
GALLERY_ORDER = (
    "cylinder_gx",
    "cylinder",
    "cylinder_gz",
    "halbach_gx",
    "halbach_gy",
    "halbach_gz",
    "biplanar_gx",
    "biplanar",
    "biplanar_gz",
    "shielded_gy",
    "shim_z0",
    "shim_z2",
    "shim_xz",
    "shim_yz",
    "shim_x2_y2",
    "shim_xy",
    "conical_harmonic",
    "cylindrical_xyz_gradient",
    "biplanar_gradient_shim",
)


def draw(ax, row):
    archive = (
        HERE / row["centerlines_file"]
        if row.get("centerlines_file")
        else ROOT / row["directory"] / "scientific_result.npz"
    )
    with np.load(archive, allow_pickle=False) as data:
        supports = [data[k] * 1000 @ PROJECTION for k in data if re.fullmatch(r"support_\d+_vertices_m", k)]
        curves = [data[k] * 1000 @ PROJECTION for k in data if re.fullmatch(r"review_conductor_\d+_points_m", k)]
    if not supports or not curves:
        raise ValueError(f"No physical support or review path: {row['channel']}")
    for p in supports:
        hull = p[ConvexHull(p[:, :2]).vertices]
        ax.fill(hull[:, 0], hull[:, 1], color="#e8f1f2", ec="#c8d8da", lw=0.6)
    segments = np.concatenate([np.stack((p[:-1], p[1:]), axis=1) for p in curves])
    depth = segments[:, :, 2].mean(axis=1)
    order = depth.argsort()
    fraction = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-12)
    far, near = np.array([0.69, 0.47, 0.32]), np.array([0.46, 0.20, 0.07])
    colors = far + fraction[:, None] * (near - far)
    ax.add_collection(LineCollection(segments[order, :, :2], colors=colors[order], linewidths=0.78))
    world = np.vstack(supports + curves)
    mid = (world.max(axis=0) + world.min(axis=0))[:2] / 2
    span = np.ptp(world[:, :2], axis=0)
    ax.set_xlim(mid[0] - max(span[0], 10) * 0.57, mid[0] + max(span[0], 10) * 0.57)
    ax.set_ylim(mid[1] - max(span[1], 10) * 0.58, mid[1] + max(span[1], 10) * 0.58)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(textwrap.fill(row["channel"], 32), fontsize=10, loc="left", color="#162b36", fontweight="bold")
    x = ax.get_xlim()[0]
    y = ax.get_ylim()[0] + 2
    ax.plot([x, x + 25], [y, y], color="#2c4650", lw=1.5)
    ax.text(x + 29, y, "25 mm", fontsize=7, va="center", color="#2c4650")
    accepted = row["smoothed_fabrication_passed"] == "True"
    ax.text(
        0.0,
        -0.09,
        f"{row['smoothed']}/{row['raw_contours']} smoothed   ·   fabrication {'passed' if accepted else 'failed'}",
        transform=ax.transAxes,
        fontsize=8,
        color="#237254" if accepted else "#804d32",
    )


def main():
    with (HERE / "smoothing.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 29:
        raise ValueError("The archived gallery requires 29 evaluated channels.")

    def family(row):
        if row["template"] in {"biplanar_gradient_shim", "cylindrical_xyz_gradient"}:
            return "systems"
        if row["template"].startswith("shim_") or row["template"] == "conical_harmonic":
            return "shims"
        return "gradients"

    for group, title in (
        ("gradients", "Gradient examples"),
        ("shims", "Shim examples"),
        ("systems", "Channels of the two example systems"),
    ):
        subset = sorted(
            (r for r in rows if family(r) == group),
            key=lambda r: (GALLERY_ORDER.index(r["template"]), int(Path(r["centerlines_file"]).stem.split("__")[-1])),
        )
        nrows = (len(subset) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(12, 1.6 + 2.93 * nrows), squeeze=False)
        fig.subplots_adjust(
            top=1 - 1.68 / fig.get_figheight(), bottom=0.065, hspace=0.57, wspace=0.26, left=0.055, right=0.98
        )
        fig.text(0.055, 0.96, title, size=19, weight="bold", color="#162b36")
        fig.text(
            0.055,
            0.925,
            "Saved conductor centerlines  ·  same world view  ·  each panel has its own scale",
            size=10,
            color="#3b5660",
        )
        for ax, row in zip(axes.flat, subset):
            draw(ax, row)
        for ax in list(axes.flat)[len(subset) :]:
            ax.set_axis_off()
        fig.text(
            0.055,
            0.016,
            "Counts refer to accepted contour fits before routing. Fabrication refers to the recorded configured checks; no case has construction release.",
            size=8.5,
            color="#3b5660",
        )
        fig.savefig(HERE / f"windings-{group}.png", dpi=155)
        plt.close(fig)


if __name__ == "__main__":
    main()
