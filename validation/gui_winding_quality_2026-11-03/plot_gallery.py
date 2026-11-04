"""Plot the recorded final conductors from full_gallery.py. No display fitting or smoothing.

Run after all gallery channels finish, or use --allow-partial --output-dir PATH to
preview only the available channels. All figures are rendered offscreen.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from scipy.spatial import ConvexHull


HERE = Path(__file__).resolve().parent
FAMILIES = {
    "gradients": (
        "Gradient windings",
        (
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
        ),
    ),
    "shims": (
        "Shim windings",
        ("shim_z0", "shim_z2", "shim_xz", "shim_yz", "shim_x2_y2", "shim_xy", "conical_harmonic"),
    ),
    "systems": ("Windings of the two coil systems", ("cylindrical_xyz_gradient", "biplanar_gradient_shim")),
}
SYSTEM_COUNTS = {"cylindrical_xyz_gradient": 3, "biplanar_gradient_shim": 9}
CAMERA = np.array([0.70, -0.88, 0.56])
CAMERA /= np.linalg.norm(CAMERA)
RIGHT = np.cross([0.0, 0.0, 1.0], CAMERA)
RIGHT /= np.linalg.norm(RIGHT)
PROJECTION = np.stack([RIGHT, np.cross(CAMERA, RIGHT), CAMERA]).T
INK, MUTED, LINE = "#18303c", "#49616a", "#d6e1e2"


def read_case(root, key):
    row = json.loads((root / f"{key}.json").read_text(encoding="utf-8"))
    if f"{row['template']}__{row['channel_index']}" != key:
        raise ValueError(f"Channel identity mismatch: {key}")
    archive = root / f"{key}.npz"
    if sha256(archive.read_bytes()).hexdigest() != row["geometry_sha256"]:
        raise ValueError(f"Changed reviewed geometry: {archive}")
    with np.load(archive, allow_pickle=False) as data:

        def project(pattern):
            result = []
            for name in sorted(data.files):
                if re.fullmatch(pattern, name):
                    points = data[name]
                    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
                        raise ValueError(f"Invalid world coordinates in {key}: {name}")
                    result.append((1000 * points) @ PROJECTION)
            return result

        supports = project(r"support_\d+_vertices_m")
        paths = project(r"review_conductor_\d+_points_m")
    if not supports or not paths or len(paths) != row["checked_bodies"]:
        raise ValueError(f"Incomplete final conductors or support: {key}")
    return row, supports, paths


def draw_winding(ax, supports, paths):
    for p in supports:
        h = p[ConvexHull(p[:, :2]).vertices]
        ax.fill(h[:, 0], h[:, 1], fc="#eaf1f1", ec="#bbcdd0", lw=0.7)
    segments = np.concatenate([np.stack([p[:-1], p[1:]], axis=1) for p in paths])
    depth = segments[:, :, 2].mean(axis=1)
    fraction = (depth - depth.min()) / max(np.ptp(depth), 1e-12)
    far, near = np.array([0.70, 0.47, 0.31]), np.array([0.40, 0.17, 0.055])
    shades = far + fraction[:, None] * (near - far)
    order = np.argsort(depth, kind="stable")
    ax.add_collection(LineCollection(segments[order, :, :2], colors=shades[order], linewidths=0.90, capstyle="round"))
    xy = np.concatenate(supports + paths)[:, :2]
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(high - low, 10.0)
    ax.set_xlim(low[0] - 0.08 * span[0], high[0] + 0.08 * span[0])
    ax.set_ylim(low[1] - 0.19 * span[1], high[1] + 0.035 * span[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.add_artist(
        AnchoredSizeBar(
            ax.transData,
            25,
            "25 mm",
            "lower left",
            pad=0.0,
            borderpad=0.1,
            sep=1,
            size_vertical=0.9,
            color=MUTED,
            frameon=False,
            fontproperties={"size": 7},
        )
    )


def render(family, title, cases, total, target):
    rows = (len(cases) + 2) // 3
    height = 1.74 + 3.64 * rows
    fig = plt.figure(figsize=(12, height), facecolor="white")
    # Physical header/footer margins keep every page and partial preview readable.
    fig.text(0.05, 1 - 0.36 / height, "COILGEN  /  GUI WINDING REVIEW", color=MUTED, fontsize=8.5, weight="bold")
    fig.text(0.05, 1 - 0.82 / height, title, color=INK, fontsize=22, weight="bold")
    count = f"{len(cases)} channels" if len(cases) == total else f"Preview · {len(cases)} of {total} channels"
    fig.text(
        0.05,
        1 - 1.10 / height,
        count + "   ·   Final reviewed centerlines   ·   Common orthographic world view",
        color=MUTED,
        fontsize=10,
    )
    grid = fig.add_gridspec(
        rows, 3, left=0.05, right=0.97, top=1 - 1.40 / height, bottom=0.65 / height, hspace=0.18, wspace=0.14
    )
    for index, (row, supports, paths) in enumerate(cases):
        tile = grid[index // 3, index % 3].subgridspec(3, 1, height_ratios=[0.16, 0.67, 0.17], hspace=0.02)
        label = fig.add_subplot(tile[0])
        label.set_axis_off()
        label.plot([0, 1], [0.98, 0.98], lw=0.9, color=LINE, transform=label.transAxes)
        label.text(
            0.0,
            0.62,
            fill(row["channel"], 31),
            fontsize=10.6,
            weight="bold",
            color=INK,
            transform=label.transAxes,
            va="center",
        )
        draw_winding(fig.add_subplot(tile[1]), supports, paths)
        metrics = fig.add_subplot(tile[2])
        metrics.set_axis_off()
        loops = row["smoothed"] + row["original_retained"]
        metrics.text(
            0.0,
            0.68,
            f"{row['smoothed']}/{loops} contour fits   ·   {row['review_rms_percent']:.2f}% field RMS",
            fontsize=8.9,
            color=INK,
            transform=metrics.transAxes,
        )
        verdict = {True: "passed", False: "failed", None: "unavailable"}[row["fabrication"]]
        metrics.text(
            0.0,
            0.21,
            f"Fabrication {verdict}   ·   P95 turn {row['review_p95_deg']:.1f}°",
            fontsize=8.9,
            color="#216647" if verdict == "passed" else "#80442b",
            transform=metrics.transAxes,
        )
    fig.text(
        0.05,
        0.30 / height,
        "Fits are counted before routing. Field RMS is relative error on the design DSV at the recorded current. "
        "P95 turn uses 3 mm stations on the final wire.",
        fontsize=8.1,
        color=MUTED,
    )
    release = sum(x[0]["construction_release"] is True for x in cases)
    fig.text(
        0.05,
        0.12 / height,
        f"Each panel has its own scale. Fabrication refers to the configured checks. "
        f"Construction release recorded for {release}/{len(cases)} channels shown.",
        fontsize=8.1,
        color=MUTED,
    )
    fig.savefig(target / f"windings-{family}.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=HERE / "reviewed_centerlines")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    keys = {
        group: [f"{name}__{i}" for name in names for i in range(SYSTEM_COUNTS.get(name, 1))]
        for group, (_, names) in FAMILIES.items()
    }
    expected = {key for group in keys.values() for key in group}
    available = {p.stem for p in args.records.glob("*.json")}
    if not available or available - expected or (expected - available and not args.allow_partial):
        raise ValueError(
            f"Expected {len(expected)} gallery records. Found {len(available)} under {args.records}. "
            "Use --allow-partial to preview available records."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for family, group in keys.items():
        cases = [read_case(args.records, key) for key in group if key in available]
        if cases:
            render(family, FAMILIES[family][0], cases, len(group), args.output_dir)


if __name__ == "__main__":
    main()
