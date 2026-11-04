"""Compare two real X²−Y² fabrication-review centerlines. No display smoothing.

Pass old and new complete run directories to replace the compact comparison.
Without arguments, redraw the figure from the included arrays and summary.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


HERE = Path(__file__).resolve().parent
RADIUS_MM = 80.0


def turn_angle_p95(points_m, closed=True, step_m=0.003):
    """Direction change in 3D at equal arclength stations; preserve the real closure."""
    p = np.asarray(points_m)
    if closed and np.linalg.norm(p[-1] - p[0]) > 1e-12:
        p = np.vstack((p, p[0]))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0, arc[-1], max(4, round(arc[-1] / step_m)) + 1)
    samples = np.stack([np.interp(s, arc, axis) for axis in p.T], axis=1)
    tangent = np.diff(samples, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    dots = np.sum(tangent * np.roll(tangent, 1, axis=0), axis=1)
    return float(np.percentile(np.degrees(np.arccos(np.clip(dots if closed else dots[1:], -1, 1))), 95))


def read_case(directory):
    from coilgen.application import DesignService

    record = DesignService.load_run(directory, array_names=("review_conductor_0_points_m",))
    meta = record.metadata
    (body,) = meta["fabrication"]["paths"]
    points = record.arrays["review_conductor_0_points_m"]
    config = json.loads((Path(directory) / "request.json").read_text(encoding="utf8"))["config"]
    rows = meta["contour_smoothing"]
    shape = [int(x) for x in config["cylinder_mesh_parameter_list"][2:4]]
    assert meta["name"] == "Second-order shim — X²−Y²" and np.isclose(config["cylinder_mesh_parameter_list"][1], 0.08)
    detail = {
        "request_sha256": meta["request_sha256"],
        "metadata_sha256": meta["metadata_sha256"],
        "mesh": shape,
        "closed": body["closed"],
        "smoothing": rows,
        "current_A": meta["wire_fields"]["contour"]["at_stream_current"]["current_A"],
        "review_field_rms_percent": meta["fabrication"]["field_metrics"]["dsv_relative_residual_percent"],
        "review_turn_p95_deg_at_3mm": turn_angle_p95(points, body["closed"]),
        "sheet_accepted": meta["summary"]["sheet_accepted"],
        "review_geometry_ready": meta["fabrication"]["ready"],
        "review_min_bend_mm": body["diagnostics"]["sweep_final_min_bend_radius_m"] * 1000,
        "failure_reasons": meta["fabrication"]["failure_reasons"],
    }
    return points, detail


def draw():
    data = json.loads((HERE / "x2y2-review.json").read_text(encoding="utf8"))
    with np.load(HERE / "x2y2-review.npz", allow_pickle=False) as archive:
        curves = [archive[name] * 1000 for name in ("previous_points_m", "updated_points_m")]
    fig = plt.figure(figsize=(11, 9), facecolor="white", layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.27, 1.22, 0.85))
    head = fig.add_subplot(grid[0, :])
    head.set_axis_off()
    head.text(0, 0.70, "Second-order X²−Y² shim", weight="bold", size=20, color="#172d39")
    head.text(
        0,
        0.14,
        "Recorded conductor after routing and mounting  ·  identical views and scales",
        size=11,
        color="#45616b",
    )
    for col, (key, title, points) in enumerate(
        zip(("previous", "updated"), ("Previous gallery", "Updated gallery"), curves)
    ):
        stats = data[key]
        if stats["closed"] and np.linalg.norm(points[-1] - points[0]) > 1e-8:
            points = np.vstack((points, points[0]))
        ax = fig.add_subplot(grid[1, col])
        ax.set_title(
            f"{title}   ·   {stats['mesh'][0]} × {stats['mesh'][1]} support",
            loc="left",
            size=13,
            weight="bold",
            color="#172d39",
        )
        segments = np.stack((points[:-1], points[1:]), axis=1)
        depth = segments[:, :, 0].mean(axis=1)
        ax.fill_between([-RADIUS_MM, RADIUS_MM], -100, 100, color="#f0f4f4")
        for mask, color, lw in ((depth < 0, "#b9cbce", 0.85), (depth >= 0, "#176068", 1.35)):
            ax.add_collection(LineCollection(segments[mask, :, 1:], colors=color, linewidths=lw, capstyle="round"))
        ax.set(xlim=(-108, 108), ylim=(-106, 106), aspect="equal", xlabel="World y (mm)", ylabel="World z (mm)")
        ax.text(
            0.02,
            -0.22,
            f"Reviewed RMS field error: {stats['review_field_rms_percent']:.2f}%   ·   Drive: {stats['current_A']:.3f} A",
            transform=ax.transAxes,
            size=10,
            color="#304b57",
        )
        ax2 = fig.add_subplot(grid[2, col])
        phi = np.arctan2(points[:, 1], points[:, 0])
        developed = np.column_stack([RADIUS_MM * phi, points[:, 2]])
        pairs = np.stack((developed[:-1], developed[1:]), axis=1)
        pairs = pairs[np.abs(np.diff(phi)) < np.pi]
        ax2.add_collection(LineCollection(pairs, colors="#176068", linewidths=1.4, capstyle="round"))
        ax2.set(xlim=(-44, 44), ylim=(12, 76), aspect="equal", xlabel="Azimuth × 80 mm (mm)", ylabel="World z (mm)")
        ax2.set_title("Same front lobe, developed onto the plane", loc="left", fontsize=10, color="#304b57")
        for axes in (ax, ax2):
            axes.spines[["top", "right"]].set_visible(False)
            axes.spines[["bottom", "left"]].set_color("#a8b6b8")
            axes.tick_params(colors="#48616c", labelsize=8)
    fig.savefig(HERE / "x2y2-reviewed-windings.png", dpi=155, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    args = parser.parse_args()
    if args.runs:
        if len(args.runs) != 2:
            parser.error("Provide both the previous and updated result directories.")
        cases = [read_case(run) for run in args.runs]
        np.savez_compressed(HERE / "x2y2-review.npz", previous_points_m=cases[0][0], updated_points_m=cases[1][0])
        (HERE / "x2y2-review.json").write_text(
            json.dumps(dict(previous=cases[0][1], updated=cases[1][1]), indent=2) + "\n", encoding="utf8"
        )
    draw()
