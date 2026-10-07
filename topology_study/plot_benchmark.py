"""Static figures for the isolated topology study (all geometry is in z=0).

Run after benchmark.py. Uses only study outputs, NumPy, and Matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from benchmark import bezier_points, intersections


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
INK = "#173535"
TEAL = "#087f8c"
ORANGE = "#d9861c"
RED = "#bf3e35"
GREEN = "#226e56"
PAPER = "#fbfcfa"


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.titlecolor": INK,
                         "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
                         "svg.fonttype": "none"})


def plane(ax, xlim: tuple, ylim: tuple, zdepth: float) -> None:
    ax.set_facecolor(PAPER)
    # Every route lies above this single planar patch; explicit ordering avoids
    # Matplotlib's collection-vs-line depth sort hiding the wire under the plane.
    ax.computed_zorder = False
    corners = [[xlim[0], ylim[0], zdepth], [xlim[1], ylim[0], zdepth],
               [xlim[1], ylim[1], zdepth], [xlim[0], ylim[1], zdepth]]
    ax.add_collection3d(Poly3DCollection([corners], facecolor="#eff3f0",
                                        edgecolor="#d6e0db", linewidth=.8, zorder=0))
    for x in np.linspace(*xlim, 9):
        ax.plot([x, x], ylim, [zdepth, zdepth], color="#dfe7e1", lw=.55, zorder=1)
    for y in np.linspace(*ylim, 9):
        ax.plot(xlim, [y, y], [zdepth, zdepth], color="#dfe7e1", lw=.55, zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(-abs(zdepth)*4, abs(zdepth)*4)
    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], (xlim[1]-xlim[0])*.10))
    ax.view_init(elev=51, azim=-68)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def line3(ax, points: np.ndarray, color: str, width: float = 2.2) -> None:
    ax.plot(points[:, 0], points[:, 1], np.zeros(len(points)), color=color,
            lw=width, solid_capstyle="round", zorder=5)


def current_arrow(ax, center: np.ndarray, radius: float, theta: float,
                  direction: int, length: float) -> None:
    point = center + radius * np.array([np.cos(theta), np.sin(theta)])
    tangent = direction * np.array([-np.sin(theta), np.cos(theta)])
    # Quiver puts its arrowhead at the endpoint; centering keeps the arrow on the arc.
    base = point - length*.5*tangent
    ax.quiver(base[0], base[1], .004, tangent[0]*length, tangent[1]*length,
              0, color=INK, lw=1.7, arrow_length_ratio=.42, zorder=15)


def export(fig, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_nesting_comparison(data: dict) -> None:
    fig = plt.figure(figsize=(14, 7.7))
    fig.suptitle("One set of directed contours. Two visit orders.", fontsize=21,
                 fontweight="bold", y=.97)
    fig.text(.5, .921, "Same cut positions and tangent-matched connector rule; every retained arc remains CCW.",
             ha="center", fontsize=11.5)
    for position, key, title, color in ((1, "best", "ADJACENT INWARD ORDER", ORANGE),
                                      (2, "scrambled", "SCRAMBLED ORDER", RED)):
        row = data[key]
        ax = fig.add_subplot(1, 2, position, projection="3d")
        plane(ax, (-53, 59), (-53, 53), -.8)
        for arc in row["arcs_mm"]:
            line3(ax, np.array(arc), TEAL, 2.25)
        for bridge in row["bridges_mm"]:
            line3(ax, np.array(bridge), color, 2.55)
        for radius, theta in ((42., 2.0), (30., 2.0), (18., 2.0), (36., 4.3)):
            current_arrow(ax, np.zeros(2), radius, theta, 1, 5.2)
        a = np.array(row["arcs_mm"][0][0])
        b = np.array(row["arcs_mm"][-1][-1])
        for point, label, offset in ((a, "A", (5., 5.)), (b, "B", (-6., -6.))):
            ax.scatter(*point, .02, s=52, color=PAPER, edgecolor=INK, linewidth=1.4,
                       depthshade=False, zorder=20)
            ax.text(point[0]+offset[0], point[1]+offset[1], .05, label,
                    color=INK, fontsize=13, fontweight="bold", zorder=30)
        if row["connector_connector_hits"]:
            hits = np.array([hit["point_mm"] for hit in row["connector_connector_hits"]])
            ax.scatter(hits[:, 0], hits[:, 1], np.full(len(hits), .035), marker="x",
                       color=RED, s=26, linewidth=1.1, depthshade=False, zorder=25)
        ax.text2D(.5, .97, title, transform=ax.transAxes, ha="center", fontsize=13,
                  fontweight="bold", color=GREEN if key == "best" else RED)
        ax.text2D(.5, .91, f'{row["connector_length_mm"]:.2f} mm connectors  |  {row["total_crossings"]} crossings',
                  transform=ax.transAxes, ha="center", fontsize=12)
        order = " → ".join(str(int(radius)) for radius in row["order_radii_mm"])
        ax.text2D(.5, .065, f"Visit radii: {order} mm", transform=ax.transAxes,
                  ha="center", fontsize=10.8)
    fig.legend(handles=[Line2D([0], [0], color=TEAL, lw=2.5, label="Retained contour"),
                        Line2D([0], [0], color=ORANGE, lw=2.5, label="Connector"),
                        Line2D([0], [0], color=INK, marker=">", lw=0, label="Current direction")],
               loc="lower center", bbox_to_anchor=(.5, .125), ncol=3, frameon=False)
    fig.text(.5, .09, "A = outer start   ·   B = inner finish   ·   Both routes are open; terminal return leads are not shown.",
             ha="center", fontsize=10.5)
    fig.text(.5, .049, "Adjacent order is shortest and uniquely crossing-free among 120 fixed-port / fixed-curve orders.",
             ha="center", fontsize=10.5)
    fig.text(.5, .021, "Equal-scale 3D views of z = 0.  Counts exclude intended joins.  Wire clearance and bend radius are not assessed.",
             ha="center", fontsize=9.5, color="#536761")
    fig.subplots_adjust(left=.025, right=.98, top=.89, bottom=.19, wspace=.00)
    export(fig, "nesting_comparison")


def directed_loop(center: np.ndarray, radius: float, gap_angle: float,
                  direction: int, half_gap: float = .23) -> dict:
    theta = gap_angle + direction * np.linspace(half_gap, 2*np.pi-half_gap, 700)
    points = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    start_tangent = direction * np.array([-np.sin(theta[0]), np.cos(theta[0])])
    end_tangent = direction * np.array([-np.sin(theta[-1]), np.cos(theta[-1])])
    return {"points": points, "center": center, "radius": radius,
            "gap_angle": gap_angle, "direction": direction,
            "start_tangent": start_tangent, "end_tangent": end_tangent}


def directed_bridge(source: dict, target: dict) -> np.ndarray:
    end, start = source["points"][-1], target["points"][0]
    handle = .20*np.linalg.norm(start-end)
    control = np.array([end, end + handle*source["end_tangent"],
                        start-handle*target["start_tangent"], start])
    return bezier_points(control, 501)


def draw_splice_orientation() -> dict:
    configurations = [
        ("Nested · same circulation", "parallel", "nested", 1, True),
        ("Nested · opposite circulation", "opposite", "nested", -1, False),
        ("Side by side · same circulation", "opposite", "side", 1, False),
        ("Side by side · opposite circulation", "parallel", "side", -1, True),
    ]
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle("A closed local splice depends on circulation", fontsize=21,
                 fontweight="bold", y=.974)
    fig.text(.5, .938, "Connect each directed exit to the other loop's entry. Both gaps must be reclosed.",
             ha="center", fontsize=11.5)
    validation = {}
    for index, (title, facing, layout, second_direction, invalid) in enumerate(configurations, 1):
        if layout == "nested":
            loops = [directed_loop(np.zeros(2), 3., 0., 1),
                     directed_loop(np.zeros(2), 1.6, 0., second_direction)]
            xlim, ylim = (-3.7, 4.1), (-3.7, 3.7)
        else:
            loops = [directed_loop(np.array([-2.5, 0.]), 1.75, 0., 1),
                     directed_loop(np.array([2.5, 0.]), 1.75, np.pi, second_direction)]
            xlim, ylim = (-4.9, 4.9), (-3.2, 3.2)
        bridges = [directed_bridge(loops[0], loops[1]), directed_bridge(loops[1], loops[0])]
        hits = intersections(*bridges)
        assert len(hits) == int(invalid), (title, hits)
        # Check each bridge against both retained arcs, excluding its own endpoints.
        unexpected_arc_hits = []
        for bridge in bridges:
            for loop in loops:
                for point in intersections(bridge, loop["points"]):
                    if min(np.linalg.norm(point-bridge[0]), np.linalg.norm(point-bridge[-1])) > 1e-7:
                        unexpected_arc_hits.append(point)
        assert not unexpected_arc_hits, (title, unexpected_arc_hits)
        validation[title] = {"bridge_crossings": len(hits), "unexpected_arc_crossings": 0}
        ax = fig.add_subplot(2, 2, index, projection="3d")
        plane(ax, xlim, ylim, -.06)
        for loop in loops:
            line3(ax, loop["points"], TEAL, 2.5)
            for theta in (1.8, 4.2):
                current_arrow(ax, loop["center"], loop["radius"], theta,
                              loop["direction"], .48)
        for bridge in bridges:
            line3(ax, bridge, RED if invalid else ORANGE, 2.8)
            ax.scatter(bridge[[0, -1], 0], bridge[[0, -1], 1], [0., 0.],
                       s=17, color=INK, depthshade=False, zorder=15)
        if len(hits):
            ax.scatter(hits[:, 0], hits[:, 1], np.full(len(hits), .02), marker="x",
                       color=RED, s=180, linewidth=2.3, depthshade=False, zorder=30)
        ax.text2D(.5, .99, title, transform=ax.transAxes, ha="center", fontsize=13.5,
                  fontweight="bold")
        ax.text2D(.5, .92, f"Facing tangents {facing}", transform=ax.transAxes,
                  ha="center", fontsize=11.5)
        ax.text2D(.5, .05, "1 crossing · needs another routing option" if invalid else "0 crossings · two separate bridge tracks",
                  transform=ax.transAxes, ha="center", fontsize=11,
                  fontweight="bold", color=RED if invalid else GREEN)
    fig.legend(handles=[Line2D([0], [0], color=TEAL, lw=2.5, label="Directed retained loop"),
                        Line2D([0], [0], color=ORANGE, lw=2.5, label="Compatible bridges"),
                        Line2D([0], [0], color=RED, marker="x", lw=2, label="Crossing pairing")],
               loc="lower center", bbox_to_anchor=(.5, .064), ncol=3, frameon=False)
    fig.text(.5, .045, "Scope: closed splice with two bridges in the pictured fixed local corridor, on one surface layer.",
             ha="center", fontsize=10.5)
    fig.text(.5, .019, "An open spiral uses a different connection problem. All diagrams show a tilted planar surface; conductor width is omitted.",
             ha="center", fontsize=10, color="#536761")
    fig.subplots_adjust(left=.025, right=.98, top=.915, bottom=.12, wspace=.0, hspace=.03)
    export(fig, "splice_orientation")
    return validation


def main() -> None:
    style()
    FIGURES.mkdir(exist_ok=True)
    data = json.loads((ROOT / "benchmark_results.json").read_text(encoding="utf-8"))
    draw_nesting_comparison(data)
    validation = draw_splice_orientation()
    print(json.dumps(validation, indent=2))
    print("Created nesting_comparison and splice_orientation as PNG and SVG.")


if __name__ == "__main__":
    main()
