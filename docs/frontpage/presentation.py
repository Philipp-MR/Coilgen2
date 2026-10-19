"""Render the GitHub gallery from verified sheets and conditioned contour paths.

Run ``python docs/frontpage/presentation.py`` from the repository root. This
script never solves, refits currents, or modifies the source NPZ evidence.
Cylinder triangles are unwrapped separately and duplicated at the seam; no
interpolation connects opposite sides of the periodic chart.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors, tri  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402


OUTPUT = Path(__file__).resolve().parent
NAVY = "#091b2b"
INK = "#15324a"
MUTED = "#62788b"
PAPER = "#f6f8fb"
COPPER = "#ffad72"
TEAL = "#47dec9"
SHEET = "#176c9d"
WIRE = "#b34d20"
NAMES = ("Cylindrical Gy", "Biplanar Gz")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#cad5df",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
            "svg.hashsalt": "CoilGen2-gallery-v1",
            "savefig.facecolor": "auto",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", dpi=260, facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT / f"{name}.svg", metadata={"Date": None}, facecolor=fig.get_facecolor())
    svg = OUTPUT / f"{name}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)


def _metrics(metrics: dict | list) -> list:
    return metrics["cases"] if isinstance(metrics, dict) else metrics


def _check(cases: list[dict], metrics: dict | list) -> list:
    info = _metrics(metrics)
    if len(cases) != 2 or len(info) != 2:
        raise ValueError("The gallery requires its two documented numerical examples.")
    for arrays, case in zip(cases, info):
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"Nonfinite gallery data: {case['name']}.")
        target = arrays["target_T"]
        if target.shape != arrays["sheet_field_T"].shape or len(target) != case["independent_point_count"]:
            raise ValueError("Field evidence and independent sample count disagree.")
        for kind in ("sheet", "contour"):
            rms = 100 * np.linalg.norm(arrays[f"{kind}_field_T"][:, 2] - target[:, 2]) / np.linalg.norm(target[:, 2])
            if not np.isclose(rms, case[f"{kind}_relative_rms_Bz_error_percent"], rtol=1e-10, atol=1e-12):
                raise ValueError("Saved field arrays and published RMS evidence disagree.")
    return info


def _dsv(ax) -> None:
    angle = np.linspace(0, 2 * np.pi, 150)
    for plane in range(3):
        circle = np.zeros((len(angle), 3))
        circle[:, (plane + 1) % 3] = 45 * np.cos(angle)
        circle[:, (plane + 2) % 3] = 45 * np.sin(angle)
        ax.plot(*circle.T, color="#a6e9f2", linewidth=0.8, alpha=0.42)


def _geometry(ax, arrays: dict, case: dict, accent: str, cylinder: bool) -> None:
    ax.set_facecolor(NAVY)
    for index in range(case["part_count"]):
        vertices = arrays[f"part_{index}_vertices_m"] * 1000
        faces = arrays[f"part_{index}_faces"]
        support = Poly3DCollection(
            vertices[faces], facecolor="#31516a", edgecolor="#426178", linewidth=0.18, alpha=0.09, zsort="average"
        )
        ax.add_collection3d(support)
        paths = [value * 1000 for key, value in arrays.items() if key.startswith(f"part_{index}_contour_")]
        # A separate artist per contour preserves every actual opening and closure.
        for path in paths:
            ax.plot(*path.T, color=accent, linewidth=1.6, solid_capstyle="round", alpha=0.97)
        if cylinder:
            angle = np.linspace(0, 2 * np.pi, 170)
            for z in (-150, 150):
                ax.plot(
                    120 * np.cos(angle),
                    120 * np.sin(angle),
                    np.full_like(angle, z),
                    color="#7fa0b5",
                    linewidth=0.8,
                    alpha=0.5,
                )
        else:
            z = vertices[:, 2].mean()
            square = np.array([[-150, -150, z], [150, -150, z], [150, 150, z], [-150, 150, z], [-150, -150, z]])
            ax.plot(*square.T, color="#7fa0b5", linewidth=0.8, alpha=0.5)
    _dsv(ax)
    limits = (-165, 165)
    ax.set(xlim=limits, ylim=limits, zlim=limits)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20 if cylinder else 23, azim=-53 if cylinder else -51)
    ax.set_proj_type("persp", focal_length=1.1)
    ax.set_axis_off()


def render_geometry(cases: list[dict], metrics: dict | list) -> None:
    """Render two actual contour families; there are no invented feeds or routes."""
    info = _check(cases, metrics)
    _style()
    fig = plt.figure(figsize=(18, 10), facecolor=NAVY)
    fig.text(0.055, 0.945, "COILGEN2  /  INVERSE MAGNETIC DESIGN", fontsize=12, color="#8ab0c9", weight="bold")
    fig.text(0.055, 0.875, "Turn field targets into winding patterns.", fontsize=30, color="#f1f7fd", weight="bold")
    fig.text(
        0.055,
        0.827,
        "Two constrained minimum-power solutions. Real geometry. Original solved currents.",
        fontsize=15,
        color="#adc1d1",
    )
    for i, (arrays, case, accent) in enumerate(zip(cases, info, (COPPER, TEAL))):
        left = 0.035 + i * 0.49
        ax = fig.add_axes((left, 0.18, 0.45, 0.64), projection="3d")
        _geometry(ax, arrays, case, accent, i == 0)
        fig.text(left + 0.02, 0.755, f"0{i + 1}", fontsize=13, color=accent, weight="bold")
        fig.text(left + 0.065, 0.752, NAMES[i], fontsize=21, color="#f1f7fd", weight="bold")
        dimensions = "Ø 240 × 300 mm support" if i == 0 else "300 × 300 mm sheets · 200 mm separation"
        fig.text(left + 0.02, 0.704, dimensions, fontsize=12, color="#9bb4c7")
        fig.text(
            left + 0.02,
            0.155,
            f"{case['contour_relative_rms_Bz_error_percent']:.2f}%",
            fontsize=31,
            color=accent,
            weight="bold",
        )
        fig.text(left + 0.15, 0.17, "contour RMS field error", fontsize=13, color="#d2e1eb")
        fig.text(
            left + 0.15,
            0.14,
            f"{case['independent_point_count']} ROI acceptance points · {case['unconnected_contour_count']} contours",
            fontsize=12,
            color="#8eabba",
        )
    fig.add_artist(plt.Line2D([0.515, 0.515], [0.14, 0.74], transform=fig.transFigure, color="#244256", linewidth=1))
    fig.text(
        0.055, 0.075, "10 mT/m target  ·  Ø 90 mm validation volume  ·  no current refit", fontsize=13, color="#bfd3df"
    )
    fig.text(
        0.055,
        0.037,
        "Smoothed, filtered unconnected contours; original currents retained. Routing and fabrication validation remain separate.",
        fontsize=11,
        color="#8eabba",
    )
    _save(fig, "coil_designs")


def _unwrap_triangles(
    vertices: np.ndarray, faces: np.ndarray, values: np.ndarray
) -> tuple[tri.Triangulation, np.ndarray]:
    """Chart P1 triangles periodically, duplicating only triangles crossing 0/2π."""
    radius = float(np.linalg.norm(vertices[:, :2], axis=1).mean())
    angle = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2 * np.pi)
    xy, scalar = [], []
    for face in faces:
        theta = angle[face].copy()
        shifts = (0,)
        if np.ptp(theta) > np.pi:
            theta[theta > np.pi] -= 2 * np.pi
            shifts = (0, 2 * np.pi)
        for shift in shifts:
            xy.extend(np.column_stack(((theta + shift) * radius * 1000, vertices[face, 2] * 1000)))
            scalar.extend(values[face])
    xy = np.asarray(xy)
    return tri.Triangulation(xy[:, 0], xy[:, 1], np.arange(len(xy)).reshape(-1, 3)), np.asarray(scalar)


def _chart_contours(ax, arrays: dict, prefix: str, cylinder: bool) -> None:
    for key, path in arrays.items():
        if not key.startswith(f"{prefix}_contour_"):
            continue
        if cylinder:
            # Contour points can lie inside faceted triangles, so use support radius.
            radius = np.linalg.norm(arrays[f"{prefix}_vertices_m"][:, :2], axis=1).mean()
            angle = np.mod(np.arctan2(path[:, 1], path[:, 0]), 2 * np.pi)
            chart = np.column_stack((angle * radius * 1000, path[:, 2] * 1000))
            cuts = np.flatnonzero(np.abs(np.diff(angle)) > np.pi) + 1
            for segment in np.split(chart, cuts):
                if len(segment) > 1:
                    ax.plot(*segment.T, color=INK, linewidth=0.55, alpha=0.6)
        else:
            ax.plot(*(path[:, :2] * 1000).T, color=INK, linewidth=0.55, alpha=0.6)


def render_stream_function(cases: list[dict], metrics: dict | list) -> None:
    """Show solved P1 stream functions in physical chart coordinates and amperes."""
    info = _check(cases, metrics)
    _style()
    fig = plt.figure(figsize=(18, 8.8), facecolor=PAPER)
    fig.text(0.055, 0.92, "The surface is the design space.", fontsize=29, weight="bold")
    fig.text(
        0.055,
        0.865,
        "Solved current potential with retained, smoothed winding paths overlaid. Before routing.",
        fontsize=14,
        color=MUTED,
    )
    panels = ((0, 0, (0.06, 0.29, 0.405, 0.43)), (1, 0, (0.535, 0.29, 0.185, 0.43)), (1, 1, (0.77, 0.29, 0.185, 0.43)))
    for case_index, part_index, bounds in panels:
        arrays, case = cases[case_index], info[case_index]
        prefix = f"part_{part_index}"
        vertices = arrays[f"{prefix}_vertices_m"]
        faces = arrays[f"{prefix}_faces"]
        sf = arrays[f"{prefix}_stream_function_A"]
        limit = max(np.max(np.abs(arrays[f"part_{i}_stream_function_A"])) for i in range(case["part_count"]))
        ax = fig.add_axes(bounds)
        if case_index == 0:
            triangulation, potential = _unwrap_triangles(vertices, faces, sf)
            ax.set_xlim(0, 2 * np.pi * 120)
            ax.set_xticks([0, 250, 500, 750])
            ax.set_xlabel("Circumferential distance Rφ [mm]", labelpad=10)
            ax.set_ylabel("z [mm]", labelpad=8)
            title = "Cylindrical Gy · periodic chart"
        else:
            triangulation = tri.Triangulation(vertices[:, 0] * 1000, vertices[:, 1] * 1000, faces)
            potential = sf
            ax.set(xlim=(-150, 150), ylim=(-150, 150), xlabel="x [mm]", ylabel="y [mm]")
            ax.set_xticks([-150, 0, 150])
            ax.set_yticks([-150, 0, 150])
            title = f"Biplanar Gz · z = {vertices[:, 2].mean() * 1000:+.0f} mm"
        image = ax.tripcolor(
            triangulation,
            potential,
            shading="gouraud",
            cmap="RdBu_r",
            norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            rasterized=True,
        )
        _chart_contours(ax, arrays, prefix, case_index == 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(-150, 150)
        ax.set_title(title, fontsize=15, pad=18)
        ax.tick_params(labelsize=11)
        bar_ax = fig.add_axes((bounds[0] + 0.025, 0.18, bounds[2] - 0.05, 0.018))
        bar = fig.colorbar(image, cax=bar_ax, orientation="horizontal")
        bar.set_label("Stream function ψ [A]", labelpad=8, fontsize=12)
        bar.ax.tick_params(labelsize=10)
        bar.outline.set_visible(False)
    fig.text(
        0.055,
        0.065,
        "Piecewise-linear values on the actual support meshes. Periodic triangles are duplicated at the cylinder seam; no seam-crossing interpolation.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.032,
        f"Color scales differ by geometry. Original turn current: cylinder {info[0]['contour_currents_A'][0]:.2f} A; biplanar sheets {info[1]['contour_currents_A'][0]:.2f} A. Smoothed paths differ from the raw level sets.",
        fontsize=11,
        color=MUTED,
    )
    _save(fig, "stream_function")


def _clean_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5ebf0", linewidth=0.65, zorder=0)
    ax.tick_params(labelsize=11)


def render_fidelity(cases: list[dict], metrics: dict | list) -> None:
    """Compare the two forward solutions on the postprocessing acceptance grid."""
    info = _check(cases, metrics)
    _style()
    fig = plt.figure(figsize=(18, 11), facecolor=PAPER)
    fig.text(0.055, 0.94, "Field fidelity, point by point.", fontsize=29, weight="bold")
    fig.text(
        0.055,
        0.892,
        f"{info[0]['independent_point_count']} spherical acceptance samples per design. Original currents; no field or current refit.",
        fontsize=14,
        color=MUTED,
    )
    axes = fig.subplots(
        2, 3, gridspec_kw={"left": 0.075, "right": 0.965, "bottom": 0.12, "top": 0.79, "wspace": 0.31, "hspace": 0.55}
    )
    for row, (arrays, case) in enumerate(zip(cases, info)):
        target = arrays["target_T"][:, 2]
        peak = np.max(np.abs(target))
        ax_scatter, ax_hist, ax_cdf = axes[row]
        for ax in axes[row]:
            _clean_axis(ax)
        lim = 1.05 * np.max(np.abs(target)) * 1e6
        ax_scatter.plot([-lim, lim], [-lim, lim], color="#a1aeb8", linewidth=1.2, linestyle="--", label="Ideal")
        residuals = [100 * (arrays[f"{kind}_field_T"][:, 2] - target) / peak for kind in ("sheet", "contour")]
        edge = max(np.max(np.abs(values)) for values in residuals) * 1.05
        bins = np.linspace(-edge, edge, 32)
        for kind, label, color, residual in zip(("sheet", "contour"), ("Sheet", "Contours"), (SHEET, WIRE), residuals):
            actual = arrays[f"{kind}_field_T"][:, 2]
            rms = case[f"{kind}_relative_rms_Bz_error_percent"]
            ax_scatter.scatter(
                target * 1e6,
                actual * 1e6,
                s=10 if kind == "sheet" else 6,
                alpha=0.4,
                color=color,
                edgecolors="none",
                rasterized=True,
                label=f"{label} · {rms:.2f}% RMS",
            )
            ax_hist.hist(
                residual,
                bins=bins,
                histtype="stepfilled" if kind == "sheet" else "step",
                color=color,
                linewidth=1.7,
                alpha=0.24 if kind == "sheet" else 1,
                label=label,
            )
            ordered = np.sort(np.abs(residual))
            fraction = 100 * np.arange(1, len(ordered) + 1) / len(ordered)
            ax_cdf.step(np.r_[0, ordered], np.r_[0, fraction], where="post", color=color, linewidth=2.2, label=label)
        ax_scatter.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Requested Bz [µT]", ylabel="Calculated Bz [µT]")
        ax_scatter.set_title(f"{case['name']} / field agreement", fontsize=15, loc="left", pad=13)
        ax_scatter.legend(loc="upper left", fontsize=10, frameon=False, markerscale=2)
        ax_hist.axvline(0, color="#99aab6", linewidth=0.9)
        ax_hist.set(xlabel="Signed residual / peak target [%]", ylabel="Validation samples")
        ax_hist.set_title("Residual distribution", fontsize=15, loc="left", pad=13)
        ax_hist.legend(fontsize=11, frameon=False)
        ax_cdf.axhline(95, color="#b1bec8", linestyle="--", linewidth=0.9)
        ax_cdf.text(0.97, 0.88, "95% of samples", transform=ax_cdf.transAxes, ha="right", fontsize=10, color=MUTED)
        ax_cdf.set(
            xlim=(0, edge),
            ylim=(0, 101),
            xlabel="Absolute residual / peak target [%]",
            ylabel="Samples within threshold [%]",
        )
        ax_cdf.set_title("Error coverage", fontsize=15, loc="left", pad=13)
        ax_cdf.legend(loc="lower right", fontsize=11, frameon=False)
    fig.text(
        0.055,
        0.055,
        "Residual = calculated Bz − requested Bz. Residual axes use one peak |target Bz| per design, including the target zero crossing.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.027,
        "RMS labels use ‖Bz − Bz,target‖₂ / ‖Bz,target‖₂ over all samples. Contours are unconnected; these results do not establish fabrication readiness.",
        fontsize=11,
        color=MUTED,
    )
    _save(fig, "validation_fidelity")


def render_all(cases: list[dict], metrics: dict | list) -> None:
    """Render all three presentations and their independently checkable provenance."""
    info = _check(cases, metrics)
    render_geometry(cases, info)
    render_stream_function(cases, info)
    render_fidelity(cases, info)
    evidence = {
        "description": "Gallery of original solved current sheets and smoothed, filtered contours with reevaluated fields.",
        "current_refitted": False,
        "contours_connected_or_routed": False,
        "fabrication_validated": False,
        "cylinder_chart": "Per-triangle periodic unwrap, with seam triangles duplicated and axes clipped to [0, 2πR].",
        "residual_normalization": "100 * (computed Bz - requested Bz) / max(abs(requested Bz)); not pointwise relative error.",
        "source_sha256": {
            name: sha256((OUTPUT / name).read_bytes()).hexdigest()
            for name in ("presentation.py", *(case.get("array_file", f"case_{i}.npz") for i, case in enumerate(info)))
        },
        "cases": [],
        "artifact_sha256": {
            f"{name}.{ext}": sha256((OUTPUT / f"{name}.{ext}").read_bytes()).hexdigest()
            for name in ("coil_designs", "stream_function", "validation_fidelity")
            for ext in ("png", "svg")
        },
    }
    for arrays, case in zip(cases, info):
        target = arrays["target_T"][:, 2]
        entry = {
            "name": case["name"],
            "postprocessing_acceptance_sample_count": len(target),
            "peak_target_Bz_T": float(np.max(np.abs(target))),
        }
        for kind in ("sheet", "contour"):
            absolute = np.abs(100 * (arrays[f"{kind}_field_T"][:, 2] - target) / np.max(np.abs(target)))
            entry[f"{kind}_absolute_residual_peak_target_percent"] = {
                "median": float(np.median(absolute)),
                "p95": float(np.percentile(absolute, 95)),
                "maximum": float(np.max(absolute)),
            }
        evidence["cases"].append(entry)
    (OUTPUT / "presentation.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    from contour_processing import load_refined_cases

    cases, metrics = load_refined_cases()
    render_all(cases, metrics)
    print("Rendered CoilGen2 gallery from verified sheets and conditioned contours.")


if __name__ == "__main__":
    main()
