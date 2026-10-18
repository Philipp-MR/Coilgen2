"""Recompute the GitHub front-page figures from generated CoilGen2 designs.

From the repository root:
    python -m pip install -e ".[research,plot]"
    python docs/frontpage/generate_figures.py

The plots distinguish the optimized sheet from unconnected iso-current
contours. Original solved currents are retained; no display-time current fit,
routing, fabrication acceptance, or hardware safety claim is made.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from coilgen import CoilGenConfig, run_coilgen  # noqa: E402
from coilgen.contours import contour_evaluation_points  # noqa: E402
from coilgen.physics import biot_savart_wire_path  # noqa: E402
from coilgen.research.storage import json_value  # noqa: E402
from coilgen.research.validation import validate_on_independent_grid  # noqa: E402


INK = "#142c44"
MUTED = "#556779"
ACCENTS = ("#087e8b", "#bd5928")
NAMES = ("Cylindrical Gy", "Biplanar Gz")


def configurations() -> tuple[CoilGenConfig, CoilGenConfig]:
    """Minimize sheet power with a 3.2% solve budget and a 4% check budget."""
    cylinder = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        # The legacy mesh-height parameter spans nz/(nz+1) of its input.
        # Compensate here to realize the stated 300 mm support length.
        cylinder_mesh_parameter_list=(0.30 * 13 / 12, 0.12, 24, 12, 1, 0, 0, 0),
        surface_is_cylinder_flag=True,
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=15,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power", "b0_T": [0, 0, 3]},
        constrained_quadratic_fallback_policy="strict",
        constrained_optimization_tolerance=1e-8,
        constrained_optimization_threads=1,
        target_error_tolerance_percent=3.2,
        target_error_constraint_acceptance_absolute_T=0,
        stream_function_smoothness_factor=0,
        turn_spacing_regularization_factor=0,
        symmetry_reduction="off",
        levels=16,
        min_point_loop_number=3,
        gauss_order=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        plot_flag=False,
        save_stl_flag=False,
        progress_flag=False,
    )
    biplanar = replace(
        cylinder,
        coil_mesh_file="create bi-planary mesh",
        biplanar_mesh_parameter_list=(0.30, 0.30, 12, 12, 0, 0, 1, 0, 0, 0, 0.20),
        surface_is_cylinder_flag=False,
        field_shape_function="z",
    )
    return cylinder, biplanar


def relative_rms(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(actual[:, 2] - target[:, 2]) / np.linalg.norm(target[:, 2]))


def support_evidence(parts, index: int) -> dict:
    """Check realized geometry against the dimensions published in the README."""
    vertices = np.vstack([part.mesh.vertices for part in parts])
    extents = np.ptp(vertices, axis=0)
    expected = (0.24, 0.24, 0.30) if index == 0 else (0.30, 0.30, 0.20)
    if len(parts) != index + 1 or not np.allclose(extents, expected, rtol=0, atol=1e-12):
        raise RuntimeError(f"Case {index}: realized support dimensions differ from the documented geometry: {extents}.")
    if not np.allclose(vertices.min(axis=0) + vertices.max(axis=0), 0, rtol=0, atol=1e-12):
        raise RuntimeError(f"Case {index}: support bounds are not centered at the world origin.")
    if index == 0:
        if not np.allclose(np.linalg.norm(vertices[:, :2], axis=1), 0.12, rtol=0, atol=1e-12):
            raise RuntimeError("Cylinder vertices do not lie at the documented 120 mm radius.")
    else:
        for part in parts:
            if not np.allclose(np.ptp(part.mesh.vertices, axis=0), (0.30, 0.30, 0), rtol=0, atol=1e-12):
                raise RuntimeError("A biplanar support does not realize the documented 300 × 300 mm plane.")
        separation = abs(parts[0].mesh.vertices[:, 2].mean() - parts[1].mesh.vertices[:, 2].mean())
        if not np.isclose(separation, 0.20, rtol=0, atol=1e-12):
            raise RuntimeError("Biplanar supports do not realize the documented 200 mm plane separation.")
    return {
        "support_extents_m": extents,
        "support_bounds_m": {"minimum_xyz": vertices.min(axis=0), "maximum_xyz": vertices.max(axis=0)},
        "part_support_extents_m": [np.ptp(part.mesh.vertices, axis=0) for part in parts],
        "documented_support_dimensions_verified": True,
    }


def compute_case(config: CoilGenConfig, name: str, index: int) -> tuple[dict, dict]:
    print(f"Computing {name}...", flush=True)
    result = run_coilgen(config, project_root=ROOT)
    support = support_evidence(result.coil_parts, index)
    validation = validate_on_independent_grid(result, resolution=12, design_margin=0.8)
    if not validation["available"] or not validation["passed"]:
        raise RuntimeError(f"{name}: independent sheet validation failed: {validation}")
    points, target = validation["points_m"], validation["target_T"]
    arrays = {
        "validation_points_m": points,
        "target_T": target,
        "sheet_field_T": validation["field_T"],
    }
    contour_field = np.zeros_like(target)
    contour_count = 0
    currents = []
    for part_index, part in enumerate(result.coil_parts):
        prefix = f"part_{part_index}"
        arrays[f"{prefix}_vertices_m"] = part.mesh.vertices
        arrays[f"{prefix}_faces"] = part.mesh.faces
        arrays[f"{prefix}_stream_function_A"] = part.stream_function
        currents.append(float(part.contour_step))
        for contour_index, contour in enumerate(part.contour_lines):
            path = contour_evaluation_points(part.mesh, contour, config)
            arrays[f"{prefix}_contour_{contour_index}_points_m"] = path
            contour_field += (
                part.contour_step * contour.turn_multiplicity * biot_savart_wire_path(path, points, worker_count=1)
            )
            contour_count += 1
    arrays["contour_field_T"] = contour_field
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError(f"{name}: nonfinite figure data.")
    design_points = result.target_field.coords
    overlap = np.any(np.all(np.isclose(points[:, None], design_points[None], rtol=0, atol=1e-12), axis=2))
    if overlap:
        raise RuntimeError(f"{name}: independent points overlap the design grid.")
    np.savez_compressed(OUTPUT / f"case_{index}.npz", **arrays)
    diagnostics = result.regularization_diagnostics
    metrics = {
        "name": name,
        "config": asdict(config),
        **support,
        "part_count": len(result.coil_parts),
        "vertex_count": sum(part.mesh.n_vertices for part in result.coil_parts),
        "triangle_count": sum(part.mesh.n_faces for part in result.coil_parts),
        "unconnected_contour_count": contour_count,
        "contour_currents_A": currents,
        "design_point_count": len(design_points),
        "independent_point_count": len(points),
        "design_grid_overlap_count": 0,
        "sheet_relative_rms_Bz_error_percent": 100 * relative_rms(validation["field_T"], target),
        "contour_relative_rms_Bz_error_percent": 100 * relative_rms(contour_field, target),
        "sheet_peak_Bz_error_uT": float(1e6 * np.max(np.abs(validation["field_T"][:, 2] - target[:, 2]))),
        "contour_peak_Bz_error_uT": float(1e6 * np.max(np.abs(contour_field[:, 2] - target[:, 2]))),
        "sheet_independent_budget_passed": bool(validation["passed"]),
        "sheet_independent_relative_rms_budget_percent": 4.0,
        "design_margin": 0.8,
        "solver": diagnostics.get("constrained_solver_name"),
        "solver_contract_met": diagnostics.get("constrained_solver_contract_met"),
        "solver_status": diagnostics.get("constrained_solver_status"),
    }
    print(
        f"{name}: sheet {metrics['sheet_relative_rms_Bz_error_percent']:.3f}%, "
        f"unconnected contours {metrics['contour_relative_rms_Bz_error_percent']:.3f}% RMS Bz error",
        flush=True,
    )
    return arrays, metrics


def plot_geometry(ax, arrays: dict, metrics: dict, accent: str, title: str) -> None:
    bound = max(np.max(np.abs(value)) for key, value in arrays.items() if key.endswith("vertices_m")) * 1e3
    sf_bound = max(np.max(np.abs(value)) for key, value in arrays.items() if key.endswith("stream_function_A"))
    norm = colors.TwoSlopeNorm(vmin=-sf_bound, vcenter=0, vmax=sf_bound)
    for index in range(metrics["part_count"]):
        vertices = arrays[f"part_{index}_vertices_m"] * 1e3
        faces = arrays[f"part_{index}_faces"]
        sf = arrays[f"part_{index}_stream_function_A"]
        face_colors = plt.colormaps["RdBu_r"](norm(np.mean(sf[faces], axis=1)))
        face_colors[:, 3] = 0.25
        mesh = Poly3DCollection(vertices[faces], facecolors=face_colors, edgecolors="none", zsort="average")
        ax.add_collection3d(mesh)
    for key, value in arrays.items():
        if "_contour_" in key:
            ax.plot(*(value * 1e3).T, color=accent, linewidth=1.35, alpha=0.95)
    theta = np.linspace(0, 2 * np.pi, 100)
    for plane in range(3):
        circle = np.zeros((len(theta), 3))
        circle[:, (plane + 1) % 3] = 45 * np.cos(theta)
        circle[:, (plane + 2) % 3] = 45 * np.sin(theta)
        ax.plot(*circle.T, color="#70869a", linewidth=0.6, alpha=0.65)
    for dimension in "xyz":
        getattr(ax, f"set_{dimension}lim")(-bound * 1.04, bound * 1.04)
        getattr(ax, f"set_{dimension}label")(f"{dimension} [mm]", fontsize=9, labelpad=0)
        axis = getattr(ax, f"{dimension}axis")
        axis.set_pane_color((0.97, 0.98, 0.99, 0.5))
        axis._axinfo["grid"]["color"] = (0.82, 0.87, 0.9, 0.5)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-52)
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    ax.tick_params(labelsize=8, pad=0)
    ax.set_title(title, fontsize=16, fontweight="bold", color=accent, pad=6)
    ax.text2D(
        0.5,
        -0.035,
        f"{metrics['unconnected_contour_count']} unconnected contours  |  "
        f"{metrics['triangle_count']} surface triangles",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color=MUTED,
    )


def save_figure(fig, stem: str) -> None:
    for extension in ("png", "svg"):
        path = OUTPUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
        if extension == "svg":
            # Hash canonical Git-checkout bytes on every platform. Matplotlib
            # uses native text newlines, and its SVG can contain trailing spaces.
            content = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
            path.write_text(content, encoding="utf-8", newline="\n")
    plt.close(fig)


def plot_figures(cases: list[dict], metrics: list[dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED})
    fig = plt.figure(figsize=(13.6, 6.6), facecolor="#ffffff")
    fig.suptitle("From a target field to surface current patterns", fontsize=23, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.89,
        "Generated supports  /  10 mT/m gradients  /  90 mm spherical target region",
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    for index, (arrays, case, accent, title) in enumerate(zip(cases, metrics, ACCENTS, NAMES)):
        ax = fig.add_axes((0.015 + index * 0.5, 0.115, 0.47, 0.72), projection="3d")
        plot_geometry(ax, arrays, case, accent, title)
    fig.text(
        0.5,
        0.025,
        "Color: relative stream function. Lines: extracted iso-current contours. Gray rings: target region.",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    save_figure(fig, "coil_designs")

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.5), facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.17, top=0.81, hspace=0.45, wspace=0.28)
    fig.suptitle("Check the field after discretizing the current sheet", fontsize=22, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.91,
        "Distinct validation samples; original current retained throughout",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    vmax = max(np.max(np.abs(case["target_T"][:, 2])) * 1e3 for case in cases)
    error_max = max(
        np.max(np.abs(case[key][:, 2] - case["target_T"][:, 2])) * 1e6
        for case in cases
        for key in ("sheet_field_T", "contour_field_T")
    )
    for row, (arrays, case, accent, title) in enumerate(zip(cases, metrics, ACCENTS, NAMES)):
        points = arrays["validation_points_m"]
        x_slice = np.min(points[points[:, 0] > 0, 0])
        selected = np.isclose(points[:, 0], x_slice, rtol=0, atol=1e-12)
        axis = np.unique(points[:, 1])
        y_index = np.searchsorted(axis, points[selected, 1])
        z_index = np.searchsorted(axis, points[selected, 2])
        data = (
            arrays["target_T"][:, 2] * 1e3,
            (arrays["sheet_field_T"][:, 2] - arrays["target_T"][:, 2]) * 1e6,
            (arrays["contour_field_T"][:, 2] - arrays["target_T"][:, 2]) * 1e6,
        )
        labels = (
            "Target Bz [mT]",
            f"Sheet residual [µT]\n{case['sheet_relative_rms_Bz_error_percent']:.2f}% relative RMS",
            f"Contour residual [µT]\n{case['contour_relative_rms_Bz_error_percent']:.2f}% relative RMS",
        )
        for column, (values, label) in enumerate(zip(data, labels)):
            ax = axes[row, column]
            grid = np.full((len(axis), len(axis)), np.nan)
            grid[z_index, y_index] = values[selected]
            scale = vmax if column == 0 else error_max
            image = ax.pcolormesh(
                axis * 1e3,
                axis * 1e3,
                np.ma.masked_invalid(grid),
                cmap="RdBu_r",
                vmin=-scale,
                vmax=scale,
                shading="nearest",
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_title(label, fontsize=11, color=INK)
            ax.set_xlabel("y [mm]", fontsize=10)
            if column == 0:
                ax.set_ylabel(f"{title}\nz [mm]", fontsize=11, color=accent)
            ax.tick_params(labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0 and column in (0, 2):
                colorbar_ax = fig.add_axes((0.945, 0.58 if column == 0 else 0.22, 0.012, 0.20))
                bar = fig.colorbar(image, cax=colorbar_ax)
                bar.set_label("Bz [mT]" if column == 0 else "Residual [µT]", fontsize=10)
                bar.ax.tick_params(labelsize=8)
    fig.text(
        0.5,
        0.055,
        "Maps show the x = +3.75 mm validation slice. RMS values use all 912 spherical validation samples.\nResidual = calculated Bz − requested Bz. Contours are unconnected; routing and fabrication are separate design stages.",
        ha="center",
        fontsize=10,
        color=MUTED,
        linespacing=1.5,
    )
    save_figure(fig, "field_validation")


def source_digest() -> str:
    digest = sha256()
    for path in sorted((ROOT / "src" / "coilgen").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    with threadpool_limits(limits=1):
        computed = [
            compute_case(config, name, index) for index, (config, name) in enumerate(zip(configurations(), NAMES))
        ]
        cases, metrics = zip(*computed)
        plot_figures(list(cases), list(metrics))
    metadata = {
        "description": "Generated magnetostatic demonstration; not routed or fabrication-validated hardware.",
        "error_definition": "100 * sqrt(sum((Bz - Bz_target)^2) / sum(Bz_target^2)); equal validation weights",
        "current_refitted": False,
        "independent_sampling": True,
        "independent_quadrature_or_mesh_convergence": False,
        "routed": False,
        "fabrication_validated": False,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_tree_sha256": source_digest(),
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "dependencies": {
            package: version(package)
            for package in ("numpy", "scipy", "cvxpy", "clarabel", "matplotlib", "threadpoolctl")
        },
        "cases": list(metrics),
        "artifact_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(OUTPUT.glob("*"))
            if path.suffix in (".npz", ".png", ".svg")
        },
    }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(json_value(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved numerical evidence and figures in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
