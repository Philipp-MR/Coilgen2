"""Compute and render a coil layout on the welcome page's windowed CAD surface.

Run with the research, gui and plot extras:
    python docs/frontpage/windowed_coil.py

The illustration shows smoothed, unconnected contours at the solved current.
It does not claim a routed circuit or fabrication acceptance. Recomputable
numerical arrays stay in tmp; the image and compact evidence accompany the page.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from coilgen import run_coilgen  # noqa: E402
from coilgen.application.presets import preset_config  # noqa: E402
from coilgen.application.project import ProjectDocument  # noqa: E402
from coilgen.application.surface_model import SurfaceModel, apply_surface, build_surface  # noqa: E402
from coilgen.contours import contour_evaluation_points  # noqa: E402
from coilgen.physics import biot_savart_wire_path  # noqa: E402
from coilgen.persistence import numerical_source_digest  # noqa: E402
from coilgen.research.storage import json_value  # noqa: E402
from coilgen.research.validation import validate_on_independent_grid  # noqa: E402

OUTPUT = ROOT / "docs" / "images"
PAPER, INK, MUTED = "#f4f7fa", "#15324a", "#526a7e"


def render(mesh, paths=()):
    """Use identical orthographic cameras; tubes only give centerlines visibility."""
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1600), lighting="none")
    try:
        plotter.set_background(PAPER)
        faces = np.column_stack((np.full(mesh.n_faces, 3), mesh.faces)).ravel()
        plotter.add_mesh(
            pv.PolyData(mesh.vertices, faces).clean(),
            color="#467d98",
            smooth_shading=True,
            ambient=0.30,
            diffuse=0.65,
            specular=0.2,
            backface_params={"color": "#a2b5bf", "ambient": 0.35},
        )
        for points in paths:
            # No interpolation, smoothing or rescaling occurs in the renderer.
            plotter.add_mesh(
                pv.lines_from_points(points).tube(radius=0.0007, n_sides=12),
                color="#efa456",
                smooth_shading=True,
                ambient=0.35,
                diffuse=0.65,
                specular=0.45,
            )
        plotter.camera_position = [(0.544, -0.832, 0.496), (0, 0, 0), (0, 0, 1)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 0.155
        for position, intensity in (((-0.6, -0.9, 1.2), 0.8), ((0.9, 0.3, 0.6), 0.6)):
            plotter.add_light(pv.Light(position=position, focal_point=(0, 0, 0), intensity=intensity))
        plotter.enable_anti_aliasing("ssaa")
        return plotter.screenshot(return_img=True)
    finally:
        plotter.close()


def relative_rms(actual, target):
    return float(100 * np.linalg.norm(actual[:, 2] - target[:, 2]) / np.linalg.norm(target[:, 2]))


def window_clearance(paths):
    """Check every straight segment against this recipe's perpendicular cutter.

    Its axis is parallel to Y, at X=0, Z=25 mm, ending at Y=0.
    This is centerline clearance, not physical conductor or fabrication clearance.
    """
    closest = np.inf
    for path in paths:
        assert np.max(np.abs(path[:, 2])) <= 0.12 + 1e-10
        for start, end in zip(path[:-1], path[1:]):
            if start[1] >= 0 and end[1] >= 0:
                continue
            if start[1] > 0:
                start = start + (end - start) * (-start[1] / (end[1] - start[1]))
            elif end[1] > 0:
                end = end + (start - end) * (-end[1] / (start[1] - end[1]))
            offset = start[[0, 2]] - [0, 0.025]
            delta = (end - start)[[0, 2]]
            denominator = float(delta @ delta)
            fraction = np.clip(-offset @ delta / denominator, 0, 1) if denominator > 0 else 0
            closest = min(closest, float(np.linalg.norm(offset + fraction * delta)))
    margin = closest - 0.045
    assert margin > 0, "A contour segment crosses the cylindrical window."
    return margin


def main():
    recipe_path = OUTPUT / "welcome-geometry-evidence.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))["figures"][0]["model"]
    model = SurfaceModel(recipe)
    mesh, scene = build_surface(model, final=True)
    assert scene["mesh_report"]["components"] == 1 and scene["mesh_report"]["boundary_loops"] == 3
    config = replace(
        preset_config("cylinder"),
        target_gradient_strength=0.01,
        target_region_radius=0.02,
        gauss_order=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        save_stl_flag=False,
        plot_flag=False,
        progress_flag=True,
        constrained_optimization_threads=1,
        parallel_worker_count=1,
    )
    project = ProjectDocument.from_config(config)
    project = apply_surface(project, model, mesh, expected_fingerprint=project.fingerprint)
    with threadpool_limits(limits=1):
        result = run_coilgen(project.config, project_root=ROOT)
        check = validate_on_independent_grid(result, resolution=16)
        assert check["available"]
        points, target = check["points_m"], check["target_T"]
        field = np.zeros_like(target)
        paths, currents = [], []
        for part in result.coil_parts:
            for contour in part.contour_lines:
                path = contour_evaluation_points(part.mesh, contour, result.input_data)
                assert np.all(np.isfinite(path)) and np.allclose(path[0], path[-1], atol=1e-10, rtol=0)
                current = float(part.contour_step * contour.turn_multiplicity)
                field += current * biot_savart_wire_path(path, points, worker_count=1)
                paths.append(path)
                currents.append(current)
    assert paths and np.isfinite(field).all()
    design_points = result.target_field.coords
    overlaps = np.any(np.all(np.isclose(points[:, None], design_points[None], rtol=0, atol=1e-12), axis=2), axis=1)
    assert not np.any(overlaps)
    margin = window_clearance(paths)
    error = relative_rms(field, target)
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    fig.text(0.04, 0.95, "From a windowed surface to a coil layout.", fontsize=30, weight="bold", color=INK)
    fig.text(
        0.04,
        0.90,
        "Same CAD surface · Gy = 10 mT/m · 40 mm spherical DSV · original solved current",
        fontsize=15,
        color=MUTED,
    )
    for i, (title, image) in enumerate(
        (("Windowed cylindrical support", render(mesh)), ("Optimized winding contours", render(mesh, paths)))
    ):
        ax = fig.add_axes((0.025 + i * 0.49, 0.12, 0.47, 0.71))
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title, fontsize=20, weight="bold", color=INK, pad=8)
    fig.text(
        0.04,
        0.062,
        f"{len(paths)} closed contours · {error:.2f}% RMS Bz error on {len(points)} independent ROI points",
        fontsize=15,
        color=INK,
    )
    fig.text(
        0.04,
        0.025,
        "Smoothed contours before routing; connections, feeds and fabrication checks remain separate. Tube width is for display.",
        fontsize=12,
        color=MUTED,
    )
    image_path = OUTPUT / "welcome-windowed-coil.png"
    fig.savefig(image_path, dpi=240, facecolor=PAPER)
    plt.close(fig)
    arrays = {
        "vertices_m": mesh.vertices,
        "faces": mesh.faces,
        "points_m": points,
        "target_T": target,
        "contour_field_T": field,
    }
    arrays.update({f"contour_{i}_points_m": path for i, path in enumerate(paths)})
    scratch = ROOT / "tmp" / "welcome-windowed-coil"
    scratch.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scratch / "numerical-arrays.npz", **arrays)
    diagnostics = result.regularization_diagnostics
    evidence = {
        "scope": "Optimized, smoothed unconnected contours; no circuit, feed, sweep or fabrication acceptance",
        "model": recipe,
        "mesh_report": scene["mesh_report"],
        "config_before_surface_application": asdict(config),
        "surface_application": {"surface_is_cylinder_flag": False, "sweep_surface_realization": "mesh"},
        "contour_count": len(paths),
        "contour_currents_A": currents,
        "current_refitted": False,
        "independent_point_count": len(points),
        "design_grid_overlap_count": int(overlaps.sum()),
        "minimum_centerline_radial_clearance_from_window_cutter_m": margin,
        "window_clearance_check": "Exact segment minimum distance to cutter axis on its Y<0 side; no conductor-width allowance",
        "stream_function_relative_rms_Bz_error_percent": relative_rms(check["field_T"], target),
        "contour_relative_rms_Bz_error_percent": error,
        "stream_function_budget_passed": check["passed"],
        "solver_contract_met": diagnostics.get("constrained_solver_contract_met"),
        "solver_status": diagnostics.get("constrained_solver_status"),
        "contour_diagnostics": [part.contour_level_diagnostics for part in result.coil_parts],
        "smoothing_diagnostics": {key: value for key, value in diagnostics.items() if key.startswith("contour_path_")},
        "independent_mesh_or_quadrature_convergence": False,
        "numerical_source_sha256": numerical_source_digest(ROOT / "src" / "coilgen"),
        "versions": {name: version(name) for name in ("numpy", "scipy", "gmsh", "pyvista", "matplotlib")},
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_sha256": sha256(image_path.read_bytes()).hexdigest(),
        "arrays_sha256": sha256((scratch / "numerical-arrays.npz").read_bytes()).hexdigest(),
    }
    (OUTPUT / "welcome-windowed-coil-evidence.json").write_text(
        json.dumps(json_value(evidence), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {image_path.name}: {len(paths)} contours, {error:.3f}% independent RMS Bz error", flush=True)


if __name__ == "__main__":
    main()
