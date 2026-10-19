"""Recompute the GitHub front-page figures from generated CoilGen2 designs.

From the repository root:
    python -m pip install -e ".[research,plot,gui]"
    python docs/frontpage/generate_figures.py

The plots distinguish the optimized sheet from smoothed, filtered unconnected
contours. Original solved currents are retained; every processed contour field
is reevaluated. No current fit, routing or fabrication acceptance is implied.
"""

from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from coilgen import CoilGenConfig, run_coilgen  # noqa: E402
from coilgen.contours import contour_evaluation_points  # noqa: E402
from coilgen.physics import biot_savart_wire_path  # noqa: E402
from coilgen.research.storage import json_value  # noqa: E402
from coilgen.research.validation import validate_on_independent_grid  # noqa: E402


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


def source_digest() -> str:
    digest = sha256()
    for path in sorted((ROOT / "src" / "coilgen").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_saved_cases() -> tuple[list[dict], list[dict], str]:
    """Reuse verified numerical evidence without changing the solved designs."""
    metadata = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    if metadata["source_tree_sha256"] != source_digest():
        raise RuntimeError("Numerical source changed. Recompute without --reuse-data.")
    if len(metadata["cases"]) != len(NAMES):
        raise RuntimeError("Saved case count differs from the configured demonstrations.")
    cases = []
    for index, (config, case) in enumerate(zip(configurations(), metadata["cases"])):
        if json_value(asdict(config)) != case["config"]:
            raise RuntimeError(f"Case {index}: configuration changed. Recompute without --reuse-data.")
        path = OUTPUT / f"case_{index}.npz"
        if sha256(path.read_bytes()).hexdigest() != metadata["artifact_sha256"][path.name]:
            raise RuntimeError(f"Case {index}: saved array checksum differs from its evidence.")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise RuntimeError(f"Case {index}: saved arrays contain nonfinite values.")
        for kind in ("sheet", "contour"):
            observed = 100 * relative_rms(arrays[f"{kind}_field_T"], arrays["target_T"])
            if not np.isclose(observed, case[f"{kind}_relative_rms_Bz_error_percent"], rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"Case {index}: saved {kind} field does not reproduce the recorded metric.")
        cases.append(arrays)
    revision = metadata.get("numerical_data_revision", metadata["git_revision"])
    print(f"Reusing verified solved designs from {revision}; evaluating the dense field maps afresh.", flush=True)
    return cases, metadata["cases"], revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="Render verified saved solutions and reevaluate dense field maps without repeating optimization.",
    )
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with threadpool_limits(limits=1):
        if args.reuse_data:
            cases, metrics, data_revision = load_saved_cases()
        else:
            computed = [
                compute_case(config, name, index) for index, (config, name) in enumerate(zip(configurations(), NAMES))
            ]
            cases, metrics = map(list, zip(*computed))
            data_revision = revision
        from contour_processing import prepare_cases
        from presentation import render_all
        from field_maps import compute_and_render
        from swept_bodies import render_all as render_swept_bodies

        refined_cases, refined_metrics = prepare_cases(cases, metrics)
        render_all(refined_cases, refined_metrics)
        compute_and_render(refined_cases, refined_metrics)
        render_swept_bodies(refined_cases, refined_metrics)
    metadata = {
        "description": "Generated magnetostatic demonstration; not routed or fabrication-validated hardware.",
        "error_definition": "100 * sqrt(sum((Bz - Bz_target)^2) / sum(Bz_target^2)); equal validation weights",
        "current_refitted": False,
        "independent_sampling": True,
        "independent_sampling_scope": "Original sheet solve versus 912 points; processed paths use those points for acceptance and a separate untouched 1472-point holdout.",
        "independent_quadrature_or_mesh_convergence": False,
        "routed": False,
        "fabrication_validated": False,
        "git_revision": revision,
        "numerical_data_revision": data_revision,
        "optimization_recomputed": not args.reuse_data,
        "source_tree_sha256": source_digest(),
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_source_sha256": {
            name: sha256((OUTPUT / name).read_bytes()).hexdigest()
            for name in ("contour_processing.py", "presentation.py", "field_maps.py", "swept_bodies.py")
        },
        "python_version": platform.python_version(),
        "dependencies": {
            package: version(package)
            for package in ("numpy", "scipy", "cvxpy", "clarabel", "matplotlib", "threadpoolctl", "pyvista", "vtk")
        },
        "cases": list(metrics),
        "refined_cases": list(refined_metrics),
        "artifact_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(OUTPUT.glob("*"))
            if path.suffix in (".npz", ".png", ".svg")
            or path.name in ("contour_processing.json", "presentation.json", "field_maps.json", "swept_bodies.json")
        },
    }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(json_value(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Saved numerical evidence and figures in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
