"""Direct dense-plane evaluation of the smoothed front-page current patterns.

Run ``python docs/frontpage/field_maps.py`` after the main figure generator.
No optimization, interpolation or current adjustment is performed. The saved
912-point vector fields must first be reproduced by both physical models.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh  # noqa: E402
from coilgen.config import CoilGenConfig  # noqa: E402
from coilgen.geometry import compute_face_normals, compute_vertex_normals  # noqa: E402
from coilgen.models import CoilPart, Mesh, TargetField  # noqa: E402
from coilgen.physics import biot_savart_wire_path, calculate_sensitivity_matrix  # noqa: E402


INK = "#122b43"
MUTED = "#526679"
ACCENTS = ("#b85a2c", "#087f8c")
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "coilgen_field", ("#093b68", "#228ebc", "#f4f7f7", "#ef9b59", "#9b302d"), N=256
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "coilgen_error", ("#46215b", "#9672af", "#faf8f2", "#e9aa49", "#9a4d12"), N=256
)


def source_digest() -> str:
    digest = sha256()
    for path in sorted((ROOT / "src" / "coilgen").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _parts(arrays: dict, case: dict) -> list[CoilPart]:
    parts = []
    for index in range(case["part_count"]):
        prefix = f"part_{index}"
        mesh = Mesh(arrays[f"{prefix}_vertices_m"], arrays[f"{prefix}_faces"])
        mesh.face_normals = compute_face_normals(mesh)
        mesh.vertex_normals = compute_vertex_normals(mesh)
        parts.append(CoilPart(mesh, stream_function=arrays[f"{prefix}_stream_function_A"]))
    calculate_one_ring_by_mesh(parts)
    calculate_basis_functions(parts)
    return parts


def _sheet_field(parts: list[CoilPart], points: np.ndarray, config: CoilGenConfig) -> np.ndarray:
    """Bound sensitivity storage to at most 512 target points at a time."""
    field = np.empty((len(points), 3))
    for start in range(0, len(points), 512):
        block = points[start : start + 512]
        target = TargetField(
            block, np.zeros_like(block), np.ones(len(block)), np.zeros(len(block), bool), np.ones(len(block), int)
        )
        calculate_sensitivity_matrix(parts, target, config)
        field[start : start + len(block)] = sum(
            np.einsum("ctv,v->tc", part.sensitivity_matrix, part.stream_function) for part in parts
        )
    for part in parts:
        part.sensitivity_matrix = None
    return field


def _contour_field(arrays: dict, case: dict, points: np.ndarray) -> np.ndarray:
    field = np.zeros((len(points), 3))
    for index, current in enumerate(case["contour_currents_A"]):
        prefix = f"part_{index}_contour_"
        keys = [key for key in arrays if key.startswith(prefix) and key.endswith("_points_m")]
        keys.sort(key=lambda key: int(key[len(prefix) : -len("_points_m")]))
        for key in keys:
            field += current * biot_savart_wire_path(arrays[key], points, worker_count=1, max_chunk_memory_mb=16)
    return field


def _verify_saved(actual: np.ndarray, expected: np.ndarray, label: str) -> dict:
    delta = actual - expected
    maximum = float(np.max(np.abs(delta)))
    relative = float(np.linalg.norm(delta) / np.linalg.norm(expected))
    # Floating-point grouping can differ between the original and blocked
    # sensitivity multiplication. No field-scale fit or sign correction is
    # permitted. This also detects any missing contour turn multiplicity.
    scale = float(np.max(np.abs(expected)))
    if not np.allclose(actual, expected, rtol=2e-12, atol=2e-14 * scale):
        raise ValueError(f"{label}: saved field was not reproduced (relative L2 difference {relative:.3g}).")
    return {"relative_L2_difference": relative, "maximum_absolute_difference_T": maximum, "passed": True}


def compute(cases: list[dict], metrics: list[dict]) -> tuple[dict, dict]:
    """Evaluate the saved processed paths on a 101 by 101 midplane grid."""
    if len(cases) != 2 or len(metrics) != 2:
        raise ValueError("The front page requires the two documented gradient cases.")
    axis = np.linspace(-0.045, 0.045, 101)
    yy, zz = np.meshgrid(axis, axis, indexing="xy")
    inside = yy**2 + zz**2 <= 0.045**2 * (1 + 8 * np.finfo(float).eps)
    points = np.column_stack((np.zeros(np.count_nonzero(inside)), yy[inside], zz[inside]))
    dense = {"axis_m": axis, "inside_roi": inside, "points_m": points}
    evidence = []
    for index, (arrays, case) in enumerate(zip(cases, metrics)):
        path = OUTPUT / case.get("array_file", f"case_{index}.npz")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], arrays[key]) for key in arrays):
                raise ValueError(f"{path.name} does not match the provided case arrays.")
        config = CoilGenConfig(gauss_order=case["config"]["gauss_order"], parallel_worker_count=1, progress_flag=False)
        parts = _parts(arrays, case)
        print(f"{case['name']}: reproducing all saved 912-point vector fields", flush=True)
        proof = {
            "sheet": _verify_saved(
                _sheet_field(parts, arrays["validation_points_m"], config), arrays["sheet_field_T"], "Sheet"
            ),
            "contours": _verify_saved(
                _contour_field(arrays, case, arrays["validation_points_m"]), arrays["contour_field_T"], "Contours"
            ),
        }
        print(f"{case['name']}: directly evaluating {len(points)} dense midplane points", flush=True)
        target = np.zeros_like(points)
        target[:, 2] = 0.01 * points[:, index + 1]
        old_target = np.zeros_like(arrays["validation_points_m"])
        old_target[:, 2] = 0.01 * arrays["validation_points_m"][:, index + 1]
        _verify_saved(old_target, arrays["target_T"], "Target expression")
        sheet = _sheet_field(parts, points, config)
        contours = _contour_field(arrays, case, points)
        for key, values in (("target_T", target), ("sheet_field_T", sheet), ("contour_field_T", contours)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Case {index}: nonfinite dense {key}.")
            dense[f"case_{index}_{key}"] = values
        evidence.append(
            {
                "name": case["name"],
                "input_file": path.name,
                "input_sha256": sha256(path.read_bytes()).hexdigest(),
                "saved_vector_field_reproduction": proof,
                "target_expression_SI": "Bz = 0.01 y T" if index == 0 else "Bz = 0.01 z T",
                "original_signed_contour_currents_A": case["contour_currents_A"],
                "saved_contour_multiplicities_verified_by_vector_field_reproduction": True,
                "volume_validation_sample_count": len(arrays["validation_points_m"]),
                "volume_sheet_relative_rms_Bz_error_percent": case["sheet_relative_rms_Bz_error_percent"],
                "volume_contour_relative_rms_Bz_error_percent": case["contour_relative_rms_Bz_error_percent"],
                "dense_plane_sheet_peak_absolute_residual_uT": float(1e6 * np.max(np.abs(sheet[:, 2] - target[:, 2]))),
                "dense_plane_contour_peak_absolute_residual_uT": float(
                    1e6 * np.max(np.abs(contours[:, 2] - target[:, 2]))
                ),
            }
        )
    metadata = {
        "description": "Direct dense-plane fields of the smoothed, filtered front-page contours and original current sheets; no current fit.",
        "grid": {
            "axis_count": len(axis),
            "axis_spacing_m": float(axis[1] - axis[0]),
            "x_m": 0.0,
            "roi_radius_m": 0.045,
            "evaluated_point_count_per_case": len(points),
        },
        "array_order": "meshgrid(axis_m, axis_m, indexing='xy'); field rows match inside_roi boolean selection",
        "interpolation": False,
        "current_refitted": False,
        "sheet_quadrature_order": int(metrics[0]["config"]["gauss_order"]),
        "sheet_evaluation": "Core calculate_sensitivity_matrix using rebuilt original mesh basis and saved stream function; blocks of 512 points.",
        "contour_evaluation": "Core analytical straight-segment Biot-Savart kernel with original signed contour current; 16 MiB chunk budget.",
        "mesh_or_quadrature_convergence_study": False,
        "routed": False,
        "fabrication_validated": False,
        "source_tree_sha256": source_digest(),
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "dependencies": {package: version(package) for package in ("numpy", "matplotlib", "threadpoolctl")},
        "cases": evidence,
    }
    return dense, metadata


def render(dense: dict, metadata: dict) -> None:
    """Render quantitative maps with shared field and residual scales."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "svg.fonttype": "path",
            "svg.hashsalt": "CoilGen2 dense field maps",
        }
    )
    fig = plt.figure(figsize=(15.5, 11.2), facecolor="white")
    fig.text(0.055, 0.955, "The field behind the pattern", fontsize=29, fontweight="bold", color=INK)
    fig.text(
        0.055, 0.912, "10 mT/m gradients  ·  90 mm target sphere  ·  original solved currents", fontsize=15, color=MUTED
    )
    positions = ((0.55, 0.255), (0.16, 0.255))
    headings = ("Requested Bz", "Contour Bz", "Sheet − target", "Contours − target")
    field_scale = max(
        float(np.max(np.abs(dense[f"case_{i}_{key}"][:, 2]))) * 1e3
        for i in range(2)
        for key in ("target_T", "sheet_field_T", "contour_field_T")
    )
    field_scale = float(0.05 * np.ceil(field_scale / 0.05))
    residual_scale = max(
        float(np.max(np.abs(dense[f"case_{i}_{key}"][:, 2] - dense[f"case_{i}_target_T"][:, 2]))) * 1e6
        for i in range(2)
        for key in ("sheet_field_T", "contour_field_T")
    )
    # Round outward to provide interpretable shared tick values, with no clipping.
    residual_scale = float(5 * np.ceil(residual_scale / 5))
    metadata["rendering"] = {
        "shared_field_scale_mT": [-field_scale, field_scale],
        "shared_residual_scale_uT": [-residual_scale, residual_scale],
        "field_or_residual_color_clipping": False,
        "svg_text": "Embedded vector glyph paths",
        "png_dpi": 280,
    }
    axis_mm = dense["axis_m"] * 1e3
    field_image = error_image = None
    for row, (bottom, height) in enumerate(positions):
        case = metadata["cases"][row]
        row_heading_y = 0.865 if row == 0 else 0.465
        fig.text(0.055, row_heading_y, case["name"], fontsize=19, fontweight="bold", color=ACCENTS[row])
        fig.text(
            0.945,
            row_heading_y + 0.002,
            f"Volume RMS error  ·  sheet {case['volume_sheet_relative_rms_Bz_error_percent']:.2f}%  /  contours {case['volume_contour_relative_rms_Bz_error_percent']:.2f}%",
            ha="right",
            fontsize=13,
            color=MUTED,
        )
        target = dense[f"case_{row}_target_T"][:, 2]
        values = (
            target * 1e3,
            dense[f"case_{row}_contour_field_T"][:, 2] * 1e3,
            (dense[f"case_{row}_sheet_field_T"][:, 2] - target) * 1e6,
            (dense[f"case_{row}_contour_field_T"][:, 2] - target) * 1e6,
        )
        for column, field in enumerate(values):
            ax = fig.add_axes((0.055 + column * 0.23, bottom, 0.20, height))
            grid = np.full(dense["inside_roi"].shape, np.nan)
            grid[dense["inside_roi"]] = field
            is_error = column >= 2
            scale = residual_scale if is_error else field_scale
            image = ax.pcolormesh(
                axis_mm,
                axis_mm,
                np.ma.masked_invalid(grid),
                shading="nearest",
                cmap=ERROR_CMAP if is_error else FIELD_CMAP,
                vmin=-scale,
                vmax=scale,
                rasterized=True,
            )
            # Only evaluated grid cells are shown. The vector outline defines
            # the actual spherical cross-section without filling masked cells.
            circle = Circle((0, 0), 45, facecolor="none", edgecolor="#607589", linewidth=0.9)
            ax.add_patch(circle)
            image.set_clip_path(circle)
            if is_error:
                error_image = image
            else:
                field_image = image
            if row == 0:
                ax.set_title(headings[column], fontsize=16, fontweight="bold", pad=12)
            ax.set(xlim=(-47, 47), ylim=(-47, 47), aspect="equal", xticks=(-40, 0, 40), yticks=(-40, 0, 40))
            ax.set_xlabel("y [mm]", fontsize=13, color=MUTED, labelpad=5)
            if column == 0:
                ax.set_ylabel("z [mm]", fontsize=13, color=MUTED, labelpad=3)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=12, colors=MUTED, length=3, width=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#ccd5dc")
    for left, image, label, ticks in (
        (0.10, field_image, "Bz [mT]", (-field_scale, 0, field_scale)),
        (0.56, error_image, "Signed Bz residual [µT]", (-residual_scale, 0, residual_scale)),
    ):
        cax = fig.add_axes((left, 0.095, 0.335, 0.012))
        bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks)
        bar.outline.set_visible(False)
        bar.set_label(label, fontsize=13, color=MUTED, labelpad=3)
        bar.ax.tick_params(labelsize=11, colors=MUTED, size=2)
    fig.text(
        0.055,
        0.007,
        f"Direct calculations at {metadata['grid']['evaluated_point_count_per_case']:,} points in the x = 0 plane; no interpolation. RMS values use 912 postprocessing acceptance samples.\nSmoothed, filtered unconnected contours; routing, feeds and fabrication validation remain separate design stages.",
        fontsize=10,
        color=MUTED,
        linespacing=1.6,
    )
    for extension in ("png", "svg"):
        path = OUTPUT / f"field_validation.{extension}"
        fig.savefig(path, dpi=280, facecolor="white", metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            content = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
            path.write_text(content, encoding="utf-8", newline="\n")
    plt.close(fig)


def compute_and_render(cases: list[dict], metrics: list[dict] | dict) -> dict:
    """Write dense field evidence and figures, accepting case or full metadata."""
    case_metrics = metrics["cases"] if isinstance(metrics, dict) else metrics
    with threadpool_limits(limits=1):
        dense, metadata = compute(cases, case_metrics)
        np.savez_compressed(OUTPUT / "field_maps.npz", **dense)
        render(dense, metadata)
    metadata["artifact_sha256"] = {
        name: sha256((OUTPUT / name).read_bytes()).hexdigest()
        for name in ("field_maps.npz", "field_validation.png", "field_validation.svg")
    }
    (OUTPUT / "field_maps.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    return metadata


def main() -> None:
    from contour_processing import load_refined_cases

    cases, metrics = load_refined_cases()
    compute_and_render(cases, metrics)


if __name__ == "__main__":
    main()
