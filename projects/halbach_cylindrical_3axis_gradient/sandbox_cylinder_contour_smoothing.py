"""Deterministic coarse-cylinder sandbox for contour-smoothing validation.

The synthetic transverse-gradient stream function produces two closed saddle
loops on a deliberately coarse circular cylinder. One loop is smoothed by
CoilGen's production path and evaluated in the cylinder's exact developed
coordinates ``(radius * azimuth, axial position)``.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parents[1]
for import_path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

from coilgen import CoilGenConfig, CoilPart, ContourLine, Mesh, regularize_contour_paths
from coilgen.contours import extract_contours
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from projects.halbach_cylindrical_3axis_gradient.paths import halbach_output_dir


CYLINDER_RADIUS_M = 0.084
CYLINDER_LENGTH_M = 0.270
ANGULAR_DIVISIONS = 24
LENGTH_DIVISIONS = 14
CONTOUR_LEVEL = 0.30

SMOOTHING_SCALE_MODE = "mesh_coupled"
SMOOTHING_MESH_GAIN = 0.10
SMOOTHING_WORK_SAMPLES_PER_MESH_LENGTH = 6
SMOOTHING_OUTPUT_SAMPLES_PER_MESH_LENGTH = 8
SMOOTHING_MAX_DISPLACEMENT_M = 0.001
SMOOTHING_TARGET_SEGMENT_LENGTH_M = 0.001
SMOOTHING_PATCH_RINGS = 4
MAX_WIGGLE_RMS_RATIO = 0.15
MAX_CURVATURE_SECOND_DERIVATIVE_RMS_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class DevelopedContourMetrics:
    high_frequency_curvature_rms_per_m: float
    curvature_second_derivative_rms_per_m3: float
    segment_count: int


@dataclass(frozen=True, slots=True)
class CylinderSmoothingSandboxResult:
    mesh: Mesh
    before: ContourLine
    after: ContourLine
    before_metrics: DevelopedContourMetrics
    after_metrics: DevelopedContourMetrics
    smoothing_diagnostics: dict[str, object]
    max_surface_reconstruction_error_m: float

    @property
    def wiggle_rms_ratio(self) -> float:
        return (
            self.after_metrics.high_frequency_curvature_rms_per_m
            / self.before_metrics.high_frequency_curvature_rms_per_m
        )

    @property
    def curvature_second_derivative_rms_ratio(self) -> float:
        return (
            self.after_metrics.curvature_second_derivative_rms_per_m3
            / self.before_metrics.curvature_second_derivative_rms_per_m3
        )


def run_sandbox() -> CylinderSmoothingSandboxResult:
    mesh = normalize_mesh(
        build_cylinder_mesh(
            CYLINDER_LENGTH_M,
            CYLINDER_RADIUS_M,
            ANGULAR_DIVISIONS,
            LENGTH_DIVISIONS,
            1.0,
            0.0,
            0.0,
            0.0,
        )
    )
    part = CoilPart(mesh=mesh)
    azimuth = np.arctan2(mesh.vertices[:, 0], mesh.vertices[:, 1])
    axial = mesh.vertices[:, 2]
    axial_half_extent = float(np.max(np.abs(axial)))
    axial_envelope = 1.0 - np.square(axial / axial_half_extent)
    part.stream_function = np.cos(azimuth) * axial_envelope
    part.potential_level_list = np.array([CONTOUR_LEVEL, -CONTOUR_LEVEL], dtype=float)
    part.contour_step = 2.0 * CONTOUR_LEVEL
    extract_contours([part], CoilGenConfig(min_point_loop_number=8))
    if len(part.contour_lines) != 2:
        raise RuntimeError(f"Expected two sandbox gradient loops, found {len(part.contour_lines)}.")
    before = next(contour for contour in part.contour_lines if contour.potential > 0.0)

    regularize_contour_paths(
        [part],
        CoilGenConfig(
            contour_path_regularization_mode="parameterized_bspline",
            contour_resample_target_segment_length=SMOOTHING_TARGET_SEGMENT_LENGTH_M,
            contour_smoothing_scale_mode=SMOOTHING_SCALE_MODE,
            contour_smoothing_mesh_gain=SMOOTHING_MESH_GAIN,
            contour_smoothing_work_samples_per_mesh_length=SMOOTHING_WORK_SAMPLES_PER_MESH_LENGTH,
            contour_smoothing_output_samples_per_mesh_length=SMOOTHING_OUTPUT_SAMPLES_PER_MESH_LENGTH,
            contour_smoothing_patch_rings=SMOOTHING_PATCH_RINGS,
            contour_smoothing_max_displacement=SMOOTHING_MAX_DISPLACEMENT_M,
            contour_smoothing_preserve_min_turn_distance=False,
            min_point_loop_number=8,
        ),
    )
    if len(part.contour_lines) != 2:
        raise RuntimeError("Smoothing removed a sandbox gradient loop.")
    after = next(contour for contour in part.contour_lines if contour.potential > 0.0)
    result = CylinderSmoothingSandboxResult(
        mesh=mesh,
        before=before,
        after=after,
        before_metrics=_developed_metrics(before),
        after_metrics=_developed_metrics(after),
        smoothing_diagnostics=dict(part.contour_level_diagnostics),
        max_surface_reconstruction_error_m=_surface_reconstruction_error(mesh, after),
    )
    validate_sandbox(result)
    return result


def validate_sandbox(result: CylinderSmoothingSandboxResult) -> None:
    diagnostics = result.smoothing_diagnostics
    if not bool(diagnostics.get("contour_path_smoothing_applied")):
        raise AssertionError(f"Cylinder smoothing was not applied: {diagnostics}")
    if diagnostics.get("contour_path_smoothing_method") != "developed_cylinder_periodic_bspline":
        raise AssertionError(f"Unexpected cylinder parameterization: {diagnostics}")
    if result.wiggle_rms_ratio >= MAX_WIGGLE_RMS_RATIO:
        raise AssertionError(
            f"Developed-plane curvature wiggle ratio {result.wiggle_rms_ratio:.3f} exceeds {MAX_WIGGLE_RMS_RATIO:.3f}."
        )
    if result.curvature_second_derivative_rms_ratio >= MAX_CURVATURE_SECOND_DERIVATIVE_RMS_RATIO:
        raise AssertionError(
            "Developed-plane curvature second-derivative ratio "
            f"{result.curvature_second_derivative_rms_ratio:.3f} exceeds "
            f"{MAX_CURVATURE_SECOND_DERIVATIVE_RMS_RATIO:.3f}."
        )
    if result.max_surface_reconstruction_error_m > 1.0e-11:
        raise AssertionError(
            f"Smoothed points left the triangle mesh by {result.max_surface_reconstruction_error_m:.3e} m."
        )


def save_sandbox_artifacts(result: CylinderSmoothingSandboxResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "cylinder_contour_smoothing_metrics.csv"
    rows = (
        {
            "curve": "before",
            **asdict(result.before_metrics),
            "max_surface_reconstruction_error_m": "",
        },
        {
            "curve": "after",
            **asdict(result.after_metrics),
            "max_surface_reconstruction_error_m": result.max_surface_reconstruction_error_m,
        },
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output_dir / "cylinder_contour_smoothing_developed_projection.png"
    _save_figure(result, figure_path)
    return metrics_path, figure_path


def _developed_projection(contour: ContourLine) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(contour.v, dtype=float)
    azimuth = np.unwrap(np.arctan2(points[:, 0], points[:, 1]))
    if azimuth[-1] < azimuth[0]:
        azimuth = -azimuth
    return CYLINDER_RADIUS_M * (azimuth - azimuth[0]), points[:, 2]


def _developed_samples(
    contour: ContourLine,
    sample_count: int = 1200,
) -> tuple[np.ndarray, np.ndarray, float]:
    circumferential, axial = _developed_projection(contour)
    points = np.column_stack((circumferential, axial))
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    query = np.linspace(0.0, total_length, int(sample_count) + 1)[:-1]
    sampled = np.column_stack(
        (
            np.interp(query, cumulative, circumferential),
            np.interp(query, cumulative, axial),
        )
    )
    return sampled[:, 0], sampled[:, 1], total_length / int(sample_count)


def _developed_curvature_trace(contour: ContourLine) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    circumferential, axial, spacing = _developed_samples(contour)
    points = np.column_stack((circumferential, axial))
    first_derivative = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / (2.0 * spacing)
    second_derivative = (np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0)) / np.square(spacing)
    speed = np.linalg.norm(first_derivative, axis=1)
    curvature = (
        first_derivative[:, 0] * second_derivative[:, 1] - first_derivative[:, 1] * second_derivative[:, 0]
    ) / np.maximum(np.power(speed, 3), 1.0e-30)
    return circumferential, axial, curvature


def _developed_metrics(contour: ContourLine) -> DevelopedContourMetrics:
    from scipy.signal import savgol_filter  # type: ignore

    _, _, curvature = _developed_curvature_trace(contour)
    _, _, spacing = _developed_samples(contour)
    low_frequency_curvature = savgol_filter(curvature, 101, 3, mode="wrap")
    curvature_first_derivative = (np.roll(curvature, -1) - np.roll(curvature, 1)) / (2.0 * spacing)
    curvature_second_derivative = (np.roll(curvature_first_derivative, -1) - np.roll(curvature_first_derivative, 1)) / (
        2.0 * spacing
    )
    return DevelopedContourMetrics(
        high_frequency_curvature_rms_per_m=float(np.sqrt(np.mean(np.square(curvature - low_frequency_curvature)))),
        curvature_second_derivative_rms_per_m3=float(np.sqrt(np.mean(np.square(curvature_second_derivative)))),
        segment_count=int(len(contour.v) - 1),
    )


def _surface_reconstruction_error(mesh: Mesh, contour: ContourLine) -> float:
    if (
        contour.surface_faces is None
        or contour.surface_barycentric is None
        or contour.segment_faces is None
        or contour.segment_barycentric is None
    ):
        return float("inf")
    reconstructed = np.asarray(
        [
            barycentric @ mesh.vertices[mesh.faces[int(face)]]
            for face, barycentric in zip(contour.surface_faces, contour.surface_barycentric)
        ],
        dtype=float,
    )
    vertex_error = float(np.max(np.linalg.norm(reconstructed - contour.v, axis=1)))
    segment_error = 0.0
    for segment_index, (face, barycentric) in enumerate(zip(contour.segment_faces, contour.segment_barycentric)):
        triangle = mesh.vertices[mesh.faces[int(face)]]
        reconstructed_segment = barycentric @ triangle
        segment_error = max(
            segment_error,
            float(
                np.max(
                    np.linalg.norm(
                        reconstructed_segment - contour.v[segment_index : segment_index + 2],
                        axis=1,
                    )
                )
            ),
        )
    return max(vertex_error, segment_error)


def _save_figure(result: CylinderSmoothingSandboxResult, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "coilgen_matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    before_x, before_z = _developed_projection(result.before)
    after_x, after_z = _developed_projection(result.after)
    before_sample_x, _, before_curvature = _developed_curvature_trace(result.before)
    after_sample_x, _, after_curvature = _developed_curvature_trace(result.after)
    # Metrics use each curve's own origin; overlays must retain their physical
    # azimuth difference instead of aligning unrelated first samples.
    before_angle = np.unwrap(np.arctan2(result.before.v[:, 0], result.before.v[:, 1]))
    after_angle = np.unwrap(np.arctan2(result.after.v[:, 0], result.after.v[:, 1]))
    before_direction = -1.0 if before_angle[-1] < before_angle[0] else 1.0
    after_direction = -1.0 if after_angle[-1] < after_angle[0] else 1.0
    difference = after_angle[0] - before_angle[0]
    offset = CYLINDER_RADIUS_M * before_direction * np.arctan2(np.sin(difference), np.cos(difference))
    direction = before_direction / after_direction
    after_x = direction * after_x + offset
    after_sample_x = direction * after_sample_x + offset
    after_curvature = direction * after_curvature
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))
    axes[0].plot(before_x * 1.0e3, before_z * 1.0e3, color="0.65", linewidth=1.2, label="mesh contour")
    axes[0].plot(after_x * 1.0e3, after_z * 1.0e3, color="tab:orange", linewidth=2.0, label="smoothed")
    axes[0].set_xlabel("developed circumferential distance [mm]")
    axes[0].set_ylabel("axial position [mm]")
    axes[0].set_title("Coarse-cylinder contour in exact developed coordinates, shared azimuth origin")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(before_sample_x * 1.0e3, before_curvature, color="0.65", linewidth=1.0, label="before")
    axes[1].plot(after_sample_x * 1.0e3, after_curvature, color="tab:orange", linewidth=1.5, label="after")
    axes[1].set_xlabel("developed circumferential distance [mm]")
    axes[1].set_ylabel("in-plane curvature [1/m]")
    axes[1].set_title(
        f"wiggle RMS ratio={result.wiggle_rms_ratio:.3f}; "
        f"curvature d2 RMS ratio={result.curvature_second_derivative_rms_ratio:.3f}"
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.94, hspace=0.42)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    sandbox_result = run_sandbox()
    sandbox_output = halbach_output_dir("cylinder_contour_smoothing_sandbox", create=True)
    saved = save_sandbox_artifacts(sandbox_result, sandbox_output)
    print(f"Developed-plane wiggle RMS ratio: {sandbox_result.wiggle_rms_ratio:.4f}")
    print(f"Curvature second-derivative RMS ratio: {sandbox_result.curvature_second_derivative_rms_ratio:.4f}")
    print(f"Maximum triangle reconstruction error: {sandbox_result.max_surface_reconstruction_error_m:.3e} m")
    for saved_path in saved:
        print(saved_path)
