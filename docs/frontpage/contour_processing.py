"""Condition the gallery's saved winding contours without refitting their current.

The saved meshes have no contour face/barycentric provenance. Consequently this
gallery adapter uses the production smoother's periodic mesh-scale B-spline fit
in exact, physical developed-cylinder/plane coordinates, rather than pretending
to reconstruct the provenance required by MeshContourBSplineSmoother. It does
not alter the application defaults or the archived optimization results.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from coilgen.contour_smoothing import (  # noqa: E402
    _evaluate_bspline,
    _mesh_coupled_periodic_bspline,
    _polyline_self_intersects,
)
from coilgen.contours import _bidirectional_polyline_distance_stats  # noqa: E402
from coilgen.physics import biot_savart_wire_path  # noqa: E402
from coilgen.research.storage import json_value  # noqa: E402


SETTINGS = {
    "method": "production_mesh_scale_periodic_cubic_bspline_on_analytic_support",
    "mesh_wavelength_m": [0.035, 0.050],
    "mesh_frequency_gain": 0.1,
    "work_samples_per_mesh_length": 8,
    "target_segment_length_m": 0.00075,
    "maximum_adjacent_segment_angle_degrees": 4.0,
    "small_loop_max_length_m": 0.400,
    "small_orbit_max_relative_rms_Bz_percent": 6.5,
    "final_relative_rms_Bz_budget_percent": 4.0,
    "symmetry_geometry_tolerance_m": 0.000020,
    "untouched_holdout_resolution": 14,
}
CONTOUR_KEY = re.compile(r"part_(\d+)_contour_(\d+)_points_m$")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _relative_rms(actual: np.ndarray, target: np.ndarray) -> float:
    return float(100 * np.linalg.norm(actual[:, 2] - target[:, 2]) / np.linalg.norm(target[:, 2]))


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _holdout(arrays: dict, metrics: dict, case_index: int) -> tuple[dict, dict]:
    """Evaluate a frozen contour selection on a new grid, without further tuning."""
    from field_maps import _contour_field, _parts, _sheet_field
    from coilgen.config import CoilGenConfig

    radius = float(metrics["config"]["target_region_radius"])
    resolution = SETTINGS["untouched_holdout_resolution"]
    axis = (np.arange(resolution) + 0.5) * 2 * radius / resolution - radius
    points = np.array(np.meshgrid(axis, axis, axis)).reshape(3, -1).T
    points = points[np.linalg.norm(points, axis=1) <= radius]
    design_axis = np.linspace(-2 * radius, 2 * radius, metrics["config"]["target_region_resolution"])
    design_points = np.array(np.meshgrid(design_axis, design_axis, design_axis)).reshape(3, -1).T
    design_points = design_points[np.linalg.norm(design_points, axis=1) <= radius]
    overlaps = {}
    for name, previous in (("optimization", design_points), ("conditioning_acceptance", arrays["validation_points_m"])):
        overlaps[name] = int(
            np.count_nonzero(
                np.any(np.all(np.isclose(points[:, None], previous[None], rtol=0, atol=1e-12), axis=2), axis=1)
            )
        )
    if any(overlaps.values()):
        raise RuntimeError("Fresh holdout points overlap an optimization or contour-conditioning sample.")
    target = np.zeros_like(points)
    target[:, 2] = 0.01 * points[:, case_index + 1]
    config = CoilGenConfig(gauss_order=metrics["config"]["gauss_order"], parallel_worker_count=1, progress_flag=False)
    sheet = _sheet_field(_parts(arrays, metrics), points, config)
    contours = _contour_field(arrays, metrics, points)
    if not all(np.all(np.isfinite(values)) for values in (sheet, contours)):
        raise RuntimeError("Fresh holdout field is nonfinite.")
    return {
        "holdout_points_m": points,
        "holdout_target_T": target,
        "holdout_sheet_field_T": sheet,
        "holdout_contour_field_T": contours,
    }, {
        "point_count": len(points),
        "resolution": resolution,
        "overlap_point_counts": overlaps,
        "target_region_radius_m": radius,
        "optimization_point_count": len(design_points),
        "conditioning_acceptance_point_count": len(arrays["validation_points_m"]),
        "sampling": "equal-weight Cartesian cell centers clipped to the 90 mm diameter spherical DSV",
        "selection_and_smoothing_frozen_before_evaluation": True,
        "used_to_choose_smoothing_or_pruning": False,
        "current_refitted": False,
        "sheet_relative_rms_Bz_error_percent": _relative_rms(sheet, target),
        "contour_relative_rms_Bz_error_percent": _relative_rms(contours, target),
        "contour_error_under_4_percent": bool(_relative_rms(contours, target) < 4.0),
        "independent_mesh_or_quadrature_convergence": False,
    }


def _area(points: np.ndarray) -> float:
    return float(0.5 * np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]))


def _chart(points: np.ndarray, case_index: int) -> np.ndarray:
    if case_index == 0:
        return np.column_stack((0.12 * np.unwrap(np.arctan2(points[:, 1], points[:, 0])), points[:, 2]))
    return points[:, :2].copy()


def _map(points: np.ndarray, case_index: int, plane_z: float) -> np.ndarray:
    if case_index == 0:
        return np.column_stack((0.12 * np.cos(points[:, 0] / 0.12), 0.12 * np.sin(points[:, 0] / 0.12), points[:, 1]))
    return np.column_stack((points, np.full(len(points), plane_z)))


def _smooth(points: np.ndarray, case_index: int) -> tuple[np.ndarray, dict]:
    uv = _chart(points, case_index)
    if not np.allclose(uv[-1], uv[0], rtol=0, atol=1e-12):
        raise RuntimeError("A gallery contour winds around the cylinder seam or is open.")
    uv[-1] = uv[0]
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))))
    parameters = cumulative / cumulative[-1]
    fit, fitted_parameters, translation, work_count = _mesh_coupled_periodic_bspline(
        uv,
        parameters,
        original_length_m=float(cumulative[-1]),
        mesh_scale_m=SETTINGS["mesh_wavelength_m"][case_index],
        mesh_gain=SETTINGS["mesh_frequency_gain"],
        work_samples_per_mesh_length=SETTINGS["work_samples_per_mesh_length"],
    )
    spline = fit, fitted_parameters, translation
    dense_u = np.linspace(0.0, 1.0, 8193)
    dense_uv = _evaluate_bspline(spline, dense_u)
    if np.sign(_area(dense_uv)) != np.sign(_area(uv)):
        raise RuntimeError("Smoothing reversed a contour's signed circulation.")
    if _polyline_self_intersects(dense_uv):
        raise RuntimeError("Smoothing produced a self-intersecting contour.")
    dense_3d = _map(dense_uv, case_index, float(points[0, 2]))
    cumulative_3d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense_3d, axis=0), axis=1))))
    sample_count = int(np.ceil(cumulative_3d[-1] / SETTINGS["target_segment_length_m"]))
    output_u = np.interp(np.linspace(0.0, cumulative_3d[-1], sample_count + 1), cumulative_3d, dense_u)
    for _ in range(8):
        output_uv = _evaluate_bspline(spline, output_u)
        output = _map(output_uv, case_index, float(points[0, 2]))
        output[-1] = output[0]
        segment = np.diff(output, axis=0)
        tangent = segment / np.linalg.norm(segment, axis=1)[:, None]
        angle = np.arccos(np.clip(np.einsum("ij,ij->i", tangent, np.roll(tangent, 1, axis=0)), -1, 1))
        tight_vertices = np.flatnonzero(angle > np.radians(SETTINGS["maximum_adjacent_segment_angle_degrees"]))
        if not len(tight_vertices):
            break
        intervals = np.unique(np.concatenate((tight_vertices, (tight_vertices - 1) % len(segment))))
        output_u = np.sort(np.concatenate((output_u, 0.5 * (output_u[intervals] + output_u[intervals + 1]))))
    else:
        raise RuntimeError("Curvature-adaptive sampling did not resolve the closed spline.")
    if case_index == 0:
        if np.max(np.abs(output[:, 2])) > 0.15 + 1e-12:
            raise RuntimeError("A smoothed contour left the cylinder's axial support.")
    elif np.max(np.abs(output[:, :2])) > 0.15 + 1e-12:
        raise RuntimeError("A smoothed contour left its planar support.")
    source_parameter_fit = _map(_evaluate_bspline(spline, parameters), case_index, float(points[0, 2]))
    segment = np.diff(output, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    tangents = segment / segment_length[:, None]
    turn_angle = np.arccos(np.clip(np.einsum("ij,ij->i", tangents, np.roll(tangents, 1, axis=0)), -1, 1))
    radius = 0.5 * (segment_length + np.roll(segment_length, 1)) / np.maximum(turn_angle, 1e-15)
    return output, {
        "source_point_count": int(len(points)),
        "output_point_count": int(len(output)),
        "source_length_m": _length(points),
        "output_length_m": _length(output),
        "source_signed_chart_area_m2": _area(uv),
        "output_signed_chart_area_m2": _area(output_uv),
        "max_same_parameter_displacement_m": float(np.max(np.linalg.norm(source_parameter_fit - points, axis=1))),
        "max_output_segment_length_m": float(segment_length.max()),
        "max_adjacent_segment_angle_degrees": float(np.degrees(turn_angle.max())),
        "minimum_sampled_centerline_bend_radius_m": float(radius.min()),
        "spline_work_sample_count": int(work_count),
        "closed": True,
        "orientation_preserved": True,
        "self_intersections": 0,
        "analytic_support": "radius_0.12_m_cylinder" if case_index == 0 else f"plane_z_{points[0, 2]:.3f}_m",
    }


def _symmetry_groups(keys: list[str], arrays: dict, case_index: int) -> list[list[str]]:
    """Match complete reflection orbits before any removal, including both planes."""
    pending = set(keys)
    groups = []
    while pending:
        key = min(pending, key=lambda item: tuple(map(int, CONTOUR_KEY.fullmatch(item).groups())))
        source = arrays[key]
        lengths = {other: _length(arrays[other]) for other in pending}
        same_length = [other for other in pending if abs(lengths[other] - lengths[key]) < 2e-5]
        group = []
        for other in same_length:
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                distance = _bidirectional_polyline_distance_stats(source * np.asarray(signs), arrays[other])["max"]
                if distance <= SETTINGS["symmetry_geometry_tolerance_m"]:
                    group.append(other)
                    break
        expected_size = 4 if case_index == 0 else 2
        if len(group) != expected_size:
            raise RuntimeError(f"Incomplete gallery symmetry orbit: {key} matches {group}.")
        group.sort(key=lambda item: tuple(map(int, CONTOUR_KEY.fullmatch(item).groups())))
        groups.append(group)
        pending.difference_update(group)
    return groups


def _prepare_case(arrays: dict, metrics: dict, case_index: int) -> tuple[dict, dict, dict]:
    points, target = arrays["validation_points_m"], arrays["target_T"]
    if len(points) != 912:
        raise RuntimeError("Gallery conditioning must reuse all 912 original independent evaluation points.")
    keys = sorted(key for key in arrays if CONTOUR_KEY.fullmatch(key))
    currents = metrics["contour_currents_A"]

    def field(key: str, path: np.ndarray) -> np.ndarray:
        part_index = int(CONTOUR_KEY.fullmatch(key).group(1))
        return currents[part_index] * biot_savart_wire_path(path, points, worker_count=1)

    raw_fields = {key: field(key, arrays[key]) for key in keys}
    raw_field = sum(raw_fields.values(), start=np.zeros_like(target))
    if not np.allclose(raw_field, arrays["contour_field_T"], rtol=1e-12, atol=1e-16):
        raise RuntimeError("The archived contour field does not match its saved geometry and current.")
    groups = _symmetry_groups(keys, arrays, case_index)
    target_norm = np.linalg.norm(target[:, 2])
    candidates = []
    for group in groups:
        group_field = sum((raw_fields[key] for key in group), start=np.zeros_like(target))
        contribution = float(100 * np.linalg.norm(group_field[:, 2]) / target_norm)
        if (
            max(_length(arrays[key]) for key in group) <= SETTINGS["small_loop_max_length_m"]
            and contribution <= SETTINGS["small_orbit_max_relative_rms_Bz_percent"]
        ):
            candidates.append((contribution, group, group_field))
    candidates.sort(key=lambda item: item[0])
    keep = set(keys)
    combined = raw_field.copy()
    removal_records = []
    joint_proposed = raw_field - sum((item[2] for item in candidates), start=np.zeros_like(target))
    joint_accepted = _relative_rms(joint_proposed, target) <= SETTINGS["final_relative_rms_Bz_budget_percent"]
    for contribution, group, group_field in candidates:
        proposed = joint_proposed if joint_accepted else combined - group_field
        accepted = _relative_rms(proposed, target) <= SETTINGS["final_relative_rms_Bz_budget_percent"]
        removal_records.append(
            {
                "source_contour_ids": group,
                "raw_group_relative_rms_Bz_contribution_percent": contribution,
                "raw_proposed_relative_rms_Bz_error_percent": _relative_rms(proposed, target),
                "raw_residual_budget_passed": accepted,
                "joint_candidate_selection": joint_accepted,
                "removed": accepted,
            }
        )
        if accepted:
            keep.difference_update(group)
            combined -= group_field
    smoothed, diagnostics, smoothed_fields = {}, {}, {}

    def condition(key: str) -> None:
        if key not in smoothed:
            smoothed[key], diagnostics[key] = _smooth(arrays[key], case_index)
            smoothed_fields[key] = field(key, smoothed[key])

    for key in sorted(keep):
        condition(key)
    final_field = sum((smoothed_fields[key] for key in sorted(keep)), start=np.zeros_like(target))
    # A small raw loop can become necessary once the rest of the winding is
    # smoothed. Restore whole symmetry orbits rather than exceeding the budget.
    for record in reversed(removal_records):
        if _relative_rms(final_field, target) <= SETTINGS["final_relative_rms_Bz_budget_percent"]:
            break
        if record["removed"]:
            for key in record["source_contour_ids"]:
                condition(key)
                final_field += smoothed_fields[key]
                keep.add(key)
            record["removed"] = False
            record["restored_after_smoothing_to_preserve_field_budget"] = True
    final_error = _relative_rms(final_field, target)
    if final_error > SETTINGS["final_relative_rms_Bz_budget_percent"]:
        raise RuntimeError(f"Conditioned case {case_index} exceeds the 4% RMS Bz field budget: {final_error}.")
    raw_retained_field = sum((raw_fields[key] for key in sorted(keep)), start=np.zeros_like(target))
    output = {key: value for key, value in arrays.items() if not CONTOUR_KEY.fullmatch(key)}
    output.update({key: smoothed[key] for key in sorted(keep)})
    output["contour_field_T"] = final_field
    refined_metrics = deepcopy(metrics)
    refined_metrics.update(
        {
            "array_file": f"refined_case_{case_index}.npz",
            "unconnected_contour_count": len(keep),
            "original_unconnected_contour_count": len(keys),
            "removed_contour_count": len(keys) - len(keep),
            "contour_relative_rms_Bz_error_percent": final_error,
            "contour_peak_Bz_error_uT": float(1e6 * np.max(np.abs(final_field[:, 2] - target[:, 2]))),
            "original_contour_relative_rms_Bz_error_percent": _relative_rms(raw_field, target),
            "contour_processing": "contour_processing.json",
            "current_refitted": False,
        }
    )
    records = []
    for key in keys:
        records.append(
            {
                "source_contour_id": key,
                "retained": key in keep,
                "signed_current_A": float(currents[int(CONTOUR_KEY.fullmatch(key).group(1))]),
                "raw_relative_rms_Bz_contribution_percent": float(
                    100 * np.linalg.norm(raw_fields[key][:, 2]) / target_norm
                ),
                **diagnostics.get(
                    key, {"source_point_count": int(len(arrays[key])), "source_length_m": _length(arrays[key])}
                ),
            }
        )
    report = {
        "name": metrics["name"],
        "source_array_file": f"case_{case_index}.npz",
        "output_array_file": refined_metrics["array_file"],
        "source_sha256": _digest(OUTPUT / f"case_{case_index}.npz"),
        "kept_source_contour_ids": sorted(keep),
        "dropped_source_contour_ids": sorted(set(keys) - keep),
        "symmetry_groups": groups,
        "pruning_group_decisions": removal_records,
        "original_relative_rms_Bz_error_percent": _relative_rms(raw_field, target),
        "raw_retained_relative_rms_Bz_error_percent": _relative_rms(raw_retained_field, target),
        "conditioned_relative_rms_Bz_error_percent": final_error,
        "field_change_relative_rms_Bz_percent": float(
            100 * np.linalg.norm(final_field[:, 2] - raw_field[:, 2]) / target_norm
        ),
        "final_field_budget_passed": True,
        "independent_point_count": len(points),
        "current_refitted": False,
        "contours": records,
    }
    holdout_arrays, holdout_report = _holdout(output, refined_metrics, case_index)
    output.update(holdout_arrays)
    report["untouched_holdout"] = holdout_report
    refined_metrics["holdout_independent_point_count"] = holdout_report["point_count"]
    refined_metrics["holdout_sheet_relative_rms_Bz_error_percent"] = holdout_report[
        "sheet_relative_rms_Bz_error_percent"
    ]
    refined_metrics["holdout_contour_relative_rms_Bz_error_percent"] = holdout_report[
        "contour_relative_rms_Bz_error_percent"
    ]
    return output, refined_metrics, report


def prepare_cases(cases: list[dict], case_metrics: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return field-checked, smooth retained paths and metrics, with verified cache."""
    # Fresh optimization metrics contain NumPy support extents/bounds, whereas
    # --reuse-data metrics have already passed through JSON. Normalize both
    # paths before hashing, copying or writing the conditioning evidence.
    case_metrics = json_value(case_metrics)
    if len(cases) != 2 or len(case_metrics) != 2:
        raise ValueError("The gallery conditioning adapter requires its two documented source cases.")
    for index, arrays in enumerate(cases):
        with np.load(OUTPUT / f"case_{index}.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(archive[key], arrays[key]) for key in arrays
            ):
                raise ValueError(f"Case {index}: provided arrays differ from the archived optimization result.")
    provenance = {
        "settings": SETTINGS,
        "conditioning_source_sha256": _digest(Path(__file__)),
        "production_smoother_sha256": _digest(ROOT / "src/coilgen/contour_smoothing.py"),
        "production_contours_sha256": _digest(ROOT / "src/coilgen/contours.py"),
        "physics_source_sha256": _digest(ROOT / "src/coilgen/physics.py"),
        "source_arrays_sha256": {
            f"case_{index}.npz": _digest(OUTPUT / f"case_{index}.npz") for index in range(len(cases))
        },
        "source_contour_currents_A": [case["contour_currents_A"] for case in case_metrics],
        "source_case_metrics_sha256": sha256(json.dumps(case_metrics, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    report_path = OUTPUT / "contour_processing.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("provenance") == provenance
            and all(
                (OUTPUT / name).exists() and _digest(OUTPUT / name) == digest
                for name, digest in report.get("output_arrays_sha256", {}).items()
            )
            and len(report.get("output_arrays_sha256", {})) == len(cases)
        ):
            loaded = []
            for index in range(len(cases)):
                with np.load(OUTPUT / f"refined_case_{index}.npz", allow_pickle=False) as archive:
                    loaded.append({key: archive[key] for key in archive.files})
            return loaded, report["refined_case_metrics"]
    prepared = [_prepare_case(case, metrics, index) for index, (case, metrics) in enumerate(zip(cases, case_metrics))]
    refined_cases, refined_metrics, reports = map(list, zip(*prepared))
    for index, arrays in enumerate(refined_cases):
        destination = OUTPUT / f"refined_case_{index}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    report = {
        "description": "Field-checked gallery contour conditioning; independent turns, no feeds or manufacturing release.",
        "provenance": provenance,
        "smoothing_geometry": "Physical developed-cylinder (120 mm radius) and planar coordinates; periodic cubic B-spline, arc-length sampled.",
        "pruning_criterion": "Complete reflection orbits only; each loop length <=400 mm, orbit RMS Bz contribution <=6.5% of target norm, and final smoothed winding RMS Bz error <=4% at unchanged signed current.",
        "manufacturing_validation": False,
        "current_refitted": False,
        "cases": reports,
        "refined_case_metrics": refined_metrics,
        "output_arrays_sha256": {
            f"refined_case_{index}.npz": _digest(OUTPUT / f"refined_case_{index}.npz") for index in range(len(cases))
        },
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return refined_cases, refined_metrics


def load_refined_cases() -> tuple[list[dict], list[dict]]:
    metadata = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    cases = []
    for index in range(len(metadata["cases"])):
        with np.load(OUTPUT / f"case_{index}.npz", allow_pickle=False) as archive:
            cases.append({key: archive[key] for key in archive.files})
    return prepare_cases(cases, metadata["cases"])


if __name__ == "__main__":
    _cases, _metrics = load_refined_cases()
    for _metric in _metrics:
        print(
            f"{_metric['name']}: {_metric['unconnected_contour_count']} smooth contours, "
            f"{_metric['removed_contour_count']} removed, "
            f"{_metric['contour_relative_rms_Bz_error_percent']:.6f}% RMS Bz error",
            flush=True,
        )
