"""Paired gallery runs. Each on/off pair starts from the identical solved sheet.

Run from the checkout with PYTHONPATH=src:
python validation/gui_review_2026-11-03/check_smoothing.py
Archived checkpoints are preferred. A changed engine digest requires a new
sheet. Use --fresh-baseline to recompute both variants explicitly.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

import coilgen
from coilgen.application.service import DesignService
from coilgen.application.templates import TEMPLATE_SPECS, template_assembly
from coilgen.persistence import numerical_source_digest
from coilgen.pipeline_stages import PipelineCheckpoint


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE.parent / "browser_workbench_2026-11-03" / "retest-matrix.csv"
RUNS = ROOT / "tmp" / "gui_review_smoothing" / "runs"
SHEET_ARRAYS = (
    "stream_function_A",
    "sheet_field_T",
    "validation_points_m",
    "validation_target_T",
    "validation_field_T",
)


def composition(result):
    def percent(scope):
        value = result["wire_fields"][scope].get("at_stream_current", {}).get("relative_rms_error")
        return None if value is None else 100 * value

    bodies = result["pipeline_sweeps"]["bodies"]
    radii = [b["diagnostics"].get("sweep_final_min_bend_radius_m") for b in bodies]
    return {
        "fixed_current_A": result["wire_fields"]["contour"]["at_stream_current"]["current_A"],
        "contour_RMS_percent": percent("contour"),
        "routed_RMS_percent": percent("routed"),
        "fabrication_RMS_percent": result["fabrication"]["field_metrics"].get("dsv_relative_residual_percent"),
        "min_pipeline_bend_mm": min((r * 1000 for r in radii if r is not None), default=None),
        "sheet_passed": result["summary"]["sheet_accepted"],
        "sweep_passed": result["pipeline_sweeps"]["checks_passed"],
        "fabrication_passed": result["fabrication"]["ready"],
        "construction_released": result["summary"]["construction_released"],
    }


def turn_angles_deg(loops, step_m=0.003):
    """World-frame tangent changes sampled at equal arc length, pooled across turns."""
    angles = []
    for points in loops:
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arclength = np.r_[0.0, np.cumsum(lengths)]
        stations = np.linspace(0.0, arclength[-1], max(4, round(arclength[-1] / step_m)) + 1)
        samples = np.column_stack([np.interp(stations, arclength, axis) for axis in points.T])
        direction = np.diff(samples, axis=0)
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        tangents = np.sum(direction * np.roll(direction, 1, axis=0), axis=1)
        angles.append(np.degrees(np.arccos(np.clip(tangents, -1.0, 1.0))))
    return np.concatenate(angles)


def compute(case):
    template, channel_index, baseline, fresh = case
    project = template_assembly(template).channels[channel_index].project
    case_name = f"{template}__{channel_index}"
    directory = RUNS / case_name
    source = Path(baseline["job_directory"])
    if fresh or not (source / "checkpoints" / "sheet.cgstage").is_file():
        source = RUNS.parent / "fresh_original" / case_name
        directory = RUNS.parent / "fresh_smoothed" / case_name
        if not (source / "result.json").exists():
            DesignService.run(project.update_config({"contour_path_regularization_mode": "off"}), source, threads=2)
        baseline_id = f"fresh/{case_name}"
    else:
        baseline_id = baseline["job_id"]
    checkpoint = source / "checkpoints" / "sheet.cgstage"
    sheet_origin = "fresh paired sheet" if baseline_id.startswith("fresh/") else "archived sheet checkpoint"
    baseline_digest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["numerical_source_sha256"]
    active_digest = numerical_source_digest(Path(coilgen.__file__).parent)
    if not (directory / "result.json").exists() and baseline_digest != active_digest:
        # Recompute the current model; never bypass a checkpoint's source fingerprint.
        directory = RUNS.parent / "recomputed" / case_name
        new_sheet = RUNS.parent / "replacement_sheets" / case_name
        if not (new_sheet / "result.json").exists():
            DesignService.run(project, new_sheet, threads=2, pipeline_stage="sheet", fabricate=False)
        checked = DesignService.load_run(new_sheet, array_names=SHEET_ARRAYS)
        checkpoint = new_sheet / "checkpoints" / "sheet.cgstage"
        with np.load(source / "scientific_result.npz", allow_pickle=False) as old:
            identical = all(np.array_equal(old[key], checked.arrays[key]) for key in SHEET_ARRAYS)
        if identical:
            sheet_origin = "recomputed sheet bit-identical to archive"
        else:
            source = RUNS.parent / "replacement_original" / case_name
            if not (source / "result.json").exists():
                raw = project.update_config({"contour_path_regularization_mode": "off"})
                DesignService.run(raw, source, threads=2, resume_checkpoint=str(checkpoint))
            baseline_id = f"replacement/{case_name}"
            sheet_origin = "fresh paired sheet"
    previous = DesignService.load_run(source, array_names=()).metadata
    if (directory / "result.json").is_file():
        result = DesignService.load_run(directory, array_names=()).metadata
        if result["project_model_fingerprint"] != project.model_fingerprint:
            raise ValueError(f"Previous run has a different model: {directory}")
    else:
        result = DesignService.run(project, directory, threads=2, resume_checkpoint=str(checkpoint))
    # Sum only the finished contour-stage records for this run, before pruning/routing.
    parts = result["contour_smoothing"]
    smoothed = sum(row["smoothed_count"] for row in parts)
    fit_rejections = sum(row["fit_rejected_count"] for row in parts)
    spacing_rejections = sum(row["spacing_rejected_count"] for row in parts)
    failures = {}
    for part in parts:
        for reason, count in part["failure_reasons"].items():
            failures[reason] = failures.get(reason, 0) + count
    with (
        np.load(source / "scientific_result.npz", allow_pickle=False) as old_arrays,
        np.load(directory / "scientific_result.npz", allow_pickle=False) as new_arrays,
    ):
        identical_sheet = all(np.array_equal(old_arrays[name], new_arrays[name]) for name in SHEET_ARRAYS)
        old_contours = [k for k in old_arrays if k.startswith("contour_") and k.endswith("_points_m")]
        new_contours = [k for k in new_arrays if k.startswith("contour_") and k.endswith("_points_m")]
        angles_before = turn_angles_deg([old_arrays[name] for name in old_contours])
        angles_after = turn_angles_deg([new_arrays[name] for name in new_contours])
        finite = all(np.isfinite(new_arrays[k]).all() for k in new_arrays if new_arrays[k].dtype.kind in "fc")
        saved_geometry = {}
        for key in new_arrays:
            if re.fullmatch(r"support_\d+_(vertices_m|faces)", key):
                np.testing.assert_array_equal(old_arrays[key], new_arrays[key])
                saved_geometry[key] = new_arrays[key]
        for prefix, arrays in (("original_", old_arrays), ("", new_arrays)):
            for key in arrays:
                if re.fullmatch(r"(contour_\d+_\d+|review_conductor_\d+)_points_m", key):
                    saved_geometry[prefix + key] = arrays[key]
    assert identical_sheet and finite
    assert old_contours == new_contours and len(new_contours) == smoothed + fit_rejections + spacing_rejections
    max_move = max(row["max_displacement_m"] for row in parts)
    assert max_move <= project.config.contour_smoothing_max_displacement * (1.0 + 1e-6)
    assert (
        result["wire_fields"]["contour"]["at_stream_current"]["current_A"]
        == previous["wire_fields"]["contour"]["at_stream_current"]["current_A"]
    )
    minimum_before, minimum_after = [], []
    original_parts = PipelineCheckpoint.load(source / "checkpoints" / "contours.cgstage").parts
    fitted_parts = PipelineCheckpoint.load(directory / "checkpoints" / "contours.cgstage").parts
    max_surface_error = 0.0
    for original, part in zip(original_parts, fitted_parts, strict=True):
        assert original.contour_step == part.contour_step
        for raw, fit in zip(original.contour_lines, part.contour_lines, strict=True):
            assert raw.potential == fit.potential and raw.current_orientation == fit.current_orientation
            assert raw.turn_multiplicity == fit.turn_multiplicity
            np.testing.assert_allclose(fit.v[0], fit.v[-1], atol=1e-12, rtol=0)
            triangles = part.mesh.vertices[part.mesh.faces[fit.segment_faces]]
            endpoints = np.einsum("nkj,nji->nki", fit.segment_barycentric, triangles)
            error = np.max(np.linalg.norm(endpoints - np.stack((fit.v[:-1], fit.v[1:]), axis=1), axis=-1))
            max_surface_error = max(max_surface_error, float(error))
        diagnostic = part.contour_level_diagnostics
        before = diagnostic["contour_path_smoothing_min_distance_before_m"]
        after = diagnostic["contour_path_smoothing_min_distance_after_m"]
        if np.isfinite(before):
            threshold = project.config.min_turn_distance
            required = min(threshold, before) if threshold > 0 else before
            assert after + max(1e-9, 1e-4 * required) >= required
            minimum_before.append(before)
            minimum_after.append(after)
    assert max_surface_error < 1e-10
    centerlines = HERE / "centerlines" / f"{case_name}.npz"
    centerlines.parent.mkdir(exist_ok=True)
    temporary = centerlines.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **saved_geometry)
    temporary.replace(centerlines)
    fields = {"template": template, "channel": project.name, "directory": str(directory.relative_to(ROOT))}
    fields.update(
        supports=len(parts),
        raw_contours=len(old_contours),
        smoothed=smoothed,
        fit_rejected=fit_rejections,
        spacing_rejected=spacing_rejections,
        failure_reasons=json.dumps(failures, sort_keys=True),
        max_displacement_mm=1000 * max_move,
        max_surface_reconstruction_error_m=max_surface_error,
        min_original_contour_spacing_mm=1000 * min(minimum_before) if minimum_before else None,
        min_smoothed_contour_spacing_mm=1000 * min(minimum_after) if minimum_after else None,
        smoothing_stage_seconds=next(
            row["stage_seconds"] for row in result["timing"] if row["stage"] == "regularized contour paths"
        ),
        sheet_and_holdout_identical=identical_sheet,
        finite_numeric_arrays=finite,
        baseline_job_id=baseline_id,
        sheet_origin=sheet_origin,
        smoothed_numerical_source_sha256=json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
            "numerical_source_sha256"
        ],
        original_request_sha256=previous["request_sha256"],
        smoothed_request_sha256=result["request_sha256"],
        centerlines_file=centerlines.relative_to(HERE).as_posix(),
        centerlines_sha256=hashlib.sha256(centerlines.read_bytes()).hexdigest(),
        original_p95_turn_angle_3mm_deg=float(np.percentile(angles_before, 95)),
        smoothed_p95_turn_angle_3mm_deg=float(np.percentile(angles_after, 95)),
    )
    for prefix, data in (("original", previous), ("smoothed", result)):
        fields.update({f"{prefix}_{k}": v for k, v in composition(data).items()})
    return fields


def main():
    # Windows terminals can lack the mathematical minus sign in a shim title.
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--fresh-baseline", action="store_true", help="Compute both variants using this installation")
    parser.add_argument("--case", action="append", help="Run selected template__index cases into a separate subset CSV")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    with BASELINE.open(encoding="utf-8-sig", newline="") as stream:
        baseline = {r["project_or_channel"]: r for r in csv.DictReader(stream)}
    cases = [
        (spec.id, index, baseline[channel.project.name], args.fresh_baseline)
        for spec in TEMPLATE_SPECS
        for index, channel in enumerate(template_assembly(spec.id).channels)
    ]
    cases.sort(key=lambda c: 0 if c[0] in {"cylinder", "biplanar", "conical_harmonic"} else 1)
    if args.case:
        cases = [case for case in cases if f"{case[0]}__{case[1]}" in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("Unknown case(s); use template__index.")
    output = HERE / "smoothing.csv" if not args.case else RUNS.parent / "subset.csv"
    rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(compute, case): case for case in cases}
        for future in as_completed(pending):
            case = pending[future]
            try:
                row = future.result()
            except Exception as exc:
                print(case[:2], type(exc).__name__, repr(exc), flush=True)
                failures.append((case[:2], exc))
                continue
            rows.append(row)
            print(row["channel"], row["smoothed"], "/", row["raw_contours"], flush=True)
            temporary = output.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: (r["template"], r["channel"])))
            temporary.replace(output)
    if failures:
        raise SystemExit(f"Failed comparisons: {failures}")


if __name__ == "__main__":
    main()
