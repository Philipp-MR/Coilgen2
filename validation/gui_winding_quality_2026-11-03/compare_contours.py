"""Compare the GUI gallery's physical contour polylines at the original sheet current.

Run from the checkout with PYTHONPATH=src. Raw contour checkpoints in the
scratch directory must match the model and numerical source used in the pair.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from coilgen.application.service import DesignService
from coilgen.application.project import ProjectDocument
from coilgen.application.templates import TEMPLATE_SPECS, template_assembly
from coilgen.contours import regularize_contour_paths
from coilgen.evaluation import calculate_unconnected_contour_field
from coilgen.persistence import numerical_source_digest
from coilgen.pipeline_stages import PipelineCheckpoint
from coilgen.research.application import _wire_field_metrics


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "gui_review_2026-11-03"
ROOT = HERE.parents[1]
SCRATCH = ROOT / "tmp/winding_quality_probes/gallery"


@cache
def numerical_source():
    import coilgen

    return numerical_source_digest(Path(coilgen.__file__).parent)


def quality(curves):
    """P95 direction change at equally spaced 3 mm and 1 mm stations."""
    result = []
    for step in (0.003, 0.001):
        angles = []
        for path in curves:
            length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            s = np.linspace(0, length[-1], max(4, round(length[-1] / step)) + 1)
            sampled = np.column_stack([np.interp(s, length, coord) for coord in path.T])
            tangent = np.diff(sampled, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            dots = np.einsum("ij,ij->i", tangent, np.roll(tangent, 1, axis=0))
            angles.extend(np.rad2deg(np.arccos(dots.clip(-1, 1))))
        result.append(float(np.percentile(angles, 95)))
    return result


def with_resolution(project, n, m):
    """Change only the mesh divisions of an analytic support, preserving its physical span."""
    config = project.config
    if config.coil_mesh_file == "create cylinder mesh":
        h, radius, _, nz, *axis = config.cylinder_mesh_parameter_list
        changes = {"cylinder_mesh_parameter_list": (h * nz / (nz + 1) * (m + 1) / m, radius, n, m, *axis)}
    elif config.coil_mesh_file == "create bi-planary mesh":
        w, h, _, _, *pose = config.biplanar_mesh_parameter_list
        changes = {"biplanar_mesh_parameter_list": (w, h, n, m, *pose)}
    elif config.coil_mesh_file == "create conical frustum mesh":
        h, r1, r2, _, _, *pose = config.conical_frustum_mesh_parameter_list
        changes = {"conical_frustum_mesh_parameter_list": (h, r1, r2, n, m, *pose)}
    else:
        raise ValueError("Explicit mesh trials require a generated cylinder, biplanar support or cone.")
    return project.update_config(changes)


def evaluate(case):
    key, index, old, mesh, scratch, reuse, source = case
    project = template_assembly(key).channels[index].project
    if source is not None:
        reference = ProjectDocument.load(source)
        if reference.name != project.name:
            raise ValueError("The alternate source must belong to the specified gallery channel.")
        project = reference
    if mesh:
        project = with_resolution(project, *mesh)
    config = project.config
    raw_model = project.update_config(
        {"contour_path_regularization_mode": "off", "contour_smoothing_local_fallback": False}
    )
    name = f"{key}__{index}"
    directory = scratch / name / raw_model.model_fingerprint[:16]
    source_hash = numerical_source()
    if reuse:
        base = reuse / name / raw_model.model_fingerprint[:16]
        for prior in (base, base / source_hash[:16]):
            path = prior / "measurement.json"
            manifest = prior / "raw/manifest.json"
            if not (path.is_file() and manifest.is_file()):
                continue
            saved = json.loads(path.read_bytes())
            if (
                saved.get("source_hash") == source_hash
                and saved.get("model_hash") == project.model_fingerprint
                and "local_fits" in saved
                and json.loads(manifest.read_bytes())["numerical_source_sha256"] == source_hash
            ):
                record = DesignService.load_run(prior / "raw", array_names=())
                assert record.metadata["project_model_fingerprint"] == raw_model.model_fingerprint
                return saved
    manifest = directory / "raw/manifest.json"
    if manifest.is_file() and json.loads(manifest.read_bytes())["numerical_source_sha256"] != source_hash:
        directory /= source_hash[:16]
    measurement = directory / "measurement.json"
    if measurement.is_file():
        saved = json.loads(measurement.read_text(encoding="utf-8"))
        if (
            saved.get("source_hash") == source_hash
            and saved.get("model_hash") == project.model_fingerprint
            and "local_fits" in saved
        ):
            return saved
    output = directory / "raw"
    if not (output / "result.json").is_file():
        DesignService.run(raw_model, output, threads=1, pipeline_stage="contours", fabricate=False)
    result = DesignService.load_run(output, array_names=()).metadata
    assert result["project_model_fingerprint"] == raw_model.model_fingerprint
    assert json.loads((output / "manifest.json").read_bytes())["numerical_source_sha256"] == source_hash
    cp = PipelineCheckpoint.load(output / "checkpoints/contours.cgstage")
    raw = [c.v.copy() for p in cp.parts for c in p.contour_lines]
    identity = [(c.current_orientation, c.potential, c.turn_multiplicity) for p in cp.parts for c in p.contour_lines]
    current = cp.parts[0].contour_step
    for part in cp.parts:
        assert part.contour_step == current

    def rms(parts):
        b = current * calculate_unconnected_contour_field(parts, config, cp.state.target_field, worker_count=1)
        return 100 * _wire_field_metrics(b, cp.state.target_field, config)["relative_rms_error"]

    raw_error = rms(cp.parts)
    smooth = regularize_contour_paths(cp.parts, config)
    curves = [c.v for p in smooth for c in p.contour_lines]
    assert identity == [
        (c.current_orientation, c.potential, c.turn_multiplicity) for p in smooth for c in p.contour_lines
    ]
    assert all(np.linalg.norm(curve[0] - curve[-1]) < 1e-11 for curve in curves)
    diagnostics = [part.contour_level_diagnostics for part in smooth]
    dmax = max(d["contour_path_smoothing_max_applied_displacement_m"] for d in diagnostics)
    assert dmax <= config.contour_smoothing_max_displacement * (1 + 1e-6)
    for part in smooth:
        for contour in part.contour_lines:
            end = np.einsum(
                "nkj,nji->nki", contour.segment_barycentric, part.mesh.vertices[part.mesh.faces[contour.segment_faces]]
            )
            np.testing.assert_allclose(end, np.stack((contour.v[:-1], contour.v[1:]), axis=1), rtol=0, atol=1e-10)
    for d in diagnostics:
        old_spacing = d["contour_path_smoothing_min_distance_before_m"]
        if np.isfinite(old_spacing):
            required = min(old_spacing, config.min_turn_distance) if config.min_turn_distance > 0 else old_spacing
            assert d["contour_path_smoothing_min_distance_after_m"] + max(1e-9, 1e-4 * required) >= required
    orig_angle, orig_1mm = quality(raw)
    angle, angle_1mm = quality(curves)
    row = dict(
        template=key,
        channel=project.name,
        mesh_vertices=result["summary"]["vertices"],
        case=name,
        source_hash=source_hash,
        model_hash=project.model_fingerprint,
        local_fallback=config.contour_smoothing_local_fallback,
        turns=len(curves),
        previous_turns=int(old["raw_contours"]),
        accepted=sum(d["contour_path_smoothing_contour_count"] for d in diagnostics),
        local_fits=sum(d.get("contour_path_smoothing_local_fit_count", 0) for d in diagnostics),
        failed=sum(d["contour_path_smoothing_failed_count"] for d in diagnostics),
        retained_for_spacing=sum(d["contour_path_smoothing_spacing_rejected_count"] for d in diagnostics),
        maximum_move_mm=1000 * dmax,
        old_p95_deg=float(old["smoothed_p95_turn_angle_3mm_deg"]),
        raw_p95_deg=orig_angle,
        smooth_p95_deg=angle,
        raw_p95_1mm_deg=orig_1mm,
        smooth_p95_1mm_deg=angle_1mm,
        previous_current_A=float(old["smoothed_fixed_current_A"]),
        current_A=current,
        old_contour_rms_percent=float(old["smoothed_contour_RMS_percent"]),
        raw_contour_rms_percent=raw_error,
        smooth_contour_rms_percent=rms(smooth),
        sheet_holdout_rms_percent=100 * result["summary"]["validated_relative_field_error"],
        sheet_passed=result["summary"]["sheet_accepted"],
    )
    measurement.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def worker(case):
    with threadpool_limits(1):
        return evaluate(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scratch", type=Path, default=SCRATCH, help="Local working directory for raw checkpoints")
    parser.add_argument(
        "--reuse", type=Path, help="Reuse verified completed comparisons from a second scratch directory"
    )
    parser.add_argument("--project", type=Path, help="Measure one specific saved source project for the named case")
    parser.add_argument("--mesh", nargs=2, type=int, help="Mesh divisions for an explicit alternative on one support")
    parser.add_argument("--case", action="append", help="Template__channel-index identifier")
    args = parser.parse_args()
    with (OLD / "smoothing.csv").open(encoding="utf-8", newline="") as f:
        old = {r["channel"]: r for r in csv.DictReader(f)}
    cases = [
        (
            spec.id,
            i,
            old[c.project.name],
            args.mesh,
            args.scratch.resolve(),
            args.reuse.resolve() if args.reuse else None,
            args.project,
        )
        for spec in TEMPLATE_SPECS
        for i, c in enumerate(template_assembly(spec.id).channels)
    ]
    if args.case:
        cases = [case for case in cases if f"{case[0]}__{case[1]}" in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("Unrecognized case")
    if args.mesh:
        if len(cases) != 1 or min(args.mesh) < 4:
            parser.error("--mesh requires one named case and at least four divisions on both axes.")
    if args.project and (args.mesh or len(cases) != 1):
        parser.error("--project must be used with one named case and cannot override its mesh.")
    sys.stdout.reconfigure(errors="backslashreplace")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(worker, case): case for case in cases}
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(row["channel"], row["accepted"], row["turns"], round(row["smooth_p95_deg"], 2), flush=True)
    dest = args.scratch / ("subset-" + args.case[0] + ".csv") if args.case else HERE / "gallery.csv"
    if args.mesh:
        dest = args.scratch / f"{args.case[0]}-{args.mesh[0]}x{args.mesh[1]}.csv"
    with dest.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["template"], row["channel"])))


if __name__ == "__main__":
    main()
