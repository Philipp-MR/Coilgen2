"""Evaluate each selected GUI example through fabrication review; record the physical winding.

Use --scratch PATH for a local (preferably unsynced) directory. Individual
families can run separately. The project fingerprint guards against accidental
reuse of a different template. Use a fresh scratch directory after engine changes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import re

import numpy as np
from threadpoolctl import threadpool_limits

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.templates import TEMPLATE_SPECS, template_assembly
from coilgen.persistence import atomic_json, file_digest


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "gui_review_2026-11-03"
ROOT = HERE.parents[1]
GEOMETRY = re.compile(r"support_\d+_vertices_m|review_conductor_\d+_points_m")


def turn_angles(points, closed):
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if closed and np.linalg.norm(points[-1] - points[0]) > 1e-12:
        arc = np.r_[arc, arc[-1] + np.linalg.norm(points[-1] - points[0])]
        points = np.vstack((points, points[0]))
    s = np.linspace(0, arc[-1], max(4, round(arc[-1] / 0.003)) + 1)
    sampled = np.column_stack([np.interp(s, arc, component) for component in points.T])
    tangents = np.diff(sampled, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    dot = np.einsum("ij,ij->i", tangents, np.roll(tangents, 1, axis=0))
    return np.degrees(np.arccos(np.clip(dot if closed else dot[1:], -1, 1)))


def evaluate(case):
    template, idx, scratch, existing, collect_only = case
    key = f"{template}__{idx}"
    proposed = template_assembly(template).channels[idx].project
    directory = Path(scratch) / key / proposed.model_fingerprint[:16]
    source = directory / "source.coilgen"
    if existing:
        directory = Path(existing)
        source = directory / "source.coilgen"
    else:
        if not source.exists() and not collect_only:
            directory.mkdir(parents=True, exist_ok=True)
            proposed.save(source)
    project = ProjectDocument.load(source)
    if (project.name, project.model_fingerprint, project.fingerprint) != (
        proposed.name,
        proposed.model_fingerprint,
        proposed.fingerprint,
    ):
        raise ValueError(f"Mismatched gallery model at {directory}")
    run = directory / "run"
    if not (run / "result.json").is_file():
        if existing or collect_only:
            raise ValueError(f"Expected completed job at {run}")
        with threadpool_limits(1):
            DesignService.run(project, run, threads=1)
    meta = DesignService.load_run(run, array_names=()).metadata
    if (meta["project_id"], meta["project_revision"], meta["project_fingerprint"]) != (
        project.id,
        project.revision,
        project.fingerprint,
    ):
        raise ValueError(f"Completed run belongs to a different source: {key}")
    with np.load(run / "scientific_result.npz", allow_pickle=False) as a:
        retained = {k: a[k] for k in a.files if GEOMETRY.fullmatch(k)}
    reports = HERE / "reviewed_centerlines"
    reports.mkdir(exist_ok=True)
    target = reports / f"{key}.npz"
    tmp = target.with_suffix(".npz.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **retained)
    tmp.replace(target)
    fab = meta["fabrication"]
    bodies = fab["paths"]
    curves = [retained[f"review_conductor_{i}_points_m"] for i in range(len(bodies))]
    angles = np.concatenate([turn_angles(p, body["closed"]) for p, body in zip(curves, bodies, strict=True)])
    # Old archives have no closure metadata. Read it from the matching verified historical report.
    from zipfile import ZipFile

    with ZipFile(PREVIOUS / "submitted-reports.zip") as z:
        previous = json.loads(z.read(f"{key}/result.json"))
    old_bodies = previous["fabrication"]["paths"]
    # Index arrays numerically (lexical order can differ for >9 bodies).
    with np.load(PREVIOUS / "centerlines" / f"{key}.npz", allow_pickle=False) as a:
        assert sum(bool(re.fullmatch(r"review_conductor_\d+_points_m", k)) for k in a.files) == len(old_bodies)
        old_angles = np.concatenate(
            [turn_angles(a[f"review_conductor_{i}_points_m"], body["closed"]) for i, body in enumerate(old_bodies)]
        )
    fits = meta["contour_smoothing"]
    row = dict(
        template=template,
        channel_index=idx,
        channel=project.name,
        model_fingerprint=project.model_fingerprint,
        project_fingerprint=project.fingerprint,
        source_hash=json.loads((run / "manifest.json").read_text("utf8"))["numerical_source_sha256"],
        request_sha256=meta["request_sha256"],
        geometry_sha256=file_digest(target),
        smoothed=sum(s["smoothed_count"] for s in fits),
        local=sum(s.get("local_fit_count", 0) for s in fits),
        original_retained=sum(s["fit_rejected_count"] + s["spacing_rejected_count"] for s in fits),
        old_review_p95_deg=float(np.percentile(old_angles, 95)),
        review_p95_deg=float(np.percentile(angles, 95)),
        previous_review_rms_percent=previous["fabrication"]["field_metrics"]["dsv_relative_residual_percent"],
        review_rms_percent=fab["field_metrics"]["dsv_relative_residual_percent"],
        current_A=meta["wire_fields"]["contour"]["at_stream_current"]["current_A"],
        previous_current_A=previous["wire_fields"]["contour"]["at_stream_current"]["current_A"],
        sheet_accepted=meta["summary"]["sheet_accepted"],
        pipeline_sweep=meta["pipeline_sweeps"]["checks_passed"],
        fabrication=fab["ready"],
        construction_release=meta["summary"]["construction_released"],
        checked_bodies=len(bodies),
        previous_bodies=len(old_bodies),
        min_fabrication_bend_mm=min(body["diagnostics"]["sweep_final_min_bend_radius_m"] for body in bodies) * 1000,
    )
    atomic_json(reports / f"{key}.json", row)
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=ROOT / "tmp/winding_quality_full")
    parser.add_argument("--case", action="append", help="template ID; includes all channels in that template")
    parser.add_argument("--channel", type=int, action="append", help="channel index in one selected template")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--collect-only", action="store_true", help="Require verified completed runs; do not compute.")
    parser.add_argument("--existing", action="append", default=[], help="key=source/run parent to reuse")
    args = parser.parse_args()
    if args.channel and (not args.case or len(args.case) != 1):
        parser.error("--channel requires exactly one --case")
    cases = []
    external = dict(spec.split("=", 1) for spec in args.existing)
    for spec in TEMPLATE_SPECS:
        if args.case and spec.id not in args.case:
            continue
        for idx in range(len(template_assembly(spec.id).channels)):
            if args.channel and idx not in args.channel:
                continue
            cases.append((spec.id, idx, args.scratch, external.get(f"{spec.id}__{idx}"), args.collect_only))
    if (
        not cases
        or (args.case and set(args.case) != {row[0] for row in cases})
        or (args.channel and set(args.channel) != {row[1] for row in cases})
    ):
        parser.error("Unknown template selection.")
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, case): case[:2] for case in cases}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                errors.append((futures[future], repr(exc)))
                print(errors[-1], flush=True)
    if errors:
        raise RuntimeError(errors)
    all_rows = [json.loads(p.read_text("utf8")) for p in sorted((HERE / "reviewed_centerlines").glob("*.json"))]
    expected = {
        (spec.id, i): (ch.project.model_fingerprint, ch.project.fingerprint)
        for spec in TEMPLATE_SPECS
        for i, ch in enumerate(template_assembly(spec.id).channels)
    }
    if {
        (r["template"], r["channel_index"]): (r["model_fingerprint"], r.get("project_fingerprint")) for r in all_rows
    } == expected:
        with (HERE / "full-gallery.csv").open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
