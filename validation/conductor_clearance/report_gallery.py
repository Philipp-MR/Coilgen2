"""Package verified offscreen runs, exported meshes, numeric witnesses and field differences."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np


def read(p):
    return json.loads(p.read_bytes())


def motion(run):
    from coilgen.pipeline_stages import PipelineCheckpoint

    checkpoint = run / "checkpoints/routed.cgstage"
    if not checkpoint.is_file():
        checkpoint = Path(read(run / "request.json")["resume_checkpoint"])
    parts = PipelineCheckpoint.load(checkpoint).parts
    peak = lateral = 0.0
    methods = []
    for p in parts:
        d = p.normal_shift_provenance
        height = np.asarray(d.get("lift_distance_m", []))
        offsets = np.asarray(d.get("lateral_displacement_m", []))
        peak = max(peak, float(np.max(abs(height), initial=0)))
        if offsets.size:
            lateral = max(lateral, float(np.max(np.linalg.norm(offsets, axis=1))))
        methods.append(p.normal_shift_diagnostics.get("normal_shift_method", "initial connector shift"))
    return peak * 1000, lateral * 1000, " | ".join(methods)


def rows_for(before, after, cases, receipts):
    source = read(cases.parent / "snapshot.json")
    reuse_path = cases.parent / "checkpoint-reuse.json"
    reuse = read(reuse_path) if reuse_path.is_file() else {}
    baseline_file = Path(__file__).with_name("baseline-channels.csv")
    with baseline_file.open(encoding="utf-8-sig", newline="") as fp:
        baseline = {r["key"]: r for r in csv.DictReader(fp)}
    for key in sorted(after):
        directory, prior = Path(after[key]), Path(before[key])
        audit = read(cases / (key + ".json"))
        old, new = (read(p / "result.json") for p in (prior, directory))
        initial = baseline.get(key, {})
        if initial.get("source_request_sha256") != old["request_sha256"]:
            initial = {}
        manifest = read(directory / "manifest.json")
        assert manifest["source_sha256"] == source["source_digest"], key
        assert manifest["numerical_source_sha256"] == source["numerical_source_digest"], key
        assert receipts[key]["request_sha256"] == audit["request_sha256"] == new["request_sha256"]
        assert receipts[key]["verified_dataset"] == "review_conductor"
        same_replay = ""
        if key in reuse:
            replay = reuse[key]
            assert replay["physical_model"] == new["project_model_fingerprint"]
            assert replay["numerical_source"] == manifest["numerical_source_sha256"]
            earlier = read(Path(replay["prior_run"]) / "result.json")
            assert earlier["request_sha256"] == replay["prior_request"]
            previous_stls = {a["name"]: a["sha256"] for a in earlier["artifacts"] if a["name"].endswith("fabrication.stl")}
            new_stls = {r["path"]: r["sha256"] for r in audit["stl_sources"].values()}
            same_replay = previous_stls == new_stls
        fa, fb = old["fabrication"], new["fabrication"]
        envelopes = [*audit["self_and_between_bodies"], *audit["short_fold_screen"]]
        skin = audit["all_station_skin"]
        body_count = len(fa["paths"])
        assert body_count > 0 and body_count == len(fb["paths"]) == len(skin) == len(audit["stl_sources"]), key
        assert len(audit["short_fold_screen"]) == body_count, key
        assert len(audit["self_and_between_bodies"]) == body_count * (body_count + 1) // 2, key
        minimum = min(
            (r["witnesses"]["minimum"]["distance_m"] for r in envelopes if "minimum" in r.get("witnesses", {})),
            default=None,
        )
        old_rms = fa["field_metrics"]["dsv_relative_residual_percent"]
        new_rms = fb["field_metrics"]["dsv_relative_residual_percent"]
        old_amps = [x["current_A"] for x in fa["electrical_metrics"]["bodies"]]
        new_amps = [x["current_A"] for x in fb["electrical_metrics"]["bodies"]]
        lift, lateral, methods = motion(directory)
        with np.load(prior / "scientific_result.npz") as a, np.load(directory / "scientific_result.npz") as b:
            traces = {k for k in a if k.startswith("contour_") and k.endswith("_points_m")}
            same_traces = traces == {k for k in b if k.startswith("contour_") and k.endswith("_points_m")}
            same_traces &= all(np.array_equal(a[k], b[k]) for k in traces)
            same_target = all(
                np.array_equal(a[k], b[k]) for k in ("target_points_m", "target_field_T", "target_weights")
            )
            sheet_delta = float(np.max(abs(a["stream_function_A"] - b["stream_function_A"])))
        yield dict(
            example_channel=key,
            bodies=len(skin),
            reused_compatible_routed_checkpoint=key in reuse,
            replayed_STLs_byte_identical=same_replay,
            previous_copper_clear=initial.get("measured_clearance_pass", ""),
            previous_remote_minimum_mm=initial.get("remote_gap_mm", ""),
            copper_clear=not any(r.get("failed", False) for r in [*envelopes, *skin]),
            required_copper_gap_mm=1000 * envelopes[0]["required_m"],
            configured_lift_limit_mm=1000 * abs(new["config"]["normal_shift_length"]),
            largest_recorded_normal_lift_mm=lift,
            largest_recorded_lateral_adjustment_mm=lateral,
            largest_reported_fold_relief_bound_mm=max(
                1000 * p["diagnostics"].get("sweep_fold_relief_max_source_bound_m", 0) for p in fb["paths"]
            ),
            methods=methods,
            smallest_measured_copper_gap_mm=None if minimum is None else 1000 * minimum,
            nonincident_contacts=sum(r["all_nonincident_contact_pairs"] for r in skin),
            incident_piercings=sum(r["incident_strict_piercing_pairs"] for r in skin),
            overlapping_coplanar_faces=sum(r["coplanar_area_overlap_pairs"] for r in skin),
            positive_gap_violations=sum(
                r.get("facet_pair_counts", {}).get("positive_subminimum", 0) for r in envelopes
            ),
            all_meshes_closed_solids=all(p["diagnostics"]["mesh_solid_ok"] for p in fb["paths"]),
            target_unchanged=bool(same_target),
            extracted_turns_unchanged=bool(same_traces),
            max_sheet_change_A=sheet_delta,
            max_current_change_A=float(np.max(np.abs(np.asarray(old_amps) - new_amps))),
            previous_fixed_current_DSV_RMS_percent=old_rms,
            fixed_current_DSV_RMS_percent=new_rms,
            DSV_change_percentage_points=new_rms - old_rms,
            previous_length_m=sum(r["length_m"] for r in fa["electrical_metrics"]["bodies"]),
            length_m=sum(r["length_m"] for r in fb["electrical_metrics"]["bodies"]),
            strict_release=fb["ready"],
            release_reasons=" | ".join(fb["failure_reasons"]),
            request_sha256=audit["request_sha256"],
        )


def csv_out(p, rows):
    with p.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def gallery_report(out, source, rows, pairs):
    """Describe the exact artifact set without inferring release from a clearance pass."""
    systems = {}
    for row in rows:
        systems.setdefault(row["example_channel"].rsplit("__", 1)[0], []).append(row)
    expected = sum(a["bodies"] * b["bodies"] for channels in systems.values() for a, b in combinations(channels, 2))
    assert len(pairs) == expected, (len(pairs), expected)
    assert len({(p["first"], p["second"]) for p in pairs}) == expected
    assembly = {s: [p for p in pairs if p["first"].rsplit("__", 1)[0] == s] for s in systems}
    margins = [r["smallest_measured_copper_gap_mm"] for r in rows if r["smallest_measured_copper_gap_mm"] is not None]
    worst = max(rows, key=lambda r: abs(r["DSV_change_percentage_points"]))
    nclear = sum(r["copper_clear"] for r in rows)
    nrelease = sum(r["strict_release"] for r in rows)
    lines = [
        "# CoilGen GUI gallery: exported conductor clearance",
        "",
        f"{len(systems)} examples, {len(rows)} channels and {sum(r['bodies'] for r in rows)} exported copper bodies. "
        f"{nclear}/{len(rows)} channels pass the checks on self-clearance and separation between their bodies; "
        f"{sum(p['copper_clear'] for p in pairs)}/{expected} interchannel body pairs pass in GUI world coordinates.",
        f"The previously audited exports had {sum(r['previous_copper_clear'] == 'True' for r in rows)} "
        f"passing channel(s) among {sum(bool(r['previous_copper_clear']) for r in rows)} matched requests.",
        "",
        f"Smallest measured copper separation within a channel: {min(margins):.6f} mm. "
        f"Minimum across channels: {min(p['minimum_gap_mm'] for p in pairs):.6f} mm. "
        "Each example requires 0.250 mm: 0.100 mm insulation on each surface and 0.050 mm manufacturing tolerance. "
        "Distances are between the exported bare-copper facets, not their centerlines.",
        "",
        "| GUI example | Channels clear | Bodies | Smallest measured in-channel gap (mm) | Interchannel min (mm) | Strict release |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s, channels in sorted(systems.items()):
        cross = assembly[s]
        measured = [
            r["smallest_measured_copper_gap_mm"] for r in channels if r["smallest_measured_copper_gap_mm"] is not None
        ]
        inside = f"{min(measured):.6f}" if measured else "—"
        outside = f"{min(p['minimum_gap_mm'] for p in cross):.6f}" if cross else "—"
        lines.append(
            f"| {s} | {sum(r['copper_clear'] for r in channels)}/{len(channels)} | "
            f"{sum(r['bodies'] for r in channels)} | {inside} | {outside} | "
            f"{sum(r['strict_release'] for r in channels)}/{len(channels)} |"
        )
    lines += [
        "",
        f"Strict fabrication release: {nrelease}/{len(rows)} channels. Review each channel's separate bend, support, frame, and other "
        "release conditions in [per-channel.csv](per-channel.csv). Clearance of the exported windings does not cover leads "
        "or additional connections absent from the STL.",
        "",
        f"Largest absolute change in fixed-current DSV RMS residual versus the original run: "
        f"{worst['example_channel']}, {worst['previous_fixed_current_DSV_RMS_percent']:.6f}% → "
        f"{worst['fixed_current_DSV_RMS_percent']:.6f}%. "
        f"Extracted contour coordinates unchanged in {sum(r['extracted_turns_unchanged'] for r in rows)}/{len(rows)} channels. "
        f"Maximum change in drive current: {max(r['max_current_change_A'] for r in rows):.4g} A.",
        "",
        "Each run was submitted and reopened through the native Qt workbench using an offscreen framebuffer. "
        "The independent check re-read each STL, verified its saved SHA-256 digest and facet count, tested skin "
        "penetration/overlap at all stations, and checked positive clearance between separate visits and bodies. "
        "Short folded returns are threshold-tested: a passing test need not report their exact minimum. "
        "See the [verification procedure](../conductor_clearance/README.md) for the arclength and turn criteria.",
        f"{sum(r['reused_compatible_routed_checkpoint'] for r in rows)} unchanged channels reused a native routing checkpoint "
        f"from the preceding full-compute campaign, matched by physical model and numerical code. The GUI regenerated their "
        f"field evaluation and fabrication exports. {sum(r['replayed_STLs_byte_identical'] is True for r in rows)} "
        "of those channels reproduced the preceding STL files byte for byte. The remaining channels ran from their gallery inputs.",
        "",
        f"Source SHA-256: `{source['source_digest']}`. Numerical source SHA-256: `{source['numerical_source_digest']}`.",
        "",
        "[Copper STL exports](copper-exports.zip) · [Reproduction files and numerical witnesses](verification-evidence.zip) · "
        "[Distances between channels](interchannel-pairs.csv). Extract both archives together for the independent recheck.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "runs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    before, after = read(args.previous), read(args.runs)
    assert set(before) == set(after)
    work = args.runs.parent
    cases = work / "cases"
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    rows = list(rows_for(before, after, cases, read(work / "native_viewport.json")))
    from stl_solid import inspect as inspect_solid
    from surface_gap import stl_triangles

    solids = []
    for row in rows:
        key = row["example_channel"]
        sources = read(cases / (key + ".json"))["stl_sources"]
        bodies = [
            dict(body=body, sha256=entry["sha256"], **inspect_solid(stl_triangles(Path(after[key]) / entry["path"])))
            for body, entry in sources.items()
        ]
        row["all_exported_STLs_closed_oriented"] = all(b["closed_oriented_solid"] for b in bodies)
        row["copper_clear"] &= row["all_exported_STLs_closed_oriented"]
        solids.extend(bodies)
    csv_out(out / "stl-solids.csv", solids)
    csv_out(out / "per-channel.csv", rows)
    pairs = []
    assembly_files = sorted([*cases.glob("*__interchannel.json"), *(work / "assemblies").glob("*__interchannel.json")])
    for file in assembly_files:
        for entry in read(file)["pairs"]:
            pairs.append(
                dict(
                    first=entry["first"],
                    second=entry["second"],
                    copper_clear=not entry["failed"],
                    minimum_gap_mm=1000 * entry["witnesses"]["minimum"]["distance_m"],
                    required_gap_mm=1000 * entry["required_m"],
                )
            )
    csv_out(out / "interchannel-pairs.csv", pairs)
    gallery_report(out, read(work / "snapshot.json"), rows, pairs)
    witness = work / "gz_opposing_cut_clearance.png"
    if witness.exists() and witness.with_suffix(".json").exists():
        for path in (witness, witness.with_suffix(".json")):
            copyfile(path, out / path.name)
        with (out/"README.md").open("a", encoding="utf-8") as fp:
            fp.write("\n[Measured Gz crossing: copper surfaces and cross section](gz_opposing_cut_clearance.png) · "
                     "[Facet IDs and witness coordinates](gz_opposing_cut_clearance.json).\n")
    with ZipFile(out / "copper-exports.zip", "w", ZIP_DEFLATED, compresslevel=5) as z:
        for key in sorted(after):
            for stl in sorted((Path(after[key]) / "fabrication").glob("*fabrication.stl")):
                z.write(stl, "jobs/" + key + "/fabrication/" + stl.name)
    with ZipFile(out / "verification-evidence.zip", "w", ZIP_DEFLATED) as z:
        frozen = Path(read(work / "snapshot.json")["source"])
        for path in sorted(frozen.rglob("*.py")):
            z.write(path, "verified-source/coilgen/" + path.relative_to(frozen).as_posix())
        for path in sorted(cases.glob("*.json")):
            z.write(path, "geometry/" + path.name)
        for path in assembly_files:
            if path.parent != cases:
                z.write(path, "geometry/" + path.name)
        for name in ("snapshot.json", "native_viewport.json", "new_runs.json"):
            z.write(work / name, name)
        if (work / "checkpoint-reuse.json").exists():
            z.write(work / "checkpoint-reuse.json", "checkpoint-reuse.json")
        if (work / "checkout_match.json").exists():
            z.write(work / "checkout_match.json", "checkout_match.json")
        z.write(out / "stl-solids.csv", "geometry/stl-solids.csv")
        for path in sorted(Path(__file__).parent.glob("*.py")):
            z.write(path, "independent-check/" + path.name)
        for name in ("offscreen_gui.py", "verify_viewport.py"):
            z.write(Path(__file__).parent / name, "gui-replay/" + name)
        z.writestr("portable-runs.json", json.dumps({k: "jobs/" + k for k in sorted(after)}, indent=2))
        z.writestr(
            "RECHECK.txt",
            "Extract this archive and copper-exports.zip to the same folder. From that folder run:\n"
            "python independent-check/audit_exports.py --runs portable-runs.json --output recheck\n"
            "The checker verifies each original STL SHA-256 before measuring copper-to-copper distances.\n",
        )
        for key in sorted(after):
            run = Path(after[key])
            for name in (
                "result.json",
                "request.json",
                "manifest.json",
                "scientific_result.npz",
                "fabrication/fabrication_report.json",
            ):
                z.write(run / name, "jobs/" + key + "/" + name)
            from coilgen.application.session_evidence import revision_key, revision_path

            original = revision_path(run.parents[1], revision_key(read(run / "result.json")))
            z.write(original, "submitted-projects/" + key + original.suffix)
    print(
        "Channels",
        len(rows),
        "bodies",
        sum(r["bodies"] for r in rows),
        "clear channels",
        sum(r["copper_clear"] for r in rows),
    )
    print("Channel pairs", len(pairs), "clear", sum(r["copper_clear"] for r in pairs))


if __name__ == "__main__":
    main()
