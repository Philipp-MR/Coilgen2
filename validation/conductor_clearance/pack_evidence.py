"""Summarize one completed 32-channel export audit and package its per-facet witnesses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def mm(x):
    return 1000 * float(x)


def pack(input_dir, output_dir, tag):
    files = sorted(input_dir.glob("*.json"))
    channels = []
    assemblies = []
    for path in files:
        data = json.loads(path.read_bytes())
        if "pairs" in data:
            pairs = data["pairs"]
            assemblies.append(
                dict(
                    system=data["system"],
                    body_pairs=len(pairs),
                    failing_pairs=sum(r["failed"] for r in pairs),
                    minimum_surface_gap_mm=mm(min(r["witnesses"]["minimum"]["distance_m"] for r in pairs)),
                )
            )
            continue
        remote = data["self_and_between_bodies"]
        short = data["short_fold_screen"]
        skin = data["all_station_skin"]
        closest = min(
            (r["witnesses"]["minimum"] for r in remote if r.get("witnesses")),
            key=lambda w: w["distance_m"],
            default=None,
        )
        row = dict(
            key=data["key"],
            job=data["job"],
            bodies=len(skin),
            required_surface_gap_mm=mm(remote[0]["required_m"]),
            measured_clearance_pass=not any(r.get("failed", False) for r in remote + short + skin),
            remote_gap_mm=mm(closest["distance_m"]) if closest else "",
            zero_remote_facet_pairs=sum(r.get("facet_pair_counts", {}).get("contact", 0) for r in remote),
            positive_subminimum_remote_facet_pairs=sum(
                r.get("facet_pair_counts", {}).get("positive_subminimum", 0) for r in remote
            ),
            positive_subminimum_short_fold_facet_pairs=sum(
                r["facet_pair_counts"]["positive_subminimum"] for r in short
            ),
            all_station_nonincident_contact_facet_pairs=sum(r["all_nonincident_contact_pairs"] for r in skin),
            all_station_piercing_facet_pairs=sum(r["strict_piercing_pairs"] for r in skin),
            incident_piercing_facet_pairs=sum(r["incident_strict_piercing_pairs"] for r in skin),
            incident_excess_line_contact_pairs=sum(r["incident_excess_line_contact_pairs"] for r in skin),
            all_station_coplanar_area_overlap_pairs=sum(r["coplanar_area_overlap_pairs"] for r in skin),
            source_request_sha256=data["request_sha256"],
        )
        channels.append(row)
    if not channels:
        raise ValueError("No audited channel cases")
    output_dir.mkdir(exist_ok=True, parents=True)
    for name, rows in (("channels", channels), ("assemblies", assemblies)):
        if not rows:
            continue
        with (output_dir / f"{tag}-{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with ZipFile(output_dir / f"{tag}-witnesses.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    return len(channels), len(assemblies)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    print(pack(args.input, args.output, args.tag))
