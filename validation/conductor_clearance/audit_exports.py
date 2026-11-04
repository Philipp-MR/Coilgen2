"""Audit strict-review STL gaps for GUI jobs; --runs maps template__channel to job directories."""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path

import numpy as np

from surface_gap import Body, compare, stl_triangles
from skin_contact import inspect_skin


def load_bodies(key, directory):
    metadata = json.loads((directory / "result.json").read_bytes())
    config = metadata["config"]
    artifact_hashes = {row["name"]: row["sha256"] for row in metadata["artifacts"] if "sha256" in row}
    fabrication = directory / "fabrication"
    records = json.loads((fabrication / "fabrication_report.json").read_bytes())["paths"]
    bodies = []
    with np.load(directory / "scientific_result.npz", allow_pickle=False) as arrays:
        for i, record in enumerate(records):
            name = record["body_id"]
            files = list(fabrication.glob(name + "*fabrication.stl"))
            if len(files) != 1:
                raise ValueError(f"{key}/{name} needs exactly one released or draft fabrication STL")
            local_file = files[0].relative_to(directory).as_posix()
            digest = sha256(files[0].read_bytes()).hexdigest()
            if artifact_hashes.get(local_file) != digest:
                raise ValueError(f"{key}/{name} STL digest differs from saved run")
            triangles = stl_triangles(files[0])
            if len(triangles) != record["diagnostics"]["face_count"]:
                raise ValueError(f"{key}/{name} facet count differs from fabrication report")
            prefix = f"review_conductor_{i}"
            owners = arrays[prefix + "_owner_kind"]
            body = Body(
                key + "/" + name,
                triangles,
                arrays[prefix + "_frame_cross_section_m"],
                arrays[prefix + "_points_m"],
                owners,
                bool(arrays[prefix + "_frame_closed"]),
                config["sweep_insulation_thickness"],
                config["sweep_manufacturing_tolerance"],
                config["sweep_bend_clearance_margin"],
            )
            body.stl_source = {"path": local_file, "sha256": digest}
            bodies.append(body)
    return bodies


def inspect(key, directory):
    bodies = load_bodies(key, directory)
    meta = json.loads((directory / "result.json").read_bytes())
    return dict(
        key=key,
        job=directory.name,
        request_sha256=meta["request_sha256"],
        project_model_fingerprint=meta["project_model_fingerprint"],
        stl_sources={b.name: b.stl_source for b in bodies},
        self_and_between_bodies=[compare(a, b) for a, b in itertools.combinations_with_replacement(bodies, 2)],
        short_fold_screen=[compare(b, b, "short_fold") for b in bodies],
        all_station_skin=[inspect_skin(b) for b in bodies],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case")
    p.add_argument("--system")
    args = p.parse_args()
    runs = {k: Path(v) for k, v in json.loads(args.runs.read_bytes()).items()}
    args.output.mkdir(exist_ok=True, parents=True)
    for key, directory in runs.items():
        if args.system or args.case and key != args.case:
            continue
        out = args.output / (key + ".json")
        out.write_text(json.dumps(inspect(key, directory), indent=2), encoding="utf-8")
        print(key, flush=True)
    if args.case:
        return
    systems = {}
    for key, directory in runs.items():
        system = key.rsplit("__", 1)[0]
        if not args.system or system == args.system:
            systems.setdefault(system, []).append((key, directory))
    for system, channels in systems.items():
        if len(channels) < 2:
            continue
        bodies = {k: load_bodies(k, d) for k, d in channels}
        rows = []
        for a, b in itertools.combinations(bodies, 2):
            rows.extend(compare(ba, bb) for ba, bb in itertools.product(bodies[a], bodies[b]))
            print(a, b, flush=True)
        out = args.output / (system + "__interchannel.json")
        out.write_text(json.dumps(dict(system=system, pairs=rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
