"""Screen a mixed set of GUI channel exports against the original assembly in world coordinates."""

import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from audit_exports import load_bodies
from surface_gap import ACCEPTANCE_EPS, compare


def audit_context(channels, changed=None, exact=False):
    """A separated bounding box certifies a lower bound; ambiguous pairs receive a complete STL test."""
    rows = []
    for (ca, aa), (cb, bb) in combinations(channels.items(), 2):
        if changed and not {ca, cb}.intersection(changed):
            continue
        for a, b in product(aa, bb):
            la, ha = a.levels[-1][0][0], a.levels[-1][1][0]
            lb, hb = b.levels[-1][0][0], b.levels[-1][1][0]
            gap = np.maximum(0.0, np.maximum(la - hb, lb - ha))
            bound = float(np.linalg.norm(gap))
            required = a.insulation + b.insulation + max(a.tolerance, b.tolerance) + max(a.margin, b.margin)
            row = {
                "first": a.name,
                "second": b.name,
                "required_m": required,
                "axis_interval_gap_xyz_m": gap.tolist(),
                "surface_gap_lower_bound_m": bound,
            }
            if not exact and bound >= required and bound > ACCEPTANCE_EPS:
                row.update(method="disjoint_world_boxes", failed=False)
            else:
                measured = compare(a, b)
                row.update(
                    method="finite_stl_facets",
                    failed=measured["failed"],
                    contained=measured["contained"],
                    witnesses=measured["witnesses"],
                    facet_pair_counts=measured["facet_pair_counts"],
                )
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--system", required=True)
    p.add_argument(
        "--replacement-runs",
        type=Path,
        help="Override one or more complete GUI channel jobs, preserving all other exports.",
    )
    p.add_argument(
        "--changed-only", action="store_true", help="Inspect only interfaces incident to an overridden channel."
    )
    p.add_argument(
        "--exact",
        action="store_true",
        help="Measure every face-to-face minimum, including pairs already certified by a bounding box.",
    )
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    base = json.loads(args.runs.read_bytes())
    jobs = {k: Path(v) for k, v in base.items() if k.rsplit("__", 1)[0] == args.system}
    repl = json.loads(args.replacement_runs.read_bytes()) if args.replacement_runs else {}
    if not set(repl) <= jobs.keys() or args.changed_only and not repl or len(jobs) < 2:
        raise ValueError("Supply at least two system channels and replacement keys from the same run map.")
    jobs.update({k: Path(v) for k, v in repl.items()})
    channels = {k: load_bodies(k, d) for k, d in jobs.items()}
    rows = audit_context(channels, set(repl) if args.changed_only else None, args.exact)
    data = {
        "system": args.system,
        "jobs": {k: d.name for k, d in jobs.items()},
        "replaced": list(repl),
        "changed_only": args.changed_only,
        "strict_stl": {b.name: b.stl_source for bodies in channels.values() for b in bodies},
        "failing_pairs": sum(r["failed"] for r in rows),
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"{len(rows)} interfaces, {data['failing_pairs']} below clearance")


if __name__ == "__main__":
    main()
