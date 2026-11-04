"""Supplementary fixed-current Bz checks on the independent grid from each saved study.

python check_wire_holdout.py --output-prefix PATH run_label=RUN_DIRECTORY [...]
Use runs with the same channel name when comparing alternatives. Within
each such group the design samples, independent grid and targets must match.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from coilgen.application.service import DesignService
from coilgen.physics import biot_savart_wire_path


def relative_rms(bz, target):
    return float(100 * np.linalg.norm(bz - target) / np.linalg.norm(target))


def evaluate(run, label, arrays, grids):
    meta = DesignService.load_run(run, array_names=()).metadata
    request = json.loads((run / "request.json").read_bytes())
    if request["config"]["target_field_component"] != "Bz":
        raise ValueError("This comparison requires a Bz target.")
    fab = meta["fabrication"]
    assert fab["field_metrics"]["current_source"] == "original_contour_step_no_refit"
    by_id = {e["body_id"]: e for e in fab["electrical_metrics"]["bodies"]}
    with np.load(run / "scientific_result.npz", allow_pickle=False) as a:
        points, target = a["fabricated_points_m"], a["target_field_T"]
        assert np.array_equal(a["target_points_m"], points) and not a["target_suppressed"].any()
        saved_vector = a["fabricated_field_T"]
        held_points, held_target = a["validation_points_m"], a["validation_target_T"]
        circuits = []
        for i, body in enumerate(fab["paths"]):
            info = by_id[body["body_id"]]
            assert info["circuit_complete"] and np.isfinite(info["current_A"]) and info["current_A"] != 0
            circuits.append((a[f"review_electrical_{i}_points_m"], info["current_A"]))

    def field(xyz):
        b = np.zeros_like(xyz, dtype=float)
        for path, current in circuits:
            b += current * biot_savart_wire_path(path, xyz, worker_count=1)
        return b

    reconstructed = field(points)
    vector_error = float(np.max(np.abs(reconstructed - saved_vector)))
    np.testing.assert_allclose(reconstructed, saved_vector, rtol=2e-11, atol=2e-18)
    rms = relative_rms(reconstructed[:, 2], target[:, 2])
    np.testing.assert_allclose(rms, fab["field_metrics"]["dsv_relative_residual_percent"], rtol=2e-11, atol=1e-10)
    # The independent sheet grid can share a few positions with the sparse
    # design grid. Use only positions outside 0.1 nm of every design sample.
    separation = cKDTree(points).query(held_points)[0]
    keep = separation > 1e-10
    excluded = int(np.count_nonzero(~keep))
    held_points, held_target = held_points[keep], held_target[keep]
    assert len(held_points) > 0
    grid = (points, target, held_points, held_target)
    name = meta["name"]
    if name in grids:
        for first, repeated in zip(grids[name], grid, strict=True):
            np.testing.assert_array_equal(first, repeated)
    else:
        grids[name] = grid
    min_separation = separation[keep].min()
    measured = field(held_points)
    residual = measured[:, 2] - held_target[:, 2]
    group = "grid_" + sha256(name.encode("utf-8")).hexdigest()[:12]
    arrays[f"{group}_points_m"], arrays[f"{group}_target_T"] = held_points, held_target
    arrays[f"{label}_Bxyz_T"] = measured
    return {
        "variant": label,
        "channel": name,
        "request_sha256": meta["request_sha256"],
        "grid_key": group,
        "currents_A": json.dumps([i for _, i in circuits]),
        "design_points": len(points),
        "holdout_points": len(held_points),
        "excluded_shared_grid_positions": excluded,
        "min_grid_separation_mm": min_separation * 1000,
        "max_design_vector_reconstruction_delta_T": vector_error,
        "design_Bz_relative_RMS_percent": rms,
        "holdout_Bz_relative_RMS_percent": relative_rms(measured[:, 2], held_target[:, 2]),
        "holdout_Bz_mean_residual_uT": 1e6 * float(residual.mean()),
        "holdout_Bz_RMS_residual_uT": 1e6 * float(np.sqrt(np.mean(residual**2))),
        "holdout_Bz_max_residual_uT": 1e6 * float(np.abs(residual).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("runs", nargs="+", help="Unique label=verified full run directory")
    args = parser.parse_args()
    rows, arrays, grids = [], {}, {}
    labels = set()
    with threadpool_limits(1):
        for item in args.runs:
            label, address = item.split("=", 1)
            if label in labels or not re.fullmatch(r"[a-zA-Z0-9_]+", label):
                parser.error("Supply unique alphanumeric/underscore labels.")
            labels.add(label)
            row = evaluate(Path(address), label, arrays, grids)
            rows.append(row)
            print(label, row["holdout_Bz_relative_RMS_percent"], flush=True)
    with args.output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
