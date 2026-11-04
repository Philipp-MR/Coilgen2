"""Probe gentler smoothing on the saved, extracted Conical Z2 turns."""

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace

import numpy as np

from check_smoothing import BASELINE, HERE, turn_angles_deg
from coilgen.contours import regularize_contour_paths
from coilgen.evaluation import calculate_unconnected_contour_field
from coilgen.pipeline_stages import PipelineCheckpoint
from coilgen.research.application import _wire_field_metrics


def trial(arguments):
    path, gain = arguments
    raw = PipelineCheckpoint.load(path)
    state = raw.state
    config = replace(
        raw.config,
        contour_path_regularization_mode="parameterized_bspline" if gain else "off",
        contour_smoothing_scale_mode="mesh_coupled",
        contour_smoothing_mesh_gain=gain or 0.1,
        contour_smoothing_max_displacement=0.0005,
        contour_smoothing_preserve_min_turn_distance=True,
    )
    parts = regularize_contour_paths(state.parts, config)
    loops = [c.v for part in parts for c in part.contour_lines]
    with np.load(HERE / "centerlines" / "conical_harmonic__0.npz", allow_pickle=False) as reference:
        if gain in (0, 0.1):
            prefix = "" if gain else "original_"
            for index, loop in enumerate(loops):
                np.testing.assert_array_equal(reference[f"{prefix}contour_0_{index}_points_m"], loop)
    assert len(loops) == 37
    current = parts[0].contour_step
    field = current * calculate_unconnected_contour_field(parts, config, state.target_field, worker_count=1)
    field_rms = _wire_field_metrics(field, state.target_field, config)["relative_rms_error"]
    diagnostics = parts[0].contour_level_diagnostics
    return {
        "gain": gain or "off",
        "accepted": diagnostics.get("contour_path_smoothing_contour_count", 0),
        "spacing_rejected": diagnostics.get("contour_path_smoothing_spacing_rejected_count", 0),
        "max_move_mm": 1000 * diagnostics.get("contour_path_smoothing_max_applied_displacement_m", 0),
        "current_A": current,
        "p95_turn_angle_3mm_deg": np.percentile(turn_angles_deg(loops), 95),
        "original_current_contour_RMS_percent": 100 * field_rms,
    }


if __name__ == "__main__":
    from pathlib import Path

    with BASELINE.open(encoding="utf-8-sig", newline="") as stream:
        archived = next(row for row in csv.DictReader(stream) if row["project_or_channel"] == "Conical Z2")
    path = Path(archived["job_directory"]) / "checkpoints" / "contours.cgstage"
    with ProcessPoolExecutor(max_workers=3) as workers:
        rows = list(workers.map(trial, [(path, gain) for gain in (0, 0.1, 0.3, 0.6)]))
    with (HERE / "cone-gains.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
