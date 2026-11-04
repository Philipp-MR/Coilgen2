"""Identify closed contour families across mesh refinements from physical raw checkpoints.

Usage: python census.py output.csv name=run/checkpoints/contours.cgstage [...]
Coordinates are in the world frame. Signed bore winding applies to cones
and cylinders coaxial with world z. Plane area applies to plates normal to x.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from coilgen.pipeline_stages import PipelineCheckpoint


def rows(path, name):
    cp = PipelineCheckpoint.load(path)
    sheet = cp.state.parts
    cone = cp.config.coil_mesh_file == "create conical frustum mesh"
    for support_index, part in enumerate(sheet):
        levels = sorted({c.potential for c in part.contour_lines})
        for contour_index, c in enumerate(part.contour_lines):
            p = c.v * 1000
            segment = np.diff(p, axis=0)
            ds = np.linalg.norm(segment, axis=1)
            midpoint = 0.5 * (p[:-1] + p[1:])
            center = np.average(midpoint, axis=0, weights=ds)
            x, y, z = p.T
            circulation = np.arctan2(x[:-1] * y[1:] - y[:-1] * x[1:], x[:-1] * x[1:] + y[:-1] * y[1:]).sum()
            yz_area = 0.5 * np.sum(y[:-1] * z[1:] - z[:-1] * y[1:])
            arc = np.r_[0, np.cumsum(ds)]
            stations = np.linspace(0, arc[-1], max(4, round(arc[-1] / 3)) + 1)
            sample = np.column_stack([np.interp(stations, arc, axis) for axis in p.T])
            tangent = np.diff(sample, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            angles = np.rad2deg(np.arccos(np.einsum("ij,ij->i", tangent, np.roll(tangent, 1, axis=0)).clip(-1, 1)))
            yield {
                "mesh": name,
                "support": support_index,
                "contour_index": contour_index,
                "level_rank_low_to_high": levels.index(c.potential),
                "potential_A": c.potential,
                "current_A": part.contour_step,
                "current_orientation": c.current_orientation,
                "multiplicity": c.turn_multiplicity,
                "xy_winding": int(round(circulation / (2 * np.pi))) if cone else None,
                "signed_yz_area_mm2": yz_area if not cone else None,
                "centroid_x_mm": center[0],
                "centroid_y_mm": center[1],
                "centroid_z_mm": center[2],
                "min_z_mm": z.min(),
                "max_z_mm": z.max(),
                "length_mm": arc[-1],
                "max_turn_3mm_deg": angles.max(),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+", help="label=path to a raw contour checkpoint")
    options = parser.parse_args()
    records = []
    for spec in options.checkpoints:
        label, checkpoint = spec.split("=", 1)
        records.extend(rows(Path(checkpoint), label))
    with options.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
