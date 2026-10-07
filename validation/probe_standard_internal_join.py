"""Inspect one unchanged standard-gradient spiral join and its supported families."""

# ruff: noqa: E402 -- Direct execution must expose repository imports first.

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from validation.replay_standard_joint_routing import json_value
from coilgen.group_routing import _edge_key, _protected_group_segments
from coilgen.joint_cut_variants import prepare_joint_group_context
from coilgen import joint_route_shapes as shapes
from coilgen.routing_objective import routing_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("GY", "GZ"), default="GZ")
    parser.add_argument("--lift-mm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "tmp" / "replay_standard_joint_routing" / (args.axis + "_before") / "baseline.pkl").open("rb") as stream:
        case = pickle.load(stream)
    case.config.normal_shift_length = args.lift_mm / 1000.0
    context = prepare_joint_group_context(case.part, case.config)
    _protected, lookup = _protected_group_segments(context)
    points = context.connected_groups[0].spiral_in_v
    identities = [lookup.get(_edge_key(a, b)) for a, b in zip(points[:-1], points[1:])]
    start = next(index for index, identity in enumerate(identities) if identity is None)
    end = start + 1
    while end < len(identities) and identities[end] is None:
        end += 1
    source, before, after = points[start:end + 1], points[start] - points[start - 1], points[end + 1] - points[end]
    evaluator = shapes.JointTransitionEvaluator(context, case.config)
    record = {"axis": args.axis, "lift_limit_m": case.config.normal_shift_length,
        "endpoints_m": source[[0, -1]], "tangent_before": before, "tangent_after": after,
        "minimum_radius_m": evaluator.minimum_radius, "clearance_m": evaluator.clearance, "raw_families": []}

    originals = {name: getattr(shapes, name) for name in ("generate_planar_return_curves", "generate_cylindrical_transition_curves")}

    def wrap(name):
        def generate(*gargs, **kwargs):
            result = originals[name](*gargs, **kwargs)
            record["raw_families"].append({"generator": name, "reason": result.reason, "rejected_reasons": result.rejected_reasons,
                "candidate_count": len(result.candidates), "family_jumps": {f"{family}:jumps{jumps}": count for (family, jumps), count in Counter(
                    (getattr(curve, "profile_family", "direct_arch"), evaluator.jump_count(curve)) for curve in result.candidates).items()}})
            return result
        return generate

    for name in originals:
        setattr(shapes, name, wrap(name))
    try:
        curves = evaluator.supported_curves(source, before, after)
    finally:
        for name, original in originals.items():
            setattr(shapes, name, original)
    record["supported_curves"] = [{"family": getattr(curve, "profile_family", "direct_arch"), "jump_count": evaluator.jump_count(curve),
        "length_m": float(np.linalg.norm(np.diff(curve.points, axis=0), axis=1).sum()),
        "cost_m": routing_cost(curve.points, case.config.b_0_direction, case.config.joint_group_routing_b0_weight),
        "maximum_height_m": getattr(curve, "maximum_height_m", None)} for curve in curves]
    record["support_rejected_reasons"] = evaluator.last_rejected_reasons
    record["evaluator_diagnostics"] = evaluator.diagnostics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, default=json_value))
    print(json.dumps(record, indent=2, default=json_value), flush=True)


if __name__ == "__main__":
    main()
