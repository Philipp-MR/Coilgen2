"""Instrument bounded joint-routing trials on trusted saved replay checkpoints.

Overrides are explicit engineering trials: production defaults are untouched.
Progress and completed proposals are flushed to JSON while a trial runs.
"""

# ruff: noqa: E402 -- Direct execution must expose repository imports first.

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from validation.replay_standard_joint_routing import export_case, json_value, summarize
from coilgen import joint_group_adoption as adoption
from coilgen.joint_route_shapes import JointTransitionEvaluator


def run_trial(axis, checkpoint, output, *, lift_mm=None, field_rel=None, stl=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(checkpoint).open("rb") as stream:
        case = pickle.load(stream)
    if case.axis != axis:
        raise ValueError("Checkpoint channel does not match requested trial.")
    original_lift = case.config.normal_shift_length
    original_field_rel = case.config.joint_group_routing_max_field_change_rel
    if lift_mm is not None:
        case.config.normal_shift_length = lift_mm / 1000.0
    if field_rel is not None:
        case.config.joint_group_routing_max_field_change_rel = field_rel
    start = time.perf_counter()
    record = {
        "axis": axis, "checkpoint": str(Path(checkpoint).resolve()), "started_local": datetime.now().isoformat(),
        "original_lift_limit_m": original_lift, "trial_lift_limit_m": case.config.normal_shift_length,
        "original_field_change_rel": original_field_rel,
        "trial_field_change_rel": case.config.joint_group_routing_max_field_change_rel,
        "fixed_current_a": case.part.contour_step,
        "context_proposals": [], "shape_proposals": [], "candidate_validations": [], "progress": [],
    }

    def save():
        (output / "trial_progress.json").write_text(json.dumps(record, indent=2, default=json_value))

    def log(message, **details):
        event = {"elapsed_s": time.perf_counter() - start, "local_time": datetime.now().isoformat(), "message": message, **details}
        record["progress"].append(event)
        print(f"{event['local_time']} [{event['elapsed_s']:.1f}s] {message}" + (" " + json.dumps(details, default=json_value) if details else ""), flush=True)
        save()

    old_curves = JointTransitionEvaluator.supported_curves
    old_propose = adoption.propose_joint_group_routes
    old_shape = adoption.shape_joint_group_route
    old_validate = adoption.validate_group_route_candidate
    last_progress = {}

    def curves(self, source_points, tangent_before, tangent_after):
        before = self.diagnostics["generated_transition_count"]
        result = old_curves(self, source_points, tangent_before, tangent_after)
        count = self.diagnostics["generated_transition_count"]
        now = time.perf_counter()
        if count != before and (not result or count % 8 == 0 or now - last_progress.get(id(self), start) >= 20):
            last_progress[id(self)] = now
            log("transition generation", generated=count, supported_last=len(result),
                unsupported=self.diagnostics["unsupported_transition_count"],
                samples=self.diagnostics["curve_sample_count"],
                family_jumps={f"{family}:jumps{jumps}": number for (family, jumps), number in Counter(
                    (getattr(curve, "profile_family", "direct_arch"), self.jump_count(curve)) for curve in result).items()},
                rejected_reasons=self.last_rejected_reasons if not result else ())
        return result

    def propose(*args, **kwargs):
        context = args[0]
        log("starting supported graph", groups=len(context.groups))
        result = old_propose(*args, **kwargs)
        proposal_path = output / f"graph_{len(record['context_proposals']) + 1:03d}.pkl"
        with proposal_path.open("wb") as stream:
            pickle.dump({"context": context, "config": case.config, "target": case.target,
                "proposals": result, "evaluator": kwargs.get("transition_evaluator")}, stream, protocol=5)
        record["context_proposals"].append({"reason": result.reason, "candidate_count": len(result.candidates), "diagnostics": result.diagnostics})
        log("supported graph completed", reason=result.reason, candidates=len(result.candidates),
            jump_counts=result.diagnostics.get("candidate_jump_counts"), detail=result.diagnostics.get("detail"))
        return result

    def shape(*args, **kwargs):
        log("starting coupled shaping")
        result = old_shape(*args, **kwargs)
        record["shape_proposals"].append({"reason": result.reason, "candidate_count": len(result.candidates), "diagnostics": result.diagnostics})
        log("coupled shaping completed", reason=result.reason, candidates=len(result.candidates))
        return result

    def validate(*args, **kwargs):
        log("starting finite-body and fixed-current validation")
        checkpoint_path = output / f"candidate_{len(record['candidate_validations']) + 1:03d}.pkl"
        with checkpoint_path.open("wb") as stream:
            pickle.dump({"args": args, "kwargs": kwargs}, stream, protocol=5)
        result = old_validate(*args, **kwargs)
        record["candidate_validations"].append({"accepted": result.accepted, "reasons": result.reasons,
            "diagnostics": result.diagnostics, "checkpoint": str(checkpoint_path.resolve())})
        log("candidate validation completed", accepted=result.accepted, reasons=result.reasons)
        return result

    JointTransitionEvaluator.supported_curves = curves
    adoption.propose_joint_group_routes = propose
    adoption.shape_joint_group_route = shape
    adoption.validate_group_route_candidate = validate
    try:
        log("starting joint routing trial", lift_limit_m=case.config.normal_shift_length,
            field_change_rel=case.config.joint_group_routing_max_field_change_rel)
        case.part = adoption.improve_group_routes([case.part], case.config, case.target)[0]
        case.assert_source_unchanged()
        with (output / "selected.pkl").open("wb") as stream:
            pickle.dump(case, stream, protocol=5)
        export_case(case, output, stl=stl)
        record["summary"] = summarize(case)
        record["completed_local"] = datetime.now().isoformat()
        log("trial completed", status=case.part.joint_route_diagnostics.get("status"), reason=case.part.joint_route_diagnostics.get("reason"))
        return case, record
    except BaseException as error:
        record["error"] = repr(error)
        log("trial failed", error=repr(error))
        raise
    finally:
        JointTransitionEvaluator.supported_curves = old_curves
        adoption.propose_joint_group_routes = old_propose
        adoption.shape_joint_group_route = old_shape
        adoption.validate_group_route_candidate = old_validate
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("GY", "GZ"), default="GZ")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lift-mm", type=float)
    parser.add_argument("--field-rel", type=float)
    parser.add_argument("--export-stl", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint or ROOT / "tmp" / "replay_standard_joint_routing" / (args.axis + "_before") / "baseline.pkl"
    run_trial(args.axis, checkpoint, args.output, lift_mm=args.lift_mm, field_rel=args.field_rel, stl=args.export_stl)


if __name__ == "__main__":
    main()
