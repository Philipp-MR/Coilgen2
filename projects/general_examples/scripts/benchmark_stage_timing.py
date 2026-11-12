"""Per-step computation-time breakdown for the GUI design templates.

Runs each selected template exactly as the GUI does (``DesignService.run_assembly``)
into a fresh output directory, so no checkpoint or completed result is reused.
It collects three levels of timing without touching the core engine:

1. Pipeline steps: the engine's own ``timing`` rows in ``result.json``.
2. Sub-steps: every progress event is time-stamped; the time until the next
   event is attributed to the operation/label/message that was active. This
   splits long steps such as "Joint routing" into transition search, topology
   shaping, finalist validation, ...
3. Overhead: wall-clock time outside the pipeline timer (request preparation,
   reports, previews, artifact writing).

Outputs (in outputs/stage_timing/<timestamp>/):
    stage_timing.csv     one row per template x pipeline step
    substep_timing.csv   one row per template x sub-step (aggregated)
    events/<id>.jsonl    raw time-stamped progress events
    summary.md           ranked tables
    runs/<id>/           the normal run output of each template

Usage (from the CoilGen2 folder):
    python projects/general_examples/scripts/benchmark_stage_timing.py
    python projects/general_examples/scripts/benchmark_stage_timing.py --templates cylinder biplanar
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    for _path in (_Path(__file__).resolve().parents[3], _Path(__file__).resolve().parents[3] / "src"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from projects._shared.example_utils import PROJECT_ROOT

DEFAULT_TEMPLATES = ("cylinder", "biplanar", "halbach_gy", "shielded_gy", "shim_x2_y2")

# Pipeline step label -> coarse phase, for the per-phase summary.
PHASES = (
    (
        "prepare",
        (
            "loaded meshes",
            "prepared coil meshes",
            "defined target field",
            "built one-ring",
            "built stream-function",
            "calculated sensitivity",
        ),
    ),
    ("solve", ("calculated resistance", "optimized stream function", "loaded preoptimized")),
    (
        "contours",
        (
            "calculated potential",
            "extracted contours",
            "filtered loop",
            "regularized contour",
            "validated contourability",
            "processed contour",
            "calculated contour distances",
        ),
    ),
    (
        "draft routing",
        ("grouped contour topology", "interconnected within", "interconnected among", "shifted return", "cleaned wire"),
    ),
    ("route check/improve", ("realized fabrication", "selected joint group routes")),
    ("sweep/build", ("finalized swept", "built grooved", "skipped topology")),
    ("evaluate", ("calculated inductance", "evaluated field errors", "evaluated gradient slice")),
)


def phase_of(step: str) -> str:
    for phase, prefixes in PHASES:
        if any(step.startswith(prefix) for prefix in prefixes):
            return phase
    return "other"


_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def normalize_message(text: str) -> str:
    text = _NUMBER.sub("#", str(text or ""))
    # Drop volatile tails (elapsed time, current candidate, phase notes) so that
    # repeated updates of one activity aggregate into one row.
    text = re.sub(r";\s*(execution time|elapsed).*$", "", text)
    text = re.sub(r"^part #/#:?\s*", "", text)
    text = re.sub(r"(failure_directed_ports|directed_ports)_#", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:90]


class EventRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.events: list[dict] = []
        self.t0 = time.perf_counter()

    def __call__(self, event) -> None:
        payload = dict(getattr(event, "payload", None) or {})
        self.events.append(
            {
                "t": time.perf_counter() - self.t0,
                "kind": getattr(event, "kind", ""),
                "operation": str(payload.get("operation") or ""),
                "label": str(payload.get("label") or ""),
                "message": str(getattr(event, "message", "") or payload.get("message") or ""),
            }
        )

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for row in self.events:
                handle.write(json.dumps(row) + "\n")

    def substeps(self, end_time: float) -> dict[tuple[str, str, str], tuple[float, int]]:
        totals: dict[tuple[str, str, str], list] = defaultdict(lambda: [0.0, 0])
        for index, row in enumerate(self.events):
            following = self.events[index + 1]["t"] if index + 1 < len(self.events) else end_time
            key = (row["operation"], row["label"], normalize_message(row["message"]))
            totals[key][0] += max(0.0, following - row["t"])
            totals[key][1] += 1
        return {key: (value[0], value[1]) for key, value in totals.items()}


def size_metrics(result: dict) -> dict[str, object]:
    """Small set of problem-size numbers, best effort (missing keys are skipped)."""
    out: dict[str, object] = {}
    summary = result.get("summary") or {}
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lowered = key.lower()
                if any(s in lowered for s in ("vert", "count", "loops", "groups", "turns", "contour", "length")):
                    out[f"summary.{key}"] = value
    parts = (result.get("routing") or {}).get("parts") or []
    out["routing.parts"] = len(parts)
    # Did the joint route search change the draft route, or keep it?
    statuses = []
    for part in parts:
        joint = part.get("joint_routing") if isinstance(part, dict) else None
        if isinstance(joint, dict) and joint.get("status"):
            statuses.append(f"{joint.get('status')}({joint.get('reason') or joint.get('selected_route_status') or ''})")
    out["joint_routing"] = "; ".join(statuses)
    return out


def run_template(template_id: str, run_root: Path, events_root: Path, threads: int) -> dict:
    from coilgen.application.service import DesignService
    from coilgen.application.templates import template_assembly

    recorder = EventRecorder(events_root / f"{template_id}.jsonl")
    out_dir = run_root / template_id
    record: dict = {"template": template_id, "status": "ok", "error": ""}
    wall_start = time.perf_counter()
    try:
        assembly = template_assembly(template_id, project_root=PROJECT_ROOT)
        enabled = [channel for channel in assembly.channels if channel.enabled]
        results = {}
        for channel in enabled:
            DesignService.validate(channel.project).raise_for_errors()
            results[channel.id] = DesignService.run(
                channel.project, out_dir / channel.id, threads=threads, progress=recorder
            )
        record["results"] = results
    except Exception as exc:  # keep going with the other templates
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    record["wall_seconds"] = time.perf_counter() - wall_start
    recorder.write()
    record["substeps"] = recorder.substeps(record["wall_seconds"])
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--templates", nargs="+", default=list(DEFAULT_TEMPLATES))
    parser.add_argument("--threads", type=int, default=1, help="threads per run (GUI default is 1)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.output or (PROJECT_ROOT / "outputs" / "stage_timing" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_root, events_root = root / "runs", root / "events"
    root.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {root}", flush=True)

    # Warm-up: imports and numba compilation are paid once here, not by the first template.
    warm = time.perf_counter()
    import coilgen.application.service  # noqa: F401
    from coilgen import geometry_kernels  # noqa: F401

    print(f"imports {time.perf_counter() - warm:.1f} s", flush=True)

    stage_rows, sub_rows, totals = [], [], []
    for template_id in args.templates:
        print(f"\n=== {template_id} ===", flush=True)
        record = run_template(template_id, run_root, events_root, args.threads)
        pipeline_total = 0.0
        sizes: dict = {}
        for channel_id, result in (record.get("results") or {}).items():
            timing = result.get("timing") or []
            for row in timing:
                if row.get("stage") == "start":
                    continue
                stage_rows.append(
                    {
                        "template": template_id,
                        "channel": channel_id,
                        "phase": phase_of(str(row["stage"])),
                        "step": row["stage"],
                        "seconds": round(float(row["stage_seconds"]), 3),
                    }
                )
            if timing:
                pipeline_total += float(timing[-1]["elapsed_seconds"])
            sizes.update(size_metrics(result))
        for (operation, label, message), (seconds, count) in sorted(
            record["substeps"].items(), key=lambda item: -item[1][0]
        ):
            if seconds < 0.05:
                continue
            sub_rows.append(
                {
                    "template": template_id,
                    "operation": operation,
                    "label": label,
                    "message": message,
                    "seconds": round(seconds, 3),
                    "events": count,
                }
            )
        totals.append(
            {
                "template": template_id,
                "status": record["status"],
                "wall_seconds": round(record["wall_seconds"], 2),
                "pipeline_seconds": round(pipeline_total, 2),
                "overhead_seconds": round(record["wall_seconds"] - pipeline_total, 2),
                "error": record["error"],
                **sizes,
            }
        )
        print(
            f"{template_id}: {record['status']} wall {record['wall_seconds']:.1f} s, pipeline {pipeline_total:.1f} s",
            flush=True,
        )
        _write_outputs(root, stage_rows, sub_rows, totals)
    print(f"\nDone. Summary: {root / 'summary.md'}", flush=True)
    return 0 if all(t["status"] == "ok" for t in totals) else 1


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(root: Path, stage_rows, sub_rows, totals) -> None:
    _write_csv(root / "stage_timing.csv", stage_rows)
    _write_csv(root / "substep_timing.csv", sub_rows)
    _write_csv(root / "totals.csv", totals)

    lines = ["# Stage timing", "", "| template | status | wall s | pipeline s | overhead s |", "|---|---|---|---|---|"]
    for t in totals:
        lines.append(
            f"| {t['template']} | {t['status']} | {t['wall_seconds']} | {t['pipeline_seconds']} | {t['overhead_seconds']} |"
        )
    phase_names = [name for name, _ in PHASES] + ["other"]
    by_phase: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in stage_rows:
        by_phase[row["template"]][row["phase"]] += row["seconds"]
    lines += ["", "## Seconds per phase", "", "| template | " + " | ".join(phase_names) + " |"]
    lines.append("|---" * (len(phase_names) + 1) + "|")
    for template, phases in by_phase.items():
        lines.append(f"| {template} | " + " | ".join(f"{phases.get(p, 0.0):.1f}" for p in phase_names) + " |")
    lines += ["", "## Slowest steps (all templates)", "", "| template | step | s |", "|---|---|---|"]
    for row in sorted(stage_rows, key=lambda r: -r["seconds"])[:20]:
        lines.append(f"| {row['template']} | {row['step']} | {row['seconds']:.1f} |")
    lines += [
        "",
        "## Slowest sub-steps (all templates)",
        "",
        "| template | operation | label | message | s |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(sub_rows, key=lambda r: -r["seconds"])[:30]:
        message = row["message"].replace("|", "/")
        lines.append(f"| {row['template']} | {row['operation']} | {row['label']} | {message} | {row['seconds']:.1f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
