"""Match observations from the hidden browser to the finished gallery by exact archive ID."""

import argparse
import json
import math
from pathlib import Path
import re

from coilgen.application.templates import TEMPLATE_SPECS, template_assembly


HERE = Path(__file__).resolve().parent


def verify(observations, index):
    cases = {}
    for spec in TEMPLATE_SPECS:
        for i, channel in enumerate(template_assembly(spec.id).channels):
            key = f"{spec.id}__{i}"
            row = json.loads((HERE / "reviewed_centerlines" / f"{key}.json").read_text("utf8"))
            saved = index["cases"][key]
            archive_id = saved["archive_id"]
            case = observations[archive_id]
            if (
                row["request_sha256"] != saved["request_sha256"]
                or saved["project_fingerprint"] != channel.project.fingerprint
                or row["project_fingerprint"] != saved["project_fingerprint"]
                or not case["label"].endswith(f" · evaluated · {archive_id[:8]}")
            ):
                raise ValueError(f"{key}: the displayed archive is not the current full evaluation.")
            counts = re.match(r"Before routing: (\d+) of (\d+) contours smoothed", case["before_routing"])
            local = re.search(r"(\d+) using local corner blends", case["before_routing"])
            if not counts or tuple(map(int, counts.groups())) != (
                row["smoothed"],
                row["smoothed"] + row["original_retained"],
            ):
                raise ValueError(f"{key}: the displayed fit count does not match the run.")
            if (int(local[1]) if local else 0) != row["local"]:
                raise ValueError(f"{key}: the displayed local fit count does not match the run.")
            if not math.isclose(float(case["reviewed_dsv_rms"].rstrip(" %")), row["review_rms_percent"], rel_tol=5e-4):
                raise ValueError(f"{key}: the displayed mounted-wire error does not match the run.")
            for titles, field in (
                (("Stream function model", "Continuous sheet"), "sheet_accepted"),
                (("Pipeline conductor sweep",), "pipeline_sweep"),
                (("Strict fabrication review",), "fabrication"),
            ):
                value = "Passed" if row[field] else "Failed"
                if not any(f"{title} · {value}" in case["checks"] for title in titles):
                    raise ValueError(f"{key}: the displayed {titles[0]} verdict does not match the run.")
            cases[key] = {**case, "archive_id": archive_id, "request_sha256": row["request_sha256"]}
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("fixture_index", type=Path)
    args = parser.parse_args()
    cases = verify(json.loads(args.observations.read_bytes()), json.loads(args.fixture_index.read_bytes()))
    (HERE / "browser-gallery.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(cases)} final browser results.")
