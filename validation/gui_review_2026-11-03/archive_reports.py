"""Keep the submitted inputs and engineering reports without the temporary stage caches."""

import csv
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    with (HERE / "smoothing.csv").open(encoding="utf-8", newline="") as src:
        cases = list(csv.DictReader(src))
    if len(cases) != 29:
        raise ValueError("Wait for all 29 comparisons before archiving.")
    target = HERE / "submitted-reports.zip"
    temp = target.with_suffix(".zip.tmp")
    try:
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for case in cases:
                run = (ROOT / case["directory"]).resolve()
                if not run.is_relative_to((ROOT / "tmp" / "gui_review_smoothing").resolve()):
                    raise ValueError(f"Unexpected run path: {run}")
                figure_data = (HERE / case["centerlines_file"]).read_bytes()
                if sha256(figure_data).hexdigest() != case["centerlines_sha256"]:
                    raise ValueError(f"Centerlines differ from the verified comparison: {run.name}")
                result = json.loads((run / "result.json").read_bytes())
                if result["request_sha256"] != case["smoothed_request_sha256"]:
                    raise ValueError(f"Report belongs to a different request: {run.name}")
                for name in ("request.json", "manifest.json", "result.json", "design_report.md"):
                    archive.write(run / name, f"{run.name}/{name}")
        with ZipFile(temp) as archive:
            if len(archive.namelist()) != 4 * len(cases) or archive.testzip() is not None:
                raise ValueError("Incomplete gallery archive.")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
