"""Preserve the inputs, original requests and text evidence for the reported full variants.

Pass roots containing candidate directories with source.coilgen and run/result.json.
The previous gallery's preserved text evidence is used only for its historical rows.
"""

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.session_evidence import run_digest


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "gui_review_2026-11-03/submitted-reports.zip"
TEXT = ("request.json", "manifest.json", "result.json", "design_report.md")


def collect(roots):
    sources = {}
    for root in roots:
        for source in root.rglob("source.coilgen"):
            run = source.parent / "run"
            if not (run / "result.json").is_file():
                continue
            meta = json.loads((run / "result.json").read_bytes())
            sources[meta["request_sha256"]] = (source, run)
    return sources


def archive(roots):
    cases = list(csv.DictReader((HERE / "route_trials.csv").open(encoding="utf-8", newline="")))
    if len(cases) != len({r["request"] for r in cases}):
        raise ValueError("Each receipt must identify a distinct run.")
    sources = collect(roots)
    target = HERE / "candidate_evidence.zip"
    partial = target.with_suffix(".zip.tmp")
    index = StringIO()
    writer = csv.DictWriter(
        index, fieldnames=("case", "variant", "request_sha256", "metadata_sha256", "has_source_project")
    )
    writer.writeheader()
    try:
        with ZipFile(partial, "w", ZIP_DEFLATED, compresslevel=6) as out, ZipFile(PREVIOUS) as earlier:
            for row in cases:
                key, variant, digest = row["case"], row["variant"], row["request"]
                prefix = f"{key}/{digest}/"
                if variant == "previous gallery":
                    files = {name: earlier.read(f"{key}/{name}") for name in TEXT}
                    meta = json.loads(files["result.json"])
                    assert meta["request_sha256"] == digest
                    for name, content in files.items():
                        out.writestr(prefix + name, content)
                    has_source = False
                else:
                    if digest not in sources:
                        raise ValueError(f"Missing exact source for {key} ({variant}): {digest}")
                    source, run = sources[digest]
                    project = ProjectDocument.load(source)
                    meta = DesignService.load_run(run, array_names=()).metadata
                    assert not project.saved_runs
                    assert (project.id, project.revision, project.fingerprint, project.model_fingerprint) == (
                        meta["project_id"],
                        meta["project_revision"],
                        meta["project_fingerprint"],
                        meta["project_model_fingerprint"],
                    )
                    assert meta["request_sha256"] == digest
                    out.write(source, prefix + "source.coilgen")
                    for name in TEXT:
                        out.write(run / name, prefix + name)
                    has_source = True
                writer.writerow(
                    dict(
                        case=key,
                        variant=variant,
                        request_sha256=digest,
                        metadata_sha256=run_digest(meta),
                        has_source_project=has_source,
                    )
                )
            out.writestr("index.csv", index.getvalue().encode("utf-8"))
        with ZipFile(partial) as checked:
            assert checked.testzip() is None
            assert len(checked.namelist()) == 1 + sum(4 if r["variant"] == "previous gallery" else 5 for r in cases)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"{len(cases)} verified candidate receipts, {target.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    archive(parser.parse_args().roots)
