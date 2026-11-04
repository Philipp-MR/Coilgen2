"""Stage completed gallery results in a dedicated local browser QA workspace without solving.

Pass --workspace HUB_ROOT/users/local and --scratch FULL_GALLERY_SCRATCH.
Use --existing KEY=DIRECTORY for cases computed elsewhere. DIRECTORY contains
source.coilgen and run/result.json. Rerun to add newly published gallery rows.
Start or restart the private browser worker after refresh so it discovers the new
saved sources and immutable result history. --check KEY verifies an end-to-end
source/result selection through WorkbenchSession and the original request reader.
Use --check all to verify the complete index in one session.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.bundles import RunArchive, project_key, revision_key
from coilgen.application.evidence import load_saved_request
from coilgen.application.session import WorkbenchSession
from coilgen.application.session_evidence import run_digest
from coilgen.persistence import atomic_json, file_digest, publish_exclusive


HERE = Path(__file__).resolve().parent
HOST_QA = Path.home() / "AppData/Local/Temp/coilgen-winding-quality-20261103"
INDEX = ".coilgen-gallery-qa.json"


def prepare_workspace(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    marker = workspace / INDEX
    if not marker.exists():
        if any(workspace.iterdir()):
            raise ValueError(f"Refusing to stage into a nonempty, unmarked workspace: {workspace}")
        atomic_json(marker, {"fixture": "coilgen-winding-gallery-qa/1", "cases": {}})
    index = json.loads(marker.read_text(encoding="utf8"))
    if index.get("fixture") != "coilgen-winding-gallery-qa/1":
        raise ValueError("Unexpected workspace marker.")
    return index


def publish_archive(workspace, record, archive):
    """Materialize the genuine RunArchive, linking only immutable payload files."""
    root = workspace / "embedded-runs"
    root.mkdir(exist_ok=True)
    destination = root / archive.id
    if destination.exists():
        if destination.resolve() != destination:
            raise ValueError("Unexpected redirected QA cache.")
        archive.materialize(root)
        return
    with TemporaryDirectory(dir=root, prefix=".qa-", ignore_cleanup_errors=True) as temp:
        staged = Path(temp) / "result"
        staged.mkdir()
        for name, payload in archive._files:
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            source = record.directory / name
            # The workbench treats these archived files as immutable. No filesystem
            # flags on a shared inode are changed. JSON and source documents are copies.
            if source.is_file() and path.suffix.lower() in {".cgstage", ".npz", ".stl", ".csv", ".step"}:
                try:
                    os.link(source, path)
                except OSError:
                    path.write_bytes(payload)
            else:
                path.write_bytes(payload)
            if file_digest(path) != sha256(payload).hexdigest():
                raise ValueError(f"Archive changed while staging: {name}")
        staged.rename(destination)


def stage_case(workspace, scratch, row, existing, cases):
    key = f"{row['template']}__{row['channel_index']}"
    directory = existing.get(key, scratch / key / row["model_fingerprint"][:16])
    source = directory / "source.coilgen"
    project = ProjectDocument.load(source)
    source_hash = file_digest(source)
    if project.model_fingerprint != row["model_fingerprint"] or project.name != row["channel"]:
        raise ValueError(f"{key}: source differs from the evaluated gallery model.")
    sources = workspace / "projects"
    sources.mkdir(exist_ok=True)
    filename = f"qa-{key}.coilgen"
    saved = sources / filename
    previous = cases.get(key)
    disk_hash = file_digest(saved) if saved.exists() else None
    if disk_hash is not None and disk_hash != source_hash and (not previous or disk_hash != previous["source_sha256"]):
        raise ValueError(f"Refusing to replace a saved file not owned by this fixture: {saved}")
    if previous and previous["request_sha256"] == row["request_sha256"] and previous["source_sha256"] == source_hash:
        staged_result = workspace / "embedded-runs" / previous["archive_id"] / "result.json"
        if saved.is_file() and staged_result.is_file():
            result_hash = file_digest(directory / "run/result.json")
            if file_digest(staged_result) == result_hash:
                stored = json.loads(staged_result.read_bytes())
                if (
                    stored.get("request_sha256") == row["request_sha256"]
                    and run_digest(stored) == previous["metadata_sha256"]
                ):
                    return {
                        **previous,
                        "metadata_checksum_recorded": bool(stored.get("metadata_sha256")),
                        "result_file_sha256": result_hash,
                    }
    record = DesignService.load_run(directory / "run", array_names=())
    meta = record.metadata_view
    if (
        project_key(project) != revision_key(meta)
        or meta["request_sha256"] != row["request_sha256"]
        or meta["summary"]["scope"] != "fabrication_review"
    ):
        raise ValueError(f"{key}: source and evaluated record disagree.")
    archive = RunArchive.capture(record, source_project=project)
    publish_archive(workspace, record, archive)
    if disk_hash != source_hash:
        with TemporaryDirectory(dir=sources, prefix=".qa-", ignore_cleanup_errors=True) as temp:
            copy = Path(temp) / filename
            shutil.copyfile(source, copy)
            if file_digest(copy) != source_hash:
                raise ValueError(f"{key}: source changed while staging.")
            if saved.exists():
                copy.replace(saved)
            else:
                publish_exclusive(copy, saved)
    return dict(
        archive_id=archive.id,
        filename=filename,
        project_id=project.id,
        revision=project.revision,
        project_fingerprint=project.fingerprint,
        model_fingerprint=project.model_fingerprint,
        request_sha256=meta["request_sha256"],
        metadata_sha256=run_digest(meta),
        metadata_checksum_recorded=bool(meta.get("metadata_sha256")),
        result_file_sha256=file_digest(record.directory / "result.json"),
        source_sha256=source_hash,
        scope=meta["summary"]["scope"],
    )


def check_case(session, case):
    """Exercise the ordinary workbench's source open, selection and request verification."""
    session.open_project(session.workspace / "projects" / case["filename"])
    matches = [
        s
        for s in session.completed_runs()
        if s.get("project_id") == session.project.id and s.get("project_fingerprint") == session.project.fingerprint
    ]
    state = next(s for s in matches if s["id"] == case["archive_id"])
    record = session.load_run(state["id"])
    req = load_saved_request(record)
    meta = record.metadata_view
    if (
        project_key(session.project) != revision_key(meta)
        or meta["request_sha256"] != case["request_sha256"]
        or run_digest(meta) != case["metadata_sha256"]
        or file_digest(record.directory / "result.json") != case["result_file_sha256"]
        or req["project_id"] != session.project.id
        or req["pipeline_stage"] != "evaluated"
        or meta["summary"]["scope"] != "fabrication_review"
        or not any(
            k.startswith("review_conductor_") and k.endswith("_points_m") for k in record.select_arrays(copy=False)
        )
    ):
        raise ValueError("Browser source, original request, reviewed geometry or run scope was lost.")
    return f"{session.project.name} · r{session.project.revision} · {req['pipeline_stage']} · {case['archive_id'][:8]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=HOST_QA / "gallery-browser/users/local")
    parser.add_argument("--scratch", type=Path, default=HOST_QA / "gallery-full")
    parser.add_argument(
        "--existing", action="append", default=[], help="KEY=DIRECTORY containing source.coilgen and run/"
    )
    parser.add_argument("--case", action="append", help="Stage these completed channel keys only.")
    parser.add_argument(
        "--check", action="append", default=[], help="Verify this staged channel through WorkbenchSession."
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    scratch = args.scratch.expanduser().resolve()
    index = prepare_workspace(workspace)
    existing = {k: Path(p).expanduser().resolve() for k, p in (x.split("=", 1) for x in args.existing)}
    records = HERE / "reviewed_centerlines"
    for path in sorted(records.glob("*.json")):
        if args.case and path.stem not in args.case:
            continue
        row = json.loads(path.read_text(encoding="utf8"))
        key = f"{row['template']}__{row['channel_index']}"
        if path.stem != key:
            raise ValueError(f"Wrong reviewed row name: {path}")
        case = stage_case(workspace, scratch, row, existing, index["cases"])
        if case != index["cases"].get(key):
            index["cases"][key] = case
            atomic_json(workspace / INDEX, index)
            print("Staged", key, case["filename"], case["archive_id"][:8], flush=True)
    missing = set(args.case or ()) - set(index["cases"])
    if missing:
        raise ValueError(f"No published result rows for {sorted(missing)}")
    if args.check:
        keys = index["cases"] if "all" in args.check else args.check
        session = WorkbenchSession(workspace)
        try:
            for key in keys:
                print("Workbench verified", check_case(session, index["cases"][key]), flush=True)
        finally:
            session.close()
    print(f"{len(index['cases'])} staged channels. Workspace: {workspace}", flush=True)


if __name__ == "__main__":
    main()
