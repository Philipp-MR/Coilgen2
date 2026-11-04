"""Portable evidence and exact source revisions for workbench sessions."""

from dataclasses import replace
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ..persistence import publish_exclusive
from ..research.storage import result_metadata_digest
from .assembly import ASSEMBLY_SCHEMA, LEGACY_ASSEMBLY_SCHEMAS, ProjectAssembly
from .bundles import SOURCE_PROJECT_FILE, RunArchive, project_key, revision_key
from .evidence import verified_run_metadata
from .portable import is_bundle, load_bundle
from .project import ProjectDocument, _json_object


def run_digest(metadata):
    return metadata.get("metadata_sha256") or result_metadata_digest(
        {key: value for key, value in metadata.items() if key != "job"}
    )


def load_assembly(path):
    if is_bundle(path):
        document = load_bundle(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object)
        if not isinstance(data, dict):
            raise ValueError("Project file must contain a JSON object.")
        loader = (
            ProjectAssembly if data.get("schema") in {ASSEMBLY_SCHEMA, *LEGACY_ASSEMBLY_SCHEMAS} else ProjectDocument
        )
        document = loader.from_dict(data, project_root=path.parent)
    assembly = document if isinstance(document, ProjectAssembly) else ProjectAssembly.new(document.name, (document,))
    if not assembly.channels:
        raise ValueError("This project has no channels.")
    return assembly


def save_new_assembly(assembly, directory, stem, extension=".coilgen.json"):
    """Publish a new default document atomically, without replacing another design."""
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory, ignore_cleanup_errors=True) as staging:
        source = assembly.save(Path(staging) / f"project{extension}")
        index = 1
        while True:
            suffix = "" if index == 1 else f"-{index}"
            destination = directory / f"{stem}{suffix}{extension}"
            try:
                publish_exclusive(source, destination)
            except FileExistsError:
                index += 1
            else:
                return destination


def revision_path(workspace, key):
    digest = sha256(json.dumps(key).encode("utf-8")).hexdigest()
    return workspace / "revisions" / f"{digest}.coilgen"


def retain_revision(workspace, project):
    path = revision_path(workspace, project_key(project))
    if not path.exists():
        project.inputs_only().save(path)


def source_project(workspace, record, channels, loader, *, search_saved=False):
    metadata = getattr(record, "metadata_view", None)
    key = revision_key(record.metadata if metadata is None else metadata)
    path = revision_path(workspace, key)
    if not path.exists():
        path = record.directory / SOURCE_PROJECT_FILE
    if path.is_file():
        source = loader(path)
        if len(source.channels) != 1:
            raise ValueError("The retained CAD source must contain the one project used by this computation.")
        project = source.channels[0].project
        if project_key(project) != key:
            raise ValueError("The retained CAD source project does not match the selected run's recorded identity.")
        return project.inputs_only()
    for channel in channels:
        if project_key(channel.project) == key:
            return channel.project.inputs_only()
    # Older workspaces recorded no submission snapshot. The filename or an
    # identical coil fingerprint alone cannot establish project ownership.
    if search_saved:
        for root in (workspace / "projects", workspace / "uploads"):
            if not root.resolve().is_relative_to(workspace):
                continue
            candidates = root.glob("*")
            if root.name == "uploads":
                children = (d for d in root.glob("*") if d.is_dir() and d.resolve().is_relative_to(root))
                candidates = chain(candidates, (path for child in children for path in child.glob("*")))
            for path in sorted(candidates):
                if (
                    path.suffix.lower() not in {".json", ".coilgen"}
                    or not path.resolve().is_relative_to(root)
                    or not path.is_file()
                ):
                    continue
                try:
                    assembly = loader(path)
                except (OSError, ValueError, TypeError, KeyError, RuntimeError):
                    continue
                for channel in assembly.channels:
                    if project_key(channel.project) == key:
                        retain_revision(workspace, channel.project)
                        return channel.project.inputs_only()
    return None


def portable_assembly(assembly, states, load_record, workspace, active_id, project_loader, *, selected_run=...):
    channels = []
    for channel in assembly.channels:
        project = channel.project
        archives = {archive.id: archive for archive in project.saved_runs}
        known = {run_digest(archive.metadata): archive.id for archive in archives.values()}
        for state in states:
            if state.get("project_id") != project.id:
                continue
            record = load_record(state["id"])
            metadata = record.metadata_view
            if metadata.get("project_id") != project.id:
                raise ValueError("The completed job's run belongs to a different project.")
            digest = run_digest(metadata)
            if digest in known:
                continue
            original = source_project(workspace, record, assembly.channels, project_loader)
            archive = RunArchive.capture(record, source_project=original)
            archives[archive.id] = archive
            known[digest] = archive.id
        session = {**project.session, "active_channel_id": active_id}
        if channel.id == active_id and selected_run is not ...:
            selected_id = None
            if selected_run is not None and selected_run.metadata_view.get("project_id") == project.id:
                digest = run_digest(selected_run.metadata_view)
                # Prefer the exact materialized archive, then the previous equivalent archive.
                for candidate in (selected_run.directory.name, session.get("selected_run_id"), known.get(digest)):
                    if (
                        isinstance(candidate, str)
                        and candidate in archives
                        and run_digest(archives[candidate].metadata) == digest
                    ):
                        selected_id = candidate
                        break
            session["selected_run_id"] = selected_id
        channels.append(replace(channel, project=project.with_results(archives.values(), session=session)))
    return replace(assembly, channels=tuple(channels))


def embedded_state(directory, metadata, *, alias=None, channel_id=None):
    return {
        "id": directory.name,
        "name": alias or metadata.get("name") or directory.name,
        "status": "completed",
        "progress": 1.0,
        "stage": "Reopened verified result",
        "embedded": True,
        "channel_id": channel_id,
        "evidence_digest": run_digest(metadata),
        **{
            key: metadata.get(key)
            for key in (
                "project_id",
                "project_revision",
                "project_fingerprint",
                "project_model_fingerprint",
                "pipeline_stage",
                "request_sha256",
                "metadata_sha256",
            )
        },
    }


def restore_embedded(directory):
    for path in sorted(directory.glob("*/result.json")):
        if len(path.parent.name) != 64 or any(c not in "0123456789abcdef" for c in path.parent.name):
            continue
        try:
            metadata, _, _ = verified_run_metadata(path.parent)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            yield {"id": path.parent.name, "embedded": True, "status": "unavailable", "error": str(exc)}
        else:
            yield embedded_state(path.parent, metadata)


def open_embedded(assembly, directory):
    states = []
    for channel in assembly.channels:
        for archive in channel.project.saved_runs:
            record = archive.materialize(directory)
            states.append(
                embedded_state(record.directory, record.metadata_view, alias=archive.alias, channel_id=channel.id)
            )
    return states
