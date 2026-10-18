"""Prepare verified project input and saved evidence without using Qt widgets."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QRunnable, Signal

from ..application.assembly import ASSEMBLY_SCHEMA, LEGACY_ASSEMBLY_SCHEMAS, ProjectAssembly
from ..application.execution import execution_reuse
from ..application.history import HistoryEntry
from ..application.portable import is_bundle, load_bundle
from ..application.project import ProjectDocument, _freeze, _json_object
from ..application.reuse import ReusePlan
from .design_tasks import deliver_task_signal
from .readiness import check_readiness
from .run_metadata import read_run_metadata


@dataclass(frozen=True, slots=True)
class PreparedHistory:
    key: tuple[str, str | None]
    records: Mapping
    entries: tuple[HistoryEntry, ...]
    ids: Mapping
    embedded_ids: Mapping
    session: Mapping
    diagnostics: tuple[str, ...]
    unavailable_directories: tuple[Path, ...]

    def __post_init__(self):
        for name in ("records", "ids", "embedded_ids", "session"):
            object.__setattr__(self, name, _freeze(dict(getattr(self, name))))


@dataclass(frozen=True, slots=True)
class PreparedProjectOpen:
    path: Path
    document: ProjectDocument | ProjectAssembly
    active_channel_id: str | None
    history: PreparedHistory
    readiness: object = None
    reuse: ReusePlan | None = None


def load_document(path, *, progress=None):
    """Use the application schema, digest and path checks on one JSON read."""
    path = Path(path).expanduser().resolve()
    if progress is not None:
        progress(0, 0, "Reading project")
    if is_bundle(path):
        document = load_bundle(path, progress=progress)
    else:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object)
        if not isinstance(value, dict):
            raise ValueError("Project file must contain a JSON object.")
        if progress is not None:
            progress(0, 0, "Validating project and owned inputs")
        loader = (
            ProjectAssembly if value.get("schema") in {ASSEMBLY_SCHEMA, *LEGACY_ASSEMBLY_SCHEMAS} else ProjectDocument
        )
        document = loader.from_dict(value, project_root=path.parent)
    if isinstance(document, ProjectAssembly) and not document.channels:
        raise ValueError("This collection has no channels. Add a channel in Python before opening it in the workbench.")
    return document


def prepare_history(project, key, history, embedded_root, *, progress=None):
    """Retain verified embedded evidence even if the optional cache is unreadable."""
    if key[0] != project.id:
        raise ValueError("Prepared history scope differs from its project identity.")
    embedded_root = Path(embedded_root).expanduser().resolve()
    records, ids, embedded_ids = {}, {}, {}
    embedded_entries, entries, diagnostics = [], (), []
    unavailable, session = (), {}
    for index, archive in enumerate(project.saved_runs):
        if progress is not None:
            progress(index, len(project.saved_runs), f"Verifying saved result: {archive.alias}")
        try:
            record = archive.materialize(embedded_root)
            if progress is not None:
                progress(index, len(project.saved_runs), f"Linking verified result: {archive.alias}")
            records[archive.id] = record
            embedded_ids[archive.id] = archive.id
            entry = HistoryEntry(
                archive.id,
                key[0],
                key[1],
                archive.alias,
                record.directory,
                read_run_metadata(record)["request_sha256"],
                record,
            )
            if history is not None:
                try:
                    entry = history.add_run(key[0], record, channel_id=key[1], alias=archive.alias)
                    if progress is not None:
                        progress(index, len(project.saved_runs), f"Restoring saved result alias: {archive.alias}")
                    history.rename_run(entry.id, archive.alias)
                    embedded_ids[archive.id] = entry.id
                    records[entry.id] = record
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    diagnostics.append(f"Local history cache unavailable; embedded evidence is restored: {exc}")
            ids[entry.id] = embedded_ids[archive.id]
            embedded_entries.append(entry)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            message = f"Embedded run {archive.alias!r} could not be restored: {exc}"
            diagnostics.append(message)
            embedded_entries.append(
                HistoryEntry(
                    archive.id,
                    key[0],
                    key[1],
                    archive.alias,
                    embedded_root / archive.id,
                    archive.metadata["request_sha256"],
                    diagnostic=message,
                )
            )
    if history is not None:
        if progress is not None:
            progress(0, 0, "Verifying local result history")
        try:
            entries = history.restore(key[0], channel_id=key[1])
            unavailable = tuple(dict.fromkeys(entry.directory for entry in entries if not entry.verified))
            records = {id_: record for id_, record in records.items() if record.directory not in unavailable}
            for entry in entries:
                if entry.verified:
                    records[entry.id] = entry.run
                    ids[entry.id] = entry.id
            if progress is not None:
                progress(0, 0, "Restoring saved view state")
            session = history.session(key[0], channel_id=key[1])
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            diagnostics.append(f"History could not be restored: {exc}")
    present = {entry.id for entry in entries}
    entries += tuple(entry for entry in embedded_entries if entry.id not in present)
    return PreparedHistory(key, records, entries, ids, embedded_ids, session, tuple(diagnostics), unavailable)


class _Signals(QObject):
    progress = Signal(object, str)
    finished = Signal(object, object, str)


class _Cancelled(Exception):
    pass


class ProjectOpenTask(QRunnable):
    """Only immutable application documents and verified records leave this worker."""

    def __init__(
        self, path, embedded_root, *, history=None, token=None, jobs_directory=None, states=(), records=(), threads=1
    ):
        super().__init__()
        self.signals = _Signals()
        self.path = Path(path).expanduser().resolve()
        self.embedded_root, self.history, self.token = embedded_root, history, token
        self.jobs_directory = jobs_directory
        self.states = tuple(deepcopy(dict(state)) for state in states)
        self.records, self.threads = tuple(records), threads
        self._cancelled = Event()

    def cancel(self):
        self._cancelled.set()

    def _check_cancelled(self):
        if self._cancelled.is_set():
            raise _Cancelled()

    def run(self):
        def progress(_completed, _total, message):
            self._check_cancelled()
            deliver_task_signal(self.signals, "progress", self.token, message)
            self._check_cancelled()

        result, error = None, ""
        try:
            self._check_cancelled()
            document = load_document(self.path, progress=progress)
            self._check_cancelled()
            channel_id = None
            project = document
            if isinstance(document, ProjectAssembly):
                channel_id = next(
                    (
                        channel.id
                        for channel in document.channels
                        if channel.project.session.get("active_channel_id") == channel.id
                    ),
                    document.channels[0].id,
                )
                project = document.channel(channel_id).project
            prepared = prepare_history(
                project,
                (project.id, channel_id),
                self.history,
                self.embedded_root,
                progress=progress,
            )
            self._check_cancelled()
            progress(0, 0, "Checking model readiness")
            readiness = check_readiness(project, pipeline_stage=project.study["pipeline_stage"])
            self._check_cancelled()
            progress(0, 0, "Verifying reusable checkpoints")
            records = {
                Path(record.directory).expanduser().resolve(): record
                for record in (*self.records, *prepared.records.values())
                if record.directory not in prepared.unavailable_directories
            }
            try:
                reuse = execution_reuse(
                    project,
                    tuple(records.values()),
                    self.states,
                    jobs_directory=self.jobs_directory,
                    threads=self.threads,
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reuse = ReusePlan((), (), (str(exc),))
            self._check_cancelled()
            result = PreparedProjectOpen(self.path, document, channel_id, prepared, readiness, reuse)
        except _Cancelled:
            error = "Cancelled"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        deliver_task_signal(self.signals, "finished", self.token, result, error)
