"""Project-linked run aliases and session state, separate from scientific evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

from ..persistence import atomic_json, json_value
from ..research.storage import DirectoryOwner
from .service import DesignService, RunRecord


HISTORY_SCHEMA = "coilgen-run-history-v1"


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    id: str
    project_id: str
    channel_id: str | None
    alias: str
    directory: Path
    request_sha256: str
    run: RunRecord | None = None
    diagnostic: str = ""

    @property
    def label(self) -> str:
        return self.alias or (self.run.metadata.get("name", self.directory.name) if self.run else self.directory.name)

    @property
    def verified(self) -> bool:
        return self.run is not None and not self.diagnostic


def _scope(project_id, channel_id):
    if not isinstance(project_id, str) or not project_id.strip() or len(project_id) > 128:
        raise ValueError("History project_id must be a nonempty stable identifier.")
    if channel_id is not None and (not isinstance(channel_id, str) or not channel_id.strip() or len(channel_id) > 128):
        raise ValueError("History channel_id must be a stable identifier or None.")
    return json.dumps([project_id, channel_id], separators=(",", ":"))


def _alias(value):
    if not isinstance(value, str) or len(value) > 120:
        raise ValueError("A run display alias must be a string of at most 120 characters.")
    return value.strip()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate history JSON key: {key}.")
        result[key] = value
    return result


class RunHistoryStore:
    """Atomic session manifest with stable project/channel filtering.

    Relative run paths preserve directory relationships when a workspace moves.
    Results, request identities and acceptance metadata are never edited. Each
    restoration rechecks artifacts and original inputs through DesignService.
    Concurrent session edits reload the manifest under one directory lock.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._read()

    def _read(self):
        if not self.path.exists():
            return {"schema": HISTORY_SCHEMA, "records": [], "sessions": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            json_value(data, nonfinite="raise")
            if not isinstance(data, dict) or data.get("schema") != HISTORY_SCHEMA:
                raise ValueError("Unsupported run-history schema.")
            if not isinstance(data.get("records"), list) or not isinstance(data.get("sessions"), dict):
                raise ValueError("Run-history records/sessions have invalid types.")
            ids = set()
            for row in data["records"]:
                if not isinstance(row, dict):
                    raise ValueError("Run-history links must be JSON objects.")
                _scope(row["project_id"], row["channel_id"])
                _alias(row["alias"])
                if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
                    raise ValueError("Run-history record IDs must be unique nonempty strings.")
                ids.add(row["id"])
                if not isinstance(row["path"], str) or not row["path"] or not isinstance(row["request_sha256"], str):
                    raise ValueError("Run-history path/identity is invalid.")
            for key, value in data["sessions"].items():
                scope = json.loads(key)
                if not isinstance(scope, list) or len(scope) != 2 or _scope(*scope) != key:
                    raise ValueError("Run-history session scope is invalid.")
                if not isinstance(value, dict) or not isinstance(value.get("view_state", {}), dict):
                    raise ValueError("Run-history session view state is invalid.")
            return data
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot read run history {self.path}: {exc}") from exc

    def _mutate(self, change):
        with DirectoryOwner(self.path.parent, filename=f".{self.path.name}.lock"):
            data = self._read()
            result = change(data)
            json_value(data, nonfinite="raise")
            atomic_json(self.path, data)
            return result

    def _directory(self, row):
        path = Path(row["path"]).expanduser()
        return (path if path.is_absolute() else self.path.parent / path).resolve()

    def _entry(self, row, *, verify=True):
        directory = self.path.parent / row["path"]
        run, diagnostic = None, ""
        if verify:
            try:
                directory = self._directory(row)
                run = DesignService.load_run(directory)
                if run.metadata["request_sha256"] != row["request_sha256"]:
                    raise ValueError("The linked directory now contains a different request identity.")
                source = row.get("run_project_id", "")
                if source and run.metadata.get("project_id") != source:
                    raise ValueError("The linked result's original project identity changed.")
            except (
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                AttributeError,
                EOFError,
                BadZipFile,
            ) as exc:
                run = None
                diagnostic = f"{row['alias'] or directory.name}: {exc}"
        return HistoryEntry(
            row["id"],
            row["project_id"],
            row["channel_id"],
            row["alias"],
            directory,
            row["request_sha256"],
            run,
            diagnostic,
        )

    def add_run(self, project_id: str, run: RunRecord | str | Path, *, channel_id=None, alias=None) -> HistoryEntry:
        """Link verified evidence; aliases never rename the underlying run."""
        _scope(project_id, channel_id)
        record = DesignService.load_run(run.directory if isinstance(run, RunRecord) else run)
        if self.path.is_relative_to(record.directory):
            raise ValueError("Run history must be stored outside immutable run output directories.")
        alias = _alias(record.metadata.get("name", record.directory.name) if alias is None else alias)
        try:
            path = Path(os.path.relpath(record.directory, self.path.parent)).as_posix()
        except ValueError:  # Different Windows volume.
            path = record.directory.as_posix()

        def add(data):
            for row in data["records"]:
                if (row["project_id"], row["channel_id"], self._directory(row)) == (
                    project_id,
                    channel_id,
                    record.directory,
                ):
                    if row["request_sha256"] != record.metadata["request_sha256"]:
                        raise ValueError(
                            "This historical link has a different request identity; choose another run directory."
                        )
                    return deepcopy(row)
            row = {
                "id": uuid4().hex,
                "project_id": project_id,
                "channel_id": channel_id,
                "alias": alias,
                "path": path,
                "request_sha256": record.metadata["request_sha256"],
                "run_project_id": record.metadata.get("project_id", ""),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            data["records"].append(row)
            return deepcopy(row)

        row = self._mutate(add)
        return HistoryEntry(
            row["id"], project_id, channel_id, row["alias"], record.directory, row["request_sha256"], record
        )

    def restore(self, project_id: str, *, channel_id=None) -> tuple[HistoryEntry, ...]:
        """Retain unavailable links as visible rows with diagnostics."""
        _scope(project_id, channel_id)
        return tuple(
            self._entry(row)
            for row in self._read()["records"]
            if row["project_id"] == project_id and row["channel_id"] == channel_id
        )

    records = restore

    def link_scope(self, project_id: str, *, source_channel_id=None, target_channel_id) -> dict[str, str]:
        """Copy links/session into another channel scope without moving evidence.

        Used when a standalone project becomes an independent-channel assembly.
        Unavailable links and pinned request hashes remain intact. Repeating the
        operation reuses existing target links and preserves target aliases and
        session choices. The returned mapping translates saved selection IDs.
        """
        source_key = _scope(project_id, source_channel_id)
        target_key = _scope(project_id, target_channel_id)

        def link(data):
            source = [
                row
                for row in data["records"]
                if row["project_id"] == project_id and row["channel_id"] == source_channel_id
            ]
            if source_key == target_key:
                return {row["id"]: row["id"] for row in source}
            target = {}
            for row in data["records"]:
                if row["project_id"] == project_id and row["channel_id"] == target_channel_id:
                    target.setdefault((row["path"], row["request_sha256"]), row)
            mapping = {}
            for row in source:
                identity = (row["path"], row["request_sha256"])
                copied = target.get(identity)
                if copied is None:
                    copied = deepcopy(row)
                    copied.update(id=uuid4().hex, channel_id=target_channel_id)
                    data["records"].append(copied)
                    target[identity] = copied
                mapping[row["id"]] = copied["id"]
            if target_key not in data["sessions"] and source_key in data["sessions"]:
                session = deepcopy(data["sessions"][source_key])
                selected = session.get("selected_id")
                session["selected_id"] = mapping.get(selected) if isinstance(selected, str) else None
                data["sessions"][target_key] = session
            return mapping

        return self._mutate(link)

    def rename_run(self, record_id: str, alias: str) -> None:
        alias = _alias(alias)

        def rename(data):
            for row in data["records"]:
                if row["id"] == record_id:
                    row["alias"] = alias
                    return
            raise ValueError("Unknown run-history record.")

        self._mutate(rename)

    def save_session(self, project_id: str, *, selected_id=None, view_state=None, channel_id=None) -> None:
        key = _scope(project_id, channel_id)
        if view_state is not None and not isinstance(view_state, dict):
            raise ValueError("Session view state must be a JSON mapping.")
        view = None if view_state is None else json_value(deepcopy(view_state), nonfinite="raise")

        def save(data):
            if selected_id is not None and not any(
                row["id"] == selected_id and row["project_id"] == project_id and row["channel_id"] == channel_id
                for row in data["records"]
            ):
                raise ValueError("Selected history record does not belong to this project/channel.")
            previous = data["sessions"].get(key, {})
            data["sessions"][key] = {
                "selected_id": selected_id,
                "view_state": previous.get("view_state", {}) if view is None else view,
            }

        self._mutate(save)

    def session(self, project_id: str, *, channel_id=None) -> dict:
        key = _scope(project_id, channel_id)
        return deepcopy(self._read()["sessions"].get(key, {"selected_id": None, "view_state": {}}))
