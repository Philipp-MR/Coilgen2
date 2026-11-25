"""Result workflows for the native workbench.

The window retains Qt widgets and signal/undo adapters; this controller
coordinates application operations without owning a second model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from PySide6.QtCore import QThreadPool
from ..research.storage import result_metadata_digest
from .run_loading import CompletedRunLoadTask
from .run_metadata import read_run_metadata


class ResultController:
    def __init__(self, window):
        self.window = window

    def _start_completed_run_load(self, state):
        window = self.window
        id_ = state["id"]
        if window._run_load_tasks or window.assembly is None or window._closing:
            return
        scopes = {
            channel.project.id: (channel.project.id, channel.id if window._collection_mode else None)
            for channel in window.assembly.channels
        }
        # Jobs from a replaced project stay in the runtime; loading them must
        # not populate the new project's history or change its selected view.
        if state.get("project_id") not in scopes:
            return
        task = CompletedRunLoadTask(
            state,
            window.jobs.directory / id_,
            window.service.load_run,
            assembly_id=window.assembly.id,
            scopes=scopes,
            history=window.history,
            reuse_inputs=(
                window.project,
                tuple(record for record in window.runs.values() if not window._run_verification_error(record)),
                tuple(state for id_, state in window.states.items() if not window._run_verification_error(id_)),
                window.jobs_directory,
                getattr(getattr(window.jobs, "policy", None), "threads_per_worker", 1),
            ),
        )
        window._run_load_tasks[id_] = task
        task.history_alias_generation = window._history_alias_generation
        task.reuse_generation = window._reuse_generation
        task.verification_generation = window._source_verification_generations.get(task.directory.resolve(), 0)
        window._loading_job_id = id_
        task.signals.finished.connect(window._completed_run_loaded)
        QThreadPool.globalInstance().start(task)

    def _completed_run_loaded(self, id_, result, error):
        window = self.window
        task = window._run_load_tasks.pop(id_, None)
        window._loading_job_id = next(iter(window._run_load_tasks), None)
        if task is None or window._closing or window.assembly is None or window.assembly.id != task.assembly_id:
            return
        owner = task.state.get("project_id")
        if not any(channel.project.id == owner for channel in window.assembly.channels):
            return
        window._loaded_ids.add(id_)
        if window._source_verification_generations.get(task.directory.resolve(), 0) != task.verification_generation:
            # A later explicit check of these files outranks this older snapshot.
            window._record_job_outcomes((window.states.get(id_, task.state),))
            window._update_execution_feedback()
            return
        if error:
            window._mark_run_verification_error(id_, error, directory=task.directory)
            window._error(f"Completed run could not be verified: {error}", open_window=False)
        else:
            record, history_result, reuse = result
            project = task.reuse_inputs[0]
            selected_reuse = None
            if window.project.id == project.id and window.project.fingerprint == project.fingerprint:
                all_reuse, selected_reuse = reuse
                if window._reuse_generation == task.reuse_generation:
                    window._reuse_plan = all_reuse
                    window._reuse_plan_fingerprint = project.fingerprint
            try:
                window._register_loaded_run(
                    record,
                    id_=id_,
                    select=None,
                    history_result=history_result,
                    selected_reuse=selected_reuse,
                    history_alias_generation=task.history_alias_generation,
                )
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                window._mark_run_verification_error(id_, exc, directory=task.directory)
                window._error(f"Completed run could not be registered: {exc}", open_window=False)
        window._record_job_outcomes((window.states.get(id_, task.state),))
        window._refresh_stage_status(refresh_reuse=False)
        window._update_execution_feedback()

    def load_run(self, path, *, id_=None, select=True, expected_record=None):
        window = self.window
        with window._visible_work("Verifying saved result · " + Path(path).name):
            return window._load_run(path, id_=id_, select=select, expected_record=expected_record)

    def _load_run(self, path, *, id_=None, select=True, expected_record=None):
        window = self.window
        try:
            record = window.service.load_run(path)
            if expected_record is not None:
                old = {key: value for key, value in read_run_metadata(expected_record).items() if key != "job"}
                current = {key: value for key, value in read_run_metadata(record).items() if key != "job"}
                if result_metadata_digest(old) != result_metadata_digest(current):
                    raise ValueError("Saved run metadata changed after selection. Reopen and verify the run.")
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            window._mark_run_verification_error(id_ or str(Path(path).expanduser().resolve()), exc, directory=path)
            raise
        return window._register_loaded_run(record, id_=id_, select=select)

    def _register_loaded_run(
        self, record, *, id_=None, select=True, history_result=None, selected_reuse=None, history_alias_generation=None
    ):
        """Register an already verified immutable record on the GUI thread."""
        window = self.window
        id_ = id_ or str(record.directory)
        window._clear_run_verification_error(record, id_=id_)
        window.runs[id_] = record
        owner = read_run_metadata(record).get("project_id")
        channels = window.assembly.channels if window.assembly is not None else ()
        channel = next((channel for channel in channels if channel.project.id == owner), None)
        scope = (owner, channel.id if window._collection_mode else None) if channel is not None else None
        if history_result is not None:
            loaded_scope, entry, entries, diagnostic = history_result
            if loaded_scope == scope and entry is not None:
                window._history_ids[entry.id] = id_
                if scope == window._history_key:
                    if history_alias_generation is not None:
                        entries = tuple(
                            replace(item, alias=window._history_alias_edits[item.id][1])
                            if item.id in window._history_alias_edits
                            and window._history_alias_edits[item.id][0] > history_alias_generation
                            else item
                            for item in entries
                        )
                    window._history_entries = entries
                    window._refresh_history_panel()
            if diagnostic:
                window.messages.appendPlainText(f"Run history link could not be saved: {diagnostic}")
        elif window.history is not None and scope is not None:
            try:
                entry = window.history.add_run(scope[0], record, channel_id=scope[1])
                window._history_ids[entry.id] = id_
                if scope == window._history_key:
                    window._history_entries = window.history.restore(scope[0], channel_id=scope[1])
                    window._refresh_history_panel()
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                window.messages.appendPlainText(f"Run history link could not be saved: {exc}")
        follow_latest = select is None
        if follow_latest:
            followed = window.live_preview.state
            select = (
                window.live_preview.following
                and window.live_preview.job_id == id_
                and window._live_owner_matches(followed)
                and window._live_owner_matches(read_run_metadata(record))
            )
        window._fill_runs()
        if select:
            if selected_reuse is not None:
                # Capability display uses the exact plan verified by the loader;
                # choosing Resume still re-verifies mutable files before use.
                threads = getattr(getattr(window.jobs, "policy", None), "threads_per_worker", 1)
                window._resume_capability_key = (
                    window.project.fingerprint,
                    window.stage_selector.currentData(),
                    threads,
                    id_,
                    id(record),
                    None,
                    (),
                    window._run_verification_error(id_),
                )
                window._resume_capability_plan = selected_reuse
            window.show_run(id_, follow_latest=follow_latest)
        window._refresh_stage_status(refresh_reuse=history_result is None)
        if select:
            window._save_session()
        return record
