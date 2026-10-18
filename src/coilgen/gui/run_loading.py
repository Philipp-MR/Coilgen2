"""Verify finished worker evidence without blocking the Qt event loop."""

from PySide6.QtCore import QObject, QRunnable, Signal


class _Signals(QObject):
    finished = Signal(str, object, str)

    def deliver(self, id_, result, error):
        try:
            self.finished.emit(id_, result, error)
        except RuntimeError as exc:
            # Qt may destroy signal sources during application shutdown while
            # an archive operation is still finishing in the thread pool.
            if "deleted" not in str(exc):
                raise


class CompletedRunLoadTask(QRunnable):
    """Only application records and file operations cross into this worker."""

    def __init__(self, state, directory, loader, *, assembly_id, scopes, history=None, reuse_inputs=None):
        super().__init__()
        self.signals = _Signals()
        self.state = dict(state)
        self.directory, self.loader = directory, loader
        self.assembly_id, self.scopes, self.history = assembly_id, dict(scopes), history
        self.reuse_inputs = reuse_inputs

    def run(self):
        result, error = None, ""
        try:
            record = self.loader(self.directory)
            metadata = record.metadata
            for key in ("request_sha256", "project_id", "project_revision", "project_fingerprint"):
                expected = self.state.get(key)
                if expected is not None and metadata.get(key) != expected:
                    raise ValueError(f"Completed result {key} differs from its submitted job.")
            scope = self.scopes.get(metadata.get("project_id"))
            entry, entries, history_error = None, (), ""
            if self.history is not None and scope is not None:
                try:
                    entry = self.history.add_run(scope[0], record, channel_id=scope[1])
                    entries = self.history.restore(scope[0], channel_id=scope[1])
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    history_error = str(exc)
            reuse = None
            if self.reuse_inputs is not None:
                from ..application.execution import execution_reuse
                from ..application.reuse import ReusePlan, plan_reuse

                project, records, states, jobs_directory, threads = self.reuse_inputs
                try:
                    all_reuse = execution_reuse(
                        project, (*records, record), states, jobs_directory=jobs_directory, threads=threads
                    )
                    selected_reuse = plan_reuse(project, (record,), threads=threads)
                    reuse = (all_reuse, selected_reuse)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    reuse = (ReusePlan((), (), (str(exc),)),) * 2
            result = (record, (scope, entry, entries, history_error), reuse)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.signals.deliver(self.state["id"], result, error)


class StoppedReuseTask(QRunnable):
    """Verify stopped checkpoints without reading evidence on the GUI thread."""

    def __init__(self, project, records, states, directory, threads, selected_state, key):
        super().__init__()
        self.signals = _Signals()
        self.project, self.records = project, tuple(records)
        self.states = tuple(dict(state) for state in states)
        self.directory, self.threads = directory, threads
        self.selected_state, self.key = dict(selected_state), key

    def run(self):
        from ..application.execution import execution_reuse
        from ..application.reuse import plan_stopped_reuse

        result, error = None, ""
        try:
            result = (
                execution_reuse(
                    self.project, self.records, self.states, jobs_directory=self.directory, threads=self.threads
                ),
                plan_stopped_reuse(self.project, (self.selected_state,), self.directory, threads=self.threads),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.signals.deliver(self.selected_state["id"], result, error)
