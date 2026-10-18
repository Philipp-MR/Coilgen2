"""Verification-heavy design actions over captured immutable project inputs."""

from PySide6.QtCore import QObject, QRunnable, Signal
from shiboken6 import isValid
from ..progress import ProgressRelay as ProgressSignalRelay


def deliver_task_signal(signals, name, *values):
    """Allow pending workers to finish after Qt destroys their signal source."""
    try:
        getattr(signals, name).emit(*values)
    except RuntimeError as exc:
        if "deleted" not in str(exc) or isValid(signals):
            raise


class _Signals(QObject):
    progress = Signal(dict)
    finished = Signal(object, str)


class DesignStudyTask(QRunnable):
    def __init__(self, operation, project, *args, **kwargs):
        super().__init__()
        self.signals = _Signals()
        self.project = project
        self.operation, self.args, self.kwargs = operation, args, kwargs

    def run(self):
        from ..progress import progress_context

        relay = ProgressSignalRelay(lambda event: deliver_task_signal(self.signals, "progress", event))
        try:
            with progress_context(relay.report):
                result = self.operation(self.project, *self.args, **self.kwargs)
        except Exception as exc:
            relay.flush()
            deliver_task_signal(self.signals, "finished", None, f"{type(exc).__name__}: {exc}")
        else:
            relay.flush()
            deliver_task_signal(self.signals, "finished", result, "")
