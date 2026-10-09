"""Verification-heavy design actions over captured immutable project inputs."""

from PySide6.QtCore import QObject, QRunnable, Signal
from ..progress import ProgressRelay as ProgressSignalRelay


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

        relay = ProgressSignalRelay(self.signals.progress.emit)
        try:
            with progress_context(relay.report):
                result = self.operation(self.project, *self.args, **self.kwargs)
        except Exception as exc:
            relay.flush()
            self.signals.finished.emit(None, f"{type(exc).__name__}: {exc}")
        else:
            relay.flush()
            self.signals.finished.emit(result, "")
