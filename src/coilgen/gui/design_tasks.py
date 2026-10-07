"""Verification-heavy design actions over captured immutable project inputs."""

from PySide6.QtCore import QObject, QRunnable, Signal


class _Signals(QObject):
    finished = Signal(object, str)


class DesignStudyTask(QRunnable):
    def __init__(self, operation, project, *args, **kwargs):
        super().__init__()
        self.signals = _Signals()
        self.project = project
        self.operation, self.args, self.kwargs = operation, args, kwargs

    def run(self):
        try:
            result = self.operation(self.project, *self.args, **self.kwargs)
        except Exception as exc:
            self.signals.finished.emit(None, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(result, "")
