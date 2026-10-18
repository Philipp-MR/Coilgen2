"""Coalesced background loading of owned, completed stage display snapshots."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ..application.stage_preview import MANIFEST_NAME, load_stage_preview


class _Signals(QObject):
    finished = Signal(int, object, object, object)


class _Load(QRunnable):
    def __init__(self, epoch, request, loader):
        super().__init__()
        self.epoch, self.request, self.loader = epoch, request, loader
        self.signals = _Signals()

    @Slot()
    def run(self):
        from .design_tasks import deliver_task_signal

        directory, identity, _signature = self.request
        preview, error = None, None
        try:
            preview = self.loader(directory, **identity)
        except Exception as exc:
            # Optional display failures must still release the loader/coalesced
            # request, including resource and unexpected archive errors.
            error = str(exc) or type(exc).__name__
        deliver_task_signal(self.signals, "finished", self.epoch, self.request, preview, error)


class LivePreviewController(QObject):
    """One loader at a time; stale callbacks cannot change the selected plot.

    Only a cheap manifest stat runs on the UI thread. Verification, decompression
    and scientific array loading use a thread; rendering remains on the UI thread.
    Display snapshots never enter run history, project results or export.
    """

    changed = Signal()
    preview_ready = Signal(str, object)

    def __init__(self, jobs_directory, parent=None, *, loader=None):
        super().__init__(parent)
        self._root = Path(jobs_directory).resolve()
        self._loader = loader or load_stage_preview
        self._state = {}
        self._following = False
        self._preview = None
        self._error = None
        self._epoch = 0
        self._seen = None
        self._worker = None
        self._pending = None
        self._closed = False

    @property
    def state(self):
        return deepcopy(self._state)

    @property
    def job_id(self):
        return self._state.get("id")

    @property
    def following(self):
        return self._following

    @property
    def preview(self):
        return self._preview

    @property
    def loading(self):
        return self._worker is not None and self._worker.epoch == self._epoch

    @property
    def working(self):
        """Actual archive loading, including discarded requests still finishing.

        ``loading`` describes the selected live view. A channel change or pause
        invalidates that view without cancelling archive decompression; global
        activity must remain visible until the retained worker has finished.
        """
        return self._worker is not None

    @property
    def error(self):
        return self._error

    def start(self, state):
        if self._closed:
            return
        self._epoch += 1
        self._state = deepcopy(state)
        self._following = True
        self._preview, self._error, self._seen, self._pending = None, None, None, None
        self.changed.emit()
        self.refresh(state)

    def pause(self):
        if self._following:
            self._epoch += 1
            self._following = False
            self._pending = None
            self.changed.emit()

    def resume(self):
        if self._closed or not self._state or self._following:
            return
        self._epoch += 1
        self._following = True
        self._seen, self._pending = None, None
        self.changed.emit()
        self.refresh(self._state)

    def stop(self):
        self._epoch += 1
        self._state, self._pending, self._seen = {}, None, None
        self._following = False
        self._preview, self._error = None, None
        self.changed.emit()

    def close(self):
        self._closed = True
        self.stop()

    def _request(self, state):
        job_id = state.get("id")
        digest = state.get("request_sha256")
        if (
            not isinstance(job_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id) is None
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(state.get("project_id"), str)
            or not state["project_id"].strip()
            or not isinstance(state.get("project_fingerprint"), str)
            or not state["project_fingerprint"].strip()
            or (
                state.get("project_revision") is not None
                and (type(state["project_revision"]) is not int or state["project_revision"] < 0)
            )
        ):
            return None
        directory = (self._root / job_id).resolve()
        if not directory.is_relative_to(self._root):
            return None
        try:
            stat = (directory / MANIFEST_NAME).stat()
        except FileNotFoundError:
            return None
        identity = {
            "expected_request_sha256": digest,
            "expected_project_id": state["project_id"],
            "expected_project_fingerprint": state["project_fingerprint"],
            "expected_project_revision": state.get("project_revision"),
        }
        return directory, identity, (stat.st_mtime_ns, stat.st_size)

    def refresh(self, state):
        if self._closed or not self._state or state.get("id") != self.job_id:
            return
        # Polling may supplement state, but cannot replace the submitted identity.
        fields = ("request_sha256", "project_id", "project_fingerprint", "project_revision")
        if any(state.get(key) != self._state.get(key) for key in fields):
            self.stop()
            return
        self._state = deepcopy(state)
        self.changed.emit()
        if not self._following:
            return
        try:
            request = self._request(state)
        except OSError as exc:
            self._error = str(exc)
            self.changed.emit()
            return
        if request is None or request == self._seen:
            return
        if self._worker is not None:
            self._pending = request
            self.changed.emit()
            return
        self._launch(request)

    def _launch(self, request):
        self._seen = request
        self._worker = _Load(self._epoch, request, self._loader)
        self._worker.signals.finished.connect(self._finished)
        QThreadPool.globalInstance().start(self._worker)
        self.changed.emit()

    @Slot(int, object, object, object)
    def _finished(self, epoch, request, preview, error):
        # Retain even an invalidated worker until its completion so at most one
        # load exists. A new generation waits/coalesces rather than racing it.
        self._worker = None
        pending, self._pending = self._pending, None
        if not self._closed and epoch == self._epoch and self._following:
            self._error = error
            if preview is not None and error is None:
                self._preview = preview
                self.preview_ready.emit(self.job_id, preview)
        if not self._closed and self._following and pending is not None:
            if epoch != self._epoch or pending != request:
                if self._worker is None:
                    # A ready handler may change projects/start a new load.
                    try:
                        current = self._request(self._state)
                    except OSError:
                        current = None
                    if current == pending:
                        self._launch(pending)
        self.changed.emit()
