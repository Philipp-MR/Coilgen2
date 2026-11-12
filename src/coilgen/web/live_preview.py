"""Live stage previews of the followed job in the browser workbench.

The server poll stats the job's ``stage_preview.json``; a changed manifest is
verified and loaded in a thread. ``_render`` shows the preview arrays while no
completed run is open. Previews are display-only and never final results.
"""

from __future__ import annotations

import asyncio

from ..application.live_preview import StagePreviewWatch, load_requested_preview, preferred_preview_dataset
from ..application.stage_preview import load_stage_preview

MODEL_DATASETS = frozenset({"support", "preview_target"})


class BrowserLivePreview:
    """Holds the latest loaded preview of ``browser.follow_job_id``."""

    def __init__(self, browser, *, loader=load_stage_preview):
        self.browser = browser
        self.watch = StagePreviewWatch(browser.session.jobs_directory)
        self.loader = loader
        self.job_id = None
        self.preview = None
        self.error = ""
        self.auto_dataset = None

    def current(self, follow_job_id):
        """The preview to draw, only while its job is still followed."""
        if follow_job_id is not None and follow_job_id == self.job_id:
            return self.preview
        return None

    async def poll(self):
        """Called by the server poll: cheap stat, then an off-thread load on change."""
        browser = self.browser
        job_id = browser.follow_job_id
        if job_id != self.job_id:
            self._clear()
            self.job_id = job_id
        state = browser.session.states.get(job_id) if job_id is not None else None
        if not state:
            return
        try:
            request = self.watch.changed(state)
        except OSError as exc:
            self.error = str(exc)
            return
        if request is None:
            return
        try:
            preview = await asyncio.to_thread(load_requested_preview, request, self.loader)
        except Exception as exc:  # A preview is optional; never stop the poll.
            self.error = str(exc) or type(exc).__name__
            return
        if preview is None or browser.follow_job_id != job_id or self.job_id != job_id:
            return
        self.preview, self.error = preview, ""
        with browser.server.state:
            self._show()

    def _show(self):
        browser, state = self.browser, self.browser.server.state
        if browser.record is not None:
            return  # A completed run stays on screen; the preview waits.
        preferred = preferred_preview_dataset(self.preview.arrays, self.preview.metadata)
        if preferred is not None and preferred != self.auto_dataset:
            # Follow the newest stage once; a later manual choice is kept until the next stage.
            self.auto_dataset = preferred
            if state.dataset != preferred:
                state.dataset = preferred  # The dataset watcher renders.
                return
        browser._render()

    def _clear(self):
        shown = self.preview is not None
        self.preview, self.error, self.auto_dataset = None, "", None
        self.watch.reset()
        browser, state = self.browser, self.browser.server.state
        if shown and browser.record is None and state.selected_run is None and state.dataset not in MODEL_DATASETS:
            with state:
                state.dataset = "support"
