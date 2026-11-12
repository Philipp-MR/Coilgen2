"""Watch a running job's stage preview manifest; shared by the desktop and browser.

Only a cheap ``stat`` of ``stage_preview.json`` runs on the caller's thread.
Callers load the announced snapshot elsewhere (a thread) with
:func:`load_requested_preview`. Previews are display-only and never final.
"""

from __future__ import annotations

from pathlib import Path
import re

from .evidence import resolve_dataset
from .scene_spec import PREFERRED_RUN_PLOTS
from .stage_preview import MANIFEST_NAME, load_stage_preview


# Plots first produced by each routing/evaluation feature preview.
PREVIEW_FEATURE_PLOTS = {
    "interconnections": ("routes",),
    "realized_centerlines": ("fabrication",),
    "sweep": ("swept",),
    "evaluated_winding": ("routed_field", "contour_field", "evaluated_winding"),
    "independent_field": ("validation",),
    "review_conductor": ("review_conductor", "review_electrical"),
    "fabricated_field": ("fabricated",),
    "winding_validation": ("winding_validation",),
}


def stage_preview_request(jobs_root, state):
    """``(directory, identity, signature)`` for a job's current manifest, or None.

    The signature is the manifest's (mtime, size); a change means a newer stage
    was announced. Raises ``OSError`` only for unexpected filesystem failures.
    """
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
    root = Path(jobs_root).resolve()
    directory = (root / job_id).resolve()
    if not directory.is_relative_to(root):
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


def load_requested_preview(request, loader=load_stage_preview):
    """Verify and load the snapshot named by a request (slow; call off the UI thread)."""
    directory, identity, _signature = request
    return loader(directory, **identity)


class StagePreviewWatch:
    """Remembers the last seen manifest so polling only reports new stages."""

    def __init__(self, jobs_directory):
        self.root = Path(jobs_directory).resolve()
        self.seen = None

    def reset(self):
        self.seen = None

    def changed(self, state):
        """The new request when the job's manifest changed since the last call."""
        request = stage_preview_request(self.root, state)
        if request is None or request == self.seen:
            return None
        self.seen = request
        return request


def preferred_preview_dataset(arrays, metadata):
    """The newest stage's plot: the feature's own plot, then the usual run order."""
    feature = (metadata or {}).get("preview_feature")
    for key in PREVIEW_FEATURE_PLOTS.get(feature, ()) + PREFERRED_RUN_PLOTS:
        if resolve_dataset(arrays, key).available:
            return key
    return None
