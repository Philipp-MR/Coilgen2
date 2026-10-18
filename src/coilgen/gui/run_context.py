"""Presentation of recorded run identity and dates, without inferred evidence."""

from .job_history import format_job_timestamp
from ..application.execution import run_compatibility as run_compatibility
from .run_metadata import read_run_metadata


def run_timestamp(source):
    """Use actual submission time; importing or linking a run is not execution."""
    metadata = read_run_metadata(source)
    job = metadata.get("job") or {}
    # Legacy records may omit these identities. Explicitly contradictory
    # bindings never supply a date for this result.
    matches = all(
        not job.get(key) or not metadata.get(key) or job[key] == metadata[key]
        for key in ("project_id", "request_sha256")
    )
    value = job.get("created_at") if matches else None
    if value is None:
        value = getattr(source, "execution_times", {}).get("created_at")
    return format_job_timestamp(value)
