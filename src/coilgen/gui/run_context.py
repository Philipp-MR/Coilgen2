"""Presentation of recorded run identity and dates, without inferred evidence."""

from .job_history import format_job_timestamp
from ..application.execution import run_compatibility as run_compatibility


def run_timestamp(metadata):
    """Use actual submission time; importing or linking a run is not execution."""
    return format_job_timestamp((metadata.get("job") or {}).get("created_at"))
