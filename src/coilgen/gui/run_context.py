"""Presentation of recorded run identity and dates, without inferred evidence."""

from .job_history import format_job_timestamp


def run_compatibility(metadata, project=None):
    """A matching full fingerprint includes both model inputs and study settings."""
    if project is None:
        return "unknown", "Identity not checked"
    job = metadata.get("job") or {}
    owner = metadata.get("project_id") or job.get("project_id")
    fingerprint = metadata.get("project_fingerprint") or job.get("project_fingerprint")
    if owner and owner != project.id:
        return "other", "Other channel"
    if not owner or not fingerprint:
        return "unknown", "Identity unverified"
    if fingerprint != project.fingerprint:
        return "earlier", "Earlier model/study"
    return "current", "Matches model/study"


def run_timestamp(metadata):
    """Use actual submission time; importing or linking a run is not execution."""
    return format_job_timestamp((metadata.get("job") or {}).get("created_at"))
