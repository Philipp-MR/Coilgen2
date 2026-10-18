"""Read immutable evidence for presentation without copying wire witnesses."""

from collections.abc import Mapping


def read_run_metadata(record):
    """Borrow an immutable application view; retain legacy adapter support."""
    if record is None:
        return {}
    if isinstance(record, Mapping):
        return record
    view = getattr(record, "metadata_view", None)
    return view if view is not None else getattr(record, "metadata", {})
