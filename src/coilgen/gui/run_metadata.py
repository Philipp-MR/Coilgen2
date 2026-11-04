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


def has_saved_request(record):
    """Presence only. The full request is reverified when the reader opens."""
    artifacts = read_run_metadata(record).get("artifacts") or ()
    matches = [item for item in artifacts if isinstance(item, Mapping) and item.get("name") == "request.json"]
    if len(matches) != 1:
        return False
    digest = matches[0].get("sha256")
    return isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower())
