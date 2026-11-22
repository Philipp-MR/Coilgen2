"""Shared fixtures for workbench project lifecycle."""

from hashlib import sha256
import json
from coilgen.application import DesignService
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.storage import result_metadata_digest
from _coilgen_test_support.run_history import write_run


def _owned_run(project, directory):
    """Construct static evidence with real integrity checks, without a solver."""
    record = write_run(directory, revision=project.revision, config=project.values)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    request.update(project_id=project.id, name=project.name, **project.study)
    digest = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    atomic_json(directory / "request.json", request)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(request=request, request_sha256=digest)
    atomic_json(directory / "manifest.json", manifest)
    metadata = record.metadata
    metadata.update(
        name=project.name,
        project_id=project.id,
        project_fingerprint=project.fingerprint,
        project_model_fingerprint=project.model_fingerprint,
        request_sha256=digest,
    )
    for artifact in metadata["artifacts"]:
        artifact["sha256"] = file_digest(directory / artifact["name"])
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(directory / "result.json", metadata)
    return DesignService.load_run(directory)
