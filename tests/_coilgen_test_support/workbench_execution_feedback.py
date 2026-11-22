"""Shared fixtures for workbench execution feedback."""

from coilgen.application.service import RunRecord


def _record(window, directory):
    return RunRecord(
        directory,
        {
            "name": window.project.name,
            "project_id": window.project.id,
            "project_revision": window.project.revision,
            "project_fingerprint": window.project.fingerprint,
            "project_model_fingerprint": window.project.model_fingerprint,
            "summary": {"scope": "continuous_sheet", "accepted": False, "sheet_accepted": False},
            "constraints": [],
            "wire_fields": {},
            "fabrication": None,
            "warnings": [],
            "artifacts": [],
        },
        {},
    )
