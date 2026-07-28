"""Centralized data and output paths for the Halbach gradient project."""

from __future__ import annotations

from pathlib import Path

from projects._shared.paths import project_data_root, project_output_dir, project_output_root, resolve_project_path


PROJECT_NAME = "halbach_cylindrical_3axis_gradient"
HALBACH_PROJECT_ROOT = Path(__file__).resolve().parent
HALBACH_DATA_ROOT = project_data_root(PROJECT_NAME)
HALBACH_OUTPUT_ROOT = project_output_root(PROJECT_NAME)


def halbach_output_dir(name: str | None = None, *, create: bool = False) -> Path:
    """Return the main output root or a named variant below it.

    Paths are lazy by default so importing a run/configuration module does not
    leave behind an empty timestamped run directory.  The solve and reporting
    layers create their output directories immediately before writing.
    """

    if name is None or name == PROJECT_NAME:
        return project_output_root(PROJECT_NAME, create=create)
    return project_output_dir(PROJECT_NAME, name, create=create)


def resolve_halbach_path(path: str | Path) -> Path:
    """Resolve Halbach inputs and generated outputs through the shared roots."""

    return resolve_project_path(HALBACH_PROJECT_ROOT, PROJECT_NAME, path)
