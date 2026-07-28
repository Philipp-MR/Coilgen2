"""Paths for the general examples project."""

from __future__ import annotations

from pathlib import Path

from projects._shared.paths import project_data_root, project_output_dir, project_output_root


GENERAL_EXAMPLES_PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERAL_EXAMPLES_DATA_ROOT = project_data_root("general_examples")
GENERAL_EXAMPLES_OUTPUT_ROOT = project_output_root("general_examples")


def general_output_dir(name: str) -> Path:
    return project_output_dir("general_examples", name)
