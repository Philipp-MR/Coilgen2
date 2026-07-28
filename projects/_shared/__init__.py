"""Shared project helpers used by runnable CoilGen study projects."""

from .object_parameters import ParameterNamespace, object_parameters, parameters_to_dict
from .paths import (
    OUTPUT_ROOT,
    REPOSITORY_ROOT,
    begin_project_run,
    latest_project_output,
    project_output_base,
    project_output_dir,
    project_output_root,
    resolve_project_path,
)

__all__ = [
    "OUTPUT_ROOT",
    "ParameterNamespace",
    "REPOSITORY_ROOT",
    "begin_project_run",
    "latest_project_output",
    "object_parameters",
    "parameters_to_dict",
    "project_output_base",
    "project_output_dir",
    "project_output_root",
    "resolve_project_path",
]
