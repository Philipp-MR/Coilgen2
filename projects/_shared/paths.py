"""Repository paths shared by runnable engineering projects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path, PureWindowsPath
from types import FrameType
import sys
from threading import RLock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
PROJECT_DATA_ROOT = REPOSITORY_ROOT / "project_data"
RUN_DIRECTORY_FORMAT = "%Y-%m-%d_%H-%M-%S-%f_%z"


@dataclass(slots=True)
class _ProjectRun:
    root: Path
    execution_frame: FrameType | None


_PROJECT_RUNS: dict[str, _ProjectRun] = {}
_PROJECT_RUN_LOCK = RLock()


def project_output_base(project_name: str, *, create: bool = False) -> Path:
    """Return the non-run-specific ``outputs/<project_name>`` directory."""

    project = _project_component(project_name, root_label="output root")
    path = OUTPUT_ROOT / project
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def begin_project_run(project_name: str, *, create: bool = True) -> Path:
    """Start and return a fresh timestamped output root for one project run.

    Use this explicitly when several runs are launched programmatically from one
    long-lived Python session. Normal command-line and Spyder script executions
    are detected automatically by :func:`project_output_root`.
    """

    project = _project_name(project_name, root_label="output root")
    with _PROJECT_RUN_LOCK:
        root = _new_run_root(project)
        _PROJECT_RUNS[project] = _ProjectRun(root=root, execution_frame=_project_execution_frame(project))
        if create:
            root.mkdir(parents=True, exist_ok=False)
        return root


def project_output_root(project_name: str, *, create: bool = False) -> Path:
    """Return this execution's ``outputs/<project>/<date_time>`` directory."""

    project = _project_name(project_name, root_label="output root")
    execution_frame = _project_execution_frame(project)
    with _PROJECT_RUN_LOCK:
        run = _PROJECT_RUNS.get(project)
        if run is None or (execution_frame is not None and execution_frame is not run.execution_frame):
            root = _new_run_root(project)
            run = _ProjectRun(root=root, execution_frame=execution_frame)
            _PROJECT_RUNS[project] = run
        if create:
            run.root.mkdir(parents=True, exist_ok=True)
        return run.root


def project_data_root(project_name: str, *, create: bool = False) -> Path:
    """Return ``project_data/<project_name>`` without allowing path traversal."""

    return project_data_dir(project_name, create=create)


def project_data_dir(project_name: str, *parts: str | Path, create: bool = False) -> Path:
    """Return a project-owned path below the repository data root.

    Input data is read-only from the perspective of project runs, so directories
    are not created unless a caller explicitly requests it.
    """

    project = _project_component(project_name, root_label="project-data root")
    path = PROJECT_DATA_ROOT / project
    for part in parts:
        path /= _safe_relative_path(part, label="data path", root_label="project-data root")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def project_output_dir(project_name: str, *parts: str | Path, create: bool = True) -> Path:
    """Return a directory below the active timestamped project-run root."""

    path = project_output_root(project_name, create=create and not parts)
    for part in parts:
        path /= _safe_relative_path(part, label="output path", root_label="output root")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def latest_project_output(project_name: str, *parts: str | Path) -> Path:
    """Return the newest prior-run artifact, with legacy-layout compatibility.

    Only paths that actually exist are selected. If no prior artifact exists,
    the returned path points into the active run so callers can report a useful
    missing path or deliberately generate a replacement without writing into an
    older run.
    """

    project = _project_name(project_name, root_label="output root")
    relative = _joined_safe_parts(parts, label="output path", root_label="output root")
    active_root = project_output_root(project, create=False)
    base = project_output_base(project, create=False)
    if base.is_dir():
        run_roots = sorted(
            (
                candidate
                for candidate in base.iterdir()
                if candidate.is_dir() and _is_run_directory(candidate.name) and candidate != active_root
            ),
            key=lambda candidate: _run_directory_sort_key(candidate.name),
            reverse=True,
        )
        for run_root in run_roots:
            candidate = run_root / relative if relative.parts else run_root
            if candidate.exists():
                return candidate
        legacy = base / relative if relative.parts else base
        if legacy.exists():
            return legacy
    return active_root / relative if relative.parts else active_root


def resolve_project_path(project_root: Path, project_name: str, path: str | Path) -> Path:
    """Resolve centralized project data, current outputs, and prior outputs.

    ``outputs/...`` addresses the active timestamped run and is safe for writes
    and same-run intermediates. ``latest_outputs/...`` reads the newest matching
    artifact from an earlier run. ``input/``, ``inputs/``, and ``data/`` resolve
    below ``project_data/<project_name>``.
    """

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts:
        prefix = candidate.parts[0].lower()
        remainder = _without_optional_project_prefix(project_name, candidate.parts[1:])
        if prefix in {"output", "outputs"}:
            return project_output_dir(project_name, *remainder, create=False)
        if prefix in {"latest_output", "latest_outputs"}:
            return latest_project_output(project_name, *remainder)
        if prefix in {"input", "inputs", "data"}:
            return project_data_dir(project_name, *remainder, create=False)
        if prefix == "project_data":
            relative = _safe_relative_path(candidate, label="data path", root_label="project-data root")
            return REPOSITORY_ROOT / relative
    return Path(project_root) / candidate


def _new_run_root(project_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime(RUN_DIRECTORY_FORMAT)
    stem = f"{timestamp}_p{os.getpid()}"
    base = project_output_base(project_name, create=False)
    candidate = base / stem
    suffix = 2
    while candidate.exists() or any(run.root == candidate for run in _PROJECT_RUNS.values()):
        candidate = base / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _project_execution_frame(project_name: str) -> FrameType | None:
    """Find the active top-level project script frame, including Spyder runs."""

    project_root = (REPOSITORY_ROOT / "projects" / project_name).resolve()
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_globals.get("__name__") == "__main__":
            filename = frame.f_globals.get("__file__")
            if filename:
                try:
                    Path(filename).resolve().relative_to(project_root)
                except (OSError, ValueError):
                    pass
                else:
                    return frame
        frame = frame.f_back
    return None


def _is_run_directory(name: str) -> bool:
    try:
        _run_directory_datetime(name)
    except ValueError:
        return False
    return True


def _run_directory_datetime(name: str) -> datetime:
    timestamp, separator, remainder = name.partition("_p")
    if not separator or not remainder.split("_", 1)[0].isdigit():
        raise ValueError(f"Not a timestamped project-run directory: {name!r}")
    return datetime.strptime(timestamp, RUN_DIRECTORY_FORMAT)


def _run_directory_sort_key(name: str) -> tuple[datetime, int]:
    timestamp = _run_directory_datetime(name)
    process_and_suffix = name.partition("_p")[2].split("_", 1)
    suffix = int(process_and_suffix[1]) if len(process_and_suffix) == 2 else 1
    return timestamp, suffix


def _project_name(project_name: str, *, root_label: str) -> str:
    return _project_component(project_name, root_label=root_label).name


def _project_component(project_name: str, *, root_label: str) -> Path:
    project = _safe_relative_path(project_name, label="project name", root_label=root_label)
    if len(project.parts) != 1:
        raise ValueError(f"Project name must be one directory component: {project_name!r}")
    return project


def _joined_safe_parts(parts: tuple[str | Path, ...], *, label: str, root_label: str) -> Path:
    path = Path()
    for part in parts:
        path /= _safe_relative_path(part, label=label, root_label=root_label)
    return path


def _without_optional_project_prefix(project_name: str, parts: tuple[str, ...]) -> tuple[str, ...]:
    if parts and parts[0].lower() == project_name.lower():
        return parts[1:]
    return parts


def _safe_relative_path(path: str | Path, *, label: str, root_label: str) -> Path:
    candidate = Path(path)
    windows_candidate = PureWindowsPath(str(path))
    if (
        not candidate.parts
        or candidate.is_absolute()
        or windows_candidate.is_absolute()
        or windows_candidate.drive
        or ".." in candidate.parts
    ):
        raise ValueError(f"{label.capitalize()} must stay below the repository {root_label}: {path!r}")
    return candidate
