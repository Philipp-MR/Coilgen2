"""Entry point for the optional PySide6 scientific workbench."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys


def _startup_message(message: str) -> None:
    """Keep the launcher console informative during slow native imports."""
    print(f"[CoilGen2] {message}", flush=True)


def launch(project=None, *, jobs_directory=None, background=False, project_path=None) -> int:
    os.environ.setdefault("QT_API", "pyside6")
    _startup_message("Loading GUI and scientific libraries...")
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication
        from .workbench import CoilGenWorkbench
    except ImportError as exc:
        raise RuntimeError(
            "The native workbench requires optional GUI dependencies. "
            'Install them with: python -m pip install -e ".[gui]"'
        ) from exc
    _startup_message("Initializing Qt...")
    application = QApplication.instance()
    own_application = application is None
    if own_application:
        application = QApplication(sys.argv[:1])
    application.setApplicationName("CoilGen2")
    application.setOrganizationName("CoilGen")
    window = CoilGenWorkbench(project, jobs_directory=jobs_directory, startup_progress=_startup_message)
    if project_path is not None:
        window.project_path = Path(project_path).expanduser().resolve()
        window.undo_stack.setClean()
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, background)
    window.show()
    # Retain a Python owner when launched from a running Qt scientific session.
    application._coilgen_workbenches = getattr(application, "_coilgen_workbenches", []) + [window]
    window_id = id(window)
    window.destroyed.connect(
        lambda *_args: setattr(
            application,
            "_coilgen_workbenches",
            [item for item in application._coilgen_workbenches if id(item) != window_id],
        )
    )
    QTimer.singleShot(0, window, lambda: _startup_message("Workspace ready."))
    return application.exec() if own_application else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CoilGen2 native scientific workbench")
    parser.add_argument("project", nargs="?", type=Path, help="Saved CoilGen project JSON")
    parser.add_argument("--jobs-directory", type=Path, help="Directory for durable local jobs")
    parser.add_argument("--background", action="store_true", help="Open without taking keyboard focus")
    arguments = parser.parse_args(argv)
    try:
        project = None
        if arguments.project:
            _startup_message(f"Loading project: {arguments.project.name}")
            from ..application import ASSEMBLY_SCHEMA, ProjectAssembly, ProjectDocument
            from ..application.assembly import LEGACY_ASSEMBLY_SCHEMAS

            value = json.loads(arguments.project.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Project file must contain a JSON object.")
            project = (
                ProjectAssembly.load(arguments.project)
                if value.get("schema") in {ASSEMBLY_SCHEMA, *LEGACY_ASSEMBLY_SCHEMAS}
                else ProjectDocument.load(arguments.project)
            )
        return launch(
            project,
            jobs_directory=arguments.jobs_directory,
            background=arguments.background,
            project_path=arguments.project,
        )
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
