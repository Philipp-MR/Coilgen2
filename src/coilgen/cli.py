"""Headless project commands using the same application API as the desktop GUI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import sys

from .attribution import AUTHOR_CREDIT, AUTHOR_ROLE


def main(argv=None) -> int:
    from .application.presets import PRESET_NAMES

    parser = argparse.ArgumentParser(
        prog="coilgen",
        description="CoilGen2 reproducible scientific projects",
        epilog=f"{AUTHOR_CREDIT} ({AUTHOR_ROLE}).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="Create an editable generated-data project")
    new.add_argument("--preset", choices=PRESET_NAMES, default="cylinder")
    new.add_argument("--name", default="Coil design")
    new.add_argument("--output", required=True, type=Path)
    inspect = commands.add_parser("inspect", help="Verify project inputs and print resolved settings")
    inspect.add_argument("project", type=Path)
    run = commands.add_parser("run", help="Execute a saved project without importing GUI dependencies")
    run.add_argument("project", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--stage", choices=("prepared", "sheet", "contours", "routed", "evaluated"))
    run.add_argument("--resume", type=Path, help="Verified compatible pipeline checkpoint")
    run.add_argument(
        "--require-accepted", action="store_true", help="Exit 2 for failed or unverified engineering acceptance"
    )
    export = commands.add_parser(
        "export-script", help="Export a reproducible Python script and its owned project assets"
    )
    export.add_argument("project", type=Path)
    export.add_argument("output", type=Path)
    commands.add_parser("capabilities", help="Print installed numerical solver capabilities")
    args = parser.parse_args(argv)
    try:
        from .application import DesignService, ProjectDocument

        if args.command == "new":
            from .application.presets import preset_config, preset_study

            project = ProjectDocument.new(args.name, preset_config(args.preset)).update_study(preset_study(args.preset))
            path = project.save(args.output)
            print(path)
            return 0
        if args.command == "capabilities":
            from .optimization_solvers import conic_solver_capabilities
            from .research.storage import environment_manifest

            print(
                json.dumps(
                    {"solvers": [asdict(s) for s in conic_solver_capabilities()], **environment_manifest()}, indent=2
                )
            )
            return 0
        project = ProjectDocument.load(args.project)
        if args.command == "inspect":
            print(
                json.dumps(
                    {
                        "id": project.id,
                        "name": project.name,
                        "revision": project.revision,
                        "fingerprint": project.fingerprint,
                        "study": project.study,
                        "config": project.values,
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        if args.command == "export-script":
            print(project.export_python(args.output))
            return 0
        from .progress import progress_context
        from .research.jobs import RuntimePolicy

        RuntimePolicy(threads_per_worker=args.threads)
        controls = {}
        if args.stage:
            controls["pipeline_stage"] = args.stage
        if args.resume:
            controls["resume_checkpoint"] = str(args.resume.expanduser().resolve())
        with progress_context(
            lambda event: print(f"{event['label']}: {event['message']}", file=sys.stderr, flush=True)
        ):
            result = DesignService.run(project, args.output, threads=args.threads, **controls)
        print(json.dumps(result["summary"], indent=2, allow_nan=False))
        return 2 if args.require_accepted and result["summary"].get("accepted") is not True else 0
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        print(f"CoilGen2: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
