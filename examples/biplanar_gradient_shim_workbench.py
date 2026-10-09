"""Create or compute a nine-channel biplanar gradient/second-order shim study.

The default writes one self-contained JSON. --gui opens the complete editable
system; --compute runs channels independently and embeds completed results.
Nominal layer separation does not certify inter-channel clearance or coupling.
See biplanar_gradient_shim_README.md for the field amplitudes and layer layout.
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
import sys

from coilgen.application import DesignService, RunArchive
from coilgen.application.system_presets import SYSTEM_EXAMPLE_TITLES, system_assembly
from coilgen.persistence import atomic_json


def _emit(message):
    # Windows redirected stdout can use a legacy code page. Preserve scientific
    # names in JSON and show escaped characters if the terminal lacks a glyph.
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(str(message).encode(encoding, errors="backslashreplace").decode(encoding), flush=True)


def main(argv=None, *, kind="biplanar_gradient_shim") -> int:
    parser = argparse.ArgumentParser(description=f"Create or compute {SYSTEM_EXAMPLE_TITLES[kind]}.")
    parser.add_argument("--output", type=Path, default=Path("outputs") / kind)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="Open all channels in the native workbench")
    mode.add_argument(
        "--compute", action="store_true", help="Run independent full studies; embed results after each channel"
    )
    parser.add_argument("--channels", nargs="+", help="Compute selected channel IDs, e.g. gx gy; default: all channels")
    parser.add_argument(
        "--stage", choices=("prepared", "sheet", "contours", "routed", "evaluated"), default="evaluated"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--no-embed-results",
        action="store_true",
        help="Keep completed results in run folders rather than the project JSON",
    )
    arguments = parser.parse_args(argv)
    if arguments.threads < 1:
        parser.error("--threads must be positive.")
    if not arguments.compute and (arguments.channels or arguments.no_embed_results or arguments.stage != "evaluated"):
        parser.error("--channels, --stage and --no-embed-results apply to --compute.")
    directory = arguments.output.expanduser().resolve()
    assembly = system_assembly(kind, project_root=directory)
    channel_ids = {channel.id for channel in assembly.channels}
    if arguments.channels and (unknown := set(arguments.channels) - channel_ids):
        parser.error(f"Unknown channels: {', '.join(sorted(unknown))}. Choose {', '.join(sorted(channel_ids))}.")
    # An existing directory may contain old results, a live job or user files.
    # Require a fresh example workspace before writing any project or run.
    if directory.exists():
        parser.error(f"Output directory already exists: {directory}. Choose a fresh --output directory.")
    for channel in assembly.channels:
        DesignService.validate(channel.project).raise_for_errors()
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / f"{kind}.coilgen.json"
    assembly.save(path)
    _emit(path)
    if arguments.gui:
        from coilgen.gui.__main__ import launch

        return launch(assembly, jobs_directory=directory / "jobs", project_path=path)
    if arguments.compute:
        selected = channel_ids if arguments.channels is None else set(arguments.channels)
        failures = {}
        for channel in assembly.channels:
            if channel.id not in selected:
                continue
            project = channel.project
            if arguments.stage != "evaluated":
                project = project.update_study({"pipeline_stage": arguments.stage, "fabricate": False})
            run_directory = directory / "runs" / channel.id
            _emit(f"{channel.name}: {project.study['pipeline_stage']} study")
            try:
                DesignService.run(project, run_directory, threads=arguments.threads)
                record = DesignService.load_run(run_directory)
                if not arguments.no_embed_results:
                    archive = RunArchive.capture(record, alias=f"{channel.name} · {arguments.stage} study")
                    project = project.with_results((*project.saved_runs, archive))
                assembly = assembly.update_channel(channel.id, project=project)
                assembly.save(path)
                summary = record.metadata.get("summary") or {}
                _emit(
                    f"{channel.name}: sheet={summary.get('sheet_accepted', summary.get('accepted'))}; "
                    f"strict geometry={(record.metadata.get('fabrication') or {}).get('ready')}; "
                    f"construction release={summary.get('construction_released', False)}"
                )
            except Exception as exc:
                failures[channel.id] = f"{type(exc).__name__}: {exc}"
                _emit(f"{channel.name}: {failures[channel.id]}")
            atomic_json(
                directory / "channel_execution.json",
                {
                    "scope": "Independent channel execution; no inter-channel qualification",
                    "requested_stage": arguments.stage,
                    "selected_channels": sorted(selected),
                    "failures": failures,
                },
            )
        return 2 if failures else 0
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
