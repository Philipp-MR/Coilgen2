"""Create a conical Z2 shim study, optionally opening or computing it.

The target is Bz = 0.002*(2*z**2-x**2-y**2) T in world coordinates.
At a 50 mm reference radius this gives +10 uT at the poles and -5 uT
at the equator. The coefficient multiplies the existing normalized real
solid Y20; it is not the pole amplitude.

The support is a 300 mm long frustum, with bottom/top radii 180/140 mm.
The spherical ROI has radius 50 mm. The study uses 1 x 1 mm copper with
18 nOhm m resistivity, a 3 mm minimum bend radius and 0.1 mm insulation.
It fits the sheet to 4% RMS and checks it against a 5% budget on a distinct,
denser grid. The 12-level connected winding retains known fixed-current
field and manufacturing failures even when the sheet passes. It is a
routing investigation, not qualified hardware. Sheet, thin-wire, thermal
and drive-feed evidence have different physical scopes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coilgen.application import DesignService, ProjectDocument, RunArchive
from coilgen.application.presets import preset_config, preset_study


def build_project() -> ProjectDocument:
    return ProjectDocument.from_config(preset_config("conical_harmonic"), name="Conical Z2").update_study(
        preset_study("conical_harmonic")
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/conical_harmonic"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="Open the editable project in the native workbench")
    mode.add_argument("--compute", action="store_true", help="Run the saved full study without Qt")
    arguments = parser.parse_args(argv)
    path = arguments.output / "conical_z2.coilgen.json"
    if path.exists():
        parser.error(f"Project already exists: {path}. Choose a fresh --output directory.")
    project = build_project()
    project.save(path)
    print(path.resolve())
    if arguments.gui:
        from coilgen.gui import launch

        return launch(project, jobs_directory=arguments.output / "jobs", project_path=path.resolve())
    if arguments.compute:
        run_directory = arguments.output / "run"
        DesignService.run(project, run_directory)
        record = DesignService.load_run(run_directory)
        project.with_results((RunArchive.capture(record, alias="Full conical Z2 study"),)).save(path)
        print(record.metadata["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
