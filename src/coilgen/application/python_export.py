"""Editable, self-contained Python source for a project revision."""

from __future__ import annotations

from dataclasses import asdict
from pprint import pformat
from textwrap import indent

from ..attribution import AUTHOR_NAME, PROJECT_URL, SOFTWARE_NAME, SOFTWARE_VERSION
from ..config import CoilGenConfig
from ..persistence import json_value


def design_script(project) -> str:
    """Keep numerical inputs explicit and owned bytes inline, without saved runs."""
    document = project.inputs_only().to_dict()
    values = document.pop("config")
    study = document.pop("study")
    defaults = json_value(asdict(CoilGenConfig()))
    # Put the design's choices before unchanged defaults, while freezing both.
    changed = {key: value for key, value in values.items() if value != defaults[key]}
    unchanged = {key: value for key, value in values.items() if key not in changed}

    def arguments(items):
        return "\n".join(
            indent(f"{key}={pformat(value, width=100, sort_dicts=False)},", "    ") for key, value in items.items()
        )

    return (
        '"""Editable CoilGen2 design; run with Python in your CoilGen environment.\n\n'
        "Geometry uses meters, magnetic fields tesla and currents amperes.\n"
        "Angle and specialized field units follow CoilGenConfig.\n"
        "Edit CONFIG and STUDY below; imported inputs are embedded at the end.\n"
        "No GUI or companion project file is needed. Use the same CoilGen version\n"
        "and numerical dependencies for reproducible results.\n"
        f"Generated with {SOFTWARE_NAME} - version {SOFTWARE_VERSION}.\n"
        f"Software maintainer: {AUTHOR_NAME}.\n"
        f"Software repository: {PROJECT_URL}\n"
        '"""\n\n'
        "from dataclasses import asdict\n"
        "from pathlib import Path\n\n"
        "from coilgen import CoilGenConfig\n"
        "from coilgen.application import DesignService, ProjectDocument\n\n\n"
        "CONFIG = CoilGenConfig(\n"
        "    # Design choices that differ from this version's defaults.\n"
        f"{arguments(changed)}\n"
        "    # Resolved defaults are explicit so later default changes cannot alter this design.\n"
        f"{arguments(unchanged)}\n"
        ")\n\n"
        "# Preserve the GUI stage, validation, fabrication and physical requirements.\n"
        f"STUDY = {pformat(study, width=100, sort_dicts=False)}\n\n\n"
        "def build_project():\n"
        '    """Construct a detached design from the editable settings and owned inputs."""\n'
        "    return ProjectDocument.from_dict(\n"
        "        {**_PROJECT, 'config': asdict(CONFIG), 'study': STUDY},\n"
        "        project_root=Path(__file__).resolve().parent,\n"
        "    )\n\n\n"
        "def run(output_dir=None, *, threads=1):\n"
        '    """Compute the selected stage and save the same result artifacts as the GUI."""\n'
        "    if output_dir is None:\n"
        "        output_dir = Path(__file__).resolve().with_suffix('.results')\n"
        "    return DesignService.run(build_project(), output_dir, threads=threads)\n\n\n"
        "def main(argv=None):\n"
        "    import argparse\n"
        "    import json\n\n"
        "    parser = argparse.ArgumentParser(description=__doc__)\n"
        "    parser.add_argument('--output', type=Path, help='Result directory (default: beside this script)')\n"
        "    parser.add_argument('--threads', type=int, default=1, help='Numerical thread budget')\n"
        "    args = parser.parse_args(argv)\n"
        "    result = run(args.output, threads=args.threads)\n"
        "    print(json.dumps(result['summary'], indent=2, allow_nan=False))\n"
        "    return result\n\n\n"
        "# Project identity and content-addressed input bytes. Keep this data with the script.\n"
        "# Asset source paths are provenance only; original files are not required.\n"
        f"_PROJECT = {pformat(document, width=100, sort_dicts=False)}\n\n\n"
        "if __name__ == '__main__':\n"
        "    from multiprocessing import freeze_support\n\n"
        "    freeze_support()\n"
        "    main()\n"
    )
