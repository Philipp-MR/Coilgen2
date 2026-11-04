"""Compute the selected GUI template with its unchanged study and retain its source."""

from pathlib import Path
import argparse

from threadpoolctl import threadpool_limits

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.templates import template_assembly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--channel", type=int, default=0, help="Zero-based index for a system template")
    parser.add_argument("--mesh", type=int, nargs=2, help="Explicit alternative angular/axial or plate divisions")
    parser.add_argument(
        "--cut-sides",
        nargs="+",
        choices=("auto", "high", "low"),
        help="Explicit group cut sides for a routing comparison",
    )
    parser.add_argument("--resume-checkpoint", type=Path, help="Verified upstream checkpoint for a routing comparison")
    args = parser.parse_args()
    project = template_assembly(args.template).channels[args.channel].project
    if args.mesh:
        from compare_contours import with_resolution

        project = with_resolution(project, *args.mesh)
    if args.cut_sides:
        project = project.update_config({"force_cut_selection": tuple(args.cut_sides)})
    args.directory.mkdir(parents=True, exist_ok=True)
    source = args.directory / "source.coilgen"
    if source.exists():
        saved = ProjectDocument.load(source)
        if (
            saved.name != project.name
            or saved.fingerprint != project.fingerprint
            or saved.model_fingerprint != project.model_fingerprint
        ):
            raise ValueError("Existing source does not match this template and study. Choose a new directory.")
        project = saved
    else:
        project.inputs_only().save(source)
    with threadpool_limits(args.threads):
        controls = {"resume_checkpoint": str(args.resume_checkpoint.resolve())} if args.resume_checkpoint else {}
        result = DesignService.run(project, args.directory / "run", threads=args.threads, **controls)
    assert result["project_fingerprint"] == project.fingerprint and result["project_revision"] == project.revision
    print(project.name, project.revision, result["request_sha256"], flush=True)


if __name__ == "__main__":
    main()
