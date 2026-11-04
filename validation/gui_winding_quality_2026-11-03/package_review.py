"""Package an exact submitted model and its verified completed winding for the GUIs."""

from argparse import ArgumentParser
from pathlib import Path

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.bundles import RunArchive, project_key, revision_key


def package(source, run, destination):
    project = ProjectDocument.load(source)
    record = DesignService.load_run(run, array_names=())
    if project_key(project) != revision_key(record.metadata):
        raise ValueError("Source document does not identify the submitted run.")
    archive = RunArchive.capture(record, source_project=project)
    session = {
        **project.session,
        "selected_run_id": archive.id,
        "view_state": {"source": "run", "dataset": "review_conductor", "model_node": "results"},
    }
    target = project.with_results((archive,), session=session).save(destination)
    restored = ProjectDocument.load(target)
    if project_key(restored) != project_key(project) or restored.saved_runs[0].id != archive.id:
        raise ValueError("Reopened bundle did not retain its exact submitted inputs and evidence.")
    print(f"{target}  ·  {record.metadata['request_sha256']}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package(args.source, args.run, args.destination)
