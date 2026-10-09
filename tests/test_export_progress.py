"""Export progress follows verified staging and exclusive publication."""

import pytest
from test_application_cad import _run

from coilgen.application.cad import export_geometry
from coilgen.application.service import DesignService
from coilgen.gui.export_panel import ArtifactExportTask, GeometryExportTask, export_verified_artifacts
from coilgen.progress import DesignCancelled, progress_context


def _percentages(events):
    return [event["percent"] for event in events if event["percent"] is not None]


def test_geometry_progress_counts_bodies_and_published_files_without_changing_evidence(tmp_path):
    record = _run(tmp_path)
    originals = {row["name"]: (record.directory / row["name"]).read_bytes() for row in record.metadata["artifacts"]}
    events = []
    with progress_context(events.append):
        package = export_geometry(record, tmp_path / "geometry")
    percentages = _percentages(events)
    assert percentages == sorted(percentages)
    assert percentages[0] == 0 and percentages[-1] == 100
    assert len(set(percentages)) >= 10
    assert any("Writing saved solid 2/2" in event["message"] for event in events)
    assert any(event.get("total") == 3 and event.get("completed") == 3 for event in events)
    for name, data in originals.items():
        assert (record.directory / name).read_bytes() == data
    step = next(row for row in package.manifest["geometry"] if row["format"] == "step")
    assert (package.directory / step["name"]).read_bytes() == originals[step["source_artifact"]]


def test_artifact_progress_counts_staging_and_publication_before_completion(tmp_path, monkeypatch):
    import coilgen.gui.export_panel as exports

    record = _run(tmp_path)
    events, publications = [], []
    original_link = exports.os.link

    def checked_link(source, destination):
        assert max(_percentages(events)) < 100
        publications.append(destination)
        return original_link(source, destination)

    monkeypatch.setattr(exports.os, "link", checked_link)
    with progress_context(events.append):
        paths = export_verified_artifacts(record, tmp_path / "artifacts", loader=DesignService.load_run)
    percentages = _percentages(events)
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100
    assert len(paths) == len(publications) == len(record.metadata["artifacts"])
    for path, row in zip(paths, record.metadata["artifacts"]):
        assert path.read_bytes() == (record.directory / row["name"]).read_bytes()


def test_cancelled_cad_publication_rolls_back_only_its_package(tmp_path):
    record = _run(tmp_path)
    destination = tmp_path / "cancelled"
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    events = []
    with progress_context(events.append, lambda: any((event.get("percent") or 0) > 96 for event in events)):
        with pytest.raises(DesignCancelled):
            export_geometry(record, destination)
    assert max(_percentages(events)) < 100
    assert not destination.exists()
    assert unrelated.read_text() == "keep"
    assert not list(tmp_path.glob(".coilgen-cad-*"))


@pytest.mark.parametrize("kind", ["cad", "artifacts"])
def test_export_workers_flush_last_progress_before_completion(tmp_path, kind):
    record = _run(tmp_path)
    if kind == "cad":
        from coilgen.application.cad import geometry_catalog

        task = GeometryExportTask(record, tmp_path / "package", [row.name for row in geometry_catalog(record)], "mm")
    else:
        task = ArtifactExportTask(record, tmp_path / "artifacts", loader=DesignService.load_run)
    observed = []
    task.signals.progress.connect(lambda event: observed.append(("progress", event)))
    task.signals.finished.connect(lambda result, error: observed.append(("finished", (result, error))))
    task.run()
    assert observed[0][0] == "progress"
    assert observed[-2][0] == "progress" and observed[-2][1]["percent"] == 100
    assert observed[-1][0] == "finished" and observed[-1][1][1] == ""
