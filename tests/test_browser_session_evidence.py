"""Browser evidence survives revisions, relocation and process restarts."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from zipfile import is_zipfile

import pytest

from coilgen.application.project import ProjectDocument
from coilgen.application.session import WorkbenchSession
from coilgen.application.session_evidence import project_key, revision_key
from coilgen.persistence import atomic_json


def finish(session):
    deadline = time.monotonic() + 180
    while session.active() and time.monotonic() < deadline:
        time.sleep(0.3)
        session.poll()
    assert not session.active()
    assert all(job["status"] == "completed" for job in session.job_states()), session.job_states()


@pytest.mark.parametrize("value", [False, None, "false", 1])
def test_only_an_explicit_boolean_can_report_construction_release(tmp_path, value):
    session = WorkbenchSession(tmp_path)
    record = SimpleNamespace(metadata={"summary": {"accepted": True, "construction_released": value}})
    assert session.run_summary(record)["construction_released"] is False


def test_new_same_name_documents_get_distinct_default_paths_and_explicit_save_as_replaces(tmp_path):
    first, second = WorkbenchSession(tmp_path), WorkbenchSession(tmp_path)
    first.new_project("Shared label")
    second.new_project("Shared label")
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda session: session.save_project(), (first, second)))
    assert len(set(paths)) == 2
    for session, path in zip((first, second), paths):
        assert json.loads(path.read_text(encoding="utf-8"))["id"] == session.assembly.id
        assert session.save_project() == path
    assert second.save_project(paths[0]) == paths[0]
    assert json.loads(paths[0].read_text(encoding="utf-8"))["id"] == second.assembly.id
    first.new_project("CON")
    assert first.save_project().name == "design-CON.coilgen.json"


def test_browser_formats_keep_separate_owned_files_on_switch_save_and_import(tmp_path):
    first, second = WorkbenchSession(tmp_path), WorkbenchSession(tmp_path)
    first.new_project("Duplicate")
    second.new_project("Duplicate")
    with ThreadPoolExecutor(max_workers=2) as pool:
        bundles = list(pool.map(lambda s: s.save_project(format="bundle"), (first, second)))
        jsons = list(pool.map(lambda s: s.save_project(format="json"), (first, second)))
    assert len(set(bundles + jsons)) == 4
    for session, bundle, text in zip((first, second), bundles, jsons):
        assert bundle.suffix == ".coilgen" and is_zipfile(bundle)
        assert text.name.endswith(".coilgen.json") and not is_zipfile(text)
        assert session.save_project(format="bundle") == bundle
        assert session.save_project(format="json") == text
        with pytest.raises(ValueError, match="must end"):
            session.save_project(text, format="bundle")
        session.open_project(bundle)
        assert session.save_project(format="bundle") == bundle
    second.save_project(bundles[0])
    taken = bundles[0].read_bytes()
    assert first.save_project(format="bundle") not in bundles
    assert bundles[0].read_bytes() == taken
    uploads = tmp_path / "uploads" / "unique"
    uploads.mkdir(parents=True)
    original = uploads / "original.coilgen"
    original.write_bytes(bundles[0].read_bytes())
    before = original.read_bytes()
    first.open_project(original)
    first.apply_settings({"levels": 9})
    copied = first.save_project(format="bundle")
    assert copied.parent == tmp_path / "projects" and copied not in bundles
    assert original.read_bytes() == before
    assert first.save_project(format="bundle") == copied
    first.new_project("Duplicate")
    assert first.save_project(format="bundle") not in [*bundles, copied]


def test_legacy_cad_sources_use_the_browser_loader_before_model_construction(tmp_path, monkeypatch):
    from coilgen.application.bundles import SOURCE_PROJECT_FILE
    from coilgen.application.session_evidence import source_project
    from coilgen.web.input_policy import load_browser_assembly

    loader = partial(load_browser_assembly, workspace=tmp_path)
    model = WorkbenchSession(tmp_path)
    model.load_template("cylinder")
    invalid = model.project.to_dict()
    invalid["config"]["field_shape_function"] = "x.__class__"
    rejected = tmp_path / "uploads" / "earlier-rejected-upload" / "case.json"
    rejected.parent.mkdir(parents=True)
    atomic_json(rejected, invalid)
    cached = tmp_path / "embedded-runs" / "fixture"
    cached.mkdir(parents=True)
    record = SimpleNamespace(directory=cached, metadata_view={})
    monkeypatch.setattr(ProjectDocument, "from_dict", lambda *a, **kw: pytest.fail("Unvalidated model constructed"))
    assert source_project(tmp_path, record, (), loader, search_saved=True) is None
    atomic_json(cached / SOURCE_PROJECT_FILE, invalid)
    with pytest.raises(ValueError, match="numerical"):
        source_project(tmp_path, record, (), loader, search_saved=True)


@pytest.mark.parametrize("full_review", (True, False))
def test_partial_boundary_remembers_full_review_after_reopening(tmp_path, full_review):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    session.apply_settings({"fabricate": full_review})
    session.set_stage("prepared")
    assert not session.project.study["fabricate"]
    path = session.save_project()
    session.close()
    reopened = WorkbenchSession(tmp_path)
    try:
        reopened.open_project(path)
        reopened.set_stage("evaluated")
        assert reopened.project.study["fabricate"] is full_review
    finally:
        reopened.close()


def test_historical_cad_matches_run_after_edit_restart_and_portable_reopen(tmp_path):
    root = tmp_path / "browser"
    session = WorkbenchSession(root)
    try:
        session.load_template("shim_z0")
        source = session.apply_config(
            {
                "levels": 3,
                "cylinder_mesh_parameter_list": (0.2125, 0.08, 32, 16, 1, 0, 0, 0),
            }
        )
        legacy = root / "uploads" / "unique-upload-id" / "original.json"
        legacy.parent.mkdir(parents=True)
        session.save_project(legacy)
        job = session.compute()
        finish(session)
        session.apply_settings({"levels": source.config.levels + 1})
        session.save_project(root / "projects" / "edited.json")
        bundle = session.save_project(format="bundle")
        assert len(session.project.saved_runs) == 1
        session.load_template("biplanar")
        record = session.load_run(job["id"])
        name = next(option.name for option in session.export_options(record) if option.format == "stl")
        # An old workspace may retain the original project under an arbitrary
        # filename, with no submission-time source snapshot.
        from coilgen.application.session_evidence import revision_path

        revision_path(root, project_key(source)).unlink()
        package = session.export(record, [name], include_project=True, allow_draft=True)
        assert package.directory.name.startswith("Uniform-B0-shim")
        companion = ProjectDocument.load(package.directory / "project.coilgen.json")
        assert project_key(companion) == project_key(source) == revision_key(record.metadata)
    finally:
        session.close()

    restarted = WorkbenchSession(root)
    try:
        assert [row["id"] for row in restarted.completed_runs()] == [job["id"]]
        restarted.open_project(bundle)
        assert [row["id"] for row in restarted.completed_runs()] == [job["id"]]
        assert len(restarted.project.saved_runs) == 1
    finally:
        restarted.close()

    relocated = tmp_path / "relocated"
    fresh = WorkbenchSession(relocated)
    try:
        fresh.open_project(bundle)
        (archived,) = fresh.completed_runs()
        record = fresh.load_run(archived["id"])
        assert record.metadata["metadata_sha256"] == session.load_run(job["id"]).metadata["metadata_sha256"]
        assert fresh.project.revision != source.revision
        package = fresh.export(record, [name], include_project=True, allow_draft=True)
        assert project_key(ProjectDocument.load(package.directory / "project.coilgen.json")) == project_key(source)
        path = fresh.save_project(relocated / "projects" / "second.json")
        assert len(fresh.project.saved_runs) == 1
    finally:
        fresh.close()
    reopened = WorkbenchSession(relocated)
    try:
        assert [row["id"] for row in reopened.completed_runs()] == [archived["id"]]
        reopened.open_project(path)
        record = reopened.load_run(archived["id"])
        # An unmodified legacy archive without an owned source revision may
        # still export geometry alone. It must never substitute its edited model.
        (record.directory / "source_project.coilgen.json").unlink()
        with pytest.raises(ValueError, match="exact source project revision is unavailable"):
            reopened.export(record, [name], include_project=True, allow_draft=True)
        assert reopened.export(record, [name], allow_draft=True).manifest["project_companion"] is None
    finally:
        reopened.close()


def test_multichannel_partial_jobs_round_trip_and_restore_terminal_history(tmp_path):
    source = tmp_path / "source"
    session = WorkbenchSession(source)
    try:
        session.load_template("cylindrical_xyz_gradient")
        session.set_channel_enabled("gy", False)
        batch = session.compute_channels("prepared")
        finish(session)
        session.select_channel("gz")
        path = session.save_project()
        assert set(batch) == {"gx", "gz"}
        assert [len(c.project.saved_runs) for c in session.channels] == [1, 0, 1]
    finally:
        session.close()
    for status in ("cancelled", "failed", "running"):
        directory = source / "jobs" / status
        directory.mkdir()
        atomic_json(directory / "job.json", {"id": status, "status": status, "project_id": "unrelated"})
    restarted = WorkbenchSession(source)
    try:
        assert len(restarted.completed_runs()) == 2
        assert restarted.states["cancelled"]["status"] == "cancelled"
        assert restarted.states["failed"]["status"] == "failed"
        assert restarted.states["running"]["status"] == "interrupted"
        restarted.open_project(path)
        assert len(restarted.completed_runs()) == 2
        assert restarted.active_channel_id == "gz"
        for channel in restarted.channels:
            for archive in channel.project.saved_runs:
                assert archive.project_id == channel.project.id
                assert archive.metadata["pipeline_stage"] == "prepared"
        previous = restarted.assembly
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["channels"][0]["project"]["results"][0]["files"]["result.json"]["data"] = "AAAA"
        bad = source / "tampered.json"
        atomic_json(bad, tampered)
        with pytest.raises(ValueError):
            restarted.open_project(bad)
        assert restarted.assembly is previous
        for id_ in ("failed", "cancelled", "running", "../../other"):
            with pytest.raises(ValueError, match="completed run"):
                restarted.load_run(id_)
    finally:
        restarted.close()
