"""Compact transport preserves immutable evidence and rejects unsafe archives."""

from hashlib import sha256
import json
import struct
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application import DesignService, RunArchive
from coilgen.persistence import atomic_json
from coilgen.application.portable import load_bundle

from test_application_project_bundle import _prepared_document


@pytest.fixture
def saved_project(tmp_path):
    project, record, archive = _prepared_document(tmp_path)
    return project.with_results((archive,), session={"selected_run_id": archive.id}), record


def _rewrite(source, destination, transform):
    with ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    transform(files)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)


@pytest.mark.parametrize("extension", (".json", ".coilgen"))
def test_full_project_roundtrip_preserves_bytes_arrays_dtype_and_owner(tmp_path, saved_project, extension):
    project, record = saved_project
    progress = []
    path = project.save(tmp_path / f"saved{extension}", progress=lambda *args: progress.append(args))
    loaded = ProjectDocument.load(path, progress=lambda *args: progress.append(args))
    assert loaded.id == project.id
    assert loaded.revision == project.revision
    assert loaded.fingerprint == project.fingerprint
    assert loaded.model_fingerprint == project.model_fingerprint
    assert loaded.values == project.values and loaded.study == project.study and loaded.session == project.session
    assert loaded.saved_runs == project.saved_runs
    assert loaded._assets == project._assets
    assert loaded.saved_runs[0].project_id == project.id
    restored = loaded.saved_runs[0].materialize(tmp_path / "cache")
    assert restored.metadata == record.metadata
    for name, data in record.arrays.items():
        np.testing.assert_array_equal(restored.array(name), data)
        assert restored.array(name).dtype == data.dtype
    if extension == ".coilgen":
        assert progress
        assert progress[-1][0] == progress[-1][1]
        with ZipFile(path) as archive:
            assert json.loads(archive.read("manifest.json"))["schema"] == "coilgen-portable-bundle-v1"
            assert not any("base64" in name for name in archive.namelist())


def test_compact_channel_assembly_deduplicates_inputs_and_keeps_distinct_owners(tmp_path, saved_project):
    project, _ = saved_project
    other = project.duplicate("Other channel")
    assembly = ProjectAssembly.new("Independent channels", (project, other))
    path = assembly.save(tmp_path / "pair.coilgen")
    loaded = ProjectAssembly.load(path)
    assert loaded.id == assembly.id and loaded.fingerprint == assembly.fingerprint
    assert [channel.id for channel in loaded.channels] == [channel.id for channel in assembly.channels]
    assert loaded.channels[0].project.id == project.id
    assert loaded.channels[1].project.id == other.id
    assert loaded.channels[0].project.saved_runs == project.saved_runs
    assert loaded.channels[1].project.saved_runs == ()
    with ZipFile(path) as archive:
        assert all(archive.namelist().count(f"blobs/{id_}") == 1 for id_ in project._assets)
    with pytest.raises(ValueError, match="assembly"):
        ProjectDocument.load(path)


def test_compact_transport_can_relocate_without_original_inputs_or_run(tmp_path, saved_project):
    import shutil

    project, record = saved_project
    path = project.save(tmp_path / "original.coilgen")
    moved = tmp_path / "moved" / "renamed.bin"
    moved.parent.mkdir()
    shutil.copyfile(path, moved)
    path.unlink()
    shutil.rmtree(record.directory)
    shutil.rmtree(tmp_path / "sources")
    loaded = ProjectDocument.load(moved)
    assert loaded.saved_runs == project.saved_runs
    loaded.saved_runs[0].materialize(tmp_path / "recreated")


def test_corrupt_payload_manifest_unlisted_path_and_duplicate_members_are_rejected(tmp_path, saved_project):
    project, _ = saved_project
    source = project.save(tmp_path / "source.coilgen")

    def corrupt_blob(files):
        name = next(name for name in files if name.startswith("blobs/"))
        files[name] = bytes([files[name][0] ^ 1]) + files[name][1:]

    def corrupt_manifest(files):
        value = json.loads(files["manifest.json"])
        value["document"]["id"] = "forged-owner"
        files["manifest.json"] = json.dumps(value).encode()

    for index, transform in enumerate(
        (corrupt_blob, corrupt_manifest, lambda files: files.update({"../escape": b"bad"}))
    ):
        path = tmp_path / f"bad{index}.coilgen"
        _rewrite(source, path, transform)
        with pytest.raises(ValueError, match="integrity|unreferenced"):
            load_bundle(path)
    path = tmp_path / "duplicates.coilgen"
    _rewrite(source, path, lambda _files: None)
    with ZipFile(path, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("manifest.json", b"{}")
    with pytest.raises(ValueError, match="duplicate"):
        load_bundle(path)
    assert not (tmp_path.parent / "escape").exists()


def test_expanded_byte_limit_checked_before_loading_payloads(tmp_path, saved_project):
    project, _ = saved_project
    path = project.save(tmp_path / "bounded.coilgen")
    with pytest.raises(ValueError, match="expanded-byte limit"):
        load_bundle(path, max_bytes=1)


@pytest.mark.parametrize("target", ("manifest", "blob"))
@pytest.mark.parametrize("damage", ("unsupported_compression", "invalid_deflate"))
def test_zip_compression_and_decompression_errors_are_normalized(tmp_path, saved_project, target, damage):
    project, _record = saved_project
    path = project.save(tmp_path / "damaged.coilgen")
    with ZipFile(path) as archive:
        member = (
            archive.getinfo("manifest.json")
            if target == "manifest"
            else next(member for member in archive.infolist() if member.filename.startswith("blobs/"))
        )
        central_offset = archive.start_dir
    data = bytearray(path.read_bytes())
    if damage == "unsupported_compression":
        # Compression is declared in both the local and central member headers.
        struct.pack_into("<H", data, member.header_offset + 8, 99)
        while data[central_offset : central_offset + 4] == b"PK\x01\x02":
            name_length, extra_length, comment_length = struct.unpack_from("<HHH", data, central_offset + 28)
            name = bytes(data[central_offset + 46 : central_offset + 46 + name_length]).decode()
            if name == member.filename:
                struct.pack_into("<H", data, central_offset + 10, 99)
                break
            central_offset += 46 + name_length + extra_length + comment_length
        else:
            pytest.fail("Member central directory entry was not found")
    else:
        name_length, extra_length = struct.unpack_from("<HH", data, member.header_offset + 26)
        # BTYPE=3 is forbidden by the DEFLATE stream format.
        data[member.header_offset + 30 + name_length + extra_length] = 0x07
    path.write_bytes(data)
    with pytest.raises(ValueError, match="corrupt|unsupported compression"):
        load_bundle(path)


def test_channel_owner_swapping_rejected_even_with_updated_transport_hashes(tmp_path, saved_project):
    project, _ = saved_project
    other = project.duplicate("Other")
    source = ProjectAssembly.new("Pair", (project, other)).save(tmp_path / "source.coilgen")

    def swap_owner(files):
        envelope = json.loads(files["manifest.json"])
        channel = envelope["document"]["channels"][0]
        channel["project"]["id"] = other.id

        def encode(value):
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

        channel["sha256"] = sha256(encode(channel["project"])).hexdigest()
        envelope["sha256"] = sha256(encode(envelope["document"])).hexdigest()
        files["manifest.json"] = encode(envelope)

    path = tmp_path / "swapped.coilgen"
    _rewrite(source, path, swap_owner)
    with pytest.raises(ValueError, match="different project owner"):
        load_bundle(path)


def test_failed_compact_save_preserves_existing_file_and_removes_staging(tmp_path, saved_project):
    project, _ = saved_project
    path = tmp_path / "previous.coilgen"
    path.write_bytes(b"previous file")

    def fail(*_args):
        raise RuntimeError("Write interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        project.save(path, progress=fail)
    assert path.read_bytes() == b"previous file"
    assert not list(tmp_path.glob(".previous.coilgen.*.tmp"))


@pytest.mark.parametrize("compact", (False, True))
def test_python_export_omits_completed_evidence_by_default_but_can_include_it(tmp_path, saved_project, compact):
    project, _ = saved_project
    script = project.export_python(tmp_path / "replay.py", compact=compact)
    companion = script.with_suffix(".coilgen" if compact else ".coilgen.json")
    loaded = ProjectDocument.load(companion)
    assert loaded.fingerprint == project.fingerprint and loaded.model_fingerprint == project.model_fingerprint
    assert loaded._assets == project._assets
    assert loaded.saved_runs == () and loaded.session == {}
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    project.export_python(script, compact=compact, include_results=True)
    assert ProjectDocument.load(companion).saved_runs == project.saved_runs


@pytest.mark.parametrize("compact", (False, True))
def test_inputs_only_companion_preserves_design_hypothesis_and_discards_ui_state(tmp_path, saved_project, compact):
    from coilgen.application.design_studies import design_study_state, set_study_notes_batch

    project, _record = saved_project
    project = set_study_notes_batch(project, hypothesis="Preserve the contour spacing rationale")
    session = {**project.session, "view_state": {"panel": "acceptance"}, "selected_run_id": project.saved_runs[0].id}
    project = project.with_results(project.saved_runs, session=session)
    script = project.export_python(tmp_path / "notes.py", compact=compact)
    loaded = ProjectDocument.load(script.with_suffix(".coilgen" if compact else ".coilgen.json"))
    assert design_study_state(loaded) == design_study_state(project)
    assert set(loaded.session) == {"design_study"} and loaded.saved_runs == ()
    assert loaded.fingerprint == project.fingerprint and loaded.revision == project.revision


@pytest.mark.parametrize("extension", (".json", ".coilgen"))
def test_future_archive_retains_actual_execution_dates_without_restoring_job_bindings(
    tmp_path, saved_project, extension
):
    project, original = saved_project
    times = {"created_at": "2026-10-11T03:27:53.159402+00:00", "completed_at": "2026-10-11T03:30:01.426467+00:00"}
    atomic_json(
        original.directory / "job.json",
        {
            "id": "live-worker-id",
            "status": "completed",
            "progress": 1,
            "project_id": project.id,
            "request_sha256": original.metadata["request_sha256"],
            "created_at": times["created_at"],
            "updated_at": times["completed_at"],
        },
    )
    record = DesignService.load_run(original.directory)
    archive = RunArchive.capture(record)
    assert archive.execution_times == times
    assert "job.json" not in dict(archive._files)
    for name in ("result.json", "request.json", "manifest.json"):
        assert dict(archive._files)[name] == (original.directory / name).read_bytes()
    path = project.with_results((archive,)).save(tmp_path / f"dated{extension}")
    loaded = ProjectDocument.load(path)
    restored = loaded.saved_runs[0].materialize(tmp_path / "cache")
    assert restored.execution_times == times
    assert "job" not in restored.metadata
    assert restored.metadata == original.metadata
    assert DesignService.load_run(restored.directory).execution_times == times
    assert RunArchive.capture(restored).execution_times == times
    detached = restored.execution_times
    detached["created_at"] = "changed"
    assert restored.execution_times == times
    from coilgen.gui.run_context import run_timestamp

    assert run_timestamp(restored).timestamp is not None
    assert run_timestamp(restored.metadata).timestamp is None
    # A sidecar whose dates changed cannot inherit the immutable cache identity.
    execution = restored.directory / "execution_times.json"
    payload = json.loads(execution.read_bytes())
    payload["times"]["created_at"] = "2020-01-01T00:00:00Z"
    atomic_json(execution, payload)
    assert DesignService.load_run(restored.directory).execution_times == {}
    with pytest.raises(ValueError, match="integrity"):
        loaded.saved_runs[0].materialize(tmp_path / "cache")


def test_other_worker_identity_or_invalid_date_is_not_archived(tmp_path, saved_project):
    project, record = saved_project
    path = record.directory / "job.json"
    for owner, timestamp in (("other-owner", "2026-10-11T04:00:00Z"), (project.id, "date unrecorded")):
        atomic_json(
            path, {"project_id": owner, "request_sha256": record.metadata["request_sha256"], "created_at": timestamp}
        )
        archive = RunArchive.capture(DesignService.load_run(record.directory))
        assert archive.execution_times == {}
