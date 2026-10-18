"""Durable links, source integrity and scoped comparisons without solver duplication."""

from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from pathlib import Path
import shutil
from threading import Event

import numpy as np
import pytest

from coilgen.application.comparison import compare_runs
from coilgen.application.history import RunHistoryStore
from coilgen.application.service import DesignService
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.storage import result_metadata_digest


def write_run(directory, *, revision=1, config=None, summary=None, refitted=False, grid_shift=0, input_path=None):
    """Small recorded evidence fixture using the real persistence verifier."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    config = {"levels": 8, "target_field_component": "Bz", "b_0_direction": [0, 0, 1], **(config or {})}
    request = {
        "name": "Original scientific name",
        "project_id": "source-project",
        "project_revision": revision,
        "config": config,
        "pipeline_stage": "evaluated",
        "validation_resolution": 12,
        "design_margin": 1,
        "fabricate": False,
        "kernel_backend": "auto",
    }
    digest = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    input_files = {}
    if input_path:
        input_files["target_field_definition_file"] = {"path": str(input_path), "sha256": file_digest(input_path)}
    atomic_json(directory / "request.json", request)
    atomic_json(
        directory / "manifest.json",
        {
            "request_sha256": digest,
            "request": request,
            "resolved_config": config,
            "input_files": input_files,
        },
    )
    points = np.array([[0, 0, 0.01], [0.01, 0, 0]]) + grid_shift
    target = np.array([[0, 0, 1e-5], [0, 0, -1e-5]])
    np.savez_compressed(
        directory / "scientific_result.npz",
        target_points_m=points,
        target_field_T=target,
        target_weights=np.ones(2),
        target_suppressed=np.zeros(2, bool),
        validation_points_m=points + 0.002,
        validation_target_T=target,
    )
    wire = {
        "available": True,
        "component": "Bz",
        "sample_count": 2,
        "current_A": 4.0,
        "current_source": "fit" if refitted else "stream_function",
        "current_refitted": refitted,
        "geometry_sources": ["oriented_unconnected_contours"],
        "field_model": "thin_wire_biot_savart",
        "rms_convention": "scalar",
        "rms_error_T": 1e-6,
        "peak_error_T": 2e-6,
        "relative_rms_error": 0.1,
        "independent_sampling": False,
        "acceptance_claim": False,
        "at_stream_current": {
            "available": True,
            "sample_count": 2,
            "current_A": 3.9,
            "current_source": "original_contour_step",
            "current_refitted": False,
            "rms_error_T": 1.2e-6,
        },
    }
    metadata = {
        "name": request["name"],
        "project_id": request["project_id"],
        "project_revision": revision,
        "request_sha256": digest,
        "pipeline_stage": "evaluated",
        # Deliberately not authoritative for parameter comparison.
        "config": {"levels": 999},
        "summary": {
            "accepted": False,
            "scope": "continuous_sheet",
            "relative_field_error": 0.02,
            "validated_relative_field_error": 0.025,
            "power_w": 1.2,
            "current_a": None,
            "resistance_ohm": None,
            "inductance_h": None,
            **(summary or {}),
        },
        "validation": {
            "available": True,
            "point_count": 2,
            "rms_error_T": 2.5e-7,
            "relative_field_error": 0.025,
            "peak_error_T": 3e-7,
            "independent_quadrature_check": False,
        },
        "wire_fields": {"contour": wire},
        "artifacts": [
            {"name": name, "kind": kind, "sha256": file_digest(directory / name)}
            for name, kind in (
                ("request.json", "request"),
                ("manifest.json", "provenance"),
                ("scientific_result.npz", "npz"),
            )
        ],
    }
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(directory / "result.json", metadata)
    return DesignService.load_run(directory)


def test_project_link_alias_and_view_state_survive_reopen_without_touching_scientific_evidence(tmp_path):
    run = write_run(tmp_path / "runs" / "first")
    before = {path.name: path.read_bytes() for path in run.directory.iterdir()}
    path = tmp_path / "project" / "session.json"
    history = RunHistoryStore(path)
    linked = history.add_run("stable-project", run, alias="Eight turns")
    history.rename_run(linked.id, "Baseline Z2")
    view = {"dataset": "routed field", "component": "Bz", "camera": [[1, 2, 3], [0, 0, 0], [0, 1, 0]]}
    history.save_session("stable-project", selected_id=linked.id, view_state=view)
    view["component"] = "Bx"
    reopened = RunHistoryStore(path)
    entries = reopened.restore("stable-project")
    assert len(entries) == 1 and entries[0].verified
    assert entries[0].label == "Baseline Z2"
    assert entries[0].run.metadata["name"] == "Original scientific name"
    assert reopened.session("stable-project")["selected_id"] == linked.id
    assert reopened.session("stable-project")["view_state"]["component"] == "Bz"
    assert {path.name: path.read_bytes() for path in run.directory.iterdir()} == before
    assert not Path(json.loads(path.read_text())["records"][0]["path"]).is_absolute()


def test_stable_project_channel_filter_and_separate_store_edits_preserve_all_revisions(tmp_path):
    first = write_run(tmp_path / "runs" / "first", revision=1)
    second = write_run(tmp_path / "runs" / "second", revision=7, config={"levels": 12})
    one = RunHistoryStore(tmp_path / "session.json")
    two = RunHistoryStore(tmp_path / "session.json")
    a = one.add_run("p", first)
    two.add_run("p", second)
    one.add_run("p", first, channel_id="stable-channel")
    two.add_run("other-project", first)
    assert [entry.run.metadata["project_revision"] for entry in one.restore("p")] == [1, 7]
    assert len(one.restore("p", channel_id="stable-channel")) == 1
    assert len(one.restore("other-project")) == 1
    one.rename_run(a.id, "Named baseline")
    assert two.add_run("p", first).label == "Named baseline"
    assert len(two.restore("p")) == 2


def test_relative_links_relocate_with_workspace(tmp_path):
    source = tmp_path / "original"
    record = write_run(source / "runs" / "a")
    store = RunHistoryStore(source / "projects" / "session.json")
    store.add_run("p", record)
    moved = tmp_path / "relocated"
    shutil.copytree(source, moved)
    restored = RunHistoryStore(moved / "projects" / "session.json").restore("p")[0]
    assert restored.verified
    assert restored.directory == moved / "runs" / "a"


def test_missing_corrupt_and_changed_input_links_remain_visible_without_discarding_valid_runs(tmp_path):
    stable_input = tmp_path / "source.npz"
    stable_input.write_bytes(b"immutable source")
    records = [
        write_run(tmp_path / name, input_path=stable_input if name == "input" else None)
        for name in ("valid", "missing", "corrupt", "input")
    ]
    store = RunHistoryStore(tmp_path / "session.json")
    for record in records:
        store.add_run("p", record)
    shutil.rmtree(records[1].directory)
    (records[2].directory / "result.json").write_text("{broken", encoding="utf-8")
    stable_input.write_bytes(b"modified source")
    entries = RunHistoryStore(store.path).restore("p")
    assert len(entries) == 4
    assert entries[0].verified
    assert all(not entry.verified and entry.diagnostic for entry in entries[1:])
    assert "changed during execution" in entries[3].diagnostic
    assert len(json.loads(store.path.read_text())["records"]) == 4


def test_valid_replacement_at_same_run_path_cannot_change_pinned_historical_identity(tmp_path):
    run = write_run(tmp_path / "run")
    store = RunHistoryStore(tmp_path / "session.json")
    store.add_run("p", run)
    shutil.rmtree(run.directory)
    replaced = write_run(run.directory, config={"levels": 12})
    entry = store.restore("p")[0]
    assert not entry.verified and "different request identity" in entry.diagnostic
    with pytest.raises(ValueError, match="different request identity"):
        store.add_run("p", replaced)


def test_session_scope_and_json_validation_are_atomic(tmp_path):
    run = write_run(tmp_path / "run")
    store = RunHistoryStore(tmp_path / "session.json")
    entry = store.add_run("p", run, channel_id="a")
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="does not belong"):
        store.save_session("p", channel_id="b", selected_id=entry.id)
    with pytest.raises(ValueError, match="Nonfinite"):
        store.save_session("p", channel_id="a", view_state={"camera": [np.nan]})
    assert store.path.read_bytes() == before
    with pytest.raises(ValueError, match="outside immutable"):
        RunHistoryStore(run.directory / "session.json").add_run("p", run)
    assert not (run.directory / "session.json").exists()


def test_scope_aware_comparison_reads_original_request_and_retains_unknown_circuit_metrics(tmp_path):
    first = write_run(tmp_path / "a")
    second = write_run(tmp_path / "b", config={"levels": 12})
    comparison = compare_runs(first, second)
    assert [(delta.field, delta.left, delta.right) for delta in comparison.parameter_deltas] == [
        ("config.levels", 8, 12)
    ]
    metrics = {metric.name: metric for metric in comparison.metrics}
    assert metrics["Sheet independent RMS error"].left == pytest.approx(0.25)
    assert metrics["Sheet independent RMS error"].comparable
    assert metrics["Contour RMS error"].comparable
    assert metrics["Contour RMS error"].reason == ""
    assert "independent acceptance=False" in metrics["Contour RMS error"].left_scope
    assert "continuous sheet" in metrics["Sheet independent RMS error"].left_scope
    for name in ("Circuit current", "Circuit resistance", "Circuit inductance"):
        assert metrics[name].left is None and not metrics[name].comparable


def test_refitted_current_and_changed_grids_do_not_become_common_scope_improvement(tmp_path):
    first = write_run(tmp_path / "a")
    refitted = write_run(tmp_path / "b", refitted=True)
    metrics = {metric.name: metric for metric in compare_runs(first, refitted).metrics}
    assert not metrics["Contour RMS error"].comparable
    assert "refitted=True" in metrics["Contour RMS error"].right_scope
    assert metrics["Contour at original current RMS error"].comparable
    assert "refitted=False" in metrics["Contour at original current RMS error"].right_scope
    shifted = write_run(tmp_path / "c", grid_shift=0.01)
    comparison = compare_runs(first, shifted)
    metrics = {metric.name: metric for metric in comparison.metrics}
    assert not metrics["Contour RMS error"].comparable
    assert not metrics["Sheet independent RMS error"].comparable
    assert any("not a common-grid improvement" in note for note in comparison.notes)


def test_comparison_rechecks_cached_records_before_reading_or_displaying_evidence(tmp_path):
    first = write_run(tmp_path / "a")
    second = write_run(tmp_path / "b")
    (second.directory / "request.json").write_text('{"config":{"levels":1}}', encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        compare_runs(first, second)


def test_malformed_historical_input_schema_is_visible_even_when_file_hashes_match(tmp_path):
    first = write_run(tmp_path / "a")
    second = write_run(tmp_path / "b")
    store = RunHistoryStore(tmp_path / "session.json")
    store.add_run("p", second)
    manifest = json.loads((second.directory / "manifest.json").read_text())
    manifest["input_files"] = []
    atomic_json(second.directory / "manifest.json", manifest)
    metadata = json.loads((second.directory / "result.json").read_text())
    for artifact in metadata["artifacts"]:
        if artifact["name"] == "manifest.json":
            artifact["sha256"] = file_digest(second.directory / "manifest.json")
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(second.directory / "result.json", metadata)
    entry = store.restore("p")[0]
    assert not entry.verified and entry.diagnostic
    with pytest.raises(ValueError, match="Cannot read comparison"):
        compare_runs(first, second)


def test_complete_circuit_values_keep_physical_units_and_scope(tmp_path):
    summary = {"scope": "fabrication_review", "current_a": 4.2, "resistance_ohm": 0.2, "inductance_h": 17e-6}
    first = write_run(tmp_path / "a", summary=summary)
    second = write_run(tmp_path / "b", summary=summary | {"current_a": 3.5, "inductance_h": 20e-6})
    metrics = {metric.name: metric for metric in compare_runs(first, second).metrics}
    assert metrics["Circuit current"].left == 4.2
    assert metrics["Circuit current"].right == 3.5
    assert metrics["Circuit resistance"].unit == "Ω"
    assert metrics["Circuit inductance"].left == pytest.approx(17)
    assert metrics["Circuit inductance"].right == pytest.approx(20)
    assert metrics["Circuit inductance"].comparable
    assert "single complete modeled circuit only" in metrics["Circuit inductance"].left_scope


def test_identical_input_content_at_different_locations_is_provenance_not_physics_delta(tmp_path):
    first_input, second_input = tmp_path / "one.map", tmp_path / "two.map"
    first_input.write_bytes(b"same field map")
    second_input.write_bytes(first_input.read_bytes())
    first = write_run(tmp_path / "a", input_path=first_input, config={"target_field_definition_file": str(first_input)})
    second = write_run(
        tmp_path / "b", input_path=second_input, config={"target_field_definition_file": str(second_input)}
    )
    comparison = compare_runs(first, second)
    assert comparison.parameter_deltas == ()
    assert any("input content is identical" in note for note in comparison.notes)
    second_input.write_bytes(b"different map")
    third = write_run(
        tmp_path / "c", input_path=second_input, config={"target_field_definition_file": str(first_input)}
    )
    comparison = compare_runs(first, third)
    assert any(delta.field == "inputs.target_field_definition_file.sha256" for delta in comparison.parameter_deltas)


def test_standalone_scope_link_preserves_aliases_unavailable_links_and_selected_view_without_moving_evidence(tmp_path):
    first = write_run(tmp_path / "a")
    second = write_run(tmp_path / "b")
    store = RunHistoryStore(tmp_path / "session.json")
    original = store.add_run("p", first, alias="Baseline before channels")
    missing = store.add_run("p", second, alias="Missing historical design")
    view = {"dataset": "contour field", "component": "Bz", "camera": [[1, 2, 3], [0, 0, 0], [0, 1, 0]]}
    store.save_session("p", selected_id=missing.id, view_state=view)
    before = json.loads(store.path.read_text())
    first_evidence = {path.name: path.read_bytes() for path in first.directory.iterdir()}
    shutil.rmtree(second.directory)
    mapping = store.link_scope("p", target_channel_id="stable-original-channel")
    assert set(mapping) == {original.id, missing.id}
    assert mapping[original.id] != original.id
    copied = store.restore("p", channel_id="stable-original-channel")
    assert [entry.label for entry in copied] == ["Baseline before channels", "Missing historical design"]
    assert copied[0].verified and not copied[1].verified and copied[1].diagnostic
    assert copied[0].request_sha256 == original.request_sha256
    assert copied[1].request_sha256 == missing.request_sha256
    assert copied[0].directory == original.directory
    assert store.session("p", channel_id="stable-original-channel") == {
        "selected_id": mapping[missing.id],
        "view_state": view,
    }
    after = json.loads(store.path.read_text())
    assert after["records"][:2] == before["records"]
    assert store.session("p") == {"selected_id": missing.id, "view_state": view}
    assert {path.name: path.read_bytes() for path in first.directory.iterdir()} == first_evidence
    assert not second.directory.exists()


def test_scope_link_is_idempotent_and_keeps_existing_target_aliases_and_session(tmp_path):
    record = write_run(tmp_path / "run")
    store = RunHistoryStore(tmp_path / "session.json")
    source = store.add_run("p", record, alias="Standalone alias")
    store.save_session("p", selected_id=source.id, view_state={"component": "Bz"})
    target = store.add_run("p", record, channel_id="a", alias="Existing channel alias")
    store.save_session("p", channel_id="a", selected_id=target.id, view_state={"component": "By"})
    assert store.link_scope("p", target_channel_id="a") == {source.id: target.id}
    assert store.link_scope("p", target_channel_id="a") == {source.id: target.id}
    assert len(store.restore("p", channel_id="a")) == 1
    assert store.restore("p", channel_id="a")[0].alias == "Existing channel alias"
    assert store.session("p", channel_id="a") == {"selected_id": target.id, "view_state": {"component": "By"}}
    assert store.link_scope("p", source_channel_id="a", target_channel_id="a") == {target.id: target.id}
    assert store.restore("p")[0].alias == "Standalone alias"


def test_completed_run_link_serializes_with_alias_and_session_edits_without_changing_evidence(tmp_path, monkeypatch):
    import coilgen.application.history as history_module

    first = write_run(tmp_path / "first")
    second = write_run(tmp_path / "second")
    store = RunHistoryStore(tmp_path / "session.json")
    entry = store.add_run("p", first, channel_id="a", alias="Before completion")
    evidence = {path: path.read_bytes() for run in (first, second) for path in run.directory.iterdir()}
    entered, release = Event(), Event()
    renamed, saved = Event(), Event()
    original_write = history_module.atomic_json

    def blocking_first_write(path, data):
        if path == store.path and not entered.is_set():
            entered.set()
            assert release.wait(5), "Test did not release the history manifest write."
        return original_write(path, data)

    def rename():
        renamed.set()
        store.rename_run(entry.id, "User's current alias")

    def save():
        saved.set()
        store.save_session("p", channel_id="a", selected_id=entry.id, view_state={"component": "By"})

    monkeypatch.setattr(history_module, "atomic_json", blocking_first_write)
    with ThreadPoolExecutor(max_workers=3) as pool:
        link = pool.submit(store.add_run, "p", second, channel_id="a")
        try:
            assert entered.wait(3)
            alias = pool.submit(rename)
            session = pool.submit(save)
            assert renamed.wait(3) and saved.wait(3)
            # Both same-instance edits must wait for the pending atomic write,
            # rather than being rejected as another runtime owner.
            for pending in (alias, session):
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.05)
        finally:
            release.set()
        assert link.result(timeout=3).verified
        alias.result(timeout=3)
        session.result(timeout=3)

    entries = RunHistoryStore(store.path).restore("p", channel_id="a")
    assert len(entries) == 2 and all(item.verified for item in entries)
    assert entries[0].id == entry.id and entries[0].alias == "User's current alias"
    assert store.session("p", channel_id="a") == {"selected_id": entry.id, "view_state": {"component": "By"}}
    assert {path: path.read_bytes() for path in evidence} == evidence


def test_same_instance_thread_lock_does_not_replace_external_history_ownership(tmp_path):
    from coilgen.research.storage import DirectoryOwner

    run = write_run(tmp_path / "run")
    store = RunHistoryStore(tmp_path / "session.json")
    entry = store.add_run("p", run, alias="Original alias")
    before = store.path.read_bytes()
    with DirectoryOwner(store.path.parent, filename=f".{store.path.name}.lock"):
        with pytest.raises(RuntimeError, match="Another runtime already owns"):
            store.rename_run(entry.id, "Cannot acquire an external owner's lock")
    assert store.path.read_bytes() == before
    store.rename_run(entry.id, "After lock release")
    assert store.restore("p")[0].alias == "After lock release"


def test_history_manifest_read_handle_finishes_before_concurrent_session_replacement(tmp_path, monkeypatch):
    store = RunHistoryStore(tmp_path / "session.json")
    store.save_session("p", view_state={"component": "Bz"})
    original_read = Path.read_text
    entered, release, saving = Event(), Event(), Event()

    def held_manifest_read(path, *args, **kwargs):
        if path == store.path and not entered.is_set():
            # Hold a real read handle as a background history restoration would.
            # On Windows it prevents os.replace until this context closes.
            with path.open("r", encoding=kwargs.get("encoding", "utf-8")) as stream:
                entered.set()
                assert release.wait(5), "Test did not release the manifest reader."
                return stream.read()
        return original_read(path, *args, **kwargs)

    def save():
        saving.set()
        store.save_session("p", view_state={"component": "By"})

    monkeypatch.setattr(Path, "read_text", held_manifest_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(store.restore, "p")
        try:
            assert entered.wait(3)
            writer = pool.submit(save)
            assert saving.wait(3)
            with pytest.raises(TimeoutError):
                writer.result(timeout=0.05)
        finally:
            release.set()
        assert reader.result(timeout=3) == ()
        writer.result(timeout=3)
    assert store.session("p") == {"selected_id": None, "view_state": {"component": "By"}}
