"""Headless evidence availability and unchanged-source integrity contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.evidence import (
    EvidenceIntegrityError,
    REQUIRED_STEPS,
    SourceHealth,
    VerificationReport,
    dataset_capabilities,
    load_saved_request,
    read_verified_artifact,
    resolve_dataset,
    verified_run_metadata,
    verify_record,
    verify_run_source,
)
from coilgen.application.service import DesignService
from coilgen.application.bundles import RunArchive
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import result_metadata_digest


def _arrays():
    return {
        "vertices_m": np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
        "faces": np.asarray([[0, 1, 2]]),
        "stream_function_A": np.zeros(3),
        "target_points_m": np.asarray([[0, 0, 0.02], [0, 0.01, 0.03]]),
        "target_field_T": np.zeros((2, 3)),
        "sheet_field_T": np.zeros((2, 3)),
        "routed_field_T": np.zeros((2, 3)),
    }


def _saved_run(tmp_path, *, checksum=True, request=True):
    directory = tmp_path / "saved"
    directory.mkdir()
    np.savez(directory / "scientific_result.npz", **_arrays())
    names = ["scientific_result.npz"]
    if request:
        submitted = ResearchRequest(
            name="Saved winding", config={"levels": 7}, project_id="fixture-project", project_revision=3
        )
        atomic_json(directory / "request.json", submitted.to_dict())
        names.append("request.json")
    atomic_json(directory / "manifest.json", {"request_sha256": "fixture-request", "input_files": {}})
    result = {
        "request_sha256": "fixture-request",
        "project_id": "fixture-project",
        "project_revision": 3,
        "summary": {"accepted": False},
        "artifacts": [
            {
                "name": name,
                "kind": "npz" if name.endswith(".npz") else "request",
                "sha256": file_digest(directory / name),
            }
            for name in names
        ],
    }
    if checksum:
        result["metadata_sha256"] = result_metadata_digest(result)
    atomic_json(directory / "result.json", result)
    return directory


def test_metadata_only_record_verification_skips_array_decompression(tmp_path, monkeypatch):
    directory = _saved_run(tmp_path)
    record = DesignService.load_run(directory)

    def unexpected_array_read(_archive, _name):
        raise AssertionError("Metadata-only verification must not decompress scientific arrays")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", unexpected_array_read)
    verified = verify_record(record, array_names=())
    assert verified.arrays == {}
    assert verified.metadata == record.metadata


def test_metadata_only_record_verification_still_rejects_corrupt_arrays(tmp_path):
    directory = _saved_run(tmp_path)
    record = DesignService.load_run(directory)
    with (directory / "scientific_result.npz").open("ab") as archive:
        archive.write(b"changed after selection")
    with pytest.raises(EvidenceIntegrityError):
        verify_record(record, array_names=())


def test_evidence_module_imports_without_gui_runtime():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import coilgen.application.evidence; assert 'PySide6' not in sys.modules"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_machine_prerequisites_and_exact_current_variant():
    available = resolve_dataset(_arrays(), "routed_residual")
    assert available.available and available.required_step == ""
    original = resolve_dataset(_arrays(), "routed_residual", original=True)
    assert not original.available and not original.invalid
    assert original.required_step == "evaluated"
    assert "routed_field_at_stream_current_T" in original.reason
    assert resolve_dataset({}, "support").required_step == "support"
    assert resolve_dataset({}, "review_electrical").required_step == "fabrication_review"


@pytest.mark.parametrize("original", [False, True])
def test_gui_catalog_is_a_presentation_of_the_headless_catalog(original):
    from coilgen.gui.dataset_capabilities import REQUIRED_STAGES, dataset_capabilities as gui_catalog

    arrays = _arrays()
    arrays["route_0_points_m"] = np.asarray([[0, 0, 0], [np.nan, 0, 0]])
    headless = dataset_capabilities(arrays, original=original)
    gui = gui_catalog(arrays, original=original)
    assert headless.keys() == gui.keys() == REQUIRED_STEPS.keys()
    for key, actual in headless.items():
        shown = gui[key]
        assert (shown.available, shown.reason, shown.invalid) == (actual.available, actual.reason, actual.invalid)
        assert shown.required_stage == (REQUIRED_STAGES[key] if actual.required_step else "")


@pytest.mark.parametrize("checksum", [False, True])
def test_metadata_verification_is_lightweight_and_reports_legacy_strength(tmp_path, monkeypatch, checksum):
    directory = _saved_run(tmp_path, checksum=checksum)
    monkeypatch.setattr(
        np, "load", lambda *_args, **_kwargs: pytest.fail("Metadata inspection loaded numerical arrays")
    )
    metadata, manifest, report = verified_run_metadata(directory)
    assert metadata["summary"]["accepted"] is False
    assert manifest["request_sha256"] == report.request_sha256 == "fixture-request"
    assert report.verified and report.metadata_checked is checksum
    assert verify_run_source(directory) == report


@pytest.mark.parametrize("mutation", ["artifact", "metadata", "manifest"])
def test_source_verification_reports_corruption_without_repair(tmp_path, mutation):
    directory = _saved_run(tmp_path)
    path = (
        directory
        / {"artifact": "scientific_result.npz", "metadata": "result.json", "manifest": "manifest.json"}[mutation]
    )
    if mutation == "artifact":
        path.write_bytes(b"changed wire evidence")
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "metadata":
            value["summary"]["accepted"] = True
        else:
            value["request_sha256"] = "another-request"
        atomic_json(path, value)
    before = path.read_bytes()
    report = verify_run_source(directory)
    assert report.status == "failed" and report.error
    assert path.read_bytes() == before


def test_verified_metadata_uses_exact_hash_addressed_relocated_inputs(tmp_path):
    directory = _saved_run(tmp_path)
    original = tmp_path / "original-target.npz"
    original.write_bytes(b"owned target sample fixture")
    digest = file_digest(original)
    destination = directory / "portable_inputs" / digest
    destination.parent.mkdir()
    destination.write_bytes(original.read_bytes())
    atomic_json(
        directory / "manifest.json",
        {"request_sha256": "fixture-request", "input_files": {"target": {"path": str(original), "sha256": digest}}},
    )
    original.unlink()
    assert verified_run_metadata(directory)[2].verified
    destination.write_bytes(b"different target")
    assert verify_run_source(directory).status == "failed"


def test_source_health_is_sticky_across_aliases_and_cached_snapshots(tmp_path):
    directory = _saved_run(tmp_path)
    cached = DesignService.load_run(directory)
    alias = SimpleNamespace(directory=directory.parent / "saved" / ".")
    health = SourceHealth()
    assert health.report(cached).status == "unverified"
    health.mark_failed(directory, "Array checksum changed")
    assert health.error(alias) == health.error(cached) == "Array checksum changed"
    assert resolve_dataset(cached.arrays, "sheet").available
    assert health.error(cached) == "Array checksum changed"
    other = tmp_path / "another-run"
    health.mark_failed(other, "Another source failed")
    report = verify_run_source(cached)
    health.mark_verified(alias, report)
    assert health.report(cached) == report and health.error(cached) == ""
    assert health.error(other) == "Another source failed"


def test_failed_or_foreign_verification_cannot_clear_health(tmp_path):
    health = SourceHealth()
    health.mark_failed(tmp_path / "saved", "Selected source failed")
    for report in (
        VerificationReport(tmp_path / "saved", "failed", "Still failed"),
        VerificationReport(tmp_path / "other", "verified"),
        VerificationReport(tmp_path / "saved", "unverified"),
    ):
        with pytest.raises(ValueError, match="successful verification"):
            health.mark_verified(tmp_path / "saved", report)
    assert health.error(tmp_path / "saved") == "Selected source failed"


def test_selected_snapshot_reverification_ignores_job_bookkeeping(tmp_path):
    directory = _saved_run(tmp_path)
    atomic_json(directory / "job.json", {"created_at": "before"})
    selected = DesignService.load_run(directory)
    atomic_json(directory / "job.json", {"created_at": "after"})
    verified = verify_record(selected)
    assert verified.metadata["job"]["created_at"] == "after"
    assert verified.metadata["summary"]["accepted"] is False


def test_legitimately_rewritten_result_cannot_replace_selected_identity(tmp_path):
    directory = _saved_run(tmp_path)
    selected = DesignService.load_run(directory)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    result["project_revision"] = 4
    result["metadata_sha256"] = result_metadata_digest(result)
    atomic_json(directory / "result.json", result)
    assert verify_run_source(directory).verified
    assert verify_run_source(selected).status == "failed"
    with pytest.raises(EvidenceIntegrityError, match="changed after it was selected"):
        read_verified_artifact(selected, "request.json")
    with pytest.raises(EvidenceIntegrityError, match="changed after it was selected"):
        RunArchive.capture(selected)


def test_archive_checks_selected_identity_after_capturing_result_bytes(tmp_path, monkeypatch):
    selected = DesignService.load_run(_saved_run(tmp_path))
    original_read = Path.read_bytes
    result_path = selected.directory / "result.json"

    def changed_read(path):
        if path == result_path:
            result = json.loads(path.read_text(encoding="utf-8"))
            result["project_revision"] = 4
            result["metadata_sha256"] = result_metadata_digest(result)
            atomic_json(path, result)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(EvidenceIntegrityError, match="changed while its archive was being captured"):
        RunArchive.capture(selected)


def test_original_request_inspection_preserves_settings_without_solver_conversion(tmp_path, monkeypatch):
    selected = DesignService.load_run(_saved_run(tmp_path))
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Settings inspection loaded unrelated maps"))
    monkeypatch.setattr(
        ResearchRequest, "to_config", lambda *_args, **_kwargs: pytest.fail("Settings inspection invoked engine")
    )
    request = load_saved_request(selected)
    assert request["config"] == {"levels": 7}
    assert request["project_id"] == "fixture-project" and request["project_revision"] == 3
    assert read_verified_artifact(selected, "request.json") == (selected.directory / "request.json").read_bytes()


def test_missing_optional_original_request_is_unavailable_not_source_corruption(tmp_path):
    selected = DesignService.load_run(_saved_run(tmp_path, request=False))
    with pytest.raises(ValueError, match="no verified original request") as caught:
        load_saved_request(selected)
    assert not isinstance(caught.value, EvidenceIntegrityError)
    assert verify_run_source(selected).verified


def test_request_byte_read_checks_digest_after_verification(tmp_path, monkeypatch):
    selected = DesignService.load_run(_saved_run(tmp_path))
    original_read = Path.read_bytes
    request_path = selected.directory / "request.json"

    def changed_read(path):
        if path == request_path:
            path.write_bytes(b'{"name": "modified request", "config": {}}')
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(EvidenceIntegrityError, match="integrity check failed"):
        load_saved_request(selected)


@pytest.mark.parametrize("name", ["../outside.npz", "C:/outside.npz", "/outside.npz", "nested\\outside.npz"])
def test_artifact_reads_reject_nonportable_or_escaping_names(tmp_path, name):
    selected = DesignService.load_run(_saved_run(tmp_path))
    with pytest.raises(ValueError, match="portable|leave"):
        read_verified_artifact(selected, name)
