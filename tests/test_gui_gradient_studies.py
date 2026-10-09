"""Focused gradient dialogs retain drafts, explicit units and study provenance."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QDialog

from coilgen.application.gradient_geometry import CylindricalGeometry
from coilgen.application.gradient_studies import (
    SubmittedCandidate,
    convergence_candidates,
    submit_convergence,
)
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.service import RunRecord
from coilgen.attribution import software_attribution
from coilgen.gui.gradient_tools import ConvergenceDialog, GradientGeometryDialog, MeasuredFieldDialog
from coilgen.research.jobs import JobManager, RuntimePolicy
from test_workbench_usability import application as application, windows as windows


def _project():
    return ProjectDocument.from_config(
        replace(preset_config("cylinder"), cylinder_mesh_parameter_list=(0.3, 0.08, 8, 3, 0, 0, 1, 0.7))
    )


def test_geometry_dialog_keeps_native_cylinder_azimuth_and_original_document(application):
    project = _project()
    original = project.to_dict()
    dialog = GradientGeometryDialog(project)
    try:
        geometry = dialog.value()
        assert geometry.primary.azimuth_rad == CylindricalGeometry.from_project(project).primary.azimuth_rad
        np.testing.assert_allclose(
            geometry.primary.mesh().vertices,
            CylindricalGeometry.from_project(project).primary.mesh().vertices,
            atol=1e-14,
        )
        dialog.axis.setText("0 0 0")
        with pytest.raises(ValueError, match="nonzero"):
            dialog.value()
        assert project.to_dict() == original
    finally:
        dialog.close()


@pytest.mark.parametrize(
    "key,text",
    (
        ("mesh_refinements", "1.5"),
        ("mesh_refinements", "4"),
        ("quadrature_orders", "3 3"),
        ("independent_resolutions", "129"),
    ),
)
def test_convergence_dialog_rejects_invalid_levels_without_generating_mesh(application, monkeypatch, key, text):
    import coilgen.application.gradient_studies as studies

    monkeypatch.setattr(studies, "convergence_candidates", lambda *_args, **_kwargs: pytest.fail("UI generated meshes"))
    dialog = ConvergenceDialog(_project())
    try:
        dialog.entries[key].setText(text)
        with pytest.raises(ValueError):
            dialog.value()
    finally:
        dialog.close()


def test_convergence_dialog_returns_explicit_levels_and_criterion_units(application):
    dialog = ConvergenceDialog(_project())
    try:
        levels, criteria = dialog.value()
        assert levels["mesh_refinements"] == (0, 1, 2)
        assert levels["quadrature_orders"] == (2, 4, 6)
        assert criteria == {"relative_tolerance": 0.01, "absolute_tolerance_T": 1e-7}
    finally:
        dialog.close()


def _measured_dialog(tmp_path, *, header=False):
    path = tmp_path / "measured.csv"
    path.write_text(("x_m,y_m,z_m,Bx_T,By_T,Bz_T\n" if header else "") + "0,0,.02,0,0,2e-5\n", encoding="utf-8")
    dialog = MeasuredFieldDialog(body_ids=("body0", "body1"))
    dialog.filename.setText(str(path))
    dialog.currents.setText("body0=2 body1=-1")
    dialog.source.setText("Independent probe scan; calibrated shunts")
    dialog.independent.setChecked(True)
    return dialog, path


@pytest.mark.parametrize("header", (False, True))
def test_measured_dialog_owns_accepted_source_bytes_and_all_body_calibration(application, tmp_path, header):
    dialog, path = _measured_dialog(tmp_path, header=header)
    try:
        dialog.rms_limit.setValue(2.5)
        dialog.peak_limit.setValue(8)
        value = dialog.value()
        original = path.read_bytes()
        path.write_text("changed after acceptance", encoding="utf-8")
        assert value["source_bytes"] == original
        assert value["currents_A"] == {"body0": 2, "body1": -1}
        assert value["independent"] is True
        assert value["rms_limit_T"] == pytest.approx(2.5e-6)
        assert value["peak_limit_T"] == pytest.approx(8e-6)
        assert value["rotation"] == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    finally:
        dialog.close()


@pytest.mark.parametrize(
    "bad", ("nonrigid", "reflection", "missing_body", "duplicate_body", "not_independent", "wrong_header_unit")
)
def test_measured_dialog_retains_invalid_input_and_does_not_accept(application, tmp_path, bad):
    dialog, _ = _measured_dialog(tmp_path, header=True)
    try:
        if bad == "nonrigid":
            dialog.rotation.setPlainText("2 0 0\n0 1 0\n0 0 1")
        elif bad == "reflection":
            dialog.rotation.setPlainText("-1 0 0\n0 1 0\n0 0 1")
        elif bad == "missing_body":
            dialog.currents.setText("body0=2")
        elif bad == "duplicate_body":
            dialog.currents.setText("body0=2 body0=-1 body1=1")
        elif bad == "not_independent":
            dialog.independent.setChecked(False)
        else:
            dialog.length_unit.setCurrentText("mm")
        with pytest.raises(ValueError):
            dialog.value()
        assert dialog.result() == QDialog.DialogCode.Rejected and dialog.result_value is None
    finally:
        dialog.close()


def test_real_queue_admission_retains_each_candidate_owner_controls_and_cancels_full_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "_monitor", lambda _self: None)
    candidates = convergence_candidates(_project(), quadrature_orders=(3,), independent_resolutions=(20,))
    with JobManager(tmp_path / "jobs", RuntimePolicy(max_pending=8)) as jobs:
        submitted = submit_convergence(candidates, jobs)
        assert len(submitted) == 3
        for item in submitted:
            persisted = json.loads((jobs.directory / item.job_id / "job.json").read_text(encoding="utf-8"))
            assert persisted["project_id"] == item.candidate.project.id
            assert persisted["project_fingerprint"] == item.candidate.project.fingerprint
            assert persisted["project_revision"] == item.candidate.project.revision
            assert persisted["validation_resolution"] == item.candidate.project.study["validation_resolution"]
    with JobManager(tmp_path / "limited", RuntimePolicy(max_pending=1)) as jobs:
        with pytest.raises(ValueError, match="queue is full"):
            submit_convergence(candidates, jobs)
        assert len(jobs.list()) == 1 and jobs.list()[0]["status"] == "cancelled"


def test_workbench_queues_in_task_and_reconstructs_manifest_without_duplicating_assets(windows, tmp_path, monkeypatch):
    import coilgen.application.gradient_studies as studies
    import coilgen.gui.gradient_tools as dialogs

    window = windows(_project())
    captured, generated = [], []
    levels = {"quadrature_orders": (3, 4)}
    criteria = {"relative_tolerance": 0.01, "absolute_tolerance_T": 1e-7}

    class Accepted:
        result_value = (levels, criteria)

        def __init__(self, *_args):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs, "ConvergenceDialog", Accepted)
    original = studies.convergence_candidates

    def generate(project, **controls):
        generated.append(project)
        return original(project, **controls)

    monkeypatch.setattr(studies, "convergence_candidates", generate)
    monkeypatch.setattr(
        studies,
        "submit_convergence",
        lambda candidates, _jobs: tuple(
            SubmittedCandidate(candidate, f"job{index}", {"id": f"job{index}"})
            for index, candidate in enumerate(candidates)
        ),
    )
    monkeypatch.setattr(
        window, "_gradient_task", lambda operation, callback, label: captured.append((operation, callback, label))
    )
    monkeypatch.setattr(window, "poll_jobs", lambda: None)
    monkeypatch.setattr(window, "_focus_job", lambda *_args, **_kwargs: None)
    window._gradient_convergence_dialog()
    assert not generated and len(captured) == 1
    operation, callback, _ = captured.pop()
    submitted, path = operation(window.project)
    callback((submitted, path))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert len(manifest["candidates"]) == 3 and "assets" in manifest["baseline"]
    assert all("assets" not in candidate for candidate in manifest["candidates"])
    assert window._session_job_ids >= {item.job_id for item in submitted}
    checks = []

    def summarize(candidates, records, *, criteria):
        checks.extend(candidates)
        assert criteria[0].absolute_tolerance == pytest.approx(0.1)  # µT
        assert set(records) == {item.candidate.project.id for item in submitted}
        return SimpleNamespace(changes=(), stable_axes=(), notes=(), to_dict=lambda: {"verified": True})

    monkeypatch.setattr(studies, "summarize_convergence", summarize)
    window._gradient_convergence_review()
    review, _callback, _label = captured.pop()
    review(window.project)
    assert json.loads(path.with_suffix(".review.json").read_text(encoding="utf-8")) == {
        "verified": True,
        "software": software_attribution(),
    }
    for candidate, original_candidate in zip(checks, submitted, strict=True):
        assert candidate.project.id == original_candidate.candidate.project.id
        assert candidate.project.revision == original_candidate.candidate.project.revision
        assert candidate.project.fingerprint == original_candidate.candidate.project.fingerprint


@pytest.mark.parametrize("kind,enabled", [("contour", False), ("routed", True)])
def test_measured_action_requires_saved_routed_or_electrical_winding(windows, tmp_path, monkeypatch, kind, enabled):
    window = windows(_project())
    record = RunRecord(tmp_path, {"wire_fields": {kind: {"available": True}}}, {})
    window.runs["saved"] = record
    window._selected_run = "saved"
    monkeypatch.setattr(window, "_run_verification_error", lambda _record: None)
    window._update_execution_feedback()
    assert window.gradient_actions["_gradient_measured_dialog"].isEnabled() is enabled


def test_malformed_owned_geometry_reports_error_without_replacing_inputs(windows, monkeypatch):
    import coilgen.gui.gradient_tools as dialogs

    window = windows(_project())
    project, count = window.project, window.undo_stack.count()
    errors = []

    def malformed(*_args):
        raise ValueError("Owned support metadata disagrees with its mesh")

    monkeypatch.setattr(dialogs, "GradientGeometryDialog", malformed)
    monkeypatch.setattr(window, "_error", errors.append)
    window._gradient_geometry_dialog()
    assert errors == ["Gradient support geometry: Owned support metadata disagrees with its mesh"]
    assert window.project is project and window.undo_stack.count() == count
