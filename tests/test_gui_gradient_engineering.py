"""Native engineering dialogs preserve verified geometry and complete specs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QPushButton

from coilgen.application import ProjectDocument, RunRecord
from coilgen.config import CoilGenConfig
from coilgen.gradient_engineering import (
    CircuitSpec,
    DriveSpec,
    GradientEngineeringSpec,
    InductanceMatrixSpec,
    ThermalSpec,
    winding_path_sha256,
)
from coilgen.gui.gradient_tools import CircuitDriveDialog, GradientRequirementsDialog


BODY = "part001_body0001"


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(application):
    opened = []

    def create(cls, *args):
        dialog = cls(*args)
        opened.append(dialog)
        return dialog

    yield create
    for dialog in opened:
        dialog.close()
        dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()


def _points(closed=False):
    if closed:
        return np.asarray([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0, 0, 0]], dtype=float)
    return np.asarray([[0, 0, 0], [0.1, 0, 0]], dtype=float)


def _circuit(points, *, closed=False, body=BODY):
    winding = points[1:] if closed else points
    driver = [-0.2, -0.2, 0]
    return CircuitSpec(
        body_id=body,
        expected_path_sha256=winding_path_sha256(points),
        supply_points_m=[driver, winding[0]],
        return_points_m=[winding[-1], [-0.3, -0.1, 0], driver],
        feed_break_segment=0 if closed else None,
        source="Saved engineering fixture",
        positive_terminal="Amplifier+",
        negative_terminal="Amplifier-",
    )


def _record(tmp_path, *, closed=False, count=1):
    arrays = {}
    metadata = {"winding_steps": [], "fabrication": {"paths": []}}
    for index in range(count):
        body = BODY if index == 0 else f"body{index}"
        arrays[f"review_conductor_{index}_points_m"] = _points(closed) + [0, 0, 0.2 * index]
        metadata["winding_steps"].append({"body_id": body})
        metadata["fabrication"]["paths"].append({"body_id": body, "closed": closed})
    return RunRecord(tmp_path / "verified-run", metadata, arrays)


def _project(engineering=None):
    project = ProjectDocument.from_config(CoilGenConfig(), name="Native gradient")
    return (
        project.update_study({"engineering": engineering.to_dict(), "fabricate": True, "pipeline_stage": "evaluated"})
        if engineering is not None
        else project
    )


def _spec(*, target=False, thermal=False, matrix=False, independent=False, circuits=None):
    circuits = circuits or (_circuit(_points()),)
    ids = tuple(value.body_id for value in circuits)
    extras = {}
    if thermal:
        extras["thermal"] = ThermalSpec(
            thermal_resistance_K_per_W=0.123456789012345,
            heat_capacity_J_per_K=123.456789012345,
            ambient_temperature_K=293.15,
            initial_temperature_K=294.1234567890123,
            max_temperature_K=340.9876543210123,
            source="Measured thermal fit",
            measured=True,
        )
    if matrix:
        extras["inductance_matrix"] = InductanceMatrixSpec(
            body_ids=ids,
            values_H=tuple(tuple(0.001 if i == j else 0 for j in range(len(ids))) for i in range(len(ids))),
            expected_electrical_path_sha256=("e" * 64,) * len(ids),
            source="Saved matrix source",
            verified=True,
        )
    if target:
        extras.update(
            target_gradient_T_per_m=0.04,
            target_slew_T_per_m_per_s=100,
            gradient_efficiencies_T_per_m_per_A=(0.0001,) * len(ids),
            gradient_efficiency_path_sha256=("e" * 64,) * len(ids),
            gradient_efficiency_source_sha256="a" * 64,
            gradient_efficiency_source="Saved independent physical-path fit",
        )
    else:
        extras.update(time_s=(0, 0.001, 0.002), currents_A=tuple(tuple(value for _ in ids) for value in (0, 2, 0)))
    drive = DriveSpec(
        body_ids=ids,
        current_limits_A=(3.123456789012345,) * len(ids),
        voltage_limits_V=(500.1234567890123,) * len(ids),
        source="Saved amplifier model",
        **extras,
    )
    return GradientEngineeringSpec(circuits=circuits, drive=drive, independent_winding_validation=independent)


@pytest.mark.parametrize(
    "target,thermal,matrix", [(False, False, False), (False, True, True), (True, False, True), (True, True, True)]
)
def test_opening_and_applying_preserves_complete_existing_spec(dialogs, tmp_path, target, thermal, matrix):
    original = _spec(target=target, thermal=thermal, matrix=matrix)
    project = _project(original)
    dialog = dialogs(CircuitDriveDialog, project, _record(tmp_path))
    assert dialog.value() == original.to_dict()
    assert project.study["engineering"] == original.to_dict()
    assert dialog.drive_mode.currentData() == ("target" if target else "waveform")
    assert dialog.thermal_enabled.isChecked() is thermal


def test_absent_engineering_remains_absent_without_edits(dialogs, tmp_path):
    project = _project()
    dialog = dialogs(CircuitDriveDialog, project, _record(tmp_path))
    assert dialog.value() is None and project.study.get("engineering") is None


def _circuit_fields(dialog):
    row = dialog.body_controls[0]
    return (
        row[3],
        *row[5:],
        dialog.rebind_controls[BODY],
        dialog.circuit_sources[BODY],
        *dialog.circuit_imports[BODY],
    )


@pytest.mark.parametrize("closed", [False, True])
def test_undefined_circuit_grays_fields_and_imports_and_ignores_incomplete_edits(
    dialogs, tmp_path, monkeypatch, closed
):
    dialog = dialogs(CircuitDriveDialog, _project(), _record(tmp_path, closed=closed))
    assert all(not control.isEnabled() for control in _circuit_fields(dialog))
    assert not dialog.body_controls[0][4].isEnabled()
    assert not dialog.drive_members[BODY].isEnabled()
    imported = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: imported.append(True))
    for button in dialog.circuit_imports[BODY]:
        button.click()
    assert not imported
    dialog.body_controls[0][7].setPlainText("Unfinished ignored circuit coordinates")
    assert dialog.value() is None


@pytest.mark.parametrize("closed", [False, True])
def test_new_circuit_definition_drive_and_closure_gate_native_controls(dialogs, tmp_path, closed):
    dialog = dialogs(CircuitDriveDialog, _project(), _record(tmp_path, closed=closed))
    defined, cut = dialog.body_controls[0][2], dialog.body_controls[0][4]
    member = dialog.drive_members[BODY]
    defined.setChecked(True)
    assert all(control.isEnabled() for control in _circuit_fields(dialog))
    assert cut.isEnabled() is closed
    assert member.isChecked() and not member.isEnabled()
    assert dialog._drive_body_ids() == [BODY]
    dialog.drive_enabled.setChecked(True)
    assert member.isEnabled() and all(control.isEnabled() for control in dialog.limit_by_body[BODY])
    defined.setChecked(False)
    assert not member.isChecked() and not member.isEnabled()
    assert not cut.isEnabled() and not dialog._drive_body_ids()
    assert all(not control.isEnabled() for control in _circuit_fields(dialog))
    assert all(not control.isEnabled() for control in dialog.limit_by_body[BODY])
    defined.setChecked(True)
    assert member.isChecked() and member.isEnabled()
    dialog.drive_enabled.setChecked(False)
    assert member.isChecked() and not member.isEnabled()
    assert all(control.isEnabled() for control in _circuit_fields(dialog))
    assert cut.isEnabled() is closed
    assert all(not control.isEnabled() for control in dialog.limit_by_body[BODY])


@pytest.mark.parametrize("target", [False, True])
def test_existing_circuit_deselection_clears_drive_membership_without_losing_saved_models(dialogs, tmp_path, target):
    original = _spec(target=target, thermal=True, matrix=True)
    dialog = dialogs(CircuitDriveDialog, _project(original), _record(tmp_path))
    defined, cut = dialog.body_controls[0][2], dialog.body_controls[0][4]
    member = dialog.drive_members[BODY]
    thermal_fields = (dialog.thermal_measured, dialog.thermal_source, *dialog.thermal_controls.values())
    assert member.isEnabled() and not cut.isEnabled()
    assert all(control.isEnabled() for control in thermal_fields)
    dialog.drive_enabled.setChecked(False)
    assert member.isChecked() and not member.isEnabled()
    assert all(not control.isEnabled() for control in thermal_fields)
    assert not dialog.retain_mutual.isEnabled() and not dialog.waveform.isEnabled()
    dialog.drive_enabled.setChecked(True)
    assert dialog.waveform.isEnabled() is not target
    assert dialog.value() == original.to_dict()
    defined.setChecked(False)
    assert not member.isChecked() and not member.isEnabled()
    assert all(not control.isEnabled() for control in _circuit_fields(dialog))
    assert all(not control.isEnabled() for control in dialog.limit_by_body[BODY])
    defined.setChecked(True)
    assert member.isEnabled() and not member.isChecked()
    member.setChecked(True)
    dialog.thermal_enabled.setChecked(False)
    assert all(not control.isEnabled() for control in thermal_fields)
    dialog.thermal_enabled.setChecked(True)
    assert all(control.isEnabled() for control in thermal_fields)
    assert dialog.value() == original.to_dict()


def test_noop_preserves_saved_circuit_and_drive_body_order(dialogs, tmp_path):
    record = _record(tmp_path, count=2)
    first = _circuit(record.array("review_conductor_0_points_m"))
    second = _circuit(record.array("review_conductor_1_points_m"), body="body1")
    spec = _spec(circuits=(second, first))
    dialog = dialogs(CircuitDriveDialog, _project(spec), record)
    assert dialog.value() == spec.to_dict()
    assert dialog._drive_body_ids() == ["body1", BODY]
    assert "body1, " + BODY in dialog.waveform_body_label.text()


def test_stale_circuit_reference_requires_explicit_rebinding(dialogs, tmp_path):
    stale = replace(_circuit(_points()), expected_path_sha256="f" * 64)
    project = _project(GradientEngineeringSpec(circuits=(stale,)))
    dialog = dialogs(CircuitDriveDialog, project, _record(tmp_path))
    with pytest.raises(ValueError, match="stale winding"):
        dialog.value()
    assert project.study["engineering"]["circuits"][0]["expected_path_sha256"] == "f" * 64
    dialog.rebind_controls[BODY].setChecked(True)
    assert dialog.value()["circuits"][0]["expected_path_sha256"] == winding_path_sha256(_points())


@pytest.mark.parametrize("closed", [False, True])
def test_new_feed_validates_open_or_closed_cut_before_applying(dialogs, tmp_path, closed):
    dialog = dialogs(CircuitDriveDialog, _project(), _record(tmp_path, closed=closed))
    circuit = _circuit(_points(closed), closed=closed)
    row = dialog.body_controls[0]
    row[2].setChecked(True)
    row[7].setPlainText("\n".join(",".join(str(v) for v in point) for point in circuit.supply_points_m))
    row[8].setPlainText("\n".join(",".join(str(v) for v in point) for point in circuit.return_points_m))
    if closed:
        with pytest.raises(ValueError, match="feed-break"):
            dialog.value()
        row[4].setValue(0)
    else:
        assert not row[4].isEnabled()
        row[4].setValue(0)
        with pytest.raises(ValueError, match="open winding"):
            dialog.value()
        row[4].setValue(-1)
    restored = GradientEngineeringSpec.from_dict(dialog.value())
    restored.circuits[0].construct_path(_points(closed), closed=closed)


def test_bad_endpoint_is_rejected_in_editor_without_accepting_or_changing_project(dialogs, tmp_path, monkeypatch):
    project = _project(GradientEngineeringSpec(circuits=(_circuit(_points()),)))
    original = project.study
    dialog = dialogs(CircuitDriveDialog, project, _record(tmp_path))
    dialog.body_controls[0][7].setPlainText("-0.2 -0.2 0\n0.3 0 0")
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, message: errors.append(message))
    dialog._accept()
    assert any("winding start" in error for error in errors)
    assert dialog.result() == QDialog.DialogCode.Rejected and dialog.result_value is None
    assert project.study == original


def test_enable_waveform_limits_and_measured_thermal_controls_serialize_full_snapshot(dialogs, tmp_path):
    spec = GradientEngineeringSpec(circuits=(_circuit(_points()),), independent_winding_validation=False)
    project = _project(spec)
    dialog = dialogs(CircuitDriveDialog, project, _record(tmp_path))
    dialog.drive_enabled.setChecked(True)
    dialog.drive_members[BODY].setChecked(True)
    dialog.waveform.setPlainText("0, 0\n0.001, 100\n0.004, 0")
    dialog.limit_by_body[BODY][0].setValue(120)
    dialog.limit_by_body[BODY][1].setValue(450)
    dialog.source.setText("Recorded amplifier operating envelope")
    dialog.thermal_enabled.setChecked(True)
    dialog.thermal_source.setText("Measured cooling pulse experiment")
    with pytest.raises(ValueError, match="measured"):
        dialog.value()
    dialog.thermal_measured.setChecked(True)
    dialog.thermal_controls["thermal_resistance_K_per_W"].setValue(0.2)
    dialog.thermal_controls["heat_capacity_J_per_K"].setValue(750)
    restored = GradientEngineeringSpec.from_dict(dialog.value())
    assert restored.circuits == spec.circuits and not restored.independent_winding_validation
    assert restored.drive.currents_A == ((0,), (100,), (0,))
    assert restored.drive.current_limits_A == (120,) and restored.drive.voltage_limits_V == (450,)
    assert restored.drive.thermal.measured and restored.drive.thermal.heat_capacity_J_per_K == 750
    assert project.study["engineering"] == spec.to_dict()


def test_target_mode_can_explicitly_change_to_waveform_while_retaining_matrix_and_thermal(dialogs, tmp_path):
    spec = _spec(target=True, thermal=True, matrix=True)
    dialog = dialogs(CircuitDriveDialog, _project(spec), _record(tmp_path))
    dialog.drive_mode.setCurrentIndex(dialog.drive_mode.findData("waveform"))
    dialog.waveform.setPlainText("0 0\n0.001 2")
    updated = GradientEngineeringSpec.from_dict(dialog.value())
    assert updated.drive.time_s == (0, 0.001)
    assert updated.drive.target_gradient_T_per_m is None
    assert updated.drive.inductance_matrix == spec.drive.inductance_matrix
    assert updated.drive.thermal == spec.drive.thermal
    dialog.retain_mutual.setChecked(False)
    assert GradientEngineeringSpec.from_dict(dialog.value()).drive.inductance_matrix is None


def test_headerless_csv_polyline_import_is_validated_against_saved_endpoint(dialogs, tmp_path, monkeypatch):
    spec = GradientEngineeringSpec(circuits=(_circuit(_points()),))
    dialog = dialogs(CircuitDriveDialog, _project(spec), _record(tmp_path))
    filename = tmp_path / "supply.csv"
    filename.write_text("-0.2,-0.2,0\n0,0,0\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(filename), ""))
    next(button for button in dialog.findChildren(QPushButton) if button.text() == "Import points…").click()
    assert dialog.value()["circuits"][0]["supply_points_m"] == [[-0.2, -0.2, 0], [0, 0, 0]]
    dialog.body_controls[0][7].setPlainText("x_m,y_m,z_m\n0,0,0\n")
    with pytest.raises(ValueError, match="numeric rows"):
        dialog.value()


def test_requirements_independent_toggle_preserves_existing_circuit_drive_and_thermal(dialogs):
    original = _spec(target=True, thermal=True, matrix=True, independent=False)
    project = _project(original)
    dialog = dialogs(GradientRequirementsDialog, project)
    assert dialog.value() == ({}, {})
    dialog.independent.setChecked(True)
    changes, study = dialog.value()
    assert changes == {} and study["fabricate"] and study["pipeline_stage"] == "evaluated"
    assert study["engineering"] == replace(original, independent_winding_validation=True).to_dict()
    dialog.independent.setChecked(False)
    assert dialog.value() == ({}, {})


def test_disabling_independent_validation_retains_study_boundary(dialogs):
    project = _project(GradientEngineeringSpec(independent_winding_validation=True))
    dialog = dialogs(GradientRequirementsDialog, project)
    dialog.independent.setChecked(False)
    changes, study = dialog.value()
    assert changes == {} and "pipeline_stage" not in study and "fabricate" not in study
    assert study["engineering"]["independent_winding_validation"] is False
