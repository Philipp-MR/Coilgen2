"""Native authoring binds selected winding evidence and plots explicit response samples."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from coilgen.application import ProjectDocument, RunRecord
from coilgen.application.dynamic_studies import CircuitReference
from coilgen.application.spatial_scene import RouteAnchor, SpatialScene
from coilgen.config import CoilGenConfig
from coilgen.gradient_engineering import (
    CircuitSpec,
    DriveSpec,
    GradientEngineeringSpec,
    evaluate_gradient_engineering,
    winding_path_sha256,
)
from coilgen.gui.dynamic_studies import DynamicReviewDialog, DynamicStudyDialog, TracePlot
from coilgen.gui.gradient_tools import CircuitDriveDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(app):
    opened = []

    def create(cls, *args, **kwargs):
        dialog = cls(*args, **kwargs)
        opened.append(dialog)
        return dialog

    yield create
    for dialog in opened:
        dialog.close()
        dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def _fixture(tmp_path, *, closed=False):
    points = np.asarray(
        [[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0, 0, 0]] if closed else [[0, 0, 0], [0.1, 0, 0]],
        dtype=float,
    )
    start, finish = (points[1], points[0]) if closed else (points[0], points[-1])
    circuit = CircuitSpec(
        "primary",
        winding_path_sha256(points),
        [[-0.1, -0.1, 0], start],
        [finish, [-0.1, -0.1, 0]],
        "Recorded fixture",
        feed_break_segment=0 if closed else None,
    )
    drive = DriveSpec(("primary",), (10,), (100,), "Fixture pulse", time_s=(0, 1, 2), currents_A=((0,), (2,), (0,)))
    engineering = GradientEngineeringSpec((circuit,), drive)
    project = ProjectDocument.from_config(CoilGenConfig(), name="Dynamic fixture").update_study(
        {"engineering": engineering.to_dict(), "fabricate": True, "pipeline_stage": "evaluated"}
    )
    record = RunRecord(
        tmp_path / "selected",
        {
            "winding_steps": [{"body_id": "primary"}],
            "fabrication": {"paths": [{"body_id": "primary", "closed": closed}]},
        },
        {"review_conductor_0_points_m": points},
    )
    return project, record, engineering, points


def test_absent_study_stays_absent_and_cancelling_leaves_project(dialogs, tmp_path):
    project, record, engineering, _ = _fixture(tmp_path)
    dialog = dialogs(DynamicStudyDialog, project, record)
    assert dialog.value() is None
    assert dialog.engineering_value() == engineering.to_dict()
    dialog.reject()
    assert project.study.get("dynamic_analysis") is None


def test_imported_reference_updates_drive_without_trusting_verified_checkbox(dialogs, tmp_path):
    project, record, _, _ = _fixture(tmp_path)
    dialog = dialogs(DynamicStudyDialog, project, record)
    reference = CircuitReference(
        "inductance", ("primary",), (dialog.paths["primary"],), ((0.002,),), "External circuit export", "a" * 64, "FEM"
    )
    dialog.references["inductance"] = reference
    study = dialog.value()
    engineering = dialog.engineering_value()
    assert engineering["drive"]["inductance_matrix"]["verified"] is False
    assert study["inductance_reference"]["source_sha256"] == "a" * 64
    updated = project.update_study({"dynamic_analysis": study, "engineering": engineering})
    assert updated.revision == project.revision + 1
    assert updated.study["dynamic_analysis"]["inductance_reference"]["body_ids"] == ["primary"]
    dialog.references["inductance"] = CircuitReference(
        "inductance", ("primary",), ("b" * 64,), ((0.002,),), "Stale export", "a" * 64, "FEM"
    )
    with pytest.raises(ValueError, match="exact complete"):
        dialog.value()


def test_target_mode_is_explicit_signed_efficiency_and_existing_limit_contract(dialogs, tmp_path):
    project, record, _, _ = _fixture(tmp_path)
    dialog = dialogs(DynamicStudyDialog, project, record)
    dialog.references["efficiency"] = CircuitReference(
        "efficiency",
        ("primary",),
        (dialog.paths["primary"],),
        (-0.02,),
        "Field regression",
        "a" * 64,
        "Bz gradient",
        field_component="Bz",
        gradient_direction_world=(0, 0, 1),
    )
    dialog.target_drive.setChecked(True)
    dialog.gradient.setValue(0.04)
    dialog.slew.setValue(100)
    engineering = GradientEngineeringSpec.from_dict(dialog.engineering_value())
    assert engineering.drive.time_s == ()
    assert engineering.drive.gradient_efficiencies_T_per_m_per_A == (-0.02,)
    assert engineering.drive.current_limits_A == (10,)


def test_waveform_csv_import_with_units_and_finite_repetition_controls(dialogs, tmp_path, monkeypatch):
    project, record, _, _ = _fixture(tmp_path)
    dialog = dialogs(CircuitDriveDialog, project, record)
    filename = tmp_path / "pulse.csv"
    filename.write_text("time_ms,primary_mA\n0,0\n1,2000\n2,0\n")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(filename), ""))
    dialog.csv_time_unit.setCurrentText("ms")
    dialog.csv_current_unit.setCurrentText("mA")
    dialog._import_waveform()
    dialog.repetitions.setValue(3)
    dialog.rest_s.setValue(0.003)
    drive = GradientEngineeringSpec.from_dict(dialog.value()).drive
    assert drive.time_s == (0, 0.001, 0.002)
    assert drive.currents_A == ((0,), (2,), (0,))
    assert drive.repetitions == 3 and drive.rest_s == 0.003
    assert "SHA-256" in drive.source


def test_scene_anchor_button_initializes_editable_cut_terminals_and_closure(dialogs, tmp_path, monkeypatch):
    project, record, _, points = _fixture(tmp_path, closed=True)
    digest = winding_path_sha256(points)
    scene = SpatialScene(
        anchors=(
            RouteAnchor("Drive+", "positive_terminal", (-0.2, -0.1, 0), "primary", digest),
            RouteAnchor("Drive-", "negative_terminal", (-0.2, 0.1, 0), "primary", digest),
            RouteAnchor("Cut", "cut", (0.05, 0, 0), "primary", digest),
        )
    )
    project = project.update_study({"scene": scene.to_dict()})
    dialog = dialogs(CircuitDriveDialog, project, record)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    dialog._use_scene_anchors("primary")
    assert not warnings
    circuit = GradientEngineeringSpec.from_dict(dialog.value()).circuits[0]
    assert circuit.feed_break_segment == 0
    assert circuit.positive_terminal == "Drive+" and circuit.negative_terminal == "Drive-"
    assert circuit.drive_closure_points_m == ((-0.2, 0.1, 0), (-0.2, -0.1, 0))
    result = circuit.construct_path(points, closed=True)
    np.testing.assert_allclose(result["positive_terminal_xyz_m"], [-0.2, -0.1, 0])


def test_drive_review_renders_explicit_interval_voltages_and_can_select_body(dialogs, tmp_path, app):
    _, _, engineering, _ = _fixture(tmp_path)
    circuit = engineering.circuits[0]
    fabrication = SimpleNamespace(
        winding_steps=[
            {
                "body_id": "primary",
                "drive_feed_designed": True,
                "expected_winding_path_sha256": circuit.expected_path_sha256,
                "electrical_path_sha256": "e" * 64,
            }
        ],
        electrical_metrics={"bodies": [{"body_id": "primary", "resistance_ohm": 1, "inductance_H": 2}]},
    )
    review = evaluate_gradient_engineering(engineering, fabrication)
    dialog = dialogs(DynamicReviewDialog, {"engineering_review": review})
    dialog.show()
    dialog.trace_body.setCurrentIndex(1)
    app.processEvents()
    plots = dialog.findChildren(TracePlot)
    assert len(plots) == 2
    np.testing.assert_allclose(plots[1].x, [0, 1, 1, 2])
    np.testing.assert_allclose(plots[1].curves[0][1], [4, 6, -2, -4])
    pixmap = QPixmap(dialog.size())
    dialog.render(pixmap)
    assert not pixmap.isNull()


def test_schema_copy_uses_exact_path_identity(dialogs, tmp_path, app):
    project, record, _, _ = _fixture(tmp_path)
    dialog = dialogs(DynamicStudyDialog, project, record)
    dialog._copy_reference_schema()
    value = json.loads(app.clipboard().text())
    assert value["path_sha256"] == [dialog.paths["primary"]]
