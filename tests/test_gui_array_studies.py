"""Array workflow captures input revisions and exposes honest evidence scopes."""

from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QFileDialog

from coilgen.application.array_studies import ArrayCase, ArrayStudySpec, CohortControls
from coilgen.gui.array_studies import ArrayCaseDialog, ArrayStudyDialog


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def _spec():
    return ArrayStudySpec((ArrayCase("case", [[0, 0, 0], [1, 0, 0]], [1e-6, 2e-6]),))


def test_registered_import_dialog_and_membership_edit(application, tmp_path):
    path = tmp_path / "map.csv"
    path.write_text("0,0,0,1\n1,0,0,2")
    editor = ArrayCaseDialog()
    editor.filename.setText(str(path))
    editor.name.setText("disturbance")
    editor.fields.setCurrentText("uT")
    editor.membership.setCurrentIndex(1)
    case = editor.value()
    np.testing.assert_allclose(case.desired_field_T, [-1e-6, -2e-6])
    assert case.membership == "held_out"
    editor.close()
    dialog = ArrayStudyDialog(None, saved_spec=_spec().to_dict())
    dialog.case_table.cellWidget(0, 1).setCurrentIndex(1)
    dialog.case_table.item(0, 3).setText("3")
    result = dialog.value()
    assert result.cases[0].membership == "held_out"
    assert result.cases[0].rms_limit_T == pytest.approx(3e-6)
    dialog.close()


def test_failed_refresh_retains_old_result_input_contract(application, tmp_path, monkeypatch):
    dialog = ArrayStudyDialog(None, saved_spec=_spec().to_dict())
    original = _spec()
    dialog._pending_spec = original
    result = {
        "scope": "fixture scalar field",
        "cases": [
            {
                "name": "case",
                "membership": "training",
                "rms_error_T": 1e-7,
                "peak_error_T": 2e-7,
                "currents_A": [1.0],
                "power_W": None,
                "saturated_channels": [False],
                "residual_T": [1e-7, 2e-7],
                "passed": None,
                "diagnostics": {"rank": 1, "channel_count": 1, "normalized_condition_number": 1.0},
            }
        ],
    }
    dialog._finished(result, "")
    assert dialog.samples.item(0, 4).text() == "0.2"  # largest residual first
    dialog._pending_spec = replace(original, cases=(replace(original.cases[0], desired_field_T=[999, 999]),))
    dialog._finished(None, "fixture failure")
    destination = tmp_path / "array.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(destination), ""))
    dialog.export_results()
    saved = json.loads(destination.read_text())
    assert saved["spec"]["cases"][0]["desired_field_T"] == [1e-6, 2e-6]
    assert dialog.result_table.item(0, 8).text() == "Unassessed"
    dialog.close()


def test_busy_study_cannot_delete_running_worker(application):
    from coilgen.gui.array_studies import _ArrayTask

    dialog = ArrayStudyDialog(None)
    dialog.worker = _ArrayTask(lambda _progress: {})
    dialog.reject()
    assert dialog.worker.cancel.is_set()
    assert "Cancellation requested" in dialog.status.text()
    dialog.worker = None
    dialog.close()


def test_cohort_controls_persist_and_individual_edits_survive_removal(application):
    spec = _spec()
    spec = replace(
        spec,
        cases=(*spec.cases, replace(spec.cases[0], name="other")),
        cohort_controls=CohortControls("sectorization", 3, 1.2, (1.0, 2.0, 3.0), 0.5),
    )
    dialog = ArrayStudyDialog(None, saved_spec=spec.to_dict())
    assert dialog.count.value() == 3
    assert dialog.strategy.currentData() == "sectorization"
    dialog.case_table.item(0, 3).setText("0.45")
    dialog.case_table.selectRow(1)
    dialog.remove_case()
    result = dialog.value()
    assert result.cases[0].rms_limit_T == pytest.approx(0.45e-6)
    assert result.cohort_controls.to_dict() == spec.cohort_controls.to_dict()
    dialog.close()
