"""Structured research editors preserve inputs and expose scoped failure maps."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from coilgen.application.gradient_studies import RigidTransform
from coilgen.application.research_requirements import (
    DesiredDataset,
    PhysicalScenario,
    RegionalRequirement,
    ResearchRequirements,
)
from coilgen.gui.research_requirements import (
    PhysicalScenarioDialog,
    RegionalRequirementDialog,
    ResearchRequirementsDialog,
    ResearchRequirementsReviewDialog,
    TransformEditor,
)
from coilgen.target_regions import TargetRegion


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(application):
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
    application.processEvents()


def requirement():
    geometry = TargetRegion(
        "box", size_m=(0.02, 0.03, 0.04), center_m=(0.01, 0.02, 0.03), rotation_axis=(0, 0, 1), rotation_angle_rad=0.3
    )
    design = DesiredDataset.analytic(geometry, ("0", "0", ".04*x"), resolution=3)
    validation = DesiredDataset.analytic(geometry, ("0", "0", ".04*x"), resolution=4, independent=True)
    return RegionalRequirement("Research ROI", design, "Bz", 1.234567890123e-7, 2e-7, 0.123, validation=validation)


def test_regional_editor_reopens_analytic_recipe_without_altering_owned_samples(dialogs):
    original = requirement()
    dialog = dialogs(RegionalRequirementDialog, original)
    actual = dialog.value()
    assert dialog.source.currentIndex() == 0
    assert dialog.validation_mode.currentIndex() == 2
    assert actual.to_dict() == original.to_dict()
    dialog.expressions[2].setText("-.04*x")
    changed = dialog.value()
    np.testing.assert_allclose(changed.design.data.field[:, 2], -original.design.data.field[:, 2])
    np.testing.assert_allclose(changed.validation.data.field[:, 2], -original.validation.data.field[:, 2])


def test_transform_editor_preserves_rotation_exactly_until_user_edits(dialogs):
    from scipy.spatial.transform import Rotation

    rotation = RigidTransform(
        tuple(map(tuple, Rotation.from_rotvec((0.1234567890123, 0.2, -0.15)).as_matrix())),
        (0.12345678901234567, 0.002, -0.001),
    )
    editor = dialogs(TransformEditor, rotation)
    assert editor.value() is rotation
    editor.translation.setText("0.01 0.02 0.03")
    actual = editor.value()
    np.testing.assert_allclose(actual.translation_m, (0.01, 0.02, 0.03))
    np.testing.assert_allclose(actual.rotation, rotation.rotation, atol=2e-13)


def test_scenario_editor_authors_physical_motion_and_heldout_membership(dialogs):
    dialog = dialogs(PhysicalScenarioDialog)
    dialog.name.setText("Phantom displaced")
    dialog.membership.setCurrentIndex(1)
    dialog.samples.translation.setText("0.001 0 0")
    actual = dialog.value()
    assert actual.membership == "validation"
    assert actual.sample_transform.translation_m == (0.001, 0.0, 0.0)
    assert actual.support_transform == RigidTransform()


def test_overview_canonical_roundtrip_and_validation_are_transactional(dialogs):
    pose = PhysicalScenario("Placement error", support_transform=RigidTransform(translation_m=(0.001, 0, 0)))
    spec = ResearchRequirements((requirement(),), (pose,))
    dialog = dialogs(ResearchRequirementsDialog, spec)
    assert dialog.value().digest == spec.digest
    assert "independent" in dialog.lists["regions"].item(0).text()
    dialog.scenarios.append(PhysicalScenario("Wrong heldout", "validation", support_transform=pose.support_transform))
    with pytest.raises(ValueError, match="repeat training"):
        dialog.value()
    assert spec.digest == ResearchRequirements.from_dict(spec.to_dict()).digest


def test_review_only_plots_available_saved_arrays_and_shows_witness(dialogs):
    points = np.array([[0, 0, 0.02], [0.01, 0, 0.02]])
    report = {
        "rows": [
            {
                "region": "ROI",
                "scenario": "Nominal",
                "scenario_membership": "design",
                "sampling": "design",
                "scope": "saved_routed_centroids",
                "available": True,
                "passed": False,
                "component": "Bz",
                "array_prefix": "requirements_0000",
                "rms_error_T": 2e-6,
                "peak_error_T": 3e-6,
                "rms_limit_T": 1e-6,
                "peak_limit_T": 2e-6,
                "peak_point_m": points[1].tolist(),
                "peak_sample_index": 1,
                "failed_sample_indices": [1],
                "dataset_sha256": "abc",
                "current_policy": "unchanged_saved_current_no_refit",
            },
            {
                "region": "Sensor",
                "scenario": "Nominal",
                "scenario_membership": "design",
                "sampling": "independent_validation",
                "scope": "complete_electrical_paths",
                "available": False,
                "passed": None,
                "reason": "Feed unavailable.",
            },
        ]
    }
    arrays = {
        "requirements_0000_points_m": points,
        "requirements_0000_field_T": [[0, 0, 1e-6], [0, 0, 3e-6]],
        "requirements_0000_target_T": np.zeros((2, 3)),
    }
    dialog = dialogs(ResearchRequirementsReviewDialog, report, arrays)
    assert "sample 1" in dialog.witness.text()
    assert len(dialog.figure.axes) == 2
    dialog.table.selectRow(1)
    assert dialog.witness.text() == "Feed unavailable."
    assert not dialog.figure.axes
