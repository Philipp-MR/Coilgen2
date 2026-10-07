"""Physical target controls preserve native fields and normalization behavior."""

from dataclasses import replace

import numpy as np
import pytest
import test_workbench_usability as usability
from PySide6.QtWidgets import QCheckBox, QComboBox, QDockWidget
from PySide6.QtCore import Qt

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.target import define_target_field

application = usability.application
windows = usability.windows


def test_linear_goal_opens_without_changing_high_precision_inputs(windows):
    project = ProjectDocument.from_config(replace(preset_config(), target_gradient_strength=0.023456789123456789))
    window = windows(project)
    window.select_node("target")
    assert window.findChild(QComboBox, "target_mode").currentText() == "Linear gradient"
    assert window.findChild(QComboBox, "linear_gradient_direction").currentData() == "y"
    assert window._draft_controls["field_shape_function"].parentWidget().isHidden()
    window.findChild(QCheckBox, "advanced_target_controls").setChecked(True)
    assert not window._draft_controls["field_shape_function"].parentWidget().isHidden()
    assert window.apply_settings()
    assert window.project is project


@pytest.mark.parametrize("axis", ("x", "z"))
def test_direction_changes_actual_bz_expression_and_normalization_axis(windows, axis):
    window = windows()
    window.select_node("target")
    direction = window.findChild(QComboBox, "linear_gradient_direction")
    direction.setCurrentIndex(direction.findData(axis))
    window._draft_controls["target_gradient_strength"].setValue(25)
    assert window.apply_settings()
    config = window.project.config
    assert config.field_shape_function == config.target_gradient_axis == axis
    field = define_target_field([], None, None, config)
    np.testing.assert_allclose(field.b[:, 2], 0.025 * field.coords[:, "xyz".index(axis)], atol=1e-18)
    np.testing.assert_array_equal(field.b[:, :2], 0)


def test_custom_expression_is_retained_when_returning_from_linear_goal(windows):
    project = ProjectDocument.from_config(
        replace(preset_config(), field_shape_function="1e-4 + 0.02*z", target_gradient_normalization="off")
    )
    window = windows(project)
    window.select_node("target")
    window.findChild(QComboBox, "target_mode").setCurrentText("Linear gradient")
    assert window.project.config.field_shape_function == "y"
    window.findChild(QComboBox, "target_mode").setCurrentText("Analytic expression")
    assert window.project.config.field_shape_function == project.config.field_shape_function
    assert window.project.config.target_gradient_normalization == "off"


def test_both_basic_and_expert_target_rows_fit_narrow_dock(windows, application):
    window = windows()
    window.select_node("target")
    window.show()
    dock = window.findChild(QDockWidget, "settings_dock")
    for expanded in (False, True):
        window.findChild(QCheckBox, "advanced_target_controls").setChecked(expanded)
        window.resizeDocks([dock], [285], Qt.Orientation.Horizontal)
        application.processEvents()
        assert window.settings.horizontalScrollBar().maximum() == 0
