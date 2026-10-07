"""Settings navigation and units refer to the model actually used by the solver."""

from types import SimpleNamespace

import numpy as np
import pytest

import test_expert_settings as expert_tests
from coilgen.application import ProjectDocument
from coilgen.application.history import HistoryEntry
from coilgen.config import CoilGenConfig
from coilgen.gui.history import HistoryPanel
from coilgen.gui.readiness import check_readiness, settings_node

application = expert_tests.application
dialogs = expert_tests.dialogs


@pytest.mark.parametrize(
    "key,node",
    [
        ("target_error_tolerance_percent", "optimization"),
        ("target_error_constraint_metric", "optimization"),
        ("target_error_constraint_acceptance_absolute_T", "optimization"),
        ("target_region_radius", "target"),
        ("secondary_target_mesh_file", "target"),
        ("cylinder_mesh_parameter_list[1]", "geometry"),
        ("skip_sweep", "winding"),
        ("skip_postprocessing", "winding"),
        ("skip_normal_shift", "winding"),
        ("min_turn_distance", "winding"),
        ("layout_terminal_strategy", "winding"),
        ("sweep_min_bend_radius", "winding"),
        ("sweep_insulation_thickness", "material"),
        ("conductor_mounting_mode", "material"),
        ("conductor dimensions", "material"),
        ("specific_conductivity_conductor", "material"),
    ],
)
def test_validation_messages_open_the_page_that_owns_the_setting(key, node):
    assert settings_node(key) == node


def _support_project(tmp_path, vertices):
    path = tmp_path / "support.npz"
    np.savez(path, schema="coilgen-mesh-v1", vertices_m=vertices, faces=np.array([[0, 1, 2]]))
    return ProjectDocument.from_config(
        CoilGenConfig(
            coil_mesh_file=str(path),
            field_shape_function="0",
            target_gradient_normalization="none",
            sf_opt_method="tikhonov",
        )
    )


@pytest.mark.parametrize(
    "vertices",
    [
        np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]),
        np.array([[0, 0, 0], [0.1, 0, 0], [0, 0, 0]]),
    ],
)
def test_exact_degenerate_support_faces_are_actionable_before_operator_assembly(tmp_path, vertices, monkeypatch):
    monkeypatch.setattr(
        "coilgen.physics.calculate_sensitivity_matrix",
        lambda *_args, **_kwargs: pytest.fail("Readiness assembled an operator"),
    )
    report = check_readiness(_support_project(tmp_path, vertices))
    issue = next(issue for issue in report.blocking if issue.key == "coil_mesh_file")
    assert issue.node == "geometry" and "triangle 1 has zero area" in issue.message
    assert "collinear" in issue.message


def test_readiness_does_not_invent_a_small_triangle_area_tolerance(tmp_path):
    project = _support_project(tmp_path, np.array([[0, 0, 0], [1e-20, 0, 0], [0, 1e-20, 0]]))
    assert check_readiness(project).ready


def test_display_unit_help_does_not_mislabel_degrees_or_millimetres_as_native_si(dialogs):
    dialog, state = dialogs()
    angle = dialog.fields["sweep_bend_arc_angle_deg"]
    length = dialog.fields["sweep_max_segment_length"]
    assert angle.control.placeholderText() == "Value in °"
    assert length.control.placeholderText() == "Value in mm"
    assert "Native stored default" in angle.toolTip()
    assert "SI JSON" not in dialog.windowTitle()
    initial = dialog.draft
    angle.control.setText("2.5")
    length.control.setText("1.25")
    assert dialog.draft["sweep_bend_arc_angle_deg"] == 2.5
    assert dialog.draft["sweep_max_segment_length"] == pytest.approx(0.00125)
    assert dialog.draft["target_gradient_strength"] == initial["target_gradient_strength"]
    assert not state["calls"]


def test_history_passed_scope_exposes_failed_sweep_in_tooltip(application, tmp_path):
    metadata = {
        "summary": {"scope": "continuous_sheet", "accepted": True, "sheet_checks_passed": True},
        "pipeline_sweeps": {"checks_passed": False, "body_count": 1},
        "fabrication": None,
    }
    entry = HistoryEntry("r", "p", None, "Baseline", tmp_path, "request", SimpleNamespace(metadata=metadata))
    panel = HistoryPanel()
    panel.set_records((entry,))
    assert panel.runs_table.horizontalHeaderItem(3).text() == "Verdict"
    assert panel.runs_table.item(0, 3).text() == "Passed"
    tooltip = panel.runs_table.item(0, 3).toolTip()
    assert "Saved validation scope: continuous_sheet" in tooltip
    assert "only to that scope" in tooltip
    assert "Sweep geometry: Failed" in tooltip
    assert "Fabrication review: Not run" in tooltip
    panel.close()
