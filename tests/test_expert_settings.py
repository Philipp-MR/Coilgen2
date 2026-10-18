"""Engineering config edits remain explicit, atomic, and scientifically portable."""

from copy import deepcopy
from dataclasses import fields
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.config import OptimizationSettings
from coilgen.gui.expert_settings import ExpertSettingsDialog
from coilgen.research.convex import ResearchPhysicsSpec


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["expert-settings-test"])


@pytest.fixture
def dialogs(application):
    created = []

    def make(config=None, callback=None):
        document = ProjectDocument.from_config(config or preset_config())
        state = {"document": document, "calls": []}

        def commit(changes, label):
            state["calls"].append((deepcopy(changes), label))
            state["document"] = state["document"].update_config(changes)
            return True

        dialog = ExpertSettingsDialog(config or preset_config(), callback or commit)
        created.append(dialog)
        return dialog, state

    yield make
    for dialog in created:
        dialog.reject()
        dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_search_matches_native_keys_labels_and_engineering_descriptions(dialogs):
    dialog, _ = dialogs()
    dialog.search.setText("sweep_insulation")
    assert not dialog.fields["sweep_insulation_thickness"].isHidden()
    assert dialog.fields["sweep_min_bend_radius"].isHidden()
    dialog.search.setText("bend radius")
    assert not dialog.fields["sweep_min_bend_radius"].isHidden()
    dialog.search.setText("triangulation")
    assert not dialog.fields["inter_group_connection_path_method"].isHidden()
    dialog.search.setText("nonexistent_parameter")
    assert not dialog.no_match.isHidden()
    assert all(group.isHidden() for group in dialog.groups.values())
    dialog.search.setText("")
    assert dialog.no_match.isHidden()


def test_typed_units_are_drafts_then_commit_through_project_validation(dialogs):
    dialog, state = dialogs()
    original = state["document"]
    dialog.fields["sweep_min_bend_radius"].control.setText("4.25")
    dialog.fields["conductor_mounting_mode"].control.setCurrentText("centered")
    assert not state["calls"]
    assert state["document"] is original
    parsed = json.loads(dialog.json_editor.toPlainText())
    assert parsed["sweep_min_bend_radius"] == pytest.approx(0.00425)
    assert dialog.apply_changes()
    assert state["document"].config.sweep_min_bend_radius == pytest.approx(0.00425)
    assert state["document"].config.conductor_mounting_mode == "centered"
    assert set(state["calls"][0][0]) == {"sweep_min_bend_radius", "conductor_mounting_mode"}
    assert dialog.apply_changes()
    assert len(state["calls"]) == 1  # Applying an unchanged draft is a no-op.


def test_regular_routing_policy_and_percent_budget_round_trip_without_changing_field_gates(dialogs):
    dialog, state = dialogs()
    config = state["document"].config
    assert dialog.fields["joint_group_routing_selection_policy"].control.currentText() == "regular"
    assert float(dialog.fields["joint_group_routing_regular_field_tolerance_rel"].control.text()) == pytest.approx(0.1)
    dialog.fields["joint_group_routing_selection_policy"].control.setCurrentText("physical")
    dialog.fields["joint_group_routing_regular_field_tolerance_rel"].control.setText("0.025")
    assert dialog.apply_changes()
    updated = state["document"].config
    assert updated.joint_group_routing_selection_policy == "physical"
    assert updated.joint_group_routing_regular_field_tolerance_rel == pytest.approx(0.00025)
    assert updated.joint_group_routing_max_field_change_rel == config.joint_group_routing_max_field_change_rel
    assert updated.joint_group_routing_field_acceptance == config.joint_group_routing_field_acceptance
    assert updated.routing_settings.joint_group_routing_regular_field_tolerance_rel == pytest.approx(0.00025)
    parsed = json.loads(dialog.json_editor.toPlainText())
    assert parsed["joint_group_routing_selection_policy"] == "physical"


def test_json_and_typed_controls_synchronize_without_resetting_omitted_fields(dialogs):
    dialog, state = dialogs()
    original = dialog.draft
    dialog.tabs.setCurrentIndex(1)
    dialog.json_editor.setPlainText(json.dumps({"sweep_min_bend_radius": 0.003, "levels": 15}))
    dialog.tabs.setCurrentIndex(0)
    assert float(dialog.fields["sweep_min_bend_radius"].control.text()) == pytest.approx(3)
    assert dialog.draft["levels"] == 15
    assert dialog.draft["cylinder_mesh_parameter_list"] == original["cylinder_mesh_parameter_list"]
    dialog.fields["sweep_insulation_thickness"].control.setText("0.2")
    assert json.loads(dialog.json_editor.toPlainText())["levels"] == 15
    assert dialog.apply_changes()
    assert state["document"].config.levels == 15
    assert state["document"].config.sweep_insulation_thickness == pytest.approx(0.0002)


def test_json_native_override_stays_complete_after_apply(dialogs):
    dialog, state = dialogs()
    dialog.tabs.setCurrentIndex(1)
    dialog.json_editor.setPlainText(json.dumps({"levels": 17}))
    assert dialog.apply_changes()
    complete = json.loads(dialog.json_editor.toPlainText())
    assert complete["levels"] == 17
    assert complete["coil_mesh_file"] == state["document"].config.coil_mesh_file
    dialog.search.setText("field_shape_function")
    dialog.find_in_json()
    assert dialog.json_editor.textCursor().selectedText() == "field_shape_function"


def test_invalid_typed_and_json_inputs_retain_draft_without_any_commit(dialogs):
    dialog, state = dialogs()
    dialog.fields["conductor_cross_section_width"].control.setText("-1")
    assert not dialog.apply_changes()
    assert "conductor" in dialog.error_label.text().lower()
    assert not dialog.fields["conductor_cross_section_width"].error.isHidden()
    assert dialog.fields["conductor_cross_section_width"].control.text() == "-1"
    assert not state["calls"]
    dialog.fields["conductor_cross_section_width"].control.setText("not a number")
    assert not dialog.apply_changes()
    dialog.tabs.setCurrentIndex(1)
    assert dialog.tabs.currentIndex() == 0
    dialog.fields["conductor_cross_section_width"].control.setText("1.5")
    dialog.tabs.setCurrentIndex(1)
    dialog.json_editor.setPlainText("{ incomplete")
    assert not dialog.apply_changes()
    assert dialog.json_editor.toPlainText() == "{ incomplete"
    dialog.tabs.setCurrentIndex(0)
    assert dialog.tabs.currentIndex() == 1
    assert not state["calls"]
    dialog.json_editor.setPlainText(json.dumps({"sweep_max_frame_rotation_deg": 181}))
    assert not dialog.apply_changes()
    assert "180" in dialog.error_label.text()
    assert not state["calls"]


def test_nested_physical_bound_preserves_custom_constraints_and_policy(dialogs):
    config = preset_config()
    config.research_physics = {
        "objective": "mixture",
        "objective_weights": {"power": 2, "magnetic_energy": 1},
        "objective_scales": {"power": 3.2, "magnetic_energy": 0.005},
        "b0_T": [1, 0, 3],
        "torque_origin_m": [0.01, 0.02, 0.03],
        "force_equality_N": [0, 0, 0],
        "torque_equality_Nm": [0, 0, 0],
        "field_constraints": [],
        "scenarios": [],
        "acceptance_absolute_by_unit": {"T": 2e-10, "N": 3e-8},
    }
    physics = deepcopy(config.research_physics)
    dialog, state = dialogs(config)
    row = dialog.fields["research_physics.magnetic_energy_limit_J"]
    row.enabled_bound.setChecked(True)
    assert not dialog.apply_changes()  # An enabled empty bound cannot become zero silently.
    row.control.setText("2.5")
    assert dialog.apply_changes()
    actual = state["document"].config.research_physics
    assert actual["magnetic_energy_limit_J"] == pytest.approx(0.0025)
    for key, value in physics.items():
        assert actual[key] == value
    assert dialog.reset_field("research_physics.magnetic_energy_limit_J")
    assert dialog.apply_changes()
    assert state["document"].config.research_physics["magnetic_energy_limit_J"] is None
    assert (
        state["document"].config.research_physics["acceptance_absolute_by_unit"]
        == physics["acceptance_absolute_by_unit"]
    )


def test_reset_is_explicit_and_affects_only_selected_draft_field(dialogs):
    config = preset_config()
    config.sweep_insulation_thickness = 0.0003
    config.sweep_min_bend_radius = 0.005
    config.conductor_cross_section_width = 0.001234567891234567
    dialog, state = dialogs(config)
    assert "Native stored default" in dialog.fields["sweep_insulation_thickness"].toolTip()
    assert dialog.reset_field("sweep_insulation_thickness")
    assert not state["calls"]
    assert dialog.draft["sweep_insulation_thickness"] == 0
    assert dialog.draft["sweep_min_bend_radius"] == config.sweep_min_bend_radius
    assert dialog.draft["conductor_cross_section_width"] == config.conductor_cross_section_width
    assert dialog.apply_changes()
    assert set(state["calls"][0][0]) == {"sweep_insulation_thickness"}


def test_native_to_physical_solver_is_an_explicit_draft_action(dialogs):
    config = preset_config()
    config.research_physics = None
    config.sf_opt_method = "tikhonov"
    dialog, state = dialogs(config)
    assert not dialog.fields["research_physics.force_limit_N"].isEnabled()
    assert not dialog.reset_field("research_physics.force_limit_N")
    assert dialog.draft["research_physics"] is None
    dialog.configure_physics.click()
    assert dialog.draft["research_physics"] == {"objective": "power"}
    assert not state["calls"]
    assert dialog.fields["research_physics.force_limit_N"].isEnabled()
    assert dialog.apply_changes()
    assert state["document"].config.research_physics == {"objective": "power"}
    assert state["document"].config.sf_opt_method == "tikhonov"  # Existing backend gives physics precedence.


def test_callback_rejection_keeps_exact_draft_and_dialog_open(dialogs):
    calls = []
    dialog, _ = dialogs(callback=lambda changes, label: calls.append((changes, label)) or False)
    dialog.fields["sweep_min_bend_radius"].control.setText("7")
    assert not dialog.apply_changes()
    assert dialog.draft["sweep_min_bend_radius"] == pytest.approx(0.007)
    assert "rejected" in dialog.error_label.text().lower()
    assert len(calls) == 1
    assert dialog.result() == 0


def test_unedited_scientific_values_and_unrepresented_json_settings_survive(dialogs):
    config = preset_config()
    config.sweep_insulation_thickness = 1.234567891234567e-18
    config.sweep_max_chord_error = 1.234567891234567e6
    config.layout_manufacturing_intent = "custom process"
    dialog, state = dialogs(config)
    dialog.fields["sweep_min_bend_radius"].control.setText("3")
    assert dialog.apply_changes()
    actual = state["document"].config
    assert actual.sweep_insulation_thickness == config.sweep_insulation_thickness
    assert actual.sweep_max_chord_error == config.sweep_max_chord_error
    assert actual.layout_manufacturing_intent == config.layout_manufacturing_intent


def test_unsupported_loaded_enum_is_retained_until_explicit_reset(application):
    config = preset_config()
    config.sweep_bend_mode = "future_custom_mode"
    calls = []
    dialog = ExpertSettingsDialog(config, lambda changes, label: calls.append(changes) or True)
    try:
        assert dialog.fields["sweep_bend_mode"].control.currentText() == "future_custom_mode"
        assert dialog.draft["sweep_bend_mode"] == "future_custom_mode"
        assert not dialog.apply_changes()
        assert not calls
        assert dialog.reset_field("sweep_bend_mode")
        assert dialog.apply_changes()
        assert calls == [{"sweep_bend_mode": "miter"}]
    finally:
        dialog.reject()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_draft_signal_tracks_typed_incomplete_reset_and_apply_states(dialogs):
    config = preset_config()
    config.sweep_insulation_thickness = 0
    dialog, _ = dialogs(config)
    received = []
    dialog.draft_changed.connect(received.append)
    dialog.fields["sweep_insulation_thickness"].control.setText("0.2")
    assert received[-1] is True and dialog.pending_changes
    dialog.close()
    assert received[-1] is True and dialog.pending_changes
    assert dialog.reset_field("sweep_insulation_thickness")
    assert received[-1] is False and not dialog.pending_changes
    dialog.fields["sweep_min_bend_radius"].control.setText("1e")
    assert received[-1] is True and dialog.pending_changes
    assert not dialog.apply_changes() and received[-1] is True
    dialog.fields["sweep_min_bend_radius"].control.setText("4")
    assert dialog.apply_changes()
    assert received[-1] is False and not dialog.pending_changes


def test_draft_signal_tracks_json_error_and_noop_merge(dialogs):
    dialog, _ = dialogs()
    received = []
    dialog.draft_changed.connect(received.append)
    dialog.tabs.setCurrentIndex(1)
    dialog.json_editor.setPlainText("{ incomplete")
    assert received[-1] is True
    assert not dialog.apply_changes() and received[-1] is True
    dialog.json_editor.setPlainText("{}")
    dialog.tabs.setCurrentIndex(0)
    assert received[-1] is False and not dialog.pending_changes


def test_rejected_expert_apply_keeps_the_draft_signal_pending(dialogs):
    dialog, _ = dialogs(callback=lambda *_args: False)
    received = []
    dialog.draft_changed.connect(received.append)
    dialog.fields["sweep_min_bend_radius"].control.setText("4")
    assert not dialog.apply_changes()
    assert received[-1] is True and dialog.pending_changes


def test_every_native_optimization_and_research_physics_field_has_a_control(dialogs):
    config = preset_config()
    config.research_physics = None
    dialog, _ = dialogs(config)
    assert {field.name for field in fields(OptimizationSettings)} <= dialog.fields.keys()
    assert {"research_physics." + field.name for field in fields(ResearchPhysicsSpec)} <= dialog.fields.keys()
    for key in (
        "optimization_solver_residual_tolerance",
        "l_curve_factor_count",
        "l_curve_factor_min_multiplier",
        "l_curve_factor_max_multiplier",
        "l_curve_regularization_factors",
        "l_curve_curvature_smoothing_window",
        "preoptimized_stream_function_match_tolerance",
        "sf_source_file",
    ):
        assert key in dialog.fields
    assert dialog.fields["gradient_slice_map_grid_size"].spec.kind == "json"
    assert dialog.fields["research_physics.energy"].spec.kind == "json"
    assert dialog.fields["preoptimized_stream_function_file"].spec.kind == "string"
    assert dialog.draft["research_physics"] is None


def test_focus_apis_preserve_original_tab_indices_and_never_activate_physics(dialogs):
    dialog, _ = dialogs()
    original = dialog.draft
    dialog.search.setText("insulation")
    assert dialog.focus_optimization()
    assert dialog.tabs.currentIndex() == 0 and dialog.optimization_only.isChecked()
    assert dialog.search.text() == ""
    assert not dialog.fields["sf_opt_method"].isHidden()
    assert not dialog.fields["research_physics.objective_weights"].isHidden()
    assert dialog.fields["sweep_insulation_thickness"].isHidden()
    dialog.search.setText("constrained_optimization_solver")
    assert not dialog.fields["constrained_optimization_solver"].isHidden()
    assert dialog.fields["tikonov_reg_factor"].isHidden()
    assert dialog.focus_native_settings() and dialog.tabs.currentIndex() == 1
    assert json.loads(dialog.json_editor.toPlainText()) == original
    assert dialog.draft == original and not dialog.pending_changes
    assert dialog.focus_optimization() and dialog.tabs.currentIndex() == 0


def test_solver_objective_basis_and_all_scalar_tuning_commit_without_hidden_resets(dialogs):
    config = preset_config()
    config.research_physics = None
    dialog, state = dialogs(config)
    original = dialog.draft
    changes = {
        "sf_opt_method": "constrained_quadratic",
        "optimization_solver": "pinv",
        "constrained_optimization_solver": "scs",
        "constrained_quadratic_fallback_policy": "strict",
        "constrained_quadratic_objective_mode": "resistance",
        "regularization_selection_method": "fixed",
        "stream_function_basis": "cotangent_spectral_lowpass",
    }
    for key, value in changes.items():
        dialog.fields[key].control.setCurrentText(value)
    dialog.fields["constrained_optimization_tolerance"].control.setText("1.234567891234567e-9")
    dialog.fields["constrained_optimization_max_iterations"].control.setText("31001")
    dialog.fields["constrained_optimization_threads"].control.setText("2")
    dialog.fields["constrained_optimization_verbose"].control.setChecked(True)
    dialog.fields["l_curve_regularization_factors"].control.setPlainText("[0.01, 0.1, 1.0]")
    assert not state["calls"] and dialog.draft["research_physics"] is None
    assert dialog.apply_changes()
    actual = state["document"].config
    for key, value in changes.items():
        assert getattr(actual, key) == value
    assert actual.constrained_optimization_tolerance == 1.234567891234567e-9
    assert actual.constrained_optimization_max_iterations == 31001 and actual.constrained_optimization_threads == 2
    assert actual.constrained_optimization_verbose and list(actual.l_curve_regularization_factors) == [0.01, 0.1, 1.0]
    assert actual.cylinder_mesh_parameter_list == original["cylinder_mesh_parameter_list"]


def test_nested_json_controls_preserve_every_other_physics_key_and_native_sequence(dialogs):
    config = preset_config()
    config.research_physics = {
        "objective": "mixture",
        "objective_weights": {"power": 2.0, "magnetic_energy": 1.0},
        "objective_scales": {"power": 3.2, "magnetic_energy": 0.005},
        "b0_T": [0.0, 0.0, 3.0],
        "energy": {"initial_order": 2},
        "force_equality_N": [0.0, 0.0, 0.0],
    }
    dialog, state = dialogs(config)
    original = deepcopy(config.research_physics)
    dialog.fields["research_physics.objective_weights"].control.setPlainText('{"power": 4.125, "minimax": 0.25}')
    dialog.fields["research_physics.b0_T"].control.setPlainText("[0.1, -0.2, 2.5]")
    dialog.fields["gradient_slice_map_grid_size"].control.setPlainText("[64, 80]")
    dialog.fields["contourability_repair_phase_offsets"].control.setPlainText("[0.0, 0.1234567891234567]")
    assert dialog.draft["research_physics"]["objective_weights"] == {"power": 4.125, "minimax": 0.25}
    assert dialog.draft["research_physics"]["energy"] == original["energy"]
    assert dialog.draft["research_physics"]["objective_scales"] == original["objective_scales"]
    assert dialog.apply_changes()
    actual = state["document"].config
    assert actual.research_physics["b0_T"] == [0.1, -0.2, 2.5]
    assert actual.research_physics["force_equality_N"] == original["force_equality_N"]
    assert list(actual.gradient_slice_map_grid_size) == [64, 80]
    assert actual.contourability_repair_phase_offsets[1] == 0.1234567891234567


def test_optional_scalar_and_native_json_values_accept_null_without_inferring_physics(dialogs):
    dialog, _ = dialogs()
    assert dialog.reset_field("research_physics")
    assert dialog.draft["research_physics"] is None
    assert "disabled" in dialog.fields["research_physics"].toolTip()
    dialog.configure_physics.click()
    row = dialog.fields["research_physics.force_equality_N"]
    row.enabled_bound.setChecked(True)
    assert not dialog.apply_changes()
    row.control.setPlainText("[0, 0, 0]")
    assert dialog.draft["research_physics"]["force_equality_N"] == [0, 0, 0]
    row.control.setPlainText("null")
    assert dialog.draft["research_physics"]["force_equality_N"] is None
    assert dialog.reset_field("research_physics") and dialog.draft["research_physics"] is None
    assert not dialog.fields["research_physics.objective_weights"].isEnabled()


def test_explicit_complete_physics_json_enables_and_disables_typed_nested_controls(dialogs):
    config = preset_config()
    config.research_physics = None
    dialog, _ = dialogs(config)
    row = dialog.fields["research_physics"]
    row.enabled_bound.setChecked(True)
    assert dialog.draft["research_physics"] is None and dialog.pending_changes
    row.control.setPlainText('{"objective": "power", "b0_T": [0, 0, 7]}')
    assert dialog.fields["research_physics.b0_T"].isEnabled()
    assert dialog.draft["research_physics"] == {"objective": "power", "b0_T": [0, 0, 7]}
    row.enabled_bound.setChecked(False)
    assert dialog.draft["research_physics"] is None
    assert not dialog.fields["research_physics.b0_T"].isEnabled()


@pytest.mark.parametrize("bad", ("{ incomplete", "[1, NaN]", '{"power": 1e999}', '"scalar"'))
def test_invalid_nested_json_retains_text_and_last_parsed_draft_across_focus_requests(dialogs, bad):
    config = preset_config()
    config.research_physics = {"objective": "power"}
    dialog, state = dialogs(config)
    row = dialog.fields["research_physics.objective_weights"]
    original = dialog.draft
    row.control.setPlainText(bad)
    assert dialog.pending_changes and dialog.draft == original
    assert not dialog.focus_native_settings() and dialog.tabs.currentIndex() == 0
    assert not dialog.apply_changes() and row.control.toPlainText() == bad
    assert not state["calls"] and not row.error.isHidden()


def test_enabling_physics_keeps_an_invalid_native_field_draft_intact(dialogs):
    dialog, _ = dialogs()
    row = dialog.fields["constrained_optimization_tolerance"]
    row.control.setText("1e")
    dialog.configure_physics.click()
    assert row.control.text() == "1e" and dialog.pending_changes
    assert not dialog.apply_changes()


def test_unknown_imported_enum_and_nested_keys_are_never_normalized_or_discarded(dialogs):
    config = preset_config()
    config.constrained_optimization_solver = "future_solver"
    config.research_physics = {"objective": "power", "future_nested": {"policy": [1, 2]}}
    dialog, state = dialogs(config)
    assert dialog.fields["constrained_optimization_solver"].control.currentText() == "future_solver"
    dialog.fields["research_physics.b0_T"].control.setPlainText("[0, 0, 5]")
    assert dialog.draft["research_physics"]["future_nested"] == {"policy": [1, 2]}
    assert not dialog.apply_changes() and not state["calls"]
    assert dialog.draft["constrained_optimization_solver"] == "future_solver"
    assert dialog.draft["research_physics"]["future_nested"] == {"policy": [1, 2]}


def test_full_float_precision_and_same_display_text_preserve_native_values(dialogs):
    config = preset_config()
    config.constrained_optimization_tolerance = 1.234567891234567e-12
    config.sweep_insulation_thickness = 1.234567891234567e-18
    dialog, _ = dialogs(config)
    assert (
        float(dialog.fields["constrained_optimization_tolerance"].control.text())
        == config.constrained_optimization_tolerance
    )
    row = dialog.fields["sweep_insulation_thickness"]
    exact = config.sweep_insulation_thickness
    text = row.control.text()
    row.control.clear()
    row.control.setText(text)
    assert dialog.draft["sweep_insulation_thickness"] == exact


def test_non_object_imported_physics_is_retained_until_explicit_configure(dialogs):
    config = preset_config()
    config.research_physics = "future_physical_contract"
    dialog, _ = dialogs(config)
    assert dialog.draft["research_physics"] == "future_physical_contract"
    assert "retained" in dialog.physics_note.text()
    dialog.configure_physics.click()
    assert dialog.draft["research_physics"] == {"objective": "power"}
    assert dialog.fields["research_physics.objective_weights"].isEnabled()


def test_actual_native_widget_render_and_focused_search_keep_controls_reachable(dialogs, application):
    dialog, state = dialogs()
    dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else ()
    if families:
        dialog.setFont(QFont(families[0], 9))
    dialog.resize(900, 780)
    dialog.show()
    assert dialog.focus_optimization()
    dialog.search.setText("constrained_optimization_solver")
    for _ in range(3):
        application.processEvents()
    control = dialog.fields["constrained_optimization_solver"].control
    assert control.isVisible() and control.isEnabled()
    assert (
        dialog.typed_scroll.viewport()
        .rect()
        .contains(control.mapTo(dialog.typed_scroll.viewport(), control.rect().center()))
    )
    assert {control.itemText(i) for i in range(control.count())} >= {"auto", "clarabel", "ecos", "scs"}
    assert not state["calls"]
    evidence = Path("outputs/optimization_settings_validation_20261011")
    evidence.mkdir(parents=True, exist_ok=True)
    assert dialog.grab().save(str(evidence / "advanced_optimization_native_900.png"))
    assert dialog.focus_native_settings()
    application.processEvents()
    assert dialog.grab().save(str(evidence / "advanced_native_json_900.png"))
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)
