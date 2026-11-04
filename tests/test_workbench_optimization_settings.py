"""Visible objectives and editable solver controls retain engineering drafts."""

from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialogButtonBox, QLabel, QPushButton

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from test_workbench_usability import application as application, windows as windows


def native_window(windows, **overrides):
    config = replace(
        preset_config(), **{"research_physics": None, "sf_opt_method": "constrained_quadratic", **overrides}
    )
    window = windows(ProjectDocument.from_config(config))
    window.select_node("optimization")
    return window


def test_objective_equation_and_solver_summary_are_visible_outside_edit_dialogs(windows):
    window = native_window(windows)
    panel = window.settings.widget()
    equation = panel.findChild(QLabel, "optimization_cost_equation")
    solver = panel.findChild(QLabel, "optimization_solver_summary")
    bounds = panel.findChild(QLabel, "optimization_constraints")
    assert "Ax − b" in bounds.text() and not bounds.isHidden()
    assert equation.text() and solver.text()
    assert panel.isAncestorOf(equation) and not equation.isHidden()
    assert panel.findChild(QPushButton, "all_optimization_parameters").text() == "All optimization parameters…"
    assert "Solver configuration" in window._settings_decisions
    assert window._draft_controls["sf_opt_method"].findText("target_error_constrained_efficiency") >= 0
    assert window._draft_controls["objective"].findText("mixture") >= 0
    assert window._draft_controls["objective"].findText("Native objective (see below)") >= 0


def test_opening_optimization_preserves_loaded_precision_and_configuration(windows):
    window = native_window(
        windows,
        tikonov_reg_factor=6.234567891234567,
        constrained_optimization_tolerance=1.234567891234567e-13,
        stream_function_smoothness_factor=0.0001234567891234567,
    )
    original = window.project
    assert not window._draft_dirty
    assert window.apply_settings()
    assert window.project is original and window.undo_stack.count() == 0


@pytest.mark.parametrize("method,conic", [("socp", True), ("tikkonov", False), ("fixed_error_efficiency", False)])
def test_imported_method_aliases_have_correct_controls_without_rewriting_values(windows, method, conic):
    window = native_window(windows, sf_opt_method=method)
    assert window._draft_controls["constrained_optimization_tolerance"].isEnabled() is conic
    factor = window._draft_controls["tikonov_reg_factor"]
    assert factor.isEnabled() is not conic
    numerical = window._settings_decisions["Numerical settings"]
    assert numerical.form.isRowVisible(factor.parentWidget()) is not conic
    assert window.project.config.sf_opt_method == method and not window._draft_dirty


def test_current_and_acceptance_drafts_reach_live_summary(windows, monkeypatch):
    import coilgen.gui.workbench as gui

    observed = []
    summary = gui.build_optimization_summary

    def capture(config, draft=None, **kwargs):
        observed.append(dict(draft or {}))
        return summary(config, draft, **kwargs)

    monkeypatch.setattr(gui, "build_optimization_summary", capture)
    window = native_window(windows)
    window._draft_controls["current_limit"].setValue(1234)
    assert observed[-1]["surface_current_density_limit_A_per_m"] == 1234
    window._draft_controls["acceptance_slack"].setValue(234)
    assert observed[-1]["target_error_constraint_acceptance_absolute_T"] == pytest.approx(234e-9)
    assert "pending draft" in window.findChild(QLabel, "optimization_summary_title").text()


def test_solver_objective_and_penalty_edits_commit_once_and_reach_compute(windows):
    window = native_window(windows)
    original = window.project
    values = {
        "constrained_optimization_solver": "clarabel",
        "constrained_optimization_tolerance": 2e-11,
        "constrained_optimization_max_iterations": 12345,
        "constrained_optimization_threads": 2,
        "constrained_optimization_verbose": True,
        "constrained_quadratic_fallback_policy": "strict",
        "stream_function_cost_function": "standard",
        "stream_function_smoothness_factor": 0.0025,
    }
    for key, value in values.items():
        control = window._draft_controls[key]
        if isinstance(value, str):
            control.setCurrentText(value)
        elif isinstance(value, bool):
            control.setChecked(value)
        else:
            control.setValue(value)
    assert "pending draft" in window.findChild(QLabel, "optimization_summary_title").text()
    assert window.project is original
    assert window.apply_settings()
    assert window.project.revision == original.revision + 1 and window.undo_stack.count() == 1
    for key, value in values.items():
        assert getattr(window.project.config, key) == value
    window.compute()
    assert len(window.service.submissions) == 1
    submitted = window.service.submissions[0][0].config
    for key, value in values.items():
        assert getattr(submitted, key) == value
    window.undo_stack.undo()
    assert window.project.fingerprint == original.fingerprint


def test_scientific_tolerance_entry_is_not_rounded_to_zero(windows, application):
    window = native_window(windows)
    section = window._settings_decisions["Solver configuration"]
    section.open()
    application.processEvents()
    control = window._draft_controls["constrained_optimization_tolerance"]
    control.lineEdit().selectAll()
    QTest.keyClicks(control.lineEdit(), "3e-14")
    QTest.keyClick(control.lineEdit(), Qt.Key.Key_Tab)
    assert window.apply_settings()
    assert window.project.config.constrained_optimization_tolerance == pytest.approx(3e-14, rel=1e-12, abs=0)


def test_absolute_only_field_budget_can_be_selected_without_removing_absolute_limit(windows):
    window = native_window(windows, target_error_constraint_metric="rms", target_error_rms_limit_T=1e-7)
    window._draft_controls["target_error_tolerance_percent"].setValue(0)
    assert window.apply_settings()
    assert window.project.config.target_error_tolerance_percent == 0
    assert window.project.config.target_error_rms_limit_T == 1e-7


def test_mixture_selection_preserves_physical_weights_scales_and_bounds(windows):
    physics = {
        "objective": "power",
        "objective_weights": {"power": 2.0, "magnetic_energy": 0.5},
        "objective_scales": {"power": 3.0, "magnetic_energy": 0.002},
        "force_limit_N": 0.2,
    }
    window = windows(ProjectDocument.from_config(replace(preset_config(), research_physics=physics)))
    window.select_node("optimization")
    strategy = window._settings_decisions["Solve strategy"].summary_label.text()
    assert "Joule power" in strategy and "balanced" not in strategy and "field_error_plus_resistance" not in strategy
    assert not window._draft_controls["stream_function_cost_function"].isEnabled()
    assert not window._draft_controls["constrained_optimization_verbose"].isEnabled()
    fallback = window._draft_controls["constrained_quadratic_fallback_policy"]
    assert not fallback.isEnabled()
    assert not window._settings_decisions["Solver configuration"].form.isRowVisible(fallback.parentWidget())
    window._draft_controls["objective"].setCurrentText("mixture")
    assert window.apply_settings()
    assert window.project.config.research_physics == {**physics, "objective": "mixture"}


@pytest.mark.parametrize("close_with", ["Cancel", "Escape", "Close"])
def test_custom_formula_cancel_restores_solver_objective_and_all_text(windows, application, close_with):
    window = native_window(windows, sf_opt_method="tikhonov")
    original = window.project
    window.findChild(QPushButton, "edit_custom_formula").click()
    section = window._settings_decisions["Solve strategy"]
    controls = window._draft_controls
    assert section.dialog.isVisible() and controls["objective"].currentText() == "custom"
    controls["custom_objective"].setText("sum_squares(residual_T / 1e-6)")
    controls["custom_constraints"].setPlainText("abs(torque_Nm[1]) <= 0.1")
    controls["custom_solver"].setCurrentText("local")
    controls["include_default_field_constraints"].setChecked(False)
    assert window._draft_dirty and window.project is original
    if close_with == "Cancel":
        section.footer.button(QDialogButtonBox.StandardButton.Cancel).click()
    elif close_with == "Escape":
        QTest.keyClick(controls["custom_constraints"], Qt.Key.Key_Escape)
    else:
        section.dialog.close()
    application.processEvents()
    assert not section.dialog.isVisible() and not window._draft_dirty and window.apply_settings()
    assert window.project is original and not window.undo_stack.count()
    assert controls["sf_opt_method"].currentText() == "tikhonov" and controls["custom_objective"].text() == ""
    assert (
        controls["custom_constraints"].toPlainText() == "" and controls["include_default_field_constraints"].isChecked()
    )


def test_custom_formulas_validate_apply_reopen_and_reach_compute(windows, tmp_path):
    physics = {"objective": "power", "force_limit_N": 1.5, "acceptance_absolute_by_unit": {"T": 3e-10}}
    window = native_window(windows, research_physics=physics)
    original = window.project
    window.findChild(QPushButton, "edit_custom_formula").click()
    controls = window._draft_controls
    controls["custom_objective"].setText("__import__('os')")
    assert not window.apply_settings() and window.project is original
    formula = "power_W / 10 + sum_squares(residual_T / 0.001)"
    limits = ["abs(torque_Nm[2]) <= 0.1", "norm_inf(force_N) <= 1"]
    controls["custom_objective"].setText(formula)
    controls["custom_constraints"].setPlainText("\n".join(limits + ["", "oops"]))
    assert not window.apply_settings() and "Constraint line 4" in window._draft_error
    controls["custom_constraints"].setPlainText("\n".join(limits))
    controls["custom_solver"].setCurrentText("local")
    controls["include_default_field_constraints"].setChecked(False)
    assert formula in window.findChild(QLabel, "optimization_cost_equation").text()
    assert "Local (SLSQP)" in window._settings_decisions["Solve strategy"].summary_label.text()
    assert window.apply_settings() and window.undo_stack.count() == 1
    expected = {
        **physics,
        "objective": "custom",
        "custom_objective": formula,
        "custom_constraints": limits,
        "custom_solver": "local",
        "include_default_field_constraints": False,
    }
    assert window.project.config.research_physics == expected
    path = tmp_path / "formula.coilgen.json"
    window.save_project(path)
    assert ProjectDocument.load(path).config.research_physics == expected
    window.compute()
    assert window.service.submissions[0][0].config.research_physics == expected
    window.select_node("optimization")
    assert window._draft_controls["custom_objective"].text() == formula
    assert window._draft_controls["custom_constraints"].toPlainText().splitlines() == limits
    window.findChild(QPushButton, "all_optimization_parameters").click()
    expert = window.expert_window
    assert expert.fields["research_physics.custom_solver"].control.currentText() == "local"
    assert expert.fields["research_physics.custom_objective"].control.text() == formula
    expert.reject()
    window._draft_controls["objective"].setCurrentText("power")
    assert window.apply_settings()
    expected.pop("custom_objective")
    assert window.project.config.research_physics == {**expected, "objective": "power"}


def test_switch_to_native_cannot_silently_drop_physical_constraints(windows):
    physics = {"objective": "power", "torque_limit_Nm": 0.3}
    window = windows(ProjectDocument.from_config(replace(preset_config(), research_physics=physics)))
    window.select_node("optimization")
    original = window.project
    window._draft_controls["objective"].setCurrentText("Native objective (see below)")
    assert not window.apply_settings()
    assert window.project is original and window.project.config.research_physics == physics
    assert "torque_limit_Nm" in window.messages.toPlainText()


def test_all_parameters_opens_focused_editor_and_applies_through_existing_undo(windows):
    window = native_window(windows)
    window.findChild(QPushButton, "all_optimization_parameters").click()
    dialog = window.expert_window
    editor = dialog.fields["constrained_optimization_max_iterations"]
    assert not editor.isHidden()
    editor.control.setText("4321")
    assert window.project.config.constrained_optimization_max_iterations != 4321
    assert dialog.apply_changes()
    assert window.project.config.constrained_optimization_max_iterations == 4321
    assert window.undo_stack.count() == 1


@pytest.mark.parametrize("metric", ["rms", "std"])
def test_explicit_field_limits_are_visible_editable_and_match_native_constraint_units(windows, metric):
    import numpy as np
    from coilgen.optimization import _dsv_constraint_block

    key = "target_error_" + metric + "_limit_T"
    window = native_window(windows, target_error_constraint_metric=metric, **{key: 1e-7})
    section = window._settings_decisions["Field and current limits"]
    control = window._draft_controls[key]
    assert section.form.isRowVisible(control.parentWidget())
    assert "100 nT" in section.summary_label.text()
    assert "overrides" in section.summary_label.text()
    assert not window._draft_controls["target_error_tolerance_percent"].isEnabled()
    control.setValue(3e-8)
    assert "30 nT" in section.summary_label.text()
    assert window.apply_settings()
    config = window.project.config
    block = _dsv_constraint_block(np.eye(2), np.array([1e-6, -1e-6]), config, weights=np.array([1.0, 4.0]))
    assert block.allowed_residual == pytest.approx(np.sqrt(5) * 3e-8)
    assert getattr(config, key) == 3e-8


def test_zero_explicit_limit_restores_relative_absolute_controls_and_loaded_precision(windows):
    window = native_window(windows, target_error_constraint_metric="rms", target_error_rms_limit_T=1.234567891234e-13)
    original = window.project
    assert not window._draft_dirty and window.apply_settings() and window.project is original
    window._draft_controls["target_error_rms_limit_T"].setValue(0)
    assert window._draft_controls["target_error_tolerance_percent"].isEnabled()
    absolute = window._draft_controls["target_error_tolerance_absolute_T"]
    assert absolute.isEnabled()
    absolute.setValue(5e-8)
    assert "50 nT" in window._settings_decisions["Field and current limits"].summary_label.text()
    assert window.apply_settings()
    assert window.project.config.target_error_tolerance_absolute_T == 5e-8
    assert window.project.config.target_error_rms_limit_T == 0


def test_disabled_physical_field_defaults_and_soft_fit_are_identified_in_primary_card(windows):
    physics = {"objective": "power", "include_default_field_constraints": False}
    window = native_window(windows, research_physics=physics)
    section = window._settings_decisions["Field and current limits"]
    assert "Default field budgets disabled" in section.summary_label.text()
    assert not window._draft_controls["target_error_tolerance_percent"].isEnabled()
    window._draft_controls["sf_opt_method"].setCurrentText("tikhonov")
    assert "Post-fit check, not a hard solve constraint" in section.summary_label.text()
