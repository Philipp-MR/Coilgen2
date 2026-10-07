"""Engineering config edits remain explicit, atomic, and scientifically portable."""

from copy import deepcopy
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.expert_settings import ExpertSettingsDialog


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
