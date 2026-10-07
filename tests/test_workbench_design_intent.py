"""Physical topology and inspection tasks preserve the numerical inputs."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import test_workbench_target_policies as target_tests
import test_native_workbench as native_tests
from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config

application = target_tests.application
windows = target_tests.windows
scientific_viewport = native_tests.scientific_viewport


def test_per_support_topology_visible_and_unrelated_edit_preserves_all_native_settings(windows):
    project = ProjectDocument.from_config(preset_config("biplanar"))
    window = windows(project)
    window.select_node("winding")
    assert window._draft_controls["winding_intent"].currentText() == "Separate circuits per support"
    window._draft_controls["levels"].setValue(project.config.levels + 2)
    assert window.apply_settings()
    expected = project.values
    expected["levels"] += 2
    assert window.project.values == expected


@pytest.mark.parametrize("initial_skip", [True, False])
def test_independent_turn_intention_retains_inactive_connected_preference(windows, initial_skip):
    config = replace(preset_config(), skip_postprocessing=initial_skip, routing_circuit_scope="per_part")
    window = windows(ProjectDocument.from_config(config))
    window.select_node("winding")
    window._draft_controls["winding_intent"].setCurrentText("Independent closed turns")
    assert window.apply_settings()
    assert window.project.config.contour_only_sweep_mode
    assert window.project.config.skip_postprocessing is initial_skip
    assert window.project.config.routing_circuit_scope == "per_part"
    assert window.project.config.skip_sweep == config.skip_sweep


def test_contours_only_disables_solids_explicitly_and_undo_restores_full_topology(windows):
    window = windows(ProjectDocument.from_config(preset_config()))
    before = window.project.values
    window.select_node("winding")
    window._draft_controls["winding_intent"].setCurrentText("Contours only")
    assert not window._draft_controls["sweep"].isEnabled()
    assert not window._draft_controls["sweep"].isChecked()
    assert window.apply_settings()
    assert window.project.config.skip_postprocessing and window.project.config.skip_sweep
    assert not window.project.config.contour_only_sweep_mode
    window.undo_stack.undo()
    assert window.project.values == before


def test_legacy_unused_solid_preference_is_preserved_on_unrelated_edit(windows):
    project = ProjectDocument.from_config(replace(preset_config(), skip_postprocessing=True, skip_sweep=False))
    window = windows(project)
    window.select_node("winding")
    assert window._draft_controls["winding_intent"].currentText() == "Contours only"
    window._draft_controls["levels"].setValue(project.config.levels + 1)
    assert window.apply_settings()
    assert window.project.config.skip_sweep is False


def test_field_inspection_selects_saved_wire_residual_without_refit(scientific_viewport):
    viewport = scientific_viewport
    arrays = {
        "target_points_m": np.array([[0, 0, 0], [0, 0.01, 0]]),
        "target_field_T": np.array([[0, 0, 1e-6], [0, 0, 2e-6]]),
        "routed_field_T": np.array([[0, 0, 1.1e-6], [0, 0, 2.1e-6]]),
    }
    before = deepcopy(arrays)
    viewport.display_run(arrays)
    current = viewport.wire_current.currentText()
    assert viewport.inspect_task("field")
    assert viewport.dataset.currentData() == "routed_residual"
    assert viewport.inspection_task.currentData() == "field"
    assert viewport.wire_current.currentText() == current
    assert "design samples" in viewport.inspection_note.text()
    for key, values in before.items():
        np.testing.assert_array_equal(viewport._arrays[key], values)


def test_wire_inspection_labels_centerlines_when_solids_are_missing(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run({"contour_0_points_m": np.array([[0.1, 0, 0], [0.1, 0.01, 0]])})
    assert viewport.inspect_task("wire")
    assert viewport.dataset.currentData() == "contours"
    assert "solids are unavailable" in viewport.inspection_note.text()
    assert "do not establish physical wire clearance" in viewport.inspection_note.text()
    before = deepcopy(viewport.view_state())
    assert not viewport.inspect_task("connections")
    assert viewport.view_state() == before
    assert viewport.dataset.currentData() == "contours"
    assert viewport.inspection_task.currentData() == "wire"
    assert not viewport.inspection_actions["connections"].isEnabled()
    # Route inspection becomes available only when actual connected paths
    # exist; their presence still makes no terminal/drive acceptance claim.
    route = np.array([[0.1, 0, 0], [0.1, 0.01, 0], [0.1, 0.01, 0.002]])
    viewport.display_run({"route_0_points_m": route})
    assert viewport.inspect_task("connections")
    assert viewport.dataset.currentData() == "routes"
    assert viewport.inspection_task.currentData() == "connections"
    assert "completeness requires separate evidence" in viewport.inspection_note.text()
    np.testing.assert_array_equal(viewport.plotter.meshes[0][0].points, route)


def test_sheet_inspection_keeps_independent_scope_when_wire_field_absent(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(
        {
            "validation_points_m": np.array([[0, 0, 0]]),
            "validation_field_T": np.array([[0, 0, 1e-6]]),
        }
    )
    assert viewport.inspect_task("field")
    assert viewport.dataset.currentData() == "validation"
    assert "Independent sheet" in viewport.inspection_note.text()
    assert "final winding evidence is unavailable" in viewport.inspection_note.text()
