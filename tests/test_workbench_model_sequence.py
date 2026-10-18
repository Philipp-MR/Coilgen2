"""Model Builder navigation preserves requests while separating studies and results."""

import test_workbench_usability as usability

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config


application = usability.application
windows = usability.windows


def _key(item):
    return item.data(0, Qt.ItemDataRole.UserRole)


def _children(item):
    return tuple(_key(item.child(index)) for index in range(item.childCount()))


def test_model_study_results_tree_has_predictable_operation_order(windows):
    window = windows()
    assert window.tree.topLevelItemCount() == 3
    model, study, results = (window.tree.topLevelItem(index) for index in range(3))

    assert model.text(0) == "Model"
    assert _children(model) == ("geometry", "target", "material", "winding")
    assert study.text(0) == "Study"
    assert _children(study) == (
        "optimization",
        "validation",
        "stage:prepared",
        "stage:sheet",
        "stage:contours",
        "stage:routed",
        "stage:evaluated",
    )
    assert _key(results) == "results"
    assert _children(results) == (
        "result:metrics",
        "result:slices",
        "result:errors",
        "result:harmonics",
        "result:coil",
        "result:checks",
        "export",
    )
    assert tuple(window.stage_items) == ("prepared", "sheet", "contours", "routed", "evaluated")


def test_study_parent_opens_requested_boundary_without_changing_model(windows):
    project = ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": "contours"})
    window = windows(project)
    window.select_node("material")

    window.tree.setCurrentItem(window.tree.topLevelItem(1))

    assert _key(window.tree.currentItem()) == "stage:contours"
    assert window.stage_selector.currentData() == "contours"
    assert window.project is project
    assert window.project.study["pipeline_stage"] == "contours"
    assert window.undo_stack.count() == 0
    assert not window.service.submissions


def test_selecting_stage_leaf_sets_the_persistent_compute_request_once(windows):
    window = windows()
    original = window.project

    window.select_node("stage:contours")

    assert _key(window.tree.currentItem()) == "stage:contours"
    assert window.stage_selector.currentData() == "contours"
    assert window.project.study["pipeline_stage"] == "contours"
    assert window.project.values == original.values
    assert window.project.revision == original.revision + 1
    assert window.undo_stack.count() == 1
    selected = window.project

    window.select_node("stage:contours")
    window.navigate_workflow("compute")

    assert window.project is selected
    assert window.undo_stack.count() == 1
    assert not window.service.submissions


def test_toolbar_stage_change_keeps_visible_settings_and_request_in_sync(windows):
    project = ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": "contours"})
    window = windows(project)
    window.navigate_workflow("compute")

    window.stage_selector.setCurrentIndex(window.stage_selector.findData("sheet"))

    assert _key(window.tree.currentItem()) == "stage:sheet"
    assert _key(window._current_node) == "stage:sheet"
    assert window.project.study["pipeline_stage"] == "sheet"
    assert window.project.values == project.values
    assert window.project.revision == project.revision + 1
    assert window.undo_stack.count() == 1
    assert not window.service.submissions


def test_reopening_already_selected_study_preserves_revision(windows):
    project = ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": "routed"})
    window = windows(project)
    window.navigate_workflow("compute")

    window.tree.setCurrentItem(window.tree.topLevelItem(1))
    window.navigate_workflow("compute")

    assert _key(window.tree.currentItem()) == "stage:routed"
    assert window.stage_selector.currentData() == "routed"
    assert window.project is project
    assert window.undo_stack.count() == 0


def test_full_study_scope_records_skipped_operations_without_acceptance_claim(windows):
    config = preset_config()
    config.skip_postprocessing = True
    config.skip_sweep = True
    config.skip_inductance_calculation = True
    project = ProjectDocument.from_config(config)
    window = windows(project)
    window.navigate_workflow("compute")
    scope = window.settings.widget().findChild(QLabel, "study_scope").text()

    assert "Interconnections skipped" in scope
    assert "Conductor solids skipped" in scope
    assert "Inductance calculation skipped" in scope
    assert "Strict fabrication review is not requested" in scope
    assert "passed" not in scope.lower()
    assert window.project is project


def test_direct_navigation_uses_sections_and_avoids_completion_numbers(windows):
    window = windows()
    assert not hasattr(window, "workflow")
    assert window.tree.headerItem().text(0) == "Model Builder"
    for key in ("geometry", "target", "material", "winding", "optimization", "validation", "results", "export"):
        window.select_node(key)
        assert window.page_label.text()
        assert "Workflow page" not in window.page_label.text()
        assert " of 8" not in window.page_label.text()
        assert window.tree_items[key].text(0) in window.page_label.text()


def test_next_navigation_follows_visible_model_then_study_then_results(windows):
    window = windows()
    original = window.project
    selected_stage = window.stage_selector.currentData()
    pages = (
        "geometry",
        "target",
        "material",
        "winding",
        "optimization",
        "validation",
        f"stage:{selected_stage}",
        "results",
        "export",
    )
    for index, expected in enumerate(pages):
        assert _key(window.tree.currentItem()) == expected
        if index < len(pages) - 1:
            window.next_button.click()

    assert not window.next_button.isEnabled()
    assert window.project is original
    assert window.undo_stack.count() == 0
    assert not window.service.submissions


def test_validation_ratio_explains_actual_acceptance_math_without_editing(windows):
    project = ProjectDocument.from_config(preset_config()).update_study({"design_margin": 0.8})
    window = windows(project)
    window.select_node("validation")
    text = "\n".join(label.text() for label in window.settings.widget().findChildren(QLabel))

    assert "Design / validation budget ratio" in text
    assert "0.8" in text and "1.25" in text
    assert "optimization" in text.lower()
    assert window._draft_controls["design_margin"].value() == 0.8
    assert window.project is project
    assert window.undo_stack.count() == 0
