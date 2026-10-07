"""The physical brief preserves intent, units and the scope of active settings."""

from dataclasses import replace
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFrame, QSizePolicy, QVBoxLayout, QWidget

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config, preset_study
from coilgen.gui.design_brief import DesignBriefPanel, design_brief


def project(kind="cylinder", **changes):
    return ProjectDocument.from_config(replace(preset_config(kind), **changes)).update_study(preset_study(kind))


def item(brief, key):
    return next(row for row in brief.items if row.key == key)


def test_brief_is_detached_intent_and_does_not_materialize_assets(monkeypatch):
    model = project()
    initial = model.to_dict()

    def resolve_forbidden(_self):
        raise AssertionError("Brief must not materialize input assets")

    monkeypatch.setattr(ProjectDocument, "config", property(resolve_forbidden))
    brief = design_brief(model)
    assert model.to_dict() == initial
    assert model._cache is None
    assert brief.revision == model.revision
    assert "20 mT/m" in item(brief, "target").value
    assert "40 mm diameter" in item(brief, "roi").value
    assert "physical span 200 mm" in item(brief, "support").value
    assert "2 × 2 mm (4 mm²)" in item(brief, "conductor").value
    assert "sheet thickness 1 mm" in item(brief, "conductor").value
    assert not any("Passed" in row.value or "accepted" in row.value for row in brief.items)


@pytest.mark.parametrize(
    "kind,word",
    [
        ("cylinder", "One connected series"),
        ("biplanar", "Separate connected circuit per support"),
        ("independent_turns", "Independent closed turns"),
    ],
)
def test_circuit_intention_describes_real_backend_modes(kind, word):
    brief = design_brief(project(kind))
    assert word in item(brief, "circuit").value
    assert "external" in item(brief, "circuit").value or "feed" in item(brief, "circuit").value
    assert "complete driven assembly" in item(brief, "terminals").scope


def test_contours_without_postprocessing_are_not_labeled_connected():
    brief = design_brief(project(skip_postprocessing=True, contour_only_sweep_mode=False))
    assert "Unconnected contours" in item(brief, "circuit").value


def test_sheet_physical_objective_and_bounds_keep_si_scope_and_zero_limits():
    physics = {
        "objective": "magnetic_energy",
        "sheet_power_limit_W": 12.0,
        "magnetic_energy_limit_J": 0.1,
        "force_limit_N": 0.0,
        "torque_equality_Nm": (0, 0, 0),
        "b0_T": (0, 2, 0),
    }
    brief = design_brief(project(research_physics=physics, b_0_direction=(1, 0, 0)))
    assert "stored magnetic energy [J]" in item(brief, "objective").value
    bounds = item(brief, "physical_bounds")
    assert "Joule loss ≤ 12 W" in bounds.value
    assert "magnetic energy ≤ 0.1 J" in bounds.value
    assert "force ≤ 0 N" in bounds.value
    assert "torque equality (0, 0, 0) N·m" in bounds.value
    assert "Continuous sheet" in bounds.scope
    assert "direction (1, 0, 0)" in item(brief, "frame").value
    assert "background (0, 2, 0) T" in item(brief, "frame").value


def test_native_objective_is_not_silently_reported_as_physical_power():
    brief = design_brief(project(research_physics=None, sf_opt_method="tikhonov"))
    assert "Native tikhonov formulation" in item(brief, "objective").value
    assert "Minimize continuous-sheet" not in item(brief, "objective").value
    assert "Solver field allowance 100 nT" in item(brief, "acceptance").value
    assert "independent-sheet allowance 0.1 nT" in item(brief, "acceptance").value


def test_field_budget_preserves_design_margin_without_claiming_winding_validation():
    brief = design_brief(project())
    field = item(brief, "field_budget")
    assert "Design relative budget 3%" in field.value
    assert "Independent-sheet relative budget 5%" in field.value
    assert "realized-wire acceptance" in field.scope
    assert "No separate absolute winding" in item(brief, "wire_budget").value


def test_default_field_constraints_can_be_explicitly_disabled():
    brief = design_brief(project(research_physics={"objective": "power", "include_default_field_constraints": False}))
    field = item(brief, "field_budget")
    assert "disabled" in field.value and "inactive" in field.value
    assert field.kind == "Scope" and field.tone == "warning"


def test_current_spacing_show_actual_effective_sheet_limit_and_policy():
    brief = design_brief(
        project(max_wire_current_A=20, min_wire_distance_m=0.004, surface_current_density_limit_A_per_m=7000)
    )
    hardware = item(brief, "hardware")
    assert "current ≤ 20 A" in hardware.value
    assert "spacing ≥ 4 mm" in hardware.value
    assert "sheet K ≤ 5000 A/m (wire_current_and_spacing)" in hardware.value
    assert "not a conductor current-density" in hardware.scope
    assert "scale_to_limit" in hardware.scope


def test_missing_hardware_and_thermal_evidence_are_not_passed_constraints():
    brief = design_brief(project())
    assert item(brief, "hardware").kind == "Scope"
    assert "No current" in item(brief, "hardware").value
    assert item(brief, "qualification").tone == "warning"
    assert "amplifier voltage" in item(brief, "qualification").value
    assert "construction release" in item(brief, "qualification").scope


def test_custom_profile_does_not_claim_fallback_rectangle_area():
    brief = design_brief(project(cross_sectional_points=((-0.0005, 0), (0, 0.0005), (0.0005, 0), (0, -0.0005))))
    conductor = item(brief, "conductor")
    assert "Custom conducting cross-section" in conductor.value
    assert "4 mm²" not in conductor.value


def test_harmonic_target_and_roi_keep_distinct_world_centres():
    brief = design_brief(project("conical_harmonic"))
    assert "Y(2,0)" in item(brief, "target").value
    assert "Coefficient is not the pole amplitude" in item(brief, "target").value
    assert "100 mm diameter" in item(brief, "roi").value
    assert "support rotations do not rotate the target" in item(brief, "frame").scope


def test_invalid_physics_remains_navigable_instead_of_crashing():
    brief = design_brief(project(research_physics=["invalid"]))
    assert brief.items[0].key == "invalid"
    assert brief.items[0].section == "solve"
    assert "settings object" in brief.items[0].value


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["coilgen-brief-test"])


@pytest.fixture
def panel(application):
    widget = DesignBriefPanel()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_panel_stays_collapsed_and_refreshes_without_navigation(panel):
    navigations = []
    panel.section_requested.connect(navigations.append)
    panel.set_project(project())
    assert panel.details.isHidden() and not panel.expand_button.isChecked()
    assert panel.value_labels["target"].textFormat() == Qt.TextFormat.PlainText
    panel.set_project(project("biplanar"), pending=True)
    assert "edits pending" in panel.revision_label.text()
    assert panel.revision_label.property("semanticTone") == "warning"
    assert "Unapplied" in panel.revision_label.toolTip()
    assert "Separate connected" in panel.value_labels["circuit"].text()
    assert navigations == []


def test_keyboard_expansion_and_navigation_use_existing_sections(panel, application):
    navigations = []
    panel.section_requested.connect(navigations.append)
    panel.set_project(project())
    panel.show()
    panel.expand_button.setFocus()
    QTest.keyClick(panel.expand_button, Qt.Key.Key_Space)
    application.processEvents()
    assert not panel.details.isHidden()
    panel.section_buttons["hardware"].setFocus()
    QTest.keyClick(panel.section_buttons["hardware"], Qt.Key.Key_Space)
    panel.section_buttons["conductor"].click()
    assert navigations == ["solve", "conductor"]


def test_brief_detail_height_is_bounded_and_scrolls_at_workbench_width(panel, application):
    panel.set_project(project())
    panel.resize(700, panel.sizeHint().height())
    panel.show()
    application.processEvents()
    assert panel.height() <= 115
    panel.expand_button.click()
    panel.resize(700, panel.sizeHint().height())
    application.processEvents()
    assert panel.details.height() <= 270
    assert panel.details.verticalScrollBar().maximum() > 0
    assert panel.details.horizontalScrollBar().maximum() == 0


@pytest.mark.parametrize("width,height", [(700, 930), (480, 768)])
def test_expand_collapse_recovers_host_plot_space(panel, application, width, height):
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(panel)
    viewport = QFrame(host)
    viewport.setMinimumHeight(140)
    viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    layout.addWidget(viewport, 1)
    history = QFrame(host)
    history.setMinimumHeight(260)
    layout.addWidget(history)
    panel.set_project(project())
    host.resize(width, height)
    host.show()
    application.processEvents()
    initial_plot_height = viewport.height()
    panel.expand_button.setChecked(True)
    application.processEvents()
    assert panel.height() <= 300
    panel.expand_button.setChecked(False)
    application.processEvents()
    assert panel.height() <= 120
    assert panel.sizeHint().height() <= 120
    assert panel.minimumSizeHint().height() <= 120
    assert viewport.height() >= initial_plot_height - 2
    assert host.height() == height
    assert panel.compact_label.y() - panel.expand_button.geometry().bottom() <= 5
    panel.setParent(None)
    host.close()
    host.deleteLater()
