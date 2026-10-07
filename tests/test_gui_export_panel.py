"""Saved-solid selection preserves units, check scope and export intent."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QScrollArea

from coilgen.application.cad import GeometryOption
from coilgen.gui.colors import item_foreground
from coilgen.gui.export_panel import GeometryExportPanel


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["geometry-export-test"])


@pytest.fixture
def panels(application):
    instances = []

    def make(*args, **kwargs):
        panel = GeometryExportPanel(*args, **kwargs)
        instances.append(panel)
        return panel

    yield make
    for panel in instances:
        panel.close()
        panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _record(*, ready=True, sheet=True):
    return SimpleNamespace(
        directory=Path("immutable_run"),
        metadata={
            "name": "Cylinder Gy",
            "project_id": "owner",
            "project_revision": 4,
            "summary": {"construction_released": False, "sheet_accepted": sheet},
            "fabrication": {"ready": ready, "failure_reasons": ["missing_return"] if ready is False else []},
        },
    )


def _options(*, status="passed", inspection=False, step_unit="m"):
    common = dict(
        body_id="part001_body0001",
        scope="fabrication_review_geometry",
        check_status=status,
        bounds_m=((0.01, 0.02, 0.03), (0.014, 0.025, 0.036)),
        inspection_only=inspection,
        source_sha256="0" * 64,
    )
    return (
        GeometryOption(name="fabrication/conductor.stl", format="stl", source_unit="m", **common),
        GeometryOption(name="fabrication/conductor_faceted.step", format="step", source_unit=step_unit, **common),
    )


def test_active_export_stays_disabled_after_selection_or_units_change(panels):
    panel = panels(_record(), _options())
    requests = []
    panel.export_requested.connect(lambda *args: requests.append(args))
    panel.set_exporting(True)
    panel.units.setCurrentIndex(1)
    panel.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    assert not panel.export_button.isEnabled()
    assert "Exporting" in panel.export_button.text()
    panel._request_export()
    assert requests == []
    panel.set_exporting(False)
    assert panel.export_button.isEnabled()
    panel._request_export()
    assert len(requests) == 1 and requests[0][1] == "m"


def test_no_record_has_no_export_or_review_action(panels):
    panel = panels()
    assert "No saved run selected" in panel.scope.text()
    assert panel.table.rowCount() == 0
    assert not panel.export_button.isEnabled()
    assert panel.selected_names() == ()
    assert panel.review_button.isHidden() and panel.problems_button.isHidden()
    assert panel.export_button.text() == "No saved solids"
    assert all(
        control.isHidden()
        for control in (panel.table, panel.units, panel.include_project, panel.bounds, panel.unit_note)
    )


def test_unverified_missing_solids_direct_review_without_enabling_export(panels):
    panel = panels(_record(ready=None), project=SimpleNamespace(id="owner", revision=7))
    assert "No solid STL or solid STEP" in panel.status.text()
    assert "Review the current revision" in panel.status.text()
    assert panel.review_button.text() == "Review current revision"
    assert "applied revision 7" in panel.review_scope.text()
    assert "saved run remains immutable" in panel.review_scope.text()
    assert panel.status.property("semanticTone") == "warning"
    assert not panel.review_button.isHidden() and panel.problems_button.isHidden()
    assert not panel.export_button.isEnabled()
    emitted = []
    panel.review_requested.connect(lambda: emitted.append("review"))
    panel.review_button.click()
    assert emitted == ["review"]
    panel.set_exporting(True)
    assert not panel.review_button.isEnabled()
    panel.set_exporting(False)
    assert panel.review_button.isEnabled()


def test_review_cannot_recompute_a_different_active_channel(panels):
    panel = panels(_record(ready=None), project=SimpleNamespace(id="another-channel", revision=7))
    emitted = []
    panel.review_requested.connect(lambda: emitted.append("review"))
    assert not panel.review_button.isEnabled()
    assert "owning channel" in panel.review_scope.text()
    panel.review_button.click()
    assert emitted == []
    panel.set_context(SimpleNamespace(id="owner", revision=9))
    assert panel.review_button.isEnabled()
    assert "applied revision 9" in panel.review_scope.text()


def test_integrity_diagnostic_cannot_be_overridden_by_stray_geometry_options(panels):
    panel = panels(
        _record(ready=None),
        _options(),
        diagnostic="Saved solid checksum changed.",
        project=SimpleNamespace(id="owner", revision=7),
    )
    emitted = []
    panel.export_requested.connect(lambda *args: emitted.append(args))
    panel._request_export()
    assert emitted == [] and not panel.export_button.isEnabled()
    assert not panel.review_button.isEnabled()
    assert "checksum" in panel.status.text()


def test_failed_review_without_solids_directs_problems_and_uses_error_tone(panels):
    panel = panels(_record(ready=False))
    assert "checks failed" in panel.status.text()
    assert panel.status.property("semanticTone") == "error"
    assert not panel.problems_button.isHidden() and panel.review_button.isHidden()
    assert not panel.export_button.isEnabled()
    emitted = []
    panel.problems_requested.connect(lambda: emitted.append("problems"))
    panel.problems_button.click()
    assert emitted == ["problems"]


def test_saved_inspection_solids_remain_exportable_when_review_failed(panels):
    options = _options(inspection=True)
    panel = panels(_record(ready=False), options)
    assert panel.export_button.isEnabled()
    assert panel.selected_names() == (options[0].name,)
    assert "inspection geometry" in panel.status.text()
    assert "checks failed" in panel.status.text()
    assert panel.status.property("semanticTone") == "error"
    assert "Not released" in panel.findChild(type(panel.scope), "geometry_export_release").text()


def test_stl_is_selected_by_default_and_step_is_explicit_opt_in(panels):
    options = _options()
    panel = panels(_record(), options)
    assert panel.units.currentData() == "mm"
    assert panel.include_project.isChecked()
    assert panel.table.item(0, 0).checkState() == Qt.CheckState.Checked
    assert panel.table.item(1, 0).checkState() == Qt.CheckState.Unchecked
    assert panel.selected_names() == (options[0].name,)
    panel.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    assert panel.selected_names() == tuple(option.name for option in options)
    panel.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert panel.selected_names() == (options[1].name,)
    panel.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not panel.export_button.isEnabled()


@pytest.mark.parametrize("step_unit", ["m", "mm"])
def test_step_retains_declared_units_while_stl_unit_control_changes(panels, step_unit):
    panel = panels(_record(), _options(step_unit=step_unit))
    assert panel.table.item(0, 1).text() == "STL · mm"
    assert panel.table.item(1, 1).text() == f"STEP · {step_unit}"
    panel.units.setCurrentIndex(panel.units.findData("m"))
    assert panel.table.item(0, 1).text() == "STL · m"
    assert panel.table.item(1, 1).text() == f"STEP · {step_unit}"


def test_extents_and_world_bounds_are_physical_mm_in_both_export_units(panels):
    panel = panels(_record(), _options())
    assert panel.table.item(0, 3).text() == "4 × 5 × 6"
    original = panel.bounds.text()
    assert "world bounds [mm]" in original
    assert "x 10…14" in original and "y 20…25" in original and "z 30…36" in original
    panel.units.setCurrentIndex(panel.units.findData("m"))
    assert panel.bounds.text() == original


def test_export_signal_carries_exact_selected_artifacts_units_and_project_intent(panels):
    options = _options()
    panel = panels(_record(), options)
    emitted = []
    panel.export_requested.connect(lambda names, unit, project: emitted.append((names, unit, project)))
    panel.export_button.click()
    assert emitted == [((options[0].name,), "mm", True)]
    panel.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    panel.units.setCurrentIndex(panel.units.findData("m"))
    panel.include_project.setChecked(False)
    panel.export_button.click()
    assert emitted[-1] == (tuple(option.name for option in options), "m", False)


def test_failed_body_checks_have_error_foreground_and_explicit_failed_text(panels):
    panel = panels(_record(), _options(status="failed", inspection=True))
    cell = panel.table.item(0, 2)
    assert cell.text() == "Inspection · Failed"
    assert cell.foreground().color() == item_foreground("error", panel.palette()).color()
    assert cell.toolTip() == "fabrication_review_geometry"
    assert panel.status.property("semanticTone") == "error"
    assert "checks failed" in panel.status.text()
    assert panel.export_button.isEnabled()  # inspectable solid, no acceptance promotion


def test_unverified_and_failed_checks_keep_distinct_semantics(panels):
    panel = panels(_record(ready=None), _options(status="unverified", inspection=True))
    assert panel.table.item(0, 2).text() == "Inspection · Unverified"
    assert panel.status.property("semanticTone") == "warning"
    assert "unverified checks" in panel.status.text()
    assert "checks failed" not in panel.status.text()


def test_field_failure_does_not_hide_geometry_or_promote_its_scope(panels):
    panel = panels(_record(sheet=False), _options(inspection=True))
    assert panel.export_button.isEnabled()
    assert "checks failed" in panel.status.text()
    assert panel.status.property("semanticTone") == "error"
    assert "field acceptance and construction release are separate" in panel.scope.text()


def test_integrity_diagnostic_without_geometry_is_error_and_not_exportable(panels):
    panel = panels(_record(), diagnostic="Artifact integrity check failed.")
    assert panel.status.text() == "Artifact integrity check failed."
    assert panel.status.property("semanticTone") == "error"
    assert not panel.export_button.isEnabled()


def test_recordless_panel_cannot_emit_export_even_with_stray_options(panels):
    panel = panels(None, _options())
    assert not panel.export_button.isEnabled()
    emitted = []
    panel.export_requested.connect(lambda *args: emitted.append(args))
    panel._request_export()
    assert emitted == []


@pytest.mark.parametrize("revision", [None, "", "  "])
def test_missing_revision_is_visibly_unverified(panels, revision):
    record = _record()
    record.metadata["project_revision"] = revision
    panel = panels(record, _options())
    assert "revision unverified" in panel.scope.text()
    assert "revision None" not in panel.scope.text()


def test_bounds_follow_current_body_while_checked_exports_stay_independent(panels):
    first = _options()[0]
    second = GeometryOption(
        name="fabrication/second.stl",
        format="stl",
        source_unit="m",
        body_id="part002_body0001",
        scope="fabrication_review_geometry",
        check_status="passed",
        inspection_only=False,
        bounds_m=((-0.1, -0.2, -0.3), (-0.09, -0.18, -0.27)),
        source_sha256="0" * 64,
    )
    panel = panels(_record(), (first, second))
    assert panel.selected_names() == (first.name, second.name)
    panel.table.setCurrentCell(1, 0)
    assert "part002_body0001" in panel.bounds.text() and "x -100…-90" in panel.bounds.text()
    assert panel.selected_names() == (first.name, second.name)
    panel.units.setCurrentIndex(panel.units.findData("m"))
    assert "part002_body0001" in panel.bounds.text() and "world bounds [mm]" in panel.bounds.text()
    panel.table.setCurrentCell(0, 0)
    assert "part001_body0001" in panel.bounds.text()
    panel.table.setCurrentCell(-1, -1)
    assert "part001_body0001" in panel.bounds.text()  # fallback to first checked solid


def test_primary_export_precedes_table_and_details(panels):
    panel = panels(_record(ready=False), _options(inspection=True))
    layout = panel.layout()
    assert layout.indexOf(panel.problems_button) < layout.indexOf(panel.units)
    assert layout.indexOf(panel.units) < layout.indexOf(panel.include_project) < layout.indexOf(panel.export_button)
    assert layout.indexOf(panel.export_button) < layout.indexOf(panel.table) < layout.indexOf(panel.bounds)
    assert panel.table.maximumHeight() <= 160
    assert [panel.table.horizontalHeaderItem(index).text() for index in range(4)] == [
        "Conductor",
        "Units",
        "Checks",
        "Extent [mm]",
    ]
    assert all(panel.table.horizontalHeaderItem(index).toolTip() for index in range(4))


@pytest.mark.parametrize("with_solids", [True, False])
def test_export_fits_narrow_settings_panel_with_primary_action_visible(application, with_solids):
    font_id = -1
    if not QFontDatabase.families():
        font_path = Path("C:/Windows/Fonts/segoeui.ttf")
        if not font_path.is_file():
            pytest.skip("Native UI font is unavailable on the offscreen platform.")
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        assert font_id >= 0
    panel = GeometryExportPanel(_record(ready=False), _options(inspection=True) if with_solids else ())
    font_family = QFontDatabase.applicationFontFamilies(font_id)[0] if font_id >= 0 else "Segoe UI"
    panel.setFont(QFont(font_family, 9))
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel)
    scroll.resize(285, 430)
    scroll.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    scroll.show()
    try:
        application.processEvents()
        assert scroll.horizontalScrollBar().maximum() == 0
        assert panel.minimumSizeHint().width() <= scroll.viewport().width()
        assert panel.export_button.isVisible()
        top_left = panel.export_button.mapTo(scroll.viewport(), QPoint(0, 0))
        bottom_right = panel.export_button.mapTo(
            scroll.viewport(), QPoint(panel.export_button.width() - 1, panel.export_button.height() - 1)
        )
        assert scroll.viewport().rect().contains(top_left)
        assert scroll.viewport().rect().contains(bottom_right)
        assert panel.export_button.isEnabled() is with_solids
    finally:
        scroll.close()
        scroll.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if font_id >= 0:
            QFontDatabase.removeApplicationFont(font_id)
