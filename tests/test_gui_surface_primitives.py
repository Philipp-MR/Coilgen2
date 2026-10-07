"""Primitive editors preserve SI geometry and expose valid existing factories."""

from dataclasses import replace
import math
import os
from pathlib import Path

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import boundary_loops, load_mesh_from_config
from coilgen.gui.editors import SUPPORT_EDITORS

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from coilgen.gui.visual_selectors import SurfaceGallery


def editor(label):
    return next(item for item in SUPPORT_EDITORS if item.label == label)


def mesh_for(item, values):
    return load_mesh_from_config(
        replace(CoilGenConfig(), coil_mesh_file=item.source, **{item.parameter_field: values}), Path.cwd()
    )


@pytest.mark.parametrize("item", SUPPORT_EDITORS, ids=lambda item: item.label)
def test_existing_defaults_make_finite_non_degenerate_mesh_and_untouched_parameters_are_exact(item):
    values = getattr(CoilGenConfig(), item.parameter_field)
    assert item.apply_edits(values, {}) == list(values)
    mesh = mesh_for(item, values)
    assert len(mesh.vertices) > 3 and len(mesh.faces) > 0
    assert np.isfinite(mesh.vertices).all()
    triangles = mesh.vertices[mesh.faces]
    twice_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    assert np.all(twice_area > 0)
    assert item.description and item.icon_name and item.category


@pytest.mark.parametrize("label", ["Partial cylinder", "Elliptical cylinder", "Superellipse"])
def test_axial_refinement_never_uses_legacy_cylinder_span_correction(label):
    item = editor(label)
    values = getattr(CoilGenConfig(), item.parameter_field)
    length = next(field for field in item.fields if field.key == "length_mm")
    divisions = next(field for field in item.fields if field.key == "length_divisions")
    assert not item.cylindrical
    assert item.display_value(length, values) == values[0] * 1000
    refined = item.apply_edits(values, {"length_divisions": values[divisions.index] + 1})
    assert refined[0] == values[0]
    assert np.ptp(mesh_for(item, refined).vertices[:, 2]) == pytest.approx(values[0])


def test_rectangle_dimension_labels_match_native_swapped_local_axes():
    item = editor("Rectangle")
    values = item.apply_edits(CoilGenConfig().planar_mesh_parameter_list, {"height_mm": 170, "width_mm": 290})
    mesh = mesh_for(item, values)
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx((0.17, 0.29, 0))
    assert "local x" in next(field.label for field in item.fields if field.key == "height_mm")
    assert "local y" in next(field.label for field in item.fields if field.key == "width_mm")


@pytest.mark.parametrize("label", ["Spherical cap", "Ellipsoidal cap"])
def test_cap_angle_uses_degrees_and_retains_one_open_rim(label):
    item = editor(label)
    values = getattr(CoilGenConfig(), item.parameter_field)
    opening = next(field for field in item.fields if field.key == "polar_angle_deg")
    assert opening.minimum > 0 and opening.maximum < 180
    assert item.display_value(opening, values) == pytest.approx(90)
    edited = item.apply_edits(values, {"polar_angle_deg": 60})
    assert edited[opening.index] == pytest.approx(math.pi / 3)
    assert len(boundary_loops(mesh_for(item, edited))) == 1
    with pytest.raises(ValueError, match="collapsed rim"):
        item.apply_edits(values, {"polar_angle_deg": 180})


@pytest.mark.parametrize("label", ["Partial cylinder", "Toroidal segment"])
def test_open_segment_angular_bounds_reject_coincident_seams_and_reversed_spans(label):
    item = editor(label)
    values = getattr(CoilGenConfig(), item.parameter_field)
    with pytest.raises(ValueError, match="coincident end seams"):
        item.apply_edits(values, {"start_angle_deg": -180, "end_angle_deg": 180})
    with pytest.raises(ValueError, match="exceed"):
        item.apply_edits(values, {"start_angle_deg": 180, "end_angle_deg": 0})
    with pytest.raises(ValueError, match="span"):
        item.apply_edits(values, {"start_angle_deg": -180, "end_angle_deg": 181})


def test_toroidal_full_tube_seam_is_welded_and_major_ends_stay_open():
    item = editor("Toroidal segment")
    values = CoilGenConfig().toroidal_segment_mesh_parameter_list
    tube_end = next(field for field in item.fields if field.key == "tube_end_angle_deg")
    assert item.display_value(tube_end, values) == pytest.approx(360)
    assert len(boundary_loops(mesh_for(item, values))) == 2
    with pytest.raises(ValueError, match="minor radius"):
        item.apply_edits(values, {"minor_radius_mm": 350})
    with pytest.raises(ValueError, match="span"):
        item.apply_edits(values, {"tube_end_angle_deg": 361})


@pytest.mark.parametrize("label", ["Annulus", "Biplanar", "Circular biplanar"])
def test_world_normal_must_be_nonzero(label):
    item = editor(label)
    values = getattr(CoilGenConfig(), item.parameter_field)
    with pytest.raises(ValueError, match="normal.*nonzero"):
        item.apply_edits(values, {f"plane_normal_{axis}": 0 for axis in "xyz"})


def test_annulus_and_disc_guard_known_empty_or_degenerate_meshes():
    annulus = editor("Annulus")
    with pytest.raises(ValueError, match="use Disc"):
        annulus.apply_edits(CoilGenConfig().annular_disc_mesh_parameter_list, {"inner_radius_mm": 0})
    with pytest.raises(ValueError, match="exceed inner"):
        annulus.apply_edits(CoilGenConfig().annular_disc_mesh_parameter_list, {"outer_radius_mm": 40})
    disc = editor("Disc")
    with pytest.raises(ValueError, match="four divisions"):
        disc.apply_edits(CoilGenConfig().circular_mesh_parameter_list, {"radial_divisions": 2})


def test_circular_biplanar_grid_step_is_physical_and_bounds_preview_cost():
    item = editor("Circular biplanar")
    values = CoilGenConfig().circular_biplanar_mesh_parameter_list
    field = next(field for field in item.fields if field.key == "mesh_edge_mm")
    assert item.display_value(field, values) == 80
    changed = item.apply_edits(values, {"mesh_edge_mm": 40, "separation_mm": 120})
    assert changed[0] == 0.04 and changed[8] == 0.12
    assert np.ptp(mesh_for(item, changed).vertices[:, 2]) == pytest.approx(0.12)
    with pytest.raises(ValueError, match="512 divisions"):
        item.apply_edits(values, {"mesh_edge_mm": 0.01})


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["surface-primitives-test"])


@pytest.fixture
def gallery(application):
    widget = SurfaceGallery()
    widget.resize(340, widget.sizeHint().height())
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_full_primitive_catalog_is_on_demand_and_keeps_three_initial_tiles(gallery):
    assert gallery.count() == 13
    assert [button.accessibleName() for button in gallery.buttons if button.isVisible()] == [
        "Cylinder",
        "Biplanar",
        "Conical frustum",
    ]
    assert gallery.more_button.isVisible()
    assert gallery.height() <= 125
    assert set(gallery.surface_actions) == {item.source for item in SUPPORT_EDITORS}
    assert all(action.text() and not action.icon().isNull() for action in gallery.surface_actions.values())


def test_native_more_surface_menu_selects_real_primitive_and_displays_its_tile(gallery, application):
    source = editor("Ellipsoidal cap").source
    events = []
    gallery.currentIndexChanged.connect(events.append)
    gallery.more_menu.popup(gallery.mapToGlobal(QPoint(0, gallery.height())))
    application.processEvents()
    action = gallery.surface_actions[source]
    rectangle = gallery.more_menu.actionGeometry(action)
    assert not rectangle.isEmpty()
    QTest.mouseClick(gallery.more_menu, Qt.MouseButton.LeftButton, pos=rectangle.center())
    assert gallery.currentData() == source
    assert events == [gallery.findData(source)]
    assert gallery.buttons[gallery.currentIndex()].isVisible()
    assert gallery.buttons[gallery.currentIndex()].isChecked()
    assert len([button for button in gallery.buttons if button.isVisible()]) == 3
    assert gallery.surface_actions[source].isChecked()


def test_loading_non_common_surface_with_blocked_signals_still_shows_active_tile(gallery):
    events = []
    gallery.currentIndexChanged.connect(events.append)
    with QSignalBlocker(gallery):
        gallery.setCurrentText("Toroidal segment")
    assert events == []
    assert gallery.currentData() == editor("Toroidal segment").source
    assert gallery.buttons[gallery.currentIndex()].isVisible()
    assert gallery.surface_actions[gallery.currentData()].isChecked()
    QTest.keyClick(gallery.buttons[gallery.currentIndex()], Qt.Key.Key_Tab)
    assert gallery.more_button.hasFocus()
