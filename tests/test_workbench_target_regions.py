"""World-coordinate target-region edits and portable mesh imports through the GUI."""

from dataclasses import replace
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest
import test_workbench_usability as usability

from PySide6.QtWidgets import QDockWidget
from PySide6.QtTest import QTest

from coilgen.application import ProjectDocument
from coilgen.application.assets import mesh_asset
from coilgen.application.presets import preset_config
from coilgen.application.targets import SolidHarmonicTarget, recognize_solid_harmonic_target
from coilgen.geometry import load_mesh_from_config, load_optional_target_mesh
from coilgen.gui.region_selectors import RegionSamplingSelector, TargetRegionGallery
from coilgen.models import Mesh
from coilgen.target import define_target_field


application = usability.application
windows = usability.windows


def _target_window(windows, **overrides):
    project = ProjectDocument.from_config(replace(preset_config(), **overrides))
    window = windows(project)
    window.select_node("target")
    return window


def _gallery(window):
    result = window.settings.findChild(TargetRegionGallery, "target_region_shape")
    assert result is not None
    return result


def _sampling(window):
    result = window.settings.findChild(RegionSamplingSelector, "target_region_sampling")
    assert result is not None
    return result


def _field(window, mesh=None):
    config = window.project.config
    region = config.target_region_geometry
    if (
        mesh is None
        and config.target_field_definition_file == "none"
        and region is not None
        and region["shape"] == "enclosing_mesh"
    ):
        mesh = load_optional_target_mesh(config.target_mesh_file, config, window.project.project_root)
    return define_target_field([], mesh, None, config)


def _choose_region(window, shape):
    gallery = _gallery(window)
    gallery.setCurrentIndex(gallery.findData(shape))


def _surface_file(tmp_path, *, closed=False):
    vertices = np.array([[10.0, 20, 30], [50, 20, 30], [10, 60, 30]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    if closed:
        vertices = np.array([[0.0, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, 40]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    path = tmp_path / ("closed_region.stl" if closed else "open_region.stl")
    lines = ["solid region"]
    for face in faces:
        lines.extend(["facet normal 0 0 0", "outer loop"])
        lines.extend("vertex " + " ".join(str(value) for value in vertices[index]) for index in face)
        lines.extend(["endloop", "endfacet"])
    lines.append("endsolid region")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path, Mesh(vertices, faces)


def test_untouched_legacy_sphere_remains_bit_exact_and_does_not_create_a_revision(windows):
    strength = 0.023456789123456789
    window = _target_window(windows, target_gradient_strength=strength)
    original = window.project
    before = _field(window)
    assert _gallery(window).currentData() == "sphere"
    assert "local bounding-box" not in window._draft_controls["target_region_resolution"].toolTip()
    assert window.apply_settings()
    assert window.project is original
    assert window.project.config.target_region_geometry is None
    assert window.project.config.target_gradient_strength == strength
    after = _field(window)
    np.testing.assert_array_equal(after.coords, before.coords)
    np.testing.assert_array_equal(after.b, before.b)
    assert window.undo_stack.count() == 0


def test_box_gallery_choice_is_one_undoable_si_region_edit(windows):
    window = _target_window(windows)
    original = window.project
    _choose_region(window, "box")
    region = window.project.config.target_region_geometry
    assert region["shape"] == "box" and region["sampling"] == "volume"
    assert region["size_m"] == pytest.approx([2 * original.config.target_region_radius] * 3)
    assert window.project.revision == original.revision + 1
    assert window.undo_stack.count() == 1
    window.undo_stack.undo()
    assert window.project is original and window.project.config.target_region_geometry is None
    window.undo_stack.redo()
    assert _gallery(window).currentData() == "box"


@pytest.mark.parametrize("shape", ("ellipsoid", "cylinder"))
def test_mm_dimensions_preserve_unrelated_high_precision_harmonic_parameters(windows, shape):
    recipe = SolidHarmonicTarget(2, 0, 1.234567891234567e-5, 0.05123456789123456, (0.001, -0.002, 0.003))
    window = _target_window(windows, **recipe.to_config_overrides())
    _choose_region(window, shape)
    controls = window._draft_controls
    if shape == "ellipsoid":
        controls["roi_size_x_mm"].setValue(30)
        controls["roi_size_y_mm"].setValue(40)
        controls["roi_size_z_mm"].setValue(50)
    else:
        controls["target_region_radius"].setValue(12)
        controls["roi_length_mm"].setValue(70)
    assert window.apply_settings()
    region = window.project.config.target_region_geometry
    if shape == "ellipsoid":
        assert region["semi_axes_m"] == pytest.approx([0.015, 0.02, 0.025])
    else:
        assert region["radius_m"] == pytest.approx(0.012)
        assert region["length_m"] == pytest.approx(0.07)
    actual = recognize_solid_harmonic_target(window.project.config.field_shape_function)
    assert actual == recipe


def test_surface_and_volume_sampling_are_distinct_world_field_snapshots(windows):
    window = _target_window(windows, field_shape_function="0.005*x+0.01*y-0.02*z", target_gradient_normalization="off")
    _choose_region(window, "box")
    volume = _field(window)
    assert "local bounding-box axis" in window._draft_controls["target_region_resolution"].toolTip()
    half = np.asarray(window.project.config.target_region_geometry["size_m"]) / 2
    assert np.all(np.abs(volume.coords) < half)
    _sampling(window).setCurrentIndex(_sampling(window).findData("surface"))
    surface = _field(window)
    assert "angular or face resolution" in window._draft_controls["target_region_resolution"].toolTip()
    assert window.project.config.target_region_geometry["sampling"] == "surface"
    assert np.all(np.any(np.isclose(np.abs(surface.coords), half, rtol=0, atol=1e-15), axis=1))
    assert not np.array_equal(volume.coords, surface.coords)
    for field in (volume, surface):
        np.testing.assert_allclose(field.b[:, 2], field.coords @ np.array([0.005, 0.01, -0.02]), rtol=1e-13, atol=1e-18)
    submitted = window.compute()
    assert submitted is not None
    assert window.service.submissions[-1][0].config.target_region_geometry["sampling"] == "surface"


def test_region_pose_moves_samples_and_preserves_harmonic_world_center(windows):
    recipe = SolidHarmonicTarget(2, 1, -1.234567891234567e-5, 0.05, (0.001, -0.002, 0.003))
    window = _target_window(windows, **recipe.to_config_overrides())
    _choose_region(window, "box")
    local = _field(window).coords.copy()
    controls = window._draft_controls
    controls["roi_center_m_x"].setValue(12)
    controls["roi_center_m_y"].setValue(-8)
    controls["roi_center_m_z"].setValue(5)
    controls["roi_angle_deg"].setValue(90)
    assert window.apply_settings()
    region = window.project.config.target_region_geometry
    assert region["center_m"] == pytest.approx([0.012, -0.008, 0.005])
    assert region["rotation_angle_rad"] == pytest.approx(np.pi / 2)
    expected = local @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T + region["center_m"]
    np.testing.assert_allclose(_field(window).coords, expected, atol=1e-17, rtol=1e-14)
    assert recognize_solid_harmonic_target(window.project.config.field_shape_function) == recipe
    assert not window.project.config.set_roi_into_mesh_center


def test_imported_map_keeps_exact_coordinates_and_disables_generated_region_controls(windows, tmp_path):
    path = tmp_path / "field.npz"
    coords = np.array([[0.001, 0.002, 0.003], [-0.004, 0.005, -0.006], [0.007, -0.008, 0.009]])
    b = np.zeros_like(coords)
    b[:, 2] = [-1e-5, 2e-5, 3e-5]
    np.savez(path, coords=coords, b=b)
    window = _target_window(windows)
    _choose_region(window, "ellipsoid")
    window.import_target(path)
    assert _gallery(window).currentData() == "imported_map"
    assert not _gallery(window).isEnabled() and not _sampling(window).isEnabled()
    assert not window._draft_controls["target_region_resolution"].isEnabled()
    for key in ("roi_size_x_mm", "roi_size_y_mm", "roi_size_z_mm"):
        if key in window._draft_controls:
            assert not window._draft_controls[key].isEnabled()
    original = window.project
    assert window.apply_settings()
    assert window.project is original
    field = _field(window)
    np.testing.assert_array_equal(field.coords, coords)
    np.testing.assert_array_equal(field.b, b)


@pytest.mark.parametrize("vertices_only", (False, True))
def test_legacy_target_mesh_tile_preserves_its_original_sampling(windows, tmp_path, vertices_only):
    path, raw = _surface_file(tmp_path)
    mesh = Mesh(raw.vertices * 0.001, raw.faces)
    path = tmp_path / "legacy-region.npz"
    path.write_bytes(mesh_asset(mesh).data)
    window = _target_window(windows, target_mesh_file=str(path), use_only_target_mesh_verts=vertices_only)
    original = window.project
    before = _field(window, mesh)
    gallery = _gallery(window)
    assert gallery.currentData() == "legacy_mesh"
    assert not gallery.buttons[gallery.currentIndex()].isEnabled()
    assert not _sampling(window).isEnabled()
    assert window.apply_settings()
    assert window.project is original and window.project.config.target_region_geometry is None
    np.testing.assert_array_equal(_field(window, mesh).coords, before.coords)
    np.testing.assert_array_equal(_field(window, mesh).b, before.b)


def test_missing_enclosing_mesh_blocks_compute_with_target_action(windows):
    window = _target_window(windows, target_region_geometry={"shape": "enclosing_mesh", "sampling": "volume"})
    assert window.compute() is None
    assert not window.service.submissions
    issues = window.readiness_panel.issues
    assert any(
        issue.blocking and issue.node == "target" and issue.action and "mesh" in issue.message.lower()
        for issue in issues
    )
    assert not window.findChild(QDockWidget, "settings_dock").isHidden()


def test_surface_region_import_is_owned_scaled_once_and_portable_after_source_deletion(windows, tmp_path):
    path, raw = _surface_file(tmp_path)
    window = _target_window(windows, imported_mesh_length_scale_to_m=0.37)
    support = window.project.config.coil_mesh_file
    window.import_region_mesh(path, length_scale_to_m=0.001, sampling="surface")
    assert "6 × resolution²" in window._draft_controls["target_region_resolution"].toolTip()
    assert window.project.values["target_mesh_file"].startswith("asset:")
    assert window.project.config.imported_mesh_length_scale_to_m == 0.37
    assert window.project.config.coil_mesh_file == support
    assert window.project.config.target_region_geometry["sampling"] == "surface"
    config = window.project.config
    assert Path(config.target_mesh_file).suffix == ".npz"
    mesh = load_optional_target_mesh(config.target_mesh_file, config, window.project.project_root)
    np.testing.assert_array_equal(np.unique(mesh.vertices, axis=0), np.unique(raw.vertices * 0.001, axis=0))
    field = _field(window, mesh)
    np.testing.assert_allclose(field.coords[:, 2], 0.03, rtol=0, atol=1e-16)
    assert np.all(field.coords[:, 0] >= 0.01) and np.all(field.coords[:, 0] <= 0.05)
    path.unlink()
    saved = window.save_project(tmp_path / "portable-region.coilgen.json")
    restored = windows(ProjectDocument.load(saved), tmp_path / "fresh-runtime" / "jobs")
    recovered = _field(restored)
    np.testing.assert_array_equal(recovered.coords, field.coords)
    np.testing.assert_array_equal(recovered.b, field.b)


def test_canonical_npz_region_coordinates_are_si_and_reject_a_second_scale(windows, tmp_path):
    _path, raw = _surface_file(tmp_path)
    source = tmp_path / "surface-si.npz"
    source.write_bytes(mesh_asset(Mesh(raw.vertices * 0.001, raw.faces)).data)
    window = _target_window(windows, imported_mesh_length_scale_to_m=0.37)
    window.import_region_mesh(source, length_scale_to_m=1.0, sampling="surface")
    np.testing.assert_allclose(_field(window).coords[:, 2], 0.03, rtol=0, atol=1e-16)
    assert window.project.config.imported_mesh_length_scale_to_m == 0.37
    original = window.project
    count = window.undo_stack.count()
    with pytest.raises(ValueError, match="NPZ|SI|scale"):
        window.import_region_mesh(source, length_scale_to_m=0.001, sampling="surface")
    assert window.project is original
    assert window.undo_stack.count() == count


def test_open_mesh_is_rejected_for_volume_without_changing_the_project(windows, tmp_path):
    path, _raw = _surface_file(tmp_path)
    window = _target_window(windows)
    original = window.project
    with pytest.raises(ValueError, match="closed|watertight|shell"):
        window.import_region_mesh(path, length_scale_to_m=0.001, sampling="volume")
    assert window.project is original
    assert window.undo_stack.count() == 0


def test_closed_mesh_volume_samples_stay_inside_actual_scaled_enclosure(windows, tmp_path):
    path, _raw = _surface_file(tmp_path, closed=True)
    window = _target_window(windows, target_region_resolution=7)
    window.import_region_mesh(path, length_scale_to_m=0.001, sampling="volume")
    coords = _field(window).coords
    assert len(coords) > 0
    assert np.all(coords > 0)
    assert np.all(np.sum(coords, axis=1) < 0.04)


@pytest.mark.parametrize(
    "region",
    [
        {"shape": "sphere", "radius_m": 0.02123456789123456},
        {"shape": "ellipsoid", "semi_axes_m": [0.01, 0.02123456789123456, 0.03]},
        {"shape": "box", "size_m": [0.02, 0.04123456789123456, 0.06]},
        {"shape": "cylinder", "radius_m": 0.02123456789123456, "length_m": 0.07},
    ],
    ids=("sphere", "ellipsoid", "box", "cylinder"),
)
def test_minimal_region_dict_keeps_implicit_defaults_until_deliberate_pose_edit(windows, region):
    original_region = deepcopy(region)
    window = _target_window(windows, target_region_geometry=region)
    original = window.project
    before = _field(window)
    assert _gallery(window).currentData() == region["shape"]
    assert _sampling(window).currentData() == "volume"
    assert window.apply_settings()
    assert window.project is original
    assert window.project.config.target_region_geometry == original_region
    assert window.undo_stack.count() == 0
    window._draft_controls["roi_center_m_x"].setValue(12)
    assert window.apply_settings()
    actual = window.project.config.target_region_geometry
    assert actual == {**original_region, "center_m": [0.012, 0.0, 0.0]}
    np.testing.assert_allclose(_field(window).coords, before.coords + [0.012, 0, 0], atol=1e-17, rtol=1e-14)
    window.undo_stack.undo()
    assert window.project is original


@pytest.mark.parametrize("switch", ("primitive", "sampling"))
def test_switch_from_legacy_centroid_region_retains_support_world_center(windows, switch):
    window = _target_window(
        windows,
        coil_mesh_file="create bi-planary mesh",
        surface_is_cylinder_flag=False,
        biplanar_mesh_parameter_list=(0.2, 0.16, 6, 5, 1, 2, -3, 0.012, -0.008, 0.005, 0.12),
        set_roi_into_mesh_center=True,
        field_shape_function="0.005*x+0.01*y-0.02*z",
        target_gradient_normalization="off",
    )
    original = window.project
    support = load_mesh_from_config(original.config, original.project_root)
    physical_center = np.mean(support.vertices, axis=0)
    legacy = define_target_field([support], None, None, original.config)
    np.testing.assert_allclose(np.mean(legacy.coords, axis=0), physical_center, atol=1e-16)
    if switch == "primitive":
        _choose_region(window, "box")
    else:
        sampling = _sampling(window)
        sampling.setCurrentIndex(sampling.findData("surface"))
    region = window.project.config.target_region_geometry
    np.testing.assert_array_equal(region["center_m"], physical_center)
    assert not window.project.config.set_roi_into_mesh_center
    field = _field(window)
    np.testing.assert_allclose(field.b[:, 2], field.coords @ [0.005, 0.01, -0.02], rtol=1e-13, atol=1e-18)
    assert window.project.config.field_shape_function == original.config.field_shape_function
    assert window.undo_stack.count() == 1
    window.undo_stack.undo()
    assert window.project is original


@pytest.mark.parametrize("selector", ("primitive", "sampling"))
def test_region_choice_preserves_incomplete_numeric_draft_and_previous_selection(windows, selector):
    window = _target_window(windows)
    original = window.project
    radius = window._draft_controls["target_region_radius"]
    radius.lineEdit().selectAll()
    QTest.keyClicks(radius.lineEdit(), "1e")
    window._draft_controls["target_region_resolution"].setValue(21)
    if selector == "primitive":
        _choose_region(window, "box")
    else:
        sampling = _sampling(window)
        sampling.setCurrentIndex(sampling.findData("surface"))
    assert window.project is original and window.undo_stack.count() == 0
    assert _gallery(window).currentData() == "sphere"
    assert _sampling(window).currentData() == "volume"
    assert radius.lineEdit().text().startswith("1e")
    assert window._draft_controls["target_region_resolution"].value() == 21
    assert window._draft_dirty and not window.draft_error_label.isHidden()
    radius.lineEdit().selectAll()
    QTest.keyClicks(radius.lineEdit(), "25")
    assert window.apply_settings()
    assert window.project.config.target_region_geometry is None
    assert window.project.config.target_region_radius == pytest.approx(0.025)
    assert window.project.config.target_region_resolution == 21


@pytest.mark.parametrize(
    "ignored_region",
    [
        {"shape": "invalid", "size_m": [-1]},
        {"shape": "ellipsoid", "semi_axes_m": [0, -1]},
        ["old", "ignored", "region"],
        "old ignored region",
        {},
    ],
    ids=("unknown-shape", "invalid-dimensions", "list", "string", "empty-dict"),
)
def test_imported_map_preserves_ignored_malformed_region_without_parsing_or_repair(windows, tmp_path, ignored_region):
    path = tmp_path / "field.npz"
    coords = np.array([[0.001, 0.002, 0.003], [-0.004, 0.005, -0.006], [0.007, -0.008, 0.009]])
    b = np.zeros_like(coords)
    b[:, 2] = [-1e-5, 2e-5, 3e-5]
    np.savez(path, coords=coords, b=b)
    window = _target_window(
        windows,
        target_field_definition_file=str(path),
        target_region_geometry=ignored_region,
        set_roi_into_mesh_center=True,
        target_gradient_normalization="off",
    )
    original = window.project
    assert _gallery(window).currentData() == "imported_map"
    assert not _gallery(window).isEnabled() and not _sampling(window).isEnabled()
    assert window._target_region_editor.region == ignored_region
    assert not window._draft_controls["target_region_radius"].isEnabled()
    assert not window._draft_controls["target_region_resolution"].isEnabled()
    window._reset_region_choices()
    assert _gallery(window).currentData() == "imported_map"
    window._switch_target_region("box")
    window._switch_target_region_sampling("surface")
    assert window.project is original
    assert window.apply_settings()
    assert window.project is original
    assert window.project.config.target_region_geometry == ignored_region
    assert window.project.config.set_roi_into_mesh_center
    assert window.undo_stack.count() == 0
    field = _field(window)
    np.testing.assert_array_equal(field.coords, coords)
    np.testing.assert_array_equal(field.b, b)


@pytest.mark.parametrize(
    "invalid_region",
    (
        {"shape": "unknown"},
        ["invalid", "active", "region"],
        {"shape": "box", "size_m": [-0.01, 0.02, 0.03]},
    ),
    ids=("unknown-shape", "non-object", "same-shape-invalid-dimensions"),
)
def test_invalid_active_region_opens_target_and_can_be_replaced_undoably(windows, invalid_region):
    window = _target_window(windows, target_region_geometry=invalid_region)
    original = window.project
    editor = window._target_region_editor
    assert editor.region == invalid_region and editor.region_error
    assert _gallery(window).currentIndex() == -1
    assert not any(button.isChecked() for button in _gallery(window).buttons)
    assert not _sampling(window).isEnabled()
    assert not window._draft_controls["target_region_radius"].isEnabled()
    assert "roi_center_m_x" not in window._draft_controls
    assert window.apply_settings()
    assert window.project is original and window.undo_stack.count() == 0
    assert window.compute() is None and not window.service.submissions
    assert any(issue.blocking and issue.node == "target" for issue in window.readiness_panel.issues)
    _choose_region(window, "box")
    assert window.project.config.target_region_geometry["shape"] == "box"
    assert _gallery(window).currentData() == "box"
    assert window.undo_stack.count() == 1
    assert len(_field(window).coords) > 0
    window.undo_stack.undo()
    assert window.project is original
    assert window._target_region_editor.region == invalid_region
    assert _gallery(window).currentIndex() == -1
