"""The 3D-print former is a visible, optional Model section with a preview before computing."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("manifold3d")

import test_workbench_usability as usability  # noqa: E402

from coilgen.application.presets import preset_config  # noqa: E402
from coilgen.application.scene_spec import model_preview, scene_layers  # noqa: E402

application = usability.application
windows = usability.windows


def test_print_former_is_listed_off_and_one_checkbox_turns_it_on(windows):
    window = windows()
    item = window.tree_items["print"]
    assert item.parent() is window._model_root and item.text(0) == "3D-print former"
    assert item.text(1) == "Add" and "Click to add" in item.toolTip(1)

    window.select_node("print")
    original = window.project
    controls = window._draft_controls
    enabled = controls["support_body_enabled"]
    assert not enabled.isChecked()
    # While off, only the switch is shown; the details appear once it is on.
    assert controls["support_body_thickness_mm"].parentWidget().isHidden()
    enabled.setChecked(True)
    assert not controls["support_body_thickness_mm"].parentWidget().isHidden()
    assert controls["support_body_side"].currentText() == "inside"
    controls["support_body_side"].setCurrentText("outside")
    controls["support_body_thickness_mm"].setValue(2.5)
    assert window.project is original
    assert window.apply_settings()
    config = window.project.config
    assert window.project.revision == original.revision + 1
    assert config.support_body_mode == "outside" and config.support_body_thickness == pytest.approx(0.0025)
    assert window.tree_items["print"].text(1) == "Ready"

    window.select_node("print")
    window._draft_controls["support_body_thickness_mm"].setValue(4)
    window._draft_controls["support_body_enabled"].setChecked(False)
    assert window.apply_settings()
    assert window.project.config.support_body_mode == "off"
    assert window.project.config.support_body_thickness == pytest.approx(0.0025)
    assert window.tree_items["print"].text(1) == "Add"
    window.undo_stack.undo()
    assert window.project.config == config
    window.undo_stack.undo()
    assert window.project.config == original.config


def test_material_page_no_longer_hides_the_former_among_conductor_settings(windows):
    window = windows()
    window.select_node("material")
    assert not any(key.startswith("support_body") for key in window._draft_controls)


@pytest.mark.parametrize("mode", ["inside", "outside"])
def test_support_preview_shows_the_former_wall_before_computing(mode):
    config = preset_config()
    _, off_arrays, _, _ = model_preview(config, Path("."))
    assert "former_preview_vertices_m" not in off_arrays
    mesh, arrays, _, _ = model_preview(replace(config, support_body_mode=mode), Path("."))
    vertices, faces = arrays["former_preview_vertices_m"], arrays["former_preview_faces"]
    assert len(vertices) == 2 * len(mesh.vertices) and faces.max() < len(vertices)
    radii = np.linalg.norm(np.asarray(vertices)[:, :2], axis=1)
    support = np.linalg.norm(np.asarray(mesh.vertices)[:, :2], axis=1)
    # The wall is the support radius plus or minus the 3 mm thickness.
    assert np.ptp(radii) == pytest.approx(config.support_body_thickness, rel=0.05)
    assert radii.min() < support.min() or radii.max() > support.max()
    layers, _ = scene_layers({"vertices_m": mesh.vertices, "faces": mesh.faces, **arrays}, "support")
    assert [layer.name for layer in layers] == ["support", "support_edges", "former_preview"]
