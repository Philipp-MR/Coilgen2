"""Body roles survive STL import, use one vocabulary, and label bodies without renaming them."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.cad import GeometryOption, geometry_option_label
from coilgen.application.support_import import (
    body_roles,
    default_part_roles,
    read_support_stl,
    support_import_summary,
    with_part_roles,
)
from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
from coilgen.manufacturing import save_mesh_as_stl
from coilgen.models import Mesh
from coilgen.part_roles import body_display_name, normalize_part_role


def _tube(radius_mm, length_mm, *, sides=24, rings=6):
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    z = np.linspace(-length_mm / 2, length_mm / 2, rings)
    vertices = np.array([[radius_mm * np.cos(t), radius_mm * np.sin(t), h] for h in z for t in theta])
    faces = []
    for ring in range(rings - 1):
        for side in range(sides):
            a, b = ring * sides + side, ring * sides + (side + 1) % sides
            faces += [[a, b, a + sides], [b, b + sides, a + sides]]
    return vertices, np.array(faces)


def _pair_stl(path, *, shield_first=True):
    """A shielded pair in millimetres; the shield's triangles come first in the file."""
    tubes = [_tube(300.0, 600.0), _tube(200.0, 500.0)]
    if not shield_first:
        tubes.reverse()
    vertices, faces, offset = [], [], 0
    for v, f in tubes:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    save_mesh_as_stl(Mesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces)), path)
    return path


def test_summary_reports_bodies_size_loops_and_defaults_inner_to_primary(tmp_path):
    mesh = read_support_stl(_pair_stl(tmp_path / "pair.stl"), 1e-3)
    summary = support_import_summary(mesh)
    assert [body["triangles"] for body in summary["bodies"]] == [240, 240]
    # Roles follow pipeline body order, not radius order: the shield was written first.
    assert summary["roles"] == ["active_shield", "primary"]
    assert summary["needs_roles"] and summary["boundary_loops"] == 4
    assert summary["size_mm"] == pytest.approx([600, 600, 600])
    assert summary["text"].startswith("2 bodies · 480 triangles · ")
    assert "4 boundary loops" in summary["text"]
    assert summary["warnings"] == []


def test_wrong_unit_is_flagged(tmp_path):
    summary = support_import_summary(read_support_stl(_pair_stl(tmp_path / "pair.stl"), 1e-6))
    assert any("Check the STL scale" in warning for warning in summary["warnings"])
    summary = support_import_summary(read_support_stl(_pair_stl(tmp_path / "pair.stl"), 1.0))
    assert any("Check the STL scale" in warning for warning in summary["warnings"])


def test_bodies_without_a_radial_gap_are_all_primaries():
    halves = [SimpleNamespace(vertices=np.array([[0.2, 0, 0], [0, 0.2, 0]])) for _ in range(2)]
    assert default_part_roles(halves, center_xy=np.zeros(2)) == ["primary", "primary"]


def test_one_vocabulary_for_the_shield_role():
    assert normalize_part_role("shield") == normalize_part_role("active_shield") == "active_shield"
    assert normalize_part_role("Active shield") == "active_shield"
    assert normalize_part_role("auxiliary") == "other"
    assert normalize_part_role("suppression") is None and normalize_part_role(None) is None
    assert body_display_name("part001_body0001", "shield") == "Active shield · part001_body0001"
    assert body_display_name("part001_body0001", None) == "part001_body0001"


def test_roles_are_validated_against_the_pipeline_split(tmp_path):
    mesh = read_support_stl(_pair_stl(tmp_path / "pair.stl"), 1e-3)
    with pytest.raises(ValueError, match="2 bodies but 1 roles"):
        with_part_roles(mesh, ["primary"])
    with pytest.raises(ValueError, match="Each body needs a role"):
        with_part_roles(mesh, ["primary", "coil"])
    tagged = with_part_roles(mesh, ["shield", "primary"])
    assert tagged.metadata["part_roles"] == ["active_shield", "primary"]
    assert "part_roles" not in mesh.metadata


def test_browser_session_import_keeps_roles_through_the_owned_asset(tmp_path):
    from coilgen.application.session import WorkbenchSession

    session = WorkbenchSession(tmp_path / "workspace")
    session.new_project("Shielded pair")
    session.import_support_mesh(_pair_stl(tmp_path / "pair.stl"), length_scale_to_m=1e-3)
    project = session.project
    loaded = load_mesh_from_config(project.config, project.project_root)
    assert loaded.metadata["part_roles"] == ["active_shield", "primary"]
    # The pipeline's own split of the reloaded mesh puts the shield first, as recorded.
    bodies = split_disconnected_mesh(loaded)
    radii = [np.mean(np.linalg.norm(body.vertices[:, :2], axis=1)) for body in bodies]
    assert radii[0] > radii[1]
    session.import_support_mesh(
        _pair_stl(tmp_path / "pair.stl"), length_scale_to_m=1e-3, part_roles=["other", "primary"]
    )
    project = session.project
    assert load_mesh_from_config(project.config, project.project_root).metadata["part_roles"] == ["other", "primary"]


def test_saved_roles_label_problems_and_exports_without_renaming_ids(tmp_path):
    from coilgen.application.service import RunRecord
    from coilgen.web.result_display import problem_groups

    metadata = {
        "fabrication": {"paths": [{"body_id": "part001_body0001", "role": "primary"}]},
        "pipeline_sweeps": {"bodies": [{"body_id": "pipeline_part002_body0001", "role": "active_shield"}]},
    }
    assert body_roles(metadata) == {
        "part001_body0001": "primary",
        "pipeline_part002_body0001": "active_shield",
    }
    issue = SimpleNamespace(
        key="fabrication:part001_body0001:minimum_bend_radius_violation",
        scope="fabrication_review",
        severity="error",
        title="Conductor bend radius is too small",
        body_id="part001_body0001",
        unit="m",
        value=None,
        limit=None,
        requirement="",
        comparator="",
        explanation="",
    )
    groups = problem_groups(RunRecord(tmp_path, metadata, {}), [issue])
    assert groups[0]["items"][0]["details"] == "Failed · Primary · part001_body0001"
    option = GeometryOption(
        "part001_body0001.stl",
        "part001_body0001",
        "stl",
        "fabrication_review_geometry",
        "passed",
        "m",
        ((0, 0, 0), (1, 1, 1)),
        False,
        "0" * 64,
        "active_shield",
    )
    assert geometry_option_label(option) == "Active shield · part001_body0001 · STL solid"


def test_desktop_import_dialog_returns_one_role_per_body(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from coilgen.gui.support_import_dialog import SupportImportDialog

    app = QApplication.instance() or QApplication([])
    summary = support_import_summary(read_support_stl(_pair_stl(tmp_path / "pair.stl"), 1e-3))
    dialog = SupportImportDialog(summary, "pair.stl")
    try:
        assert dialog.roles() == ["active_shield", "primary"]
        dialog.role_boxes[0].setCurrentIndex(dialog.role_boxes[0].findData("other"))
        assert dialog.roles() == ["other", "primary"]
        assert dialog.summary.text() == summary["text"] and dialog.warnings.isHidden()
    finally:
        dialog.deleteLater()
        app.processEvents()
