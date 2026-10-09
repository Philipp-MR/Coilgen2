from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application import ProjectDocument
from coilgen.application.spatial_scene import (
    ScenePart,
    SpatialScene,
    RouteAnchor,
    path_mesh_clearance,
    evaluate_spatial_scene,
    mesh_scene_clearance,
)
from coilgen.config import CoilGenConfig
from coilgen.gradient_engineering import winding_path_sha256


def test_clearance_checks_continuous_segments_not_just_endpoints():
    box = ScenePart.box("Beam window", size_m=(2, 2, 2))
    assert path_mesh_clearance([[-2, 0, 0], [2, 0, 0]], box)["passed"] is False
    assert path_mesh_clearance([[-2, 0, 0], [2, 0, 0]], box)["witness_segment_points_m"] == [[-2, 0, 0], [2, 0, 0]]
    assert path_mesh_clearance([[0, 0, 0], [0.1, 0, 0]], box)["passed"] is False
    clear = path_mesh_clearance([[-2, 2, 0], [2, 2, 0]], box, radius_m=0.2)
    assert clear["passed"] is True
    assert clear["minimum_centerline_surface_distance_m"] == pytest.approx(1)
    assert path_mesh_clearance([[-2, 1.1, 0], [2, 1.1, 0]], box, radius_m=0.2)["passed"] is False


def test_corridor_checks_envelope_and_never_promotes_surface_contact():
    box = ScenePart.box("Feed corridor", size_m=(2, 2, 2), role="feed_corridor", clearance_m=0.1)
    assert path_mesh_clearance([[0, 0, 0], [0.5, 0, 0]], box, radius_m=0.1)["passed"] is True
    assert path_mesh_clearance([[0, 0, 0], [0.9, 0, 0]], box, radius_m=0.1)["passed"] is False
    assert path_mesh_clearance([[0, 0, 0], [2, 0, 0]], box)["passed"] is False


def test_scene_rejects_open_and_inconsistently_oriented_exclusions():
    part = ScenePart.box("Box")
    with pytest.raises(ValueError, match="closed manifold"):
        ScenePart("Open", "keepout", part.vertices_m, part.faces[:-1])
    faces = part.faces.copy()
    faces[0] = faces[0, ::-1]
    with pytest.raises(ValueError, match="consistent orientation"):
        ScenePart("Reversed", "obstacle", part.vertices_m, faces)


def test_realized_facet_collision_is_checked_beyond_centerline_capsules():
    exclusion = ScenePart.box("Small obstacle", center_m=(0, 0, 0), size_m=(0.2, 0.2, 0.2))
    overhanging = ScenePart.box("Miter extension", center_m=(0.5, 0, 0), size_m=(1, 0.3, 0.3))
    assert mesh_scene_clearance(overhanging._mesh, exclusion)["passed"] is False
    clear = ScenePart.box("Clear", center_m=(2, 0, 0), size_m=(0.2, 0.2, 0.2))
    assert mesh_scene_clearance(clear._mesh, exclusion)["passed"] is True
    enclosing = ScenePart.box("Enclosing", size_m=(1, 1, 1))
    assert mesh_scene_clearance(enclosing._mesh, exclusion)["passed"] is False


def test_containment_checks_every_disconnected_solid_component():
    first = ScenePart.box("Far component", center_m=(5, 0, 0), size_m=(0.2, 0.2, 0.2))
    second = ScenePart.box("Nested component", size_m=(0.2, 0.2, 0.2))
    combined = ScenePart(
        "Multipart obstacle",
        "keepout",
        np.vstack([first.vertices_m, second.vertices_m]),
        np.vstack([first.faces, second.faces + len(first.vertices_m)]),
    )
    conductor = ScenePart.box("Conductor", size_m=(1, 1, 1))
    assert mesh_scene_clearance(conductor._mesh, combined)["passed"] is False
    assert mesh_scene_clearance(combined._mesh, conductor)["passed"] is False


def test_native_swept_facet_failure_is_visible_even_when_centerline_passes():
    mesh = ScenePart.box("Recorded miter", center_m=(0.5, 0, 0), size_m=(1, 0.3, 0.3))._mesh
    part = SimpleNamespace(wire_path=SimpleNamespace(v=np.asarray([[0.5, 1, 0], [0.5, 2, 0]])), swept_body=mesh)
    result = SimpleNamespace(input_data=CoilGenConfig(), coil_parts=[part])
    scene = SpatialScene((ScenePart.box("Sensor", size_m=(0.2, 0.2, 0.2)),))
    report = evaluate_spatial_scene(result, scene)
    row = report["rows"][0]
    assert row["minimum_centerline_surface_distance_m"] > row["required_envelope_gap_m"]
    assert row["passed"] is False
    assert "swept conductor" in row["reason"]
    assert report["checks"][0]["reason"] == row["reason"]
    assert "recorded swept facets" in report["checks"][0]["scope"]


def test_npz_metres_cannot_be_silently_scaled_again(tmp_path):
    part = ScenePart.box("SI cube", size_m=(1, 1, 1))
    path = tmp_path / "cube.npz"
    np.savez(path, schema="coilgen-mesh-v1", vertices_m=part.vertices_m, faces=part.faces)
    with pytest.raises(ValueError, match="already SI"):
        ScenePart.import_mesh(path, name="Cube", role="keepout", length_unit="mm")
    imported = ScenePart.import_mesh(path, name="Cube", role="keepout", length_unit="m")
    assert np.ptp(imported.vertices_m[:, 0]) == 1


def test_scene_inputs_owned_portably_and_do_not_change_default_model(tmp_path):
    project = ProjectDocument.from_config(CoilGenConfig())
    default = project.to_dict()
    assert "scene" not in project.study
    scene = SpatialScene((ScenePart.box("Metal", role="conducting_metal", material="Aluminium"),))
    edited = project.update_study({"scene": scene.to_dict()})
    assert edited.fingerprint != project.fingerprint
    assert edited.model_fingerprint == project.model_fingerprint
    path = edited.save(tmp_path / "scene.json")
    loaded = ProjectDocument.load(path)
    assert loaded.fingerprint == edited.fingerprint
    assert SpatialScene.from_dict(loaded.study["scene"]).digest == scene.digest
    assert edited.update_study({"scene": None}).study == project.study
    assert project.to_dict() == default


def test_complete_feed_obstacle_and_original_cut_interval_witness():
    winding = np.asarray([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0], [-2, -2, 0]], float)
    electrical = np.asarray([[-3, 0, 0], [3, 0, 0], [3, 3, 0], [-3, 0, 0]], float)
    body = SimpleNamespace(
        body_id="body",
        points_m=winding,
        path=SimpleNamespace(cross_section_m=np.asarray([[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01]])),
    )
    plan = {
        "body_id": "body",
        "electrical_points_m": electrical,
        "circuit_complete": True,
        "feed_break_segment": 0,
        "positive_terminal_xyz_m": electrical[0],
        "negative_terminal_xyz_m": electrical[-1],
        "steps": [{"segment_role": "external_supply", "start_sample_index": 0, "end_sample_index": 1}],
    }
    fabrication = SimpleNamespace(paths=[body], winding_steps=[plan])
    result = SimpleNamespace(input_data=CoilGenConfig(), coil_parts=[])
    anchor = RouteAnchor("Cut midpoint", "cut", [0, -2, 0], "body", winding_path_sha256(winding), tolerance_m=1e-9)
    scene = SpatialScene((ScenePart.box("Protected sensor", size_m=(2, 2, 2)),), (anchor,))
    report = evaluate_spatial_scene(result, scene, fabrication=fabrication)
    assert report["feeds_modeled"] is True
    assert report["rows"][0]["passed"] is False  # supply crosses the sensor
    assert report["rows"][1]["passed"] is True  # midpoint lies on removed interval
    stale = SpatialScene(anchors=(RouteAnchor("Stale", "cut", [0, -2, 0], "body", "f" * 64),))
    assert evaluate_spatial_scene(result, stale, fabrication=fabrication)["rows"][0]["passed"] is None


def test_gui_scene_apply_and_explicit_pick_preserve_si_data():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from coilgen.gui.spatial_scene import SpatialSceneDialog

    application = QApplication.instance() or QApplication([])
    project = ProjectDocument.from_config(CoilGenConfig()).update_study(
        {"scene": SpatialScene((ScenePart.box("Corridor", role="feed_corridor"),)).to_dict()}
    )
    dialog = SpatialSceneDialog(project)
    dialog.picked_point([0.001, 0.002, 0.003])
    assert np.allclose([float(v) for v in dialog.point.text().split()], [0.001, 0.002, 0.003])
    assert dialog.value() == project.study["scene"]
    dialog.table.item(0, 2).setText("0.002")
    assert dialog.value()["parts"][0]["clearance_m"] == 0.002
    assert dialog._capture_edits()
    dialog._refresh()
    assert dialog.value()["parts"][0]["clearance_m"] == 0.002
    dialog.close()
    dialog.deleteLater()
    application.processEvents()
