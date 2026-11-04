from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.service import RunRecord
from coilgen.application.shielding import shielding_sources, shielding_surface_fields
from coilgen.scientific_arrays import shielding_contribution_arrays
from coilgen.physics import biot_savart_wire_path


def test_wire_contributions_use_recorded_signed_current_for_every_support(tmp_path):
    mask = np.array([True, False, True, False])
    points = np.arange(12, dtype=float).reshape(-1, 3) * 0.01
    target = SimpleNamespace(suppressed=mask)
    current = -2.5
    parts = [
        SimpleNamespace(mesh=SimpleNamespace(metadata={"role": role}), field_by_layout=coefficient * points)
        for role, coefficient in (("active_shield", -1.5), ("primary", 1), ("primary", 1))
    ]
    arrays = shielding_contribution_arrays(parts, target, "routed", current_A=current)
    arrays.update(target_suppressed=mask, target_points_m=points, routed_field_T=current * 0.5 * points)
    wire = {"routed": {"current_A": current, "current_source": "target_fit"}}
    record = RunRecord(tmp_path, {"wire_fields": wire}, arrays)
    assert shielding_sources(record) == ("routed",)
    saved = shielding_surface_fields(record, "routed")
    np.testing.assert_allclose(saved.primary_T, 2 * current * points[mask])
    np.testing.assert_allclose(saved.shield_T, -1.5 * current * points[mask])
    np.testing.assert_allclose(saved.total_T, saved.primary_T + saved.shield_T)
    assert saved.current_A == current and saved.current_source == "target_fit"


def test_historical_suppression_samples_use_saved_net_vectors_only(tmp_path):
    fields = np.arange(12, dtype=float).reshape(-1, 3) * 1e-6
    arrays = {
        "target_points_m": fields * 100,
        "sheet_field_T": fields,
        "target_field_T": fields * 2,
        "target_suppressed": [False, True, False, True],
        "contour_field_T": fields[:, 2],  # A scalar cannot supply a vector map.
    }
    record = RunRecord(tmp_path, {}, arrays)
    assert shielding_sources(record) == ("sheet",)
    saved = shielding_surface_fields(record)
    assert saved.primary_T is saved.shield_T is saved.normals is saved.area_weights_m2 is saved.faces is None
    np.testing.assert_array_equal(saved.total_T, fields[[1, 3]])
    np.testing.assert_array_equal(saved.target_T, 2 * fields[[1, 3]])
    for invalid in ([False] * 4, [False] * 3, [0, 1, 2, 0]):
        assert shielding_sources(RunRecord(tmp_path, {}, {**arrays, "target_suppressed": invalid})) == ()
    with pytest.raises(ValueError, match="disagree"):
        shielding_surface_fields(
            RunRecord(
                tmp_path,
                {},
                {**arrays, "shielding_sheet_primary_T": fields[:2], "shielding_sheet_shield_T": fields[:2]},
            )
        )


def test_fabricated_maps_require_identical_samples_and_complete_electrical_current_records(tmp_path):
    path = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], float)
    coords = path[:3] + 0.5
    arrays = {
        "target_points_m": coords,
        "target_suppressed": [False, True, True],
        "fabricated_points_m": coords,
        "fabricated_field_T": coords * 1e-6,
        "review_electrical_0_points_m": path,
    }
    drive = {"current_source": "original_contour_current_no_refit", "currents_A": {"shield": -1.25}}
    metadata = {
        "fabricated_physics": drive,
        "winding_steps": [{"body_id": "shield", "circuit_complete": True}],
        "fabrication": {"electrical_metrics": {"bodies": [{"body_id": "shield", "current_A": -1.25}]}},
    }
    record = RunRecord(tmp_path, metadata, arrays)
    assert shielding_sources(record) == ("fabricated",)
    saved = shielding_surface_fields(record, "fabricated")
    assert saved.primary_T is saved.shield_T is None
    assert saved.current_A == -1.25
    np.testing.assert_array_equal(saved.total_T, arrays["fabricated_field_T"][1:])
    for new_arrays, new_metadata in (
        ({"fabricated_points_m": coords[::-1]}, {}),
        ({"review_electrical_0_points_m": path[:-1]}, {}),
        ({}, {"winding_steps": [{"body_id": "shield", "circuit_complete": False}]}),
        ({}, {"fabricated_physics": {**drive, "currents_A": {"shield": 1.25}}}),
    ):
        assert shielding_sources(RunRecord(tmp_path, {**metadata, **new_metadata}, {**arrays, **new_arrays})) == ()


def test_direct_routed_quadrature_uses_actual_field_and_area_on_new_points(tmp_path):
    from coilgen.progress import DesignCancelled

    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0, 0, 0]])
    points = np.array([[0.02, 0.02, 0.01], [0.08, 0.02, 0.01], [0.04, 0.08, 0.07]])
    faces = np.array([[0, 1, 2]])
    current = -5.5
    field = current * biot_savart_wire_path(path, points)
    arrays = {
        "target_points_m": points,
        "target_suppressed": [True] * 3,
        "target_field_T": field * 3,
        "shielding_surface_faces": faces,
        "routed_field_T": field,
        "evaluated_winding_0_points_m": path,
    }
    metadata = {
        "wire_fields": {
            "routed": {
                "available": True,
                "current_A": current,
                "current_source": "target_fit",
                "geometry_sources": ["connected_wire"],
            }
        }
    }
    record = RunRecord(tmp_path, metadata, arrays)
    assert shielding_sources(record) == ("routed", "routed_interior")
    out = shielding_surface_fields(record, "routed_interior")
    expected = np.array(
        [(4 * a + b + c) / 6 for a, b, c in (points, np.roll(points, 1, axis=0), np.roll(points, 2, axis=0))]
    )
    np.testing.assert_allclose(np.sort(out.points_m, axis=0), np.sort(expected, axis=0))
    np.testing.assert_allclose(out.total_T, current * biot_savart_wire_path(path, out.points_m))
    assert out.sampling == "triangle_interior" and out.target_T is None and out.faces is None
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    np.testing.assert_allclose(out.area_weights_m2, np.linalg.norm(normal) / 6)
    np.testing.assert_allclose(out.normals, np.tile(normal / np.linalg.norm(normal), (3, 1)))
    assert out.current_A == current and out.current_source == "target_fit"
    moved = RunRecord(tmp_path, metadata, {**arrays, "evaluated_winding_0_points_m": path + [0, 0, 0.02]})
    with pytest.raises(ValueError, match="do not reproduce"):
        shielding_surface_fields(moved, "routed_interior")
    with pytest.raises(DesignCancelled):
        shielding_surface_fields(record, "routed_interior", cancel_check=lambda: True)
    assert "routed_interior" not in shielding_sources(
        RunRecord(tmp_path, metadata, {**arrays, "shielding_surface_faces": np.tile(faces, (20_001, 1))})
    )
