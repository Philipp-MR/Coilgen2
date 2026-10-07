"""Physical realization must preserve sweep geometry without certifying a mesh."""

from copy import deepcopy

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
import coilgen.manufacturing as manufacturing
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry, WireSegmentOwner


def _case(kind):
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    if kind == "closed_surface":
        angle = np.linspace(0, 2 * np.pi, 33)
        points = 0.03 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
        points[-1] = points[0]
        geometry = [WirePathGeometry.SURFACE] * (len(points) - 1)
    elif kind == "surface_fillet":
        points = np.array([[-0.03, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.03, 0.0]])
        geometry = [WirePathGeometry.SURFACE] * 2
    elif kind == "ring_repair":
        points = np.array([[-0.03, 0.0, 0.0], [-0.03 + 1e-8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
        geometry = [WirePathGeometry.SURFACE] * 3
    else:
        points = np.array([[-0.03, 0.0, 0.0], [0.0, 0.0, 0.0], [0.02, 0.015, 0.01], [0.04, 0.015, 0.01]])
        geometry = [WirePathGeometry.SURFACE, WirePathGeometry.FREE_3D, WirePathGeometry.OFFSET_SURFACE]
    owners = [
        WireSegmentOwner("contour_loop", group_index=index // 2, loop_index=index, connector_role="realization_test")
        for index in range(len(geometry))
    ]
    # The asymmetric profile exercises both transverse and normal centroid
    # corrections in addition to the support mounting offset.
    section = np.array([[0.0001, -0.0002], [0.0006, -0.0002], [0.0005, 0.0007], [0.0001, 0.0007]])
    config = CoilGenConfig(
        skip_sweep=False,
        cross_sectional_points=section,
        conductor_mounting_mode="surface_outer",
        sweep_bend_mode="fillet",
        sweep_min_bend_radius=0.004,
        sweep_max_segment_length=0.004,
        sweep_max_frame_rotation_deg=5.0,
    )
    if kind == "ring_repair":
        config.sweep_bend_mode = "miter"
        config.sweep_min_bend_radius = 0.0
        config.sweep_path_simplification = "off"
    return CoilPart(mesh=mesh, wire_path=WirePath(points, segment_owners=owners, segment_geometry=geometry)), config


def _assert_same_fabrication_path(actual, expected):
    for name in ("points_m", "tangents", "normals", "transverse", "cross_section_m", "source_points_m"):
        actual_values, expected_values = getattr(actual, name), getattr(expected, name)
        assert actual_values.shape == expected_values.shape
        assert actual_values.tobytes() == expected_values.tobytes(), name
    assert actual.closed == expected.closed
    assert actual.source_kind == expected.source_kind
    assert actual.segment_geometry == expected.segment_geometry
    assert actual.segment_owners == expected.segment_owners
    assert actual.wire_points_m.tobytes() == expected.wire_points_m.tobytes()


@pytest.mark.parametrize("kind", ["closed_surface", "surface_fillet", "mixed_geometry", "ring_repair"])
def test_realized_path_matches_full_sweep_byte_for_byte(kind):
    part, config = _case(kind)
    full_sweep = deepcopy(part)
    original_route = deepcopy(part.wire_path)
    manufacturing.create_swept_coil_meshes([full_sweep], config)

    parts = [part]
    assert manufacturing.realize_coil_paths(parts, config) is parts
    _assert_same_fabrication_path(part.fabrication_path, full_sweep.fabrication_path)
    assert part.wire_path.v.tobytes() == original_route.v.tobytes()
    assert part.wire_path.segment_owners == original_route.segment_owners
    assert part.wire_path.segment_geometry == original_route.segment_geometry
    for key, value in part.manufacturing_diagnostics.items():
        if key not in {"fabrication_path_created", "swept_body_created", "sweep_geometry_role"}:
            assert value == full_sweep.manufacturing_diagnostics[key], key
    assert part.manufacturing_diagnostics["fabrication_path_created"]
    assert part.manufacturing_diagnostics["sweep_centerline_offset_m"] == 0.0002
    if kind == "ring_repair":
        assert part.manufacturing_diagnostics["sweep_degenerate_ring_points_removed"] == 1

    realized_path = part.fabrication_path
    manufacturing.create_swept_coil_meshes([part], config)
    _assert_same_fabrication_path(part.fabrication_path, realized_path)
    assert part.swept_body.vertices.tobytes() == full_sweep.swept_body.vertices.tobytes()
    assert part.swept_body.faces.tobytes() == full_sweep.swept_body.faces.tobytes()
    assert part.manufacturing_diagnostics == full_sweep.manufacturing_diagnostics


def test_realization_discards_stale_meshes_without_tessellation_or_mesh_certification(monkeypatch):
    part, config = _case("surface_fillet")
    manufacturing.create_swept_coil_meshes([part], config)
    assert part.swept_body is not None
    assert "manufacturing_ok" in part.manufacturing_diagnostics

    def fail_if_called(*args, **kwargs):
        pytest.fail("Physical realization invoked conductor mesh generation or certification")

    for name in (
        "_triangulate_sweep_laterals",
        "_swept_mesh_quality_diagnostics",
        "_sweep_section_conservation_diagnostics",
        "_support_conformity_diagnostics",
        "_final_mesh_collision_diagnostics",
        "mesh_manufacturing_diagnostics",
        "_finalize_manufacturing_diagnostics",
    ):
        monkeypatch.setattr(manufacturing, name, fail_if_called)
    manufacturing.realize_coil_paths([part], config)
    assert part.fabrication_path is not None
    assert part.swept_body is None
    assert part.swept_bodies == []
    assert not part.manufacturing_diagnostics["swept_body_created"]
    for key in ("manufacturing_ok", "mesh_solid_ok", "watertight", "sweep_mesh_collision_broad_phase"):
        assert key not in part.manufacturing_diagnostics


@pytest.mark.parametrize("mode", ["skip_sweep", "contour_only_sweep_mode", "missing_wire", "short_wire"])
def test_non_routed_realization_clears_previous_physical_state(mode):
    part, config = _case("surface_fillet")
    manufacturing.create_swept_coil_meshes([part], config)
    if mode in {"skip_sweep", "contour_only_sweep_mode"}:
        setattr(config, mode, True)
    elif mode == "missing_wire":
        part.wire_path = None
    else:
        part.wire_path = WirePath(np.zeros((1, 3)))
    manufacturing.realize_coil_paths([part], config)
    assert part.fabrication_path is None
    assert part.swept_body is None
    assert part.swept_bodies == []
    assert part.manufacturing_diagnostics == {"fabrication_path_created": False, "swept_body_created": False}
