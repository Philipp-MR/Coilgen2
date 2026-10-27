from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import (
    CoilGenConfig,
    CoilPart,
    ContourLine,
    Mesh,
    WirePath,
    create_swept_coil_bodies,
    create_swept_coil_meshes,
)
from coilgen.manufacturing import (
    _SurfaceProjector,
    _finalize_manufacturing_diagnostics,
    _segment_distance,
    _swept_mesh_self_intersection_diagnostics,
    manufacturing_diagnostics_rows,
    save_swept_bodies_as_step,
    save_swept_bodies_as_stl,
)
from coilgen.geometry import build_cylinder_mesh, normalize_mesh


def _closed_circle(radius: float, *, count: int = 64) -> np.ndarray:
    # The frame limit applies to the complete frame, including tangent turn.
    # Keep this valid-export fixture below the default 10-degree bound.
    angles = 2.0 * np.pi * np.arange(count, dtype=float) / count
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)))
    return np.vstack((points, points[0]))


def _planar_contour_part() -> CoilPart:
    support = Mesh(
        vertices=np.asarray(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ]
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    return CoilPart(
        mesh=support,
        contour_lines=[
            ContourLine(v=_closed_circle(1.2), potential=1.0),
            ContourLine(v=_closed_circle(0.7), potential=2.0),
        ],
    )


def _octagonal_sweep_config() -> CoilGenConfig:
    angles = 2.0 * np.pi * np.arange(8, dtype=float) / 8.0
    radius = 0.025
    return CoilGenConfig(
        contour_only_sweep_mode=True,
        skip_postprocessing=True,
        skip_sweep=False,
        conductor_cross_section_width=2.0 * radius,
        conductor_cross_section_height=2.0 * radius,
        cross_sectional_points=tuple(zip(radius * np.cos(angles), radius * np.sin(angles))),
        sweep_path_simplification="off",
        layout_terminal_strategy="waived",
    )


def test_contour_only_sweep_creates_independent_closed_manifold_bodies() -> None:
    part = _planar_contour_part()

    create_swept_coil_meshes([part], _octagonal_sweep_config())

    assert part.wire_path is None
    assert len(part.swept_bodies) == 2
    assert part.swept_body is not None
    assert part.swept_body.n_vertices == sum(body.mesh.n_vertices for body in part.swept_bodies)
    assert part.swept_body.n_faces == sum(body.mesh.n_faces for body in part.swept_bodies)
    assert part.manufacturing_diagnostics["sweep_body_count"] == 2
    assert part.manufacturing_diagnostics["manufacturing_ok"]

    for contour_index, body in enumerate(part.swept_bodies, start=1):
        diagnostics = body.diagnostics
        assert body.contour_index == contour_index
        assert diagnostics["sweep_source"] == "contour"
        assert diagnostics["sweep_closed_path"]
        assert diagnostics["sweep_end_cap_count"] == 0
        assert diagnostics["sweep_section_point_count"] == 8
        assert diagnostics["manufacturing_ok"]
        assert diagnostics["watertight"]
        assert diagnostics["degenerate_face_count"] == 0
        assert diagnostics["open_edge_count"] == 0
        assert diagnostics["nonmanifold_edge_count"] == 0
        assert diagnostics["misoriented_edge_count"] == 0
        assert diagnostics["signed_volume_m3"] > 0.0


@pytest.mark.parametrize(
    ("preliminary", "final", "expected_ok"),
    [
        (True, False, True),
        (False, True, False),
        (True, None, False),
    ],
)
def test_manufacturing_curvature_uses_final_mounted_path_when_available(
    preliminary: bool, final: bool | None, expected_ok: bool
) -> None:
    # Two submicrometre ring samples can be removed after the preliminary
    # curvature check; only the final centroid is fabricated.
    diagnostics = {
        "mesh_solid_ok": True,
        "sweep_curvature_envelope_violation": preliminary,
    }
    if final is not None:
        diagnostics["sweep_final_curvature_envelope_violation"] = final
    part = CoilPart(mesh=_planar_contour_part().mesh, manufacturing_diagnostics=diagnostics)
    _finalize_manufacturing_diagnostics(part)
    assert part.manufacturing_diagnostics["manufacturing_ok"] is expected_ok
    assert ("curvature_envelope_violation" in part.manufacturing_diagnostics["manufacturing_failure_reasons"]) is (
        not expected_ok
    )


def test_contour_only_sweep_exports_one_stl_and_step_solid_per_contour(tmp_path) -> None:
    part = _planar_contour_part()
    create_swept_coil_meshes([part], _octagonal_sweep_config())
    result = SimpleNamespace(coil_parts=[part])

    stl_files = save_swept_bodies_as_stl(result, tmp_path / "stl", prefix="contour_body")
    step_files = save_swept_bodies_as_step(result, tmp_path / "step", prefix="contour_body")
    rows = manufacturing_diagnostics_rows(result)

    assert [path.name for path in stl_files] == [
        "contour_body_part1_contour0001.stl",
        "contour_body_part1_contour0002.stl",
        "contour_body_part1_current_carrying_surface.stl",
    ]
    assert [path.name for path in step_files] == [
        "contour_body_part1_contour0001.step",
        "contour_body_part1_contour0002.step",
    ]
    assert len(rows) == 2
    assert [row["contour_index"] for row in rows] == [1, 2]
    for path in step_files:
        step_text = path.read_text(encoding="ascii")
        assert "MANIFOLD_SOLID_BREP(" in step_text
        assert "CLOSED_SHELL(" in step_text


def test_contour_only_sweep_rejects_open_contours() -> None:
    part = _planar_contour_part()
    part.contour_lines = [ContourLine(v=_closed_circle(1.0)[:-1], potential=1.0)]

    with pytest.raises(ValueError, match="is open"):
        create_swept_coil_meshes([part], _octagonal_sweep_config())


def test_insulation_clearance_failure_blocks_otherwise_valid_sweep_export(tmp_path) -> None:
    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 33)
    arc = np.column_stack((0.01 + 0.0005 * np.cos(angles), 0.0005 * np.sin(angles), np.zeros(len(angles))))
    part = _planar_contour_part()
    part.contour_lines = []
    part.wire_path = WirePath(v=np.vstack(([-0.01, -0.0005, 0.0], arc, [-0.01, 0.0005, 0.0])))
    create_swept_coil_meshes(
        [part],
        CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.0002,
            conductor_cross_section_height=0.0002,
            sweep_insulation_thickness=0.0006,
            sweep_max_segment_length=0.0005,
            sweep_max_chord_error=1e-6,
        ),
    )
    diagnostics = part.manufacturing_diagnostics

    assert diagnostics["mesh_solid_ok"]
    assert not diagnostics["sweep_capsule_self_intersection"]
    assert not diagnostics["sweep_mesh_self_intersection"]
    assert diagnostics["sweep_capsule_clearance_violation"]
    assert not diagnostics["manufacturing_ok"]
    assert "sweep_clearance_violation" in diagnostics["manufacturing_failure_reasons"]
    with pytest.raises(ValueError, match="sweep_clearance_violation"):
        save_swept_bodies_as_stl(SimpleNamespace(coil_parts=[part]), tmp_path, fail_on_violations=True)
    assert not list(tmp_path.glob("*.stl"))


@pytest.mark.parametrize("mounting_mode", ["centered", "surface_inner"])
def test_cylindrical_clearance_uses_mounted_conductor_centerline(mounting_mode: str) -> None:
    radius = 0.02
    parameters = (0.1, radius, 32, 4, 1, 0, 0, 0)
    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 33)
    developed = np.column_stack((0.00101 * np.sin(angles), 0.02 + 0.00101 * np.cos(angles)))
    developed = np.vstack(([-0.00101, -0.02], developed, [0.00101, -0.02]))
    points = np.column_stack(
        (radius * np.cos(developed[:, 0] / radius), radius * np.sin(developed[:, 0] / radius), developed[:, 1])
    )
    part = CoilPart(mesh=normalize_mesh(build_cylinder_mesh(*parameters)), wire_path=WirePath(v=points))
    create_swept_coil_meshes(
        [part],
        CoilGenConfig(
            skip_sweep=False,
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=parameters,
            surface_is_cylinder_flag=True,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            sweep_insulation_thickness=0.0003,
            sweep_max_segment_length=0.001,
            sweep_frame_method="surface_normal",
            conductor_mounting_mode=mounting_mode,
        ),
    )
    diagnostics = part.manufacturing_diagnostics
    mounted_radius = radius - (0.0005 if mounting_mode == "surface_inner" else 0.0)
    expected_gap = 2.0 * mounted_radius * np.sin(0.00101 / radius)

    assert diagnostics["sweep_capsule_min_centerline_distance_m"] == pytest.approx(expected_gap)
    assert diagnostics["sweep_capsule_clearance_violation"] is (mounting_mode == "surface_inner")


@pytest.mark.parametrize("scale", [1e-6, 1e-4, 1.0])
@pytest.mark.parametrize("gap", [0.0, 0.25])
def test_segment_distance_is_scale_invariant_for_interior_crossings(scale: float, gap: float) -> None:
    first = scale * np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    second = scale * np.asarray([[0.0, -0.5, gap], [0.0, 0.5, gap]])

    for a, b in ((first, second), (second, first), (first[::-1], second[::-1])):
        assert _segment_distance(*a, *b) == pytest.approx(scale * gap, abs=scale * 1e-12)


@pytest.mark.parametrize("scale", [1e-5, 1e-3, 1.0])
@pytest.mark.parametrize("offset, intersects", [(0.2, True), (1.2, False)])
def test_swept_triangle_collision_is_scale_invariant(scale: float, offset: float, intersects: bool) -> None:
    first = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    second = np.asarray([[offset, 0.2, -1.0], [offset, 0.2, 1.0], [offset + 0.6, 0.2, 0.0]])
    mesh = Mesh(vertices=scale * np.vstack((first, second)), faces=np.asarray([[0, 1, 2], [3, 4, 5]]))

    diagnostics = _swept_mesh_self_intersection_diagnostics(mesh)

    assert diagnostics["sweep_mesh_self_intersection"] is intersects


def test_nonplanar_triangular_support_uses_validated_projected_fillet() -> None:
    support = Mesh(
        vertices=np.asarray(
            [
                [-0.1, -0.1, 0.0],
                [0.1, -0.1, 0.0],
                [0.1, 0.1, 0.02],
                [-0.1, 0.1, -0.01],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    points, _normals = _SurfaceProjector(support).project(
        np.asarray([[-0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.04, 0.0]], dtype=float)
    )
    required_radius = 0.005
    part = create_swept_coil_bodies(
        [CoilPart(mesh=support, wire_path=WirePath(v=points))],
        CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=required_radius,
            sweep_bend_arc_angle_deg=8.0,
            sweep_max_chord_error=5.0e-5,
        ),
    )[0]
    diagnostics = part.manufacturing_diagnostics

    assert diagnostics["sweep_surface_fillet_backend"] == "triangular_mesh"
    assert diagnostics["sweep_surface_corner_eligible_count"] == 1
    assert diagnostics["sweep_surface_corner_resolved_count"] == 1
    assert diagnostics["sweep_surface_corner_validation_failure_count"] == 0
    assert diagnostics["sweep_min_resampled_bend_radius_m"] >= 0.99 * required_radius
    assert diagnostics["sweep_folded_quad_count"] == 0
    assert diagnostics["sweep_mesh_self_intersection_count"] == 0
    assert diagnostics["mesh_solid_ok"]
    assert diagnostics["manufacturing_ok"]
