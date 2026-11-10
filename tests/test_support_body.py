from __future__ import annotations

import builtins
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("manifold3d")

from coilgen import CoilGenConfig, CoilPart, Mesh, WirePath, create_swept_coil_meshes  # noqa: E402
from coilgen.application.cad import _body_sources  # noqa: E402
from coilgen.application.scene_spec import scene_layers  # noqa: E402
from coilgen.application.session import SETTINGS  # noqa: E402
from coilgen.config_validation import validate_config  # noqa: E402
from coilgen.geometry import build_cylinder_mesh  # noqa: E402
from coilgen.manufacturing import mesh_manufacturing_diagnostics, save_swept_bodies_as_stl  # noqa: E402
from coilgen.models import FabricationPath, SweptBody  # noqa: E402
from coilgen.scientific_arrays import scientific_arrays  # noqa: E402
from coilgen.support_body import (  # noqa: E402
    build_grooved_support,
    create_grooved_support_bodies,
    groove_cutter_profile,
    thickened_support_mesh,
)

WIDTH, HEIGHT, THICKNESS = 0.002, 0.001, 0.003


def _grid(length=0.05, count=40):
    xs = np.linspace(-length / 2, length / 2, count + 1)
    x, y = np.meshgrid(xs, xs)
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    faces = []
    for i in range(count):
        for j in range(count):
            a = i * (count + 1) + j
            faces += [(a, a + 1, a + count + 2), (a, a + count + 2, a + count + 1)]
    return Mesh(vertices=vertices, faces=np.asarray(faces))


def _mount(mode):
    return {"surface_inner": -HEIGHT / 2, "centered": 0.0, "surface_outer": HEIGHT / 2}[mode]


def _plane_loop(mode, radius=0.015, count=720):
    """Centroid path of a rectangular conductor mounted on the z = 0 plane (+z normals)."""
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    support = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros(count)))
    tangents = np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(count)))
    normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    profile = np.array(
        [[-WIDTH / 2, -HEIGHT / 2], [WIDTH / 2, -HEIGHT / 2], [WIDTH / 2, HEIGHT / 2], [-WIDTH / 2, HEIGHT / 2]]
    )
    return FabricationPath(
        points_m=support + _mount(mode) * normals,
        tangents=tangents,
        normals=normals,
        transverse=np.cross(normals, tangents),
        cross_section_m=profile,
        source_points_m=support,
        closed=True,
    )


def _config(mode="inside", mounting="surface_inner", **kwargs):
    settings = {
        "support_body_mode": mode,
        "support_body_thickness": THICKNESS,
        "support_body_groove_clearance": 0.0,
        "conductor_mounting_mode": mounting,
        "conductor_cross_section_width": WIDTH,
        "conductor_cross_section_height": HEIGHT,
    }
    return replace(CoilGenConfig(), **{**settings, **kwargs})


@pytest.mark.parametrize(
    ("side", "mounting", "depth"),
    [
        ("inside", "surface_inner", HEIGHT),
        ("inside", "centered", HEIGHT / 2),
        ("outside", "surface_outer", HEIGHT),
        ("outside", "centered", HEIGHT / 2),
    ],
)
def test_plane_groove_volume_matches_slab_minus_annular_groove(side, mounting, depth):
    radius = 0.015
    body = build_grooved_support(_grid(), [_plane_loop(mounting, radius)], _config(side, mounting))
    d = body.diagnostics

    assert d["support_body_ok"], d["support_body_failure_reasons"]
    assert d["support_body_watertight"]
    assert d["support_body_component_count"] == 1
    assert mesh_manufacturing_diagnostics(body.mesh)["watertight"]
    slab = 0.05 * 0.05 * THICKNESS
    assert d["support_slab_manifold_volume_m3"] == pytest.approx(slab, rel=1e-9)
    groove = 2 * np.pi * radius * WIDTH * depth
    assert d["support_groove_removed_volume_m3"] == pytest.approx(groove, rel=1e-3)
    assert d["support_body_volume_m3"] == pytest.approx(slab - groove, rel=1e-4)
    assert d["support_groove_depth_m"] == pytest.approx(depth)
    assert d["support_groove_floor_thickness_m"] == pytest.approx(THICKNESS - depth)


def test_slab_side_is_opposite_for_inside_and_outside():
    path = [_plane_loop("centered")]
    inside = build_grooved_support(_grid(), path, _config("inside", "centered")).mesh.vertices[:, 2]
    outside = build_grooved_support(_grid(), path, _config("outside", "centered")).mesh.vertices[:, 2]
    assert inside.min() == pytest.approx(-THICKNESS) and inside.max() == pytest.approx(0.0, abs=1e-12)
    assert outside.min() == pytest.approx(0.0, abs=1e-12) and outside.max() == pytest.approx(THICKNESS)


def test_only_conductors_on_this_support_choose_the_slab_side():
    # A second plate 20 mm above with flipped frame normals must not flip this plate's slab.
    own = _plane_loop("centered")
    other = _plane_loop("centered")
    other = replace(
        other,
        points_m=other.points_m + [0.0, 0.0, 0.02],
        normals=-other.normals,
        transverse=-other.transverse,
        segment_owners=(),
        segment_geometry=(),
    )
    many = [other, other, other, own]
    z = build_grooved_support(_grid(), many, _config("inside", "centered")).mesh.vertices[:, 2]
    assert z.min() == pytest.approx(-THICKNESS)


def test_conductor_on_the_far_side_is_reported_not_silently_ignored():
    body = build_grooved_support(_grid(), [_plane_loop("surface_outer")], _config("inside", "surface_outer"))
    d = body.diagnostics
    assert not d["support_body_ok"]
    assert "conductor_does_not_reach_support_body" in d["support_body_failure_reasons"]
    assert d["support_groove_removed_volume_m3"] == pytest.approx(0.0, abs=1e-15)
    assert d["support_body_watertight"]


def test_grooves_deeper_than_the_wall_are_rejected():
    config = _config(support_body_groove_extra_depth=THICKNESS)
    d = build_grooved_support(_grid(), [_plane_loop("surface_inner")], config).diagnostics
    assert not d["support_body_ok"]
    assert "grooves_cut_through_support_body" in d["support_body_failure_reasons"]


def test_clearance_and_insulation_grow_the_groove_and_open_it_straight():
    config = _config(support_body_groove_clearance=0.0002, sweep_insulation_thickness=0.0001)
    profile, info = groove_cutter_profile(_plane_loop("surface_inner").cross_section_m, config)
    growth = 0.0003
    assert info["support_groove_depth_m"] == pytest.approx(HEIGHT + growth)
    assert info["support_groove_width_at_face_m"] == pytest.approx(WIDTH + 2 * growth, rel=1e-6)
    # The channel walls continue past the face (w = 0) so the wire can be laid in.
    assert profile[:, 1].max() > 0.0
    assert info["support_groove_floor_thickness_m"] == pytest.approx(THICKNESS - HEIGHT - growth)


def test_groove_cutter_is_one_connected_solid():
    from coilgen.support_body import _cutter_manifold

    # Touching-only hull pieces would stay separate solids and leave slivers.
    closed = _cutter_manifold(_plane_loop("surface_inner", count=180), _config())
    assert len(closed.decompose()) == 1 and closed.genus() == 1
    path = _plane_loop("surface_inner", count=180)
    open_path = replace(path, closed=False, segment_owners=(), segment_geometry=())
    assert len(_cutter_manifold(open_path, _config()).decompose()) == 1


def test_cylinder_ring_groove_volume():
    radius, length = 0.04, 0.03
    support = build_cylinder_mesh(length, radius, 256, 12, 1, 0, 0, 0)
    count = 1024
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    radial = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(count)))
    tangents = np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(count)))
    profile = np.array(
        [[-WIDTH / 2, -HEIGHT / 2], [WIDTH / 2, -HEIGHT / 2], [WIDTH / 2, HEIGHT / 2], [-WIDTH / 2, HEIGHT / 2]]
    )
    path = FabricationPath(
        points_m=(radius - HEIGHT / 2) * radial,
        tangents=tangents,
        normals=radial,
        transverse=np.cross(radial, tangents),
        cross_section_m=profile,
        source_points_m=radius * radial,
        closed=True,
    )
    body = build_grooved_support(support, [path], _config("inside", "surface_inner"))
    d = body.diagnostics

    assert d["support_body_ok"], d["support_body_failure_reasons"]
    assert d["support_body_watertight"]
    assert d["support_slab_volume_ratio"] == pytest.approx(1.0, abs=0.05)
    assert d["support_body_genus"] == 1  # a tube stays a tube
    groove = np.pi * (radius**2 - (radius - HEIGHT) ** 2) * WIDTH
    assert d["support_groove_removed_volume_m3"] == pytest.approx(groove, rel=0.02)
    # The former is the slab (inside the support) with the groove opening outward.
    r = np.linalg.norm(body.mesh.vertices[:, :2], axis=1)
    assert r.max() <= radius + 1e-9
    assert r.min() == pytest.approx(radius - THICKNESS, rel=0.02)


def test_open_and_closed_supports_thicken_into_closed_solids():
    open_slab, info = thickened_support_mesh(_grid(0.02, 4), THICKNESS, -1.0)
    assert mesh_manufacturing_diagnostics(open_slab)["watertight"]
    assert info["support_slab_volume_m3"] == pytest.approx(0.02 * 0.02 * THICKNESS)
    # Closed octahedron -> hollow shell (two nested closed surfaces).
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float) * 0.05
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    shell, info = thickened_support_mesh(Mesh(vertices=vertices, faces=faces), 0.002, -1.0)
    assert mesh_manufacturing_diagnostics(shell)["watertight"]
    assert info["support_surface_open_edge_count"] == 0
    assert 0 < info["support_slab_volume_m3"]


def _swept_plane_part():
    angle = np.linspace(0, 2 * np.pi, 97)
    points = np.column_stack((0.015 * np.cos(angle), 0.015 * np.sin(angle), np.zeros_like(angle)))
    points[-1] = points[0]
    return CoilPart(mesh=_grid(), wire_path=WirePath(v=points), contour_step=1.0)


def test_pipeline_stage_cuts_the_real_sweep_and_exports_the_former(tmp_path):
    config = replace(_config("inside", "surface_inner"), sweep_max_chord_error=0.0)
    part = _swept_plane_part()
    create_swept_coil_meshes([part], config)
    assert part.swept_bodies and part.swept_bodies[0].fabrication_path is not None

    create_grooved_support_bodies([part], config)
    d = part.support_body.diagnostics
    assert d["support_body_ok"], d["support_body_failure_reasons"]
    assert d["support_groove_depth_m"] == pytest.approx(HEIGHT)
    # The groove holds the conductor: it removes at least the conductor volume inside the slab.
    conductor = part.swept_bodies[0].diagnostics["absolute_volume_m3"]
    assert d["support_groove_removed_volume_m3"] == pytest.approx(conductor, rel=0.02)

    files = save_swept_bodies_as_stl(SimpleNamespace(coil_parts=[part]), tmp_path, prefix="coil")
    assert tmp_path / "coil_part1_grooved_support.stl" in files

    arrays = scientific_arrays(
        [part],
        SimpleNamespace(coords=np.zeros((1, 3)), b=np.zeros((1, 3)), suppressed=np.zeros(1, bool), weights=np.ones(1)),
    )
    assert "grooved_support_0_vertices_m" in arrays
    layers, _ = scene_layers(arrays, "swept")
    assert [layer.name for layer in layers][0] == "grooved_support_0"
    assert "support" not in [layer.name for layer in layers]


def test_off_mode_and_missing_backend(monkeypatch):
    part = _swept_plane_part()
    create_grooved_support_bodies([part], CoilGenConfig())
    assert part.support_body is None

    real_import = builtins.__import__

    def without_manifold(name, *args, **kwargs):
        if name == "manifold3d":
            raise ImportError("no manifold3d")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_manifold)
    part.swept_bodies = [SweptBody(mesh=_grid(), fabrication_path=_plane_loop("surface_inner"))]
    create_grooved_support_bodies([part], _config())
    d = part.support_body.diagnostics
    assert not d["support_body_ok"]
    assert "manifold3d" in d["support_body_failure_reasons"]


def test_config_validation_and_front_end_fields():
    assert CoilGenConfig().support_body_mode == "off"
    errors = validate_config(replace(CoilGenConfig(), support_body_mode="sideways")).errors
    assert any(error.field == "support_body_mode" for error in errors)
    errors = validate_config(replace(CoilGenConfig(), support_body_mode="inside", support_body_thickness=0.0)).errors
    assert any(error.field == "support_body_thickness" for error in errors)

    fields = {field.key: field for field in SETTINGS}
    thickness = fields["support_body_thickness_mm"]
    assert thickness.node == "print"
    assert thickness.get(CoilGenConfig(), {}) == pytest.approx(3.0)
    assert thickness.set(2.5, CoilGenConfig(), {}) == {"support_body_thickness": pytest.approx(0.0025)}
    assert fields["support_body_mode"].choices == ("off", "inside", "outside")


def test_export_catalog_identifies_reviewed_support_bodies():
    metadata = {"fabrication": {"paths": [], "support_bodies": [{"body_id": "part001_grooved_support"}]}}
    assert _body_sources(metadata)["part001_grooved_support"][0] == "print_former_geometry"
