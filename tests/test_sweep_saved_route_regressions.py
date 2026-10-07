"""Small geometry regressions from the saved Halbach sweep review.

The embedded connector is already normal-shifted. Its preservation/rejection
does not establish that the unavailable original routing stage was corrected.
Full saved-file replay is optional in outputs/sweep_implementation_20260914.
"""

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import (
    _finalize_manufacturing_diagnostics,
    _sweep_wire_path_with_diagnostics,
    mesh_manufacturing_diagnostics,
)
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry


PROFILE = np.array([[-0.00098, -0.0009], [0.00098, -0.0009], [0.0008, 0.0009], [-0.0008, 0.0009]])


def _config(**overrides):
    options = dict(
        coil_mesh_file="none",
        surface_is_cylinder_flag=False,
        cross_sectional_points=PROFILE,
        sweep_bend_mode="fillet",
        sweep_min_bend_radius=0.0025,
        sweep_max_segment_length=0.003,
        sweep_max_chord_error=5e-5,
        sweep_max_frame_rotation_deg=5.0,
        sweep_self_intersection_check="capsule_fast",
    )
    options.update(overrides)
    return CoilGenConfig(**options)


def _plane():
    return normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )


def _run(part, config=None):
    mesh, diagnostics = _sweep_wire_path_with_diagnostics(part, PROFILE, _config() if config is None else config)
    part.manufacturing_diagnostics = {**mesh_manufacturing_diagnostics(mesh), **diagnostics}
    _finalize_manufacturing_diagnostics(part)
    return mesh, part.manufacturing_diagnostics


def test_straight_free_conductor_is_not_rejected_by_failure_detection():
    points = np.array([[-0.03, 0.0, 0.005], [0.03, 0.0, 0.005]])
    part = CoilPart(mesh=_plane(), wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D]))
    mesh, diagnostics = _run(part)
    assert diagnostics["manufacturing_ok"]
    assert diagnostics["watertight"]
    assert diagnostics["sweep_folded_quad_count"] == 0
    expected_volume = 0.06 * 0.0018 * (0.00196 + 0.0016) / 2
    assert np.isclose(diagnostics["absolute_volume_m3"], expected_volume, rtol=1e-10)
    assert np.isfinite(mesh.vertices).all()


def test_watertight_remote_crossing_is_not_manufacturing_valid():
    points = np.array([[-0.03, -0.02, 0.005], [0.03, 0.02, 0.005], [0.03, -0.02, 0.005], [-0.03, 0.02, 0.005]])
    part = CoilPart(mesh=_plane(), wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * 3))
    _mesh, diagnostics = _run(part)
    assert diagnostics["watertight"]
    assert not diagnostics["manufacturing_ok"]
    assert diagnostics["sweep_mesh_self_intersection"]
    assert diagnostics["sweep_capsule_clearance_violation_count"] > 0


def test_physical_support_crease_cannot_be_certified_by_projection_and_sampling():
    # Two genuine planar faces meet at 45 degrees. Refining their triangulation
    # cannot produce a regular finite-radius transverse curve in exact contact.
    mesh = normalize_mesh(
        Mesh(
            np.array(
                [
                    [-0.05, -0.05, 0.0],
                    [-0.05, 0.05, 0.0],
                    [0.0, -0.05, 0.0],
                    [0.0, 0.05, 0.0],
                    [0.05, -0.05, 0.05],
                    [0.05, 0.05, 0.05],
                ]
            ),
            np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]]),
        )
    )
    points = np.array([[-0.03, 0.0, 0.0], [0.0, 0.0, 0.0], [0.03, 0.0, 0.03]])
    part = CoilPart(mesh=mesh, wire_path=WirePath(points))
    _body, diagnostics = _run(part)
    assert not diagnostics["manufacturing_ok"]
    assert diagnostics["sweep_min_bend_radius_violation"]
    np.testing.assert_array_equal(part.wire_path.v, points)
    assert part.wire_path.segment_geometry == [WirePathGeometry.SURFACE] * 2


def test_saved_post_shift_connector_backtrack_remains_explicit_and_source_unchanged():
    # Saved GX point_index 2154..2164 (1-based), run 2026-09-13_16-35-20.
    # These small coordinates are the CSV data, not a reconstructed pre-shift route.
    points = np.array(
        [
            [0.07663972524807271, 0.004923324734449552, 0.02802286527905723],
            [0.07881354262562884, 0.004922510188483757, 0.022774808909447213],
            [0.0808415352716521, 0.004922460704836113, 0.01787880076241937],
            [0.08278250526845929, 0.004922195892017736, 0.01319288691331272],
            [0.08473665618415215, 0.004922253521795158, 0.008475143114840818],
            [0.08696007118877326, 0.004922150251399867, 0.002510612838268154],
            [0.08673112268779884, 0.005318509367714385, 0.0036600814535010266],
            [0.0867289897474835, 0.005948394095405975, 0.003665231434267061],
            [0.08672554455468112, 0.006578239104294828, 0.0036735494768220978],
            [0.08672561056148118, 0.007190357728622205, 0.003673390122010192],
            [0.08672406146945237, 0.007802463165375022, 0.0036771299680353163],
        ]
    )
    # Compact support with the same eight physical facet planes. It intentionally
    # does not reproduce the full saved remeshing's interpolated vertex normals.
    angle = np.arange(8) * np.pi / 4
    ring = np.column_stack((0.085 * np.cos(angle), np.zeros(8), 0.085 * np.sin(angle)))
    vertices = np.vstack((ring + [0.0, -0.05, 0.0], ring + [0.0, 0.05, 0.0]))
    faces = np.array(
        [triangle for i in range(8) for triangle in ([i, (i + 1) % 8, (i + 1) % 8 + 8], [i, (i + 1) % 8 + 8, i + 8])]
    )
    support = normalize_mesh(Mesh(vertices, faces))
    geometry = [WirePathGeometry.OFFSET_SURFACE] * (len(points) - 1)
    part = CoilPart(mesh=support, wire_path=WirePath(points.copy(), segment_geometry=geometry))
    _body, diagnostics = _run(part)
    np.testing.assert_array_equal(part.wire_path.v, points)
    assert part.wire_path.segment_geometry == geometry
    assert diagnostics["sweep_max_turn_angle_deg"] > 159.0
    assert not diagnostics["manufacturing_ok"]
    assert diagnostics["sweep_min_bend_radius_violation"]
