import copy

import numpy as np
import pytest

from coilgen.bridge_realization import bridge_realization_check, minimum_bridge_radius
from coilgen.config import CoilGenConfig
from coilgen.geometry import load_mesh_from_config
from coilgen.models import CoilPart, WirePath, WirePathGeometry, WireSegmentOwner


FREE = WirePathGeometry.FREE_3D
SURFACE = WirePathGeometry.SURFACE


def _cylinder(tmp_path):
    angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.25, 0.067, 28, 20, 1, 0, 0, 0),
        surface_is_cylinder_flag=True,
        conductor_cross_section_width=0.0014,
        conductor_cross_section_height=0.0014,
        cross_sectional_points=0.0007 * np.column_stack((np.cos(angles), np.sin(angles))),
        normal_shift_length=0.001,
        sweep_bend_mode="miter",
        sweep_path_simplification="off",
    )
    return CoilPart(mesh=load_mesh_from_config(config, tmp_path)), config


def _faceted_attachment(part):
    angle = np.pi / 28
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    tangent = np.array([-normal[1], normal[0], 0.0])
    attachment = 0.067 * np.cos(angle) * normal
    points = attachment + np.array([-0.003, -0.002, 0.0, 0.002, 0.004])[:, None] * tangent
    candidate = attachment + np.linspace(-0.002, 0.0, 21)[:, None] * tangent
    geometry = [FREE, FREE, SURFACE, SURFACE]
    owner = WireSegmentOwner(kind="contour_group", group_index=0)
    part.wire_path = WirePath(points.copy(), segment_owners=[owner] * 4, segment_geometry=geometry.copy())
    return points, candidate, geometry


def test_rejects_smooth_faceted_ramp_whose_surface_attachment_moves(tmp_path):
    part, config = _cylinder(tmp_path)
    points, candidate, geometry = _faceted_attachment(part)
    original_points = points.copy()
    original_candidate = candidate.copy()
    original_path = copy.deepcopy(part.wire_path)
    accepted, diagnostics = bridge_realization_check(part, points, geometry, 1, 2, candidate, config, False)
    assert not accepted
    assert diagnostics["bridge_realization_checked"]
    assert diagnostics["bridge_realization_reason"] == "realized_attachment_curvature"
    assert diagnostics["bridge_realization_attachment_retraction_m"] > 0.0004
    assert diagnostics["bridge_realization_minimum_radius_m"] < minimum_bridge_radius(config)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(candidate, original_candidate)
    np.testing.assert_array_equal(part.wire_path.v, original_path.v)
    assert part.wire_path.segment_owners == original_path.segment_owners
    assert part.wire_path.segment_geometry == original_path.segment_geometry
    assert geometry == [FREE, FREE, SURFACE, SURFACE]


@pytest.mark.parametrize("position", ["start", "end"])
def test_closed_seam_keeps_surface_neighbor_for_attachment_check(tmp_path, position):
    part, config = _cylinder(tmp_path)
    points, candidate, _geometry = _faceted_attachment(part)
    if position == "start":
        candidate = candidate[::-1].copy()
        points = np.vstack((candidate[0], candidate[-1], points[0], points[4], candidate[0]))
        geometry, start, end = [FREE, FREE, FREE, SURFACE], 0, 1
    else:
        points = np.vstack((candidate[-1], points[3], points[4], candidate[0], candidate[-1]))
        geometry, start, end = [SURFACE, FREE, FREE, FREE], 3, 4
    accepted, diagnostics = bridge_realization_check(part, points, geometry, start, end, candidate, config, True)
    assert not accepted
    assert diagnostics["bridge_realization_checked"]
    assert diagnostics["bridge_realization_reason"] == "realized_attachment_curvature"


@pytest.mark.parametrize("case", ["mesh_support", "flat_support", "free_neighbors"])
def test_unaffected_attachments_skip_preparation(tmp_path, monkeypatch, case):
    part, config = _cylinder(tmp_path)
    points, candidate, geometry = _faceted_attachment(part)
    if case == "mesh_support":
        config.sweep_surface_realization = "mesh"
    elif case == "flat_support":
        config.surface_is_cylinder_flag = False
    else:
        geometry = [FREE] * 4

    def unexpected(*args, **kwargs):
        raise AssertionError("An unaffected attachment should not prepare a sweep.")

    monkeypatch.setattr("coilgen.manufacturing._prepare_sweep_centerline", unexpected)
    accepted, diagnostics = bridge_realization_check(part, points, geometry, 1, 2, candidate, config, False)
    assert accepted
    assert not diagnostics["bridge_realization_checked"]


def test_analytic_attachment_still_checks_the_realized_neighbor_tangent(tmp_path):
    part, config = _cylinder(tmp_path)
    config.sweep_max_segment_length = 1e-4
    radius = 0.067
    angle = 2.0 * np.pi / 28
    attachment = np.array([radius, 0.0, 0.0])
    following = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
    tangent = following - attachment
    tangent /= np.linalg.norm(tangent)
    points = np.vstack((attachment - 0.003 * tangent, attachment - 0.002 * tangent, attachment, following))
    candidate = attachment + np.linspace(-0.002, 0.0, 21)[:, None] * tangent

    # The free bridge and original surface chord are exactly collinear, and
    # their shared endpoint needs no radial correction. Analytic realization
    # curves the surface chord, leaving an abrupt free/surface attachment.
    accepted, diagnostics = bridge_realization_check(
        part, points, [FREE, FREE, SURFACE], 1, 2, candidate, config, False
    )

    assert not accepted
    assert diagnostics["bridge_realization_checked"]
    assert diagnostics["bridge_realization_attachment_retraction_m"] < 1e-12
    assert diagnostics["bridge_realization_reason"] == "realized_attachment_curvature"
    assert 0.0006 < diagnostics["bridge_realization_minimum_radius_m"] < 0.0008
    assert diagnostics["bridge_realization_minimum_radius_m"] < minimum_bridge_radius(config)


def test_already_analytic_attachment_with_smooth_neighbor_is_checked_and_accepted(tmp_path):
    part, config = _cylinder(tmp_path)
    points, candidate, geometry = _faceted_attachment(part)
    normal = points[2] / np.linalg.norm(points[2])
    correction = 0.067 * normal - points[2]
    points += correction
    candidate += correction

    accepted, diagnostics = bridge_realization_check(part, points, geometry, 1, 2, candidate, config, False)

    assert accepted
    assert diagnostics["bridge_realization_checked"]
    assert diagnostics["bridge_realization_attachment_retraction_m"] < 1e-12
    assert diagnostics["bridge_realization_reason"] == "accepted"
    assert diagnostics["bridge_realization_minimum_radius_m"] > 0.05


def test_attachment_movement_is_rejected_without_mutating_source(tmp_path):
    part, config = _cylinder(tmp_path)
    points, candidate, geometry = _faceted_attachment(part)
    candidate[-1, 2] += 1e-5
    accepted, diagnostics = bridge_realization_check(part, points, geometry, 1, 2, candidate, config, False)
    assert not accepted
    assert diagnostics["bridge_realization_reason"] == "attachment_moved"


def test_retracted_attachment_with_adequate_radius_is_accepted(tmp_path):
    part, config = _cylinder(tmp_path)
    _points, original, geometry = _faceted_attachment(part)
    tangent = original[-1] - original[0]
    tangent /= np.linalg.norm(tangent)
    attachment = original[-1]
    points = attachment + np.array([-0.006, -0.004, 0.0, 0.002, 0.004])[:, None] * tangent
    candidate = attachment + np.linspace(-0.004, 0.0, 5)[:, None] * tangent
    accepted, diagnostics = bridge_realization_check(part, points, geometry, 1, 2, candidate, config, False)
    assert accepted
    assert diagnostics["bridge_realization_checked"]
    assert diagnostics["bridge_realization_minimum_radius_m"] > 0.0025
    assert diagnostics["bridge_realization_attachment_retraction_m"] > 0.0004


def test_closed_implicit_source_is_supported(tmp_path):
    part, config = _cylinder(tmp_path)
    points, candidate, _geometry = _faceted_attachment(part)
    points = np.vstack((candidate[-1], points[3], points[4], candidate[0]))
    candidate = candidate[::-1].copy()
    # The replacement starts at the seam; its preceding SURFACE interval wraps.
    points = np.vstack((candidate[0], candidate[-1], points[1], points[2]))
    accepted, diagnostics = bridge_realization_check(
        part, points, [FREE, FREE, FREE, SURFACE], 0, 1, candidate, config, True
    )
    assert not accepted
    assert diagnostics["bridge_realization_checked"]
