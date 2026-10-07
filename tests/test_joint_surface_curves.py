"""Physical corridor regressions for compact support-facet return proposals."""

import numpy as np
import pytest

from coilgen.connector_bridge_validation import _validate_support
from coilgen.geometry import normalize_mesh
from coilgen.geometry_kernels import segment_distances
from coilgen.joint_surface_curves import generate_planar_return_curves
from coilgen.manufacturing import _SurfaceProjector
from coilgen.models import Mesh
from coilgen.models import WirePathGeometry


def _rotation():
    axis = np.array([1.0, -0.4, 0.2])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(0.7) * skew + (1 - np.cos(0.7)) * (skew @ skew)


def _case(scale=1.0, rotation=np.eye(3)):
    # One cylinder facet unfolded into a plane. Its seam is the y=0 edge;
    # protected turn ends start at y=5mm, leaving a 5mm-clear seam corridor.
    vertices = np.array([[0, 0, -0.15], [0, 0.015, -0.15], [0, 0.015, 0.15], [0, 0, 0.15]])
    mesh = normalize_mesh(Mesh(vertices=vertices * scale @ rotation.T, faces=np.array([[0, 1, 2], [0, 2, 3]])))
    source = np.array([[0, 0.005, -0.03], [0, 0.005, 0.03]]) * scale @ rotation.T
    obstacles = np.array([[[0, 0.005, z], [0, 0.014, z]] for z in (-0.02, -0.01, 0, 0.01, 0.02)])
    obstacles = obstacles * scale @ rotation.T
    arguments = dict(
        tangent_before=np.array([0, -1, 0]) @ rotation.T,
        tangent_after=np.array([0, 1, 0]) @ rotation.T,
        normal=np.array([1, 0, 0]) @ rotation.T,
        obstacle_segments=obstacles,
        maximum_height_m=0.001 * scale,
        minimum_radius_m=0.0014 * scale,
        maximum_segment_length_m=0.0005 * scale,
        chord_tolerance_m=0.0000175 * scale,
        maximum_tangent_angle_deg=2.0,
        maximum_candidates=64,
        b0_direction=np.array([0, 0, 1]) @ rotation.T,
        clearance_score_cap_m=0.005 * scale,
    )
    return _SurfaceProjector(mesh), source, arguments


def _supported(projector, source, arguments):
    generation = generate_planar_return_curves(projector, source, **arguments)
    supported = []
    for candidate in generation.candidates:
        segments = np.stack((candidate.points[:-1], candidate.points[1:]), axis=1)
        reasons = _validate_support(
            projector,
            source[0],
            segments,
            arguments["normal"],
            arguments["maximum_height_m"],
            1e-8,
            {},
        )
        if not reasons:
            supported.append(candidate)
    return generation, supported


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_seam_return_clears_turns_without_lift_and_preserves_ports(scale):
    projector, source, arguments = _case(scale)
    generation, supported = _supported(projector, source, arguments)
    assert generation.candidates and supported, generation.rejected_reasons
    seam = max(supported, key=lambda curve: curve.local_clearance_m)
    np.testing.assert_array_equal(seam.points[[0, -1]], source)
    assert seam.profile_family == "planar_biarc_return"
    assert seam.height_m == 0 and not np.any(seam.lift_distances_m)
    assert seam.minimum_radius_m >= arguments["minimum_radius_m"]
    assert seam.local_clearance_m == pytest.approx(0.005 * scale, abs=1e-11)
    assert np.max(np.abs(seam.points[:, 0])) < 1e-12
    assert np.min(seam.points[:, 1]) >= -1e-12
    lengths = np.linalg.norm(np.diff(seam.points, axis=0), axis=1)
    assert np.max(lengths) <= arguments["maximum_segment_length_m"] * (1 + 1e-9)
    for tangent, interval in (
        (arguments["tangent_before"], seam.points[1] - seam.points[0]),
        (arguments["tangent_after"], seam.points[-1] - seam.points[-2]),
    ):
        assert tangent @ interval / np.linalg.norm(interval) >= np.cos(np.radians(1.01))
    # Every segment is measured, including the terminal bends.
    first = seam.points[:-1, None]
    directions = np.diff(seam.points, axis=0)[:, None]
    other = arguments["obstacle_segments"]
    distances = segment_distances(first, directions, other[None, :, 0], np.diff(other, axis=1)[None, :, 0])
    assert float(distances.min()) >= 0.005 * scale - 1e-10


def test_return_family_is_covariant_under_rotation():
    projector, source, arguments = _case()
    reference, _ = _supported(projector, source, arguments)
    rotation = _rotation()
    turned_projector, turned_source, turned_arguments = _case(rotation=rotation)
    turned, _ = _supported(turned_projector, turned_source, turned_arguments)
    assert len(reference.candidates) == len(turned.candidates)
    for before, after in zip(reference.candidates, turned.candidates):
        np.testing.assert_allclose(after.points, before.points @ rotation.T, atol=2e-12)
        assert after.minimum_radius_m == pytest.approx(before.minimum_radius_m)


def test_opposing_attachment_faces_are_not_assumed_coplanar():
    projector, source, arguments = _case()

    class DifferentFaces:
        vertices = projector.vertices

        def project_with_face_normals(self, points):
            return points, points, np.array([[1, 0, 0], [0, 1, 0]])

    result = generate_planar_return_curves(DifferentFaces(), source, **arguments)
    assert not result.candidates
    assert result.reason == "no_common_planar_attachment_facet"


def test_insufficient_corridor_room_is_not_repaired_by_relaxing_clearance():
    projector, source, arguments = _case()
    # Move all protected endpoints 1mm toward the seam. Supported proposals
    # may still be generated, but none meets the unchanged 5mm requirement.
    arguments["obstacle_segments"][:, :, 1] -= 0.001
    generation, supported = _supported(projector, source, arguments)
    assert generation.candidates and supported
    assert max(candidate.local_clearance_m for candidate in supported) < arguments["clearance_score_cap_m"]


def test_declared_distance_budget_failure_is_preserved():
    from coilgen.joint_transition_curves import _DistanceBudgetExceeded

    projector, source, arguments = _case()

    def exhausted(_count):
        raise _DistanceBudgetExceeded("fixture_pair_budget_exceeded")

    result = generate_planar_return_curves(projector, source, **arguments, distance_budget=exhausted)
    assert not result.candidates
    assert result.reason == "fixture_pair_budget_exceeded"


def test_raised_middle_keeps_terminal_arcs_nominal_and_has_a_flat_plateau():
    projector, source, arguments = _case()
    generation, supported = _supported(projector, source, arguments)
    raised = [curve for curve in supported if curve.profile_family == "planar_biarc_lifted_return"]
    assert raised, generation.rejected_reasons
    candidate = max(raised, key=lambda curve: curve.height_m)
    assert candidate.height_m == arguments["maximum_height_m"]
    assert candidate.minimum_radius_m >= arguments["minimum_radius_m"]
    np.testing.assert_array_equal(candidate.points[[0, -1]], source)
    surface = np.array([value == WirePathGeometry.SURFACE for value in candidate.segment_geometry])
    assert len(surface) == len(candidate.points) - 1 and surface[0] and surface[-1]
    assert np.any(~surface)
    assert not np.any(candidate.lift_distances_m[:-1][surface])
    plateau = candidate.lift_distances_m == candidate.height_m
    assert np.count_nonzero(plateau) > 10
    assert np.ptp(candidate.points[plateau, 0]) < 1e-14
    intervals = np.diff(candidate.points, axis=0)
    lengths = np.linalg.norm(intervals, axis=1)
    assert lengths.max() <= arguments["maximum_segment_length_m"] * (1 + 1e-9)
    # A numerical independent circumcircle estimate verifies the complete
    # sampled lifted path, including transitions back to the terminal arcs.
    before, after = intervals[:-1], intervals[1:]
    denominator = (
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) * np.linalg.norm(before + after, axis=1)
    )
    curvature = 2 * np.linalg.norm(np.cross(before, after), axis=1) / denominator
    assert 1 / curvature.max() >= arguments["minimum_radius_m"] * 0.999


def test_only_admitted_raised_profiles_consume_the_shape_budget():
    projector, source, arguments = _case()
    charges = []
    result = generate_planar_return_curves(projector, source, **arguments, shape_trial_budget=charges.append)
    assert result.candidates and 0 < sum(charges) <= 32 * 6
    arguments["maximum_height_m"] = 0
    charges.clear()
    result = generate_planar_return_curves(projector, source, **arguments, shape_trial_budget=charges.append)
    assert result.candidates and not charges
    assert all(curve.profile_family == "planar_biarc_return" for curve in result.candidates)
