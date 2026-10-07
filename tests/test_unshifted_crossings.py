"""Crossing repair must not depend on where the original connector was sampled."""

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.geometry_kernels import segment_distances
from coilgen.layout import find_min_wire_segment_distance_3d, find_wire_segment_contacts_3d, shift_return_paths
from coilgen.models import CoilPart, Mesh, WirePath, WireSegmentOwner


def _sparse_crossing(interior_x=()):
    """A 40 mm owned span crosses a remote contour on a planar support."""
    connector = np.column_stack(([-0.02, *interior_x, 0.02], np.zeros((len(interior_x) + 2, 2))))
    points = np.vstack(
        (
            [[0, -0.03, 0], [0, 0.03, 0], [-0.04, 0.03, 0], [-0.04, 0, 0]],
            connector,
            [[0.04, 0, 0], [0.04, -0.04, 0], [0, -0.04, 0], [0, -0.03, 0]],
        )
    )
    owner = WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="primary")
    owners = [WireSegmentOwner("contour_loop", loop_index=0)] * (len(points) - 1)
    geometry = ["surface"] * (len(points) - 1)
    owners[4 : 3 + len(connector)] = [owner] * (len(connector) - 1)
    geometry[4 : 3 + len(connector)] = ["free_3d"] * (len(connector) - 1)
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    return CoilPart(mesh=mesh, wire_path=WirePath(points, segment_owners=owners, segment_geometry=geometry))


def _protected_segments(path):
    indices = [i for i, owner in enumerate(path.segment_owners) if owner.kind == "contour_loop"]
    return np.stack((path.v[indices], path.v[np.asarray(indices) + 1]), axis=1)


@pytest.mark.parametrize(
    "interior_x", [(), (-0.012,), (0.0,)], ids=["no_anchor", "off_crossing_anchor", "center_anchor"]
)
@pytest.mark.parametrize(
    "target_clearance", [0.0, 0.002, 0.006], ids=["no_spacing_target", "achievable_target", "larger_target"]
)
def test_sparse_owned_crossing_gets_physical_clearance_without_moving_contours(interior_x, target_clearance):
    part = _sparse_crossing(interior_x)
    original = part.wire_path
    protected = _protected_segments(original).copy()
    config = CoilGenConfig(
        normal_shift_length=0.003,
        min_wire_distance_m=target_clearance,
        conductor_cross_section_width=0.00196,
        conductor_cross_section_height=0.0018,
        sweep_min_bend_radius=0.00272,
        normal_sweep_smooth_factor=3,
    )

    shift_return_paths([part], config)

    np.testing.assert_array_equal(_protected_segments(part.wire_path), protected)
    connector_indices = np.array(
        [i for i, owner in enumerate(part.wire_path.segment_owners) if owner.kind == "inter_group_connector"]
    )
    starts = part.wire_path.v[connector_indices]
    directions = part.wire_path.v[connector_indices + 1] - starts
    underpass = original.v[:2]
    gaps = segment_distances(
        starts,
        directions,
        np.broadcast_to(underpass[0], starts.shape),
        np.broadcast_to(underpass[1] - underpass[0], starts.shape),
    )
    # The conservative envelope diameter is 2.661 mm; a 3 mm bridge can
    # clear that conductor even when the requested 6 mm spacing cannot fit.
    physical_clearance = np.hypot(config.conductor_cross_section_width, config.conductor_cross_section_height)
    assert np.min(gaps) >= physical_clearance - 1e-8
    assert np.max(part.wire_path.v[:, 2]) <= config.normal_shift_length + 1e-12
    assert part.normal_shift_diagnostics["normal_shift_introduced_backtrack_count"] == 0
    if target_clearance > config.normal_shift_length:
        assert part.normal_shift_diagnostics["normal_shift_lift_below_clearance_target"]
        assert part.normal_shift_diagnostics["normal_shift_unresolved_contact_count"] > 0
    else:
        assert part.normal_shift_diagnostics["normal_shift_unresolved_contact_count"] == 0


def test_short_lift_without_spacing_target_reports_required_swept_body_check():
    part = _sparse_crossing((0.0,))
    shift_return_paths(
        [part],
        CoilGenConfig(
            normal_shift_length=0.001,
            min_wire_distance_m=0.0,
            conductor_cross_section_width=0.00196,
            conductor_cross_section_height=0.0018,
        ),
    )
    diagnostics = part.normal_shift_diagnostics
    assert diagnostics["normal_shift_physical_clearance_candidate_count"] > 0
    assert diagnostics["normal_shift_physical_clearance_status"] == "requires_swept_body_check"
    assert diagnostics["normal_shift_unresolved_contact_count"] == 0
    assert np.max(part.wire_path.v[:, 2]) <= 0.001 + 1e-12


@pytest.mark.parametrize("short_local_loop", [False, True])
def test_unrepairable_exact_centerline_crossing_is_failure_without_spacing_target(short_local_loop):
    part = _sparse_crossing()
    if short_local_loop:
        # The metric local-continuation exclusion may hide this crossing at a
        # conductor-sized search radius; a strict centerline check must not.
        part.wire_path = WirePath(np.array([[-0.001, 0, 0], [0.001, 0, 0], [0, -0.001, 0], [0, 0.001, 0]]))
    part.wire_path.segment_owners = [WireSegmentOwner("contour_loop", loop_index=0)] * (len(part.wire_path.v) - 1)
    original = part.wire_path.v.copy()
    shift_return_paths([part], CoilGenConfig(normal_shift_length=0.003, min_wire_distance_m=0.0))
    np.testing.assert_array_equal(part.wire_path.v, original)
    assert part.normal_shift_diagnostics["normal_shift_quality_failure"]
    assert part.normal_shift_diagnostics["normal_shift_status"] == "unresolved"


@pytest.mark.parametrize(
    "method, spatial_minimum", [("all_pairs", 3000), ("spatial", 3000), ("auto", 3000), ("auto", 0)]
)
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("gap", [0.0, 0.5e-9, 2e-9])
def test_nonadjacent_local_crossing_is_retained_consistently_by_distance_backends(method, spatial_minimum, closed, gap):
    points = np.array([[-0.001, 0, 0], [0.001, 0, 0], [0, -0.001, gap], [0, 0.001, gap]])
    if closed:
        points = np.vstack((points, points[:1]))
    config = CoilGenConfig(
        wire_distance_method=method,
        wire_distance_spatial_min_segments=spatial_minimum,
        wire_distance_chunk_size=2,
        min_wire_distance_m=0.003,
    )
    contacts = find_wire_segment_contacts_3d(points, 0.003, config=config)
    minimum = find_min_wire_segment_distance_3d(points, config=config)
    if gap <= 1e-9:
        assert len(contacts) == 1
        assert contacts[0][:2] == (0, 2)
        assert contacts[0][2] == pytest.approx(gap, abs=1e-15)
        assert minimum[:2] == (0, 2)
        assert minimum[2] == pytest.approx(gap, abs=1e-15)
    else:
        assert contacts == []  # Ordinary metric continuation is still excluded.
        assert minimum[:2] == (-1, -1)


@pytest.mark.parametrize("method", ["all_pairs", "spatial", "auto"])
def test_overlapping_aabbs_do_not_restore_nonintersecting_local_bend(method):
    points = 0.001 * np.array([[-1, -1, 0], [1, 1, 0], [-1, 1, -0.5], [1, -1, 1]])
    config = CoilGenConfig(wire_distance_method=method, wire_distance_spatial_min_segments=0, min_wire_distance_m=0.003)
    assert find_wire_segment_contacts_3d(points, 0.003, config=config) == []
    assert find_min_wire_segment_distance_3d(points, config=config)[:2] == (-1, -1)


@pytest.mark.parametrize("method", ["all_pairs", "spatial", "auto"])
def test_topological_neighbors_and_closed_seam_remain_excluded(method):
    points = 0.001 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    config = CoilGenConfig(wire_distance_method=method, wire_distance_spatial_min_segments=0, min_wire_distance_m=0.003)
    assert find_wire_segment_contacts_3d(points, 0.003, config=config) == []
    assert find_min_wire_segment_distance_3d(points, config=config)[:2] == (-1, -1)
