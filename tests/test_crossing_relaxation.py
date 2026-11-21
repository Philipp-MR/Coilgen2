from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.crossing_relaxation import (
    _CutMotion,
    _constraint_axes,
    _positive_gap_normal,
    _source_quality,
    _swept_score,
)
from coilgen.geometry import normalize_mesh
from coilgen.models import Mesh, WireSegmentOwner
from coilgen.paired_cut_lifts import _crossed_cuts


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("arrangement", ["parallel", "crossed", "coincident"])
def test_station_separation_keeps_analytic_gap_and_axis_ties_across_units(scale, arrangement):
    length, width, height, gap = 0.01, 0.0004, 0.0002, 0.003
    first = np.array([[0.0, 0, 0], [length, 0, 0]])
    copper = np.array(
        [[x, y, z] for x in (0, length) for y in (-width / 2, width / 2) for z in (-height / 2, height / 2)]
    )
    tangent = np.array([1.0, 0, 0])
    if arrangement == "crossed":
        second = np.array([[length / 2, -length / 2, gap], [length / 2, length / 2, gap]])
        other = copper[:, [1, 0, 2]] + [length / 2, -length / 2, gap]
        other_tangent = np.array([0.0, 1, 0])
        expected_gap, direction = gap - height, [0, 0, -1]
    else:
        offset = [0, gap if arrangement == "parallel" else 0, 0]
        second, other = first + offset, copper + offset
        other_tangent = tangent
        expected_gap = gap - width if arrangement == "parallel" else 0
        direction = [0, -1, 0] if arrangement == "parallel" else [0, 0, 0]
    physical = np.concatenate((first, second)) * scale
    window = (np.array([0]), np.array([1.0]))
    normal = np.array([0.0, 0, 1])
    cache = {
        0: (0, 0.0, window, copper * scale, first.mean(axis=0) * scale, tangent, normal),
        2: (1, 0.0, window, other * scale, second.mean(axis=0) * scale, other_tangent, normal),
    }
    pairs = [(0, 2, None), (2, 0, 0.0), (0, 2, 0.0001)]
    result = list(_constraint_axes(physical, cache, pairs))
    assert [row[:3] for row in result] == pairs
    axes, distances = result[0][3:]
    assert np.isfinite(axes).all() and np.argmax(distances) == 0
    np.testing.assert_array_equal(axes[0], direction)
    assert distances[0] == pytest.approx(expected_gap * scale, abs=10 * np.finfo(float).eps * scale)


def test_clearance_score_requires_a_completed_solid():
    diag = {
        "exact_collision_completed": True,
        "exact_intersection_count": 0,
        "copper_gap_violating_facet_pairs": 0,
        "mesh_solid_ok": False,
    }
    p = SimpleNamespace(swept_bodies=[SimpleNamespace(diagnostics=diag)])
    assert _swept_score(p) is None
    diag["mesh_solid_ok"] = True
    assert _swept_score(p) == (0, 0)


def test_positive_gap_uses_actual_facets_not_overlapping_bounding_boxes():
    # Two nonparallel triangles with the nearest point on a face and the
    # opposing vertex. Their axis-aligned boxes overlap in x and y.
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    y = np.array([[0.5, 0.5, 0.00024], [2.0, 1.0, 0.02], [1.0, 2.0, 0.02]])
    v = np.vstack((x, y))
    mesh = Mesh(v, np.array([[0, 1, 2], [2, 1, 0], [3, 4, 5], [5, 4, 3]]))
    direction = _positive_gap_normal(SimpleNamespace(mesh=mesh), 0, 1, 1)
    np.testing.assert_allclose(direction, [0, 0, -1], atol=1e-12)


@pytest.mark.parametrize("sign", [-1, 1])
def test_cut_motion_caps_and_unmodified_open_endpoints(sign):
    cfg = CoilGenConfig(normal_shift_length=sign * 0.004, interconnection_cut_width=0.006)
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]])
    normal = np.tile([0.0, 0.0, 1.0], (4, 1))
    owner = WireSegmentOwner("inter_group_connector")
    prov = dict(
        points_after_m=pts,
        points_before_m=pts,
        normals=normal,
        output_source_segment_indices=np.array([0, 1, 2, 2]),
        output_source_segment_fractions=np.array([0.0, 0.0, 0.0, 1.0]),
        output_segment_owners=[vars(owner)] * 3,
        output_segment_geometry=["surface"] * 3,
        segment_geometry_before=["surface"] * 3,
        lift_distance_m=np.zeros(4),
        lateral_displacement_m=np.zeros_like(pts),
    )
    v = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    m = normalize_mesh(Mesh(v, np.array([[0, 1, 2], [0, 2, 3]])))
    cm = _CutMotion(SimpleNamespace(normal_shift_provenance=prov, mesh=m), cfg)
    assert np.all(cm.authorized[[0, -1]] == 0.0)
    coords = cm.radial_limit(pts[[1, 2]] + np.array([[0, 0.5, 0.4 * sign], [0, 0.5, -0.5 * sign]]), np.array([1, 2]))
    np.testing.assert_allclose(coords[:, 1], 0.003)
    np.testing.assert_allclose(coords[:, 2], [sign * 0.004, 0.0])


def test_partial_improvement_can_retain_an_existing_crossing_but_cannot_certify_it():
    wire = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    part = SimpleNamespace(
        normal_shift_diagnostics={
            "normal_shift_physical_clearance_m": 0.001,
            "normal_shift_centerline_intersection_count": 1,
        }
    )
    motion = SimpleNamespace(
        detail={
            "points_before_m": wire,
            "output_source_segment_indices": np.array([0, 1, 2, 2]),
            "output_source_segment_fractions": np.array([0.0, 0.0, 0.0, 1.0]),
        }
    )
    config = CoilGenConfig()
    assert _source_quality(part, motion, wire, config, (1, 1))[2:] == (1, 0)
    assert _source_quality(part, motion, wire, config, (0, 0)) is None
    part.normal_shift_diagnostics["normal_shift_centerline_intersection_count"] = 0
    assert _source_quality(part, motion, wire, config, (1, 1)) is None


def test_paired_levels_require_an_interior_world_space_cut_crossing():
    p = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    owners = [
        WireSegmentOwner("inter_group_connector", connector_index=2, connector_role="primary"),
        WireSegmentOwner("contour_loop"),
        WireSegmentOwner("contour_loop"),
        WireSegmentOwner("inter_group_connector", connector_index=2, connector_role="closing"),
    ]
    assert list(_crossed_cuts(p, owners, 0.1)) == [2]
    p[-2:, 2] = 0.01
    assert list(_crossed_cuts(p, owners, 0.1)) == []
    p[-2:] = [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert list(_crossed_cuts(p, owners, 0.1)) == []
    p[-2:] = p[:2][::-1]
    assert list(_crossed_cuts(p, owners, 0.1)) == [2]
