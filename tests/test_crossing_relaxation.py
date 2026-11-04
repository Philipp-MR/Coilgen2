from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.crossing_relaxation import _CutMotion, _positive_gap_normal, _source_quality, _swept_score
from coilgen.geometry import normalize_mesh
from coilgen.models import Mesh, WireSegmentOwner
from coilgen.paired_cut_lifts import _crossed_cuts


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
