import numpy as np
import pytest

from coilgen.models import WirePathGeometry
from coilgen.routing_jumps import raised_span_metrics, fabricated_jump_metrics
from coilgen.routing_objective import RoutingSearchCost, route_is_better, route_order_key
from coilgen.joint_state_search import search_group_states


class Plane:
    def project(self, points):
        result = np.array(points, copy=True)
        result[:, 2] = 0
        return result, np.tile([0, 0, 1], (len(result), 1))


def count(points, **kwargs):
    return raised_span_metrics(points, Plane(), maximum_sample_spacing_m=0.1, **kwargs)["jump_count"]


def test_continuous_transfer_counts_once_independent_of_crossed_turns():
    points = np.array([[0, 0, 0], [1, 0, 1], [10, 0, 1], [11, 0, 0]])
    assert count(points) == 1
    # Landing between optional humps creates distinct transfers.
    humps = np.array([[i, 0, i % 2] for i in range(7)])
    assert count(humps) == 3


def test_cyclic_span_is_invariant_to_storage_seam_and_direction():
    ring = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 1], [3, 0, 0], [3, 1, 0], [0, 1, 0]])
    for start in range(len(ring)):
        shifted = np.roll(ring, start, axis=0)
        assert count(shifted, closed=True) == 1
        assert count(shifted[::-1], closed=True) == 1


def test_mixed_surface_attachment_measurements_are_invariant_to_cycle_seam():
    ring = np.array([[0, 0, 0.01], [1, 0, 0.01], [2, 1, 0.2], [0, 1, 0.2]])
    # Nominal surface endpoints may differ from the projection used for free
    # intervals (for example, analytic versus faceted support realization).
    geometry = np.array([WirePathGeometry.SURFACE, *[WirePathGeometry.FREE_3D] * 3], dtype=object)
    measured = []
    for start in range(len(ring)):
        measured.append(
            raised_span_metrics(
                np.roll(ring, start, axis=0),
                Plane(),
                closed=True,
                segment_geometry=np.roll(geometry, start),
                maximum_sample_spacing_m=0.1,
            )
        )
    assert [value["jump_count"] for value in measured] == [1] * len(ring)
    np.testing.assert_allclose([value["raised_length_m"] for value in measured], measured[0]["raised_length_m"])


def test_collinear_resampling_and_duplicate_points_do_not_change_count():
    points = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 0], [3, 0, 1], [4, 0, 0]], dtype=float)
    dense = np.vstack([np.linspace(a, b, 19) for a, b in zip(points[:-1], points[1:])])
    assert count(points) == count(dense) == 2


def test_free_space_label_does_not_make_a_surface_connector_a_jump():
    assert count([[0, 0, 0], [1, 0, 0]], segment_geometry=[WirePathGeometry.FREE_3D]) == 0
    assert count([[0, 0, 0], [1, 0, -1], [2, 0, 0]]) == 1


def test_nominal_curved_surface_chords_are_not_detached_transfers():
    class Cylinder:
        def project(self, points):
            projected = np.array(points, copy=True)
            projected[:, :2] /= np.linalg.norm(projected[:, :2], axis=1)[:, None]
            return projected, projected

    points = np.array([[1.0, 0, 0], [0, 1, 0]])
    result = raised_span_metrics(points, Cylinder(), segment_geometry=[WirePathGeometry.SURFACE])
    assert result["jump_count"] == 0


def test_a_fully_detached_closed_cycle_is_one_span():
    assert count([[0, 0, 1], [1, 0, 1], [1, 1, 1]], closed=True) == 1


def test_sampling_exhaustion_never_returns_a_zero_jump_estimate():
    with pytest.raises(ValueError, match="budget"):
        raised_span_metrics([[0, 0, 0], [1, 0, 1]], Plane(), maximum_samples=3)
    with pytest.raises(ValueError, match="budget"):
        raised_span_metrics([[0, 0, 0], [1, 0, 1]], Plane(), maximum_sample_spacing_m=1e-320)


@pytest.mark.parametrize("height", [0.0, 0.5e-6, 2e-6, 0.006])
@pytest.mark.parametrize("rotated", [False, True])
def test_complete_curve_jump_queries_use_exact_faceted_distances(monkeypatch, height, rotated):
    from types import SimpleNamespace

    from coilgen.geometry import build_cylinder_mesh
    from coilgen.joint_cylinder_curves import _cylinder_chart
    from coilgen.manufacturing import _SurfaceProjector

    mesh = build_cylinder_mesh(0.25, 0.066, 8, 3, 1, 0, 0, 0)
    if rotated:
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_rotvec([0.51, -0.22, 0.83]).as_matrix()
        mesh.vertices = mesh.vertices @ rotation.T + [0.016, -0.019, 0.031]
    projector = _SurfaceProjector(mesh)
    chart = _cylinder_chart(projector)
    points = chart.map(np.array([[0.2, 0, 0], [0.5, 0, 0], [0.8, 0, 0]]) * chart.lengths[0])
    points[1] += height * np.cross(chart.directions[0], chart.axis)
    reference = raised_span_metrics(
        points, SimpleNamespace(project=projector.project), maximum_sample_spacing_m=0.0001
    )

    def unexpected_projection(_points):
        pytest.fail("Verified facets should avoid the general projection query.")

    monkeypatch.setattr(projector, "project", unexpected_projection)
    result = raised_span_metrics(points, projector, maximum_sample_spacing_m=0.0001)
    assert result["jump_count"] == reference["jump_count"] == int(height > 1e-6)
    assert result["sample_count"] == reference["sample_count"]
    assert result["raised_length_m"] == pytest.approx(reference["raised_length_m"], abs=1e-12)


def test_fabricated_mounting_offset_is_not_a_jump():
    from types import SimpleNamespace
    from coilgen.config import CoilGenConfig

    config = CoilGenConfig(conductor_mounting_mode="surface_outer")
    points = np.array([[0.0, 0, 0.001], [1, 0, 0.001], [1, 1, 0.001], [0, 1, 0.001]])
    trace = SimpleNamespace(
        points_m=points,
        closed=True,
        normals=np.tile([0.0, 0, 1], (4, 1)),
        transverse=np.tile([0.0, 1, 0], (4, 1)),
        cross_section_m=np.array([[-0.001, -0.001], [0.001, -0.001], [0.001, 0.001], [-0.001, 0.001]]),
        segment_geometry=[WirePathGeometry.FREE_3D] * 4,
    )
    part = SimpleNamespace(fabrication_path=trace)
    assert fabricated_jump_metrics(part, config, projector=Plane())["jump_count"] == 0


def test_selection_uses_jumps_then_actual_field_then_length():
    def metrics(j, error, length):
        return {"jump_count": j, "longitudinal_rms_T": error, "length_m": length}

    assert route_is_better(metrics(1, 1e-4, 100), metrics(2, 0, 1))
    assert route_is_better(metrics(1, 1e-8, 100), metrics(1, 1e-7, 1))
    assert route_is_better(metrics(1, 1e-8, 1), metrics(1, 1e-8, 100))
    assert not route_is_better(metrics(2, 0, 1), metrics(1, 1e-4, 100))


def test_graph_prefers_longer_lower_jump_cycle_including_closure():
    costs = [[RoutingSearchCost(0, 0)], [RoutingSearchCost(0, 0), RoutingSearchCost(0, 0)]]

    def edge(a, b):
        state = a[1] if a[0] == 1 else b[1]
        return RoutingSearchCost(int(state == 0), 1 if state == 0 else 100)

    result = search_group_states(costs, edge, beam_width=1, candidate_count=2, maximum_expanded_states=100)
    assert result.circuits[0][1] == ((0, 0), (1, 1))
    assert result.circuits[0][0].jumps == 0
    assert float(result.circuits[0][0]) == 200


def test_field_ties_have_a_transitive_order_independent_of_enumeration():
    from itertools import permutations

    candidates = [
        {"jump_count": 1, "longitudinal_rms_T": error, "length_m": length}
        for error, length in ((0.0, 3), (0.75e-15, 2), (1.5e-15, 1))
    ]
    for order in permutations(candidates):
        winner = order[0]
        for candidate in order[1:]:
            if route_is_better(candidate, winner):
                winner = candidate
        assert winner is min(candidates, key=route_order_key)
