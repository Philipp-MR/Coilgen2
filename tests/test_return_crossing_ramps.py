"""Return approach repairs must use route identity and fixed attachments."""

import numpy as np
import pytest

from coilgen import connector_bridges, layout
from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.geometry_kernels import segment_distances
from coilgen.layout import _NormalShiftRoute, _prepared_return_obstacles, _return_crossing_ramp_spans
from coilgen.manufacturing import _prepare_sweep_centerline, conductor_cross_section_points
from coilgen.models import CoilPart, Mesh, WirePath, WireSegmentOwner


def _approach():
    points = (
        np.array(
            [[-10, 0, 0], [0, 0, 0], [10, 0, 0], [10, -8, 0], [10, -20, 0], [100, 100, 0], [7.5, 20, 0], [7.5, -20, 0]]
        )
        * 1e-3
    )
    owner = WireSegmentOwner("contour_group", group_index=0)
    source = WirePath(points, segment_owners=[owner] * 7, segment_geometry=["surface"] * 7)
    route = _NormalShiftRoute(("within_group_return", 0, 0), 2, 0, (1, 2, 3))
    routes = [[], [route], [route], [route], [], [], []]
    factors = np.array([0.0, 0, 1, 1, 0, 0, 0, 0])
    trial = points.copy()
    trial[:, 2] += 0.003 * factors
    anchors = [(1, 6, 1.75, 6.5, 0.0)]
    config = CoilGenConfig(
        normal_shift_length=0.003,
        conductor_cross_section_width=0.00196,
        conductor_cross_section_height=0.0018,
        sweep_min_bend_radius=0.0025,
    )
    return source, routes, anchors, trial, factors, config


def test_sparse_return_entry_includes_corner_and_preserves_contour_attachment():
    args = _approach()
    assert _return_crossing_ramp_spans(*args) == [(1, 3, (6,))]


def test_shared_group_ownership_without_return_membership_does_not_authorize_repair():
    source, _routes, anchors, trial, factors, config = _approach()
    assert _return_crossing_ramp_spans(source, [[] for _ in range(7)], anchors, trial, factors, config) == []


def test_return_approach_with_sufficient_physical_gap_does_not_move():
    source, routes, anchors, trial, factors, config = _approach()
    trial[6:, 2] = -0.003
    assert _return_crossing_ramp_spans(source, routes, anchors, trial, factors, config) == []


def test_return_approach_requires_a_lifted_tail_before_the_next_attachment():
    source, routes, anchors, trial, factors, config = _approach()
    factors[3] = 0
    assert _return_crossing_ramp_spans(source, routes, anchors, trial, factors, config) == []


def _obstacle_part(points, geometry, *, crease=False):
    vertices = np.array(
        [[-0.05, -0.05, 0], [-0.05, 0.05, 0], [0, -0.05, 0], [0, 0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0]]
    )
    if crease:
        vertices[:, 2] = -0.4 * np.abs(vertices[:, 0])
    faces = np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]])
    return CoilPart(mesh=normalize_mesh(Mesh(vertices, faces)), wire_path=WirePath(points, segment_geometry=geometry))


def _obstacle_config(**kwargs):
    return CoilGenConfig(
        surface_is_cylinder_flag=False,
        conductor_cross_section_width=0.0002,
        conductor_cross_section_height=0.0002,
        sweep_min_bend_radius=0.002,
        sweep_max_segment_length=0.002,
        sweep_max_chord_error=1e-5,
        sweep_max_frame_rotation_deg=5,
        sweep_path_simplification="off",
        **kwargs,
    )


@pytest.mark.parametrize("geometry_kind", ["surface", "free_3d"])
def test_obstacles_realize_faceted_surface_chord_but_preserve_free_space(geometry_kind):
    points = np.array([[-0.03, 0, -0.012], [-0.01, 0, -0.004], [0.01, 0, -0.004], [0.03, 0, -0.012]])
    part = _obstacle_part(points, [geometry_kind] * 3, crease=True)
    config = _obstacle_config()
    original = points.copy()
    mesh_vertices = part.mesh.vertices.copy()
    geometry = list(part.wire_path.segment_geometry)
    part.normal_shift_diagnostics = {"sentinel": "unchanged"}

    obstacle = _prepared_return_obstacles(part, points, geometry, [1], np.arange(len(points)), config)

    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(part.mesh.vertices, mesh_vertices)
    assert part.wire_path.segment_geometry == geometry
    assert part.normal_shift_diagnostics == {"sentinel": "unchanged"}
    assert part.fabrication_path is None
    assert part.swept_body is None
    if geometry_kind == "surface":
        assert obstacle[:, :, 2].max() > -0.001  # Millimetre-scale correction of the buried chord.
    else:
        # Densification is allowed; changing the original free-space polyline is not.
        samples = obstacle.reshape(-1, 3)
        distances = segment_distances(
            samples[:, None], np.zeros((len(samples), 1, 3)), points[None, :-1], np.diff(points, axis=0)[None]
        )
        assert np.max(np.min(distances, axis=1)) < 1e-12
        assert obstacle[:, :, 2].max() == pytest.approx(-0.004, abs=1e-14)


def _sorted_segments(segments):
    rows = segments.reshape(-1, 6)
    return rows[np.lexsort(rows.T[::-1])]


def test_closed_underpass_context_is_invariant_to_seam_position():
    unique = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0]])
    config = _obstacle_config(sweep_bend_mode="fillet")
    results = []
    for shift in (0, 2):
        shifted = np.roll(unique, -shift, axis=0)
        points = np.vstack((shifted, shifted[:1]))
        part = _obstacle_part(points, ["surface"] * 4)
        results.append(
            _prepared_return_obstacles(
                part, points, part.wire_path.segment_geometry, [(-shift) % 4], np.arange(5), config
            )
        )
    np.testing.assert_allclose(_sorted_segments(results[0]), _sorted_segments(results[1]), atol=1e-12)
    # Both sides of the seam corner must be available to the 2 mm fillet.
    assert np.min(np.linalg.norm(results[0].reshape(-1, 3), axis=1)) > 0.0005


def test_overlapping_closed_context_prepares_one_complete_loop():
    points = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0], [0, 0, 0]])
    part = _obstacle_part(points, ["surface"] * 4)
    config = _obstacle_config(sweep_bend_mode="fillet")
    prepared, _, _ = _prepare_sweep_centerline(
        part, points[:-1], part.wire_path.segment_geometry, conductor_cross_section_points(config), config, closed=True
    )
    expected = np.stack((prepared, np.roll(prepared, -1, axis=0)), axis=1)
    actual = _prepared_return_obstacles(
        part, points, part.wire_path.segment_geometry, [0, 3, 1, 2, 0], np.arange(5), config
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("closed", [False, True])
def test_prepared_obstacle_locates_omitted_source_interval_inside_current_edge(closed):
    final = [0, 0, 0] if closed else [0, 0.04, 0]
    points = np.array([[0, 0, 0], [0.04, 0, 0], [0.04, 0.04, 0], final])
    part = _obstacle_part(points, ["free_3d"] * 3)
    # Original vertex3 was removed during resampling; its interval to4 now
    # occupies part of the final current edge, whose source endpoints are2,4.
    obstacles = _prepared_return_obstacles(
        part, points, part.wire_path.segment_geometry, [3], np.array([0, 1, 2, 4]), _obstacle_config()
    )
    midpoint = 0.5 * (points[-2] + points[-1])
    gaps = segment_distances(midpoint, np.zeros(3), obstacles[:, 0], obstacles[:, 1] - obstacles[:, 0])
    assert len(obstacles) > 1
    assert np.min(gaps) < 1e-12


def test_shift_return_paths_reconstructs_owned_entry_and_preserves_lifted_tail(monkeypatch):
    # The return crosses a fixed contour 6 mm after its entry. Its interior
    # plateau remains lifted while both contour attachments stay fixed.
    points = 1e-3 * np.array(
        [
            [-10, 0, 0],
            [0, 0, 0],
            [10, 0, 0],
            [10, -8, 0],
            [10, -20, 0],
            [10, -32, 0],
            [40, 40, 0],
            [6, 20, 0],
            [6, -20, 0],
        ]
    )
    contour_owner = WireSegmentOwner("contour_loop", group_index=0, loop_index=0)
    return_owner = WireSegmentOwner("within_group_return", group_index=0)
    owners = [contour_owner, *([return_owner] * 4), *([contour_owner] * 3)]
    part = _obstacle_part(points, ["surface"] * 8)
    part.wire_path = WirePath(points, segment_owners=owners)
    protected = np.array([0, 1, 5, 6, 7, 8])
    original = points.copy()
    config = CoilGenConfig(
        normal_shift_length=0.003,
        normal_sweep_smooth_factor=3,
        min_wire_distance_m=0,
        conductor_cross_section_width=0.00196,
        conductor_cross_section_height=0.0018,
        sweep_min_bend_radius=0.0025,
        sweep_max_frame_rotation_deg=5,
        sweep_max_chord_error=5e-5,
        sweep_max_segment_length=0.005,
    )
    prepared_calls, generator_obstacles = [], []
    prepare = layout._prepared_return_obstacles
    generate = connector_bridges.generate_return_ramps

    def record_preparation(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prepared_calls.append(result.copy())
        return result

    def record_generation(source, incoming, outgoing, normal, obstacles, **kwargs):
        generator_obstacles.append(obstacles.copy())
        return generate(source, incoming, outgoing, normal, obstacles, **kwargs)

    monkeypatch.setattr(layout, "_prepared_return_obstacles", record_preparation)
    monkeypatch.setattr(connector_bridges, "generate_return_ramps", record_generation)
    layout.shift_return_paths([part], config)

    provenance = part.normal_shift_provenance
    (event,) = provenance["bridge_events"]
    assert event["accepted"] and event["kind"] == "return_approach"
    assert (event["source_start_index"], event["source_end_index"]) == (1, 3)
    assert event["owner"]["kind"] == "within_group_return"
    assert event["obstacle_reference"] == "support_prepared_underpass_centerline"
    assert len(prepared_calls) == len(generator_obstacles) == 1
    assert len(prepared_calls[0]) > 1  # The actual support preparation subdivides the underpass.
    np.testing.assert_array_equal(generator_obstacles[0], prepared_calls[0])
    assert part.normal_shift_diagnostics["normal_shift_return_ramp_count"] == 1
    assert not part.normal_shift_diagnostics["normal_shift_quality_failure"]
    assert part.normal_shift_diagnostics["normal_shift_centerline_intersection_count"] == 0

    output_positions = provenance["output_source_segment_indices"] + provenance["output_source_segment_fractions"]
    for index in protected:
        (output_index,) = np.flatnonzero(np.isclose(output_positions, index, atol=1e-12, rtol=0))
        np.testing.assert_array_equal(part.wire_path.v[output_index], original[index])
    (start,) = np.flatnonzero(np.isclose(output_positions, 1, atol=1e-12, rtol=0))
    (end,) = np.flatnonzero(np.isclose(output_positions, 3, atol=1e-12, rtol=0))
    np.testing.assert_array_equal(
        part.wire_path.v[[start, end]], provenance["points_after_lift_before_repair_m"][[1, 3]]
    )
    assert part.shift_array[start] == pytest.approx(0, abs=1e-12)
    assert part.shift_array[end] == pytest.approx(1, abs=1e-12)
    assert part.shift_array[end + 1] == pytest.approx(1, abs=1e-12)
    assert part.wire_path.v[end, 2] == pytest.approx(config.normal_shift_length)
    assert all(owner == return_owner for owner in part.wire_path.segment_owners[start:end])
    assert all(value == "free_3d" for value in part.wire_path.segment_geometry[start:end])
    assert np.max(part.wire_path.v[:, 2]) <= config.normal_shift_length + 1e-8
