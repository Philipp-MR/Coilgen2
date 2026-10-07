from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import coilgen.layout as layout
from coilgen.config import CoilGenConfig
from coilgen.layout import (
    GroupConnectionCandidate,
    GroupConnectionSearchPath,
    InterGroupCutAlignmentState,
    _axially_aligned_group_candidates,
    _group_connection_candidate_is_valid,
    _sort_group_connection_candidates,
    calculate_group_centers,
    generate_group_cutshapes,
    interconnect_among_groups,
    interconnect_within_groups,
)
from coilgen.models import CoilPart, ContourLine, LoopGroup, Mesh
from coilgen.surface_distance import SurfaceGeodesicSolver


_SAMPLES = 48
_RADIUS = 0.1


def _rotation(tilted: bool) -> np.ndarray:
    if not tilted:
        return np.eye(3)
    axis = np.asarray([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = 0.63
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _ring(z: float, seam: int, reverse: bool, *, subdivide: bool = False) -> np.ndarray:
    angles = np.arange(_SAMPLES) * (2.0 * np.pi / _SAMPLES)
    points = np.column_stack((_RADIUS * np.cos(angles), _RADIUS * np.sin(angles), np.full(_SAMPLES, z)))
    if subdivide:
        # Add collinear samples, changing neither the polygon nor its enclosed area.
        points = np.insert(points, 6, points[5] + np.arange(1, 8)[:, None] / 8.0 * (points[6] - points[5]), axis=0)
    points = np.roll(points, -seam, axis=0)
    if reverse:
        points = points[::-1]
    return np.vstack((points, points[:1]))


def _fixture(*, seam: int = 0, reverse: bool = False, subdivide: bool = False, tilted: bool = False):
    rotation = _rotation(tilted)
    origin = np.asarray([0.31, -0.22, 0.4]) if tilted else np.zeros(3)

    def transform(points):
        return np.asarray(points) @ rotation.T + origin

    vertices = np.vstack((_ring(-0.15, 0, False)[:-1], _ring(0.15, 0, False)[:-1]))
    faces = []
    for first in range(_SAMPLES):
        following = (first + 1) % _SAMPLES
        faces.extend(((first, following, first + _SAMPLES), (following, following + _SAMPLES, first + _SAMPLES)))
    mesh = Mesh(
        vertices=transform(vertices),
        faces=np.asarray(faces),
        boundaries=[np.arange(_SAMPLES), np.arange(_SAMPLES, 2 * _SAMPLES)],
    )
    groups = []
    contours = []
    for group_index, levels in enumerate(((0.04, 0.065, 0.09), (-0.04, -0.065, -0.09))):
        loops = [
            ContourLine(
                transform(
                    _ring(z, seam + 7 * group_index + 3 * index, reverse != bool(group_index), subdivide=subdivide)
                ),
                float(index + 1) * (-1 if group_index else 1),
            )
            for index, z in enumerate(levels)
        ]
        first = len(contours)
        groups.append(
            LoopGroup(loops=loops, loop_indices=list(range(first, first + len(loops))), outer_loop_indices=(first,))
        )
        contours.extend(loops)
    part = CoilPart(mesh=mesh, contour_lines=contours, groups=groups, group_levels=[[0, 1]], level_positions=[[]])
    calculate_group_centers([part])
    axis = rotation[:, 2]
    radial = np.asarray([1.0, 0.0, 0.0]) - axis * axis[0]
    radial /= np.linalg.norm(radial)
    return part, origin, axis, radial


def _cuts(part, index, b0, *, align=True):
    return generate_group_cutshapes(
        part.groups[index], part.group_centers_v[index], part.mesh, 0.002, b0, align_axial_rings=align
    )


@pytest.mark.parametrize(
    "seam,reverse,subdivide", [(0, False, False), (7, False, False), (19, True, False), (11, True, True)]
)
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("b0_sign", [-1.0, 1.0])
def test_axis_encircling_ring_cuts_share_one_plane_and_side(seam, reverse, subdivide, tilted, b0_sign):
    part, origin, axis, radial = _fixture(seam=seam, reverse=reverse, subdivide=subdivide, tilted=tilted)
    transverse = np.cross(axis, radial)
    baseline, _, _, _ = _fixture(tilted=tilted)
    for index in range(2):
        actual = _cuts(part, index, b0_sign * axis)
        expected = _cuts(baseline, index, axis)
        for cut, reference in zip(actual, expected):
            np.testing.assert_allclose(cut.high_cut.v, reference.high_cut.v, atol=2e-13, rtol=0)
            np.testing.assert_allclose(cut.low_cut.v, reference.low_cut.v, atol=2e-13, rtol=0)
            assert abs(float((cut.high_cut.v - origin) @ transverse)) < 2e-13
            assert abs(float((cut.low_cut.v - origin) @ transverse)) < 2e-13
            assert float((cut.high_cut.v - origin) @ radial) > 0.09
            assert float((cut.low_cut.v - origin) @ radial) < -0.09


def test_reordering_groups_does_not_change_the_shared_cut_frame():
    part, _, axis, _ = _fixture(seam=13, reverse=True)
    before = [_cuts(part, index, axis) for index in range(2)]
    part.groups.reverse()
    calculate_group_centers([part])
    for index in range(2):
        after = _cuts(part, index, axis)
        for cut, original in zip(after, before[1 - index]):
            np.testing.assert_allclose(cut.high_cut.v, original.high_cut.v, atol=1e-13, rtol=0)
            np.testing.assert_allclose(cut.low_cut.v, original.low_cut.v, atol=1e-13, rtol=0)


@pytest.mark.parametrize("case", ["local_loop", "open_contour", "transverse_b0", "missing_axis"])
def test_nonaxial_or_nonencircling_contours_keep_existing_cut_behavior(case):
    part, _, axis, _ = _fixture(seam=7)
    if case == "local_loop":
        for group in part.groups:
            for loop in group.loops:
                loop.v[:, :2] = 0.12 * loop.v[:, :2] + [0.07, 0.02]
    elif case == "open_contour":
        for group in part.groups:
            for loop in group.loops:
                loop.v = loop.v[:-4]
    elif case == "transverse_b0":
        axis = np.asarray([0.0, 1.0, 0.0])
    else:
        part.mesh.boundaries = []
    calculate_group_centers([part])
    for index in range(2):
        if case == "open_contour":
            for align in (False, True):
                with pytest.raises(ValueError, match="Could not generate two cut points"):
                    _cuts(part, index, axis, align=align)
            continue
        for actual, legacy in zip(_cuts(part, index, axis), _cuts(part, index, axis, align=False)):
            np.testing.assert_array_equal(actual.high_cut.v, legacy.high_cut.v)
            np.testing.assert_array_equal(actual.low_cut.v, legacy.low_cut.v)


@pytest.mark.parametrize("alignment,cylinder", [("off", True), ("disabled", True), ("auto", False)])
def test_alignment_opt_out_preserves_the_original_within_group_route(alignment, cylinder):
    part, _, axis, _ = _fixture(seam=7)
    config = CoilGenConfig(
        interconnection_cut_width=0.002,
        b_0_direction=tuple(axis),
        inter_group_cut_alignment=alignment,
        surface_is_cylinder_flag=cylinder,
    )
    legacy = [_cuts(part, index, axis, align=False) for index in range(2)]
    interconnect_within_groups([part], config)
    for group, cut_infos in zip(part.groups, legacy):
        selected = [cut.low_cut if group.within_group_cut_side == "low" else cut.high_cut for cut in cut_infos]
        # Sphere opening's midpoint lies just inside the polygon; the cut center
        # is recovered by its outward radial direction rather than its radius.
        for opening, cut in zip(group.opening_cuts_v, selected):
            center = opening.mean(axis=0)
            np.testing.assert_allclose(
                center[:2] / np.linalg.norm(center[:2]), cut.v[:2] / np.linalg.norm(cut.v[:2]), atol=1e-12
            )


def _signed_axial_area(points, axis):
    centered = np.asarray(points) - np.asarray(points).mean(axis=0)
    return float(np.sum(np.cross(centered[:-1], centered[1:]), axis=0) @ axis)


def test_connector_alignment_does_not_reward_mirrored_azimuth_errors():
    origin = np.asarray([0.3, -0.2, 0.4])
    state = InterGroupCutAlignmentState(
        origin_v=origin,
        axis_v=np.asarray([0.0, 0.0, 1.0]),
        preferred_radial_v=np.asarray([1.0, 0.0, 0.0]),
        distance_rel_tol=0.05,
        candidate_factor=4,
        return_corridor_aligned=True,
        axial_ring_alignment=True,
    )

    def candidate(first_angle, second_angle):
        angles = np.deg2rad([first_angle, second_angle])
        endpoints = np.column_stack((0.1 * np.cos(angles), 0.1 * np.sin(angles), [0.04, -0.04])) + origin
        distance = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        return GroupConnectionCandidate(
            local_a=0,
            local_b=1,
            point_a_v=endpoints[0],
            point_b_v=endpoints[1],
            direct_distance=distance,
            geodesic_distance=np.inf,
            selection_distance=distance,
            used_geodesic_selection=False,
        )

    axial = candidate(15.0, 15.0)
    mirrored_straddle = candidate(12.0, -12.0)
    # The straddling connector's midpoint lies on the preferred azimuth, but
    # its endpoints produce unnecessary transverse current and broken symmetry.
    assert _sort_group_connection_candidates([mirrored_straddle, axial], state)[0] is axial


@pytest.mark.parametrize("method", ["direct_3d", "surface_geodesic"])
def test_within_group_routes_are_coherent_without_reversing_contour_current(method):
    part, origin, axis, radial = _fixture(seam=13)
    originals = deepcopy(part.groups)
    config = CoilGenConfig(
        interconnection_cut_width=0.002, b_0_direction=tuple(axis), inter_group_connection_path_method=method
    )
    interconnect_within_groups([part], config)
    selected_sides = {group.within_group_cut_side for group in part.groups}
    assert len(selected_sides) == 1
    return_radials = []
    for original, group, connected in zip(originals, part.groups, part.connected_groups):
        for before, after in zip(original.loops, group.opened_loops):
            assert _signed_axial_area(before.v, axis) * _signed_axial_area(after.v, axis) > 0
        for before, retained in zip(original.loops, group.loops):
            np.testing.assert_array_equal(retained.v, before.v)
        relative = connected.return_path_v - origin
        perpendicular = relative - np.outer(relative @ axis, axis)
        return_radials.append(perpendicular.mean(axis=0))
        assert np.max(np.linalg.norm(perpendicular - perpendicular.mean(axis=0), axis=1)) < 0.0021
        assert abs(float(perpendicular.mean(axis=0) @ np.cross(axis, radial))) < 0.0011
    assert float(return_radials[0] @ return_radials[1]) > 0

    interconnect_among_groups([part], config)
    assert part.wire_path is not None
    assert len(part.opening_cuts_among_groups) == 1
    for connection in part.opening_cuts_among_groups:
        for points in (connection.connection_path_v, connection.closing_connection_path_v):
            displacement = points[-1] - points[0]
            transverse = displacement - axis * float(displacement @ axis)
            # Paired conductor separation may span one opening, but not drift
            # around the cylinder between upper and lower groups.
            assert np.linalg.norm(transverse) < 0.003


@pytest.mark.parametrize("forced", [("low",), ("high", "low")])
def test_explicit_cut_side_selection_is_preserved(forced):
    part, origin, axis, radial = _fixture(seam=13)
    config = CoilGenConfig(interconnection_cut_width=0.002, b_0_direction=tuple(axis), force_cut_selection=forced)
    interconnect_within_groups([part], config)
    expected = list(forced) if len(forced) > 1 else list(forced) * len(part.groups)
    assert [group.within_group_cut_side for group in part.groups] == expected
    for side, connected in zip(expected, part.connected_groups):
        signed_radius = float((connected.return_path_v.mean(axis=0) - origin) @ radial)
        assert signed_radius < -0.09 if side == "low" else signed_radius > 0.09


def _staggered_edge_case(*, reverse=False):
    part, origin, axis, radial = _fixture()
    loops, searches, fixed = [], [], []
    for z, fraction, label in ((0.04, 0.3, 0), (-0.04, 0.7, 1)):
        points = _ring(z, 0, False)
        inserted = points[4] + fraction * (points[5] - points[4])
        search_points = np.asarray([points[4], inserted, points[5]])
        points = np.insert(points, 5, inserted, axis=0)
        if reverse:
            points = points[::-1]
        loops.append(ContourLine(points, float(label)))
        searches.append(GroupConnectionSearchPath(search_points, np.full(3, label)))
        fixed.append(inserted)
    part.contour_lines = loops
    distance = float(np.linalg.norm(fixed[1] - fixed[0]))
    candidate = GroupConnectionCandidate(
        local_a=0,
        local_b=1,
        point_a_v=fixed[0],
        point_b_v=fixed[1],
        direct_distance=distance,
        geodesic_distance=np.inf,
        selection_distance=distance,
        used_geodesic_selection=False,
        loop_a_index=0,
        loop_b_index=1,
    )
    state = InterGroupCutAlignmentState(origin, axis, radial, 0.05, 4, True, True)
    return part, searches, candidate, state


@pytest.mark.parametrize("reverse", [False, True])
def test_corresponding_azimuth_candidates_interpolate_staggered_contour_samples(reverse):
    part, searches, candidate, state = _staggered_edge_case(reverse=reverse)
    before = [loop.v.copy() for loop in part.contour_lines]
    aligned = _axially_aligned_group_candidates(part, searches, [candidate], state, None)
    assert len(aligned) == 2
    interpolated_endpoints = 0
    for result in aligned:
        assert (result.local_a, result.local_b, result.loop_a_index, result.loop_b_index) == (0, 1, 0, 1)
        np.testing.assert_allclose(result.point_a_v[:2], result.point_b_v[:2], atol=1e-14, rtol=0)
        assert result.direct_distance == pytest.approx(0.08, abs=1e-14)
        for point, source in ((result.point_a_v, before[0]), (result.point_b_v, before[1])):
            interpolated_endpoints += int(np.min(np.linalg.norm(source - point, axis=1)) > 1e-10)
    assert interpolated_endpoints == 2
    for loop, original in zip(part.contour_lines, before):
        np.testing.assert_array_equal(loop.v, original)


@pytest.mark.parametrize("reverse", [False, True])
def test_corresponding_azimuth_candidates_do_not_interpolate_across_removed_return_gap(reverse):
    part, searches, candidate, state = _staggered_edge_case(reverse=reverse)
    # The lower contour retains the candidate and its following endpoint, but
    # loses the endpoint of the interval containing the desired matching cut.
    searches[1] = GroupConnectionSearchPath(searches[1].v[1:], np.full(2, 1))
    aligned = _axially_aligned_group_candidates(part, searches, [candidate], state, None)
    assert len(aligned) == 1
    np.testing.assert_array_equal(aligned[0].point_b_v, candidate.point_b_v)
    assert np.linalg.norm(aligned[0].point_a_v - candidate.point_a_v) > 1e-4

    # A search path containing isolated retained samples has no admissible
    # source intervals, even if connecting its samples would cross the seam.
    searches = [GroupConnectionSearchPath(search.v[1:2], np.asarray([label])) for label, search in enumerate(searches)]
    assert _axially_aligned_group_candidates(part, searches, [candidate], state, None) == []


def test_intergroup_bridges_keep_axial_symmetry_with_staggered_collinear_sampling():
    part, _, axis, _ = _fixture(seam=13)
    for group_index, group in enumerate(part.groups):
        for loop in group.loops:
            starts, ends = loop.v[:-1], loop.v[1:]
            fraction = 0.3 if group_index == 0 else 0.7
            points = np.empty((2 * len(starts) + 1, 3))
            points[:-1:2] = starts
            points[1::2] = starts + fraction * (ends - starts)
            points[-1] = points[0]
            loop.v = points
    calculate_group_centers([part])
    before = [loop.v.copy() for loop in part.contour_lines]
    config = CoilGenConfig(interconnection_cut_width=0.002, b_0_direction=tuple(axis))
    interconnect_within_groups([part], config)
    interconnect_among_groups([part], config)
    assert len(part.opening_cuts_among_groups) == 1
    cut = part.opening_cuts_among_groups[0]
    assert {cut.loop1_index, cut.loop2_index} == {0, 3}
    for path in (cut.connection_path_v, cut.closing_connection_path_v):
        delta = path[-1] - path[0]
        np.testing.assert_allclose(delta - axis * float(delta @ axis), 0.0, atol=1e-12, rtol=0)
    for loop, original in zip(part.contour_lines, before):
        np.testing.assert_array_equal(loop.v, original)
    owners = part.wire_path.segment_owners
    assert len(owners) == len(part.wire_path.v) - 1
    assert {owner.loop_index for owner in owners if owner.kind == "contour_loop"} == set(range(6))
    for first, second, owner in zip(part.wire_path.v[:-1], part.wire_path.v[1:], owners):
        if owner.kind != "contour_loop":
            continue
        original = before[owner.loop_index]
        starts, directions = original[:-1], np.diff(original, axis=0)
        midpoint = 0.5 * (first + second)
        fractions = np.clip(np.sum((midpoint - starts) * directions, axis=1) / np.sum(directions**2, axis=1), 0, 1)
        distances = np.linalg.norm(starts + fractions[:, None] * directions - midpoint, axis=1)
        nearest = int(np.argmin(distances))
        assert distances[nearest] < 1e-10
        assert float((second - first) @ directions[nearest]) > 0

    # Matching azimuths must not bypass the existing clearance gate: an
    # unrelated contour placed between the groups intersects either bridge.
    candidate = GroupConnectionCandidate(
        local_a=0,
        local_b=1,
        point_a_v=np.asarray(cut.selected_point1_v).reshape(3),
        point_b_v=np.asarray(cut.selected_point2_v).reshape(3),
        direct_distance=cut.direct_distance,
        geodesic_distance=np.inf,
        selection_distance=cut.direct_distance,
        used_geodesic_selection=False,
        loop_a_index=cut.loop1_index,
        loop_b_index=cut.loop2_index,
    )
    groups = [cut.group1_index, cut.group2_index]
    assert _group_connection_candidate_is_valid(part, part.connected_groups, groups, candidate, config, None)
    part.contour_lines.append(ContourLine(_ring(0.0, 0, False), 0.0))
    config.min_inter_group_connector_clearance = 0.001
    assert not _group_connection_candidate_is_valid(part, part.connected_groups, groups, candidate, config, None)


@pytest.mark.parametrize("strict", [False, True])
def test_rejected_aligned_candidate_never_replaces_the_original_diagnostic_fallback(monkeypatch, strict):
    part, origin, axis, radial = _fixture(seam=13)
    config = CoilGenConfig(interconnection_cut_width=0.002, b_0_direction=tuple(axis))
    interconnect_within_groups([part], config)
    # This closed conductor blocks every bridge through the axial gap, so
    # neither the diagonal vertex proposal nor its aligned versions are valid.
    part.contour_lines.append(ContourLine(_ring(0.0, 0, False), 0.0))
    config.min_inter_group_connector_clearance = 0.001
    config.inter_group_connection_path_method = "surface_geodesic" if strict else "direct_3d"
    first = _ring(0.04, 0, False)[4]
    second = _ring(-0.04, 0, False)[5]
    distance = float(np.linalg.norm(second - first))
    original = GroupConnectionCandidate(
        local_a=0,
        local_b=1,
        point_a_v=first,
        point_b_v=second,
        direct_distance=distance,
        geodesic_distance=np.inf,
        selection_distance=distance,
        used_geodesic_selection=False,
        loop_a_index=0,
        loop_b_index=3,
    )
    monkeypatch.setattr(
        layout,
        "_group_connection_candidates",
        lambda local_a, local_b, *_args: [original] if (local_a, local_b) == (0, 1) else [],
    )
    validated = []

    def observe_validation(*args, **kwargs):
        result = _group_connection_candidate_is_valid(*args, **kwargs)
        validated.append((args[3], result))
        return result

    monkeypatch.setattr(layout, "_group_connection_candidate_is_valid", observe_validation)
    result = layout._select_group_connection(
        part,
        part.connected_groups,
        [0, 1],
        np.vstack([group.return_path_v for group in part.connected_groups]),
        config,
        SurfaceGeodesicSolver(part.mesh) if strict else None,
        alignment_state=InterGroupCutAlignmentState(origin, axis, radial, 0.05, 4, True, True),
    )
    assert any(candidate is original for candidate, _ in validated)
    assert any(candidate is not original for candidate, _ in validated)
    assert all(not valid for _, valid in validated)
    if strict:
        assert result is None
        assert not original.used_clearance_fallback
    else:
        assert result is original
        assert result.used_clearance_fallback
