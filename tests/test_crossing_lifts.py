"""Regression coverage for short connector approaches and multi-crossing layer assignment."""

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.crossing_lifts import _crossing_plateaus, _straighten_reversed_returns, apply_crossing_lifts
from coilgen.geometry import normalize_mesh
from coilgen.layout import find_wire_segment_contacts_3d
from coilgen.models import CoilPart, ConnectedGroup, InterGroupCut, Mesh, WirePath, WireSegmentOwner


def _short_connector(seam):
    # Two cuts are 4 mm apart. A 6 mm bend radius requires lift shoulders
    # on the adjoining winding. The vertical underpass crosses their midpoint.
    points = 0.001 * np.array(
        [
            [0, -30, 0],
            [0, 30, 0],
            [-30, 30, 0],
            [-30, 0, 0],
            [-2, 0, 0],
            [2, 0, 0],
            [30, 0, 0],
            [30, -30, 0],
            [0, -30, 0],
        ]
    )
    contour = WireSegmentOwner("contour_loop", group_index=0, loop_index=0)
    transfer = WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="primary")
    owners = [contour] * 8
    owners[4] = transfer
    connector = points[4:6].copy()
    indices = (np.arange(8) + seam) % 8
    v = points[np.r_[indices, indices[0]]]
    path = WirePath(v, segment_owners=[owners[i] for i in indices])
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]), np.array([[0, 1, 2], [0, 2, 3]])
        )
    )
    # Construction metadata may have a midpoint that was removed by stitching.
    record = InterGroupCut(
        np.empty((0, 3)), np.empty((0, 3)), np.vstack([connector[0], connector.mean(axis=0), connector[-1]])
    )
    group = ConnectedGroup(points[5:7], np.empty((0, 3)), points[5:7], np.empty((0, 3)))
    return CoilPart(mesh=mesh, wire_path=path, opening_cuts_among_groups=[record], connected_groups=[group])


@pytest.mark.parametrize("seam", [0, 4, 5])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_short_connector_lift_preserves_footprint_ownership_and_clearance_across_seam(seam, sign):
    p = _short_connector(seam)
    source = p.wire_path
    cfg = CoilGenConfig(
        normal_shift_length=sign * 0.004,
        sweep_min_bend_radius=0.006,
        conductor_cross_section_width=0.002,
        conductor_cross_section_height=0.002,
    )
    assert apply_crossing_lifts(p, cfg)
    meta = p.normal_shift_provenance
    positions = meta["output_source_segment_indices"] + meta["output_source_segment_fractions"]
    assert np.all(np.diff(positions) > 0)
    assert np.max(np.abs(meta["lift_distance_m"])) <= 0.004
    assert np.max(sign * p.wire_path.v[:, 2]) > 0.002
    np.testing.assert_array_equal(
        source.v[:, :2], p.wire_path.v[np.searchsorted(positions, np.arange(len(source.v))), :2]
    )
    np.testing.assert_array_equal(p.wire_path.v[0], p.wire_path.v[-1])
    assert not find_wire_segment_contacts_3d(p.wire_path.v, 0.0028, config=cfg)
    cut = p.opening_cuts_among_groups[0].connection_path_v
    assert len(cut) > 2 and np.min(sign * cut[:, 2]) > 0.0028
    # Group's historical contour points now follow the same lifted approach.
    group = p.connected_groups[0].spiral_in_v
    assert sign * group[0, 2] > 0.0028 and abs(group[-1, 2]) < 1e-12


@pytest.mark.parametrize(
    "settings",
    [
        {"skip_normal_shift": True},
        {"normal_shift_length": 0.0},
        {"min_wire_distance_m": 0.006, "normal_shift_length": 0.001},
    ],
)
def test_disabled_and_unachievable_hard_spacing_leave_input_untouched(settings):
    p = _short_connector(0)
    path = p.wire_path
    cut = p.opening_cuts_among_groups[0].connection_path_v
    assert not apply_crossing_lifts(p, CoilGenConfig(**settings))
    assert p.wire_path is path and p.opening_cuts_among_groups[0].connection_path_v is cut
    assert p.normal_shift_provenance == {}


def test_shared_overpass_choice_satisfies_both_exact_crossings_before_near_miss():
    # Three separate approaches form an odd graph once the conservative
    # proximity A/C is included; only A/B and B/C actually cross.
    owners = [WireSegmentOwner("contour_loop") for _ in range(16)]
    owners[2] = WireSegmentOwner("inter_group_connector", connector_index=0)
    owners[9] = WireSegmentOwner("inter_group_connector", connector_index=1)
    owners[14] = WireSegmentOwner("within_group_return", group_index=0)
    arc = np.arange(17) * 0.01
    mask, selected = _crossing_plateaus(owners, [(2, 14, 0.0019), (2, 9, 0), (9, 14, 0)], arc, 0.003, False)
    assert mask[2] != mask[9] and mask[9] != mask[14]
    assert len(selected) == 2 and all(d == 0 for _, _, d in selected)
    levels, selected = _crossing_plateaus(owners, [(2, 14, 0), (2, 9, 0), (9, 14, 0)], arc, 0.003, False, layer_count=3)
    assert set(levels[[2, 9, 14]]) == {0, 1, 2} and len(selected) == 3


def test_protected_contour_samples_and_segments_remain_exact():
    p = _short_connector(0)
    source = p.wire_path
    assert apply_crossing_lifts(p, CoilGenConfig(normal_shift_length=0.004), protect_contours=True)

    def protected(path):
        i = np.array([i for i, o in enumerate(path.segment_owners) if o.kind == "contour_loop"])
        return np.stack((path.v[i], path.v[i + 1]), axis=1)

    np.testing.assert_array_equal(protected(p.wire_path), protected(source))


def test_narrow_retraced_return_shortcut_keeps_endpoints_and_protected_winding():
    part = _short_connector(0)
    points = 0.001 * np.array(
        [[0, 25, 0], [0, 0, 0], [0, -15, 0], [1, -30, 0], [2, -20, 0], [1, -10, 0], [0, 5, 0], [0, 24, 0], [10, 24, 0]],
        dtype=float,
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    loop = WireSegmentOwner("contour_loop", loop_index=2)
    rail = WireSegmentOwner("within_group_return", group_index=1)
    owners = [loop, *([rail] * 6), loop]
    repaired, directors, events = _straighten_reversed_returns(
        part, points, normals, owners, CoilGenConfig(interconnection_cut_width=0.006), 0.002
    )
    assert len(events) == 1 and events[0]["removed_axial_backtrack_m"] > 0.025
    np.testing.assert_array_equal(repaired[[0, 1, 7, 8]], points[[0, 1, 7, 8]])
    assert np.all(np.diff(repaired[1:8, 1]) >= 0)
    np.testing.assert_allclose(directors, normals)


@pytest.mark.parametrize("source_chord_offset", [0.0, 0.0002])
def test_return_elbow_uses_a_material_radius_at_the_same_fixed_attachments(source_chord_offset):
    part = _short_connector(0)
    points = 0.001 * np.array([[-25, 0, 0], [-12, 0, 0], [0, 0, 0], [0, 12, 0], [0, 25, 0]], dtype=float)
    points[:, 2] = source_chord_offset
    rail = WireSegmentOwner("within_group_return", group_index=0)
    part.wire_path = WirePath(points, segment_owners=[rail] * 4)
    part.opening_cuts_among_groups = []
    part.connected_groups = []
    cfg = CoilGenConfig(normal_shift_length=0.004, sweep_min_bend_radius=0.006)
    assert apply_crossing_lifts(part, cfg, round_returns=True)
    np.testing.assert_array_equal(part.wire_path.v[[0, -1]], points[[0, -1]])
    interior = part.wire_path.v[(part.wire_path.v[:, 0] < -1e-9) & (part.wire_path.v[:, 1] > 1e-9)]
    assert len(interior) > 8
    np.testing.assert_allclose(part.wire_path.v[:, 2], source_chord_offset, atol=1e-11)
    np.testing.assert_allclose(
        np.linalg.norm(interior - np.array([-0.006, 0.006, source_chord_offset]), axis=1), 0.006, atol=1e-11
    )


def test_contact_provenance_collapses_only_coincident_attachment_samples():
    from coilgen.crossing_lifts import _proposal_contacts

    points = np.array([[-1, 0, 0], [0, 0, 0], [2e-12, 0, 0], [3e-12, 0, 0], [1, 0, 0], [1, 1, 0], [-1, -1, 0]])
    pairs = _proposal_contacts(points, 1e-7, CoilGenConfig())
    # Nonlocal diagonal 5 actually crosses both sides of the attachment.
    assert [(a, b) for a, b, _ in pairs] == [(0, 5), (3, 5)]

    # A real 0.39-nm connector edge is above the vertex-merging tolerance,
    # but is still just the common continuation of its two oriented legs.
    through = np.array([[-1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [3e-10, 2.5e-10, 0.0], [1.0, 1.0, 0.0]])
    assert not _proposal_contacts(through, 1e-7, CoilGenConfig())
    through[-1] = [-1.0, -1.0, 0.0]
    through[-1, 0] += 1e-4  # open, with an actual returning leg
    assert _proposal_contacts(through, 1e-7, CoilGenConfig())


def test_group_metadata_follows_a_second_resampled_proposal():
    from dataclasses import replace

    from coilgen.crossing_lifts import apply_lift_proposal, crossing_lift_candidate

    p = _short_connector(0)
    cfg = CoilGenConfig(normal_shift_length=0.004)
    assert apply_crossing_lifts(p, cfg)
    prior = p.connected_groups[0].spiral_in_v.copy()
    # A second isolated proposal has different sample stations on the same
    # connector approaches. Keep an independent copy of the source-index map.
    flipped = replace(cfg, normal_shift_length=-0.004)
    path, detail = crossing_lift_candidate(p, flipped, ramp_scale=1.3)
    assert apply_lift_proposal(p, flipped, (path, detail))
    curve = p.connected_groups[0].spiral_in_v
    assert prior[0, 2] > 0.0 and curve[0, 2] < 0.0
    source_pos = detail["source_positions"]
    expected = path.v[(source_pos >= 5) & (source_pos <= 6)]
    np.testing.assert_allclose(curve[[0, -1]], expected[[0, -1]], atol=1e-13)
    # Extra shared historical stations may lie in the interior of the newly
    # sampled edges. Every stored group point must still lie on the true wire.
    np.testing.assert_allclose(curve[:, 2], np.interp(curve[:, 0], expected[:, 0], expected[:, 2]), atol=1e-13)


def test_incumbent_proposal_preserves_signed_legacy_sampling_and_geometry():
    from dataclasses import replace

    from coilgen.layout import shift_return_paths
    from coilgen.mounted_crossing_lifts import incumbent_lift_proposal, mounted_gap_proposal

    p = _short_connector(0)
    cfg = CoilGenConfig(normal_shift_length=-0.004)
    shift_return_paths([p], cfg)
    path, detail = incumbent_lift_proposal(p, cfg)
    np.testing.assert_array_equal(path.v, p.wire_path.v)
    assert path.segment_geometry == p.wire_path.segment_geometry
    assert path.segment_owners == p.wire_path.segment_owners
    reconstructed = (
        detail["output_footprint_m"]
        + detail["lift_distance_m"][:, None] * detail["output_normals"]
        + detail["lateral_displacement_m"]
    )
    np.testing.assert_allclose(reconstructed, path.v, rtol=0, atol=1e-12)
    assert incumbent_lift_proposal(p, replace(cfg, skip_normal_shift=True)) is None
    assert mounted_gap_proposal(p, cfg) is None


@pytest.mark.parametrize("closed", [True, False])
def test_short_lateral_edges_use_final_ring_stations_and_exclude_caps(closed):
    from types import SimpleNamespace

    from coilgen.crossing_conditioning import _short_ring_stations

    profile = np.array([[0.0, 0.0], [0.002, 0.0], [0.0, 0.002]])
    z = np.array([0.0, 0.01, 0.0100005, 0.03])
    rings = np.array([np.column_stack((profile, np.full(len(profile), t))) for t in z])
    if closed:
        # Also exercise the periodic interval between the final and first ring.
        rings[-1, 0] = rings[0, 0] + np.array([0.0, 0.0, 1e-6])
    vertices = rings.reshape(-1, 3)
    if not closed:
        vertices = np.vstack((vertices, rings[0].mean(axis=0), rings[-1].mean(axis=0)))
    path = SimpleNamespace(points_m=rings.mean(axis=1), cross_section_m=profile, closed=closed)
    body = SimpleNamespace(
        fabrication_path=path, mesh=SimpleNamespace(vertices=vertices), diagnostics={"edge_length_tolerance_m": 1e-6}
    )
    part = SimpleNamespace(manufacturing_diagnostics={"short_edge_count": 3}, swept_bodies=[body])
    assert _short_ring_stations(part) == ({1, 3} if closed else {1})
    part.manufacturing_diagnostics["short_edge_count"] = 0
    assert not _short_ring_stations(part)
