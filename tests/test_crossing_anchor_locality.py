"""Distinct physical crossing minima must survive event clustering."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.layout import (
    _crossing_event_segment_selections,
    _crossing_lift_anchors,
    _normal_shift_routes_by_segment,
    _owned_crossing_bridge_spans,
    find_wire_segment_contacts_3d,
)
from coilgen.models import WirePath, WirePathGeometry, WireSegmentOwner


def _nearby_crossings(*, same_owner, adjacent=False, underpass_owner=False):
    # Two overpass crossings are only 4 mm apart, inside a 15 mm event window.
    # The second has a nonzero physical gap; it is not a duplicate of the first.
    over = [[-2, -1, 0], [-2, 1, 0]]
    over += [[6, -1, 1]] if adjacent else [[2, 1, 0.5], [2, -1, 0.5]]
    points = np.asarray(over + [[30, -20, 30], [30, 30, 30], [-5, 0, 0], [10, 0, 0]]) * 1e-3
    contour = WireSegmentOwner("contour_loop", group_index=0, loop_index=0)
    first = WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="primary")
    second = (
        first if same_owner else WireSegmentOwner("inter_group_connector", connector_index=2, connector_role="primary")
    )
    owners = [contour] * (len(points) - 1)
    geometry = [WirePathGeometry.SURFACE] * len(owners)
    over_segments = (0, len(over) - 2)
    for index in range(len(over) - 1) if same_owner else over_segments:
        owners[index] = first if index == 0 else second
        geometry[index] = WirePathGeometry.FREE_3D
    under = len(points) - 2
    if underpass_owner:
        owners[under] = WireSegmentOwner("inter_group_connector", connector_index=1, connector_role="primary")
        geometry[under] = WirePathGeometry.FREE_3D
    path = WirePath(points, segment_owners=owners, segment_geometry=geometry)
    part = SimpleNamespace(wire_path=path, connected_groups=[], opening_cuts_among_groups=[])
    routes = _normal_shift_routes_by_segment(part)
    # Keep all real contacts to this underpass, including the higher-gap sample
    # between the separate minima. Remote path joins do not concern this event.
    contacts = [contact for contact in find_wire_segment_contacts_3d(points, 0.002661) if contact[1] == under]
    events = _crossing_event_segment_selections(contacts, routes, points, 0.015)
    return path, contacts, events, over_segments, under


@pytest.mark.parametrize("same_owner", [True, False])
def test_distinct_crossings_with_different_gaps_both_receive_anchors(same_owner):
    path, contacts, events, over, under = _nearby_crossings(same_owner=same_owner)
    distances = {(a, b): gap for a, b, gap in contacts}
    assert distances[(over[0], under)] == pytest.approx(0.0, abs=1e-12)
    assert distances[(over[1], under)] == pytest.approx(0.0005, abs=1e-12)
    anchors = _crossing_lift_anchors(events, contacts, path.v)
    assert {(a, b) for a, b, *_ in anchors} >= {(over[0], under), (over[1], under)}
    if not same_owner:
        spans = _owned_crossing_bridge_spans(path, events, [(a, b, d) for a, b, _x, _y, d in anchors])
        assert {start for start, _end, _under in spans} >= set(over)


def test_adjacent_overpass_segments_can_have_distinct_interior_crossings():
    path, contacts, events, over, under = _nearby_crossings(same_owner=True, adjacent=True)
    anchors = _crossing_lift_anchors(events, contacts, path.v)
    by_pair = {(a, b): (x, y, d) for a, b, x, y, d in anchors}
    assert {(over[0], under), (over[1], under)} <= by_pair.keys()
    assert 0.0 < by_pair[(over[0], under)][0] < 1.0
    assert 1.0 < by_pair[(over[1], under)][0] < 2.0
    assert by_pair[(over[1], under)][2] > 1e-4
    assert abs(by_pair[(over[0], under)][1] - by_pair[(over[1], under)][1]) > 0.1


def test_neighboring_crossings_keep_their_individual_route_priorities():
    path, contacts, events, over, under = _nearby_crossings(same_owner=False, underpass_owner=True)
    anchors = _crossing_lift_anchors(events, contacts, path.v)
    # Route 1 wins against route 0 at the first crossing, but loses against
    # route 2 at the second. A majority vote over the event reverses one role.
    selected = {(a, b) for a, b, *_ in anchors}
    assert (under, over[0]) in selected
    assert (over[1], under) in selected
    assert (over[0], under) not in selected


def test_collinear_subdivision_preserves_crossing_locations():
    path, contacts, events, _over, _under = _nearby_crossings(same_owner=True)
    original = _crossing_lift_anchors(events, contacts, path.v)
    points = np.empty((2 * len(path.v) - 1, 3))
    points[::2] = path.v
    points[1::2] = (path.v[:-1] + path.v[1:]) / 2
    refined_path = WirePath(
        points,
        segment_owners=[owner for owner in path.segment_owners for _ in range(2)],
        segment_geometry=[geometry for geometry in path.segment_geometry for _ in range(2)],
    )
    part = SimpleNamespace(wire_path=refined_path, connected_groups=[], opening_cuts_among_groups=[])
    under_start = len(points) - 3
    refined_contacts = [
        contact for contact in find_wire_segment_contacts_3d(points, 0.002661) if contact[1] >= under_start
    ]
    refined_events = _crossing_event_segment_selections(
        refined_contacts, _normal_shift_routes_by_segment(part), points, 0.015
    )
    refined = _crossing_lift_anchors(refined_events, refined_contacts, points)

    def locations(vertices, anchors):
        result = set()
        for over, under, over_position, under_position, _gap in anchors:
            first = vertices[over] + (over_position - over) * (vertices[over + 1] - vertices[over])
            second = vertices[under] + (under_position - under) * (vertices[under + 1] - vertices[under])
            result.add(tuple(np.round(np.r_[first, second], 12)))
        return result

    assert locations(path.v, original) == locations(points, refined)


def test_closed_seam_endpoint_is_not_minimum_when_previous_interval_is_closer():
    points = 1e-3 * np.array([[0, 0, 0], [1, 0, 0], [-2, -3, 0], [2, -3, 0], [-1, -1, 0], [0, 0, 0]], dtype=float)
    # Segment 0 is closest to segment 2 at the seam. Moving backward into
    # segment 4 decreases the distance, so the seam is not a crossing minimum.
    anchors = _crossing_lift_anchors([((0,), (2,))], [(0, 2, 0.003)], points)
    assert anchors == []


def test_closed_seam_true_minimum_is_retained_from_both_incident_intervals():
    points = 1e-3 * np.array([[0, 0, 0], [1, 0, 0], [0, -2, 1], [0, 2, 1], [-1, 0, 0], [0, 0, 0]], dtype=float)
    anchors = _crossing_lift_anchors([((0, 4), (2,))], [(0, 2, 0.001), (4, 2, 0.001)], points)
    assert {(a, b) for a, b, *_ in anchors} == {(0, 2), (4, 2)}
    assert {position for _a, _b, position, *_ in anchors} == {0.0, 5.0}
