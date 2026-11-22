"""Added-route costs must never include protected winding intervals."""

from coilgen.config import CoilGenConfig
from coilgen.layout import _assign_wire_path_segment_owners
from coilgen.models import WirePathGeometry

from _coilgen_test_support.regular_route_provenance import (
    _part as _part,
)


def test_regular_route_labels_exact_winding_join_and_return_separately():
    part = _part()
    _assign_wire_path_segment_owners(part, config=CoilGenConfig(joint_group_routing_selection_policy="regular"))
    owners = part.wire_path.segment_owners
    assert [owner.kind for owner in owners] == [
        "contour_loop",
        "contour_group",
        "within_group_return",
        "within_group_return",
    ]
    assert owners[0].loop_index == 7
    assert owners[1].connector_role == "joint_spiral_join"
    assert all(value == WirePathGeometry.SURFACE for value in part.wire_path.segment_geometry)


def test_physical_route_keeps_historical_group_ownership():
    part = _part()
    _assign_wire_path_segment_owners(part, config=CoilGenConfig(joint_group_routing_selection_policy="physical"))
    assert [owner.kind for owner in part.wire_path.segment_owners] == [
        "contour_loop",
        "contour_group",
        "contour_group",
        "contour_group",
    ]
    assert not any(owner.connector_role for owner in part.wire_path.segment_owners)


def test_regular_route_unknown_intervals_stay_ambiguous():
    part = _part()
    # A changed interval cannot inherit protected or return ownership simply
    # from its neighbouring points. Its lower-bound metric stays explicit.
    part.wire_path.v[2] = [1.1, 1, 0]
    _assign_wire_path_segment_owners(part, config=CoilGenConfig(joint_group_routing_selection_policy="regular"))
    assert all(
        owner.kind == "contour_group" and not owner.connector_role for owner in part.wire_path.segment_owners[1:3]
    )
    assert part.wire_path.segment_owners[0].kind == "contour_loop"
    assert part.wire_path.segment_owners[-1].kind == "within_group_return"
