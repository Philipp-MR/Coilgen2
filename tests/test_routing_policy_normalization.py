"""Every routing stage honors the same normalized engineering enum."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import layout, surface_distance
from coilgen.config import CoilGenConfig
from coilgen.config_validation import validate_config
from coilgen.group_routing import propose_joint_group_routes
from coilgen.routing_objective import routing_selection_policy
from _group_routing_cases import make_group_routing_case
from test_regular_ring_traversal import _config, _part, _positions
from test_regular_route_provenance import _part as provenance_part


@pytest.mark.parametrize("raw,expected", [(" REGULAR ", "regular"), (" PhYsIcAl ", "physical")])
def test_validated_enum_controls_ring_order_corridor_activation_and_provenance(monkeypatch, raw, expected):
    config = _config(raw)
    config.inter_group_connection_path_method = "mesh_geodesic"
    assert not any(item.field == "joint_group_routing_selection_policy" for item in validate_config(config).errors)
    assert routing_selection_policy(config) == expected
    part = layout.topological_loop_grouping([_part()], config)[0]
    positions = np.asarray(_positions(part))
    ordered = np.all(np.diff(positions) > 0) or np.all(np.diff(positions) < 0)
    assert bool(ordered) == (expected == "regular")
    calls = []
    original = surface_distance.SurfaceGeodesicSolver

    def solver(mesh, *, refine_routes=False):
        calls.append(refine_routes)
        return original(mesh, refine_routes=refine_routes)

    monkeypatch.setattr(layout, "SurfaceGeodesicSolver", solver)
    monkeypatch.setattr(surface_distance, "SurfaceGeodesicSolver", solver)
    layout.calculate_group_centers([part])
    layout.interconnect_within_groups([part], config)
    case = make_group_routing_case()
    case.config.joint_group_routing_selection_policy = raw
    case.config.inter_group_connection_path_method = "mesh_geodesic"
    propose_joint_group_routes(case.part, case.config)
    assert len(calls) >= 2 and all(value == (expected == "regular") for value in calls)
    part = provenance_part()
    layout._assign_wire_path_segment_owners(part, config=config)
    assert bool(part.wire_path.segment_owners[1].connector_role) == (expected == "regular")


def test_historical_object_and_missing_config_retain_physical_policy():
    assert routing_selection_policy(None) == "physical"
    assert routing_selection_policy(SimpleNamespace()) == "physical"
    assert routing_selection_policy(CoilGenConfig(joint_group_routing_selection_policy="Unknown")) == "unknown"
