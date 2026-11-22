"""Shared fixtures for winding sketch."""

import numpy as np
from coilgen.interconnection_editor import InterconnectionEditor
from coilgen.models import CoilPart, Mesh, WirePath, WireSegmentOwner
from _coilgen_test_support.interconnection_editor import routed as routed


def crossing_editor():
    mesh = Mesh(
        np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]), np.array([[0, 1, 2], [0, 2, 3]])
    )
    points = np.array([[-0.03, 0, 0], [0.03, 0, 0], [0, -0.03, 0], [0, 0.03, 0], [-0.03, 0, 0]])
    owners = [
        WireSegmentOwner("within_group_connector" if i % 2 == 0 else "contour_group", group_index=i) for i in range(4)
    ]
    return InterconnectionEditor(CoilPart(mesh=mesh, wire_path=WirePath(points, owners), contour_step=1.0))
