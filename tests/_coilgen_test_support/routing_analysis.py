"""Shared fixtures for routing analysis."""

import numpy as np
from coilgen.application.service import RunRecord
from coilgen.models import FabricationPath
from coilgen.scientific_arrays import fabrication_frame_arrays


def plane(offset=0.0):
    vertices = np.array([[offset, -0.1, -0.1], [offset, 0.1, -0.1], [offset, 0.1, 0.1], [offset, -0.1, 0.1]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def record(tmp_path, *, two_planes=False, frames=False):
    vertices, faces = plane(0.05)
    points = np.array(
        [[0.05, -0.02, -0.02], [0.05, 0.02, 0.02], [0.05, -0.02, 0.02], [0.05, 0.02, -0.02], [0.05, -0.02, -0.02]]
    )
    arrays = {
        "vertices_m": vertices,
        "faces": faces,
        "support_0_vertices_m": vertices,
        "support_0_faces": faces,
        "route_0_points_m": points,
        "route_0_owner_kind": np.array(
            ["contour_loop", "contour_group", "within_group_return", "inter_group_connector"]
        ),
        "route_0_owner_connector_role": np.array(["", "joint_spiral_join", "", "primary"]),
        "route_0_owner_group_index": np.array([0, 0, 0, -1]),
    }
    metadata = {
        "name": "Exact routing evidence",
        "summary": {"accepted": False},
        "fabrication": {"ready": False},
        "routing": {
            "parts": [{"part_index": 0, "current_A": -2.0, "joint_routing": {"selected_route_status": "invalid"}}]
        },
    }
    if two_planes:
        arrays.update(
            {
                "support_1_vertices_m": vertices - [0.1, 0, 0],
                "support_1_faces": faces,
                "route_1_points_m": points - [0.1, 0, 0],
            }
        )
    if frames:
        path = FabricationPath(
            points[:-1] + [0.001, 0, 0],
            np.tile([0, 0, 1.0], (4, 1)),
            np.tile([1.0, 0, 0], (4, 1)),
            np.tile([0, 1.0, 0], (4, 1)),
            np.array([[0.0, 0.0], [0.002, 0.0], [0.0, 0.001]]),
            points[:-1],
            closed=True,
        )
        prefix = "review_conductor_0"
        arrays[prefix + "_points_m"] = path.wire_points_m
        arrays.update(fabrication_frame_arrays(prefix, path))
        metadata["fabrication"]["paths"] = [
            {
                "body_id": "part001_body0001",
                "part_index": 1,
                "diagnostics": {
                    "manufacturing_ok": False,
                    "manufacturing_failure_reasons": "minimum_bend_radius_violation",
                    "sweep_ring_count": 4,
                    "sweep_final_bend_point_count": 4,
                    "sweep_final_min_bend_point_index": 1,
                    "sweep_final_min_bend_radius_m": 0.001,
                    "sweep_required_bend_radius_m": 0.006,
                },
            }
        ]
    return RunRecord(tmp_path, metadata, arrays)
