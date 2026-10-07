"""Small directed coil groups for joint return/inter-group routing regressions.

Coordinates and dimensional configuration values are SI.  These are routing
inputs, not certified manufactured outputs: the legacy turnaround contains a
crossing that a joint route must either remove or lift.  Every protected opened
contour is a smooth circular arc with a known traversal direction.
"""

from dataclasses import dataclass
import itertools

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.models import CoilPart, ConnectedGroup, ContourLine, LoopGroup, Mesh


@dataclass
class GroupRoutingCase:
    name: str
    part: CoilPart
    config: CoilGenConfig
    protected_segments: dict[str, tuple[np.ndarray, np.ndarray]]
    protected_contours: tuple[np.ndarray, ...]
    description: str
    expected_constraints: dict[str, object]
    support_normal: np.ndarray


def make_group_routing_case(
    name: str = "shared_corridor",
    *,
    group_count: int = 2,
    scale: float = 1.0,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    gap_angles_degrees: tuple[float, ...] | None = None,
    current_directions: tuple[int, ...] | None = None,
) -> GroupRoutingCase:
    """Build independent 2/3/4-group fixtures with exact directed segment IDs.

    ``rotation`` is a proper 3D rotation; scaling also scales the conductor,
    clearance, radius and lift limits.  ``greedy_trap`` fixes three groups and
    ``blocked_corridor`` adds a third group across the first pair's chord.
    A stress case is not a proof that every possible global route is infeasible.
    """
    if name not in {
        "shared_corridor",
        "facing_corridor",
        "opposed_direction",
        "blocked_corridor",
        "insufficient_bend_lift",
        "greedy_trap",
    }:
        raise ValueError(f"Unknown routing fixture {name!r}.")
    if group_count not in {2, 3, 4} or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Use 2, 3 or 4 groups and a finite positive scale.")
    transform = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    offset = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    if (
        transform.shape != (3, 3)
        or offset.shape != (3,)
        or not np.all(np.isfinite(transform))
        or not np.all(np.isfinite(offset))
        or not np.allclose(transform.T @ transform, np.eye(3), atol=1e-12)
        or not np.isclose(np.linalg.det(transform), 1.0, atol=1e-12)
    ):
        raise ValueError("Fixture transforms require a finite proper rotation and translation.")

    centers = np.column_stack(((np.arange(group_count) - (group_count - 1) / 2) * 0.036, np.zeros(group_count)))
    gap_angles = np.full(group_count, -90.0)
    directions = np.ones(group_count, dtype=int)
    half_gap_degrees = 17.5
    descriptions = {
        "shared_corridor": "Disjoint nested circles share a south-facing open return corridor.",
        "facing_corridor": "Two disjoint nested groups expose their opening corridors toward each other.",
        "opposed_direction": "The second group has the opposite prescribed current circulation; reversing it is forbidden.",
        "blocked_corridor": "A third disjoint group crosses the direct connector between groups zero and one.",
        "insufficient_bend_lift": "Narrow fixed openings and a 50 micrometre lift stress local bend and crossing feasibility.",
        "greedy_trap": "Three separated groups give an explicit nearest-next directed-port length trap.",
    }
    if name == "facing_corridor":
        if group_count != 2:
            raise ValueError("The facing corridor fixture has exactly two groups.")
        gap_angles = np.asarray([0.0, 180.0])
    elif name == "opposed_direction":
        directions[1] = -1
    elif name == "blocked_corridor":
        centers = np.asarray([[-0.036, 0.0], [0.036, 0.0], [0.0, -0.012]])
        gap_angles = np.asarray([-90.0, -90.0, 90.0])
        directions = np.ones(3, dtype=int)
    elif name == "insufficient_bend_lift":
        half_gap_degrees = 3.0
    elif name == "greedy_trap":
        centers = np.asarray([[0.0, 0.0], [0.045, 0.0], [0.018, 0.046]])
        gap_angles = np.asarray([60.0, 90.0, 270.0])
        directions = np.ones(3, dtype=int)
    if gap_angles_degrees is not None:
        gap_angles = np.asarray(gap_angles_degrees, dtype=float)
        if gap_angles.shape != (len(centers),) or not np.all(np.isfinite(gap_angles)):
            raise ValueError("Provide one finite opening angle per group.")
    if current_directions is not None:
        directions = np.asarray(current_directions, dtype=int)
        if directions.shape != (len(centers),) or not np.all(np.isin(directions, [-1, 1])):
            raise ValueError("Provide one prescribed +1/-1 current direction per group.")

    def xyz(points):
        values = np.asarray(points, dtype=float)
        if values.shape[-1] == 2:
            values = np.column_stack((values, np.zeros(len(values))))
        return scale * values @ transform.T + offset

    lower = np.min(centers, axis=0) - 0.032
    upper = np.max(centers, axis=0) + 0.032
    mesh = normalize_mesh(
        Mesh(
            vertices=xyz([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]]),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
        )
    )
    part = CoilPart(mesh=mesh, contour_step=1.0)
    protected = {}
    opened_contours = []
    for group_index, (center, gap_angle, direction) in enumerate(zip(centers, gap_angles, directions)):
        full_loops, opened_loops, indices = [], [], []
        for local_index, radius in enumerate((0.012, 0.008)):
            start = np.deg2rad(gap_angle + direction * half_gap_degrees)
            arc = np.deg2rad(360 - 2 * half_gap_degrees)
            angles = start + direction * np.linspace(0.0, arc, int(np.ceil(np.rad2deg(arc) / 2.0)) + 1)
            points = xyz(center + radius * np.column_stack((np.cos(angles), np.sin(angles))))
            # Retain the exact opened samples in the full source contour.
            closure_angles = angles[-1] + direction * np.linspace(0.0, np.deg2rad(2 * half_gap_degrees), 20)[1:-1]
            closure = xyz(center + radius * np.column_stack((np.cos(closure_angles), np.sin(closure_angles))))
            full_points = np.vstack((points, closure, points[:1]))
            potential = float(direction * (local_index + 1))
            full = ContourLine(full_points, potential=potential, current_orientation=float(direction))
            opened = ContourLine(points.copy(), potential=potential, current_orientation=float(direction))
            indices.append(len(part.contour_lines))
            part.contour_lines.append(full)
            full_loops.append(full)
            opened_loops.append(opened)
            opened_contours.append(points.copy())
            for segment_index, (a, b) in enumerate(zip(points[:-1], points[1:])):
                protected[f"g{group_index}:l{local_index}:s{segment_index}"] = (a.copy(), b.copy())
        outer, inner = (loop.v for loop in opened_loops)
        spiral_in = np.vstack((outer, inner))
        spiral_out = np.vstack((inner, outer))
        # This is the same explicit turnaround + outward rail decomposition as
        # layout._connect_opened_loops, without reversing either contour.
        return_path = np.vstack((inner[-1], inner[0], outer[0]))
        part.connected_groups.append(
            ConnectedGroup(
                v=np.vstack((spiral_in, return_path, spiral_in[:1])),
                return_path_v=return_path,
                spiral_in_v=spiral_in,
                spiral_out_v=spiral_out,
                outer_loop_indices=(indices[0],),
            )
        )
        part.groups.append(
            LoopGroup(
                loops=full_loops,
                loop_indices=indices,
                opened_loops=opened_loops,
                opening_cuts_v=[loop.v[[0, -1]].copy() for loop in opened_loops],
                outer_loop_indices=(indices[0],),
            )
        )
        part.loop_groups.append(indices)
    n_groups = len(part.groups)
    part.group_levels = [list(range(n_groups))]
    part.level_positions = [[]]
    part.group_centers_v = xyz(centers)
    config = CoilGenConfig(
        surface_is_cylinder_flag=False,
        inter_group_connection_path_method="direct_3d",
        inter_group_cut_alignment="none",
        interconnection_cut_width=0.006 * scale,
        min_inter_group_connector_clearance=0.001 * scale,
        min_wire_distance_m=0.001 * scale,
        conductor_cross_section_width=0.0006 * scale,
        conductor_cross_section_height=0.0004 * scale,
        normal_shift_length=(0.00005 if name == "insufficient_bend_lift" else 0.002) * scale,
        sweep_min_bend_radius=(0.003 if name == "insufficient_bend_lift" else 0.0012) * scale,
        sweep_bend_mode="fillet",
        sweep_max_chord_error=0.000005 * scale,
        sweep_max_segment_length=0.0008 * scale,
        sweep_max_frame_rotation_deg=5.0,
        sweep_bend_arc_angle_deg=2.0,
        sweep_self_intersection_check="capsule_fast",
    )
    constraints = {
        "group_count": n_groups,
        "group_centers_m": xyz(centers).tolist(),
        "gap_angles_degrees_in_fixture_plane": gap_angles.tolist(),
        "current_directions": directions.tolist(),
        "protected_contour_radii_m": [0.012 * scale, 0.008 * scale],
        "opening_angle_degrees": 2 * half_gap_degrees,
        "every_directed_protected_segment_required_once": True,
        "source_is_manufacturing_certificate": False,
    }
    return GroupRoutingCase(
        name, part, config, protected, tuple(opened_contours), descriptions[name], constraints, transform[:, 2]
    )


def directed_port_length_witness(case: GroupRoutingCase) -> dict[str, object]:
    """Compare a bounded greedy order against exhaustive directed port costs.

    This deliberately measures only connector chord length and makes no support,
    clearance or finite-body assertion.  Starting group and variant are fixed
    identically for both searches, so the comparison is reproducible.
    """
    groups = case.part.connected_groups
    ports = [(group.spiral_in_v, group.spiral_out_v) for group in groups]

    def length(order, variants):
        return float(
            sum(
                np.linalg.norm(
                    ports[order[i]][variants[i]][-1]
                    - ports[order[(i + 1) % len(order)]][variants[(i + 1) % len(order)]][0]
                )
                for i in range(len(order))
            )
        )

    order, variants = [0], [0]
    while len(order) < len(groups):
        _, group_index, variant = min(
            (float(np.linalg.norm(ports[order[-1]][variants[-1]][-1] - ports[g][v][0])), g, v)
            for g in range(len(groups))
            if g not in order
            for v in range(2)
        )
        order.append(group_index)
        variants.append(variant)
    exhaustive = min(
        (length((0, *rest), (0, *tail)), (0, *rest), (0, *tail))
        for rest in itertools.permutations(range(1, len(groups)))
        for tail in itertools.product(range(2), repeat=len(groups) - 1)
    )
    return {
        "greedy_order": order,
        "greedy_variants": variants,
        "greedy_connector_length_m": length(order, variants),
        "best_order": list(exhaustive[1]),
        "best_variants": list(exhaustive[2]),
        "best_connector_length_m": exhaustive[0],
        "is_physical_feasibility_certificate": False,
    }
