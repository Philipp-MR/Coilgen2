"""Bounded directed attachment windows and shared return-corridor proposals.

Ports are positions on the original contour polylines, in metres of directed
arc length. No source contour is moved or reversed. These are geometric search
contexts, not certificates of clearance, support, field accuracy or feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .models import ContourLine
from .regular_rings import ring_geometry, ring_support_axis


class ContourArc:
    def __init__(self, loop):
        points = np.array(loop.v, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.all(np.isfinite(points)):
            raise ValueError("Attachment search needs a finite closed source contour")
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack((points, points[0]))
        self.points = points
        self.delta = np.diff(points, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        if np.any(self.lengths <= 1e-12):
            raise ValueError("Ambiguous degenerate source contour")
        self.positions = np.r_[0.0, np.cumsum(self.lengths)]
        self.perimeter = float(self.positions[-1])
        self.loop = loop

    def point(self, position):
        s = float(position) % self.perimeter
        index = min(np.searchsorted(self.positions, s, side="right") - 1, len(self.lengths) - 1)
        fraction = (s - self.positions[index]) / self.lengths[index]
        if fraction < 1e-12:
            return self.points[index].copy()
        if fraction > 1 - 1e-12:
            return self.points[index + 1].copy()
        return self.points[index] + fraction * self.delta[index]

    def position(self, point):
        fraction = np.clip(np.einsum("ij,ij->i", point - self.points[:-1], self.delta) / self.lengths**2, 0, 1)
        error = np.linalg.norm(self.points[:-1] + fraction[:, None] * self.delta - point, axis=1)
        index = int(np.argmin(error))
        if error[index] > max(1e-12, self.perimeter * 1e-10):
            raise ValueError("Attachment is not on its original contour")
        return float(self.positions[index] + fraction[index] * self.lengths[index]) % self.perimeter

    def opening_center(self, opened):
        end, start = self.position(opened.v[-1]), self.position(opened.v[0])
        gap = (start - end) % self.perimeter
        if gap <= 1e-12 or gap >= self.perimeter - 1e-12:
            raise ValueError("Source contour has no unambiguous opening")
        return (end + 0.5 * gap) % self.perimeter, gap

    def open(self, center, span):
        if not np.isfinite(center) or not np.isfinite(span) or not 1e-10 < span < 0.5 * self.perimeter:
            raise ValueError("A bounded opening must remove less than half the source contour")
        start = (center + span / 2) % self.perimeter
        travel = self.perimeter - span
        offsets = (self.positions[:-1] - start) % self.perimeter
        kept = np.flatnonzero((offsets > 1e-12) & (offsets < travel - 1e-12))
        kept = kept[np.argsort(offsets[kept], kind="stable")]
        points = np.vstack((self.point(start), self.points[kept], self.point(start + travel)))
        kwargs = {}
        faces, bary = self.loop.segment_faces, self.loop.segment_barycentric
        if faces is not None and bary is not None and len(faces) == len(self.lengths):
            absolute = np.r_[start, start + offsets[kept], start + travel]
            out_faces, out_bary = [], []
            for a, b in zip(absolute[:-1], absolute[1:]):
                # Locate the interval interior: modulo at its seam endpoint
                # can round to P-ulp and select the preceding source face.
                local = ((a + b) / 2) % self.perimeter
                j = min(np.searchsorted(self.positions, local, side="right") - 1, len(self.lengths) - 1)
                middle = (local - self.positions[j]) / self.lengths[j]
                half = (b - a) / (2 * self.lengths[j])
                fa, fb = middle - half, middle + half
                tolerance = 32 * np.finfo(float).eps * max(1.0, self.perimeter / self.lengths[j])
                if fa < -tolerance or fb > 1 + tolerance:
                    raise ValueError("Opened interval does not retain one source segment")
                fa, fb = np.clip([fa, fb], 0.0, 1.0)
                out_faces.append(faces[j])
                out_bary.append(bary[j, 0] + np.array([fa, fb])[:, None] * (bary[j, 1] - bary[j, 0]))
            kwargs.update(segment_faces=np.asarray(out_faces), segment_barycentric=np.asarray(out_bary))
        opened = ContourLine(
            points,
            self.loop.potential,
            self.loop.current_orientation,
            turn_multiplicity=self.loop.turn_multiplicity,
            **kwargs,
        )
        return opened, points[[0, -1]].copy()


@dataclass(frozen=True)
class LoopPort:
    center_m: float
    span_m: float


@dataclass(frozen=True)
class GroupPortPlan:
    side: str
    ports: tuple[LoopPort, ...]
    return_fraction: float = 0.5


@dataclass(frozen=True)
class RoutingPortPlan:
    name: str
    groups: dict[int, GroupPortPlan]
    diagnostics: dict


def _source_orientation(arc, center, gap, normal, origin=None):
    """Orient a seam displacement in the support's intrinsic local frame.

    Saddle contours have zero projected area along an annular support's axis;
    its floating-point sign cannot identify their traversal direction. Use the
    directed original opening chord at the attachment instead. A nearly axial
    chord uses the axial coordinate explicitly, rather than the noise in its
    circumferential component. Unresolved directions do not authorize a move.
    """
    tolerance = 1e-8
    if origin is not None:
        point = arc.point(center)
        radial = point - origin
        radial -= (radial @ normal) * normal
        radius = float(np.linalg.norm(radial))
        tangent = arc.point(center + gap / 2) - arc.point(center - gap / 2)
        length = float(np.linalg.norm(tangent))
        if radius <= 1e-12 or length <= 1e-12:
            return None, "degenerate_attachment"
        tangent /= length
        circumferential = np.cross(normal, radial / radius)
        for basis, direction in (("local_circumferential", circumferential), ("local_axial", normal)):
            dot = float(tangent @ direction)
            if abs(dot) > tolerance:
                return (1 if dot > 0 else -1), basis
        return None, "unresolved_attachment_direction"
    relative = arc.points - arc.points[0]
    area = np.cross(relative[:-1], relative[1:]).sum(axis=0)
    length = float(np.linalg.norm(area))
    dot = float(area @ normal)
    if length <= 1e-24 or abs(dot) <= tolerance * length:
        return None, "unresolved_planar_orientation"
    return (1 if dot > 0 else -1), "planar_signed_area"


def _routing_bands(part):
    frame = ring_support_axis(part.mesh)
    if frame is None:
        return [], None
    origin, axis = frame
    all_rows = []
    for gi, group in enumerate(part.groups):
        rows = []
        for li, loop in enumerate(group.loops):
            geometry = ring_geometry(loop.v, origin, axis)
            if geometry is None:
                break
            rows.append((geometry["position"], gi, li, geometry))
        else:
            all_rows.extend(rows)
    all_rows.sort(key=lambda row: row[0])
    blocks = []
    for _position, gi, li, geometry in all_rows:
        if not blocks or blocks[-1]["group_index"] != gi:
            blocks.append(
                {
                    "group_index": gi,
                    "local_loop_indices": [],
                    "minimum_m": geometry["minimum"],
                    "maximum_m": geometry["maximum"],
                }
            )
        blocks[-1]["local_loop_indices"].append(li)
        blocks[-1]["maximum_m"] = max(blocks[-1]["maximum_m"], geometry["maximum"])
    return blocks, frame


def propose_routing_port_plans(part, config, *, maximum_candidates, failure_witnesses=()):
    """Jointly allocate bounded seam lanes, then move ports continuously on sources.

    The opening and lane bounds are explicit configuration controls. Physical
    constraints are not changed. Contiguous axial bands are planning records;
    they never replace electromagnetic groups or reverse directed turns.
    """
    from .layout import _force_cut_selection_for_group, _within_group_cut_width
    from .manufacturing import (
        conductor_cross_section_points,
        _cross_section_envelope_radius,
        _required_centerline_clearance,
    )

    if maximum_candidates <= 0 or not config.joint_group_routing_search_cuts:
        return []
    arcs, seeds = {}, {}
    free = [
        gi
        for gi, group in enumerate(part.groups)
        if _force_cut_selection_for_group(config, gi) is None and group.opened_loops
    ]
    for gi in free:
        group = part.groups[gi]
        if len(group.opened_loops) != len(group.loops) or any(loop.turn_multiplicity != 1 for loop in group.loops):
            continue
        try:
            arcs[gi] = [ContourArc(loop) for loop in group.loops]
            seeds[gi] = [arc.opening_center(opened) for arc, opened in zip(arcs[gi], group.opened_loops)]
        except ValueError:
            arcs.pop(gi, None)
    if not arcs:
        return []
    section = conductor_cross_section_points(config)
    radius = max(float(config.sweep_min_bend_radius), 2 * _cross_section_envelope_radius(section))
    clearance = _required_centerline_clearance(section, config)
    nominal = _within_group_cut_width(config)
    max_factor = float(config.joint_group_routing_cut_span_max_factor)
    extent = float(config.joint_group_routing_seam_search_extent_rel)
    blocks, frame = _routing_bands(part)
    origin = None
    if frame is None:
        triangles = part.mesh.vertices[part.mesh.faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]).sum(axis=0)
        if np.linalg.norm(normal) <= 1e-15:
            return []
        normal = normal / np.linalg.norm(normal)
    else:
        origin, normal = frame
    orientations, unavailable = {}, {}
    for gi in list(arcs):
        values = [_source_orientation(arc, base, gap, normal, origin) for arc, (base, gap) in zip(arcs[gi], seeds[gi])]
        if any(sign is None for sign, _basis in values):
            unavailable[gi] = [basis for sign, basis in values if sign is None]
            del arcs[gi]
        else:
            orientations[gi] = values
    if not arcs:
        return []
    # Witnesses prioritize the failing source group without relying on world axes.
    failed_groups = {int(w.get("owner", {}).get("group_index", -1)) for w in failure_witnesses if isinstance(w, dict)}
    order = sorted(arcs, key=lambda gi: (gi not in failed_groups, gi))
    allocation = {gi: (rank - (len(order) - 1) / 2) * clearance for rank, gi in enumerate(order)}
    patterns = [
        (0.0, False, 0.5),
        (0.0, True, 0.5),
        (0.5, False, 0.25),
        (-0.5, False, 0.75),
        (1.0, True, 0.25),
        (-1.0, True, 0.75),
        (0.25, True, 0.5),
        (-0.25, True, 0.5),
    ]
    for phase in np.linspace(-0.875, 0.875, 8):
        for separated, return_fraction in ((True, 0.25), (False, 0.75), (True, 0.5)):
            patterns.append((float(phase), separated, return_fraction))
    plans = []
    for phase, separated, return_fraction in patterns:
        groups, rows = {}, []
        for gi in sorted(arcs):
            ports = []
            for li, (arc, (base, gap)) in enumerate(zip(arcs[gi], seeds[gi])):
                lane = allocation[gi] if separated else 0.0
                span = min(
                    max(gap, 2.5 * radius + 2 * clearance + 2 * abs(lane)), max_factor * nominal, 0.45 * arc.perimeter
                )
                if span < gap * (1 - 1e-8):
                    # A factor of one preserves existing openings rather than shrinking them.
                    span = gap
                sign, basis = orientations[gi][li]
                offset = sign * (phase * extent * arc.perimeter + lane)
                offset = float(np.clip(offset, -extent * arc.perimeter, extent * arc.perimeter))
                port = LoopPort((base + offset) % arc.perimeter, span)
                ports.append(port)
                rows.append(
                    {
                        "group_index": gi,
                        "local_loop_index": li,
                        "center_arclength_m": port.center_m,
                        "removed_arc_m": span,
                        "source_perimeter_m": arc.perimeter,
                        "nominal_chord_m": nominal,
                        "seam_shift_m": offset,
                        "lane_offset_m": lane,
                        "directed_attachment_sign": sign,
                        "orientation_basis": basis,
                    }
                )
            groups[gi] = GroupPortPlan(part.groups[gi].within_group_cut_side, tuple(ports), return_fraction)
        diagnostics = {
            "scope": "bounded_shared_corridor_and_directed_port_proposal",
            "ports": rows,
            "axial_routing_blocks": blocks,
            "group_membership_unchanged": True,
            "lane_separation_target_m": clearance,
            "minimum_radius_m": radius,
            "cut_span_max_factor": max_factor,
            "seam_search_extent_rel": extent,
            "return_fraction": return_fraction,
            "separated_lane_proposal": separated,
            "physical_constraints_relaxed": False,
            "feasibility_proven": False,
            "unavailable_orientation_groups": unavailable,
        }
        plans.append(RoutingPortPlan(f"shared_ports_{len(plans)}", groups, diagnostics))
        if len(plans) >= maximum_candidates:
            break
    return plans


def connected_group_from_ports(group, plan, solver, *, allow_direct_fallback, corner_step_deg):
    """Build a continuous return seed through a reserved lane, not each loop start."""
    from .layout import _connect_opened_loops, _connect_path_sections_with_connectors, _stack_path_sections

    if len(group.loops) != len(plan.ports):
        raise ValueError("Attachment plan must cover every source loop exactly once")
    arcs = [ContourArc(loop) for loop in group.loops]
    opened, cuts = zip(*(arc.open(port.center_m, port.span_m) for arc, port in zip(arcs, plan.ports)))
    group.opened_loops, group.opening_cuts_v = list(opened), list(cuts)
    connected = _connect_opened_loops(
        list(opened),
        solver,
        allow_direct_fallback=allow_direct_fallback,
        corner_radius=0.0,
        corner_step_deg=corner_step_deg,
    )
    lane = [
        arc.point(port.center_m + (plan.return_fraction - 0.5) * port.span_m) for arc, port in zip(arcs, plan.ports)
    ]
    sections = [opened[-1].v[-1:], np.asarray(lane[-1:]), np.asarray(lane[:1]), opened[0].v[:1]]
    sections = [
        section
        for i, section in enumerate(sections)
        if i == 0 or np.linalg.norm(section[0] - sections[i - 1][-1]) > 1e-12
    ]
    return_path = _connect_path_sections_with_connectors(sections, solver, allow_direct_fallback=allow_direct_fallback)
    if len(return_path) < 2:
        raise ValueError("Continuous supported return lane unavailable")
    connected.return_path_v = return_path
    connected.v = _stack_path_sections((connected.spiral_in_v, return_path))
    connected.outer_loop_indices = tuple(group.outer_loop_indices)
    group.within_group_cut_side = plan.side
    group.within_group_cut_pair_indices = ()
    group.within_group_cut_metrics = {
        "planned_cut_context": 1.0,
        "minimum_removed_arc_m": min(port.span_m for port in plan.ports),
        "maximum_removed_arc_m": max(port.span_m for port in plan.ports),
    }
    return connected


def audit_shared_corridor_reservations(part, config, *, maximum_pairs=2000000, maximum_broad_phase_pairs=8000000):
    """Rank cross-group added corridors with exact 3D segment distances.

    This seed audit cannot certify a smooth or finite-width body. It excludes
    retained turns and never removes a family merely because lift is needed.
    A bounded unfinished audit remains explicitly unavailable.
    """
    from .geometry_kernels import segment_distances
    from .manufacturing import _required_centerline_clearance, conductor_cross_section_points

    required = _required_centerline_clearance(conductor_cross_section_points(config), config)
    corridors = []
    for group, connected in zip(part.groups, part.connected_groups):
        protected = {(tuple(a), tuple(b)) for loop in group.opened_loops for a, b in zip(loop.v[:-1], loop.v[1:])}
        joins = [
            np.array([a, b])
            for a, b in zip(connected.spiral_in_v[:-1], connected.spiral_in_v[1:])
            if (tuple(a), tuple(b)) not in protected
        ]
        returns = [np.array([a, b]) for a, b in zip(connected.return_path_v[:-1], connected.return_path_v[1:])]
        corridors.append(np.asarray(joins + returns).reshape(-1, 2, 3))
    pairs, broad_pairs, exact_contacts, violations, minimum = 0, 0, 0, 0, float("inf")
    for gi, first in enumerate(corridors):
        for gj in range(gi + 1, len(corridors)):
            second = corridors[gj]
            if not len(first) or not len(second):
                continue
            # Conservative expanded AABBs keep distant pairs out of numerical work.
            for section in np.array_split(first, max(1, int(np.ceil(len(first) / 64)))):
                for other in np.array_split(second, max(1, int(np.ceil(len(second) / 1024)))):
                    work = len(section) * len(other)
                    if broad_pairs + work > maximum_broad_phase_pairs:
                        return {
                            "completed": False,
                            "reason": "corridor_broad_phase_budget_exhausted",
                            "evaluated_pairs": pairs,
                            "broad_phase_pairs": broad_pairs,
                        }
                    broad_pairs += work
                    a, b = np.nonzero(
                        np.all(section.min(axis=1)[:, None] <= other.max(axis=1)[None] + required, axis=2)
                        & np.all(other.min(axis=1)[None] <= section.max(axis=1)[:, None] + required, axis=2)
                    )
                    if pairs + len(a) > maximum_pairs:
                        return {
                            "completed": False,
                            "reason": "corridor_pair_budget_exhausted",
                            "evaluated_pairs": pairs,
                            "broad_phase_pairs": broad_pairs,
                        }
                    pairs += len(a)
                    if not len(a):
                        continue
                    distances = segment_distances(
                        section[a, 0], section[a, 1] - section[a, 0], other[b, 0], other[b, 1] - other[b, 0]
                    )
                    minimum = min(minimum, float(np.min(distances)))
                    exact_contacts += int(np.count_nonzero(distances <= 1e-10))
                    violations += int(np.count_nonzero(distances < required))
    return {
        "completed": True,
        "scope": "cross_group_added_seed_segments_only",
        "evaluated_pairs": pairs,
        "broad_phase_pairs": broad_pairs,
        "intersection_pair_count": exact_contacts,
        "clearance_pair_count": violations,
        "required_clearance_m": required,
        "minimum_distance_m": minimum if np.isfinite(minimum) else None,
        "minimum_is_capped_at_required_clearance": minimum >= required,
        "finite_body_certified": False,
        "infeasibility_proven": False,
    }
