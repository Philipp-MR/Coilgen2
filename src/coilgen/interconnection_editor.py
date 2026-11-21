"""Directed, transactional edits of an existing automatic winding.

All coordinates are SI. Contours are immutable; only bounded openings and
explicitly owned connections can change. The directed sections form one circuit,
including deliberately disconnected draft edges. Rendering owns no geometry.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re

import numpy as np

from .models import CoilPart, WirePath, WirePathGeometry, WireSegmentOwner
from .routing_ports import ContourArc


PLAN_SCHEMA = "coilgen.interconnection-editor/2"
MAX_POINTS = 50000


def _points(values):
    points = np.array(values, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or not 2 <= len(points) <= MAX_POINTS:
        raise ValueError("A connection needs 2–50,000 points in metres.")
    if not np.all(np.isfinite(points)) or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12):
        raise ValueError("Connection points must be finite and consecutive points distinct.")
    return points


def route_fingerprint(part):
    """Bind edits to exact support, directed contours, provenance and current."""
    digest = sha256()

    def array(value):
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())

    array(part.mesh.vertices)
    array(part.mesh.faces)
    array(part.wire_path.v)
    for contour in part.contour_lines:
        array(contour.v)
        for name in (
            "segment_faces",
            "segment_barycentric",
            "surface_faces",
            "surface_barycentric",
            "surface_edge_vertices",
            "surface_edge_t",
        ):
            value = getattr(contour, name)
            digest.update(name.encode())
            if value is None:
                digest.update(b"none")
            else:
                array(value)
        digest.update(
            json.dumps(
                (float(contour.potential), float(contour.current_orientation), int(contour.turn_multiplicity)),
                allow_nan=False,
            ).encode()
        )
    digest.update(json.dumps(None if part.contour_step is None else float(part.contour_step), allow_nan=False).encode())
    digest.update(json.dumps([asdict(owner) for owner in part.wire_path.segment_owners], sort_keys=True).encode())
    digest.update(json.dumps([value.value for value in part.wire_path.segment_geometry]).encode())
    return digest.hexdigest()


@dataclass
class RouteSection:
    key: str
    points: np.ndarray
    owner: WireSegmentOwner
    geometry: tuple[WirePathGeometry, ...]
    editable: bool = False
    cut: tuple[float, float] | None = None
    locked: bool = False
    disconnected: bool = False
    guides: tuple[tuple[float, float, float], ...] = ()
    display_name: str = ""
    target: str | None = None
    bridge: dict | None = None
    cut_shape: dict | None = None
    cut_object: str | None = None

    @property
    def label(self):
        if self.display_name:
            return self.display_name
        if self.owner.kind == "contour_loop":
            return f"Turn {self.owner.loop_index + 1}" + (
                " · movable cut" if self.cut is not None else " · retained arc"
            )
        if not self.editable:
            return f"Group {self.owner.group_index + 1} · retained segment"
        if self.owner.kind == "within_group_return":
            return f"Return · group {self.owner.group_index + 1}"
        if self.owner.kind == "inter_group_connector":
            return f"Bridge · group {self.owner.group1_index + 1} → {self.owner.group2_index + 1}"
        return f"Connection · group {self.owner.group_index + 1}"


class InterconnectionEditor:
    """One owned circuit, exact import, bounded history and atomic operations."""

    def __init__(self, part: CoilPart):
        self._source = deepcopy(part)
        if getattr(part, "routing_editor_baseline", None) is not None:
            self._source.wire_path = deepcopy(part.routing_editor_baseline)
        path = self._source.wire_path
        if path is None or len(path.v) < 2 or len(path.segment_owners) != len(path.v) - 1:
            raise ValueError("Editing requires a routed checkpoint with complete segment ownership.")
        _points(path.v)
        self.fingerprint = route_fingerprint(self._source)
        self.closed = np.array_equal(path.v[0], path.v[-1])
        self._sections = self._split(path)
        self._baseline = deepcopy(self._sections)
        self._undo, self._redo = [], []
        self.revision = 0
        self._solver = self._projector = self._chart = None

    def _split(self, path):
        owners, geometry, points = list(path.segment_owners), list(path.segment_geometry), path.v.copy()
        # Put a closed circuit's seam on an ownership boundary. Retain the exact
        # original array for the no-edit round trip (wire_path below).
        boundaries = [i for i in range(1, len(owners)) if owners[i] != owners[i - 1] or geometry[i] != geometry[i - 1]]
        if self.closed and owners[0] == owners[-1] and geometry[0] == geometry[-1] and boundaries:
            k = boundaries[0]
            points = np.vstack((points[k:-1], points[: k + 1]))
            owners, geometry = owners[k:] + owners[:k], geometry[k:] + geometry[:k]
        stops = (
            [0]
            + [i for i in range(1, len(owners)) if owners[i] != owners[i - 1] or geometry[i] != geometry[i - 1]]
            + [len(owners)]
        )
        sections = []
        for index, (a, b) in enumerate(zip(stops[:-1], stops[1:])):
            owner = deepcopy(owners[a])
            editable = owner.kind in {"inter_group_connector", "within_group_return", "within_group_connector"}
            editable |= owner.kind == "contour_group" and owner.connector_role == "joint_spiral_join"
            section = RouteSection(f"s{index}", points[a : b + 1].copy(), owner, tuple(geometry[a:b]), editable)
            if owner.kind == "contour_loop" and 0 <= owner.loop_index < len(self._source.contour_lines):
                try:
                    arc = ContourArc(self._source.contour_lines[owner.loop_index])
                    positions = np.array([arc.position(p) for p in section.points])
                    forward = np.mod(np.diff(positions), arc.perimeter)
                    span = arc.perimeter - forward.sum()
                    # Partial or reversed provenance is retained but not editable.
                    if 1e-10 < span < arc.perimeter / 2 and np.allclose(
                        forward, np.linalg.norm(np.diff(section.points, axis=0), axis=1), rtol=1e-8, atol=1e-11
                    ):
                        section.cut = ((positions[-1] + span / 2) % arc.perimeter, float(span))
                except ValueError:
                    pass
            sections.append(section)
        counts = {}
        for section in sections:
            if section.owner.kind == "contour_loop":
                counts[section.owner.loop_index] = counts.get(section.owner.loop_index, 0) + 1
        for section in sections:
            if counts.get(section.owner.loop_index, 0) > 1:
                section.cut = None
        # Name directed connections by their neighboring turns. Geometry-split
        # returns can share those endpoints, so distinguish their ordered spans.
        names = []
        for i, section in enumerate(sections):
            name = section.label
            if section.editable:
                turns = []
                for direction in (-1, 1):
                    for distance in range(1, len(sections)):
                        neighbor = i + direction * distance
                        if not self.closed and not 0 <= neighbor < len(sections):
                            break
                        owner = sections[neighbor % len(sections)].owner
                        if owner.kind == "contour_loop":
                            turns.append(owner.loop_index + 1)
                            break
                if len(turns) == 2:
                    name += f" · turns {turns[0]} → {turns[1]}"
            names.append(name)
        seen = {}
        for section, name in zip(sections, names):
            seen[name] = seen.get(name, 0) + 1
            section.display_name = name + (f" · segment {seen[name]}" if names.count(name) > 1 else "")
        for index, section in enumerate(sections):
            section.target = (
                sections[(index + 1) % len(sections)].key if self.closed or index + 1 < len(sections) else None
            )
        return sections

    @property
    def chart(self):
        from .winding_surface import WindingSurfaceChart

        if self._chart is None:
            self._chart = WindingSurfaceChart(self._source.mesh)
        return self._chart

    @property
    def sections(self):
        return deepcopy(self._sections)

    @property
    def source(self):
        return deepcopy(self._source)

    @property
    def support(self):
        return self._source.mesh.copy()

    @property
    def automatic_path(self):
        return deepcopy(self._source.wire_path)

    @property
    def undo_label(self):
        return self._undo[-1][0] if self._undo else ""

    @property
    def redo_label(self):
        return self._redo[-1][0] if self._redo else ""

    def section(self, key):
        return deepcopy(self._sections[self._index(key)])

    def _index(self, key):
        for i, section in enumerate(self._sections):
            if section.key == key:
                return i
        raise ValueError("Unknown winding section.")

    def _change(self, label, operation):
        draft = deepcopy(self._sections)
        operation(draft)
        self._check_joins(draft)
        self._refresh_names(draft)
        if self._signature(draft) == self._signature(self._sections):
            return
        self._undo.append((label, self._sections))
        self._undo = self._undo[-100:]
        self._redo.clear()
        self._sections = draft
        self.revision += 1

    @staticmethod
    def _signature(sections):
        return tuple(
            (
                s.points.tobytes(),
                s.geometry,
                s.cut,
                s.locked,
                s.disconnected,
                s.guides,
                s.target,
                json.dumps(s.bridge, sort_keys=True),
                json.dumps(s.cut_shape, sort_keys=True),
                s.cut_object,
            )
            for s in sections
        )

    def undo(self):
        if self._undo:
            label, value = self._undo.pop()
            self._redo.append((label, self._sections))
            self._sections = value
            self.revision += 1

    def redo(self):
        if self._redo:
            label, value = self._redo.pop()
            self._undo.append((label, self._sections))
            self._sections = value
            self.revision += 1

    def _check_joins(self, sections):
        if sum(len(s.points) - 1 for s in sections) > MAX_POINTS:
            raise ValueError("Edited route exceeds the 50,000-segment limit.")
        by_key = {s.key: s for s in sections}
        occupied = set()
        for section in sections:
            if section.target is None:
                if self.closed or section.key != self._sections[-1].key:
                    raise ValueError("An internal connection cannot become an implicit feed terminal.")
                continue
            if section.target not in by_key:
                raise ValueError("Unknown connection endpoint.")
            if section.disconnected:
                continue
            if section.target in occupied:
                raise ValueError("That endpoint is occupied. Cut its incoming connection first.")
            occupied.add(section.target)
            if not np.array_equal(section.points[-1], by_key[section.target].points[0]):
                raise ValueError("Connection endpoints do not match their directed neighboring sections.")

    def _refresh_names(self, sections):
        by_key = {s.key: s for s in sections}
        for row, baseline in zip(sections, self._baseline):
            row.display_name = baseline.display_name
            if row.target == baseline.target:
                continue
            target, seen = row.target, set()
            while target is not None and target not in seen:
                seen.add(target)
                destination = by_key[target]
                if destination.owner.kind == "contour_loop":
                    # Only replace the destination turn, never group provenance.
                    row.display_name = re.sub(
                        r"( · turns \d+ → )\d+",
                        lambda m: m[1] + str(destination.owner.loop_index + 1),
                        baseline.display_name,
                    )
                    break
                target = destination.target

    @property
    def ports(self):
        occupied = {s.target for s in self._sections if not s.disconnected}
        allowed = {s.target for s in self._baseline if s.editable}
        result = []
        for section in self._sections:
            if section.disconnected and section.editable and not section.locked:
                result.append(
                    dict(key=section.key, role="out", point=section.points[0].tolist(), label="From " + section.label)
                )
            if section.key in allowed and section.key not in occupied:
                result.append(
                    dict(key=section.key, role="in", point=section.points[0].tolist(), label="Into " + section.label)
                )
        return result

    def reconnect_endpoints(self, source_key, target_key):
        section = self.section(source_key)
        available = {(p["key"], p["role"]) for p in self.ports}
        if (source_key, "out") not in available or (target_key, "in") not in available:
            raise ValueError("Choose a free outgoing endpoint and a free incoming endpoint.")
        if section.bridge:
            raise ValueError("Remove this connection's bridge before changing its endpoint.")
        target = self.section(target_key)

        def edit(rows):
            row = rows[self._index(source_key)]
            mode = self._mode(row)
            row.points = self._route(row.points[0], target.points[0], (), mode)
            row.geometry = (mode,) * (len(row.points) - 1)
            row.target, row.disconnected, row.guides = target_key, False, ()

        self._change("Reconnect endpoints", edit)

    def drawing_ports(self, key):
        """Free ports plus the selected connection's existing attachments."""
        ports = self.ports
        row = self.section(key)
        if row.editable and not row.locked and not row.bridge and not row.disconnected and row.target is not None:
            ports.extend(
                [
                    dict(key=key, role="out", point=row.points[0].tolist(), label="From " + row.label),
                    dict(key=row.target, role="in", point=row.points[-1].tolist(), label="To " + row.label),
                ]
            )
        return ports

    def draw_connection(self, source_key, target_key, stroke, *, flat=False):
        """Route through a drawn surface path, with one atomic undo operation."""
        row = self.section(source_key)
        if not row.editable or row.locked or row.bridge:
            raise ValueError("Choose an unlocked connection without a bridge.")
        if row.disconnected:
            available = {(p["key"], p["role"]) for p in self.ports}
            if (source_key, "out") not in available or (target_key, "in") not in available:
                raise ValueError("Finish on a free input. Cut its existing connection first.")
        elif target_key != row.target:
            raise ValueError("Cut connections before exchanging their endpoints.")
        values = np.asarray(stroke, float)
        if (
            values.ndim != 2
            or values.shape[1] != (2 if flat else 3)
            or not 2 <= len(values) <= 512
            or not np.isfinite(values).all()
        ):
            raise ValueError("Draw a path with 2 to 512 finite points.")
        mode = self._mode(row)
        if mode == WirePathGeometry.FREE_3D:
            raise ValueError("Use the 3D guide handles for a free-space connection.")
        values = (
            np.asarray([self.chart.pick(p) for p in values]) if flat else np.asarray([self.project(p) for p in values])
        )
        target = self.section(target_key)
        values[0], values[-1] = self.project(row.points[0]), self.project(target.points[0])
        # Simplify input jitter to 0.25 mm; this is not a clearance tolerance.
        keep, pending = {0, len(values) - 1}, [(0, len(values) - 1)]
        while pending:
            a, b = pending.pop()
            if b <= a + 1:
                continue
            v = values[b] - values[a]
            t = np.clip((values[a + 1 : b] - values[a]) @ v / max(float(v @ v), 1e-30), 0, 1)
            distance = np.linalg.norm(values[a + 1 : b] - values[a] - t[:, None] * v, axis=1)
            i = a + 1 + int(np.argmax(distance))
            if distance[i - a - 1] > 0.00025:
                keep.add(i)
                pending.extend([(a, i), (i, b)])
        guides = values[sorted(keep)[1:-1]]
        if len(guides) > 128:
            raise ValueError("This sketch is too detailed; draw a shorter, smoother path.")
        if mode == WirePathGeometry.OFFSET_SURFACE and len(guides):
            feet, normals = self._projector.project(np.vstack([row.points[0], target.points[0]]))
            heights = np.einsum("ij,ij->i", np.vstack([row.points[0], target.points[0]]) - feet, normals)
            full = values[sorted(keep)]
            station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))]
            fraction = station[1:-1] / max(station[-1], 1e-30)
            surface, normals = self._projector.project(guides)
            guides = surface + ((1 - fraction) * heights[0] + fraction * heights[1])[:, None] * normals

        def edit(rows):
            section = rows[self._index(source_key)]
            section.guides = tuple(map(tuple, guides))
            section.points = self._route(section.points[0], target.points[0], section.guides, mode)
            section.geometry = (mode,) * (len(section.points) - 1)
            section.target, section.disconnected = target_key, False

        self._change("Draw connection", edit)

    def drag_guide(self, key, index, point, *, flat=False):
        """Surface dragging changes the footprint, preserving an existing lift."""
        section = self.section(key)
        guides = list(section.guides) or [tuple(section.points[len(section.points) // 2])]
        if not 0 <= index < len(guides):
            raise ValueError("Unknown guide point.")
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("A guide needs three finite coordinates in metres.")
        mode = self._mode(section)
        # Offset-surface connections retain their attachment height in set_guides.
        # A flat sketch of a free-space guide retains that guide's normal depth.
        if flat and mode == WirePathGeometry.FREE_3D:
            self.project(point)
            feet, normals = self._projector.project(np.asarray([guides[index], point]))
            height = float((np.asarray(guides[index]) - feet[0]) @ normals[0])
            point = feet[1] + height * normals[1]
        guides[index] = tuple(point)
        self.set_guides(key, guides)

    def set_locked(self, key, locked):
        i = self._index(key)
        self._change(
            "Lock section" if locked else "Unlock section", lambda rows: setattr(rows[i], "locked", bool(locked))
        )

    def disconnect(self, key, disconnected=True):
        i = self._index(key)
        section = self._sections[i]
        if not section.editable or section.locked:
            raise ValueError("Unlock an owned connection before cutting or reconnecting it.")
        self._change(
            "Cut connection" if disconnected else "Reconnect",
            lambda rows: setattr(rows[i], "disconnected", bool(disconnected)),
        )

    def project(self, point):
        from .manufacturing import _SurfaceProjector

        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("A surface point needs three finite coordinates.")
        if self._projector is None:
            self._projector = _SurfaceProjector(self._source.mesh)
        return self._projector.project(point[None])[0][0]

    def nearest_cut_position(self, key, point):
        section = self.section(key)
        if section.cut is None:
            raise ValueError("This section does not contain a complete directed contour opening.")
        arc = ContourArc(self._source.contour_lines[section.owner.loop_index])
        fraction = np.clip(np.einsum("ij,ij->i", point - arc.points[:-1], arc.delta) / arc.lengths**2, 0, 1)
        nearest = arc.points[:-1] + fraction[:, None] * arc.delta
        index = int(np.argmin(np.linalg.norm(nearest - point, axis=1)))
        return float(arc.positions[index] + fraction[index] * arc.lengths[index])

    def cut_point(self, key):
        section = self.section(key)
        if section.cut is None:
            raise ValueError("No editable cut for this section.")
        return ContourArc(self._source.contour_lines[section.owner.loop_index]).point(section.cut[0])

    def cut_curve(self, key):
        """Detached original contour for view-only picking, without copying field arrays."""
        section = self.section(key)
        if section.cut is None:
            raise ValueError("No editable cut for this section.")
        return self._source.contour_lines[section.owner.loop_index].v.copy()

    def cut_at_position(self, key, position):
        section = self.section(key)
        return ContourArc(self._source.contour_lines[section.owner.loop_index]).point(position)

    def _route(self, start, end, guides, geometry):
        from .surface_distance import SurfaceGeodesicSolver

        if geometry != WirePathGeometry.FREE_3D and self._solver is None:
            self._solver = SurfaceGeodesicSolver(self._source.mesh, refine_routes=True)
        anchors = [start, *(np.asarray(p) for p in guides), end]
        sections = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            if np.linalg.norm(a - b) <= 1e-12:
                continue
            if geometry != WirePathGeometry.FREE_3D:
                projected_a, projected_b = (
                    (self.project(a), self.project(b)) if geometry == WirePathGeometry.OFFSET_SURFACE else (a, b)
                )
                result = self._solver.route_between_points(projected_a, projected_b)
                if not np.isfinite(result.distance) or len(result.path_v) < 2:
                    raise ValueError("No route exists on this support between the selected points.")
                path = result.path_v.copy()
                if geometry == WirePathGeometry.OFFSET_SURFACE:
                    projected, normals = self._projector.project(np.vstack((a, b)))
                    height = np.einsum("ij,ij->i", np.vstack((a, b)) - projected, normals)
                    station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
                    fraction = station / station[-1]
                    surface, guide_normals = self._projector.project(path)
                    path = surface + ((1 - fraction) * height[0] + fraction * height[1])[:, None] * guide_normals
                path[0], path[-1] = a, b
            else:
                path = np.vstack((a, b))
            sections.append(path if not sections else path[1:])
        points = np.vstack(sections)
        # Geodesic endpoint refinement can emit sub-picometre duplicate samples.
        # Keep exact attachments and the existing point-distinctness threshold.
        compact = [points[0]]
        for point in points[1:-1]:
            if np.linalg.norm(point - compact[-1]) > 1e-12:
                compact.append(point)
        while len(compact) > 1 and np.linalg.norm(points[-1] - compact[-1]) <= 1e-12:
            compact.pop()
        compact.append(points[-1])
        return _points(np.asarray(compact))

    @staticmethod
    def _mode(section):
        kinds = set(section.geometry)
        if kinds == {WirePathGeometry.SURFACE}:
            return WirePathGeometry.SURFACE
        if len(kinds) == 1:
            return next(iter(kinds))
        raise ValueError("This connection has mixed or offset geometry; its attachment contract must be preserved.")

    def set_guides(self, key, guides):
        i = self._index(key)
        section = self._sections[i]
        if not section.editable or section.locked or section.disconnected:
            raise ValueError("Select an unlocked, connected routing section.")
        if section.bridge:
            raise ValueError("Remove the bridge before reshaping its base connection.")
        mode = self._mode(section)
        values = np.asarray(guides, dtype=float).reshape(-1, 3)
        if len(values) > 128 or not np.all(np.isfinite(values)):
            raise ValueError("Use at most 128 finite guide points.")
        if mode == WirePathGeometry.SURFACE:
            values = np.asarray([self.project(p) for p in values]).reshape(-1, 3)
        elif mode == WirePathGeometry.OFFSET_SURFACE and len(values):
            middle = section.points[len(section.points) // 2]
            self.project(middle)
            origin, normal = self._projector.project(middle[None])
            height = float((middle - origin[0]) @ normal[0])
            surface, normals = self._projector.project(values)
            values = surface + height * normals

        def edit(rows):
            row = rows[i]
            row.guides = tuple(tuple(float(x) for x in p) for p in values)
            row.points = self._route(row.points[0], row.points[-1], row.guides, mode)
            row.geometry = (mode,) * (len(row.points) - 1)

        self._change("Guide connection", edit)

    def move_cut(self, key, center_m, span_m=None):
        self._change("Move contour cut", lambda rows: self._move_cuts(rows, [(key, center_m, span_m)]))

    def _move_cuts(self, rows, changes):
        if not self.connectivity["complete"]:
            raise ValueError("Complete the endpoint connections before moving cuts.")
        by_key = {s.key: s for s in rows}
        reroute = set()
        for key, position, width in changes:
            row = by_key[key]
            if row.cut is None or row.locked:
                raise ValueError("Select an unlocked contour with an editable opening.")
            incoming = [s for s in rows if s.target == key and not s.disconnected]
            outgoing = by_key.get(row.target)
            if len(incoming) != 1 or outgoing is None:
                raise ValueError("A terminal opening cannot be moved without an explicit feed route.")
            before, after = incoming[0], outgoing
            if before.key == after.key or any(not s.editable or s.locked for s in (before, after)):
                raise ValueError("Both neighboring connections must be owned and unlocked to move this cut.")
            if before.bridge or after.bridge:
                raise ValueError("Remove the neighboring bridge before moving its attachment cut.")
            arc = ContourArc(self._source.contour_lines[row.owner.loop_index])
            width = row.cut[1] if width is None else float(width)
            opened, _ = arc.open(float(position), width)
            row.points = opened.v.copy()
            row.geometry = (WirePathGeometry.SURFACE,) * (len(row.points) - 1)
            row.cut = (float(position) % arc.perimeter, width)
            row.cut_shape = None
            row.cut_object = None
            before.points[-1], after.points[0] = row.points[0], row.points[-1]
            reroute.update((before.key, after.key))
        for key in sorted(reroute):
            row = by_key[key]
            mode = self._mode(row)
            row.points = self._route(row.points[0], row.points[-1], row.guides, mode)
            row.geometry = (mode,) * (len(row.points) - 1)

    def preview_seam(self, stroke, keys=None, *, flat=False):
        from .winding_surface import seam_intersections

        stroke = np.asarray(stroke, dtype=float)
        dimension = 2 if flat else 3
        if (
            stroke.ndim != 2
            or stroke.shape[1] != dimension
            or not 2 <= len(stroke) <= 512
            or not np.isfinite(stroke).all()
        ):
            raise ValueError("A seam needs 2–512 finite points on the support.")
        chart = self.chart
        if not flat:
            developed = chart.project(stroke)
            if np.any(
                np.linalg.norm(np.diff(developed, axis=0), axis=1)
                > 3 * np.linalg.norm(np.diff(stroke, axis=0), axis=1) + 1e-6
            ):
                raise ValueError("The stroke crosses the flat chart boundary. Draw separate strokes on each side.")
            stroke = developed
        else:
            for point in stroke:
                chart.pick(point)
        selected = set(keys) if keys is not None else {s.key for s in self._sections if s.cut is not None}
        if not selected <= {s.key for s in self._sections}:
            raise ValueError("Unknown seam contour.")
        cuts = []
        for row in self._sections:
            if row.key not in selected or row.cut is None:
                continue
            hits = seam_intersections(chart, self._source.contour_lines[row.owner.loop_index], stroke)
            if len(hits) > 1:
                raise ValueError(f"The seam crosses {row.label} more than once. Draw a shorter stroke.")
            if hits:
                cuts.append(
                    dict(
                        key=row.key,
                        label=row.label,
                        position_m=hits[0],
                        span_m=row.cut[1],
                        point=self.cut_at_position(row.key, hits[0]).tolist(),
                    )
                )
        if not cuts:
            raise ValueError("The stroke did not cross a movable cut contour in the selected group.")
        # Locks and geometry are checked atomically when the preview is applied.
        return dict(cuts=cuts, stroke=stroke.tolist(), revision=self.revision)

    def default_cut_shape(self, kind, key):
        from .winding_cut_shapes import CutShape

        selected = self.section(key)
        rows = [s for s in self._sections if s.cut is not None]
        if not rows:
            raise ValueError("This circuit has no movable contour openings.")
        row = min(rows, key=lambda s: np.linalg.norm(self.cut_point(s.key) - selected.points[0]))
        if selected.cut is not None:
            row = selected
        if row.cut_shape and row.cut_shape["kind"] == kind:
            return deepcopy(row.cut_shape)
        center = self.cut_point(row.key)
        tangent = self.cut_at_position(row.key, row.cut[0] + 0.0001) - center
        extent = np.linalg.norm(np.ptp(self.support.vertices, axis=0))
        return CutShape.from_dict(
            dict(kind=kind, center=center, normal=tangent, size=[row.cut[1] / 2, extent * 0.15, extent * 0.15])
        ).to_dict()

    @property
    def cut_objects(self):
        """Persistent cutters, with deterministic grouping for pre-object drafts."""
        objects = {}
        for row in self._sections:
            if not row.cut_shape:
                continue
            identity = row.cut_object or sha256(json.dumps(row.cut_shape, sort_keys=True).encode()).hexdigest()[:16]
            obj = objects.setdefault(identity, dict(id=identity, shape=deepcopy(row.cut_shape), keys=[], locked=False))
            obj["keys"].append(row.key)
            obj["locked"] |= row.locked
        extent = float(np.linalg.norm(np.ptp(self.support.vertices, axis=0)))
        from .winding_cut_shapes import CutShape

        for index, obj in enumerate(objects.values(), 1):
            obj["label"] = f"{obj['shape']['kind'].capitalize()} {index}"
            obj["display"] = CutShape.from_dict(obj["shape"]).display(extent)
        return list(objects.values())

    def preview_cut_shape(self, value, keys=None, *, object_id=None, choices=None):
        from .winding_cut_shapes import CutShape

        shape = CutShape.from_dict(value)
        selected = set(keys) if keys is not None else {s.key for s in self._sections if s.cut is not None}
        if not selected <= {s.key for s in self._sections}:
            raise ValueError("Unknown cut-shape contour.")
        choices = dict(choices or {})
        if not set(choices) <= selected:
            raise ValueError("An opening choice must belong to the selected contours.")
        if object_id is not None and object_id not in {o["id"] for o in self.cut_objects}:
            raise ValueError("Unknown cut object.")
        cuts, alternatives, multiple = [], [], 0
        for row in self._sections:
            if row.key not in selected or row.cut is None:
                continue
            arc = ContourArc(self._source.contour_lines[row.owner.loop_index])
            intervals = shape.intervals(arc)
            if not intervals:
                continue
            multiple += int(len(intervals) > 1)

            def distance(interval):
                position = ((interval[0] + interval[1]) / 2) % arc.perimeter
                delta = abs(position - row.cut[0])
                return min(delta, arc.perimeter - delta)

            chosen = choices.get(row.key, intervals.index(min(intervals, key=distance)))
            if not isinstance(chosen, int) or not 0 <= chosen < len(intervals):
                raise ValueError("Unknown cut intersection. Place the shape again.")
            for index, (a, b) in enumerate(intervals):
                alternatives.append(
                    dict(
                        key=row.key,
                        index=index,
                        selected=index == chosen,
                        point=self.cut_at_position(row.key, (a + b) / 2 % arc.perimeter).tolist(),
                        opening=[
                            self.cut_at_position(row.key, s % arc.perimeter).tolist() for s in np.linspace(a, b, 25)
                        ],
                    )
                )
            start, end = intervals[chosen]
            position, width = ((start + end) / 2) % arc.perimeter, end - start
            arc.open(position, width)  # Reject empty/degenerate retained arcs before preview.
            cuts.append(
                dict(
                    key=row.key,
                    label=row.label,
                    position_m=position,
                    span_m=width,
                    point=self.cut_at_position(row.key, position).tolist(),
                    opening=[
                        self.cut_at_position(row.key, s % arc.perimeter).tolist() for s in np.linspace(start, end, 25)
                    ],
                )
            )
        return dict(
            shape=shape.to_dict(),
            object_id=object_id
            or sha256(
                json.dumps([shape.to_dict(), sorted(selected), self.revision], sort_keys=True).encode()
            ).hexdigest()[:16],
            keys=sorted(selected),
            choices=choices,
            alternatives=alternatives,
            cuts=cuts,
            multiple=multiple,
            revision=self.revision,
            display=shape.display(float(np.linalg.norm(np.ptp(self.support.vertices, axis=0)))),
        )

    def set_cut_object_locked(self, identity, locked):
        obj = next((o for o in self.cut_objects if o["id"] == identity), None)
        if obj is None:
            raise ValueError("Unknown cut object.")

        def edit(rows):
            for row in rows:
                if row.key in obj["keys"]:
                    row.locked = bool(locked)

        self._change("Lock cut object" if locked else "Unlock cut object", edit)

    def apply_seam(self, preview):
        if preview.get("revision") != self.revision:
            raise ValueError("The seam preview is stale. Draw it again on the current revision.")
        changes = [(c["key"], float(c["position_m"]), float(c["span_m"])) for c in preview["cuts"]]
        if not changes or len({c[0] for c in changes}) != len(changes):
            raise ValueError("A seam must contain distinct contour cuts.")

        def edit(rows):
            self._move_cuts(rows, changes)
            if "shape" in preview:
                from .winding_cut_shapes import CutShape

                shape = CutShape.from_dict(preview["shape"]).to_dict()
                identity = preview.get("object_id")
                old_keys = {key for obj in self.cut_objects if obj["id"] == identity for key in obj["keys"]}
                for row in rows:
                    if row.key in old_keys and row.key not in {c[0] for c in changes}:
                        if row.locked:
                            raise ValueError("Unlock the cut before changing its cut object membership.")
                        row.cut_shape = row.cut_object = None
                for key, _, _ in changes:
                    rows[self._index(key)].cut_shape = deepcopy(shape)
                    rows[self._index(key)].cut_object = identity

        self._change("Apply cut shape" if "shape" in preview else "Move seam", edit)

    def _preview_clone(self):
        clone = copy(self)
        clone._sections = deepcopy(self._sections)
        clone._undo, clone._redo = [], []
        return clone

    def complete_preview(self, preview, minimum_radius=0.0):
        """Route the exact proposed edit on a detached draft; never mutate history."""
        from .winding_review import quick_review

        result = deepcopy(preview)
        clone = self._preview_clone()
        try:
            clone.apply_preview(preview)
            result["routes"] = [
                dict(key=s.key, points=s.points.tolist(), guides=list(s.guides))
                for s, before in zip(clone._sections, self._sections)
                if s.editable and not np.array_equal(s.points, before.points)
            ]
            result["warnings"] = quick_review(clone, minimum_radius)
            result["error"] = ""
        except ValueError as error:
            result.update(routes=[], warnings=[], error=str(error))
        return result

    def preview_drawing(self, source, target, stroke, *, flat=False, minimum_radius=0.0):
        preview = dict(
            revision=self.revision,
            cuts=[],
            drawing=dict(source=source, target=target, stroke=np.asarray(stroke, dtype=float).tolist(), flat=flat),
        )
        return self.complete_preview(preview, minimum_radius)

    def apply_preview(self, preview):
        if preview.get("revision") != self.revision:
            raise ValueError("This preview is stale. Create it again on the current revision.")
        if preview.get("error"):
            raise ValueError(preview["error"])
        if "drawing" in preview:
            d = preview["drawing"]
            self.draw_connection(d["source"], d["target"], d["stroke"], flat=d["flat"])
        else:
            self.apply_seam(preview)

    def route_unlocked(self):
        def edit(rows):
            for row in rows:
                if row.editable and not row.locked and not row.disconnected:
                    if row.bridge:
                        continue  # An explicit over/under decision is retained.
                    mode = self._mode(row)
                    row.points = self._route(row.points[0], row.points[-1], row.guides, mode)
                    row.geometry = (mode,) * (len(row.points) - 1)

        self._change("Route unlocked connections", edit)

    def bridge_crossing(self, key, other_key, relation, separation_m, span_m, crossing_point=None):
        from .winding_bridges import crossing_stations, surface_crossing_stations, realize_bridge

        if relation not in {"over", "under"} or key == other_key:
            raise ValueError("Choose two different crossing sections and over or under.")
        raised_key, lower_key = (key, other_key) if relation == "over" else (other_key, key)
        raised, lower = self.section(raised_key), self.section(lower_key)
        if not raised.editable or raised.locked or raised.disconnected or lower.disconnected:
            raise ValueError(
                "The upper path must be an unlocked, connected routing section. Protected contours stay fixed."
            )
        if not np.isfinite([separation_m, span_m]).all() or separation_m <= 0 or span_m <= 0:
            raise ValueError("Bridge separation and ramp span must be positive distances.")
        base = np.asarray(raised.bridge["base_points_m"]) if raised.bridge else raised.points
        labels = raised.bridge["base_geometry"] if raised.bridge else [g.value for g in raised.geometry]
        if WirePathGeometry.FREE_3D.value in labels:
            raise ValueError("A surface overpass needs a surface or offset-surface connection.")
        try:
            chart = self.chart
        except ValueError:
            self.project(base[0])
            hits = surface_crossing_stations(self._projector, base, lower.points)
        else:
            hits = crossing_stations(chart, base, lower.points)
        if not hits:
            raise ValueError("These sections do not cross on the support. Choose the intersecting path.")
        if crossing_point is None:
            if len(hits) > 1:
                raise ValueError("These sections cross more than once. Click the intended crossing in a view.")
            selected = hits[0]
        else:
            point = np.asarray(crossing_point, float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("A crossing needs three finite physical coordinates.")
            selected = min(hits, key=lambda h: np.linalg.norm(h[2] - point))
        station, _, high_point, low_point = selected
        self.project(high_point)
        _, normal = self._projector.project(high_point[None])
        height = max(0.0, float(separation_m + (low_point - high_point) @ normal[0]))
        spec = dict(
            other=lower_key,
            center_m=float(station),
            height_m=height,
            span_m=float(span_m),
            separation_m=float(separation_m),
            base_points_m=base.tolist(),
            base_geometry=list(labels),
        )
        points, labels = realize_bridge(self._projector, spec)

        def edit(rows):
            row = rows[self._index(raised_key)]
            row.bridge, row.points, row.geometry = spec, _points(points), labels

        self._change("Bridge over crossing" if relation == "over" else "Bridge under crossing", edit)
        return raised_key

    def bridge_handle(self, key):
        spec = self.section(key).bridge
        if not spec:
            return None
        base = np.asarray(spec["base_points_m"])
        station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))]
        point = np.array([np.interp(spec["center_m"], station, base[:, i]) for i in range(3)])
        self.project(point)
        _, normals = self._projector.project(point[None])
        return dict(
            point=(point + spec["height_m"] * normals[0]).tolist(), base=point.tolist(), normal=normals[0].tolist()
        )

    def set_bridge_height(self, key, point):
        from .winding_bridges import realize_bridge

        row = self.section(key)
        if not row.bridge or row.locked:
            raise ValueError("Select an unlocked bridge height handle.")
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("A bridge handle needs a finite point.")
        handle = self.bridge_handle(key)
        spec = deepcopy(row.bridge)
        height = float((point - handle["base"]) @ handle["normal"])
        spec["separation_m"] += height - spec["height_m"]
        spec["height_m"] = height
        points, labels = realize_bridge(self._projector, spec)

        def edit(rows):
            item = rows[self._index(key)]
            item.bridge, item.points, item.geometry = spec, _points(points), labels

        self._change("Change bridge height", edit)

    def remove_bridge(self, key):
        row = self.section(key)
        if row.locked or not row.bridge:
            raise ValueError("Select an unlocked bridge to remove.")

        def edit(rows):
            section = rows[self._index(key)]
            section.points = _points(section.bridge["base_points_m"])
            section.geometry = tuple(WirePathGeometry(g) for g in section.bridge["base_geometry"])
            section.bridge = None

        self._change("Remove bridge", edit)

    @property
    def connectivity(self):
        gaps = sum(s.disconnected for s in self._sections)
        graph = {s.key: set() for s in self._sections}
        for row in self._sections:
            if not row.disconnected and row.target is not None:
                graph[row.key].add(row.target)
                graph[row.target].add(row.key)
        unseen, components = set(graph), 0
        while unseen:
            pending = [unseen.pop()]
            components += 1
            while pending:
                neighbors = graph[pending.pop()] & unseen
                unseen -= neighbors
                pending.extend(neighbors)
        return {
            "complete": gaps == 0 and components == 1,
            "components": components,
            "free_ends": 2 * gaps + (0 if self.closed else 2),
            "closed": self.closed,
        }

    def wire_path(self):
        if not self.connectivity["complete"]:
            raise ValueError(
                "Reconnect all cut connections into one directed circuit before validation or application."
            )
        if all(
            np.array_equal(a.points, b.points) and a.geometry == b.geometry and a.target == b.target
            for a, b in zip(self._sections, self._baseline)
        ):
            return deepcopy(self._source.wire_path)
        by_key = {s.key: s for s in self._sections}
        ordered, key = [], self._sections[0].key
        while key is not None and len(ordered) < len(self._sections):
            ordered.append(by_key[key])
            key = by_key[key].target
        points = np.vstack([s.points if i == 0 else s.points[1:] for i, s in enumerate(ordered)])
        owners = [deepcopy(s.owner) for s in ordered for _ in range(len(s.points) - 1)]
        return WirePath(points, owners, [g for s in ordered for g in s.geometry])

    def build_part(self):
        part = deepcopy(self._source)
        part.routing_editor_baseline = deepcopy(self._source.wire_path)
        part.wire_path = self.wire_path()
        for section in self._sections:
            if section.cut is not None:
                arc = ContourArc(part.contour_lines[section.owner.loop_index])
                opened, endpoints = arc.open(*section.cut)
                for group in part.groups:
                    if section.owner.loop_index in group.loop_indices:
                        local = group.loop_indices.index(section.owner.loop_index)
                        if local < len(group.opened_loops):
                            group.opened_loops[local] = opened
                        if local < len(group.opening_cuts_v):
                            group.opening_cuts_v[local] = endpoints
            owner = section.owner
            if owner.kind == "inter_group_connector" and 0 <= owner.connector_index < len(
                part.opening_cuts_among_groups
            ):
                cut = part.opening_cuts_among_groups[owner.connector_index]
                attr = "closing_connection_path_v" if owner.connector_role == "closing" else "connection_path_v"
                setattr(cut, attr, section.points.copy())
        # A revised path has no inherited field, electrical, body or support result.
        for name in (
            "fabrication_path",
            "swept_body",
            "support_body",
            "pcb_layout",
            "field_by_layout",
            "shift_array",
            "points_to_shift",
        ):
            setattr(part, name, None)
        part.swept_bodies = []
        part.wire_self_contacts = []
        for name in (
            "manufacturing_diagnostics",
            "route_diagnostics",
            "normal_shift_diagnostics",
            "normal_shift_provenance",
            "joint_route_diagnostics",
        ):
            setattr(part, name, {})
        part.coil_resistance = part.coil_inductance = part.coil_length = part.coil_cross_section = 0.0
        part.topology_diagnostics = deepcopy(part.topology_diagnostics)
        part.topology_diagnostics["interconnection_editor"] = {
            "source_fingerprint": self.fingerprint,
            "revision": self.revision,
            "validation": "pending",
        }
        return part

    def to_plan(self):
        rows = []
        for section, baseline in zip(self._sections, self._baseline):
            row = {"key": section.key, "locked": section.locked, "disconnected": section.disconnected}
            if section.target != baseline.target:
                row["target"] = section.target
            if section.bridge:
                row["bridge"] = deepcopy(section.bridge)
            if section.cut_shape:
                row["cut_shape"] = deepcopy(section.cut_shape)
                if section.cut_object:
                    row["cut_object"] = section.cut_object
            if section.cut != baseline.cut:
                row["cut_m"] = list(section.cut)
            if section.editable and (not np.array_equal(section.points, baseline.points) or section.guides):
                row.update(
                    points_m=section.points.tolist(),
                    geometry=[x.value for x in section.geometry],
                    guides_m=section.guides,
                )
            if len(row) > 3 or row["locked"] or row["disconnected"]:
                rows.append(row)
        return {"source_fingerprint": self.fingerprint, "sections": rows}

    def load_plan(self, plan):
        if not isinstance(plan, dict) or plan.get("source_fingerprint") != self.fingerprint:
            raise ValueError("Manual routing belongs to a different automatic route; reopen and reconcile the edits.")
        rows = plan.get("sections")
        if not isinstance(rows, list) or len(rows) > len(self._sections):
            raise ValueError("Invalid manual routing section list.")
        draft = deepcopy(self._baseline)
        seen = set()
        for value in rows:
            if not isinstance(value, dict):
                raise ValueError("Each edited section must be an object.")
            key = value.get("key")
            if key in seen:
                raise ValueError("Duplicate edited section.")
            seen.add(key)
            section = draft[self._index(key)]
            for attr in ("locked", "disconnected"):
                if not isinstance(value.get(attr, False), bool):
                    raise ValueError("Lock and connection states must be booleans.")
                setattr(section, attr, value.get(attr, False))
            if section.disconnected and not section.editable:
                raise ValueError("Only connections may be disconnected.")
            if "target" in value:
                if not section.editable or value["target"] not in {
                    s.target for s in self._baseline if s.editable and s.target is not None
                }:
                    raise ValueError("Only an owned connection may target a valid endpoint.")
                section.target = value["target"]
            if "cut_m" in value:
                if section.cut is None or len(value["cut_m"]) != 2:
                    raise ValueError("Invalid contour opening edit.")
                arc = ContourArc(self._source.contour_lines[section.owner.loop_index])
                opened, _ = arc.open(*value["cut_m"])
                section.points = opened.v
                section.geometry = (WirePathGeometry.SURFACE,) * (len(opened.v) - 1)
                section.cut = tuple(value["cut_m"])
            if "cut_shape" in value:
                from .winding_cut_shapes import CutShape

                if section.cut is None:
                    raise ValueError("Only a movable contour opening may have a cut shape.")
                shape = CutShape.from_dict(value["cut_shape"])
                arc = ContourArc(self._source.contour_lines[section.owner.loop_index])
                position, width = section.cut
                if not any(
                    abs(end - start - width) < 1e-9
                    and abs(((start + end) / 2 - position + arc.perimeter / 2) % arc.perimeter - arc.perimeter / 2)
                    < 1e-9
                    for start, end in shape.intervals(arc)
                ):
                    raise ValueError("Saved cut shape does not match its contour opening.")
                section.cut_shape = shape.to_dict()
                identity = value.get("cut_object")
                if identity is not None and (
                    not isinstance(identity, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", identity)
                ):
                    raise ValueError("Invalid cut object identity.")
                section.cut_object = identity
            if "points_m" in value:
                if not section.editable:
                    raise ValueError("Source contour vertices cannot be edited.")
                mode = self._mode(section)
                section.points = _points(value["points_m"])
                geometry = tuple(WirePathGeometry(v) for v in value["geometry"])
                if len(geometry) != len(section.points) - 1 or (not value.get("bridge") and set(geometry) != {mode}):
                    raise ValueError("A saved connection cannot change its geometric contract.")
                section.geometry = geometry
                guides = np.asarray(value.get("guides_m", []), dtype=float).reshape(-1, 3)
                if len(guides) > 128 or not np.all(np.isfinite(guides)):
                    raise ValueError("Invalid saved guide points.")
                section.guides = tuple(tuple(float(x) for x in p) for p in guides)
                if value.get("bridge"):
                    from .winding_bridges import realize_bridge

                    spec = deepcopy(value["bridge"])
                    if spec.get("other") not in {s.key for s in draft} or spec["other"] == section.key:
                        raise ValueError("A bridge needs a different, known crossing section.")
                    _points(spec["base_points_m"])
                    if set(spec["base_geometry"]) != {mode.value}:
                        raise ValueError("A bridge cannot change its base attachment contract.")
                    self.project(section.points[0])
                    realized, labels = realize_bridge(self._projector, spec)
                    if not np.array_equal(realized, section.points) or labels != geometry:
                        raise ValueError("Saved bridge geometry does not match its ramp definition.")
                    section.bridge = spec
            elif value.get("bridge"):
                raise ValueError("Bridge geometry is missing.")
        objects = {}
        for section in draft:
            if section.cut_object:
                shape = json.dumps(section.cut_shape, sort_keys=True)
                if objects.setdefault(section.cut_object, shape) != shape:
                    raise ValueError("A cut object must have one shared shape definition.")
        self._check_joins(draft)
        self._refresh_names(draft)
        self._sections = draft
        self._undo.clear()
        self._redo.clear()
        self.revision += 1


def validate_editor_plan(plan):
    if plan is None:
        return
    if (
        not isinstance(plan, dict)
        or plan.get("schema") not in {PLAN_SCHEMA, "coilgen.interconnection-editor/1"}
        or not isinstance(plan.get("circuits"), list)
    ):
        raise ValueError("Invalid interconnection editor plan.")
    if not 1 <= len(plan["circuits"]) <= 256:
        raise ValueError("An editor plan needs 1–256 circuit entries.")
    for circuit in plan["circuits"]:
        if (
            not isinstance(circuit, dict)
            or not isinstance(circuit.get("source_fingerprint"), str)
            or not isinstance(circuit.get("sections"), list)
        ):
            raise ValueError("An editor circuit needs source identity and section edits.")


def apply_editor_plan(parts, plan):
    """Apply after automatic routing, before final manufacturing and field checks."""
    validate_editor_plan(plan)
    if plan is None:
        return parts
    if len(parts) != len(plan["circuits"]):
        raise ValueError("The automatic circuit count changed; reconcile manual routing.")
    result = []
    for part, settings in zip(parts, plan["circuits"]):
        editor = InterconnectionEditor(part)
        editor.load_plan(settings)
        result.append(editor.build_part())
    return result
