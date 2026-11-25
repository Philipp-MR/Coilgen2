"""Physical PCB artwork, development and checks, independent of either GUI.

Coordinates are metres. A seam is a real board cut; periodic copies are only
used to clip artwork, never to establish electrical continuity across it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import pi

import numpy as np


@dataclass(frozen=True)
class PcbSettings:
    mode: str = "wire"
    seam_deg: float = 0.0
    width_m: float = 0.001
    clearance_m: float = 0.0003
    edge_m: float = 0.001
    copper_m: float = 70e-6
    layer_gap_m: float = 0.0001

    def __post_init__(self):
        if self.mode not in {"wire", "spirals"}:
            raise ValueError("Choose the saved wire or two-layer group spirals.")
        for name in ("seam_deg", "width_m", "clearance_m", "edge_m", "copper_m", "layer_gap_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0 <= self.seam_deg < 360:
            raise ValueError("Seam angle must be in [0, 360) degrees.")
        if not 1e-6 <= self.width_m <= 0.1 or not 1e-6 <= self.copper_m <= 0.01:
            raise ValueError("Trace width and copper thickness must be positive physical dimensions.")
        if not 0 <= self.clearance_m <= 0.1 or not 0 <= self.edge_m <= 0.1 or not 0 <= self.layer_gap_m <= 0.1:
            raise ValueError("Clearances and layer separation must lie between 0 and 100 mm.")


def _axis_sign(axis):
    return axis * (1 if axis[np.argmax(abs(axis))] >= 0 else -1)


@dataclass
class BoardFrame:
    kind: str
    origin: np.ndarray
    axes: np.ndarray
    radius: float
    lower: np.ndarray
    size: np.ndarray
    seam: float = 0.0

    @classmethod
    def fit(cls, mesh, seam_deg=0):
        v = np.asarray(mesh.vertices, float)
        if len(v) < 3 or not np.isfinite(v).all():
            raise ValueError("The support needs a finite triangulated surface.")
        center = v.mean(axis=0)
        _, singular, axes = np.linalg.svd(v - center, full_matrices=False)
        scale = max(float(singular[0]), 1e-12)
        if singular[-1] <= scale * 1e-8:
            normal = _axis_sign(axes[-1])
            seed = np.eye(3)[np.argmin(abs(normal))]
            x = seed - normal * np.dot(seed, normal)
            x /= np.linalg.norm(x)
            axes = np.array([x, np.cross(normal, x), normal])
            uv = (v - center) @ axes[:2].T
            lower = uv.min(axis=0)
            return cls("plane", center, axes, 0.0, lower, np.ptp(uv, axis=0))
        triangles = v[mesh.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals[lengths > 1e-15] / lengths[lengths > 1e-15, None]
        if len(normals) < 3:
            raise ValueError("This nonplanar support has too few valid faces to develop without distortion.")
        _, sv, na = np.linalg.svd(normals, full_matrices=False)
        if sv[1] < 1e-8 * sv[0] or sv[2] > 1e-5 * sv[1]:
            raise ValueError(
                "PCB development supports planes and circular cylinders. This surface cannot be flattened without distortion."
            )
        z = _axis_sign(na[-1])
        seed = np.eye(3)[np.argmin(abs(z))]
        x = seed - z * np.dot(seed, z)
        x /= np.linalg.norm(x)
        axes = np.array([x, np.cross(z, x), z])
        local = (v - center) @ axes.T
        circle, _, rank, _ = np.linalg.lstsq(
            np.column_stack((2 * local[:, :2], np.ones(len(v)))), np.sum(local[:, :2] ** 2, axis=1), rcond=None
        )
        if rank != 3:
            raise ValueError("The cylindrical support does not define a stable radius.")
        origin = center + circle[:2] @ axes[:2]
        local = (v - origin) @ axes.T
        radii = np.linalg.norm(local[:, :2], axis=1)
        radius = float(np.mean(radii))
        if radius <= 1e-9 or np.max(abs(radii - radius)) > max(1e-7, radius * 1e-5):
            raise ValueError("The support is not a constant-radius cylinder.")
        return cls(
            "cylinder",
            origin,
            axes,
            radius,
            np.array([0, local[:, 2].min()]),
            np.array([2 * pi * radius, np.ptp(local[:, 2])]),
            np.radians(seam_deg),
        )

    def project(self, points, *, continuous=True):
        local = (np.asarray(points, float) - self.origin) @ self.axes.T
        if self.kind == "plane":
            return local[:, :2] - self.lower
        theta = np.mod(np.arctan2(local[:, 1], local[:, 0]) - self.seam, 2 * pi)
        if continuous:
            theta = np.unwrap(theta)
        return np.column_stack((self.radius * theta, local[:, 2] - self.lower[1]))

    def wrap(self, uv, layer="top", separation=0.0):
        uv = np.asarray(uv, float)
        offset = separation if layer == "top" else 0.0
        if self.kind == "plane":
            local = np.column_stack((uv + self.lower, np.full(len(uv), offset)))
        else:
            theta = uv[:, 0] / self.radius + self.seam
            local = np.column_stack(
                (
                    (self.radius + offset) * np.cos(theta),
                    (self.radius + offset) * np.sin(theta),
                    uv[:, 1] + self.lower[1],
                )
            )
        return self.origin + local @ self.axes

    def distance(self, points):
        local = (np.asarray(points) - self.origin) @ self.axes.T
        return abs(local[:, 2]) if self.kind == "plane" else abs(np.linalg.norm(local[:, :2], axis=1) - self.radius)

    def path_points(self, uv, layer="top", separation=0.0):
        """Resolve wrapped straight artwork to at most 1 µm cylindrical chord error."""
        if self.kind == "plane":
            return self.wrap(uv, layer, separation)
        max_step = min(0.02, np.sqrt(8e-6 / (self.radius + separation))) * self.radius
        points = [uv[0]]
        for a, b in zip(uv[:-1], uv[1:]):
            count = max(1, int(np.ceil(abs(b[0] - a[0]) / max_step)))
            points.extend(a + np.arange(1, count + 1)[:, None] / count * (b - a))
        return self.wrap(points, layer, separation)


@dataclass
class PcbTrace:
    key: str
    label: str
    layer: str
    uv: np.ndarray
    source: np.ndarray | None = None


@dataclass
class PcbIssue:
    code: str
    text: str
    trace: str = ""
    uv: list = field(default_factory=list)
    severity: str = "error"


def polygon_rings(geometry):
    """Exterior followed by holes per polygon; retain holes in preview and export."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [[np.asarray(geometry.exterior.coords), *[np.asarray(r.coords) for r in geometry.interiors]]]
    return [rings for g in getattr(geometry, "geoms", ()) for rings in polygon_rings(g)]


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [np.asarray(geometry.coords)]
    return [p for g in getattr(geometry, "geoms", ()) for p in line_parts(g)]


def _periodic(geometry, frame):
    from shapely.affinity import translate

    if frame.kind == "plane":
        return [geometry]
    w = frame.size[0]
    lo, _, hi, _ = geometry.bounds
    # Only translations which intersect the physical board, including copper at a seam.
    return [translate(geometry, xoff=k * w) for k in range(int(np.ceil(-hi / w)), int(np.floor((w - lo) / w)) + 1)]


def _board(mesh, frame):
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    face_shapes = []
    for face in mesh.faces:
        triangle = Polygon(frame.project(mesh.vertices[face]))
        if triangle.is_valid and triangle.area > 1e-18:
            face_shapes.extend(_periodic(triangle, frame))
    board = unary_union(face_shapes).intersection(box(0, 0, *frame.size))
    # Remove sub-nanometre triangulation cracks without changing engineering dimensions.
    board = board.buffer(1e-10, join_style=2).buffer(-1e-10, join_style=2)
    if board.is_empty or not board.is_valid:
        raise ValueError("Could not construct a valid flat board from the support triangles.")
    return board


class PcbDesign:
    """One detached PCB candidate. Rebuild after edits; export this exact geometry."""

    def __init__(self, part, settings, edits=None):
        from shapely.geometry import LineString, Point, box
        from shapely.ops import unary_union

        self.settings = settings
        self.frame = BoardFrame.fit(part.mesh, settings.seam_deg)
        self.board = _board(part.mesh, self.frame)
        self.traces, self.issues, self.vias = [], [], []
        self.polygons, self.flat_lines = {}, {}
        self.terminals = []
        self.path = None
        edits = edits or {}
        layers = edits.get("layers", {})
        if settings.mode == "wire":
            if part.wire_path is None or len(part.wire_path.v) < 2:
                raise ValueError("This part has no saved routed wire. Choose group spirals or finish routing.")
            sources = [("wire", "Saved winding", "top", part.wire_path.v)]
        else:
            sources = [
                (f"g{i + 1}-{side}", f"Group {i + 1} · {side}", layer, getattr(group, f"spiral_{side}_v"))
                for i, group in enumerate(part.connected_groups)
                for side, layer in (("in", "top"), ("out", "bottom"))
            ]
            if not sources:
                raise ValueError("This run has no saved group spirals. Use the saved winding.")
        for key, label, layer, points in sources:
            points = np.asarray(points, float)
            if len(points) < 2 or not np.isfinite(points).all():
                raise ValueError(f"{label} has no valid path.")
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
            points = points[keep]
            if len(points) < 2:
                raise ValueError(f"{label} has zero length.")
            layer = layers.get(key, layer)
            if layer not in {"top", "bottom"}:
                raise ValueError("Copper layers must be top or bottom.")
            self.traces.append(PcbTrace(key, label, layer, self.frame.project(points), points.copy()))
            distances = self.frame.distance(points)
            deviation = float(distances.max())
            if deviation > 5e-5:
                witness = self.frame.project(points[[int(distances.argmax())]])[0]
                if self.frame.kind == "cylinder":
                    witness[0] %= self.frame.size[0]
                self.issues.append(
                    PcbIssue(
                        "off_surface",
                        f"{label}: flattening moves the saved path by up to {deviation * 1000:.3f} mm. Review lifted connections.",
                        key,
                        witness.tolist(),
                    )
                )
        self.endpoints = {}
        for trace in self.traces:
            for side, index in (("start", 0), ("end", -1)):
                p = trace.uv[index].copy()
                if self.frame.kind == "cylinder":
                    p[0] %= self.frame.size[0]
                self.endpoints[f"{trace.key}:{side}"] = dict(
                    trace=trace.key, layer=trace.layer, uv=p, label=f"{trace.label} · {side}"
                )
        for index, link in enumerate(edits.get("connections", [])):
            a, b = (self.endpoints.get(link.get(k)) for k in ("a", "b"))
            if a is None or b is None or link.get("a") == link.get("b"):
                raise ValueError("Choose two different saved endpoints to connect.")
            layer = link.get("layer", "top")
            if layer not in {"top", "bottom"}:
                raise ValueError("Connection layer must be top or bottom.")
            uv = np.array([a["uv"], b["uv"]])
            if np.linalg.norm(uv[1] - uv[0]) > 1e-10:
                self.traces.append(PcbTrace(f"link{index + 1}", f"Connection {index + 1}", layer, uv))
            for endpoint in (a, b):
                if endpoint["layer"] != layer:
                    self.vias.append(endpoint["uv"].copy())
        terminal_keys = edits.get("terminals", ["wire:start", "wire:end"] if settings.mode == "wire" else [])
        if len(terminal_keys) != len(set(terminal_keys)):
            raise ValueError("Terminal assignments must be distinct.")
        for key in terminal_keys:
            if key not in self.endpoints:
                raise ValueError("A terminal no longer belongs to this winding layout.")
            self.terminals.append(dict(key=key, **self.endpoints[key]))
        page = box(0, 0, *self.frame.size)
        for trace in self.traces:
            line = LineString(trace.uv)
            if not line.is_simple:
                self.issues.append(PcbIssue("self_crossing", f"{trace.label}: centerline crosses itself.", trace.key))
            copies = _periodic(line, self.frame)
            self.flat_lines[trace.key] = [p for copy in copies for p in line_parts(copy.intersection(page))]
            raw_copper = unary_union([copy.buffer(settings.width_m / 2, quad_segs=12) for copy in copies])
            # A cylindrical seam clips periodic copies in X only. Keep excess Y
            # (and all planar excess) for checks before trimming export artwork.
            checked_copper = raw_copper
            if self.frame.kind == "cylinder" and not raw_copper.is_empty:
                _, low, _, high = raw_copper.bounds
                checked_copper = raw_copper.intersection(box(0, low - 1, self.frame.size[0], high + 1))
            polygons = raw_copper.intersection(page)
            self.polygons[trace.key] = polygons
            if not polygons.is_valid:
                self.issues.append(PcbIssue("polygon", f"{trace.label}: invalid copper polygon.", trace.key))
            if not self.board.buffer(1e-9).covers(checked_copper):
                witness = checked_copper.difference(self.board).representative_point()
                self.issues.append(
                    PcbIssue(
                        "outside",
                        f"{trace.label}: copper leaves the board or crosses a support opening.",
                        trace.key,
                        list(witness.coords[0]),
                    )
                )
            if not self.board.buffer(-settings.edge_m).buffer(1e-9).covers(checked_copper):
                witness = checked_copper.difference(self.board.buffer(-settings.edge_m)).representative_point()
                self.issues.append(
                    PcbIssue(
                        "edge",
                        f"{trace.label}: copper is within {settings.edge_m * 1000:g} mm of a board edge.",
                        trace.key,
                        list(witness.coords[0]),
                    )
                )
            if self.frame.kind == "cylinder":
                w = self.frame.size[0]
                for a, b in zip(trace.uv[:-1], trace.uv[1:]):
                    if abs(b[0] - a[0]) < 1e-14:
                        continue
                    lo, hi = sorted((a[0], b[0]))
                    for k in range(int(np.floor(lo / w)) + 1, int(np.floor(hi / w)) + 1):
                        t = (k * w - a[0]) / (b[0] - a[0])
                        pos = a + t * (b - a)
                        self.issues.append(
                            PcbIssue(
                                "seam",
                                f"{trace.label}: trace cut at the opening seam; move the seam or explicitly redesign the connection.",
                                trace.key,
                                [0.0, float(pos[1])],
                            )
                        )
        self._clearances()
        if len({t.layer for t in self.traces}) > 1 and settings.layer_gap_m <= 0:
            self.issues.append(PcbIssue("layer_stack", "Two copper layers require a positive layer separation."))
        self._circuit()
        if self.vias:
            self.issues.append(
                PcbIssue(
                    "via_spec",
                    "Layer transitions have landing copper; drill sizes, plating and pad dimensions still need definition in PCB CAD.",
                    severity="warning",
                )
            )
        self.copper = {}
        for layer in ("top", "bottom"):
            copper = [self.polygons[t.key] for t in self.traces if t.layer == layer]
            copper.extend(Point(p).buffer(settings.width_m / 2, quad_segs=12) for p in self.vias)
            self.copper[layer] = unary_union(copper).intersection(page)

    def _clearances(self):
        from shapely.geometry import LineString, Point
        from shapely.ops import nearest_points, unary_union
        from shapely.strtree import STRtree

        threshold = self.settings.width_m + self.settings.clearance_m
        for via in self.vias:
            for trace in self.traces:
                if any(np.linalg.norm(via - p) < 1e-8 for p in (trace.uv[0], trace.uv[-1])):
                    continue
                if any(Point(via).distance(LineString(p)) < threshold - 1e-10 for p in self.flat_lines[trace.key]):
                    self.issues.append(
                        PcbIssue(
                            "via_clearance", f"Layer transition is too close to {trace.label}.", trace.key, via.tolist()
                        )
                    )
        for ti, trace in enumerate(self.traces):
            for other in self.traces[ti + 1 :]:
                if trace.layer != other.layer:
                    continue
                a = unary_union([LineString(p) for p in self.flat_lines[trace.key]])
                b = unary_union([LineString(p) for p in self.flat_lines[other.key]])
                # Exempt only the local neighbourhood of a deliberate end-to-end join.
                joints = [
                    p
                    for p in (trace.uv[0], trace.uv[-1])
                    for q in (other.uv[0], other.uv[-1])
                    if np.linalg.norm(p - q) < 1e-8
                ]
                for p in joints:
                    disk = Point(p).buffer(threshold * 1.5)
                    a, b = a.difference(disk), b.difference(disk)
                if not a.is_empty and not b.is_empty and a.distance(b) < threshold - 1e-10:
                    p, _ = nearest_points(a, b)
                    self.issues.append(
                        PcbIssue(
                            "clearance",
                            f"{trace.label} / {other.label}: same-layer copper gap is below {self.settings.clearance_m * 1000:g} mm.",
                            trace.key,
                            list(p.coords[0]),
                        )
                    )
            # Nonlocal intervals of the same winding also require separation.
            for points in self.flat_lines[trace.key]:
                source_line = LineString(points)
                if source_line.length / (threshold / 2) + len(points) > 200000:
                    self.issues.append(
                        PcbIssue(
                            "check_limit",
                            "Turn-clearance check exceeds 200,000 intervals; increase the physical width/gap or divide the winding for review.",
                            trace.key,
                        )
                    )
                    continue
                # Subdivide long straight intervals so excluding the local bend
                # neighbourhood never exempts an entire parallel return leg.
                points = np.asarray(source_line.segmentize(threshold / 2).coords)
                segments = [
                    LineString([a, b]) for a, b in zip(points[:-1], points[1:]) if np.linalg.norm(b - a) > 1e-12
                ]
                if not segments:
                    continue
                lengths = np.array([s.length for s in segments])
                along = np.r_[0, np.cumsum(lengths)]
                tree = STRtree(segments)
                closed = np.linalg.norm(points[0] - points[-1]) < 1e-8
                found = False
                for i, segment in enumerate(segments):
                    for j in tree.query(segment.buffer(threshold)):
                        j = int(j)
                        if j <= i + 1:
                            continue
                        gap = along[j] - along[i + 1]
                        if closed:
                            gap = min(gap, along[-1] - along[j + 1] + along[i])
                        if gap < threshold * 1.5:
                            continue
                        if segment.distance(segments[j]) < threshold - 1e-10:
                            p, _ = nearest_points(segment, segments[j])
                            self.issues.append(
                                PcbIssue(
                                    "turn_clearance",
                                    f"{trace.label}: nonadjacent turns are closer than the copper width plus required gap.",
                                    trace.key,
                                    list(p.coords[0]),
                                )
                            )
                            found = True
                            break
                    if found:
                        break

    def _circuit(self):
        # Endpoint graph is built in board space, never modulo circumference.
        nodes, edges = [], []

        def node(p, layer):
            for i, (q, side) in enumerate(nodes):
                if side == layer and np.linalg.norm(p - q) < 1e-8:
                    return i
            nodes.append((np.asarray(p), layer))
            return len(nodes) - 1

        for trace in self.traces:
            pieces = self.flat_lines[trace.key]
            for p in pieces:
                edges.append(
                    (
                        node(p[0], trace.layer),
                        node(p[-1], trace.layer),
                        self.frame.path_points(p, trace.layer, self.settings.layer_gap_m),
                    )
                )
        for p in self.vias:
            a, b = node(p, "top"), node(p, "bottom")
            if a != b and not any({x, y} == {a, b} for x, y, _ in edges):
                edges.append(
                    (
                        a,
                        b,
                        np.vstack(
                            (
                                self.frame.wrap([p], "top", self.settings.layer_gap_m),
                                self.frame.wrap([p], "bottom", self.settings.layer_gap_m),
                            )
                        ),
                    )
                )
        adjacency = [[] for _ in nodes]
        for i, (a, b, _) in enumerate(edges):
            adjacency[a].append(i)
            adjacency[b].append(i)
        free = [i for i, v in enumerate(adjacency) if len(v) == 1]
        bad = [i for i, v in enumerate(adjacency) if len(v) > 2]
        if bad:
            self.issues.append(
                PcbIssue(
                    "branch",
                    "The endpoint graph branches; a series winding must not split into parallel paths.",
                    uv=nodes[bad[0]][0].tolist(),
                )
            )
        if len(self.terminals) != 2 or len(free) != 2:
            self.issues.append(
                PcbIssue(
                    "terminals",
                    f"Define one continuous circuit with two terminals. Found {len(free)} open ends and {len(self.terminals)} assigned terminals.",
                )
            )
            return
        ends = []
        for terminal in self.terminals:
            matches = [
                i
                for i in free
                if nodes[i][1] == terminal["layer"] and np.linalg.norm(nodes[i][0] - terminal["uv"]) < 1e-8
            ]
            if not matches:
                self.issues.append(
                    PcbIssue("terminal_location", "Assigned terminals must be the two free ends of the circuit.")
                )
                return
            ends.append(matches[0])
        if len(set(ends)) != 2 or bad:
            return
        used = set()
        current = ends[0]
        path = []
        while True:
            choices = [i for i in adjacency[current] if i not in used]
            if not choices:
                break
            index = choices[0]
            used.add(index)
            a, b, points = edges[index]
            if current == a:
                current = b
            else:
                current = a
                points = points[::-1]
            path.extend(points if not path else points[1:])
        if len(used) != len(edges) or current != ends[1]:
            self.issues.append(
                PcbIssue("disconnected", "The artwork contains disconnected paths or a closed isolated loop.")
            )
        elif not any(
            i.code
            in {
                "seam",
                "outside",
                "self_crossing",
                "clearance",
                "turn_clearance",
                "via_clearance",
                "layer_stack",
                "check_limit",
            }
            for i in self.issues
        ):
            self.path = np.array(path)

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == "error"]

    def scene(self):
        def rings(g):
            return [[r.tolist() for r in group] for group in polygon_rings(g)]

        def comparison(trace):
            if trace.source is None:
                return None
            proposed = self.frame.wrap(trace.uv, trace.layer, self.settings.layer_gap_m)
            distances = np.linalg.norm(proposed - trace.source, axis=1)
            index = int(distances.argmax())
            uv = self.frame.project(trace.source[[index]], continuous=False)[0]
            return dict(
                distance_mm=float(distances[index]) * 1000,
                uv=uv.tolist(),
                saved=trace.source[index].tolist(),
                proposed=proposed[index].tolist(),
            )

        grid = []
        if self.frame.kind == "cylinder":
            for v in (0, self.frame.size[1]):
                grid.append(
                    self.frame.wrap(np.column_stack((np.linspace(0, self.frame.size[0], 97), np.full(97, v)))).tolist()
                )
            for u in np.linspace(0, self.frame.size[0], 9):
                grid.append(self.frame.wrap([[u, 0], [u, self.frame.size[1]]]).tolist())
        else:
            grid = [self.frame.wrap(r).tolist() for group in polygon_rings(self.board) for r in group]
        return dict(
            kind=self.frame.kind,
            seam_line=self.frame.wrap([[0, 0], [0, self.frame.size[1]]]).tolist()
            if self.frame.kind == "cylinder"
            else [],
            size_m=self.frame.size.tolist(),
            settings=asdict(self.settings),
            board=rings(self.board),
            support=grid,
            traces=[
                dict(
                    key=t.key,
                    label=t.label,
                    layer=t.layer,
                    original=t.source.tolist() if t.source is not None else [],
                    comparison=comparison(t),
                    flat=[p.tolist() for p in self.flat_lines[t.key]],
                    polygons=rings(self.polygons[t.key]),
                    wrapped=[
                        self.frame.path_points(p, t.layer, self.settings.layer_gap_m).tolist()
                        for p in self.flat_lines[t.key]
                    ],
                )
                for t in self.traces
            ],
            endpoints=[
                dict(
                    key=k,
                    label=v["label"],
                    trace=v["trace"],
                    layer=v["layer"],
                    uv=v["uv"].tolist(),
                    xyz=self.frame.wrap([v["uv"]], v["layer"], self.settings.layer_gap_m)[0].tolist(),
                )
                for k, v in self.endpoints.items()
            ],
            vias=[
                dict(uv=p.tolist(), xyz=self.frame.wrap([p], "top", self.settings.layer_gap_m)[0].tolist())
                for p in self.vias
            ],
            terminals=[v["key"] for v in self.terminals],
            issues=[dict(**asdict(i), xyz=self.frame.wrap([i.uv])[0].tolist() if i.uv else []) for i in self.issues],
            circuit_complete=self.path is not None,
            errors=len(self.errors),
        )
