"""Immutable, geometry-bound inspection of saved winding interconnections.

Developed views are projections of recorded samples, never collision tests.
Only authenticated saved body diagnostics provide engineering witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import re

import numpy as np

from ..group_route_convex_shell import convex_circular_shell
from ..models import Mesh
from .diagnostics import ProblemIssue, run_problems, run_verdicts


COLORS = {
    "contour": "#146b80",
    "join": "#e48a22",
    "return": "#8b5bb5",
    "connector": "#d82b47",
    "unclassified_added": "#667587",
    "unknown": "#a0a8ad",
}
ROLE_LABELS = {"unclassified_added": "Added span (role unrecorded)"}


def _readonly(value, *, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _points(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or value.ndim != 2 or value.shape[1:] != (3,) or len(value) < 2:
        raise ValueError(f"{name} requires an exact nonempty Nx3 real path.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains nonfinite coordinates.")
    return _readonly(value, dtype=float)


@dataclass(frozen=True, slots=True)
class SupportView:
    vertices_m: np.ndarray
    faces: np.ndarray
    kind: str
    origin_m: np.ndarray
    axis: np.ndarray
    basis: np.ndarray
    panels_3d: np.ndarray
    panels_2d: np.ndarray
    scope: str

    @classmethod
    def from_mesh(cls, vertices, faces):
        vertices = _points(vertices, "Support vertices")
        raw = np.asarray(faces)
        if raw.dtype.kind not in "iu" or raw.ndim != 2 or raw.shape[1:] != (3,) or not len(raw):
            raise ValueError("Support faces require triangular integer indices.")
        if np.any(raw < 0) or np.any(raw >= len(vertices)):
            raise ValueError("Support triangle indices leave the saved mesh.")
        faces = _readonly(raw, dtype=np.int64)
        origin = vertices.mean(axis=0)
        _, singular, frame = np.linalg.svd(vertices - origin, full_matrices=False)
        scale = max(float(np.ptp(vertices, axis=0).max()), 1e-12)
        tolerance = max(scale * 1e-9, 1e-12)
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        if np.any(np.linalg.norm(normals, axis=1) <= tolerance**2):
            raise ValueError("Degenerate support triangles cannot define diagnostic frames.")
        if singular[-1] <= tolerance * np.sqrt(len(vertices)):
            normal = frame[-1]
            if np.dot(normal, normals.sum(axis=0)) < 0:
                normal = -normal
            # Stable world-axis convention retains y/z for the usual x-normal planes.
            world = np.eye(3)[int(np.argmin(np.abs(normal)))]
            first = world - normal * float(world @ normal)
            first /= np.linalg.norm(first)
            second = np.cross(normal, first)
            return cls(
                vertices,
                faces,
                "plane",
                _readonly(origin),
                _readonly(normal),
                _readonly([first, second]),
                _readonly(np.empty((0, 4, 3))),
                _readonly(np.empty((0, 4, 2))),
                "Physical planar coordinates; height follows saved face orientation.",
            )
        shell, reason = convex_circular_shell(Mesh(vertices, faces), tolerance)
        if shell is None:
            return cls(
                vertices,
                faces,
                "spatial",
                _readonly(origin),
                _readonly(frame[-1]),
                _readonly(frame[:2]),
                _readonly(np.empty((0, 4, 3))),
                _readonly(np.empty((0, 4, 2))),
                "Unrecognized support: world 3D remains authoritative; no developed or signed-height view. " + reason,
            )
        bottom = shell.plane_points.copy()
        axis = shell.axis
        # Ensure the chosen circular order advances consistently around the actual shell.
        cross = np.sum(np.cross(bottom - shell.origin, np.roll(bottom, -1, axis=0) - shell.origin), axis=0)
        if cross @ axis < 0:
            bottom = bottom[::-1]
        # The cut is deterministic in world coordinates; it has no engineering meaning.
        helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
        radial = helper - axis * float(helper @ axis)
        radial /= np.linalg.norm(radial)
        angles = np.arctan2((bottom - shell.origin) @ np.cross(axis, radial), (bottom - shell.origin) @ radial)
        start = int(np.argmin(np.mod(angles, 2 * np.pi)))
        bottom = np.roll(bottom, -start, axis=0)
        top = (
            shell.origin
            + shell.axial_length * axis
            + (bottom - shell.origin) * (shell.top_radius / shell.bottom_radius)
        )
        panels = np.stack((bottom, np.roll(bottom, -1, axis=0), np.roll(top, -1, axis=0), top), axis=1)
        developed = []
        first_bottom, first_top = np.zeros(2), np.array([0.0, np.linalg.norm(top[0] - bottom[0])])
        for panel in panels:
            generator = panel[3] - panel[0]
            generator /= np.linalg.norm(generator)
            edge = panel[1] - panel[0]
            transverse = edge - generator * float(edge @ generator)
            transverse /= np.linalg.norm(transverse)
            vertical = (first_top - first_bottom) / np.linalg.norm(first_top - first_bottom)
            horizontal = np.array([vertical[1], -vertical[0]])
            local = panel - panel[0]
            unfolded = (
                first_bottom + (local @ generator)[:, None] * vertical + (local @ transverse)[:, None] * horizontal
            )
            developed.append(unfolded)
            first_bottom, first_top = unfolded[1], unfolded[2]
        return cls(
            vertices,
            faces,
            "shell",
            _readonly(shell.origin),
            _readonly(axis),
            _readonly(frame[:2]),
            _readonly(panels),
            _readonly(developed),
            "Full polygonal support unfolded panel by panel in metres; each facet is isometric. "
            "The display cut is omitted across the seam. Free 3D joins are projections; crossings here do not prove collisions.",
        )

    def coordinates(self, points):
        """Physical facet coordinates and signed face height; not a gap certificate."""
        points = np.asarray(points, dtype=float)
        if self.kind == "plane":
            relative = points - self.origin_m
            return relative @ self.basis.T, relative @ self.axis, np.zeros(len(points), dtype=int)
        if self.kind != "shell":
            return (points - self.origin_m) @ self.basis.T, np.full(len(points), np.nan), np.full(len(points), -1)
        panels = self.panels_3d
        if len(points) * len(panels) > 50_000_000:
            raise ValueError(
                "Developed projection exceeds its 50-million-facet-sample work budget; geometry is not decimated."
            )
        generator = panels[:, 3] - panels[:, 0]
        generator /= np.linalg.norm(generator, axis=1)[:, None]
        transverse = panels[:, 1] - panels[:, 0]
        transverse -= generator * np.einsum("ij,ij->i", transverse, generator)[:, None]
        transverse /= np.linalg.norm(transverse, axis=1)[:, None]
        normals = np.cross(transverse, generator)
        radial = panels[:, 0] - self.origin_m
        radial -= (radial @ self.axis)[:, None] * self.axis
        normals *= np.where(np.einsum("ij,ij->i", normals, radial) > 0, 1.0, -1.0)[:, None]
        # Maximal outward halfspace selects the local side facet. At edges this
        # is a signed support-plane height, not Euclidean distance to the solid.
        selected, height = np.empty(len(points), dtype=int), np.empty(len(points))
        for start in range(0, len(points), 4096):
            signed = np.einsum("pij,ij->pi", points[start : start + 4096, None] - panels[:, 0], normals)
            local = np.argmax(signed, axis=1)
            selected[start : start + len(local)] = local
            height[start : start + len(local)] = signed[np.arange(len(local)), local]
        relative = points - panels[selected, 0]
        vertical = self.panels_2d[:, 3] - self.panels_2d[:, 0]
        vertical /= np.linalg.norm(vertical, axis=1)[:, None]
        horizontal = np.column_stack((vertical[:, 1], -vertical[:, 0]))
        xy = self.panels_2d[selected, 0] + (
            np.einsum("ij,ij->i", relative, generator[selected])[:, None] * vertical[selected]
            + np.einsum("ij,ij->i", relative, transverse[selected])[:, None] * horizontal[selected]
        )
        return xy, height, selected

    def visible_intervals(self, panels):
        if self.kind != "shell":
            return np.ones(max(len(panels) - 1, 0), dtype=bool)
        # Never draw a line across the arbitrary display cut.
        return np.abs(np.diff(panels)) < max(len(self.panels_3d) - 1, 1)


@dataclass(frozen=True, slots=True)
class RoutingSource:
    key: str
    label: str
    part_index: int
    body_id: str
    points_m: np.ndarray
    support: SupportView
    roles: tuple[str, ...]
    group_indices: np.ndarray
    current_A: float | None
    rings_m: np.ndarray | None
    status: str
    issues: tuple[ProblemIssue, ...]
    scope: str
    circuit_status: str = "Drive feed design status not recorded."
    body_measurements: tuple[tuple[str, float, float], ...] = ()

    @property
    def arclength_m(self):
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points_m, axis=0), axis=1))]


@dataclass(frozen=True, slots=True)
class RoutingAnalysis:
    title: str
    run_directory: str
    fabrication_status: str
    sources: tuple[RoutingSource, ...]


def routing_analysis_available(record):
    if record is None:
        return False
    names = getattr(record, "array_names", None)
    if names is None:
        names = getattr(record, "arrays", ())
    return any(
        isinstance(key, str)
        and re.fullmatch(
            r"(?:route_\d+|fabrication_\d+(?:_\d+)?|review_conductor_\d+)_points_m",
            key,
        )
        for key in names
    )


def _connected_supports(vertices, faces):
    """Legacy merged meshes: split by actual connectivity, never by plot overlap."""
    parent = np.arange(len(vertices))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for face in faces:
        for other in face[1:]:
            first, second = find(int(face[0])), find(int(other))
            parent[second] = first
    labels = np.array([find(index) for index in range(len(vertices))])
    output = []
    for root in np.unique(labels):
        chosen = np.flatnonzero(labels == root)
        selected_faces = faces[np.isin(faces[:, 0], chosen)]
        if not len(selected_faces):
            continue
        indices = np.full(len(vertices), -1)
        indices[chosen] = np.arange(len(chosen))
        output.append((vertices[chosen], indices[selected_faces]))
    return output


def _roles(owners, roles, count):
    if owners is None:
        return ("unknown",) * count
    if np.shape(owners) != (count,) or np.asarray(owners).dtype.kind not in "SU":
        raise ValueError("Saved ownership must contain one typed string per exact path interval.")
    if roles is not None and (np.shape(roles) != (count,) or np.asarray(roles).dtype.kind not in "SU"):
        raise ValueError("Saved connector roles must correspond to every exact interval.")
    result = []
    for index, owner in enumerate(owners):
        role = str(roles[index]) if roles is not None else ""
        result.append(
            "connector"
            if owner == "inter_group_connector"
            else "contour"
            if owner == "contour_loop"
            else "return"
            if owner == "within_group_return"
            else "join"
            if role == "joint_spiral_join"
            else "unclassified_added"
            if owner == "contour_group"
            else "unknown"
        )
    return tuple(result)


def _circuit_status(metadata, body_id, part):
    winding = (metadata.get("fabrication") or {}).get("winding", ())
    candidates = (
        [row for row in winding if row.get("body_id") == body_id]
        if body_id
        else [row for row in winding if str(row.get("body_id", "")).startswith(f"part{part + 1:03d}_")]
    )
    if not candidates:
        return "Drive feed design status not recorded."
    if all(row.get("drive_feed_designed") is True for row in candidates):
        return "Explicit drive paths recorded; saved lead and feed-break qualification still applies."
    if all(row.get("circuit_complete") is True for row in candidates):
        return "Closed mathematical winding; drive feed and physical feed break are not designed."
    return "External supply/return or drive closure is not completely designed."


def _body_measurements(diagnostics):
    values = []
    for label, value_key, limit_key in (
        (
            "Final centroid bend radius / required minimum",
            "sweep_final_min_bend_radius_m",
            "sweep_required_bend_radius_m",
        ),
        (
            "Capsule centerline separation / required minimum (conservative screen)",
            "sweep_capsule_min_centerline_distance_m",
            "sweep_required_centerline_clearance_m",
        ),
        (
            "Support penetration / permitted maximum",
            "sweep_support_max_penetration_m",
            "sweep_support_conformity_tolerance_m",
        ),
    ):
        value, limit = diagnostics.get(value_key), diagnostics.get(limit_key)
        if (
            isinstance(value, (float, int))
            and isinstance(limit, (float, int))
            and not isinstance(value, bool)
            and not isinstance(limit, bool)
            and np.isfinite(value)
            and np.isfinite(limit)
        ):
            values.append((label, float(value), float(limit)))
    return tuple(values)


def _saved_frames(record, prefix, body_id, points):
    keys = ("points_m", "normals", "transverse", "cross_section_m", "closed")
    names = set(record.array_names)
    frames = None
    if all(f"{prefix}_frame_{key}" in names for key in keys):
        frames = {key: record.array(f"{prefix}_frame_{key}", copy=False) for key in keys}
        for key in ("kind", "group_index", "connector_role"):
            name = f"{prefix}_owner_{key}"
            if name in names:
                frames["owner_" + key] = record.array(name, copy=False)
    elif body_id:
        # Historical runs retain authenticated frames as a separate artifact.
        from .evidence import read_verified_artifact

        matches = [
            item["name"]
            for item in record.metadata_view.get("artifacts", ())
            if item.get("name", "").replace("\\", "/").split("/")[-1] == f"{body_id}_fabrication_frames.npz"
        ]
        if len(matches) == 1:
            with np.load(BytesIO(read_verified_artifact(record, matches[0])), allow_pickle=False) as archive:
                frames = {key: np.array(archive[key], copy=True) for key in keys}
                frames.update(
                    {
                        key: np.array(archive[key], copy=True)
                        for key in ("owner_kind", "owner_group_index", "owner_connector_role")
                        if key in archive
                    }
                )
    if frames is None:
        return None
    origins = _points(frames["points_m"], "Saved frame centroids")
    closed = np.asarray(frames["closed"])
    if closed.dtype.kind != "b" or closed.size != 1:
        raise ValueError("Saved frame closure must be a single boolean.")
    expected = np.vstack((origins, origins[0])) if bool(closed.item()) else origins
    if not np.array_equal(points, expected):
        raise ValueError("Saved frames do not exactly correspond to the selected geometry.")
    for key in ("normals", "transverse"):
        value = np.asarray(frames[key])
        if value.shape != origins.shape or not np.all(np.isfinite(value)):
            raise ValueError("Saved frame axes must match every physical ring.")
    profile = np.asarray(frames["cross_section_m"])
    if profile.ndim != 2 or profile.shape[1:] != (2,) or len(profile) < 3 or not np.all(np.isfinite(profile)):
        raise ValueError("Saved conductor profile requires finite physical polygon coordinates.")
    if len(origins) * len(profile) > 2_000_000:
        raise ValueError("Physical profile display exceeds its two-million-vertex work budget.")
    return frames


def analyze_routes(record) -> RoutingAnalysis:
    """Read an already verified RunRecord; no solve, rerouting or current fit."""
    metadata, names = record.metadata_view, set(record.array_names)
    if not routing_analysis_available(record):
        raise ValueError("This saved run has no routed or realized conductor geometry.")
    parts = {row["part_index"]: row for row in metadata.get("routing", {}).get("parts", ())}
    supports, legacy = {}, None
    issues = run_problems(record)
    current_bodies = {
        row["body_id"]: row.get("current_A")
        for row in (metadata.get("fabrication") or {}).get("electrical_metrics", {}).get("bodies", ())
    }
    definitions = []
    for name in sorted(names):
        match = re.fullmatch(r"route_(\d+)_points_m", name)
        if match:
            index = int(match[1])
            joint = parts.get(index, {}).get("joint_routing", {})
            status = (
                "Retained baseline · invalid"
                if joint.get("selected_route_status") == "invalid"
                else "Selected routing candidate"
                if joint.get("accepted_improvement_found_within_budget")
                else "Stored route · adoption not recorded"
            )
            definitions.append(
                (name[:-9], index, "", "Native route", status, parts.get(index, {}).get("current_A"), {})
            )
    for section, label in (
        (metadata.get("pipeline_sweeps") or {}, "Pipeline conductor"),
        (metadata.get("fabrication") or {}, "Strict-review conductor"),
    ):
        rows = section.get("bodies", section.get("paths", ()))
        for ordinal, row in enumerate(rows):
            part = int(row.get("part_index", 1)) - 1
            body = int(row.get("body_index", 1)) - 1
            prefix = f"fabrication_{part}_{body}" if label.startswith("Pipeline") else f"review_conductor_{ordinal}"
            if prefix + "_points_m" not in names and label.startswith("Pipeline"):
                prefix = f"fabrication_{part}"
            if prefix + "_points_m" not in names:
                continue
            ok = row.get("manufacturing_ok", (row.get("diagnostics") or {}).get("manufacturing_ok"))
            status = (
                "Body checks passed"
                if ok is True
                else "Body checks failed"
                if ok is False
                else "Body checks not recorded"
            )
            definitions.append(
                (
                    prefix,
                    part,
                    row.get("body_id", ""),
                    label,
                    status,
                    current_bodies.get(row.get("body_id"), parts.get(part, {}).get("current_A")),
                    row.get("diagnostics") or {},
                )
            )
    sources = []
    for prefix, part, body_id, label, status, current, diagnostics in definitions:
        points = _points(record.array(prefix + "_points_m", copy=False), prefix)
        if len(points) > 100_000:
            raise ValueError("Path display exceeds its 100,000-point work budget; no geometry is decimated.")
        if part not in supports:
            if f"support_{part}_vertices_m" in names and f"support_{part}_faces" in names:
                geometry = (
                    record.array(f"support_{part}_vertices_m", copy=False),
                    record.array(f"support_{part}_faces", copy=False),
                )
            else:
                if legacy is None:
                    legacy = _connected_supports(
                        record.array("vertices_m", copy=False), record.array("faces", copy=False)
                    )
                # Historical component order is not assumed. Match the path to
                # the actual nearest component using sampled vertex distances.
                from scipy.spatial import cKDTree

                sample = points[np.linspace(0, len(points) - 1, min(32, len(points))).astype(int)]
                distances = [float(np.mean(cKDTree(vertices).query(sample)[0])) for vertices, _ in legacy]
                order = np.argsort(distances)
                if len(order) > 1 and abs(distances[order[0]] - distances[order[1]]) < 1e-9:
                    raise ValueError("Historical merged support cannot be uniquely associated with this route.")
                geometry = legacy[int(order[0])]
            supports[part] = SupportView.from_mesh(*geometry)
        frames = _saved_frames(record, prefix, body_id if label.startswith("Strict") else "", points)
        rings = None
        kinds = record.array(prefix + "_owner_kind", copy=False) if prefix + "_owner_kind" in names else None
        roles = (
            record.array(prefix + "_owner_connector_role", copy=False)
            if prefix + "_owner_connector_role" in names
            else None
        )
        groups = (
            record.array(prefix + "_owner_group_index", copy=False) if prefix + "_owner_group_index" in names else None
        )
        if frames is not None:
            from ..manufacturing import _polygon_centroid

            profile = np.asarray(frames["cross_section_m"])
            offset = profile - _polygon_centroid(profile)
            rings = (
                np.asarray(frames["points_m"])[:, None]
                + offset[None, :, :1] * np.asarray(frames["transverse"])[:, None]
                + offset[None, :, 1:] * np.asarray(frames["normals"])[:, None]
            )
            if bool(np.asarray(frames["closed"]).item()):
                rings = np.concatenate((rings, rings[:1]))
            kinds, roles, groups = (
                frames.get("owner_" + key, value)
                for key, value in (("kind", kinds), ("connector_role", roles), ("group_index", groups))
            )
        count = len(points) - 1
        if groups is not None and (np.shape(groups) != (count,) or np.asarray(groups).dtype.kind not in "iu"):
            raise ValueError("Saved group IDs must contain one integer per exact path interval.")
        groups = _readonly(groups, dtype=int) if groups is not None else _readonly(np.full(count, -1))
        current = float(current) if isinstance(current, (int, float)) and np.isfinite(current) else None
        source_issues = tuple(
            issue
            for issue in issues
            if issue.body_id == body_id
            and issue.scope == ("fabrication_review" if label.startswith("Strict") else "pipeline_sweep_geometry")
        )
        if frames is not None:
            from .diagnostics import _BODY_REASONS, _frame_witness

            saved_frames = {**frames, "closed": bool(np.asarray(frames["closed"]).item())}
            located = []
            for issue in source_issues:
                reason = issue.key.rsplit(":", 1)[-1]
                definition = _BODY_REASONS.get(reason)
                if definition is not None and definition[-1]:
                    witnesses, intervals, description = _frame_witness(saved_frames, diagnostics, definition[-1])
                    issue = replace(issue, points_m=witnesses, segments=intervals, location_description=description)
                located.append(issue)
            source_issues = tuple(located)
        sources.append(
            RoutingSource(
                prefix,
                label,
                part,
                body_id,
                points,
                supports[part],
                _roles(kinds, roles, count),
                groups,
                current,
                None if rings is None else _readonly(rings),
                status,
                source_issues,
                "Exact saved samples; no display smoothing. "
                + (
                    "Physical profile reconstructed from matching saved final frames."
                    if rings is not None
                    else "Physical profile frames unavailable for this source; centerlines only."
                )
                + " Direction arrows follow signed saved current; unavailable current uses path direction.",
                _circuit_status(metadata, body_id, part),
                _body_measurements(diagnostics),
            )
        )
    return RoutingAnalysis(
        str(metadata.get("name", "Saved winding")),
        str(record.directory),
        run_verdicts(metadata)["fabrication"],
        tuple(sources),
    )


def _closeup_window(source, issue, radius_m):
    if issue is None or not issue.points_m or issue not in source.issues:
        raise ValueError("A close-up requires a located recorded issue belonging to this geometry.")
    witness = np.asarray(issue.points_m)
    center = np.mean(witness, axis=0)
    if "capsule" in issue.explanation.lower() and len(issue.segments) == 2:
        from ..geometry_kernels import segment_closest_parameters

        (a, b), (c, d) = issue.segments
        first, second = segment_closest_parameters(
            witness[a], witness[b] - witness[a], witness[c], witness[d] - witness[c]
        )
        center = (witness[a] + first * (witness[b] - witness[a]) + witness[c] + second * (witness[d] - witness[c])) / 2
    elif "bend" in issue.key and len(witness) == 3:
        center = witness[1]
    if radius_m is None:
        limits = [row[2] for row in source.body_measurements if "minimum" in row[0]]
        radius_m = max(0.003, 1.5 * max(limits, default=0.0))
    if isinstance(radius_m, (bool, np.bool_)) or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Close-up half-width must be a finite positive distance in metres.")
    if source.support.kind == "shell":
        index = source.support.coordinates(center[None])[2][0]
        panel = source.support.panels_3d[index]
        generator = panel[3] - panel[0]
        generator /= np.linalg.norm(generator)
        transverse = panel[1] - panel[0]
        transverse -= generator * (transverse @ generator)
        transverse /= np.linalg.norm(transverse)
        basis = np.asarray([transverse, generator])
    else:
        basis = source.support.basis
    return center, float(radius_m), basis


def draw_routing_closeup(figure, source: RoutingSource, issue, *, envelopes=True, radius_m=None):
    """Export one authenticated witness in physical millimetres without changing geometry.

    The local view projects exact saved samples onto a tangent plane. A capsule
    closest point locates the camera only; it does not certify a solid contact.
    ``radius_m`` is the local world-space half-width, not a geometry tolerance.
    """
    return draw_routing_analysis(figure, source, issue=issue, envelopes=envelopes, closeup=True, radius_m=radius_m)


def draw_routing_analysis(figure, source: RoutingSource, *, issue=None, envelopes=True, closeup=False, radius_m=None):
    """One support at a time; exact 3D is authoritative for every witness."""
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if issue is not None and issue not in source.issues:
        raise ValueError("The selected issue does not belong to this saved geometry.")
    figure.set_layout_engine(None)
    figure.clear()
    developed, spatial, height_axis = (
        figure.add_subplot(221),
        figure.add_subplot(222, projection="3d"),
        figure.add_subplot(212),
    )
    points, support = source.points_m, source.support
    xy, height, panel_indices = support.coordinates(points)
    visible = support.visible_intervals(panel_indices)
    segments = np.stack((points[:-1], points[1:]), axis=1)
    included = np.ones(len(segments), dtype=bool)
    if closeup:
        center, radius, basis = _closeup_window(source, issue, radius_m)
        xy = (points - center) @ basis.T
        profile_radius = (
            float(np.max(np.linalg.norm(source.rings_m - points[:, None], axis=2)))
            if source.rings_m is not None
            else 0.0
        )
        extent = radius + profile_radius
        included = np.all(np.max(segments, axis=1) >= center - extent, axis=1) & np.all(
            np.min(segments, axis=1) <= center + extent, axis=1
        )
        visible = included.copy()
    colors = [COLORS[role] for role in source.roles]
    planar_segments = np.stack((xy[:-1], xy[1:]), axis=1)
    developed.add_collection(
        LineCollection(planar_segments[visible] * 1e3, colors=np.asarray(colors)[visible], linewidths=1.3)
    )
    spatial.add_collection3d(
        Line3DCollection(segments[included] * 1e3, colors=np.asarray(colors)[included], linewidths=1.2)
    )
    triangles = support.vertices_m[support.faces] * 1e3
    if closeup:
        triangles = triangles[
            np.all(np.max(triangles, axis=1) >= (center - radius) * 1e3, axis=1)
            & np.all(np.min(triangles, axis=1) <= (center + radius) * 1e3, axis=1)
        ]
    spatial.add_collection3d(Poly3DCollection(triangles, facecolors="#9ba6ac", alpha=0.07, edgecolors="none"))
    if support.kind == "shell" and not closeup:
        developed.add_collection(
            PolyCollection(
                support.panels_2d * 1e3, facecolors="#e5eaed", edgecolors="#aebbc4", linewidths=0.4, alpha=0.3
            )
        )
    if envelopes and source.rings_m is not None:
        rings = source.rings_m
        quads = np.stack(
            (rings[:-1], rings[1:], np.roll(rings[1:], -1, axis=1), np.roll(rings[:-1], -1, axis=1)), axis=2
        )
        spatial.add_collection3d(
            Poly3DCollection(
                quads[included].reshape(-1, 4, 3) * 1e3,
                facecolors="#587e8a",
                edgecolors="none",
                alpha=0.15 if closeup else 0.1,
            )
        )
        if closeup:
            polygons = (quads[included] - center) @ basis.T
            keep = np.ones(polygons.shape[:2], dtype=bool)
        else:
            envelope_xy, _, envelope_panels = support.coordinates(quads.reshape(-1, 3))
            polygons = envelope_xy.reshape(len(points) - 1, rings.shape[1], 4, 2)
            # A polygon across the display seam would create an invented wide band.
            keep = np.broadcast_to(visible[:, None], polygons.shape[:2]).copy()
        if support.kind == "shell" and not closeup:
            cuts = envelope_panels.reshape(len(points) - 1, rings.shape[1], 4)
            keep &= np.ptp(cuts, axis=2) < len(support.panels_3d) - 1
        developed.add_collection(
            PolyCollection(polygons[keep].reshape(-1, 4, 2) * 1e3, facecolors="#587e8a", edgecolors="none", alpha=0.1)
        )
    # Direction arrows and group IDs use recorded sample intervals and owners.
    count = min(12, len(points) - 1)
    arclength = source.arclength_m
    indices = np.minimum(
        np.searchsorted(arclength, np.linspace(0, arclength[-1], count, endpoint=False)), len(points) - 2
    )
    if closeup:
        indices = np.flatnonzero(included)
        indices = (
            indices[np.linspace(0, len(indices) - 1, min(12, len(indices))).astype(int)] if len(indices) else indices
        )
    sign = -1 if source.current_A is not None and source.current_A < 0 else 1
    for index in indices:
        anchor = points[index + 1] if sign < 0 else points[index]
        spatial.quiver(
            *(anchor * 1e3),
            *((points[index + 1] - points[index]) * sign * 1e3),
            color=colors[index],
            arrow_length_ratio=0.3,
            linewidth=0.8,
        )
        if visible[index]:
            first, last = planar_segments[index] * 1e3
            if sign < 0:
                first, last = last, first
            annotation = developed.annotate(
                "",
                xy=last,
                xytext=first,
                annotation_clip=True,
                arrowprops={"arrowstyle": "->", "color": colors[index], "lw": 0.8},
            )
            annotation.arrow_patch.set_clip_path(developed.patch)
    for group in np.unique(source.group_indices[source.group_indices >= 0])[:16]:
        candidates = np.flatnonzero((source.group_indices == group) & included)
        if not len(candidates):
            continue
        index = int(candidates[0])
        developed.annotate(f"G{group + 1}", xy=xy[index] * 1e3, xytext=(3, 4), textcoords="offset points", fontsize=8)
    shown_issues = [issue] if issue is not None else [value for value in source.issues if value.points_m]
    for shown in shown_issues:
        if not shown.points_m:
            continue
        witnesses = np.asarray(shown.points_m)
        projected = (witnesses - center) @ basis.T if closeup else support.coordinates(witnesses)[0]
        developed.scatter(*((projected * 1e3).T), c="#cc203b", marker="x", s=40, zorder=10)
        spatial.scatter(*((witnesses * 1e3).T), c="#cc203b", marker="x", s=40, depthshade=False)
        for first, last in shown.segments:
            developed.plot(*(projected[[first, last]] * 1e3).T, c="#cc203b", lw=2)
            spatial.plot(*(witnesses[[first, last]] * 1e3).T, c="#cc203b", lw=2)
    developed.autoscale_view()
    developed.set_aspect("equal", adjustable="box")
    developed.set_title(
        "Local tangent projection · exact samples"
        if closeup
        else "Full developed support"
        if support.kind != "spatial"
        else "PCA projection · geometry unrecognized"
    )
    developed.set_xlabel("Local distance [mm]" if closeup else "Developed distance [mm]")
    developed.set_ylabel("Local distance [mm]" if closeup else "Developed distance [mm]")
    bounds = np.vstack((points, support.vertices_m)) * 1e3
    spatial.auto_scale_xyz(bounds[:, 0], bounds[:, 1], bounds[:, 2])
    spatial.set_box_aspect(np.maximum(np.ptp(bounds, axis=0), 1e-6))
    extents = np.ptp(bounds, axis=0)
    for index, axis in enumerate((spatial.xaxis, spatial.yaxis, spatial.zaxis)):
        if extents[index] < 0.05 * max(extents):
            axis.set_ticks([float(np.mean(bounds[:, index]))])
    spatial.set_title("Exact 3D conductor / current direction")
    spatial.set_xlabel("x [mm]")
    spatial.set_ylabel("y [mm]")
    spatial.set_zlabel("z [mm]")
    if closeup:
        developed.set_xlim(-radius * 1e3, radius * 1e3)
        developed.set_ylim(-radius * 1e3, radius * 1e3)
        for axis, middle in zip((spatial.set_xlim, spatial.set_ylim, spatial.set_zlim), center):
            axis((middle - radius) * 1e3, (middle + radius) * 1e3)
        spatial.set_box_aspect((1.0, 1.0, 1.0))
    distance = source.arclength_m * 1e3
    height_axis.add_collection(
        LineCollection(
            np.stack(
                (np.column_stack((distance[:-1], height[:-1])), np.column_stack((distance[1:], height[1:]))), axis=1
            )
            * np.array([1.0, 1e3]),
            colors=colors,
            linewidths=1.1,
        )
    )
    height_axis.autoscale_view()
    height_axis.axhline(0, color="#607080", linewidth=0.7)
    height_axis.set_xlabel("Cumulative 3D path length [mm]")
    height_axis.set_ylabel("Signed support-face height [mm]")
    height_axis.set_title("Saved sample heights · positive follows outward shell / saved plane orientation")
    if closeup:
        height_axis.clear()
        height_axis.set_axis_off()
        measurement = (
            "Measured value not recorded"
            if issue.value is None
            else f"Measured {issue.value * (1e3 if issue.unit == 'm' else 1):.6g} {'mm' if issue.unit == 'm' else issue.unit}"
        )
        if issue.limit is not None:
            measurement += f" / limit {issue.limit * (1e3 if issue.unit == 'm' else 1):.6g} {'mm' if issue.unit == 'm' else issue.unit}"
        rows = [issue.title, measurement, issue.explanation]
        rows.extend(
            f"Saved body: {name}: {value * 1e3:.6g} / {limit * 1e3:.6g} mm"
            for name, value, limit in source.body_measurements
        )
        rows.extend(
            (
                "Local tangent projection does not certify clearance. Capsule overlap is a conservative screen, not an exact solid intersection.",
                "Exact saved centerlines and final profile rings; no smoothing, offset or rerouting."
                if source.rings_m is not None
                else "Exact saved centerlines; profile rings unavailable, so no conductor width is invented.",
            )
        )
        height_axis.text(
            0.0, 1.0, "\n".join(rows), transform=height_axis.transAxes, va="top", fontsize=9, linespacing=1.6, wrap=True
        )
    if support.kind == "spatial":
        height_axis.text(
            0.5, 0.5, "Signed support height unavailable for this support", transform=height_axis.transAxes, ha="center"
        )
    for axis in (developed, height_axis):
        axis.grid(alpha=0.2)
    for axis in (developed, spatial, height_axis):
        axis.title.set_fontsize(10)
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_fontsize(9)
        axis.yaxis.label.set_fontsize(9)
    spatial.zaxis.label.set_fontsize(9)
    legend = [
        Line2D([], [], color=color, label=ROLE_LABELS.get(role, role.capitalize()))
        for role, color in COLORS.items()
        if role in source.roles
    ]
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=5, fontsize=8)
    current = f"{source.current_A:+.6g} A" if source.current_A is not None else "Current unrecorded; arrows follow path"
    figure.suptitle(
        f"Support {source.part_index + 1} · {source.label} · {source.status} · {current}", fontsize=11, y=0.995
    )
    figure.text(0.02, 0.025, source.circuit_status, fontsize=8, color="#6e3a24")
    if not closeup:
        count = sum(bool(value.points_m) for value in source.issues)
        figure.text(
            0.02,
            0.055,
            f"{'Selected' if issue else f'All {count} located'} saved check witnesses shown. "
            "Projection crossings are not collision evidence; capsule overlaps are conservative screens.",
            fontsize=7,
        )
    figure.subplots_adjust(top=0.84, bottom=0.11, hspace=0.8, wspace=0.22)
    return developed, spatial, height_axis
