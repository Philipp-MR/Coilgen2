"""Owned scene geometry and explicit constraints on realized electrical paths.

Scene checks do not alter the stream-function solution. They gate acceptance
of the recorded winding, including supplied feeds, and retain failure witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from numbers import Real
from pathlib import Path

import numpy as np

from ..models import Mesh
from ..geometry_kernels import triangle_pair_distances, segment_distances
from .gradient_studies import RigidTransform, _xyz


SCENE_SCHEMA = "coilgen-spatial-scene-v1"
ROLES = {"obstacle", "keepout", "feed_corridor", "conducting_metal", "reference"}


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{label} requires 1–200 characters.")
    return value


def _length(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative in metres.")
    return float(value)


@dataclass(frozen=True, slots=True)
class ScenePart:
    name: str
    role: str
    vertices_m: object
    faces: object
    clearance_m: float = 0.0
    source: str = "User-authored world geometry"
    material: str = "unspecified"
    _mesh: Mesh = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _text(self.name, "Scene name")
        _text(self.source, "Scene source")
        _text(self.material, "Material")
        if self.role not in ROLES:
            raise ValueError("Unsupported scene role.")
        vertices = _xyz(self.vertices_m, "Scene vertices", 3)
        raw_faces = np.asarray(self.faces)
        if raw_faces.dtype.kind not in "iu" or raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("Scene triangles must be integer vertex triples.")
        faces = raw_faces.astype(np.int64, copy=True)
        if not 4 <= len(faces) <= 200_000 or len(vertices) > 200_000:
            raise ValueError("Scene meshes require 4–200,000 triangles and at most 200,000 vertices.")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("Scene triangle indices leave the vertex table.")
        triangles = vertices[faces]
        if np.any(
            np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) == 0
        ):
            raise ValueError("Scene triangles cannot be degenerate.")
        if self.role != "reference":
            edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
            _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
            if np.any(counts != 2):
                raise ValueError("Exclusions and corridors require closed manifold surfaces.")
            # The solid-angle containment predicate requires consistent orientation.
            directed = np.sort(edges, axis=1)
            _, inverse = np.unique(directed, axis=0, return_inverse=True)
            signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
            if np.any(np.bincount(inverse, weights=signs) != 0):
                raise ValueError("Scene surface triangles require consistent orientation.")
        faces.setflags(write=False)
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "clearance_m", _length(self.clearance_m, "Scene clearance"))
        object.__setattr__(self, "_mesh", Mesh(vertices, faces))

    @property
    def geometry_sha256(self):
        return sha256(
            np.asarray(self.vertices_m, dtype="<f8").tobytes() + np.asarray(self.faces, dtype="<i8").tobytes()
        ).hexdigest()

    def to_dict(self):
        return {
            "name": self.name,
            "role": self.role,
            "vertices_m": self.vertices_m.tolist(),
            "faces": self.faces.tolist(),
            "clearance_m": self.clearance_m,
            "source": self.source,
            "material": self.material,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "name",
            "role",
            "vertices_m",
            "faces",
            "clearance_m",
            "source",
            "material",
        }:
            raise ValueError("Malformed scene part.")
        return cls(**value)

    @classmethod
    def box(cls, name, *, center_m=(0, 0, 0), size_m=(0.01, 0.01, 0.01), role="keepout", transform=None, **kwargs):
        center = np.asarray(center_m, dtype=float)
        size = np.asarray(size_m, dtype=float)
        if center.shape != (3,) or size.shape != (3,) or not np.all(np.isfinite([center, size])) or np.any(size <= 0):
            raise ValueError("Box center and positive full dimensions require finite xyz metres.")
        vertices = center + 0.5 * size * np.asarray(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
        )
        if transform is not None:
            if not isinstance(transform, RigidTransform):
                raise ValueError("Box registration requires a rigid transform.")
            vertices = vertices @ np.asarray(transform.rotation).T + np.asarray(transform.translation_m)
        faces = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
        return cls(name, role, vertices, faces, **kwargs)

    @classmethod
    def import_mesh(cls, path, *, name, role, length_unit, transform=RigidTransform(), **kwargs):
        from ..geometry import read_stl, read_mesh_npz

        units = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "um": 1e-6}
        if length_unit not in units or not isinstance(transform, RigidTransform):
            raise ValueError("Explicit mesh length units and a proper rigid registration are required.")
        path = Path(path)
        if path.suffix.lower() == ".npz" and length_unit != "m":
            raise ValueError("NPZ vertices_m are already SI metres; select m for NPZ imports.")
        mesh = (
            read_stl(path)
            if path.suffix.lower() == ".stl"
            else read_mesh_npz(path)
            if path.suffix.lower() == ".npz"
            else None
        )
        if mesh is None:
            raise ValueError("Import an STL or NPZ scene mesh.")
        vertices = mesh.vertices * units[length_unit]
        vertices = vertices @ np.asarray(transform.rotation).T + np.asarray(transform.translation_m)
        return cls(
            name,
            role,
            vertices,
            mesh.faces,
            source=f"{path.name} · SHA-256 {sha256(path.read_bytes()).hexdigest()}",
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class RouteAnchor:
    name: str
    kind: str
    point_m: object
    body_id: str
    expected_path_sha256: str
    tolerance_m: float = 0.001

    def __post_init__(self):
        _text(self.name, "Anchor name")
        _text(self.body_id, "Anchor body")
        if self.kind not in {"cut", "positive_terminal", "negative_terminal", "waypoint"}:
            raise ValueError("Unsupported route anchor kind.")
        point = _xyz([self.point_m], "Anchor point")[0]
        point.setflags(write=False)
        object.__setattr__(self, "point_m", point)
        digest = self.expected_path_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Anchors require the exact recorded winding SHA-256.")
        object.__setattr__(self, "tolerance_m", _length(self.tolerance_m, "Anchor tolerance"))

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "point_m": self.point_m.tolist(),
            "body_id": self.body_id,
            "expected_path_sha256": self.expected_path_sha256,
            "tolerance_m": self.tolerance_m,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "name",
            "kind",
            "point_m",
            "body_id",
            "expected_path_sha256",
            "tolerance_m",
        }:
            raise ValueError("Malformed route anchor.")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SpatialScene:
    parts: tuple[ScenePart, ...] = ()
    anchors: tuple[RouteAnchor, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        if (
            len(self.parts) > 64
            or len(self.anchors) > 256
            or any(not isinstance(p, ScenePart) for p in self.parts)
            or any(not isinstance(a, RouteAnchor) for a in self.anchors)
        ):
            raise ValueError("Scenes support at most 64 parts and 256 bound anchors.")
        names = [p.name for p in (*self.parts, *self.anchors)]
        if len(set(names)) != len(names):
            raise ValueError("Scene and anchor names must be unique.")

    def to_dict(self):
        return {
            "schema": SCENE_SCHEMA,
            "parts": [p.to_dict() for p in self.parts],
            "anchors": [a.to_dict() for a in self.anchors],
        }

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "parts", "anchors"}
            or value["schema"] != SCENE_SCHEMA
        ):
            raise ValueError("Malformed spatial scene.")
        return cls(
            tuple(ScenePart.from_dict(p) for p in value["parts"]),
            tuple(RouteAnchor.from_dict(a) for a in value["anchors"]),
        )

    @property
    def digest(self):
        return sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


def path_mesh_clearance(points_m, part: ScenePart, *, radius_m=0.0):
    """Continuous segment distances, with containment; no point-sampling gaps.

    A circumscribed circular conductor envelope is conservative for any profile
    inside that radius. Contact/overlap may be conservative; a pass is certified
    for that envelope and the imported faceted, consistently oriented surface.
    """
    from ..research.fabrication import _inside_closed_mesh

    points = _xyz(points_m, "Electrical path", 2)
    radius = _length(radius_m, "Envelope radius")
    required = radius + part.clearance_m
    triangles = part.vertices_m[part.faces]
    # Degenerate triangles encode segments; the shared distance kernel handles
    # them exactly, including intersections with triangle interiors.
    best, witness = np.inf, None
    inside = _inside_closed_mesh(points[0], part._mesh)
    starts, ends = points[:-1], points[1:]
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        segment = np.asarray([start, end, end])
        lower = np.linalg.norm(
            np.maximum(
                0, np.maximum(triangles.min(axis=1) - segment.max(axis=0), segment.min(axis=0) - triangles.max(axis=1))
            ),
            axis=1,
        )
        eligible = np.flatnonzero(lower <= best)
        for chunk_start in range(0, len(eligible), 4096):
            ids = eligible[chunk_start : chunk_start + 4096]
            distance = triangle_pair_distances(
                np.broadcast_to(segment, (len(ids), 3, 3)), triangles[ids], backend="numpy"
            ).values_m
            local = int(np.argmin(distance))
            if distance[local] < best:
                best, witness = float(distance[local]), index
        if best == 0:
            break
    corridor = part.role == "feed_corridor"
    # The continuous distance check detects every crossing. A single initial
    # containment test therefore suffices, even for concave closed surfaces.
    containment_ok = inside if corridor else not inside
    passed = containment_ok and best >= required and (best > 0 or required > 0)
    return {
        "passed": bool(passed),
        "minimum_centerline_surface_distance_m": float(best),
        "required_envelope_gap_m": required,
        "inside": bool(inside),
        "witness_segment": witness,
        "witness_segment_points_m": points[witness : witness + 2].tolist() if witness is not None else None,
        "geometry_model": "circumscribed_conductor_capsules_and_faceted_scene",
        "conservative_overlap": not passed,
    }


def evaluate_spatial_scene(result, scene: SpatialScene, *, fabrication=None):
    """Review exact saved centroid/electrical paths, preserving their currents."""
    from ..evaluation import layout_field_geometry
    from ..gradient_engineering import winding_path_sha256
    from ..manufacturing import _polygon_centroid

    rows, paths, radius = [], {}, {}
    config = result.input_data
    profile = np.asarray(config.cross_sectional_points, dtype=float)
    fallback_radius = 0.5 * np.hypot(config.conductor_cross_section_width, config.conductor_cross_section_height)
    envelope = (
        float(np.max(np.linalg.norm(profile - _polygon_centroid(profile), axis=1)))
        if len(profile) > 2
        else float(fallback_radius)
    )
    envelope += config.sweep_insulation_thickness + config.sweep_manufacturing_tolerance
    plans, winding_digests, original_windings, conductor_meshes = {}, {}, {}, {}
    if fabrication is not None:
        plans = {p["body_id"]: p for p in fabrication.winding_steps}
        for body in fabrication.paths:
            plan = plans[body.body_id]
            paths[body.body_id] = plan["electrical_points_m"]
            winding_digests[body.body_id] = winding_path_sha256(body.points_m)
            original_windings[body.body_id] = body.points_m
            conductor_meshes[body.body_id] = (body.mesh,) if hasattr(body, "mesh") else ()
            cross = body.path.cross_section_m
            radius[body.body_id] = (
                float(np.max(np.linalg.norm(cross - _polygon_centroid(cross), axis=1)))
                + config.sweep_insulation_thickness
                + config.sweep_manufacturing_tolerance
            )
        source = "fabricated_electrical_paths"
    else:
        for index, part in enumerate(result.coil_parts):
            points, _ = layout_field_geometry(part, config)
            if points is not None:
                id_ = f"evaluated_winding_{index}"
                paths[id_] = points
                winding_digests[id_] = winding_path_sha256(points)
                radius[id_] = envelope
                meshes = tuple(body.mesh for body in getattr(part, "swept_bodies", ()))
                legacy_mesh = getattr(part, "swept_body", None)
                conductor_meshes[id_] = meshes or ((legacy_mesh,) if legacy_mesh is not None else ())
        source = "saved_evaluated_centroid_paths_feeds_excluded"
    for part in scene.parts:
        if part.role == "reference":
            continue
        if not paths:
            rows.append(
                {"name": part.name, "passed": None, "reason": "No realized winding is available.", "scope": source}
            )
        for id_, points in paths.items():
            checked = points
            if part.role == "feed_corridor":
                plan = plans.get(id_, {})
                feed_intervals = [
                    op
                    for op in plan.get("steps", ())
                    if op.get("segment_role") in {"external_supply", "external_return", "drive_closure"}
                ]
                if not feed_intervals:
                    rows.append(
                        {
                            "name": f"{part.name} · {id_}",
                            "passed": None,
                            "reason": "No role-separated modeled feeds are available.",
                            "scope": source,
                        }
                    )
                    continue
                checked_paths = [
                    points[int(op["start_sample_index"]) : int(op["end_sample_index"]) + 1] for op in feed_intervals
                ]
            else:
                checked_paths = [checked]
            for index, checked in enumerate(checked_paths):
                row = path_mesh_clearance(checked, part, radius_m=radius[id_])
                meshes = conductor_meshes.get(id_, ())
                if meshes and part.role != "feed_corridor":
                    solids = [
                        {
                            "body_index": i,
                            **mesh_scene_clearance(
                                mesh,
                                part,
                                margin_m=config.sweep_insulation_thickness + config.sweep_manufacturing_tolerance,
                            ),
                        }
                        for i, mesh in enumerate(meshes)
                    ]
                    solid = min(
                        solids,
                        key=lambda value: (
                            value["passed"],
                            value["minimum_surface_distance_m"]
                            if value["minimum_surface_distance_m"] is not None
                            else float("inf"),
                        ),
                    )
                    solid = {**solid, "body_checks": solids}
                    row["realized_conductor_surface"] = solid
                    row["passed"] = row["passed"] and solid["passed"]
                    if not solid["passed"]:
                        row["reason"] = (
                            "Recorded swept conductor and scene solid overlap by containment."
                            if solid["containment"]
                            else "Recorded swept conductor surface fails the required scene clearance."
                        )
                rows.append(
                    {
                        "name": f"{part.name} · {id_}" + (f" · feed {index + 1}" if len(checked_paths) > 1 else ""),
                        "scope": source,
                        "body_id": id_,
                        "scene_geometry_sha256": part.geometry_sha256,
                        **row,
                    }
                )
    for anchor in scene.anchors:
        points = paths.get(anchor.body_id)
        row = {"name": anchor.name, "scope": source, "body_id": anchor.body_id, "passed": None}
        if points is None or winding_digests.get(anchor.body_id) != anchor.expected_path_sha256:
            row["reason"] = "Anchor winding identity is absent or stale; pick against the intended verified path."
        else:
            plan = plans.get(anchor.body_id, {})
            selected = points
            if anchor.kind in {"positive_terminal", "negative_terminal"}:
                key = "positive_terminal_xyz_m" if anchor.kind == "positive_terminal" else "negative_terminal_xyz_m"
                value = plan.get(key)
                if value is None:
                    row["reason"] = "Explicit terminal geometry is unavailable."
                    rows.append(row)
                    continue
                selected = np.asarray([value, value])
            elif anchor.kind == "cut":
                removed = plan.get("feed_break_segment")
                if removed is None:
                    row["reason"] = "An explicit geometry-bound feed break is unavailable."
                    rows.append(row)
                    continue
                original = original_windings.get(anchor.body_id)
                if original is None or not 0 <= removed < len(original) - 1:
                    row["reason"] = "The recorded cut interval cannot be resolved on its source winding."
                    rows.append(row)
                    continue
                selected = original[removed : removed + 2]
            distances = segment_distances(anchor.point_m, np.zeros(3), selected[:-1], np.diff(selected, axis=0))
            distance = float(np.min(distances))
            row.update(distance_m=distance, limit_m=anchor.tolerance_m, passed=distance <= anchor.tolerance_m)
        rows.append(row)
    checks = []
    for row in rows:
        solid = row.get("realized_conductor_surface", {})
        surface_failed = solid.get("passed") is False
        checks.append(
            {
                "name": f"Spatial: {row['name']}",
                "passed": row["passed"],
                "scope": row["scope"] + ("; recorded swept facets" if surface_failed else ""),
                "value": solid.get("minimum_surface_distance_m")
                if surface_failed
                else row.get("minimum_centerline_surface_distance_m", row.get("distance_m")),
                "limit": solid.get("required_surface_distance_m")
                if surface_failed
                else row.get("required_envelope_gap_m", row.get("limit_m")),
                "unit": "m",
                "reason": row.get("reason", ""),
            }
        )
    return {
        "schema": "coilgen-spatial-review-v1",
        "scene_sha256": scene.digest,
        "scope": source,
        "coverage": "recorded_world_geometry_only; finite posed field studies do not establish posed clearance",
        "rows": rows,
        "checks": checks,
        "passed": all(r["passed"] is True for r in rows) if rows else None,
        "feeds_modeled": bool(plans) and all(p.get("circuit_complete") is True for p in plans.values()),
        "construction_released": False,
        "routing_policy": "Review realized geometry; constraints do not silently reroute the numerical design.",
    }


def mesh_scene_clearance(mesh, part: ScenePart, *, margin_m=0):
    """Check actual swept facets, including miter extensions beyond capsules."""
    from ..research.fabrication import _candidate_pairs, _inside_closed_mesh

    required = part.clearance_m + _length(margin_m, "Conductor surface offset")
    tolerance = 1e-10
    threshold = required + tolerance
    triangles = mesh.vertices[mesh.faces]
    other = part.vertices_m[part.faces]
    pairs = _candidate_pairs(triangles, other, threshold)
    gap, witness = np.inf, None
    for start in range(0, len(pairs), 4096):
        block = pairs[start : start + 4096]
        distances = triangle_pair_distances(triangles[block[:, 0]], other[block[:, 1]], backend="numpy").values_m
        index = int(np.argmin(distances))
        if distances[index] < gap:
            gap, witness = float(distances[index]), block[index].tolist()
    contained = any(_inside_closed_mesh(point, part._mesh) for point in _component_points(mesh)) or any(
        _inside_closed_mesh(point, mesh) for point in _component_points(part._mesh)
    )
    exact_gap = gap <= threshold
    return {
        "passed": bool(not contained and gap > tolerance and gap >= required),
        "minimum_surface_distance_m": gap if exact_gap else None,
        "certified_surface_distance_lower_bound_m": threshold if not exact_gap else None,
        "required_surface_distance_m": required,
        "containment": bool(contained),
        "witness_triangle_pair": witness,
        "geometry_model": "recorded_swept_facets_and_scene_surface",
    }


def _component_points(mesh):
    """One used vertex per connected surface component, including nested solids."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.concatenate((mesh.faces[:, :2], mesh.faces[:, 1:]))
    graph = coo_matrix((np.ones(len(edges), dtype=bool), edges.T), shape=(len(mesh.vertices), len(mesh.vertices)))
    _, labels = connected_components(graph, directed=False)
    used = np.unique(mesh.faces)
    _, first = np.unique(labels[used], return_index=True)
    return mesh.vertices[used[first]]
