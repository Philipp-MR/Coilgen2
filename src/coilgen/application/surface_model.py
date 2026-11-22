"""Editable CAD recipes and their explicit, SI surface-mesh handoff.

OpenCASCADE (through Gmsh) owns the geometry. Display tessellation is disposable;
only final meshing produces an input for the numerical pipeline. No kernel is
imported at module import time, so saved meshes work without the CAD extra.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real
import re
from threading import RLock
from uuid import uuid4

import numpy as np

from ..geometry import boundary_loops, compute_face_normals, compute_vertex_normals, load_mesh_from_config
from ..geometry import split_disconnected_mesh
from ..models import Mesh
from ..geometry_collision import accelerated_mesh_collision_diagnostics


SCHEMA = "coilgen-surface-model-v1"
PRIMITIVES = {
    "cylinder": {"radius": 0.08, "height": 0.24},
    "sphere": {"radius": 0.06},
    "box": {"width": 0.16, "depth": 0.16, "height": 0.24},
    "plane": {"width": 0.16, "height": 0.24},
}
MAX_NODES = 24
MAX_TRIANGLES = 200_000
_LOCK = RLock()


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite, between {low:g} and {high:g}.")
    return float(value)


def _vector(value, name, limit):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} needs three values.")
    return [_number(v, name, -limit, limit) for v in value]


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SurfaceModel:
    """Validated, copied construction tree. Dimensions/metres; XYZ rotations/radians.

    Rotations apply about fixed X, Y, then Z axes through the primitive's centre:
    Rz @ Ry @ Rx. Operations consume inputs, which remain editable in the tree.
    Face references include exact geometric signatures: changed faces must be
    reselected, never silently substituted by a kernel's recycled entity tag.
    """

    _value: dict

    def __post_init__(self):
        value = deepcopy(self._value)
        if not isinstance(value, dict) or set(value) != {"schema", "nodes", "excluded_faces", "edge_length_m"}:
            raise ValueError("Malformed surface model.")
        if value["schema"] != SCHEMA:
            raise ValueError("Unsupported surface-model schema.")
        nodes = value["nodes"]
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
            raise ValueError(f"A surface model needs 1–{MAX_NODES} objects and operations.")
        dimensions, consumed = {}, set()
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("Each object must be a mapping.")
            key, kind = node.get("id"), node.get("kind")
            if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", key) or key in dimensions:
                raise ValueError("Object identifiers must be unique ASCII names.")
            if not isinstance(node.get("name"), str) or not 1 <= len(node["name"]) <= 80:
                raise ValueError("Each object needs a name of 1–80 characters.")
            if kind in PRIMITIVES:
                if set(node) != {"id", "name", "kind", "parameters", "center_m", "rotation_rad"}:
                    raise ValueError("Malformed primitive object.")
                if not isinstance(node["parameters"], dict) or set(node["parameters"]) != set(PRIMITIVES[kind]):
                    raise ValueError(f"Invalid {kind} dimensions.")
                node["parameters"] = {k: _number(v, k + " (m)", 1e-5, 10) for k, v in node["parameters"].items()}
                node["center_m"] = _vector(node["center_m"], "Centre (m)", 10)
                node["rotation_rad"] = _vector(node["rotation_rad"], "Rotation (rad)", 2 * math.pi)
                dimensions[key] = 2 if kind == "plane" else 3
            elif kind == "subtract":
                if set(node) != {"id", "name", "kind", "base", "tool"}:
                    raise ValueError("Malformed subtraction.")
                base, tool = node["base"], node["tool"]
                if base not in dimensions or tool not in dimensions or base == tool:
                    raise ValueError("Subtraction needs two distinct, earlier objects.")
                if base in consumed or tool in consumed:
                    raise ValueError("Choose two current result objects, not already consumed operands.")
                if dimensions[tool] != 3:
                    raise ValueError("The subtraction tool must be a closed solid. A plane is a sheet.")
                dimensions[key] = dimensions[base]
                consumed.update((base, tool))
            else:
                raise ValueError(f"Unknown geometry operation: {kind!r}.")
        excluded = value["excluded_faces"]
        if (
            not isinstance(excluded, list)
            or len(excluded) > 500
            or any(not isinstance(k, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+:[0-9a-f]{64}", k) for k in excluded)
            or len(set(excluded)) != len(excluded)
        ):
            raise ValueError("Invalid excluded-face references.")
        value["edge_length_m"] = _number(value["edge_length_m"], "Mesh edge length (m)", 1e-5, 1)
        object.__setattr__(self, "_value", value)

    @classmethod
    def initial(cls):
        return cls(dict(schema=SCHEMA, nodes=[primitive("cylinder", "object1")], excluded_faces=[], edge_length_m=0.01))

    def to_dict(self):
        return deepcopy(self._value)

    @property
    def identity(self):
        return _digest(self._value)

    @property
    def roots(self):
        consumed = {n[k] for n in self._value["nodes"] if n["kind"] == "subtract" for k in ("base", "tool")}
        return [n["id"] for n in self._value["nodes"] if n["id"] not in consumed]

    @classmethod
    def from_project(cls, project):
        if project is None or str(project.config.coil_mesh_file).lower() == "none":
            return None
        mesh = load_mesh_from_config(project.config, project.project_root)
        recipe = mesh.metadata.get("surface_model")
        if recipe is None:
            return None
        model = cls(recipe)
        if mesh.metadata.get("surface_model_identity") != model.identity:
            raise ValueError("The saved surface recipe failed its integrity check.")
        if mesh.metadata.get("surface_mesh_digest") != mesh_digest(mesh):
            raise ValueError("The saved surface mesh disagrees with its authoring receipt.")
        return model


def primitive(kind, key, *, center_m=(0, 0, 0)):
    if kind not in PRIMITIVES:
        raise ValueError("Unknown primitive.")
    return dict(
        id=key,
        name=kind.capitalize() + " " + key.removeprefix("object"),
        kind=kind,
        parameters=dict(PRIMITIVES[kind]),
        center_m=list(center_m),
        rotation_rad=[0, 0, 0],
    )


def mesh_digest(mesh):
    digest = sha256()
    digest.update(np.asarray(mesh.vertices, dtype="<f8").tobytes())
    digest.update(np.asarray(mesh.faces, dtype="<i8").tobytes())
    return digest.hexdigest()


@contextmanager
def _kernel():
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Geometry studio needs Gmsh. Run Setup CoilGen2.cmd, or install coilgen-python[geometry]."
        ) from exc
    with _LOCK:
        owned = not gmsh.isInitialized()
        previous = ""
        options = {}
        created = False
        try:
            if owned:
                gmsh.initialize([], readConfigFiles=False, interruptible=False)
            else:
                previous = gmsh.model.getCurrent()
            settings = {
                "General.Terminal": 0,
                "General.NumThreads": 1,
                "Mesh.MaxNumThreads2D": 1,
                "Mesh.Algorithm": 6,
                "Mesh.ElementOrder": 1,
                "Mesh.RecombineAll": 0,
                "Mesh.MeshSizeFromPoints": 0,
                "Mesh.MeshSizeFromCurvature": 0,
                "Mesh.MeshSizeExtendFromBoundary": 0,
            }
            options = {k: gmsh.option.getNumber(k) for k in settings}
            for key, val in settings.items():
                gmsh.option.setNumber(key, val)
            gmsh.model.add("coilgen-" + uuid4().hex)
            created = True
            yield gmsh
        finally:
            if gmsh.isInitialized():
                if created:
                    gmsh.model.remove()
                if owned:
                    gmsh.finalize()
                else:
                    for key, val in options.items():
                        gmsh.option.setNumber(key, val)
                    if previous:
                        gmsh.model.setCurrent(previous)


def _build(gmsh, model):
    occ = gmsh.model.occ
    shapes = {}
    for node in model._value["nodes"]:
        kind, p = node["kind"], node.get("parameters", {})
        if kind == "subtract":
            entities, _ = occ.cut(shapes[node["base"]], shapes[node["tool"]], removeObject=True, removeTool=True)
            if not entities:
                raise ValueError(f"{node['name']} removes the complete object; no surface remains.")
        else:
            if kind == "cylinder":
                tag = occ.addCylinder(0, 0, -p["height"] / 2, 0, 0, p["height"], p["radius"])
            elif kind == "sphere":
                tag = occ.addSphere(0, 0, 0, p["radius"])
            elif kind == "box":
                tag = occ.addBox(
                    -p["width"] / 2, -p["depth"] / 2, -p["height"] / 2, p["width"], p["depth"], p["height"]
                )
            else:
                tag = occ.addRectangle(-p["width"] / 2, -p["height"] / 2, 0, p["width"], p["height"])
            entities = [(2 if kind == "plane" else 3, tag)]
            for axis, angle in zip(np.eye(3), node["rotation_rad"]):
                if angle:
                    occ.rotate(entities, 0, 0, 0, *axis, angle)
            occ.translate(entities, *node["center_m"])
        shapes[node["id"]] = entities
    occ.synchronize()
    faces, signatures = [], set()
    names = {n["id"]: n["name"] for n in model._value["nodes"]}
    for root in model.roots:
        for dim, tag in shapes[root]:
            boundaries = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=True) if dim == 3 else [(2, tag)]
            for _, signed in boundaries:
                face = abs(signed)
                signature = {
                    "type": gmsh.model.getType(2, face),
                    "area": round(occ.getMass(2, face), 13),
                    "center": [round(v, 11) for v in occ.getCenterOfMass(2, face)],
                    "bounds": [round(v, 11) for v in occ.getBoundingBox(2, face)],
                    "orientation": 1 if signed > 0 else -1,
                }
                key = root + ":" + _digest(signature)
                if key in signatures:
                    raise ValueError(
                        "Coincident CAD faces have ambiguous identities. Separate or subtract the objects."
                    )
                signatures.add(key)
                faces.append(
                    dict(
                        key=key,
                        body=root,
                        label=f"{names[root]} · {signature['type']} {len(faces) + 1}",
                        tag=face,
                        sign=1 if signed > 0 else -1,
                        area_m2=signature["area"],
                        excluded=key in model._value["excluded_faces"],
                    )
                )
    missing = sorted(set(model._value["excluded_faces"]) - signatures)
    return faces, missing


def _mesh_arrays(gmsh, faces):
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    tags, xyz = np.asarray(tags), np.asarray(xyz).reshape(-1, 3)
    order = np.argsort(tags)
    tags, xyz = tags[order], xyz[order]
    triangles, face_indices = [], []
    for index, face in enumerate(faces):
        types, _, blocks = gmsh.model.mesh.getElements(2, face["tag"])
        for type_, block in zip(types, blocks):
            if type_ != 2:
                raise ValueError("The mesher returned a non-triangular or higher-order surface element.")
            tri = np.searchsorted(tags, np.asarray(block).reshape(-1, 3))
            # Gmsh emits the oriented OCC face (including reversed caps and
            # cavity walls). Applying getBoundary's sign again reverses it twice.
            triangles.append(tri)
            face_indices.extend([index] * len(tri))
    if not triangles:
        raise ValueError("No triangles remain. Restore a face or choose a different surface.")
    triangles = np.vstack(triangles)
    if len(triangles) > MAX_TRIANGLES:
        raise ValueError(f"Mesh exceeds {MAX_TRIANGLES:,} triangles. Increase the edge length.")
    used, inverse = np.unique(triangles, return_inverse=True)
    return Mesh(xyz[used], inverse.reshape(-1, 3)), np.asarray(face_indices, dtype=int)


def audit_surface(mesh):
    """Topology/element checks; does not claim field accuracy or fabrication readiness."""
    v, f = mesh.vertices, mesh.faces
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area2 = np.linalg.norm(cross, axis=1)
    if not np.isfinite(v).all() or np.any(area2 <= np.finfo(float).eps * max(np.ptp(v, axis=0).max() ** 2, 1e-20)):
        raise ValueError("The surface contains nonfinite or degenerate triangles.")
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("The surface has non-manifold edges.")
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    balance = np.bincount(inverse, weights=directions)
    if np.any(balance[counts == 2] != 0):
        raise ValueError("The surface has inconsistent face orientations.")
    boundary = unique[counts == 1]
    if len(boundary):
        _, degree = np.unique(boundary, return_counts=True)
        if np.any(degree != 2):
            raise ValueError("The surface has a branching or pinched boundary.")
    mesh.face_normals = compute_face_normals(mesh)
    mesh.vertex_normals = compute_vertex_normals(mesh)
    mesh.boundaries = boundary_loops(mesh)
    lengths = np.stack([np.linalg.norm(v[f[:, (i + 1) % 3]] - v[f[:, i]], axis=1) for i in range(3)], axis=1)
    quality = 2 * math.sqrt(3) * area2 / np.sum(lengths**2, axis=1)
    components = len(split_disconnected_mesh(mesh))
    return dict(
        vertices=mesh.n_vertices,
        triangles=mesh.n_faces,
        components=components,
        boundary_loops=len(mesh.boundaries),
        area_m2=float(area2.sum() / 2),
        min_triangle_quality=float(quality.min()),
    )


def build_surface(model: SurfaceModel, *, final=False):
    """Rebuild CAD and tessellate. Excluded faces remain selectable in the preview.

    Final meshing removes volume ownership and excluded faces first. It is a
    fresh CAD surface mesh, never a filtered copy of display triangles.
    """
    with _kernel() as gmsh:
        faces, missing = _build(gmsh, model)
        if final and missing:
            raise ValueError(
                "Face selections changed after a geometry edit. Restore/reselect the affected faces before meshing."
            )
        retained = [f for f in faces if not f["excluded"]]
        if final and not retained:
            raise ValueError("No retained faces. Restore at least one face before meshing.")
        edge = model._value["edge_length_m"]
        if not final:
            bounds = np.asarray([gmsh.model.getBoundingBox(2, f["tag"]) for f in faces])
            edge = max(float(np.max(bounds[:, 3:].max(axis=0) - bounds[:, :3].min(axis=0))) / 22, 1e-5)
        area = sum(f["area_m2"] for f in (retained if final else faces))
        if area / (0.3 * edge**2) > MAX_TRIANGLES:
            raise ValueError("Requested mesh is too large. Increase the target edge length.")
        if final:
            gmsh.model.occ.remove(gmsh.model.getEntities(3), recursive=False)
            gmsh.model.occ.remove([(2, f["tag"]) for f in faces if f["excluded"]], recursive=True)
            gmsh.model.occ.synchronize()
            faces = retained
        # The private model owns these per-entity sizes; global mesh-size bounds
        # are restored even when the caller already owns a Gmsh session.
        old = {k: gmsh.option.getNumber(k) for k in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax")}
        try:
            gmsh.option.setNumber("Mesh.MeshSizeMin", edge)
            gmsh.option.setNumber("Mesh.MeshSizeMax", edge)
            gmsh.model.mesh.generate(2)
            mesh, indices = _mesh_arrays(gmsh, faces)
        finally:
            for key, val in old.items():
                gmsh.option.setNumber(key, val)
        report = audit_surface(mesh) if final else None
        if final:
            # Keep interactive meshing independent of first-use JIT compilation.
            collisions = accelerated_mesh_collision_diagnostics(mesh, backend="numpy")
            if collisions["sweep_mesh_self_intersection"]:
                raise ValueError(
                    "The retained surface intersects itself. Subtract overlapping objects, "
                    "remove unwanted faces, or move the objects apart before meshing."
                )
            report["self_intersections"] = 0
        public = [{k: v for k, v in f.items() if k not in {"tag", "sign"}} for f in faces]
        scene = dict(
            identity=model.identity,
            model=model.to_dict(),
            roots=model.roots,
            vertices=mesh.vertices.tolist(),
            triangles=mesh.faces.tolist(),
            face_indices=indices.tolist(),
            faces=public,
            unresolved_faces=missing,
            mesh_report=report,
            final=final,
        )
        if final:
            mesh.metadata.update(
                geometry_type="cad_surface",
                units="SI",
                surface_model=model.to_dict(),
                surface_model_identity=model.identity,
                surface_mesh_digest=mesh_digest(mesh),
                mesher="Gmsh " + gmsh.__version__,
                surface_mesh_report=report,
            )
        return mesh, scene


def apply_surface(project, model, mesh, *, expected_fingerprint):
    if project.fingerprint != expected_fingerprint:
        raise ValueError("The project changed while the surface editor was open. Reopen it before applying.")
    if mesh.metadata.get("surface_model_identity") != model.identity or mesh.metadata.get(
        "surface_mesh_digest"
    ) != mesh_digest(mesh):
        raise ValueError("Build a fresh final mesh after editing the geometry.")
    report = audit_surface(mesh)
    changes = dict(surface_is_cylinder_flag=False, sweep_surface_realization="mesh")
    if report["components"] > 1:
        changes["routing_circuit_scope"] = "per_part"
    return project.with_mesh(mesh, field="coil_mesh_file", changes=changes)


class SurfaceEditor:
    """Atomic draft transactions shared by Qt and the browser transport."""

    def __init__(self, model=None):
        self.model = model or SurfaceModel.initial()
        self.scene = None
        self.mesh = None
        self.mesh_scene = None
        self.undo = []
        self.redo = []

    def preview(self):
        _, self.scene = build_surface(self.model)
        return self.snapshot()

    def edit(self, value):
        model = SurfaceModel(value)
        if model.identity == self.model.identity:
            return self.snapshot()
        _, scene = build_surface(model)
        self.undo.append(self.model)
        self.undo = self.undo[-40:]
        self.redo.clear()
        self.model, self.scene, self.mesh, self.mesh_scene = model, scene, None, None
        return self.snapshot()

    def travel(self, forward=False):
        source, destination = (self.redo, self.undo) if forward else (self.undo, self.redo)
        if source:
            model = source[-1]
            _, scene = build_surface(model)
            source.pop()
            destination.append(self.model)
            self.model, self.scene, self.mesh, self.mesh_scene = model, scene, None, None
        return self.snapshot()

    def generate(self):
        mesh, scene = build_surface(self.model, final=True)
        self.mesh, self.mesh_scene = mesh, scene
        return self.snapshot()

    def snapshot(self):
        if self.scene is None:
            return None
        return dict(
            self.scene,
            undo=bool(self.undo),
            redo=bool(self.redo),
            mesh_preview=self.mesh_scene,
            can_apply=self.mesh is not None,
        )
