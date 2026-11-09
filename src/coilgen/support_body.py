"""Grooved support bodies: a thickened support surface with conductor channels.

The support surface of each coil part is offset along its vertex normals into a
closed slab of ``support_body_thickness``. The realized conductor paths are then
swept with a groove profile (the conductor profile grown by insulation and
``support_body_groove_clearance``, extended straight out through the slab face
so the wire can be laid in) and subtracted. The result is a directly printable
former.

Booleans use the optional ``manifold3d`` package. Each groove is the union of
convex hulls of consecutive profile rings, so tight bends or overlapping path
sections never hand a self-intersecting cutter to the boolean kernel.

The solid side is defined relative to the conductor frame normals, i.e. the
same normal that ``conductor_mounting_mode`` uses:

``inside``
    The slab spans ``[-t, 0]`` along the normal. ``surface_inner`` conductors
    are fully recessed, ``centered`` ones half, ``surface_outer`` ones not at all.
``outside``
    The slab spans ``[0, t]``; the roles of inner and outer mounting swap.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .geometry import compute_face_normals
from .models import CoilPart, FabricationPath, Mesh, SupportBody

SUPPORT_BODY_MODES = ("off", "inside", "outside")
_PROFILE_ROUNDING_SAMPLES = 12
_HULL_OVERLAP_FRACTION = 1e-3
_MIN_SLAB_VOLUME_RATIO = 0.8
_MAX_SLAB_VOLUME_RATIO = 1.25


class SupportBodyError(ValueError):
    """The grooved support body cannot be constructed."""


def support_body_mode(config) -> str:
    text = str(getattr(config, "support_body_mode", "off") or "off").strip().lower().replace("-", "_")
    if text not in SUPPORT_BODY_MODES:
        raise SupportBodyError(f"Unsupported support_body_mode {text!r}; use one of {', '.join(SUPPORT_BODY_MODES)}.")
    return text


def support_body_enabled(config) -> bool:
    return support_body_mode(config) != "off"


def boolean_backend_available() -> bool:
    try:
        import manifold3d  # noqa: F401
    except ImportError:
        return False
    return True


def _manifold():
    try:
        import manifold3d
    except ImportError as exc:  # pragma: no cover - exercised through monkeypatching
        raise SupportBodyError(
            "The grooved support body needs the optional 'manifold3d' package (python -m pip install manifold3d)."
        ) from exc
    return manifold3d


# --------------------------------------------------------------------------- profiles
def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Counterclockwise convex hull (Andrew's monotone chain), collinear points dropped."""

    unique = np.unique(np.round(np.asarray(points, dtype=float), 15), axis=0)
    if len(unique) < 3:
        raise SupportBodyError("Groove profile is degenerate.")
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    pts = unique[order]
    scale = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-15)
    tolerance = 1e-12 * scale * scale

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise SupportBodyError("Groove profile is degenerate.")
    return hull


def _grown_profile(profile: np.ndarray, radius: float) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    if radius <= 0.0:
        return _convex_hull_2d(profile)
    angles = np.linspace(0.0, 2.0 * np.pi, _PROFILE_ROUNDING_SAMPLES, endpoint=False)
    circle = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return _convex_hull_2d((profile[:, None, :] + circle[None, :, :]).reshape(-1, 2))


def _mounting_and_reference(cross_section: np.ndarray, config) -> tuple[float, np.ndarray]:
    from .manufacturing import _mounting_centerline_offset, _path_reference_point

    return (
        float(_mounting_centerline_offset(config, cross_section)),
        np.asarray(_path_reference_point(cross_section, config), dtype=float),
    )


def groove_cutter_profile(cross_section: np.ndarray, config) -> tuple[np.ndarray, dict[str, float]]:
    """Return the groove profile in support coordinates ``(transverse, normal)``.

    Coordinates are relative to the support point under the conductor: ``w = 0``
    is the support surface (the slab face where the grooves open). The profile is
    the convex hull of the conductor profile grown by insulation + clearance,
    extended straight out through the face and deepened by the extra depth.
    """

    mode = support_body_mode(config)
    if mode == "off":
        raise SupportBodyError("support_body_mode is 'off'.")
    cross_section = np.asarray(cross_section, dtype=float)
    mounting, _ = _mounting_and_reference(cross_section, config)
    radius = float(getattr(config, "sweep_insulation_thickness", 0.0)) + float(
        getattr(config, "support_body_groove_clearance", 0.0)
    )
    thickness = float(config.support_body_thickness)
    extra = float(getattr(config, "support_body_groove_extra_depth", 0.0))
    grown = _grown_profile(cross_section, radius) + np.asarray([0.0, mounting])
    # Opening direction: away from the solid.
    opening = 1.0 if mode == "inside" else -1.0
    w = opening * grown[:, 1]
    # Move a copy fully past the face so the channel walls run straight out.
    lift = max(0.0, -float(np.min(w))) + 0.5 * thickness + radius
    parts = [grown, grown + np.asarray([0.0, opening * lift])]
    if extra > 0.0:
        parts.append(grown - np.asarray([0.0, opening * extra]))
    profile = _convex_hull_2d(np.vstack(parts))
    into_solid = -opening * profile[:, 1]
    depth = float(np.max(into_solid))
    width_at_face = _profile_width_at(profile, 0.0)
    return profile, {
        "support_groove_depth_m": depth,
        "support_groove_floor_thickness_m": thickness - depth,
        "support_groove_width_at_face_m": width_at_face,
        "support_groove_growth_m": radius,
        "support_groove_mounting_offset_m": mounting,
    }


def _profile_width_at(profile: np.ndarray, w: float) -> float:
    xs = []
    count = len(profile)
    for index in range(count):
        a, b = profile[index], profile[(index + 1) % count]
        if (a[1] - w) * (b[1] - w) <= 0.0 and a[1] != b[1]:
            t = (w - a[1]) / (b[1] - a[1])
            xs.append(a[0] + t * (b[0] - a[0]))
    return float(max(xs) - min(xs)) if len(xs) >= 2 else 0.0


# --------------------------------------------------------------------------- cutters
def _support_points(path: FabricationPath, cross_section: np.ndarray, config) -> np.ndarray:
    mounting, reference = _mounting_and_reference(cross_section, config)
    return (
        np.asarray(path.points_m, dtype=float)
        - (mounting + reference[1]) * np.asarray(path.normals, dtype=float)
        - reference[0] * np.asarray(path.transverse, dtype=float)
    )


def groove_cutter_rings(path: FabricationPath, config) -> np.ndarray:
    """``(n_rings, n_profile, 3)`` groove-profile rings along a realized path."""

    cross_section = np.asarray(path.cross_section_m, dtype=float)
    profile, _ = groove_cutter_profile(cross_section, config)
    base = _support_points(path, cross_section, config)
    return (
        base[:, None, :]
        + profile[None, :, 0, None] * np.asarray(path.transverse)[:, None, :]
        + profile[None, :, 1, None] * np.asarray(path.normals)[:, None, :]
    )


def _cutter_manifold(path: FabricationPath, config):
    m3d = _manifold()
    rings = groove_cutter_rings(path, config)
    count = len(rings)
    pairs = [(index, index + 1) for index in range(count - 1)]
    if path.closed and count > 2:
        pairs.append((count - 1, 0))
    # Neighbouring hulls only touching at a shared ring stay separate solids under
    # symbolic perturbation, so each piece is stretched a hair past its rings.
    # Open-path ends are left exact.
    extent = float(np.max(np.ptp(rings[0], axis=0)))
    overlap = _HULL_OVERLAP_FRACTION * max(extent, 1e-9)
    centers = rings.mean(axis=1)
    last = count - 1
    hulls = []
    for a, b in pairs:
        direction = centers[b] - centers[a]
        length = float(np.linalg.norm(direction))
        direction = direction / length if length > 0.0 else np.asarray(path.tangents[a], dtype=float)
        start = rings[a] - (0.0 if a == 0 and not path.closed else overlap) * direction
        end = rings[b] + (0.0 if b == last and not path.closed else overlap) * direction
        hull = m3d.Manifold.hull_points(np.array(np.vstack((start, end)), dtype=np.float64, order="C"))
        if not hull.is_empty():
            hulls.append(hull)
    if not hulls:
        return m3d.Manifold()
    return m3d.Manifold.batch_boolean(hulls, m3d.OpType.Add)


# --------------------------------------------------------------------------- slab
def _boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    directed: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(int(u), int(v))] += 1
    undirected: dict[tuple[int, int], int] = defaultdict(int)
    for (u, v), count in directed.items():
        undirected[(min(u, v), max(u, v))] += count
    return [(u, v) for (u, v) in directed if undirected[(min(u, v), max(u, v))] == 1]


def _angle_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    face_normals = np.divide(
        face_normals, lengths[:, None], out=np.zeros_like(face_normals), where=lengths[:, None] > 0
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        e1 = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        e2 = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        denominator = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        cosine = np.divide(np.einsum("ij,ij->i", e1, e2), denominator, out=np.ones(len(faces)), where=denominator > 0)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        np.add.at(normals, faces[:, corner], face_normals * angle[:, None])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0)
    return normals, face_normals


def _offset_distances(vertex_normals: np.ndarray, face_normals: np.ndarray, faces: np.ndarray, thickness: float):
    """Offset per vertex so the wall stays ``thickness`` thick next to creases."""

    worst = np.ones(len(vertex_normals))
    for corner in range(3):
        cosine = np.einsum("ij,ij->i", vertex_normals[faces[:, corner]], face_normals)
        np.minimum.at(worst, faces[:, corner], cosine)
    return thickness / np.clip(worst, 0.5, 1.0)


def _frame_normal_agreement(support: Mesh, vertex_normals: np.ndarray, paths: list[FabricationPath]) -> float:
    """Sign that maps support vertex normals onto the conductor frame normals.

    Only path samples lying on this support vote, so conductors on another part
    (the opposite plate of a biplanar pair, a shield) cannot flip the side.
    """

    vertices = np.asarray(support.vertices, dtype=float)
    faces = np.asarray(support.faces, dtype=np.int64)
    edges = vertices[faces] - vertices[np.roll(faces, -1, axis=1)]
    longest_edge = float(np.max(np.linalg.norm(edges, axis=2))) if len(faces) else 0.0
    total = 0.0
    for path in paths:
        step = max(1, len(path.points_m) // 64)
        points = np.asarray(path.points_m, dtype=float)[::step]
        normals = np.asarray(path.normals, dtype=float)[::step]
        reach = longest_edge + 2.0 * float(np.max(np.ptp(np.asarray(path.cross_section_m, dtype=float), axis=0)))
        for start in range(0, len(points), 128):
            chunk = points[start : start + 128]
            distances = np.linalg.norm(chunk[:, None, :] - vertices[None, :, :], axis=2)
            nearest = np.argmin(distances, axis=1)
            near = distances[np.arange(len(chunk)), nearest] <= reach
            dots = np.einsum("ij,ij->i", vertex_normals[nearest], normals[start : start + 128])
            total += float(dots[near].sum())
    return 1.0 if total >= 0.0 else -1.0


def thickened_support_mesh(support: Mesh, thickness: float, direction: float) -> tuple[Mesh, dict]:
    """Close ``support`` into a slab offset by ``direction * thickness`` along its vertex normals.

    Open surfaces get side walls along every boundary loop; closed surfaces
    become a hollow shell. Faces are oriented outward (positive volume).
    """

    vertices = np.asarray(support.vertices, dtype=float)
    faces = np.asarray(support.faces, dtype=np.int64)
    if not len(faces):
        raise SupportBodyError("Support mesh has no triangles.")
    from .manufacturing import _edge_topology_counts, _signed_mesh_volume

    open_edges, nonmanifold, misoriented = _edge_topology_counts(faces)
    if nonmanifold or misoriented:
        raise SupportBodyError(
            f"Support mesh is not an oriented manifold surface ({nonmanifold} non-manifold, "
            f"{misoriented} inconsistently oriented edges); repair it before thickening."
        )
    vertex_normals, face_normals = _angle_weighted_vertex_normals(vertices, faces)
    if not np.all(np.linalg.norm(vertex_normals, axis=1) > 0.5):
        raise SupportBodyError("Support mesh has vertices without a defined normal.")
    distances = _offset_distances(vertex_normals, face_normals, faces, thickness)
    offset = vertices + direction * distances[:, None] * vertex_normals
    count = len(vertices)
    offset_face_normals = compute_face_normals(Mesh(vertices=offset, faces=faces))
    folded = int(np.count_nonzero(np.einsum("ij,ij->i", offset_face_normals, face_normals) <= 0.0))
    walls = []
    for a, b in _boundary_edges(faces):
        walls.append((b, a, a + count))
        walls.append((b, a + count, b + count))
    all_faces = np.vstack((faces, faces[:, ::-1] + count, np.asarray(walls, dtype=np.int64).reshape(-1, 3)))
    all_vertices = np.vstack((vertices, offset))
    volume = _signed_mesh_volume(all_vertices, all_faces)
    if volume < 0.0:
        all_faces = all_faces[:, ::-1].copy()
        volume = -volume
    triangles = vertices[faces]
    area = float(
        0.5
        * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum()
    )
    expected = area * thickness
    return Mesh(vertices=all_vertices, faces=all_faces, metadata={"role": "support_slab"}), {
        "support_surface_area_m2": area,
        "support_surface_open_edge_count": int(open_edges),
        "support_slab_volume_m3": float(volume),
        "support_slab_volume_ratio": float(volume / expected) if expected > 0 else 0.0,
        "support_slab_folded_face_count": folded,
    }


# --------------------------------------------------------------------------- boolean
def _to_manifold(mesh: Mesh):
    m3d = _manifold()
    data = m3d.Mesh64(
        # Fresh writable copies: the bindings reject read-only array views.
        vert_properties=np.array(mesh.vertices, dtype=np.float64, order="C"),
        tri_verts=np.array(mesh.faces, dtype=np.uint64, order="C"),
    )
    solid = m3d.Manifold(data)
    return solid, str(solid.status()).rsplit(".", 1)[-1]


def _from_manifold(solid) -> Mesh:
    data = solid.to_mesh64()
    vertices = np.asarray(data.vert_properties, dtype=float)[:, :3]
    faces = np.asarray(data.tri_verts, dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces, metadata={"role": "grooved_support"})


def _paths_of(part: CoilPart) -> list[FabricationPath]:
    paths = [body.fabrication_path for body in part.swept_bodies if body.fabrication_path is not None]
    if not paths and part.fabrication_path is not None:
        paths = [part.fabrication_path]
    return paths


def build_grooved_support(support: Mesh, paths: list[FabricationPath], config) -> SupportBody:
    """Thicken ``support`` and subtract grooves along every realized path in ``paths``."""

    from .manufacturing import mesh_manufacturing_diagnostics

    mode = support_body_mode(config)
    if mode == "off":
        raise SupportBodyError("support_body_mode is 'off'.")
    thickness = float(config.support_body_thickness)
    if not thickness > 0.0:
        raise SupportBodyError("support_body_thickness must be positive.")
    m3d = _manifold()
    vertex_normals, _ = _angle_weighted_vertex_normals(
        np.asarray(support.vertices, dtype=float), np.asarray(support.faces, dtype=np.int64)
    )
    agreement = _frame_normal_agreement(support, vertex_normals, paths)
    side = -1.0 if mode == "inside" else 1.0
    slab_mesh, diagnostics = thickened_support_mesh(support, thickness, side * agreement)
    slab, status = _to_manifold(slab_mesh)
    diagnostics.update(
        {
            "support_body_mode": mode,
            "support_body_thickness_m": thickness,
            "support_body_groove_clearance_m": float(getattr(config, "support_body_groove_clearance", 0.0)),
            "support_body_groove_extra_depth_m": float(getattr(config, "support_body_groove_extra_depth", 0.0)),
            "support_slab_status": status,
            "support_groove_path_count": int(len(paths)),
            "support_normal_agreement": float(agreement),
        }
    )
    reasons: list[str] = []
    if status != "NoError" or slab.is_empty():
        reasons.append(f"slab_not_manifold:{status}")
    if diagnostics["support_slab_folded_face_count"]:
        reasons.append("slab_offset_folds_thickness_exceeds_curvature_radius")
    ratio = diagnostics["support_slab_volume_ratio"]
    if not _MIN_SLAB_VOLUME_RATIO <= ratio <= _MAX_SLAB_VOLUME_RATIO:
        reasons.append(f"slab_volume_ratio_{ratio:.3f}")

    cutters = []
    depths, floors, widths = [], [], []
    for path in paths:
        _, profile_info = groove_cutter_profile(np.asarray(path.cross_section_m, dtype=float), config)
        depths.append(profile_info["support_groove_depth_m"])
        floors.append(profile_info["support_groove_floor_thickness_m"])
        widths.append(profile_info["support_groove_width_at_face_m"])
        cutter = _cutter_manifold(path, config)
        if not cutter.is_empty():
            cutters.append(cutter)
    if depths:
        diagnostics.update(
            {
                "support_groove_depth_m": float(max(depths)),
                "support_groove_floor_thickness_m": float(min(floors)),
                "support_groove_width_at_face_m": float(max(widths)),
            }
        )
    if not paths:
        reasons.append("no_conductor_paths")
    elif max(depths) <= 0.0:
        reasons.append("conductor_does_not_reach_support_body")
    elif min(floors) <= 0.0:
        reasons.append("grooves_cut_through_support_body")

    slab_volume = float(slab.volume()) if not slab.is_empty() else 0.0
    if cutters:
        groove = m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)
        result = slab - groove
        groove_volume = float(groove.volume())
    else:
        result, groove_volume = slab, 0.0
    result_volume = float(result.volume()) if not result.is_empty() else 0.0
    mesh = _from_manifold(result) if not result.is_empty() else Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
    removed = slab_volume - result_volume
    if paths and max(depths) > 0.0 and removed <= 1e-6 * max(slab_volume, 1e-30):
        reasons.append("grooves_remove_no_material")
    checks = mesh_manufacturing_diagnostics(mesh)
    if not checks.get("watertight") or not checks.get("positive_volume"):
        reasons.append("grooved_support_not_closed_solid")
    slab_pieces = len(slab.decompose()) if not slab.is_empty() else 0
    body_pieces = len(result.decompose()) if not result.is_empty() else 0
    if body_pieces > slab_pieces:
        # Grooves split the former, or left loose slivers between channels.
        reasons.append(f"grooves_split_support_into_{body_pieces}_pieces")
    diagnostics.update(
        {
            "support_slab_component_count": int(slab_pieces),
            "support_body_component_count": int(body_pieces),
            "support_slab_manifold_volume_m3": slab_volume,
            "support_groove_cutter_volume_m3": groove_volume,
            "support_groove_removed_volume_m3": float(removed),
            "support_body_volume_m3": result_volume,
            "support_body_genus": int(result.genus()) if not result.is_empty() else 0,
            "support_body_vertex_count": int(len(mesh.vertices)),
            "support_body_face_count": int(len(mesh.faces)),
            "support_body_watertight": bool(checks.get("watertight", False)),
            "support_body_open_edge_count": int(checks.get("open_edge_count", 0)),
            "support_body_nonmanifold_edge_count": int(checks.get("nonmanifold_edge_count", 0)),
            "support_body_ok": not reasons,
            "support_body_failure_reasons": ";".join(reasons),
            "support_body_boolean_backend": "manifold3d",
        }
    )
    mesh.metadata.update({"support_body_mode": mode, "support_body_thickness_m": thickness})
    return SupportBody(mesh=mesh, diagnostics=diagnostics)


def create_grooved_support_bodies(parts: list[CoilPart], config) -> list[CoilPart]:
    """Attach a grooved support body to every part (pipeline stage).

    Every part's slab is cut by the grooves of all parts, so a conductor that
    runs across another part's support is never buried in it.
    """

    if not support_body_enabled(config):
        for part in parts:
            part.support_body = None
        return parts
    paths = [path for part in parts for path in _paths_of(part)]
    for part in parts:
        try:
            part.support_body = build_grooved_support(part.mesh, paths, config)
        except SupportBodyError as exc:
            part.support_body = SupportBody(
                mesh=Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3))),
                diagnostics={
                    "support_body_mode": support_body_mode(config),
                    "support_body_ok": False,
                    "support_body_failure_reasons": str(exc),
                },
            )
    return parts


__all__ = [
    "SUPPORT_BODY_MODES",
    "SupportBodyError",
    "boolean_backend_available",
    "build_grooved_support",
    "create_grooved_support_bodies",
    "groove_cutter_profile",
    "groove_cutter_rings",
    "support_body_enabled",
    "support_body_mode",
    "thickened_support_mesh",
]
