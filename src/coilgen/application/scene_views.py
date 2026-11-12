"""View helpers shared by the desktop and browser viewports.

Open support edges, the Primary / Shield body filter, camera presets named by
the direction they look from, and the region-of-interest bounds of a field map.
Everything here changes what is drawn, never the saved data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from ..part_roles import ACTIVE_SHIELD, PRIMARY, normalize_part_role
from .evidence import FIELD_KEYS, PATH_PREFIXES, field_points_name

OUTLINE_COLOR = "#b4651e"  # copper: open edges (cylinder ends and windows)
HIGHLIGHT_COLOR = "#e87525"

# Camera presets: name -> (direction from the focus towards the camera, view up).
DEFAULT_VIEW = "Oblique"
VIEW_PRESETS = {
    "Oblique": ((1.0, 0.6, 0.4), (0.0, 0.0, 1.0)),
    "From +x (YZ)": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "From −x (YZ)": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "From +y (XZ)": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "From −y (XZ)": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "From +z (XY)": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}
# Names used before the presets said where they look from.
LEGACY_VIEW_NAMES = {"Isometric": "Oblique", "YZ": "From +x (YZ)", "XZ": "From −y (XZ)", "XY": "From +z (XY)"}

BOTH = "both"
BODY_VIEWS = ((BOTH, "Both"), (PRIMARY, "Primary"), (ACTIVE_SHIELD, "Shield"))
_PER_VERTEX = ("stream_function_A",)
_PER_FACE = ("sheet_current_density_A_per_m", "sheet_current_density_points_m")
_PATH = re.compile("(?:" + "|".join(PATH_PREFIXES.values()) + r")\d+(?:_\d+)?_points_m")
_SURFACES = ("swept_", "grooved_support_", "review_grooved_support_", "former_preview_")


# -- camera ---------------------------------------------------------------
def view_preset_name(name) -> str | None:
    """Current preset name for ``name``; older plane names ("YZ", "Isometric") still work."""
    name = LEGACY_VIEW_NAMES.get(name, name)
    return name if name in VIEW_PRESETS else None


def apply_view_preset(plotter, name, bounds=None) -> bool:
    """Point a pyvista plotter along a preset and fit ``bounds`` (default: the whole scene)."""
    name = view_preset_name(name)
    if name is None:
        return False
    vector, up = VIEW_PRESETS[name]
    plotter.view_vector(vector, viewup=up, bounds=bounds)
    return True


def roi_bounds(arrays, key, *, margin=0.08):
    """Bounds of a field map's region of interest, without shielding samples.

    Shielding (suppression) samples are flagged by ``target_suppressed``; they
    sit far outside the imaging region, so fitting to them shows only the coil.
    Returns ``(xmin, xmax, ymin, ymax, zmin, zmax)`` or ``None``.
    """
    if key not in FIELD_KEYS:
        return None
    points = arrays.get(field_points_name(arrays, key))
    if points is None:
        return None
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return None
    suppressed = arrays.get("target_suppressed")
    if suppressed is not None:
        suppressed = np.asarray(suppressed).astype(bool).reshape(-1)
        if suppressed.shape == (len(points),) and suppressed.any() and not suppressed.all():
            points = points[~suppressed]
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return None
    low, high = points.min(axis=0), points.max(axis=0)
    pad = max(float(np.max(high - low)) * margin, 1e-3)
    return tuple(float(value) for axis in range(3) for value in (low[axis] - pad, high[axis] + pad))


def scene_diagonal(bounds) -> float:
    values = np.asarray(bounds, dtype=float).reshape(3, 2)
    return float(np.linalg.norm(values[:, 1] - values[:, 0]))


def witness_bounds(points, *, minimum=0.0025):
    """Bounds around located evidence, the same framing the desktop uses."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points)):
        return None
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low)) * 0.625, minimum)
    return tuple(float(value) for axis in range(3) for value in (center[axis] - radius, center[axis] + radius))


def valid_witness(points, segments):
    """``(points, segments)`` as arrays when they form a drawable witness, else ``None``."""
    try:
        points = np.asarray(points, dtype=float)
        segments = [tuple(int(index) for index in pair) for pair in segments or ()]
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.all(np.isfinite(points)):
        return None
    if any(len(pair) != 2 or not all(0 <= index < len(points) for index in pair) for pair in segments):
        return None
    return points, segments


# -- open edges -----------------------------------------------------------
def boundary_loops(faces) -> list[np.ndarray]:
    """Vertex-index loops along edges used by exactly one triangle (closed loops repeat their start)."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        return []
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    adjacency = defaultdict(list)
    for a, b in boundary.tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)
    used, loops = set(), []
    for start, step in boundary.tolist():
        if (start, step) in used:
            continue
        used.add((start, step))
        loop, current = [start, step], step
        while current != start:
            step = next((n for n in adjacency[current] if (min(current, n), max(current, n)) not in used), None)
            if step is None:
                break
            used.add((min(current, step), max(current, step)))
            loop.append(step)
            current = step
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def outline_paths(vertices, faces) -> tuple[np.ndarray, ...]:
    """Open support edges as point paths in metres: cylinder ends and window outlines."""
    vertices = np.asarray(vertices, dtype=float)
    return tuple(vertices[loop] for loop in boundary_loops(faces) if len(loop) > 1)


def _encircles_axis(points, center_xy) -> bool:
    angles = np.unwrap(np.arctan2(points[:, 1] - center_xy[1], points[:, 0] - center_xy[0]))
    return abs(float(angles[-1] - angles[0])) > np.pi


def opening_loops(vertices, faces) -> tuple[np.ndarray, ...]:
    """Edges of holes cut into the support: loops that do not go round its z axis.

    The two ends of a cylinder encircle the axis; a window does not.
    """
    vertices = np.asarray(vertices, dtype=float)
    if not len(vertices):
        return ()
    center = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    return tuple(path for path in outline_paths(vertices, faces) if not _encircles_axis(path, center))


# -- bodies ---------------------------------------------------------------
def body_labels(faces, vertex_count) -> tuple[np.ndarray, int]:
    """Body index per vertex (-1 when unused), numbered like ``geometry.split_disconnected_mesh``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces, dtype=np.int64)
    labels = np.full(int(vertex_count), -1, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        return labels, 0
    rows = faces[:, [0, 1, 2]].reshape(-1)
    cols = faces[:, [1, 2, 0]].reshape(-1)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(labels), len(labels)))
    _, components = connected_components(graph, directed=False)
    # Pipeline order: by the first triangle of each body.
    face_components = components[faces[:, 0]]
    found, first = np.unique(face_components, return_index=True)
    order = found[np.argsort(first)]
    rank = np.full(components.max() + 1, -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    used = np.zeros(len(labels), dtype=bool)
    used[faces.reshape(-1)] = True
    labels[used] = rank[components[used]]
    return labels, len(order)


def _support(arrays):
    if "vertices_m" not in arrays or "faces" not in arrays:
        return None
    vertices = np.asarray(arrays["vertices_m"], dtype=float)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        return None
    return vertices, faces


def saved_part_roles(metadata) -> dict[int, str]:
    """Part index -> role, from the conductor bodies a saved run recorded (``part001_body0001``)."""
    from .support_import import body_roles

    roles = {}
    for body_id, role in body_roles(metadata or {}).items():
        match = re.search(r"part(\d+)", body_id)
        if match and int(match.group(1)) >= 1:
            roles.setdefault(int(match.group(1)) - 1, role)
    return roles


def support_roles(arrays, roles=None, metadata=None) -> list[str]:
    """One role per support body: recorded roles first, else inner radius primary / outer shield."""
    from .support_import import default_part_roles

    support = _support(arrays)
    if support is None:
        return []
    vertices, faces = support
    labels, count = body_labels(faces, len(vertices))
    if isinstance(roles, (list, tuple)) and len(roles) == count:
        normalized = [normalize_part_role(role) for role in roles]
        if None not in normalized:
            return normalized
    recorded = saved_part_roles(metadata)
    if count and all(index in recorded for index in range(count)):
        return [recorded[index] for index in range(count)]
    bodies = [SimpleNamespace(vertices=vertices[labels == index]) for index in range(count)]
    return default_part_roles(bodies)


def body_view_choices(roles) -> list[dict]:
    """Primary / Shield / Both, offered only when the support has both roles."""
    roles = set(roles or ())
    if not {PRIMARY, ACTIVE_SHIELD} <= roles:
        return []
    return [{"value": value, "title": title} for value, title in BODY_VIEWS]


def _filter_surface(vertices, faces, keep):
    vertex_keep = np.asarray(keep, dtype=bool)
    face_keep = vertex_keep[faces].all(axis=1) if len(faces) else np.zeros(0, dtype=bool)
    index = np.cumsum(vertex_keep) - 1
    return vertices[vertex_keep], index[faces[face_keep]], vertex_keep, face_keep


def filter_body_view(arrays, view, roles):
    """Arrays with only the bodies of ``view`` ("primary", "active_shield" or "both").

    The support, stream function and current density are cut to the chosen
    bodies; conductor paths and solids are kept when they lie on them. Field
    samples are left as they are: the field is the field of the whole coil.
    """
    support = _support(arrays)
    if view in (None, "", BOTH) or support is None:
        return arrays
    vertices, faces = support
    labels, count = body_labels(faces, len(vertices))
    if len(roles or ()) != count:
        return arrays
    wanted = [index for index, role in enumerate(roles) if role == view]
    if not wanted or len(wanted) == count:
        return arrays
    from scipy.spatial import cKDTree

    keep_vertex = np.isin(labels, wanted)
    out = dict(arrays)
    out["vertices_m"], out["faces"], _, face_keep = _filter_surface(vertices, faces, keep_vertex)
    for name in _PER_VERTEX:
        if name in arrays and len(arrays[name]) == len(vertices):
            out[name] = np.asarray(arrays[name])[keep_vertex]
    for name in _PER_FACE:
        if name in arrays and len(arrays[name]) == len(faces):
            out[name] = np.asarray(arrays[name])[face_keep]
    used = labels >= 0
    tree, owners = cKDTree(vertices[used]), labels[used]

    def on_wanted(points):
        _, nearest = tree.query(points)
        return np.isin(owners[nearest], wanted)

    for name in list(out):
        if _PATH.fullmatch(name):
            path = np.asarray(out[name], dtype=float)
            if path.ndim == 2 and path.shape[1] == 3 and len(path):
                sample = path[np.linspace(0, len(path) - 1, min(len(path), 64)).astype(int)]
                if np.count_nonzero(on_wanted(sample)) * 2 <= len(sample):
                    del out[name]
        elif name.startswith(_SURFACES) and name.endswith("_vertices_m"):
            stem = name.removesuffix("_vertices_m")
            body = np.asarray(out[name], dtype=float)
            body_faces = out.get(stem + "_faces")
            if body_faces is None or body.ndim != 2 or body.shape[1] != 3 or not len(body):
                continue
            kept, kept_faces, _, _ = _filter_surface(body, np.asarray(body_faces, dtype=np.int64), on_wanted(body))
            if len(kept_faces):
                out[name], out[stem + "_faces"] = kept, kept_faces
            else:
                del out[name], out[stem + "_faces"]
    return out


__all__ = [
    "BODY_VIEWS",
    "BOTH",
    "DEFAULT_VIEW",
    "HIGHLIGHT_COLOR",
    "LEGACY_VIEW_NAMES",
    "OUTLINE_COLOR",
    "VIEW_PRESETS",
    "apply_view_preset",
    "body_labels",
    "body_view_choices",
    "boundary_loops",
    "filter_body_view",
    "opening_loops",
    "outline_paths",
    "roi_bounds",
    "saved_part_roles",
    "scene_diagonal",
    "support_roles",
    "valid_witness",
    "view_preset_name",
    "witness_bounds",
]
