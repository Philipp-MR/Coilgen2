"""Unwrapped (φ, z) view of cylindrical supports, shared by both workbenches.

Each cylinder (primary, shield) is cut along φ = ±180° and laid flat, side by
side: stream function as colour, contours as lines, openings in copper. This is
the clearest view of windows and the wires that bend around them. Matplotlib
only, like the shielding plot; nothing is recomputed.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np

from ..part_roles import ACTIVE_SHIELD, PRIMARY, part_role_label
from .evidence import path_array_names
from .scene_views import OUTLINE_COLOR, body_labels, opening_loops, support_roles

# Relative spread of the radius about the z axis below which a body counts as a cylinder.
CYLINDER_TOLERANCE = 0.03
_ROLE_ORDER = {PRIMARY: 0, ACTIVE_SHIELD: 1}


def _support(arrays):
    if "vertices_m" not in arrays or "faces" not in arrays:
        return None
    vertices = np.asarray(arrays["vertices_m"], dtype=float)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        return None
    return vertices, faces


def cylinder_bodies(arrays):
    """``[(vertex indices, mean radius m)]`` per body, or ``[]`` unless every body is a z-axis cylinder."""
    support = _support(arrays)
    if support is None:
        return []
    vertices, faces = support
    labels, count = body_labels(faces, len(vertices))
    center = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    radius = np.linalg.norm(vertices[:, :2] - center, axis=1)
    bodies = []
    for index in range(count):
        members = np.flatnonzero(labels == index)
        mean = float(radius[members].mean()) if len(members) else 0.0
        span = float(np.ptp(vertices[members, 2])) if len(members) else 0.0
        if mean <= 0 or span <= 0 or float(radius[members].std()) > CYLINDER_TOLERANCE * mean:
            return []
        # A cylinder goes most of the way round: at least 27 of 36 ten-degree sectors.
        phi = np.arctan2(vertices[members, 1] - center[1], vertices[members, 0] - center[0])
        if len(np.unique(np.floor((phi + np.pi) / (2 * np.pi) * 36))) < 27:
            return []
        bodies.append((members, mean))
    return bodies


def is_cylindrical(arrays) -> bool:
    return bool(cylinder_bodies(arrays))


def _angles(points, center):
    return np.degrees(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))


def _break_seam(phi, z):
    """Insert gaps where a line crosses φ = ±180°, so it is not drawn across the map."""
    jumps = np.flatnonzero(np.abs(np.diff(phi)) > 180) + 1
    return np.insert(phi, jumps, np.nan), np.insert(z, jumps, np.nan)


def draw_unwrapped(figure, arrays, *, roles=None, metadata=None, title=None) -> str:
    """Draw one φ–z panel per cylinder; returns a one-line note."""
    from matplotlib.colors import Normalize

    bodies = cylinder_bodies(arrays)
    if not bodies:
        raise ValueError("The unwrapped view needs cylinders about the z axis.")
    vertices, faces = _support(arrays)
    labels, _ = body_labels(faces, len(vertices))
    roles = support_roles(arrays, roles, metadata)
    center = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    phi, z_mm = _angles(vertices, center), vertices[:, 2] * 1e3
    stream = arrays.get("stream_function_A")
    stream = np.asarray(stream, dtype=float).reshape(-1) if stream is not None else None
    if stream is not None and len(stream) != len(vertices):
        stream = None
    key = "contours" if path_array_names(arrays, "contours") else "routes"
    paths = [np.asarray(arrays[name], dtype=float) for name in path_array_names(arrays, key)]
    path_bodies = []
    if paths:
        from scipy.spatial import cKDTree

        used = labels >= 0
        tree, owners = cKDTree(vertices[used]), labels[used]
        path_bodies = [int(np.bincount(owners[tree.query(path)[1]]).argmax()) for path in paths]
    norm = None
    if stream is not None:
        limit = float(np.max(np.abs(stream))) or 1.0
        norm = Normalize(-limit, limit)
    order = sorted(range(len(bodies)), key=lambda index: (_ROLE_ORDER.get(roles[index], 2), index))
    figure.clear()
    axes = figure.subplots(1, len(order), sharey=True, squeeze=False)[0]
    mesh_artist = None
    openings = 0
    for axis, index in zip(axes, order):
        members, radius = bodies[index]
        body_faces = faces[labels[faces[:, 0]] == index]
        if stream is not None and len(body_faces):
            face_phi = phi[body_faces].copy()
            seam = np.ptp(face_phi, axis=1) > 180
            face_phi[seam] = np.where(face_phi[seam] < 0, face_phi[seam] + 360, face_phi[seam])
            triangles = np.arange(face_phi.size).reshape(-1, 3)
            mesh_artist = axis.tripcolor(
                face_phi.reshape(-1),
                z_mm[body_faces].reshape(-1),
                stream[body_faces].reshape(-1),
                triangles=triangles,
                shading="gouraud",
                cmap="coolwarm",
                norm=norm,
            )
        for path, owner in zip(paths, path_bodies):
            if owner == index:
                axis.plot(*_break_seam(_angles(path, center), path[:, 2] * 1e3), color="#1f2933", linewidth=0.8)
        body_mask = np.zeros(len(vertices), dtype=bool)
        body_mask[members] = True
        for loop in opening_loops(vertices[body_mask], _local_faces(body_faces, body_mask)):
            axis.plot(*_break_seam(_angles(loop, center), loop[:, 2] * 1e3), color=OUTLINE_COLOR, linewidth=2.4)
            openings += 1
        if stream is None and len(members):
            # Without a stream function, shade the cylinder so its length reads at a glance.
            low, high = float(z_mm[members].min()), float(z_mm[members].max())
            axis.fill_between([-180, 180], low, high, color="#e6ebef", zorder=0, linewidth=0)
        label = part_role_label(roles[index]) or f"Body {index + 1}"
        axis.set_title(f"{label} · r {radius * 1e3:.4g} mm", fontsize=10)
        axis.set_xlim(-180, 180)
        axis.set_xticks(range(-180, 181, 90))
        axis.set_xlabel("φ [°]")
        axis.grid(True, color="#dddddd", linewidth=0.5)
    # Span every cylinder's full length, not only what was drawn on it (openings alone are short).
    all_members = np.concatenate([members for members, _ in bodies])
    z_low, z_high = float(z_mm[all_members].min()), float(z_mm[all_members].max())
    margin = 0.03 * max(z_high - z_low, 1e-6)
    axes[0].set_ylim(z_low - margin, z_high + margin)
    axes[0].set_ylabel("z [mm]")
    if title:
        figure.suptitle(title, fontsize=11)
    if mesh_artist is not None:
        bar = figure.colorbar(mesh_artist, ax=list(axes), shrink=0.85, pad=0.02)
        bar.set_label("Stream function [A]")
    parts = [
        f"{len(order)} cylinder{'s' if len(order) != 1 else ''}",
        f"{openings} opening{'s' if openings != 1 else ''}",
    ]
    parts.append("stream function shown" if stream is not None else "no stream function in this view")
    if paths:
        parts.append(f"{len(paths)} {'contours' if key == 'contours' else 'routes'}")
    return " · ".join(parts) + " · cut at φ = ±180°, seen from outside"


def _local_faces(body_faces, body_mask):
    index = np.cumsum(body_mask) - 1
    return index[body_faces]


def unwrapped_png(arrays, *, roles=None, metadata=None, title=None, dpi=110) -> tuple[bytes, str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 5), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    note = draw_unwrapped(figure, arrays, roles=roles, metadata=metadata, title=title)
    buffer = BytesIO()
    canvas.print_png(buffer)
    return buffer.getvalue(), note


def unwrapped_image(arrays, *, roles=None, metadata=None, title=None) -> tuple[str, str]:
    """PNG as a ``data:`` URL for the browser, and the note."""
    data, note = unwrapped_png(arrays, roles=roles, metadata=metadata, title=title)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii"), note
