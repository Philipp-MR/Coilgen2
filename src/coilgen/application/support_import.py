"""Support-STL import shared by both clients: what was read, and which body is which.

The pipeline splits a support into bodies with :func:`geometry.split_disconnected_mesh`
and tags them from ``mesh.metadata["part_roles"]`` in that order. This module uses
the same function, so the roles a user picks here land on the same bodies there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..part_roles import (
    ACTIVE_SHIELD,
    OTHER,
    PART_ROLE_LABELS,
    PART_ROLES,
    PRIMARY,
    body_display_name,
    normalize_part_role,
)

ROLE_CHOICES = tuple({"value": role, "title": PART_ROLE_LABELS[role]} for role in PART_ROLES)
# A support this small or large almost always means the wrong STL unit.
SMALLEST_PLAUSIBLE_M = 1e-3
LARGEST_PLAUSIBLE_M = 10.0
# Coaxial bodies form two radius groups when the gap between them is this
# fraction of the outer radius or more (a shield sits well outside its primary).
_RADIUS_GAP_FRACTION = 0.05


def read_support_stl(path, length_scale_to_m: float = 1.0):
    """Read an STL, scale it to metres and weld it exactly as the pipeline will."""
    from ..geometry import normalize_mesh, read_stl

    scale = _scale(length_scale_to_m)
    mesh = read_stl(Path(path))
    mesh.vertices = np.asarray(mesh.vertices, dtype=float) * scale
    mesh.metadata = {
        **(getattr(mesh, "metadata", None) or {}),
        "source_file": Path(path).name,
        "source_length_scale_to_m": scale,
        "units": "SI",
    }
    return normalize_mesh(mesh)


def _scale(value) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Set a positive, finite STL scale (0.001 for millimetres).") from exc
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Set a positive, finite STL scale (0.001 for millimetres).")
    return scale


def support_bodies(mesh):
    """The bodies in pipeline order (the function the pipeline itself uses)."""
    from ..geometry import split_disconnected_mesh

    return split_disconnected_mesh(mesh)


def _mean_radii(bodies, center_xy):
    return [float(np.mean(np.linalg.norm(body.vertices[:, :2] - center_xy, axis=1))) for body in bodies]


def default_part_roles(bodies, center_xy=None) -> list[str]:
    """Inner radius group is the primary, the outer group the active shield.

    Radii are measured about the z axis through the support's centre, which is
    where cylindrical gradient supports are built. Bodies without a clear radial
    gap between them are all primaries (for example split primary halves).
    """
    if len(bodies) < 2:
        return [PRIMARY] * len(bodies)
    if center_xy is None:
        points = np.concatenate([body.vertices for body in bodies])
        center_xy = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
    radii = np.asarray(_mean_radii(bodies, np.asarray(center_xy, dtype=float)))
    order = np.argsort(radii)
    gaps = np.diff(radii[order])
    split = int(np.argmax(gaps))
    if gaps[split] < _RADIUS_GAP_FRACTION * max(float(radii.max()), 1e-12):
        return [PRIMARY] * len(bodies)
    outer = set(order[split + 1 :].tolist())
    return [ACTIVE_SHIELD if index in outer else PRIMARY for index in range(len(bodies))]


def _size_mm(vertices) -> list[float]:
    if not len(vertices):
        return [0.0, 0.0, 0.0]
    return [float(value) * 1e3 for value in np.ptp(vertices, axis=0)]


def _format_size(size_mm) -> str:
    return " × ".join(f"{value:.4g}" for value in size_mm) + " mm"


def _loops(count: int) -> str:
    return f"{count} boundary loop" + ("" if count == 1 else "s")


def support_import_summary(mesh) -> dict:
    """What the STL holds, in the units the design will use.

    Bodies, triangles, size, boundary loops (open ends and windows) and a unit
    sanity warning, plus one default role per body for supports with two or more.
    """
    from ..surface_topology import mesh_surface_diagnostics

    bodies = support_bodies(mesh)
    vertices = np.asarray(mesh.vertices, dtype=float)
    center_xy = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2 if len(vertices) else np.zeros(2)
    radii = _mean_radii(bodies, center_xy) if len(vertices) else [0.0] * len(bodies)
    roles = default_part_roles(bodies, center_xy)
    rows = []
    for index, (body, radius, role) in enumerate(zip(bodies, radii, roles)):
        size = _size_mm(body.vertices)
        loops = len(body.boundaries)
        rows.append(
            {
                "index": index,
                "name": f"Body {index + 1}",
                "triangles": int(body.n_faces),
                "size_mm": size,
                "mean_radius_mm": radius * 1e3,
                "boundary_loops": loops,
                "role": role,
                "text": f"{body.n_faces} triangles · {_format_size(size)} · mean radius {radius * 1e3:.4g} mm · "
                + _loops(loops),
            }
        )
    diagnostics = mesh_surface_diagnostics(mesh)
    size = _size_mm(vertices)
    largest = max(size) / 1e3 if size else 0.0
    warnings = []
    if largest < SMALLEST_PLAUSIBLE_M:
        warnings.append(f"The support is only {_format_size(size)} across. Check the STL scale (0.001 for mm).")
    elif largest > LARGEST_PLAUSIBLE_M:
        warnings.append(f"The support is {_format_size(size)} across. Check the STL scale (1 for metres).")
    if diagnostics.get("non_manifold_edge_count"):
        warnings.append(f"{diagnostics['non_manifold_edge_count']} edges are shared by more than two triangles.")
    if diagnostics.get("zero_area_face_count"):
        warnings.append(f"{diagnostics['zero_area_face_count']} triangles have zero area.")
    loops = sum(row["boundary_loops"] for row in rows)
    text = (
        f"{len(rows)} bod{'y' if len(rows) == 1 else 'ies'} · {mesh.n_faces} triangles · "
        f"{_format_size(size)} · {_loops(loops)} (open ends and windows)"
    )
    return {
        "bodies": rows,
        "triangles": int(mesh.n_faces),
        "size_mm": size,
        "boundary_loops": loops,
        "warnings": warnings,
        "roles": roles,
        "needs_roles": len(rows) >= 2,
        "text": text,
    }


def with_part_roles(mesh, roles):
    """Return a copy of ``mesh`` that records one role per body in pipeline order.

    One body needs no role. ``roles=None`` keeps any roles the mesh already carries.
    """
    if roles is None:
        return mesh
    roles = [normalize_part_role(role) for role in roles]
    if any(role is None for role in roles):
        raise ValueError("Each body needs a role: primary, active shield or other.")
    count = len(support_bodies(mesh))
    if len(roles) != count:
        raise ValueError(f"The support has {count} bodies but {len(roles)} roles were given.")
    tagged = mesh.copy()
    tagged.metadata = dict(mesh.metadata)
    if count < 2:
        tagged.metadata.pop("part_roles", None)
    else:
        tagged.metadata["part_roles"] = roles
    return tagged


def body_roles(metadata) -> dict[str, str]:
    """Saved conductor body id → role, from fabrication review and pipeline sweep evidence."""
    roles = {}
    sources = (
        ((metadata.get("fabrication") or {}).get("paths") or ()),
        ((metadata.get("pipeline_sweeps") or {}).get("bodies") or ()),
    )
    for bodies in sources:
        for body in bodies:
            if isinstance(body, dict) and isinstance(body.get("body_id"), str):
                role = normalize_part_role(body.get("role"))
                if role is not None:
                    roles[body["body_id"]] = role
    return roles


def body_label(roles: dict, body_id) -> str:
    """``"Primary · part001_body0001"`` when the run recorded a role, else the id itself."""
    if not body_id:
        return ""
    return body_display_name(body_id, roles.get(body_id))


__all__ = [
    "ACTIVE_SHIELD",
    "OTHER",
    "PRIMARY",
    "ROLE_CHOICES",
    "body_label",
    "body_roles",
    "default_part_roles",
    "read_support_stl",
    "support_bodies",
    "support_import_summary",
    "with_part_roles",
]
