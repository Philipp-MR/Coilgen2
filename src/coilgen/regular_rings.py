"""Conservative geometric traversal for parallel essential surface rings."""

from __future__ import annotations

import numpy as np


def closed_curve_centroid(points: np.ndarray) -> np.ndarray | None:
    """Arc-length centroid, invariant to cyclic starts and collinear sampling."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.all(np.isfinite(points)):
        return None
    if np.linalg.norm(points[0] - points[-1]) > 1e-12:
        points = np.vstack((points, points[:1]))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 1e-15:
        return None
    return np.sum(0.5 * (points[:-1] + points[1:]) * lengths[:, None], axis=0) / total


def ring_support_axis(mesh):
    """Identify an annular strip from its two planar, parallel boundaries."""
    boundaries = getattr(mesh, "boundaries", []) or []
    if len(boundaries) != 2:
        return None
    curves = []
    for boundary in boundaries:
        nodes = np.asarray(boundary, dtype=np.int64)
        if len(nodes) < 3 or np.any(nodes < 0) or np.any(nodes >= len(mesh.vertices)):
            return None
        points = np.asarray(mesh.vertices[nodes], dtype=float)
        if np.linalg.norm(points[0] - points[-1]) > 1e-12:
            points = np.vstack((points, points[:1]))
        center = closed_curve_centroid(points)
        if center is None:
            return None
        relative = points - points[0]
        normal = np.sum(np.cross(relative[:-1], relative[1:]), axis=0)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-24:
            return None
        normal /= norm
        extent = float(np.max(np.linalg.norm(points - center, axis=1)))
        if np.max(np.abs((points - center) @ normal)) > max(1e-10, 1e-6 * extent):
            return None
        curves.append((center, normal))
    vector = curves[1][0] - curves[0][0]
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return None
    axis = vector / length
    if any(np.linalg.norm(np.cross(axis, normal)) >= 0.01 for _, normal in curves):
        return None
    return 0.5 * (curves[0][0] + curves[1][0]), axis


def ring_geometry(points: np.ndarray, origin: np.ndarray, axis: np.ndarray):
    """Verify a closed, predominantly transverse once-encircling ring.

    This excludes local saddle contours, multiple winding and tilted rings.
    Axial variation is allowed; nonoverlapping axial bands are checked by the
    caller, rather than assuming that optimized contours are exact circles.
    """
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 4
        or not np.all(np.isfinite(points))
        or np.linalg.norm(points[0] - points[-1]) > 1e-10
    ):
        return None
    centered = points - points[0]
    area_vector = np.sum(np.cross(centered[:-1], centered[1:]), axis=0)
    area_length = float(np.linalg.norm(area_vector))
    if area_length <= 1e-24 or np.linalg.norm(np.cross(axis, area_vector / area_length)) >= 0.01:
        return None
    relative = points - origin
    axial = relative @ axis
    radial = relative - np.outer(axial, axis)
    lengths = np.linalg.norm(radial, axis=1)
    if np.any(lengths <= 1e-12):
        return None
    radial /= lengths[:, None]
    angles = np.arctan2(
        np.cross(radial[:-1], radial[1:]) @ axis,
        np.einsum("ij,ij->i", radial[:-1], radial[1:]),
    )
    if abs(abs(float(np.sum(angles))) - 2.0 * np.pi) > 1e-6:
        return None
    centroid = closed_curve_centroid(points)
    if centroid is None:
        return None
    return {
        "position": float((centroid - origin) @ axis),
        "minimum": float(np.min(axial)),
        "maximum": float(np.max(axial)),
        "orientation": 1 if float(area_vector @ axis) > 0 else -1,
    }


def ring_height_profile(points, origin, axis):
    """A once-encircling, azimuth-monotone polygon as height over angle.

    Optimized rings on a tapered support need not be planar or have their area
    normal parallel to the support axis. Keep their exact points and use their
    angular order only to propose a traversal. Manufacturing checks still own
    clearance, curvature and support acceptance.
    """
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 4
        or not np.isfinite(points).all()
        or np.linalg.norm(points[0] - points[-1]) > 1e-10
    ):
        return None
    radial_axis = np.eye(3)[int(np.argmin(np.abs(axis)))]
    radial_axis = radial_axis - axis * (radial_axis @ axis)
    radial_axis /= np.linalg.norm(radial_axis)
    second_axis = np.cross(axis, radial_axis)
    relative = points - origin
    xy = np.column_stack((relative @ radial_axis, relative @ second_axis))
    if np.any(np.linalg.norm(xy, axis=1) <= 1e-12):
        return None
    angle = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    turns = float(angle[-1] - angle[0])
    if abs(abs(turns) - 2 * np.pi) > 1e-6:
        return None
    sign = 1 if turns > 0 else -1
    if np.any(sign * np.diff(angle) <= 1e-12):
        return None
    height = relative @ axis
    if sign < 0:
        angle, xy, height = angle[::-1], xy[::-1], height[::-1]
    return dict(angle=angle, xy=xy, height=height, orientation=sign)


def _ring_heights(profile, angles):
    """Intersect the original straight segments with the requested radial planes."""
    angle = profile["angle"]
    query = (angles - angle[0]) % (2 * np.pi) + angle[0]
    index = np.clip(np.searchsorted(angle, query, side="right") - 1, 0, len(angle) - 2)
    a, b = profile["xy"][index], profile["xy"][index + 1]
    ray = np.column_stack((np.cos(query), np.sin(query)))

    def cross(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    fraction = -cross(a, ray) / cross(b - a, ray)
    return profile["height"][index] + fraction * (profile["height"][index + 1] - profile["height"][index])


def _profile_ring_order(part, indices, origin, axis):
    profiles = {i: ring_height_profile(part.contour_lines[i].v, origin, axis) for i in indices}
    if any(p is None for p in profiles.values()):
        return None
    signs = {p["orientation"] for p in profiles.values()}
    if len(signs) != 1:
        return None
    sign = next(iter(signs))
    positions = {i: float((closed_curve_centroid(part.contour_lines[i].v) - origin) @ axis) * sign for i in indices}
    ordered = sorted(indices, key=positions.get)
    if not separated_ring_profiles([profiles[i] for i in ordered], sign):
        return None
    return ordered, positions, sign


def separated_ring_profiles(profiles, sign=1):
    """Reject height-order reversals before proposing a common cut meridian."""
    for a, b in zip(profiles, profiles[1:]):
        nodes = np.unique(np.r_[a["angle"] % (2 * np.pi), b["angle"] % (2 * np.pi), 0.0, 2 * np.pi])
        # Include interval midpoints; this is a conservative ordering proposal,
        # never a replacement for the independent conductor collision audit.
        query = np.sort(np.r_[nodes, (nodes[:-1] + nodes[1:]) / 2])
        separation = sign * (_ring_heights(b, query) - _ring_heights(a, query))
        if np.min(separation) <= 1e-8:
            return False
    return True


def regular_ring_order(part, loop_indices, signatures, contains, axis_result):
    """Return a monotone order only for verified, separate essential rings.

    Contour geometry, current direction, group membership and multiplicity are
    never changed. Axis sign follows directed ring current, rather than world
    coordinates, contour indices, potential values or a sampled start vertex.
    """
    original = [int(index) for index in loop_indices]
    diagnostics = {"method": "unchanged", "original_loop_indices": original, "ordered_loop_indices": original}

    def unchanged(reason):
        diagnostics["reason"] = reason
        return original, diagnostics

    if len(original) <= 1:
        return unchanged("single_loop")
    by_index = {signature.loop_index: signature for signature in signatures}
    if any(
        index not in by_index or not by_index[index].boundary_associated or by_index[index].ambiguous
        for index in original
    ):
        return unchanged("not_verified_parallel_essential_rings")
    boundary_keys = {by_index[index].boundary_signature for index in original}
    if len(boundary_keys) != 1 or np.any(np.asarray(contains, dtype=bool)[np.ix_(original, original)]):
        return unchanged("incompatible_or_nested_topology")
    # Several unrelated boundary holes do not identify a single axial strip.
    if axis_result is None or len(getattr(part.mesh, "boundaries", []) or []) != 2:
        return unchanged("ambiguous_support_axis")
    origin, axis = (np.asarray(value, dtype=float) for value in axis_result)
    geometry = {index: ring_geometry(part.contour_lines[index].v, origin, axis) for index in original}
    if any(value is None for value in geometry.values()):
        profile_order = _profile_ring_order(part, original, origin, axis)
        if profile_order is None:
            return unchanged("not_closed_transverse_once_encircling_rings")
        ordered, positions, sign = profile_order
        diagnostics.update(
            method="directed_support_axis_monotone",
            reason="ordered_azimuth_monotone_essential_rings",
            ordered_loop_indices=ordered,
            support_origin_m=origin.tolist(),
            traversal_axis=(sign * axis).tolist(),
            original_axial_travel_m=float(np.abs(np.diff([positions[i] for i in original])).sum()),
            ordered_axial_travel_m=float(np.abs(np.diff([positions[i] for i in ordered])).sum()),
            axial_positions_m=[positions[i] for i in ordered],
        )
        return ordered, diagnostics
    signs = {value["orientation"] for value in geometry.values()}
    if len(signs) != 1:
        return unchanged("incompatible_directed_ring_orientation")
    sign = next(iter(signs))
    axis = sign * axis
    if sign < 0:
        geometry = {
            index: {
                **value,
                "position": -value["position"],
                "minimum": -value["maximum"],
                "maximum": -value["minimum"],
            }
            for index, value in geometry.items()
        }
    ordered = sorted(original, key=lambda index: geometry[index]["position"])
    scale = max(float(np.ptp([value["position"] for value in geometry.values()])), 1e-3)
    tolerance = max(1e-10, 1e-8 * scale)
    if any(
        geometry[second]["minimum"] - geometry[first]["maximum"] <= tolerance
        for first, second in zip(ordered, ordered[1:])
    ):
        return unchanged("overlapping_or_ambiguous_axial_bands")

    def travel(indices):
        return float(np.sum(np.abs(np.diff([geometry[index]["position"] for index in indices]))))

    diagnostics.update(
        method="directed_support_axis_monotone",
        reason="verified_parallel_essential_rings",
        ordered_loop_indices=ordered,
        support_origin_m=origin.tolist(),
        traversal_axis=axis.tolist(),
        original_axial_travel_m=travel(original),
        ordered_axial_travel_m=travel(ordered),
        axial_positions_m=[geometry[index]["position"] for index in ordered],
    )
    return ordered, diagnostics
