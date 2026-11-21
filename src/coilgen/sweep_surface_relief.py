"""Bounded physical realization on verified convex supports.

The nominal directed winding is immutable. Candidate smoothing, local bend
fairing and crossing relief are proposals only: the existing body, support,
clearance and source-corridor predicates must all accept before adoption.
"""

from dataclasses import dataclass
from copy import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_dilation, label
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .group_route_convex_cap import convex_mesh_cap
from .group_route_convex_shell import convex_circular_shell
from .models import FabricationPath, Mesh, WirePathGeometry
from .wire_provenance import ordered_source_stations
from .progress import ProgressReporter


@dataclass(frozen=True)
class _Envelope:
    origin: np.ndarray
    basis: np.ndarray
    radius: float
    slope: float
    bottom: float
    top: float | None
    spherical: bool = False

    @classmethod
    def from_mesh(cls, mesh, tolerance):
        from .group_route_support import _oriented_surface_reason

        if _oriented_surface_reason(mesh, 200_000):
            return None
        shell, _ = convex_circular_shell(mesh, tolerance)
        if shell is not None:
            axis, origin = shell.axis, shell.origin
            radius, slope = shell.bottom_radius, (shell.top_radius - shell.bottom_radius) / shell.axial_length
            bottom, top, spherical = 0.0, shell.axial_length, False
        else:
            cap, _ = convex_mesh_cap(mesh, tolerance)
            if cap is None:
                return None
            # Fit geometry, never metadata. Translation before fitting avoids
            # the poorly conditioned squared-coordinate system at large origins.
            center = mesh.vertices.mean(axis=0)
            xyz = mesh.vertices - center
            solution, _, rank, _ = np.linalg.lstsq(
                np.column_stack((2 * xyz, np.ones(len(xyz)))), np.sum(xyz * xyz, axis=1), rcond=None
            )
            if rank != 4:
                return None
            origin = center + solution[:3]
            radii = np.linalg.norm(mesh.vertices - origin, axis=1)
            radius = float(radii.mean())
            if radius <= tolerance or np.max(abs(radii - radius)) > tolerance:
                return None
            axis = -cap.rim_normal
            slope, bottom, top, spherical = 0.0, float((cap.rim_origin - origin) @ axis), None, True
        radial = mesh.vertices - origin
        radial -= (radial @ axis)[:, None] * axis
        first = radial[np.argmax(np.linalg.norm(radial, axis=1))]
        first /= np.linalg.norm(first)
        basis = np.column_stack((first, np.cross(axis, first), axis))
        return cls(origin, basis, radius, slope, bottom, top, spherical)

    def local(self, points):
        return (points - self.origin) @ self.basis

    def global_points(self, points):
        return points @ self.basis.T + self.origin

    def normal(self, points):
        p = self.local(points)
        if self.spherical:
            n = p / np.linalg.norm(p, axis=1)[:, None]
        else:
            n = np.column_stack((p[:, :2] / np.linalg.norm(p[:, :2], axis=1)[:, None], np.full(len(p), -self.slope)))
            n /= np.linalg.norm(n, axis=1)[:, None]
        return n @ self.basis.T

    def height(self, points):
        p = self.local(points)
        if self.spherical:
            return np.linalg.norm(p, axis=1) - self.radius
        return (np.linalg.norm(p[:, :2], axis=1) - self.radius - self.slope * p[:, 2]) / np.hypot(1, self.slope)

    def constrain(self, points, offset, section, config, *, radial_width=None):
        p = self.local(points)
        width = config.sweep_min_bend_radius / 4
        tolerance = config.sweep_manufacturing_tolerance

        def soft_bound(value, low, high=None, transition=width):
            result = low + transition * np.logaddexp(0, (value - low) / transition)
            if high is not None:
                result -= transition * np.logaddexp(0, (value - high) / transition)
            return result

        footprint = np.max(np.linalg.norm(section, axis=1)) + 2 * tolerance
        p[:, 2] = soft_bound(p[:, 2], self.bottom + footprint, None if self.top is None else self.top - footprint)
        radial = np.linalg.norm(p if self.spherical else p[:, :2], axis=1)
        base = self.radius if self.spherical else self.radius + self.slope * p[:, 2]
        scale = 1.0 if self.spherical else np.hypot(1, self.slope)
        proposed = soft_bound(
            radial,
            base + (offset + 0.6 * tolerance) * scale,
            base + (offset + config.normal_shift_length - 2 * tolerance) * scale,
            transition=width if radial_width is None else radial_width,
        )
        if self.spherical:
            p *= (proposed / radial)[:, None]
        else:
            p[:, :2] *= (proposed / radial)[:, None]
        return self.global_points(p)

    def lift(self, points, heights):
        p = self.local(points)
        radial = np.linalg.norm(p if self.spherical else p[:, :2], axis=1)
        scale = 1 if self.spherical else np.hypot(1, self.slope)
        if self.spherical:
            p *= (1 + heights / radial)[:, None]
        else:
            p[:, :2] *= (1 + heights * scale / radial)[:, None]
        return self.global_points(p)


def _closed_arc(points):
    closed = np.vstack((points, points[0]))
    return closed, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]


def _uniform(points, spacing):
    closed, arc = _closed_arc(points)
    keep = np.r_[True, np.diff(arc) > 1e-12]
    closed, arc = closed[keep], arc[keep]
    count = int(np.ceil(arc[-1] / spacing))
    if count < 12 or count > 150_000:
        return None
    stations = np.linspace(0, arc[-1], count, endpoint=False)
    return np.column_stack([np.interp(stations, arc, closed[:, j]) for j in range(3)]), arc[-1] / count


def _source_corridor(source, target):
    """Continuous directed correspondence, including every source/target corner.

    Between consecutive merged knots both curves are affine, so the norm of
    their difference is convex and its maximum occurs at a checked endpoint.
    This is a complete-polyline bound, not nearest-vertex sampling or an
    unordered match to a neighbouring turn. A failed correspondence rejects.
    """
    source = np.asarray(source)
    if np.linalg.norm(source[0] - source[-1]) > 1e-9:
        return None
    target_closed, target_arc = _closed_arc(target)
    indices, fractions = ordered_source_stations(source, target_closed)
    source_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
    sl, tl = source_arc[-1], target_arc[-1]
    if min(sl, tl) <= 1e-12:
        return None
    mid = (target_arc[:-1] + target_arc[1:]) / 2
    matched = source_arc[indices] + fractions * np.diff(source_arc)[indices]
    matched = np.unwrap(matched * (2 * np.pi / sl)) * (sl / (2 * np.pi))
    if np.any(np.diff(matched) < -1e-10) or matched[-1] - matched[0] > sl:
        return None
    matched = np.maximum.accumulate(matched)
    u, v = np.r_[mid, mid[0] + tl], np.r_[matched, matched[0] + sl]
    first_period = int(np.floor(v[0] / sl))
    source_knots = np.unique(np.concatenate([source_arc + k * sl for k in range(first_period, first_period + 3)]))
    source_knots = source_knots[(source_knots > v[0]) & (source_knots < v[-1])]
    knots = np.unique(
        np.r_[u, np.interp(source_knots, v, u), target_arc[target_arc >= u[0]], target_arc[target_arc < u[0]] + tl]
    )
    s, t = np.interp(knots, u, v) % sl, knots % tl
    a = np.column_stack([np.interp(s, source_arc, source[:, j]) for j in range(3)])
    b = np.column_stack([np.interp(t, target_arc, target_closed[:, j]) for j in range(3)])
    delta = a - b
    return np.linalg.norm(delta, axis=1), t, delta, tl, indices


def _frames(points, envelope):
    from .manufacturing import _path_tangents

    tangent = _path_tangents(points, closed=True)
    normal = envelope.normal(points)
    normal -= np.einsum("ij,ij->i", normal, tangent)[:, None] * tangent
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(lengths < 1e-6):
        return None
    normal /= lengths[:, None]
    return tangent, normal, np.cross(tangent, normal)


def _fair_patch(points, envelope, radius, bound):
    p = envelope.local(points)
    angular = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    base, scale = angular.mean(), np.linalg.norm(p[:, :2], axis=1).mean()
    initial = np.column_stack(((angular - base) * scale, p[:, 2]))
    height = np.linalg.norm(p[:, :2], axis=1) - (envelope.radius + envelope.slope * p[:, 2])

    def physical(uv):
        angle = base + uv[:, 0] / scale
        r = envelope.radius + envelope.slope * uv[:, 1] + height
        return np.column_stack((r * np.cos(angle), r * np.sin(angle), uv[:, 1]))

    count, fixed = len(p), 3
    original_lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)

    def residual(value):
        uv = initial.copy()
        uv[fixed:-fixed] = value.reshape(-1, 2)
        differences = np.diff(physical(uv), axis=0)
        lengths = np.linalg.norm(differences, axis=1)
        units = differences / np.maximum(lengths[:, None], 1e-12)
        curvature = 2 * np.linalg.norm(np.diff(units, axis=0), axis=1) / np.maximum(lengths[:-1] + lengths[1:], 1e-12)
        return np.r_[
            ((uv - initial) * 20).ravel(),
            (lengths - original_lengths) * 2000,
            np.maximum(curvature * radius - 1, 0) * 10,
            (np.diff(uv, 2, axis=0) - np.diff(initial, 2, axis=0)).ravel() * 10,
        ]

    jacobian = lil_matrix((count * 2 + count - 1 + count - 2 + (count - 2) * 2, (count - 2 * fixed) * 2), dtype=int)
    for point in range(fixed, count - fixed):
        columns = [(point - fixed) * 2, (point - fixed) * 2 + 1]
        jacobian[point * 2 : point * 2 + 2, columns] = 1
        for i in range(max(0, point - 1), min(count - 2, point) + 1):
            jacobian[count * 2 + i, columns] = 1
        for i in range(max(0, point - 2), min(count - 3, point) + 1):
            jacobian[count * 3 - 1 + i, columns] = 1
            jacobian[count * 4 - 3 + 2 * i : count * 4 - 3 + 2 * i + 2, columns] = 1
    initial_value = initial[fixed:-fixed].ravel()
    fit = least_squares(
        residual,
        initial_value,
        jac_sparsity=jacobian.tocsr(),
        bounds=(initial_value - bound, initial_value + bound),
        max_nfev=250,
        ftol=1e-8,
        xtol=1e-9,
        gtol=1e-8,
    )
    chart = initial.copy()
    chart[fixed:-fixed] = fit.x.reshape(-1, 2)
    return envelope.global_points(physical(chart))


def _fair_bends(points, envelope, config, spacing):
    from .manufacturing import _polyline_circumradii

    radius = config.sweep_min_bend_radius
    if envelope.spherical:
        return points
    points = points.copy()
    padding = max(6, int(np.ceil(4.2 * radius / spacing)))
    for _ in range(3):
        radii, indices = _polyline_circumradii(points, closed=True, minimum_turn_angle_deg=1e-6, return_indices=True)
        bad = np.zeros(len(points), dtype=bool)
        bad[indices[radii < radius * 1.006]] = True
        if not bad.any():
            break
        # Rotate into a safe gap so a patch spanning the winding seam is one
        # continuous optimisation problem, with its endpoints fixed.
        expanded = binary_dilation(np.tile(bad, 3), iterations=padding)[len(bad) : 2 * len(bad)]
        gaps = np.flatnonzero(~expanded)
        if not len(gaps):
            return None
        shift = int(gaps[len(gaps) // 2])
        rotated = np.roll(points, -shift, axis=0)
        groups, count = label(np.roll(expanded, -shift))
        for group in range(1, count + 1):
            ProgressReporter.from_config(config, label="Conductor relief").report(
                None, f"Fair constrained bend {group}/{count}", force=True
            )
            selected = np.flatnonzero(groups == group)
            if len(selected) <= 6 or len(selected) > 1200:
                return None
            rotated[selected] = _fair_patch(
                rotated[selected],
                envelope,
                radius * 1.05,
                min(radius * 5 / 6, config.sweep_corner_cluster_max_source_deviation_m),
            )
        points = np.roll(rotated, shift, axis=0)
    return points


def _restore_corridor(points, envelope, offset, source, config):
    bound = config.sweep_corner_cluster_max_source_deviation_m
    for _ in range(4):
        frames = _frames(points, envelope)
        if frames is None:
            return None
        origins = points - offset * frames[1]
        corridor = _source_corridor(source, origins)
        if corridor is None:
            return None
        distance, stations, corrections, length, _ = corridor
        if distance.max() <= bound:
            return points
        _, arc = _closed_arc(origins)
        selected, total = [], np.zeros_like(points)
        width = config.sweep_min_bend_radius * 8 / 3
        for index in np.argsort(distance)[::-1]:
            if distance[index] <= bound * 0.967:
                break
            station = stations[index]
            if any(min(abs(station - other), length - abs(station - other)) < width * 3 for other in selected):
                continue
            selected.append(station)
            delta = corrections[index] * (1 - bound * 0.933 / distance[index])
            d = abs(arc[:-1] - station)
            d = np.minimum(d, length - d)
            total += delta[None, :] * np.exp(-0.5 * (d / width) ** 2)[:, None]
        points = points + total
    return None


def _crossing_relief(points, envelope, certificate, config):
    witnesses = certificate.get("copper_clearance_witness_intervals", ())
    if not witnesses:
        return None
    _, arc = _closed_arc(points)
    length = arc[-1]
    heights = envelope.height(points)
    width = config.sweep_min_bend_radius * 8 / 3
    selected, bump = [], np.zeros(len(points))
    # Stable order matters: preserve the existing over/under order and group
    # overlapping facet witnesses into one smooth bridge adjustment.
    for a, b, _ in witnesses:
        high = a if heights[a] > heights[b] else b
        station = arc[high]
        if any(min(abs(station - other), length - abs(station - other)) < 3 * width for other in selected):
            continue
        selected.append(station)
        d = abs(arc[:-1] - station)
        d = np.minimum(d, length - d)
        # A bounded proposal, not a substitute for an exact gap certificate.
        amount = max(
            3 * config.sweep_manufacturing_tolerance,
            0.225 * config.conductor_cross_section_height if certificate.get("exact_intersection_count", 0) else 0,
        )
        bump += amount * np.exp(-0.5 * (d / width) ** 2)
    return envelope.lift(points, bump)


def _candidate(part, points, section, envelope, offset, config):
    from . import manufacturing as m

    frames = _frames(points, envelope)
    if frames is None:
        return None
    tangent, normal, transverse = frames
    origins = points - offset * normal
    corridor = _source_corridor(part.wire_path.v, origins)
    if corridor is None or corridor[0].max() > config.sweep_corner_cluster_max_source_deviation_m:
        return None
    path = FabricationPath(
        points,
        tangent,
        normal,
        transverse,
        section,
        part.wire_path.v,
        True,
        segment_owners=tuple(part.wire_path.segment_owners[i] for i in corridor[4])
        if part.wire_path.segment_owners
        else (),
        segment_geometry=(WirePathGeometry.OFFSET_SURFACE,) * len(points),
    )
    vertices = (
        points[:, None] + section[None, :, :1] * transverse[:, None] + section[None, :, 1:] * normal[:, None]
    ).reshape(-1, 3)
    faces, loft = m._triangulate_sweep_laterals(
        vertices, len(points), len(section), closed=True, optimize_diagonals=True
    )
    mesh = Mesh(vertices, np.asarray(faces, dtype=np.int64))
    diagnostics = {
        **m._final_path_bend_diagnostics(points, section, config, closed=True),
        **m._physical_segment_length_diagnostics(points, section, config, closed=True),
        **m._final_frame_and_profile_diagnostics(
            origins, tangent, normal, transverse, section, config, closed=True, mounting_offset=offset
        ),
        **loft,
        **m._swept_mesh_quality_diagnostics(mesh, config),
        **m._support_conformity_diagnostics(part, mesh, origins, config),
        **m._sweep_section_conservation_diagnostics(
            mesh, vertices, origins, normal, transverse, section, offset, closed=True
        ),
        **m.mesh_manufacturing_diagnostics(mesh),
        "sweep_relief_source_deviation_bound_m": float(corridor[0].max()),
        "sweep_relief_source_deviation_limit_m": config.sweep_corner_cluster_max_source_deviation_m,
        "sweep_relief_source_certificate": "continuous_directed_polyline_correspondence",
        "sweep_ring_count": len(points),
        "sweep_output_point_count": len(points),
        "sweep_section_point_count": len(section),
        "sweep_closed_path": True,
        "sweep_end_cap_count": 0,
        "sweep_centerline_offset_m": offset,
        "conductor_path_reference": "centroid",
        "conductor_mounting_mode": "surface_outer",
        "sweep_surface_realization": config.sweep_surface_realization,
        "sweep_physics_geometry": "mounted_profile_centroid",
        "sweep_frame_method": "surface_normal_relief",
    }
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    diagnostics["sweep_relief_max_segment_length_m"] = float(lengths.max())
    if config.sweep_max_segment_length > 0 and lengths.max() > config.sweep_max_segment_length:
        return None
    # Cubic tangent interpolation is bounded by its control polygon. This
    # checks complete intervals against the existing chord error, not midpoints.
    direction = (np.roll(points, -1, axis=0) - points) / lengths[:, None]
    tangent_end = np.roll(tangent, -1, axis=0)
    flatness = (
        lengths
        / 3
        * np.maximum(
            np.linalg.norm(tangent - np.einsum("ij,ij->i", tangent, direction)[:, None] * direction, axis=1),
            np.linalg.norm(tangent_end - np.einsum("ij,ij->i", tangent_end, direction)[:, None] * direction, axis=1),
        )
    )
    diagnostics["sweep_relief_cubic_flatness_bound_m"] = float(flatness.max())
    if config.sweep_max_chord_error > 0 and flatness.max() > config.sweep_max_chord_error:
        return None
    trial = copy(part)
    trial.fabrication_path, trial.swept_body, trial.manufacturing_diagnostics = path, mesh, diagnostics
    m._finalize_manufacturing_diagnostics(trial)
    if not diagnostics["manufacturing_ok"]:
        return None
    return trial


def repair_surface_sweep(part, section, config, original_diagnostics):
    """Return an independently certified replacement, or leave the part intact."""
    from . import manufacturing as m

    path = part.fabrication_path
    if (
        path is None
        or not path.closed
        or config.sweep_corner_cluster_max_source_deviation_m <= 0
        or config.sweep_min_bend_radius <= 0
        or config.normal_shift_length <= 0
        or str(config.conductor_mounting_mode).lower() != "surface_outer"
        or m._uses_profile_origin(config)
        or str(config.sweep_bend_mode).lower() != "fillet"
        or str(config.sweep_self_intersection_check).lower() == "off"
        or m._manufacturing_route_topology_failure_reasons(part)
    ):
        return None
    tolerance = max(config.sweep_manufacturing_tolerance, 1e-8)
    reporter = ProgressReporter.from_config(config, label="Conductor relief")
    reporter.report(None, "Verify support geometry and the permitted repair corridor", force=True)
    envelope = _Envelope.from_mesh(part.mesh, tolerance)
    if envelope is None:
        return None
    radius = config.sweep_min_bend_radius
    spacing = min(radius * 0.15, config.sweep_max_segment_length or radius * 0.15)
    sampled = _uniform(path.points_m, spacing)
    if sampled is None:
        return None
    dense, spacing = sampled
    # A broad soft bound needlessly lifts the long contour runs and attenuates
    # their field. Prefer a narrower radial transition; retain the broad
    # candidate for difficult bends. Both undergo identical exact acceptance.
    for radial_width in (radius / 16, radius / 4):
        candidate = _repair_surface_candidate(
            part, section, config, original_diagnostics, envelope, dense, spacing, radial_width
        )
        if candidate is not None:
            return candidate
    return None


def _repair_surface_candidate(part, section, config, original_diagnostics, envelope, dense, spacing, radial_width):
    from . import manufacturing as m
    from .group_route_support import certify_group_route_support
    from .group_route_validation import GroupRouteValidationLimits, _exact_collision_audit
    from .swept_clearance import required_copper_gap

    reporter = ProgressReporter.from_config(config, label="Conductor relief")
    radius = config.sweep_min_bend_radius
    width = radius * (5 / 6 if abs(envelope.slope) > 1e-10 else 1)
    points = gaussian_filter1d(dense, width / spacing, axis=0, mode="wrap")
    offset = m._mounting_centerline_offset(config, section)
    points = envelope.constrain(points, offset, section, config, radial_width=radial_width)
    points = _fair_bends(points, envelope, config, spacing)
    if points is None:
        return None
    points = _restore_corridor(points, envelope, offset, part.wire_path.v, config)
    if points is None:
        return None
    for iteration in range(5):
        reporter.report(None, f"Check repaired body and source corridor ({iteration + 1}/5)", force=True)
        trial = _candidate(part, points, section, envelope, offset, config)
        if trial is None:
            return None
        support = certify_group_route_support(trial, config)
        if not support.accepted:
            return None
        reporter.report(None, "Certify exact copper clearance on the repaired body", force=True)
        certificate, reason = _exact_collision_audit(
            trial.swept_body,
            GroupRouteValidationLimits(),
            fabrication_path=trial.fabrication_path,
            required_gap_m=required_copper_gap(config),
        )
        if not reason:
            diagnostics = trial.manufacturing_diagnostics
            diagnostics.update(certificate)
            diagnostics.update(
                sweep_surface_relief_applied=True,
                sweep_surface_relief_iterations=iteration,
                sweep_relief_radial_transition_width_m=radial_width,
                sweep_relief_input_diagnostics=dict(original_diagnostics),
                sweep_relief_support_certificate=support.diagnostics,
                copper_clearance_failure="",
                sweep_mesh_self_intersection=False,
                sweep_mesh_collision_status="checked",
            )
            m._finalize_manufacturing_diagnostics(trial)
            if diagnostics["manufacturing_ok"]:
                return trial
            return None
        points = _crossing_relief(points, envelope, certificate, config)
        if points is None:
            return None
    return None
