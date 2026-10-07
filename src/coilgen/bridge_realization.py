"""Reject bridge attachments whose final support realization breaks curvature."""

from __future__ import annotations

import numpy as np

from .config import CoilGenConfig
from .models import CoilPart, WirePathGeometry


def minimum_bridge_radius(config: CoilGenConfig) -> float:
    """Keep the profile-envelope margin used during bridge generation."""
    from .manufacturing import _cross_section_envelope_radius, conductor_cross_section_points

    envelope = _cross_section_envelope_radius(conductor_cross_section_points(config))
    limit = float(config.sweep_max_curvature_envelope_ratio)
    return max(float(config.sweep_min_bend_radius), envelope / (0.98 * limit) if limit > 0 else 0.0, 1e-9)


def bridge_realization_check(
    part: CoilPart,
    points: np.ndarray,
    geometry: list[WirePathGeometry],
    start: int,
    end: int,
    candidate_points: np.ndarray,
    config: CoilGenConfig,
    closed: bool,
) -> tuple[bool, dict[str, object]]:
    """Check a free bridge and its retained neighbors in the sweep's frame.

    Analytic surface retraction moves shared ``SURFACE`` endpoints while the
    adjacent free bridge remains fixed. A smooth bridge on the mesh can thereby
    acquire a sharp attachment. Only that mixed analytic attachment is checked
    here; this is neither a full-body nor a clearance certificate. No input
    coordinates, ownership, or geometry labels are changed.
    """
    from .manufacturing import (
        _analytic_cylinder_surface_constraint,
        _prepare_sweep_centerline,
        conductor_cross_section_points,
    )

    diagnostics: dict[str, object] = {"bridge_realization_checked": False}
    support = _analytic_cylinder_surface_constraint(part, config)
    if support is None:
        return True, diagnostics

    points = np.asarray(points, dtype=float)
    candidate = np.asarray(candidate_points, dtype=float)
    count = len(geometry)
    explicit_closure = len(points) == count + 1
    if not (explicit_closure or (closed and len(points) == count)):
        raise ValueError("Bridge source geometry must label every source interval.")
    if not (0 <= start < end < len(points)) or candidate.ndim != 2 or candidate.shape[1] != 3 or len(candidate) < 2:
        raise ValueError("Bridge replacement needs an ordered source span and at least two 3D points.")
    if closed and explicit_closure and np.linalg.norm(points[0] - points[-1]) > 1e-10:
        raise ValueError("Closed bridge source must repeat its first point.")
    if not np.all(np.isfinite(candidate)):
        return False, {**diagnostics, "bridge_realization_reason": "nonfinite_candidate"}
    if not np.allclose(candidate[[0, -1]], points[[start, end]], rtol=0.0, atol=1e-12):
        return False, {**diagnostics, "bridge_realization_reason": "attachment_moved"}
    if closed and end - start == count:
        return True, diagnostics

    has_before = closed or start > 0
    has_after = closed or end < count
    before_index = (start - 1) % count
    after_index = end % count
    surface_attachments = []
    if has_before and WirePathGeometry(geometry[before_index]) == WirePathGeometry.SURFACE:
        surface_attachments.append(candidate[0])
    if has_after and WirePathGeometry(geometry[after_index]) == WirePathGeometry.SURFACE:
        surface_attachments.append(candidate[-1])
    if not surface_attachments:
        return True, diagnostics
    attachments = np.asarray(surface_attachments)
    projected, _normals = support.project(attachments)
    correction = float(np.max(np.linalg.norm(projected - attachments, axis=1)))
    diagnostics["bridge_realization_attachment_retraction_m"] = correction
    # Include the true preceding/following source interval even at a closed
    # route's storage seam. Boundary vertices retain both incident types.
    # Even an attachment already on the analytic surface can acquire a kink:
    # retracting its neighboring chord changes the incident surface tangent.
    chunks = [candidate]
    local_geometry = [WirePathGeometry.FREE_3D] * (len(candidate) - 1)
    if has_before:
        chunks.insert(0, points[before_index : before_index + 1])
        local_geometry.insert(0, WirePathGeometry(geometry[before_index]))
    if has_after:
        following = (end + 1) % count if closed else end + 1
        chunks.append(points[following : following + 1])
        local_geometry.append(WirePathGeometry(geometry[after_index]))
    local_points = np.vstack(chunks)
    prepared, _prepared_geometry, prepared_diagnostics = _prepare_sweep_centerline(
        part,
        local_points,
        local_geometry,
        conductor_cross_section_points(config),
        config,
        closed=False,
    )
    radius = float(prepared_diagnostics["sweep_min_resampled_bend_radius_m"])
    required = minimum_bridge_radius(config)
    # A zero reported radius means that the preparation found no curved triple.
    minimum = radius if radius > 0.0 else float("inf")
    vectors = np.diff(prepared, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    products = lengths[:-1] * lengths[1:]
    cosines = np.divide(
        np.einsum("ij,ij->i", vectors[:-1], vectors[1:]),
        products,
        out=np.ones(len(products)),
        where=products > 1e-24,
    )
    reversal = bool(np.any(cosines <= -1.0 + 1e-10))
    accepted = bool(np.all(np.isfinite(prepared)) and not np.isnan(radius) and not reversal and minimum >= required)
    diagnostics.update(
        bridge_realization_checked=True,
        bridge_realization_reason="accepted" if accepted else "realized_attachment_curvature",
        bridge_realization_minimum_radius_m=minimum,
        bridge_realization_required_radius_m=required,
        bridge_realization_point_count=len(prepared),
        bridge_realization_collinear_reversal=reversal,
    )
    return accepted, diagnostics
