"""Physical raised transfers on an open or cyclic reference conductor trace."""

from __future__ import annotations

import numpy as np

from .faceted_support import faceted_cylinder_support
from .models import WirePathGeometry


def raised_span_metrics(
    points,
    projector,
    *,
    closed=False,
    tolerance_m=1e-6,
    segment_geometry=None,
    maximum_sample_spacing_m=0.001,
    maximum_samples=100000,
):
    """Count continuous excursions from the nominal attachment surface.

    Input points are unmounted profile origins. SURFACE intervals describe the
    nominal surface-following geometry, whose polygonal chords need not lie on
    a curved support. Other intervals are measured, never classified merely
    from FREE_3D/offset labels. Independent support validation remains required.

    Collinear subdivisions are removed before distance sampling. Detachment is
    resolved at the declared spacing and height tolerance, not inferred from
    how many obstacles a transfer crosses. Budget exhaustion raises explicitly.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("Jump counting requires finite (n, 3) geometry.")
    if (
        not np.isfinite(tolerance_m)
        or tolerance_m <= 0
        or not np.isfinite(maximum_sample_spacing_m)
        or maximum_sample_spacing_m <= 0
    ):
        raise ValueError("Jump height tolerance and sampling spacing must be positive.")
    if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int) or maximum_samples < 3:
        raise ValueError("Jump sampling requires an integer budget of at least three points.")
    if closed and not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    geometry = tuple(segment_geometry) if segment_geometry is not None else ()
    if geometry and len(geometry) != len(points) - 1:
        raise ValueError("Jump provenance must match reference intervals.")
    attached = np.array([WirePathGeometry(value) == WirePathGeometry.SURFACE for value in geometry], dtype=bool)
    if not len(attached):
        attached = np.zeros(len(points) - 1, dtype=bool)
    # Preserve corners and attachment changes; eliminate sampling-only vertices.
    vertices, labels = [points[0]], []
    for index, endpoint in enumerate(points[1:]):
        if np.linalg.norm(endpoint - vertices[-1]) <= 1e-14:
            continue
        if len(vertices) > 1 and labels[-1] == attached[index]:
            a, b = vertices[-1] - vertices[-2], endpoint - vertices[-1]
            scale = np.linalg.norm(a) * np.linalg.norm(b)
            if a @ b > 0 and np.linalg.norm(np.cross(a, b)) <= 1e-12 * scale:
                vertices[-1] = endpoint
                continue
        vertices.append(endpoint)
        labels.append(bool(attached[index]))
    if not labels:
        return {"jump_count": 0, "raised_length_m": 0.0, "sample_count": 1}
    sections, surface, lengths = [], [], []
    sample_count = 1
    for start, end, nominal in zip(vertices[:-1], vertices[1:], labels):
        length = float(np.hypot.reduce(end - start))
        if not np.isfinite(length):
            raise ValueError("Jump interval length is not finite.")
        with np.errstate(over="ignore"):
            required = np.divide(length, maximum_sample_spacing_m)
        if not nominal and (not np.isfinite(required) or required > maximum_samples - sample_count):
            raise ValueError("jump_sampling_budget_exceeded")
        divisions = 1 if nominal else max(2, int(np.ceil(required)))
        sample_count += divisions
        if sample_count > maximum_samples:
            raise ValueError("jump_sampling_budget_exceeded")
        section = np.linspace(start, end, divisions + 1)
        sections.append(section if not sections else section[1:])
        surface.extend([nominal] * divisions)
        lengths.extend([length / divisions] * divisions)
    sampled = np.vstack(sections)
    surface = np.asarray(surface, dtype=bool)
    nominal_points = np.zeros(len(sampled), dtype=bool)
    nominal_points[:-1] |= surface
    nominal_points[1:] |= surface
    if closed:
        nominal_points[0] = nominal_points[-1] = bool(nominal_points[0] or nominal_points[-1])
    distance = np.zeros(len(sampled))
    query = ~nominal_points
    if np.any(query):
        # Only distances are used here. A verified faceted extrusion supplies
        # the exact closest lateral point without constructing mounting frames.
        distance_projector = faceted_cylinder_support(projector) or projector
        projected = np.asarray(distance_projector.project(sampled[query])[0])
        if projected.shape != sampled[query].shape or not np.all(np.isfinite(projected)):
            raise ValueError("Jump support projection is unavailable.")
        distance[query] = np.linalg.norm(sampled[query] - projected, axis=1)
    raised = distance > tolerance_m
    if closed:
        # Endpoint occurrences are one physical location, not two transfers.
        raised[0] = raised[-1] = bool(raised[0] or raised[-1])
        ring = raised[:-1]
        count = int(np.count_nonzero(ring & ~np.roll(ring, 1)))
        if len(ring) and np.all(ring):
            count = 1
    else:
        count = int(raised[0]) + int(np.count_nonzero(raised[1:] & ~raised[:-1]))
    # Length is diagnostic only; threshold crossings are linearly interpolated.
    fractions = np.zeros(len(lengths))
    both = raised[:-1] & raised[1:]
    fractions[both] = 1.0
    crossing = raised[:-1] != raised[1:]
    low = np.minimum(distance[:-1], distance[1:])
    high = np.maximum(distance[:-1], distance[1:])
    fractions[crossing] = (high[crossing] - tolerance_m) / (high[crossing] - low[crossing])
    return {
        "jump_count": count,
        "raised_length_m": float(np.dot(lengths, fractions)),
        "sample_count": len(sampled),
        "height_tolerance_m": float(tolerance_m),
        "maximum_sample_spacing_m": float(maximum_sample_spacing_m),
        "closed": bool(closed),
    }


def fabricated_jump_metrics(part, config, *, projector=None):
    """Undo the configured mounting/profile offset before measuring excursions."""
    from .manufacturing import (
        _SurfaceProjector,
        _analytic_cylinder_surface_constraint,
        _mounting_centerline_offset,
        _polygon_centroid,
    )

    trace = part.fabrication_path
    if trace is None or not trace.closed:
        raise ValueError("A closed fabricated trace is required for cyclic jump counting.")
    centroid = _polygon_centroid(trace.cross_section_m)
    offset = _mounting_centerline_offset(config, trace.cross_section_m)
    origins = trace.points_m - (offset + centroid[1]) * trace.normals - centroid[0] * trace.transverse
    surface = _analytic_cylinder_surface_constraint(part, config)
    surface = surface if surface is not None else (projector or _SurfaceProjector(part.mesh))
    return raised_span_metrics(
        origins,
        surface,
        closed=True,
        segment_geometry=trace.segment_geometry,
        tolerance_m=config.routing_jump_height_tolerance_m,
        maximum_sample_spacing_m=config.routing_jump_sample_spacing_m,
    )
