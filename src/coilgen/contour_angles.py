from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CoilPart, ContourLine, Mesh


@dataclass(frozen=True)
class IntrinsicAngleTrace:
    """Intrinsic turning diagnostics sampled along one contour."""

    arclength_m: np.ndarray
    heading_angle_rad: np.ndarray
    turn_angle_rad: np.ndarray
    geodesic_curvature_per_m: np.ndarray
    curvature_variation_per_m2: np.ndarray
    curvature_second_derivative_per_m3: np.ndarray
    high_frequency_curvature_per_m: np.ndarray
    sample_weight_m: np.ndarray
    closed: bool


def intrinsic_contour_angle_trace(mesh: Mesh, contour: ContourLine) -> IntrinsicAngleTrace | None:
    """Measure in-surface contour turning after unfolding adjacent triangles.

    The routine is diagnostic only: it neither resamples nor moves the contour.
    ``None`` is returned when exact segment-to-face provenance is unavailable.
    """

    return _intrinsic_contour_angle_trace(mesh, contour, _unit_face_normals(mesh))


def analyze_contour_angle_stability(parts: list[CoilPart]) -> list[CoilPart]:
    """Attach compact intrinsic-angle stability metrics to every coil part."""

    for part in parts:
        face_normals = _unit_face_normals(part.mesh)
        traces = [
            trace
            for contour in part.contour_lines
            if (trace := _intrinsic_contour_angle_trace(part.mesh, contour, face_normals)) is not None
        ]
        part.contour_level_diagnostics.update(_angle_stability_metrics(traces, len(part.contour_lines)))
    return parts


def _intrinsic_contour_angle_trace(
    mesh: Mesh,
    contour: ContourLine,
    face_normals: np.ndarray,
) -> IntrinsicAngleTrace | None:
    points = np.asarray(contour.v, dtype=float)
    if contour.segment_faces is None or len(points) < 3:
        return None

    segment_faces = np.asarray(contour.segment_faces, dtype=np.int64)
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_count = len(segment_vectors)
    if (
        len(segment_faces) != segment_count
        or np.any(segment_faces < 0)
        or np.any(segment_faces >= len(mesh.faces))
        or np.any(segment_lengths <= 1.0e-14)
    ):
        return None

    tangents = segment_vectors / segment_lengths[:, None]
    scale = max(float(np.max(segment_lengths)), float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    closed = bool(np.linalg.norm(points[0] - points[-1]) <= 1.0e-10 * scale)
    current_segments = np.arange(segment_count, dtype=np.int64) if closed else np.arange(1, segment_count)
    if len(current_segments) == 0:
        return None

    turn_angles: list[float] = []
    turn_scales: list[float] = []
    for current_segment in current_segments:
        previous_segment = (int(current_segment) - 1) % segment_count
        current_segment = int(current_segment)
        previous_face = int(segment_faces[previous_segment])
        current_face = int(segment_faces[current_segment])
        if previous_face != current_face:
            shared_vertices = np.intersect1d(mesh.faces[previous_face], mesh.faces[current_face])
            if len(shared_vertices) != 2:
                return None

        transported = _parallel_transport_tangent(
            tangents[previous_segment],
            face_normals[previous_face],
            face_normals[current_face],
        )
        if transported is None:
            return None

        normal = face_normals[current_face]
        incoming = _unit_in_face(transported, normal)
        outgoing = _unit_in_face(tangents[current_segment], normal)
        if incoming is None or outgoing is None:
            return None
        turn_angles.append(
            float(
                np.arctan2(
                    np.dot(normal, np.cross(incoming, outgoing)),
                    np.clip(np.dot(incoming, outgoing), -1.0, 1.0),
                )
            )
        )
        turn_scales.append(0.5 * (segment_lengths[previous_segment] + segment_lengths[current_segment]))

    turn_angle = np.asarray(turn_angles, dtype=float)
    sample_weight = np.asarray(turn_scales, dtype=float)
    curvature = turn_angle / sample_weight
    if closed:
        arclength = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
        curvature_variation = (np.roll(curvature, -1) - curvature) / segment_lengths
        curvature_second_derivative = _closed_curvature_second_derivative(curvature, segment_lengths)
        high_frequency = _closed_high_frequency_curvature(curvature, segment_lengths)
    else:
        arclength = np.cumsum(segment_lengths)[:-1]
        curvature_variation = (
            np.gradient(curvature, arclength, edge_order=1) if len(curvature) > 1 else np.zeros_like(curvature)
        )
        curvature_second_derivative = _open_curvature_second_derivative(curvature, arclength)
        high_frequency = _open_high_frequency_curvature(curvature, arclength)

    heading = np.cumsum(turn_angle)
    heading -= heading[0]
    return IntrinsicAngleTrace(
        arclength_m=arclength,
        heading_angle_rad=heading,
        turn_angle_rad=turn_angle,
        geodesic_curvature_per_m=curvature,
        curvature_variation_per_m2=curvature_variation,
        curvature_second_derivative_per_m3=curvature_second_derivative,
        high_frequency_curvature_per_m=high_frequency,
        sample_weight_m=sample_weight,
        closed=closed,
    )


def _unit_face_normals(mesh: Mesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    result = np.zeros_like(normals)
    valid = lengths > 1.0e-14
    result[valid] = normals[valid] / lengths[valid, None]
    return result


def _parallel_transport_tangent(
    tangent: np.ndarray,
    source_normal: np.ndarray,
    target_normal: np.ndarray,
) -> np.ndarray | None:
    axis = np.cross(source_normal, target_normal)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(np.dot(source_normal, target_normal), -1.0, 1.0))
    if sine <= 1.0e-14:
        return tangent.copy() if cosine > 0.0 else None
    axis /= sine
    return tangent * cosine + np.cross(axis, tangent) * sine + axis * np.dot(axis, tangent) * (1.0 - cosine)


def _unit_in_face(vector: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = vector - np.dot(vector, normal) * normal
    length = float(np.linalg.norm(projected))
    return None if length <= 1.0e-14 else projected / length


def _closed_high_frequency_curvature(curvature: np.ndarray, segment_lengths: np.ndarray) -> np.ndarray:
    previous_distance = np.roll(segment_lengths, 1)
    next_distance = segment_lengths
    baseline = (next_distance * np.roll(curvature, 1) + previous_distance * np.roll(curvature, -1)) / (
        previous_distance + next_distance
    )
    return curvature - baseline


def _closed_curvature_second_derivative(curvature: np.ndarray, segment_lengths: np.ndarray) -> np.ndarray:
    previous_distance = np.roll(segment_lengths, 1)
    next_distance = segment_lengths
    previous_slope = (curvature - np.roll(curvature, 1)) / previous_distance
    next_slope = (np.roll(curvature, -1) - curvature) / next_distance
    return 2.0 * (next_slope - previous_slope) / (previous_distance + next_distance)


def _open_high_frequency_curvature(curvature: np.ndarray, arclength: np.ndarray) -> np.ndarray:
    residual = np.zeros_like(curvature)
    if len(curvature) < 3:
        return residual
    previous_distance = arclength[1:-1] - arclength[:-2]
    next_distance = arclength[2:] - arclength[1:-1]
    baseline = (next_distance * curvature[:-2] + previous_distance * curvature[2:]) / (
        previous_distance + next_distance
    )
    residual[1:-1] = curvature[1:-1] - baseline
    return residual


def _open_curvature_second_derivative(curvature: np.ndarray, arclength: np.ndarray) -> np.ndarray:
    derivative = np.zeros_like(curvature)
    if len(curvature) < 3:
        return derivative
    previous_distance = arclength[1:-1] - arclength[:-2]
    next_distance = arclength[2:] - arclength[1:-1]
    previous_slope = (curvature[1:-1] - curvature[:-2]) / previous_distance
    next_slope = (curvature[2:] - curvature[1:-1]) / next_distance
    derivative[1:-1] = 2.0 * (next_slope - previous_slope) / (previous_distance + next_distance)
    return derivative


def _angle_stability_metrics(traces: list[IntrinsicAngleTrace], contour_count: int) -> dict[str, object]:
    if not traces:
        return {
            "contour_angle_analyzed_count": 0,
            "contour_angle_skipped_count": int(contour_count),
            "contour_intrinsic_turn_rms_deg": 0.0,
            "contour_intrinsic_turn_p95_abs_deg": 0.0,
            "contour_geodesic_curvature_rms_per_m": 0.0,
            "contour_curvature_variation_rms_per_m2": 0.0,
            "contour_curvature_second_derivative_rms_per_m3": 0.0,
            "contour_high_frequency_curvature_rms_per_m": 0.0,
            "contour_high_frequency_curvature_fraction": 0.0,
            "contour_worst_high_frequency_curvature_rms_per_m": 0.0,
        }

    weights = np.concatenate([trace.sample_weight_m for trace in traces])
    turns = np.concatenate([trace.turn_angle_rad for trace in traces])
    curvature = np.concatenate([trace.geodesic_curvature_per_m for trace in traces])
    variation = np.concatenate([trace.curvature_variation_per_m2 for trace in traces])
    second_derivative = np.concatenate([trace.curvature_second_derivative_per_m3 for trace in traces])
    high_frequency = np.concatenate([trace.high_frequency_curvature_per_m for trace in traces])
    curvature_rms = _weighted_rms(curvature, weights)
    high_frequency_rms = _weighted_rms(high_frequency, weights)
    worst_high_frequency = max(
        _weighted_rms(trace.high_frequency_curvature_per_m, trace.sample_weight_m) for trace in traces
    )
    return {
        "contour_angle_analyzed_count": int(len(traces)),
        "contour_angle_skipped_count": int(contour_count - len(traces)),
        "contour_intrinsic_turn_rms_deg": float(np.degrees(_weighted_rms(turns, weights))),
        "contour_intrinsic_turn_p95_abs_deg": float(np.degrees(np.percentile(np.abs(turns), 95.0))),
        "contour_geodesic_curvature_rms_per_m": float(curvature_rms),
        "contour_curvature_variation_rms_per_m2": float(_weighted_rms(variation, weights)),
        "contour_curvature_second_derivative_rms_per_m3": float(_weighted_rms(second_derivative, weights)),
        "contour_high_frequency_curvature_rms_per_m": float(high_frequency_rms),
        "contour_high_frequency_curvature_fraction": float(
            high_frequency_rms / curvature_rms if curvature_rms > 1.0e-14 else 0.0
        ),
        "contour_worst_high_frequency_curvature_rms_per_m": float(worst_high_frequency),
    }


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    return 0.0 if weight_sum <= 0.0 else float(np.sqrt(np.sum(weights * np.square(values)) / weight_sum))
