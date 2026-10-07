"""Arclength-weighted, oriented frames for swept conductor paths.

The surface guide is signed: reversing it changes an asymmetric conductor or
its mounting side. Free spans use double-reflection rotation-minimizing
transport. Guided spans minimize a one-dimensional connection energy; the
linear systems are tridiagonal (cyclic for closed paths), never dense.
"""

from __future__ import annotations

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-15)


def transverse_guide(tangent: np.ndarray, guide: np.ndarray) -> np.ndarray:
    normal = guide - tangent * np.dot(tangent, guide)
    if np.linalg.norm(normal) < 1e-10:
        axis = np.eye(3)[int(np.argmin(np.abs(tangent)))]
        normal = axis - tangent * np.dot(tangent, axis)
    return unit(normal)


def transport_normal(
    normal: np.ndarray, first_tangent: np.ndarray, last_tangent: np.ndarray, chord: np.ndarray
) -> np.ndarray:
    """Double-reflection RMF step, with finite coincident/reversal limits."""
    chord_sq = float(np.dot(chord, chord))
    if chord_sq <= 1e-30:
        return transverse_guide(last_tangent, normal)
    reflected_normal = normal - 2.0 * np.dot(chord, normal) / chord_sq * chord
    reflected_tangent = first_tangent - 2.0 * np.dot(chord, first_tangent) / chord_sq * chord
    second = last_tangent - reflected_tangent
    second_sq = float(np.dot(second, second))
    if second_sq > 1e-30:
        reflected_normal -= 2.0 * np.dot(second, reflected_normal) / second_sq * second
    return transverse_guide(last_tangent, reflected_normal)


def _tridiagonal_solve(lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, right: np.ndarray) -> np.ndarray:
    diagonal = diagonal.astype(complex, copy=True)
    output = np.array(right, dtype=complex, copy=True)
    for index in range(1, len(diagonal)):
        multiplier = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= multiplier * upper[index - 1]
        output[index] -= multiplier * output[index - 1]
    output[-1] /= diagonal[-1]
    for index in range(len(diagonal) - 2, -1, -1):
        output[index] = (output[index] - upper[index] * output[index + 1]) / diagonal[index]
    return output


def _connection_solve(
    diagonal: np.ndarray, lower: np.ndarray, right: np.ndarray, closing_lower: complex | None
) -> np.ndarray:
    upper = np.conjugate(lower)
    if closing_lower is None:
        return _tridiagonal_solve(lower, diagonal, upper, right)
    # The two corner entries are a rank-two update of the open tridiagonal.
    extra = np.zeros((len(diagonal), 3), dtype=complex)
    extra[:, 0] = right
    extra[0, 1] = 1.0
    extra[-1, 2] = 1.0
    solved = _tridiagonal_solve(lower, diagonal, upper, extra)
    edge = complex(closing_lower)
    small = np.eye(2, dtype=complex) + np.array(
        [
            edge * solved[-1, 1:],
            np.conjugate(edge) * solved[0, 1:],
        ]
    )
    correction = np.linalg.solve(small, np.array([edge * solved[-1, 0], np.conjugate(edge) * solved[0, 0]]))
    return solved[:, 0] - solved[:, 1:] @ correction


def free_transport_frames(
    points: np.ndarray, tangents: np.ndarray, guides: np.ndarray, *, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Pure RMF, with a geometrically chosen seam for an unconstrained loop."""
    count = len(points)
    start = int(np.lexsort(points.T[::-1])[0]) if closed else 0
    order = (np.arange(count) + start) % count
    normals = np.zeros_like(points)
    normals[order[0]] = transverse_guide(tangents[order[0]], guides[order[0]])
    for first, last in zip(order[:-1], order[1:]):
        normals[last] = transport_normal(normals[first], tangents[first], tangents[last], points[last] - points[first])
    if closed and count > 2:
        first, last = int(order[0]), int(order[-1])
        closing = transport_normal(normals[last], tangents[last], tangents[first], points[first] - points[last])
        residual = np.arctan2(
            np.dot(tangents[first], np.cross(closing, normals[first])), np.dot(closing, normals[first])
        )
        lengths = np.linalg.norm(points[np.roll(order, -1)] - points[order], axis=1)
        distance = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
        fractions = distance / max(float(lengths.sum()), 1e-15)
        for index, fraction in zip(order, fractions):
            angle = residual * fraction
            normal = normals[index]
            normals[index] = np.cos(angle) * normal + np.sin(angle) * np.cross(tangents[index], normal)
    return normals, np.cross(tangents, normals)


def _interpolate_free_spans(
    points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    attached: np.ndarray,
    *,
    closed: bool,
) -> np.ndarray:
    """Interpolate material roll by arclength between signed attachments.

    Normalizing a harmonic complex chord is singular at opposite endpoint
    directors. The transported endpoint angle has a regular constant-twist
    solution, including an antipodal pair (either half-turn is admissible).
    """
    anchors = np.flatnonzero(attached)
    if not len(anchors) or len(anchors) == len(points):
        return normals
    count = len(points)
    pairs = list(zip(anchors[:-1], anchors[1:]))
    if closed:
        pairs.append((anchors[-1], anchors[0] + count))
    for first, last in pairs:
        if last - first <= 1:
            continue
        indices = np.arange(first, last + 1) % count
        transported = np.empty((len(indices), 3))
        transported[0] = normals[first]
        distance = np.zeros(len(indices))
        for step, (a, b) in enumerate(zip(indices[:-1], indices[1:]), 1):
            chord = points[b] - points[a]
            transported[step] = transport_normal(transported[step - 1], tangents[a], tangents[b], chord)
            distance[step] = distance[step - 1] + np.linalg.norm(chord)
        end = indices[-1]
        sine = float(np.dot(tangents[end], np.cross(transported[-1], normals[end])))
        cosine = float(np.dot(transported[-1], normals[end]))
        angle = np.pi if abs(sine) < 1e-12 and cosine < 0.0 else np.arctan2(sine, cosine)
        roll = angle * distance[1:-1] / max(distance[-1], 1e-15)
        middle = indices[1:-1]
        normals[middle] = np.cos(roll)[:, None] * transported[1:-1] + np.sin(roll)[:, None] * np.cross(
            tangents[middle], transported[1:-1]
        )
    if not closed:
        # A free tail has only one director constraint, hence pure RMF.
        for indices in (range(int(anchors[0]), 0, -1), range(int(anchors[-1]), count - 1)):
            direction = -1 if indices.step < 0 else 1
            for first in indices:
                last = first + direction
                normals[last] = transport_normal(
                    normals[first],
                    tangents[first],
                    tangents[last],
                    points[last] - points[first],
                )
    return normals


def constrained_frames(
    points: np.ndarray,
    tangents: np.ndarray,
    guides: np.ndarray,
    surface_segments: np.ndarray,
    strength: float,
    length_scale: float,
    *,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize transport variation plus signed surface-guide attachment.

    Edge energy is weighted by inverse arclength. Guide attachment uses the
    adjacent SURFACE half-interval lengths, so adding samples does not multiply
    the physical guide strength. Strength one fixes attached frames exactly;
    strength zero is pure RMF. The roll field is normalized after solving.
    """
    count = len(points)
    if count < 2:
        normals = np.array([transverse_guide(tangent, guide) for tangent, guide in zip(tangents, guides)]).reshape(
            (-1, 3)
        )
        return normals, np.cross(tangents, normals)
    if strength <= 0.0 or not np.any(surface_segments):
        return free_transport_frames(points, tangents, guides, closed=closed)
    normals = np.array([transverse_guide(tangent, guide) for tangent, guide in zip(tangents, guides)])
    transverse = np.cross(tangents, normals)
    starts = np.arange(count if closed else count - 1)
    ends = (starts + 1) % count
    lengths = np.linalg.norm(points[ends] - points[starts], axis=1)
    mass = np.zeros(count)
    np.add.at(mass, starts, 0.5 * lengths * surface_segments)
    np.add.at(mass, ends, 0.5 * lengths * surface_segments)
    if strength >= 1.0:
        normals = _interpolate_free_spans(points, tangents, normals, mass > 0.0, closed=closed)
        return normals, np.cross(tangents, normals)
    weights = 1.0 / np.maximum(lengths, 1e-12)
    phases = np.empty(len(starts), dtype=complex)
    for edge, (first, last) in enumerate(zip(starts, ends)):
        transported = transport_normal(normals[first], tangents[first], tangents[last], points[last] - points[first])
        phases[edge] = complex(np.dot(transported, normals[last]), np.dot(transported, transverse[last]))
    attachment = mass * (strength / max(1.0 - strength, 1e-12)) / max(length_scale, 1e-9) ** 2
    diagonal = attachment.copy()
    np.add.at(diagonal, starts, weights)
    np.add.at(diagonal, ends, weights)
    if np.all(attachment <= 8.0 * np.finfo(float).eps * diagonal):
        # Below floating-point resolution the open connection Laplacian loses
        # its only gauge constraint. Use the strength-zero limit, not a singular
        # Thomas solve followed by normalization of NaNs.
        return free_transport_frames(points, tangents, guides, closed=closed)
    lower = -weights[: count - 1] * phases[: count - 1]
    closing = -weights[-1] * phases[-1] if closed else None
    right = attachment.astype(complex)
    rolls = _connection_solve(diagonal, lower, right, closing)
    magnitudes = np.abs(rolls)
    rolls = np.divide(rolls, magnitudes, out=np.ones_like(rolls), where=magnitudes > 1e-12)
    # A guide is an oriented mounting constraint, not an unoriented line field.
    # A conflicting soft solution may not put an attached profile inside-out.
    rolls[(mass > 0.0) & (rolls.real <= 0.0)] = 1.0
    normals = rolls.real[:, None] * normals + rolls.imag[:, None] * transverse
    normals = _interpolate_free_spans(points, tangents, normals, mass > 0.0, closed=closed)
    return normals, np.cross(tangents, normals)
