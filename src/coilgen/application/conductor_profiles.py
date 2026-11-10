"""Conductor-profile editing contract in metres and physical local axes.

Coordinate 0 is surface-transverse (across the wire); coordinate 1 is the
support-surface normal. Profiles remain relative to the configured wire path:
editing never recentres them or changes the submitted winding direction.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConductorProfile:
    points_m: tuple[tuple[float, float], ...]
    area_m2: float
    width_m: float
    height_m: float
    centroid_m: tuple[float, float]
    clockwise: bool

    @property
    def bounds_min_m(self) -> tuple[float, float]:
        return tuple(min(point[axis] for point in self.points_m) for axis in (0, 1))

    @property
    def bounds_max_m(self) -> tuple[float, float]:
        return tuple(max(point[axis] for point in self.points_m) for axis in (0, 1))


def is_rectangular_profile(points: Sequence[Sequence[float]]) -> bool:
    """Recognize the legacy rectangle sentinel, without treating bad input as one."""
    try:
        values = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(values.shape == (1, 2) and np.all(values == 0.0))


def canonical_profile_points(points: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    """Return SI row-wise points, accepting the engine's legacy 2-by-N format.

    A repeated final point closes the polygon implicitly and is removed only
    when exactly equal to the first. All remaining coordinates/order are exact.
    This function also preserves the rectangle sentinel.
    """
    try:
        values = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Conductor coordinates must be numeric.") from exc
    if values.ndim != 2:
        raise ValueError("Conductor coordinates require N points with two coordinates each.")
    if values.shape[0] == 2 and values.shape[1] != 2:
        values = values.T
    if values.shape[1:] != (2,):
        raise ValueError("Conductor coordinates require N points with two coordinates each.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Conductor coordinates must be finite.")
    if len(values) > 1 and np.array_equal(values[0], values[-1]):
        values = values[:-1]
    return tuple((float(point[0]), float(point[1])) for point in values)


MIN_CIRCLE_SEGMENTS = 6
MAX_CIRCLE_SEGMENTS = 256
DEFAULT_CIRCLE_SEGMENTS = 16


def _circle_segments(segments) -> int:
    try:
        whole = not isinstance(segments, bool) and int(segments) == segments
    except (TypeError, ValueError, OverflowError):
        whole = False
    if not whole:
        raise ValueError("The number of circle points must be a whole number.")
    segments = int(segments)
    if not MIN_CIRCLE_SEGMENTS <= segments <= MAX_CIRCLE_SEGMENTS:
        raise ValueError(
            f"Use between {MIN_CIRCLE_SEGMENTS} and {MAX_CIRCLE_SEGMENTS} points around a circle (got {segments})."
        )
    return segments


def equal_area_radius_scale(segments: int) -> float:
    """Radius factor that gives a regular inscribed ``segments``-gon the circle's area.

    An inscribed polygon underestimates the circle area (16 points: 97.4 %,
    8 points: 90.0 %, 6 points: 82.7 %), and therefore overestimates the DC
    resistance computed from the profile area.
    """
    segments = _circle_segments(segments)
    return float(np.sqrt(2.0 * np.pi / (segments * np.sin(2.0 * np.pi / segments))))


def round_wire_profile(
    diameter_m: float, segments: int = DEFAULT_CIRCLE_SEGMENTS, *, equal_area: bool = False
) -> tuple[tuple[float, float], ...]:
    """Regular polygon approximating a round (litz) wire centred on the origin.

    ``segments`` is the number of polygon points; fewer points give smaller
    swept STL files. With ``equal_area`` the polygon is enlarged slightly so
    its area (used for resistance) equals that of the round wire.
    """
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("A round profile needs a positive diameter.")
    segments = _circle_segments(segments)
    radius = diameter_m / 2.0 * (equal_area_radius_scale(segments) if equal_area else 1.0)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    return _snap_round_off(((radius * np.cos(angle), radius * np.sin(angle)) for angle in angles), radius)


def round_wire_options(points, diameter_m):
    """Recognize a saved round preset by exact geometry, preserving custom polygons."""
    try:
        points = canonical_profile_points(points)
    except ValueError:
        return None
    segments = len(points)
    if not MIN_CIRCLE_SEGMENTS <= segments <= MAX_CIRCLE_SEGMENTS or not np.isfinite(diameter_m) or diameter_m <= 0:
        return None
    for equal_area in (False, True):
        if points == round_wire_profile(diameter_m, segments, equal_area=equal_area):
            return segments, equal_area
    return None


def _snap_round_off(points, scale) -> tuple[tuple[float, float], ...]:
    """Replace trigonometric round-off such as cos(pi/2) = 6e-17 by exact zeros."""
    tolerance = 1e-12 * scale
    return tuple(tuple(0.0 if abs(value) < tolerance else float(value) for value in point) for point in points)


def u_shape_profile(
    width_m: float,
    height_m: float,
    segments: int = DEFAULT_CIRCLE_SEGMENTS,
    *,
    arc_side: str = "top",
) -> tuple[tuple[float, float], ...]:
    """Rectangle with a half circle on one side (a "U" or "D" profile).

    The half circle has diameter ``width_m``; ``height_m`` is the total height,
    so ``height_m == width_m / 2`` gives a plain half disc. ``segments`` is the
    number of points around a *full* circle, i.e. the half circle uses
    ``segments // 2`` straight segments (the same angular resolution as the
    round wire). ``arc_side="top"`` points the arc away from the support
    (+n, flat side on the support); ``"bottom"`` points it towards the support.
    The profile is centred on its bounding box, like the rectangle.
    """
    if not all(np.isfinite(value) and value > 0 for value in (width_m, height_m)):
        raise ValueError("A U-shape profile needs a positive width and height.")
    if arc_side not in {"top", "bottom"}:
        raise ValueError("The U-shape arc side must be 'top' or 'bottom'.")
    radius = width_m / 2.0
    straight = height_m - radius
    tolerance = 1e-9 * height_m
    if straight < -tolerance:
        raise ValueError("A U-shape needs a height of at least half its width (the half circle radius).")
    arc_segments = _circle_segments(segments) // 2
    base = -height_m / 2.0
    straight = straight if straight > tolerance else 0.0
    centre = base + straight
    points = [(-radius, base), (radius, base)]
    for angle in np.linspace(0.0, np.pi, arc_segments + 1):
        # A plain half disc shares its arc end points with the base corners.
        if straight > 0.0 or 0.0 < angle < np.pi:
            points.append((radius * np.cos(angle), centre + radius * np.sin(angle)))
    if arc_side == "bottom":
        points = [(u, -n) for u, n in reversed(points)]
    return _snap_round_off(points, height_m)


def parse_profile_table(text: str, *, length_scale_to_m: float = 1e-3) -> tuple[tuple[float, float], ...]:
    """Read a two-column (u, n) point table, e.g. pasted from a spreadsheet.

    Accepted separators are ``;``, tab, ``,`` or whitespace. With ``;`` or tab
    separators a decimal comma is also accepted (German spreadsheet export).
    Empty lines and lines starting with ``#`` are ignored; a non-numeric first
    line is treated as a header. Values are multiplied by
    ``length_scale_to_m`` (default: millimetres). Geometry is not validated
    here, so a faulty table can still be loaded and corrected.
    """
    if not np.isfinite(length_scale_to_m) or length_scale_to_m <= 0:
        raise ValueError("The table length scale must be positive.")
    points = []
    header_allowed = True
    for number, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ";" in line or "\t" in line:
            cells = [cell.strip().replace(",", ".") for cell in line.replace("\t", ";").split(";")]
        elif "," in line:
            cells = [cell.strip() for cell in line.split(",")]
        else:
            cells = line.split()
        cells = [cell for cell in cells if cell]
        try:
            values = [float(cell) for cell in cells]
        except ValueError:
            if header_allowed:
                header_allowed = False
                continue
            raise ValueError(f"Line {number}: expected two numbers (u, n), got {line!r}.") from None
        header_allowed = False
        if len(values) != 2:
            raise ValueError(f"Line {number}: expected two columns (u, n), found {len(values)}.")
        if not all(np.isfinite(values)):
            raise ValueError(f"Line {number}: coordinates must be finite.")
        points.append((values[0] * length_scale_to_m, values[1] * length_scale_to_m))
    if len(points) < 3:
        raise ValueError("A profile table needs at least three rows of u, n coordinates.")
    return canonical_profile_points(points)


def read_profile_table(path, *, length_scale_to_m: float = 1e-3) -> tuple[tuple[float, float], ...]:
    """Read a :func:`parse_profile_table` point table from a CSV/TSV/TXT file."""
    from pathlib import Path

    content = Path(path).read_bytes()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    return parse_profile_table(text, length_scale_to_m=length_scale_to_m)


def omega_holder_profile(
    wire_diameter_m: float,
    wall_m: float = 0.0008,
    base_width_m: float | None = None,
    opening_angle_deg: float = 70.0,
    segments: int = 20,
) -> tuple[tuple[float, float], ...]:
    """Printable omega-shaped holder around a round wire at the origin.

    The holder is a ring of thickness ``wall_m`` around the wire channel, open
    away from the support (+n) by ``opening_angle_deg`` so the wire can be
    snapped in, standing on a flat foot of ``base_width_m`` on the support (-n).
    Sweep it with ``conductor_path_reference = "profile_origin"`` so the
    exported path and the field source follow the wire centre.
    """
    if not (np.isfinite(wire_diameter_m) and wire_diameter_m > 0 and np.isfinite(wall_m) and wall_m > 0):
        raise ValueError("Wire diameter and wall thickness must be positive.")
    if not 10.0 <= opening_angle_deg <= 160.0:
        raise ValueError("The omega opening angle must lie between 10 and 160 degrees.")
    inner = wire_diameter_m / 2.0
    outer = inner + wall_m
    foot = base_width_m if base_width_m is not None else 2.0 * outer + 2.0 * wall_m
    half_gap = np.radians(opening_angle_deg) / 2.0
    # Outer arc from the right lip, around the bottom, to the left lip (clockwise in u-n).
    start, stop = np.pi / 2 - half_gap, -3 * np.pi / 2 + half_gap
    outer_arc = [(outer * np.cos(t), outer * np.sin(t)) for t in np.linspace(start, stop, segments)]
    inner_arc = [(inner * np.cos(t), inner * np.sin(t)) for t in np.linspace(stop, start, segments)]
    bottom = -outer
    lowest = [index for index, point in enumerate(outer_arc) if point[1] <= -outer * np.cos(np.radians(35))]
    first, last = lowest[0], lowest[-1]
    footprint = [
        (foot / 2.0, outer_arc[first][1]),
        (foot / 2.0, bottom - wall_m),
        (-foot / 2.0, bottom - wall_m),
        (-foot / 2.0, outer_arc[last][1]),
    ]
    points = outer_arc[: first + 1] + footprint + outer_arc[last:] + inner_arc
    return tuple((float(u), float(n)) for u, n in points)


def validate_conductor_profile(
    points: Sequence[Sequence[float]], *, width_m: float | None = None, height_m: float | None = None
) -> ConductorProfile:
    """Validate a sweepable simple profile and report physical dimensions.

    Rectangle-sentinel input requires positive finite ``width_m``/``height_m``.
    Width and height are only used for that sentinel; custom coordinates define
    the physical profile. Concave simple polygons (e.g. omega holders) are
    accepted; open-path end caps are then ear-clipped instead of fanned.
    Geometry tolerances match the sweep: max(1e-12 * span, 1e-15) metres
    for edge lengths and max(1e-12 * span**2, 1e-30) square metres for area.
    Segment contact checks also detect collinear overlaps and touching edges.
    """
    values = canonical_profile_points(points)
    if is_rectangular_profile(values):
        try:
            dimensions = np.asarray((width_m, height_m), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Rectangle width and height must be positive finite lengths.") from exc
        if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
            raise ValueError("Rectangle width and height must be positive finite lengths.")
        width, height = dimensions
        values = (
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        )
    profile = np.asarray(values, dtype=float)
    if len(profile) < 3:
        raise ValueError("A conductor profile requires at least three distinct points.")
    with np.errstate(over="ignore", invalid="ignore"):
        extent = np.ptp(profile, axis=0)
    if not np.all(np.isfinite(extent)) or np.max(extent) > np.sqrt(np.finfo(float).max) / 4:
        raise ValueError("Conductor coordinate span is too large for reliable geometric calculations.")
    scale = max(float(np.max(extent)), 1e-15)
    length_tolerance = max(1e-12 * scale, 1e-15)
    area_tolerance = max(1e-12 * scale**2, 1e-30)
    edges = np.roll(profile, -1, axis=0) - profile
    if np.any(np.linalg.norm(edges, axis=1) <= length_tolerance):
        raise ValueError("Conductor profile contains a zero-length or repeated segment.")
    for first in range(len(profile)):
        for second in range(first + 1, len(profile)):
            if (first + 1) % len(profile) == second or (second + 1) % len(profile) == first:
                continue
            if _segments_contact(
                profile[first],
                profile[(first + 1) % len(profile)],
                profile[second],
                profile[(second + 1) % len(profile)],
                length_tolerance,
                area_tolerance,
            ):
                raise ValueError("Conductor segments intersect, overlap, or touch away from a shared endpoint.")
    relative = profile - profile[0]
    following = np.roll(relative, -1, axis=0)
    cross = relative[:, 0] * following[:, 1] - relative[:, 1] * following[:, 0]
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) <= area_tolerance:
        raise ValueError("Conductor profile has zero or numerically negligible area.")
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(np.abs(turns) <= area_tolerance):
        raise ValueError("Remove collinear profile points: every corner must turn.")
    # Preserve the scientific kernel's acceptance boundary; this call does not
    # mutate submitted points even when the sweep internally reverses winding.
    from coilgen.manufacturing import _validated_cross_section

    _validated_cross_section(profile)
    centroid = profile[0] + np.sum((relative + following) * cross[:, None], axis=0) / (6 * signed_area)
    return ConductorProfile(
        values,
        abs(signed_area),
        float(extent[0]),
        float(extent[1]),
        tuple(float(coordinate) for coordinate in centroid),
        signed_area < 0,
    )


def _segments_contact(a, b, c, d, length_tolerance, area_tolerance) -> bool:
    def orientation(p, q, r):
        edge, offset = q - p, r - p
        return float(edge[0] * offset[1] - edge[1] * offset[0])

    def on_segment(p, q, point):
        return bool(
            np.all(point >= np.minimum(p, q) - length_tolerance)
            and np.all(point <= np.maximum(p, q) + length_tolerance)
        )

    orientations = (orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b))
    if (
        (orientations[0] > area_tolerance and orientations[1] < -area_tolerance)
        or (orientations[0] < -area_tolerance and orientations[1] > area_tolerance)
    ) and (
        (orientations[2] > area_tolerance and orientations[3] < -area_tolerance)
        or (orientations[2] < -area_tolerance and orientations[3] > area_tolerance)
    ):
        return True
    return any(
        abs(value) <= area_tolerance and on_segment(p, q, point)
        for value, p, q, point in (
            (orientations[0], a, b, c),
            (orientations[1], a, b, d),
            (orientations[2], c, d, a),
            (orientations[3], c, d, b),
        )
    )
