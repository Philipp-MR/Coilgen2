from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from .config import CoilGenConfig
from .models import CoilGenResult, CoilPart, PcbLayout, PcbWirePart


def generate_cylindrical_pcb_print(parts: list[CoilPart], config: CoilGenConfig) -> list[CoilPart]:
    """Generate 2D PCB artwork by unrolling cylindrical wire paths."""

    if not (config.surface_is_cylinder_flag and config.make_cylndrical_pcb):
        for part in parts:
            part.pcb_layout = None
        return parts

    for part in parts:
        if part.wire_path is None or len(part.wire_path.v) < 2:
            part.pcb_layout = None
            continue
        frame = cylindrical_unwrap_frame(part)
        track_width = _track_width(config)
        boundary = unwrap_points(part.mesh.vertices, frame)
        upper: list[PcbWirePart] = []
        lower: list[PcbWirePart] = []

        if config.pcb_interconnection_method == "spiral_in_out" and part.connected_groups:
            for group in part.connected_groups:
                upper.extend(_wire_parts_from_centerline(unwrap_points(group.spiral_in_v, frame), track_width))
                lower.extend(_wire_parts_from_centerline(unwrap_points(group.spiral_out_v, frame), track_width))
        else:
            upper.extend(_wire_parts_from_centerline(unwrap_points(part.wire_path.v, frame), track_width))

        part.pcb_layout = PcbLayout(
            upper_layer=upper,
            lower_layer=lower,
            boundary=_boundary_rectangle(boundary),
            track_width=track_width,
            cylinder_radius=frame.radius,
        )
    return parts


def save_pcb_layouts_as_svg(result: CoilGenResult, output_dir: str | Path, *, prefix: str = "pcb_layout") -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        if part.pcb_layout is None:
            continue
        saved.append(
            _save_single_pcb_svg(
                part.pcb_layout, output / f"{prefix}_part{part_ind}.svg", title=f"{prefix} part {part_ind}"
            )
        )
    return saved


def save_pcb_layouts_as_dxf(result: CoilGenResult, output_dir: str | Path, *, prefix: str = "pcb_layout") -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        if part.pcb_layout is None:
            continue
        saved.append(_save_single_pcb_dxf(part.pcb_layout, output / f"{prefix}_part{part_ind}.dxf"))
    return saved


def save_pcb_centerlines_as_csv(
    result: CoilGenResult, output_dir: str | Path, *, prefix: str = "pcb_centerlines"
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for part_ind, part in enumerate(result.coil_parts, start=1):
        if part.pcb_layout is None:
            continue
        path = output / f"{prefix}_part{part_ind}.csv"
        lines = ["part,layer,wire_part,point_index,x_m,y_m"]
        for layer_name, layer_parts in (("upper", part.pcb_layout.upper_layer), ("lower", part.pcb_layout.lower_layer)):
            for wire_ind, wire_part in enumerate(layer_parts, start=1):
                for point_ind, point in enumerate(wire_part.centerline, start=1):
                    lines.append(f"{part_ind},{layer_name},{wire_ind},{point_ind},{point[0]:.12g},{point[1]:.12g}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        saved.append(path)
    return saved


class CylindricalUnwrapFrame:
    def __init__(self, origin: np.ndarray, axis_z: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray, radius: float):
        self.origin = origin
        self.axis_z = axis_z
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.radius = radius


def cylindrical_unwrap_frame(part: CoilPart) -> CylindricalUnwrapFrame:
    vertices = part.mesh.vertices
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axis_z = axes[0]
    radial = centered - np.outer(centered @ axis_z, axis_z)
    _, _, radial_axes = np.linalg.svd(radial, full_matrices=False)
    axis_x = radial_axes[0]
    axis_y = np.cross(axis_z, axis_x)
    axis_y /= max(np.linalg.norm(axis_y), 1e-15)
    x = centered @ axis_x
    y = centered @ axis_y
    radius = float(np.median(np.hypot(x, y)))
    return CylindricalUnwrapFrame(vertices.mean(axis=0), axis_z, axis_x, axis_y, radius)


def unwrap_points(points: np.ndarray, frame: CylindricalUnwrapFrame) -> np.ndarray:
    centered = np.asarray(points, dtype=float) - frame.origin
    x = centered @ frame.axis_x
    y = centered @ frame.axis_y
    z = centered @ frame.axis_z
    theta = np.unwrap(np.arctan2(y, x))
    return np.column_stack((theta * frame.radius, z))


def _wire_parts_from_centerline(centerline: np.ndarray, track_width: float) -> list[PcbWirePart]:
    if len(centerline) < 2:
        return []
    parts = _split_large_wrap_jumps(centerline)
    return [
        PcbWirePart(centerline=part, track_shape=_track_polygon(part, track_width)) for part in parts if len(part) >= 2
    ]


def _split_large_wrap_jumps(centerline: np.ndarray) -> list[np.ndarray]:
    if len(centerline) < 2:
        return [centerline]
    jumps = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    median = float(np.median(jumps[jumps > 0])) if np.any(jumps > 0) else 0.0
    if median <= 0:
        return [centerline]
    cut_after = np.where(jumps > 8.0 * median)[0]
    if len(cut_after) == 0:
        return [centerline]
    parts = []
    start = 0
    for cut in cut_after:
        parts.append(centerline[start : cut + 1])
        start = cut + 1
    if start < len(centerline):
        parts.append(centerline[start:])
    return parts


def _track_polygon(centerline: np.ndarray, track_width: float) -> np.ndarray:
    if len(centerline) < 2:
        return centerline.copy()
    tangents = np.zeros_like(centerline)
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    if len(centerline) > 2:
        tangents[1:-1] = centerline[2:] - centerline[:-2]
    lengths = np.linalg.norm(tangents, axis=1)
    tangents[lengths > 0] /= lengths[lengths > 0, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals[normal_lengths > 0] /= normal_lengths[normal_lengths > 0, None]
    offset = normals * (track_width / 2.0)
    polygon = np.vstack((centerline + offset, (centerline - offset)[::-1], centerline[0] + offset[0]))
    return polygon


def _boundary_rectangle(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=float)
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)
    return np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]], dtype=float)


def _track_width(config: CoilGenConfig) -> float:
    return max(float(config.conductor_cross_section_width), 1e-9)


def _save_single_pcb_svg(layout: PcbLayout, path: Path, *, title: str) -> Path:
    all_points = [part.track_shape for part in layout.upper_layer + layout.lower_layer]
    if layout.boundary is not None and len(layout.boundary):
        all_points.append(layout.boundary)
    stacked = np.vstack(all_points) if all_points else np.zeros((2, 2), dtype=float)
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    pad = span * 0.05
    mins -= pad
    maxs += pad
    width_mm = max(float((maxs[0] - mins[0]) * 1000.0), 1.0)
    height_mm = max(float((maxs[1] - mins[1]) * 1000.0), 1.0)

    def map_points(points: np.ndarray) -> np.ndarray:
        mapped = (points - mins) * 1000.0
        mapped[:, 1] = height_mm - mapped[:, 1]
        return mapped

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width_mm:.3f}mm" height="{height_mm:.3f}mm" viewBox="0 0 {width_mm:.3f} {height_mm:.3f}">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if layout.boundary is not None and len(layout.boundary):
        boundary = map_points(layout.boundary.copy())
        lines.append(_polyline_svg(boundary, stroke="#666666", stroke_width=0.1, fill="none"))
    for part in layout.upper_layer:
        lines.append(_polygon_svg(map_points(part.track_shape.copy()), fill="#1f6feb"))
    for part in layout.lower_layer:
        lines.append(_polygon_svg(map_points(part.track_shape.copy()), fill="#d73a49"))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _save_single_pcb_dxf(layout: PcbLayout, path: Path) -> Path:
    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$INSUNITS",
        "70",
        "6",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]
    if layout.boundary is not None and len(layout.boundary):
        lines.extend(_dxf_lwpolyline(layout.boundary, "boundary", closed=True))
    for part in layout.upper_layer:
        lines.extend(_dxf_lwpolyline(part.track_shape, "upper", closed=True))
    for part in layout.lower_layer:
        lines.extend(_dxf_lwpolyline(part.track_shape, "lower", closed=True))
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _dxf_lwpolyline(points: np.ndarray, layer: str, *, closed: bool) -> list[str]:
    points = np.asarray(points, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-12:
        points = points[:-1]
    lines = [
        "0",
        "LWPOLYLINE",
        "8",
        layer,
        "90",
        str(len(points)),
        "70",
        "1" if closed else "0",
    ]
    for point in points:
        lines.extend(["10", f"{point[0] * 1000.0:.9g}", "20", f"{point[1] * 1000.0:.9g}"])
    return lines


def _polygon_svg(points: np.ndarray, *, fill: str) -> str:
    coords = " ".join(f"{point[0]:.4f},{point[1]:.4f}" for point in points)
    return f'<polygon points="{coords}" fill="{fill}" stroke="none" shape-rendering="geometricPrecision"/>'


def _polyline_svg(points: np.ndarray, *, stroke: str, stroke_width: float, fill: str) -> str:
    coords = " ".join(f"{point[0]:.4f},{point[1]:.4f}" for point in points)
    return f'<polyline points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:g}"/>'
