"""Render production sweep meshes around the field-validated smooth contours.

The round 1.8 mm profile is an illustration choice, centered on the same
conditioned paths used for the thin-wire field calculation. VTK shades the
actual mesh: it never substitutes a display-only spline or inflated tube.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from coilgen import CoilGenConfig  # noqa: E402
from coilgen.geometry_kernels import geometry_acceleration  # noqa: E402
from coilgen.manufacturing import create_swept_contour_body  # noqa: E402
from coilgen.models import CoilPart, ContourLine, Mesh  # noqa: E402
from coilgen.persistence import json_value, source_digest  # noqa: E402


NAVY = "#091b2b"
PALE = "#eef7ff"
MUTED = "#a4b7c8"
COPPER = "#e8a16f"
TEAL = "#39c9ba"
RADIUS_M = 0.0009
SECTION_SIDES = 32
SWEEP_SETTINGS = {
    "conductor_mounting_mode": "centered",
    "sweep_surface_realization": "analytic",
    "sweep_bend_mode": "miter",
    "sweep_frame_method": "surface_normal",
    "sweep_path_simplification": "off",
    "sweep_max_segment_length": 0.001,
    "sweep_max_chord_error": 0.00001,
}


def _identity(cases: list[dict], info: list[dict]) -> dict:
    inputs = []
    for arrays, case in zip(cases, info):
        digest = sha256()
        for key, value in sorted(arrays.items()):
            array = np.ascontiguousarray(value)
            digest.update(key.encode())
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        inputs.append({"case_arrays_sha256": digest.hexdigest(), "config": case["config"]})
    return {
        "inputs": inputs,
        "core_source_sha256": source_digest(ROOT / "src" / "coilgen"),
        "nominal_profile_radius_m": RADIUS_M,
        "profile_sides": SECTION_SIDES,
        "sweep_settings": SWEEP_SETTINGS,
        "require_valid_independent_bodies": True,
        "geometry_acceleration_backend": "numpy",
    }


def compute_bodies(cases: list[dict], metrics: dict | list) -> tuple[list[list[dict]], dict]:
    """Use the production sweep API, retaining all resulting diagnostics."""
    info = metrics.get("refined_cases", metrics["cases"]) if isinstance(metrics, dict) else metrics
    identity = _identity(cases, info)
    evidence_path = OUTPUT / "swept_bodies.json"
    arrays_path = OUTPUT / "swept_bodies.npz"
    if evidence_path.exists() and arrays_path.exists():
        previous = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            previous.get("geometry_identity") == identity
            and previous.get("mesh_arrays_sha256") == sha256(arrays_path.read_bytes()).hexdigest()
        ):
            with np.load(arrays_path, allow_pickle=False) as archive:
                cached = []
                for case_index, case in enumerate(previous["cases"]):
                    bodies = []
                    for body_index, record in enumerate(case["bodies"]):
                        prefix = f"case_{case_index}_body_{body_index}"
                        bodies.append(
                            {
                                **record,
                                **{
                                    key: archive[f"{prefix}_{key}"]
                                    for key in ("vertices_m", "faces", "source_points_m", "realized_centerline_m")
                                },
                            }
                        )
                    cached.append(bodies)
            print("Reusing verified actual swept conductor meshes.", flush=True)
            return cached, previous

    theta = np.arange(SECTION_SIDES) * 2 * np.pi / SECTION_SIDES
    section = RADIUS_M * np.column_stack((np.cos(theta), np.sin(theta)))
    cache = {"cross_section_m": section}
    all_bodies, records = [], []
    with threadpool_limits(limits=1), geometry_acceleration("numpy"):
        for case_index, (arrays, case) in enumerate(zip(cases, info)):
            config = replace(
                CoilGenConfig(**case["config"]),
                skip_sweep=False,
                progress_flag=False,
                **SWEEP_SETTINGS,
            )
            bodies, summaries = [], []
            for part_index in range(case["part_count"]):
                part = CoilPart(Mesh(arrays[f"part_{part_index}_vertices_m"], arrays[f"part_{part_index}_faces"]))
                keys = sorted(
                    (key for key in arrays if key.startswith(f"part_{part_index}_contour_")),
                    key=lambda key: int(key.split("_")[3]),
                )
                for key in keys:
                    contour_index = int(key.split("_")[3])
                    points = arrays[key]
                    body = create_swept_contour_body(
                        part,
                        ContourLine(points, potential=0.0),
                        config,
                        part_index=part_index + 1,
                        contour_index=contour_index + 1,
                        cross_section=section,
                        validate=True,
                    )
                    mesh = body.mesh
                    if not mesh.n_faces or not np.all(np.isfinite(mesh.vertices)):
                        raise RuntimeError("The core sweep returned an empty or nonfinite conductor body.")
                    if mesh.faces.min() < 0 or mesh.faces.max() >= mesh.n_vertices:
                        raise RuntimeError("The core sweep returned invalid triangle indices.")
                    realized = body.fabrication_path.wire_points_m
                    # The core can insert collinear support samples. Its own
                    # two-way source-displacement audit remains authoritative.
                    summary = {
                        "part_index_zero_based": part_index,
                        "contour_index_zero_based": contour_index,
                        "vertex_count": mesh.n_vertices,
                        "triangle_count": mesh.n_faces,
                        "diagnostics": json_value(body.diagnostics),
                        "centerline_vertex_max_displacement_m": float(np.max(np.linalg.norm(realized - points, axis=1)))
                        if realized.shape == points.shape
                        else None,
                    }
                    geometry = {
                        "vertices_m": mesh.vertices,
                        "faces": mesh.faces,
                        "source_points_m": points,
                        "realized_centerline_m": realized,
                    }
                    prefix = f"case_{case_index}_body_{len(bodies)}"
                    cache.update({f"{prefix}_{name}": value for name, value in geometry.items()})
                    bodies.append({**summary, **geometry})
                    summaries.append(summary)
                print(
                    f"Swept case {case_index + 1}, support {part_index + 1}: {len(keys)} independent bodies.",
                    flush=True,
                )
            all_bodies.append(bodies)
            records.append(
                {
                    "name": case["name"],
                    "body_count": len(bodies),
                    "triangle_count": sum(body["triangle_count"] for body in bodies),
                    "bodies": summaries,
                }
            )
    np.savez_compressed(arrays_path, **cache)
    summaries = [body for case in records for body in case["bodies"]]
    vertex_displacements = [body["centerline_vertex_max_displacement_m"] for body in summaries]
    same_centerline_vertices = all(value is not None and value < 1e-12 for value in vertex_displacements)
    evidence = {
        "description": "Actual production conductor-sweep triangle boundaries; illustrative independent contours.",
        "api": "coilgen.manufacturing.create_swept_contour_body(validate=True)",
        "geometry_identity": identity,
        "mesh_arrays_sha256": sha256(arrays_path.read_bytes()).hexdigest(),
        "nominal_diameter_mm": 2 * RADIUS_M * 1e3,
        "cross_section_area_m2": SECTION_SIDES * RADIUS_M**2 * np.sin(2 * np.pi / SECTION_SIDES) / 2,
        "joined": False,
        "electrical_circuit_completed": False,
        "connecting_leads_and_feeds_designed": False,
        "fabrication_validated": False,
        "independent_body_geometry_validation_passed": True,
        "geometry_diagnostics_computed": True,
        "between_body_clearance_checked": False,
        "geometry_modified_by_core_projection_and_mounting": not same_centerline_vertices,
        "field_and_sweep_centerline_vertices_agree_within_m": 1e-12 if same_centerline_vertices else None,
        "field_recomputed_for_swept_geometry": False,
        "field_geometry": "same conditioned contour centerlines; no mounting offset",
        "renderer": "VTK Phong lighting with smooth vertex normals on the actual production triangle mesh",
        "body_check_counts": {
            key: sum(bool(body["diagnostics"].get(key, False)) for body in summaries)
            for key in (
                "mesh_solid_ok",
                "manufacturing_ok",
                "sweep_mesh_self_intersection",
                "sweep_mesh_self_intersection_truncated",
                "sweep_self_intersection_skipped",
                "sweep_capsule_self_intersection",
                "sweep_final_frame_rotation_violation",
                "sweep_curvature_envelope_violation",
                "sweep_profile_jacobian_violation",
            )
        },
        "cases": records,
    }
    return all_bodies, evidence


def _combine(bodies: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, offset = [], [], 0
    for body in bodies:
        vertices.append(body["vertices_m"])
        faces.append(body["faces"] + offset)
        offset += len(body["vertices_m"])
    return np.vstack(vertices), np.vstack(faces)


def _render_mesh(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    *,
    color: str,
    camera: tuple,
    scale_m: float,
    size: tuple[int, int],
) -> np.ndarray:
    """Light actual core triangles; smooth normals change no vertex positions."""
    connectivity = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    mesh = pv.PolyData(vertices_m, connectivity)
    plotter = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    try:
        plotter.background_color = NAVY
        plotter.add_mesh(
            mesh,
            color=color,
            smooth_shading=True,
            split_sharp_edges=False,
            ambient=0.24,
            diffuse=0.72,
            specular=0.75,
            specular_power=65,
            show_edges=False,
        )
        plotter.add_light(pv.Light(position=(-0.45, -0.55, 0.8), focal_point=(0, 0, 0), intensity=0.9))
        plotter.add_light(pv.Light(position=(0.5, 0.25, 0.3), focal_point=(0, 0, 0), intensity=0.55))
        plotter.add_light(pv.Light(position=(0.1, -0.2, -0.5), focal_point=(0, 0, 0), intensity=0.3))
        plotter.camera_position = camera
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = scale_m
        plotter.enable_anti_aliasing("ssaa")
        return plotter.screenshot(return_img=True)
    finally:
        plotter.close()


def _bend_detail(body: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly orient a real rounded bend without stretching its dimensions."""
    source = body["realized_centerline_m"][:-1]
    before = np.roll(source, 1, axis=0)
    after = np.roll(source, -1, axis=0)
    tangent = after - before
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    curvature = after - 2 * source + before
    curvature -= np.sum(curvature * tangent, axis=1, keepdims=True) * tangent
    # An extreme in the longest principal direction gives the U-shaped end
    # of the winding, rather than an arbitrary low-curvature side section.
    _, _, basis = np.linalg.svd(source - source.mean(axis=0), full_matrices=False)
    index = int(np.argmax((source - source.mean(axis=0)) @ basis[0]))
    x_axis = tangent[index]
    y_axis = curvature[index] / np.linalg.norm(curvature[index])
    z_axis = np.cross(x_axis, y_axis)
    transform = np.column_stack((x_axis, y_axis, z_axis))
    return (body["vertices_m"] - source[index]) @ transform, body["faces"]


def _image(fig: plt.Figure, bounds: tuple, pixels: np.ndarray) -> None:
    ax = fig.add_axes(bounds)
    ax.imshow(pixels)
    ax.set_axis_off()


def render_all(cases: list[dict], metrics: dict | list) -> None:
    """Build/cache actual swept bodies and render their overview and details."""
    bodies, evidence = compute_bodies(cases, metrics)
    evidence.update(connecting_leads_and_feeds_designed=False, between_body_clearance_checked=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "CoilGen2-swept-bodies-v2"})
    fig = plt.figure(figsize=(16, 12), facecolor=NAVY)
    fig.text(0.055, 0.955, "SMOOTH PATHS. REAL CONDUCTOR BODIES.", fontsize=11, color=TEAL, weight="bold")
    fig.text(0.055, 0.910, "Windings that flow", fontsize=32, color=PALE, weight="bold")
    fig.text(
        0.055,
        0.872,
        "Smoothed, field-checked contours  /  1.8 mm round profile  /  actual CoilGen2 sweep meshes",
        fontsize=13,
        color=MUTED,
    )
    locations = ((0.025, 0.352, 0.475, 0.510), (0.50, 0.327, 0.475, 0.535))
    cameras = (
        ((0.5, -0.85, 0.35), (0, 0, 0), (0, 0, 1)),
        ((0.55, -0.8, 0.65), (0, 0, 0), (0, 0, 1)),
    )
    for index, (body_list, accent) in enumerate(zip(bodies, (COPPER, TEAL))):
        vertices, faces = _combine(body_list)
        pixels = _render_mesh(
            vertices,
            faces,
            color=accent,
            camera=cameras[index],
            scale_m=0.195 if index == 0 else 0.235,
            size=(1800, 1600),
        )
        _image(fig, locations[index], pixels)
        fig.text(
            0.075 + index * 0.49,
            0.342,
            "Cylindrical Gy" if index == 0 else "Biplanar Gz",
            fontsize=19,
            color=accent,
            weight="bold",
        )
        fig.text(
            0.075 + index * 0.49, 0.313, f"{len(body_list)} independent conductor bodies", fontsize=12, color=MUTED
        )

    vertices, faces = _bend_detail(bodies[0][0])
    pixels = _render_mesh(
        vertices,
        faces,
        color=COPPER,
        camera=((0, -0.015, 0.16), (0, 0.010, 0), (0, 1, 0)),
        scale_m=0.022,
        size=(1400, 750),
    )
    _image(fig, (0.045, 0.052, 0.43, 0.225), pixels)
    fig.text(0.075, 0.276, "Rounded bend · actual mesh detail", color=PALE, fontsize=13, weight="bold")

    profile = fig.add_axes((0.515, 0.083, 0.15, 0.16))
    profile.set_facecolor(NAVY)
    theta = np.arange(SECTION_SIDES) * 2 * np.pi / SECTION_SIDES
    section_mm = RADIUS_M * 1e3 * np.column_stack((np.cos(theta), np.sin(theta)))
    profile.add_patch(Polygon(section_mm, facecolor=COPPER, edgecolor="#ffe0bb", lw=1.4))
    profile.annotate("", xy=(-0.9, -1.13), xytext=(0.9, -1.13), arrowprops={"arrowstyle": "|-|", "color": MUTED})
    profile.text(0, -1.40, "1.8 mm", color=PALE, ha="center", fontsize=11)
    profile.set(xlim=(-1.55, 1.55), ylim=(-1.65, 1.2), aspect="equal")
    profile.axis("off")
    fig.text(0.70, 0.240, "A smooth, finite conductor", color=PALE, fontsize=14, weight="bold")
    fig.text(
        0.70,
        0.145,
        "32-sided circular cross-section\nCentered on the validated contour\nSmooth normals, true mesh geometry",
        color=MUTED,
        fontsize=12,
        linespacing=1.9,
    )
    fig.text(
        0.055,
        0.025,
        "Illustrative independent bodies. Small contours are filtered with field validation; connecting leads and feeds are not designed. No manufacturing release.",
        color=MUTED,
        fontsize=10.5,
    )
    for extension in ("png", "svg"):
        path = OUTPUT / f"swept_bodies.{extension}"
        fig.savefig(path, dpi=280, facecolor=NAVY, metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    plt.close(fig)
    evidence["generator_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    evidence["figure_sha256"] = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (OUTPUT / "swept_bodies.png", OUTPUT / "swept_bodies.svg")
    }
    (OUTPUT / "swept_bodies.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print("Saved actual swept conductor-body geometry and gallery.", flush=True)


if __name__ == "__main__":
    from contour_processing import load_refined_cases

    saved_cases, saved_metrics = load_refined_cases()
    render_all(saved_cases, saved_metrics)
