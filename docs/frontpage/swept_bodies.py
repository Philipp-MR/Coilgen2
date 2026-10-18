"""Render actual conductor-boundary meshes around the saved closed contours.

The 1.8 mm diameter profile and surface mounting are illustration choices.
Each contour remains independent: there are no fabricated joins, feeds or
return paths, and these bodies are not released for manufacture. Field plots
continue to use the original current-sheet and thin-wire geometries.
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
from matplotlib import colors  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
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
SECTION_SIDES = 16


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
        "mounting_mode": "surface_outer",
        "bend_mode": "miter",
        "surface_realization": "analytic",
        "geometry_acceleration_backend": "numpy",
    }


def compute_bodies(cases: list[dict], metrics: dict | list) -> tuple[list[list[dict]], dict]:
    """Use the production sweep API, retaining all resulting diagnostics."""
    info = metrics["cases"] if isinstance(metrics, dict) else metrics
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
                conductor_mounting_mode="surface_outer",
                sweep_surface_realization="analytic",
                sweep_bend_mode="miter",
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
                        validate=False,
                    )
                    mesh = body.mesh
                    if not mesh.n_faces or not np.all(np.isfinite(mesh.vertices)):
                        raise RuntimeError("The core sweep returned an empty or nonfinite conductor body.")
                    if mesh.faces.min() < 0 or mesh.faces.max() >= mesh.n_vertices:
                        raise RuntimeError("The core sweep returned invalid triangle indices.")
                    summary = {
                        "part_index_zero_based": part_index,
                        "contour_index_zero_based": contour_index,
                        "vertex_count": mesh.n_vertices,
                        "triangle_count": mesh.n_faces,
                        "diagnostics": json_value(body.diagnostics),
                    }
                    geometry = {
                        "vertices_m": mesh.vertices,
                        "faces": mesh.faces,
                        "source_points_m": points,
                        "realized_centerline_m": body.fabrication_path.wire_points_m,
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
    evidence = {
        "description": "Actual production conductor-sweep triangle boundaries; illustrative independent contours.",
        "api": "coilgen.manufacturing.create_swept_contour_body(validate=False)",
        "geometry_identity": identity,
        "mesh_arrays_sha256": sha256(arrays_path.read_bytes()).hexdigest(),
        "nominal_diameter_mm": 2 * RADIUS_M * 1e3,
        "cross_section_area_m2": SECTION_SIDES * RADIUS_M**2 * np.sin(2 * np.pi / SECTION_SIDES) / 2,
        "joined": False,
        "electrical_circuit_completed": False,
        "connecting_leads_and_feeds_designed": False,
        "fabrication_validated": False,
        "geometry_diagnostics_computed": True,
        "between_body_clearance_checked": False,
        "geometry_modified_by_core_projection_and_mounting": True,
        "field_recomputed_for_swept_geometry": False,
        "cases": records,
    }
    return all_bodies, evidence


def _metal_triangles(ax, vertices_m, faces, color, *, focus=None) -> None:
    triangles = vertices_m[faces] * 1e3
    if focus is not None:
        center, half_width = focus
        selected = np.all(np.abs(triangles.mean(axis=1) - center) <= half_width, axis=1)
        triangles = triangles[selected]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    light = np.array([-0.4, -0.55, 0.73])
    light /= np.linalg.norm(light)
    intensity = np.abs(normal @ light)
    base = np.array(colors.to_rgb(color))
    rgb = base[None, :] * (0.50 + 0.50 * intensity[:, None])
    rgb += (1 - rgb) * (0.40 * intensity[:, None] ** 10)
    collection = Poly3DCollection(triangles, facecolors=rgb, linewidth=0, edgecolors="none", zsort="average")
    collection.set_rasterized(True)
    ax.add_collection3d(collection)


def _scene(ax, limits, *, elev=24, azim=-58) -> None:
    ax.set_facecolor(NAVY)
    for index, dimension in enumerate("xyz"):
        getattr(ax, f"set_{dimension}lim")(*limits[index])
    ax.set_box_aspect([high - low for low, high in limits])
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def _combine(bodies: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, offset = [], [], 0
    for body in bodies:
        vertices.append(body["vertices_m"])
        faces.append(body["faces"] + offset)
        offset += len(body["vertices_m"])
    return np.vstack(vertices), np.vstack(faces)


def render_all(cases: list[dict], metrics: dict | list) -> None:
    """Build/cache actual swept bodies and render their overview and details."""
    bodies, evidence = compute_bodies(cases, metrics)
    evidence.update(connecting_leads_and_feeds_designed=False, between_body_clearance_checked=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "CoilGen2-swept-bodies-v1"})
    fig = plt.figure(figsize=(16, 11.5), facecolor=NAVY)
    fig.text(0.055, 0.948, "WINDINGS, WITH VOLUME", fontsize=11, color=TEAL, weight="bold")
    fig.text(0.055, 0.897, "From contour lines to conductor bodies", fontsize=29, color=PALE, weight="bold")
    fig.text(
        0.055,
        0.858,
        "Actual CoilGen2 sweep meshes  /  1.8 mm nominal diameter  /  16-sided round profile",
        fontsize=13,
        color=MUTED,
    )
    locations = ((0.01, 0.32, 0.53, 0.52), (0.50, 0.32, 0.49, 0.52))
    for index, (body_list, bounds, accent) in enumerate(
        zip(bodies, (((-135, 135), (-135, 135), (-163, 163)), ((-165, 165), (-165, 165), (-115, 115))), (COPPER, TEAL))
    ):
        ax = fig.add_axes(locations[index], projection="3d")
        _scene(ax, bounds, elev=24 if index == 0 else 30, azim=-52 if index == 0 else -58)
        vertices, faces = _combine(body_list)
        _metal_triangles(ax, vertices, faces, accent)
        if index == 0:
            theta = np.linspace(0, 2 * np.pi, 200)
            for z in (-150, 150):
                ax.plot(
                    120 * np.cos(theta), 120 * np.sin(theta), np.full_like(theta, z), color="#66839b", alpha=0.5, lw=0.7
                )
        fig.text(
            0.075 + index * 0.49,
            0.345,
            "Cylindrical Gy" if index == 0 else "Biplanar Gz",
            fontsize=19,
            color=accent,
            weight="bold",
        )
        fig.text(
            0.075 + index * 0.49, 0.315, f"{len(body_list)} independent conductor bodies", fontsize=12, color=MUTED
        )

    detail = fig.add_axes((0.025, 0.045, 0.45, 0.25), projection="3d")
    first = bodies[0][0]
    source = first["realized_centerline_m"][:-1]
    corner = int(np.argmax(source[:, 0]))
    center = source[corner]
    incoming = center - source[(corner - 1) % len(source)]
    outgoing = source[(corner + 1) % len(source)] - center
    incoming /= np.linalg.norm(incoming)
    normal = np.cross(incoming, outgoing)
    normal /= np.linalg.norm(normal)
    local_basis = np.column_stack((incoming, np.cross(normal, incoming), normal))
    # A rigid camera-coordinate transform shows the actual bend face-on.
    # Geometry and conductor dimensions are never stretched for the inset.
    local_vertices = (first["vertices_m"] - center) @ local_basis
    _scene(detail, ((-9, 3), (-3, 9), (-2, 2)), elev=62, azim=-42)
    _metal_triangles(detail, local_vertices, first["faces"], COPPER)
    fig.text(0.075, 0.285, "Conductor mesh detail", color=PALE, fontsize=13, weight="bold")

    profile = fig.add_axes((0.51, 0.075, 0.16, 0.18))
    profile.set_facecolor(NAVY)
    theta = np.arange(SECTION_SIDES) * 2 * np.pi / SECTION_SIDES
    section_mm = RADIUS_M * 1e3 * np.column_stack((np.cos(theta), np.sin(theta)))
    profile.add_patch(Polygon(section_mm, facecolor=COPPER, edgecolor="#ffe0bb", lw=1.4))
    profile.annotate("", xy=(-0.9, -1.13), xytext=(0.9, -1.13), arrowprops={"arrowstyle": "|-|", "color": MUTED})
    profile.text(0, -1.40, "1.8 mm", color=PALE, ha="center", fontsize=11)
    profile.set(xlim=(-1.55, 1.55), ylim=(-1.65, 1.2), aspect="equal")
    profile.axis("off")
    fig.text(0.71, 0.24, "A finite conductor profile", color=PALE, fontsize=14, weight="bold")
    fig.text(
        0.71,
        0.145,
        "Production sweep backend\nPhysical profile dimensions\nExplicit support-normal mounting",
        color=MUTED,
        fontsize=12,
        linespacing=1.9,
    )
    fig.text(
        0.055,
        0.025,
        "Illustrative profile and mounting. Contours remain independent; connecting leads and feeds are not designed. No manufacturing release.",
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
    saved_metrics = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    saved_cases = []
    for case_index in range(len(saved_metrics["cases"])):
        with np.load(OUTPUT / f"case_{case_index}.npz", allow_pickle=False) as archive:
            saved_cases.append({key: archive[key] for key in archive.files})
    render_all(saved_cases, saved_metrics)
