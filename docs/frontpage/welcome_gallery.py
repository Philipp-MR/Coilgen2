"""Render welcome-page geometry from the actual CoilGen2 CAD/mesh APIs.

Run from the repository root with the gui and plot extras installed:
    python docs/frontpage/welcome_gallery.py

These are geometry illustrations, not optimized or fabrication-qualified coils.
Existing screenshots and numerical result figures are never modified.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coilgen import CoilGenConfig  # noqa: E402
from coilgen.application.surface_model import SurfaceModel, build_surface, primitive  # noqa: E402
from coilgen.application.support_parameters import SUPPORT_EDITORS  # noqa: E402
from coilgen.geometry import load_mesh_from_config  # noqa: E402
from coilgen.models import Mesh  # noqa: E402

OUTPUT = ROOT / "docs" / "images"
PAPER, INK, MUTED = "#f4f7fa", "#15324a", "#526a7e"
BLUE, TEAL, COPPER = "#4b93bd", "#39a38a", "#d78a4c"


def render(layers, *, fixed=False):
    """Render real triangles with orthographic projection and explicit lighting."""
    plotter = pv.Plotter(off_screen=True, window_size=(1100, 1050), lighting="none")
    try:
        plotter.set_background(PAPER)
        for mesh, color, opacity, edges in layers:
            faces = np.column_stack((np.full(mesh.n_faces, 3), mesh.faces)).ravel()
            poly = pv.PolyData(mesh.vertices, faces).clean()
            plotter.add_mesh(
                poly,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                split_sharp_edges=True,
                ambient=0.40,
                diffuse=0.62,
                specular=0.25,
                specular_power=32,
                show_edges=edges,
                edge_color="#21475c",
                line_width=0.9,
                backface_params={"color": "#bad7df", "ambient": 0.5, "diffuse": 0.5},
            )
        vertices = np.vstack([layer[0].vertices for layer in layers])
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        scale = np.linalg.norm(np.ptp(vertices, axis=0))
        if fixed:
            center, scale = np.zeros(3), 0.32
        camera = center + np.array([1.7, -2.6, 1.55]) * scale
        plotter.camera_position = [camera, center, (0, 0, 1)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 0.168 if fixed else scale * 0.48
        for offset, intensity in (([-2, -3, 4], 0.85), ([3, 1, 2], 0.65), ([-1, 2, -2], 0.25)):
            plotter.add_light(
                pv.Light(position=center + np.array(offset) * scale, focal_point=center, intensity=intensity)
            )
        plotter.enable_anti_aliasing("ssaa")
        return plotter.screenshot(return_img=True)
    finally:
        plotter.close()


def figure(title, subtitle, rows, columns):
    fig = plt.figure(figsize=(16, 2 + 4.3 * rows), facecolor=PAPER)
    fig.text(0.035, 0.965, title, fontsize=30, weight="bold", color=INK, va="top")
    fig.text(0.035, 0.885, subtitle, fontsize=15, color=MUTED, va="top")
    grid = fig.add_gridspec(rows, columns, left=0.025, right=0.975, bottom=0.035, top=0.80, hspace=0.22, wspace=0.06)
    axes = [fig.add_subplot(grid[i, j]) for i in range(rows) for j in range(columns)]
    for ax in axes:
        ax.set_axis_off()
    return fig, axes


def save(fig, name):
    path = OUTPUT / name
    fig.savefig(path, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(path.name, flush=True)
    return {"file": name, "sha256": sha256(path.read_bytes()).hexdigest()}


def cad_workflow():
    value = SurfaceModel.initial().to_dict()
    cutter = primitive("cylinder", "cutter", center_m=(0, -0.09, 0.025))
    cutter.update(name="Window cutter", parameters={"radius": 0.045, "height": 0.18}, rotation_rad=[np.pi / 2, 0, 0])
    value["nodes"].append(cutter)
    _, source = build_surface(SurfaceModel(value))
    layers = []
    for key, color in (("object1", BLUE), ("cutter", COPPER)):
        face_ids = {i for i, face in enumerate(source["faces"]) if face["body"] == key}
        triangles = [tri for tri, index in zip(source["triangles"], source["face_indices"]) if index in face_ids]
        layers.append((Mesh(np.array(source["vertices"]), np.array(triangles)), color, 1, False))
    value["nodes"].append(dict(id="cut", name="Windowed cylinder", kind="subtract", base="object1", tool="cutter"))
    _, cut = build_surface(SurfaceModel(value))
    wall = max((f for f in cut["faces"] if "Cylinder" in f["label"]), key=lambda f: f["area_m2"])
    value["excluded_faces"] = [f["key"] for f in cut["faces"] if f["key"] != wall["key"]]
    value["edge_length_m"] = 0.012
    model = SurfaceModel(value)
    mesh, final = build_surface(model, final=True)
    assert final["mesh_report"]["boundary_loops"] == 3
    assert final["mesh_report"]["components"] == 1
    fig, axes = figure(
        "Build the surface. Then choose the mesh.",
        "Editable CAD primitives → Boolean subtraction and face removal → Solver surface",
        1,
        3,
    )
    panels = [
        ("01  Position the primitives", render(layers, fixed=True)),
        ("02  Keep the required faces", render([(mesh, TEAL, 1, False)], fixed=True)),
        ("03  Mesh the final surface", render([(mesh, BLUE, 1, True)], fixed=True)),
    ]
    for ax, (title, pixels) in zip(axes, panels):
        ax.imshow(pixels)
        ax.set_title(title, fontsize=18, weight="bold", color=INK, pad=8)
    receipt = save(fig, "welcome-cad-workflow.png")
    receipt.update(model=model.to_dict(), report=final["mesh_report"], mesher=mesh.metadata["mesher"])
    return receipt


def support_gallery():
    names = [
        "Cylinder",
        "Biplanar",
        "Conical frustum",
        "Annulus",
        "Elliptical cylinder",
        "Superellipse",
        "Spherical cap",
        "Toroidal segment",
    ]
    editors = {e.label: e for e in SUPPORT_EDITORS}
    fig, axes = figure(
        "Start with the support your coil needs.",
        "Eight examples from the generated surface library · each panel uses its own viewing scale",
        2,
        4,
    )
    records = []
    for i, (name, ax) in enumerate(zip(names, axes)):
        editor = editors[name]
        config = CoilGenConfig(coil_mesh_file=editor.source, surface_is_cylinder_flag=editor.cylindrical)
        divisions = {
            f.key: 128 if "angular" in f.key or "major" in f.key else 64 if "minor" in f.key else 32
            for f in editor.fields
            if f.integer
        }
        config = replace(
            config, **{editor.parameter_field: editor.apply_edits(getattr(config, editor.parameter_field), divisions)}
        )
        mesh = load_mesh_from_config(config, ROOT)
        assert np.isfinite(mesh.vertices).all() and mesh.n_faces > 0
        ax.imshow(render([(mesh, (BLUE, TEAL, COPPER)[i % 3], 1, False)]))
        ax.set_title(name, fontsize=19, color=INK, weight="bold", pad=4)
        records.append(
            {
                "name": name,
                "source": editor.source,
                "parameters": list(getattr(config, editor.parameter_field)),
                "vertices": mesh.n_vertices,
                "triangles": mesh.n_faces,
            }
        )
    receipt = save(fig, "welcome-support-library.png")
    receipt["supports"] = records
    return receipt


def workflow():
    stages = [
        ("01", "SURFACE", "Geometry & boundaries"),
        ("02", "TARGET", "Field & region"),
        ("03", "CURRENT", "Physical optimization"),
        ("04", "WINDING", "Contours & connections"),
        ("05", "CONDUCTOR", "Profile & mounting"),
        ("06", "REVIEW", "Fields & engineering"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="230" viewBox="0 0 1800 230" role="img" aria-label="CoilGen2 workflow from support surface to field and engineering review">',
        '<rect width="1800" height="230" rx="18" fill="#091b2b"/>',
    ]
    for i, (number, title, detail) in enumerate(stages):
        x = 30 + i * 295
        color = "#47dec9" if i >= 3 else "#ffad72"
        parts.append(
            f'<text x="{x}" y="58" fill="{color}" font-family="DejaVu Sans,Arial,sans-serif" font-size="24">{number}</text>'
        )
        parts.append(
            f'<text x="{x}" y="108" fill="#f4f8fc" font-family="DejaVu Sans,Arial,sans-serif" font-size="27" font-weight="700">{title}</text>'
        )
        parts.append(
            f'<text x="{x}" y="150" fill="#a8c0d0" font-family="DejaVu Sans,Arial,sans-serif" font-size="20">{detail.replace("&", "&amp;")}</text>'
        )
        if i < 5:
            parts.append(f'<path d="M{x + 245} 94h28m-9 -9 9 9-9 9" fill="none" stroke="#658498" stroke-width="2"/>')
    parts.append("</svg>")
    (OUTPUT / "welcome-workflow.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": PAPER})
    receipts = [cad_workflow(), support_gallery()]
    workflow()
    evidence = {
        "scope": "Geometry illustrations from production APIs; no field optimization or fabrication acceptance",
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "figures": receipts,
    }
    (OUTPUT / "welcome-geometry-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
