"""CAD edits must preserve geometry, topology, units and project provenance."""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("gmsh")

from coilgen.application.project import ProjectDocument  # noqa: E402
from coilgen.application.surface_model import (  # noqa: E402
    SurfaceEditor,
    SurfaceModel,
    apply_surface,
    build_surface,
    primitive,
)
from coilgen.application.presets import preset_config  # noqa: E402


def sphere_cut():
    value = SurfaceModel.initial().to_dict()
    value["nodes"].extend(
        [
            primitive("sphere", "sphere1", center_m=(0.065, -0.035, 0.04)),
            dict(id="cut1", kind="subtract", name="Access recess", base="object1", tool="sphere1"),
        ]
    )
    return SurfaceModel(value)


def test_closed_cylinder_area_volume_and_outward_normals():
    mesh, scene = build_surface(SurfaceModel.initial(), final=True)
    t = mesh.vertices[mesh.faces]
    volume = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
    assert volume == pytest.approx(math.pi * 0.08**2 * 0.24, rel=0.006)
    assert scene["mesh_report"]["area_m2"] == pytest.approx(2 * math.pi * 0.08 * (0.08 + 0.24), rel=0.006)
    assert scene["mesh_report"]["boundary_loops"] == 0
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx([0.16, 0.16, 0.24], abs=1e-4)


def test_subtract_sphere_and_delete_caps_and_cavity():
    model = sphere_cut()
    _, scene = build_surface(model)
    value = model.to_dict()
    value["excluded_faces"] = [f["key"] for f in scene["faces"] if "Plane" in f["label"] or "Sphere" in f["label"]]
    mesh, result = build_surface(SurfaceModel(value), final=True)
    assert result["mesh_report"]["boundary_loops"] == 3
    assert result["mesh_report"]["components"] == 1
    assert np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]) == pytest.approx(0.08, abs=2e-9)
    assert np.linalg.norm(mesh.vertices - [0.065, -0.035, 0.04], axis=1).min() >= 0.06 - 1e-8


def test_plane_rotation_translation_and_surface_cut():
    value = SurfaceModel.initial().to_dict()
    plane = primitive("plane", "plane")
    plane.update(center_m=[0.1, -0.2, 0.3], rotation_rad=[math.pi / 2, 0, 0])
    value["nodes"] = [plane]
    mesh, _ = build_surface(SurfaceModel(value), final=True)
    assert mesh.vertices[:, 1] == pytest.approx(-0.2, abs=1e-12)
    assert mesh.vertices.mean(axis=0) == pytest.approx([0.1, -0.2, 0.3], abs=0.005)
    sphere = primitive("sphere", "tool", center_m=(0.1, -0.2, 0.3))
    value["nodes"] += [sphere, dict(id="hole", kind="subtract", name="Hole", base="plane", tool="tool")]
    _, result = build_surface(SurfaceModel(value), final=True)
    assert result["mesh_report"]["boundary_loops"] == 2


def test_face_references_are_reproducible_and_stale_deletions_block_mesh():
    value = sphere_cut().to_dict()
    _, a = build_surface(SurfaceModel(value))
    _, b = build_surface(SurfaceModel(value))
    assert [f["key"] for f in a["faces"]] == [f["key"] for f in b["faces"]]
    value["excluded_faces"] = [a["faces"][0]["key"]]
    value["nodes"][1]["center_m"][0] += 0.01
    _, changed = build_surface(SurfaceModel(value))
    assert changed["unresolved_faces"]
    with pytest.raises(ValueError, match="Face selections changed"):
        build_surface(SurfaceModel(value), final=True)


def test_project_reopen_owns_recipe_and_mesh_and_preserves_target(tmp_path):
    project = ProjectDocument.from_config(preset_config("cylinder"))
    model = sphere_cut()
    mesh, _ = build_surface(model, final=True)
    updated = apply_surface(project, model, mesh, expected_fingerprint=project.fingerprint)
    assert updated.revision == project.revision + 1
    assert updated.values["target_region_geometry"] == project.values["target_region_geometry"]
    assert updated.study == project.study
    path = updated.save(tmp_path / "surface.coilgen")
    reopened = ProjectDocument.load(path)
    assert SurfaceModel.from_project(reopened).to_dict() == model.to_dict()
    with pytest.raises(ValueError, match="project changed"):
        apply_surface(updated, model, mesh, expected_fingerprint=project.fingerprint)
    mesh.vertices[0, 0] += 0.001
    with pytest.raises(ValueError, match="fresh final mesh"):
        apply_surface(project, model, mesh, expected_fingerprint=project.fingerprint)


def test_transaction_failure_and_undo_invalidate_final_mesh():
    editor = SurfaceEditor()
    editor.preview()
    editor.generate()
    previous = editor.model.identity
    invalid = editor.model.to_dict()
    invalid["nodes"][0]["parameters"]["radius"] = float("nan")
    with pytest.raises(ValueError):
        editor.edit(invalid)
    assert editor.model.identity == previous and editor.mesh is not None
    value = editor.model.to_dict()
    value["nodes"][0]["center_m"] = [0.1, 0, 0]
    editor.edit(value)
    assert editor.mesh is None
    editor.travel()
    assert editor.model.identity == previous and editor.mesh is None
    editor.travel(True)
    assert editor.model.identity != previous


def test_kernel_isolated_between_threads_and_existing_session():
    import gmsh

    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(lambda _: build_surface(SurfaceModel.initial(), final=True)[1]["mesh_report"], range(2))
        )
    assert results[0] == results[1]
    gmsh.initialize([], interruptible=False)
    try:
        gmsh.model.add("external-model")
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        before = gmsh.option.getNumber("Mesh.MeshSizeMax")
        build_surface(SurfaceModel.initial())
        assert gmsh.model.getCurrent() == "external-model"
        assert len(gmsh.model.getEntities(3)) == 1
        assert gmsh.option.getNumber("Mesh.MeshSizeMax") == before
    finally:
        gmsh.finalize()


def test_intersecting_roots_block_meshing_but_separate_supports_are_valid():
    value = SurfaceModel.initial().to_dict()
    value["edge_length_m"] = 0.03
    value["nodes"].append(primitive("sphere", "sphere", center_m=(0.07, 0, 0)))
    with pytest.raises(ValueError, match="intersects itself"):
        build_surface(SurfaceModel(value), final=True)
    value["nodes"][-1]["center_m"] = [0.3, 0, 0]
    model = SurfaceModel(value)
    mesh, scene = build_surface(model, final=True)
    assert scene["mesh_report"]["components"] == 2
    assert scene["mesh_report"]["self_intersections"] == 0
    project = ProjectDocument.from_config(preset_config("cylinder"))
    updated = apply_surface(project, model, mesh, expected_fingerprint=project.fingerprint)
    assert updated.config.routing_circuit_scope == "per_part"


@pytest.mark.parametrize("shape,boundaries", [("trimmed", 3), ("sphere", 0), ("box", 0)])
def test_cad_surface_reaches_numerical_preparation(shape, boundaries, tmp_path):
    from coilgen import CoilGenConfig
    from coilgen.application import DesignService
    from coilgen.pipeline_stages import PipelineCheckpoint

    value = sphere_cut().to_dict() if shape == "trimmed" else SurfaceModel.initial().to_dict()
    if shape == "trimmed":
        _, scene = build_surface(SurfaceModel(value))
        value["excluded_faces"] = [f["key"] for f in scene["faces"] if "Plane" in f["label"] or "Sphere" in f["label"]]
    else:
        value["nodes"] = [primitive(shape, "object1")]
    value["edge_length_m"] = 0.035
    model = SurfaceModel(value)
    mesh, _ = build_surface(model, final=True)
    config = CoilGenConfig(
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_geometry={"shape": "sphere", "radius_m": 0.015, "sampling": "volume"},
        target_region_resolution=4,
        sf_opt_method="tikhonov",
        research_physics=None,
        iteration_num_mesh_refinement=0,
        plot_flag=False,
        progress_flag=False,
    )
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "prepared", "fabricate": False})
    project = apply_surface(project, model, mesh, expected_fingerprint=project.fingerprint)
    result = DesignService.run(project, tmp_path / "run", threads=1)
    assert result["pipeline_stage"] == "prepared"
    state = PipelineCheckpoint.load(tmp_path / "run" / result["checkpoint"]).state
    assert len(state.parts) == 1 and len(state.parts[0].mesh.boundaries) == boundaries
    part = state.parts[0]
    assert np.isfinite(part.sensitivity_matrix).all() and np.any(part.sensitivity_matrix)
    assert part.sensitivity_matrix.shape == (3, len(state.target_field.coords), part.mesh.n_vertices)
    for basis in part.basis_elements:
        assert np.all(basis.area > 0) and np.isfinite(basis.current).all()


@pytest.mark.parametrize("change", ["cycle", "sheet_tool", "tiny_mesh", "all_faces"])
def test_invalid_construction_cannot_produce_solver_mesh(change):
    value = sphere_cut().to_dict()
    if change == "cycle":
        value["nodes"][-1]["base"] = "cut1"
    elif change == "sheet_tool":
        value["nodes"][1] = primitive("plane", "sphere1")
    elif change == "tiny_mesh":
        value["edge_length_m"] = 1e-5
    else:
        _, scene = build_surface(SurfaceModel(value))
        value["excluded_faces"] = [f["key"] for f in scene["faces"]]
    with pytest.raises(ValueError):
        build_surface(SurfaceModel(value), final=True)
