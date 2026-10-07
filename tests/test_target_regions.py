"""SI geometry, world-target, enclosure and legacy-contract regressions."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application import DesignService, ProjectDocument
from coilgen.config_validation import validate_config
from coilgen.geometry import load_optional_target_mesh
from coilgen.gradient_slice_maps import (
    GradientSliceMapSpec,
    _slice_map_source_signature,
    evaluate_gradient_slice_maps,
    target_region_slice_map_limitation,
)
from coilgen.models import CoilPart, Mesh, TargetField
from coilgen.object_registry import register_in_memory_target_field
from coilgen.pipeline_stages import PipelineStage, config_field_stage, stage_fingerprints
from coilgen.research.validation import validate_on_independent_grid
from coilgen.target import define_target_field, evaluate_field_expression
from coilgen.target_regions import (
    TargetRegion,
    sample_target_region,
    sample_target_region_from_config,
    target_region_summary,
    validate_enclosing_mesh,
)


REGIONS = [
    {"shape": "sphere", "radius_m": 0.02},
    {"shape": "ellipsoid", "semi_axes_m": [0.015, 0.025, 0.035]},
    {"shape": "box", "size_m": [0.03, 0.05, 0.07]},
    {"shape": "cylinder", "radius_m": 0.02, "length_m": 0.06},
]


def support():
    return Mesh(
        np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2], [0, 0, 2]], float) * 0.05,
        np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
    )


def tetra():
    return Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * 0.04,
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )


def concave_enclosure():
    polygon = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float) * 0.02
    vertices = np.vstack([np.column_stack((polygon, np.repeat(z, 6))) for z in (-0.02, 0.02)])
    faces = []
    for i in range(1, 5):
        faces.extend([[0, i + 1, i], [6, i + 6, i + 7]])
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    return Mesh(vertices, np.array(faces))


def coordinates(points):
    return {tuple(row) for row in points}


@pytest.mark.parametrize("spec", REGIONS)
@pytest.mark.parametrize("sampling", ["volume", "surface"])
def test_geometry_placement_boundary_and_distinct_sampling(spec, sampling):
    region = TargetRegion.from_dict({**spec, "sampling": sampling})
    points = sample_target_region(region, 7)
    validation = sample_target_region(region, 10, independent=True)
    assert not coordinates(points) & coordinates(validation)
    assert np.all(np.isfinite(points))
    scaled = points / region.half_extents_m
    if region.shape in {"sphere", "ellipsoid"}:
        distance = np.linalg.norm(scaled, axis=1)
    elif region.shape == "box":
        distance = np.max(np.abs(scaled), axis=1)
    else:
        distance = np.maximum(np.linalg.norm(scaled[:, :2], axis=1), np.abs(scaled[:, 2]))
    if sampling == "surface":
        assert distance == pytest.approx(np.ones(len(points)), abs=1e-14)
    else:
        assert np.all(distance < 1)
    moved = TargetRegion.from_dict(
        {
            **region.to_dict(),
            "center_m": [0.1, -0.04, 0.07],
            "rotation_axis": [0, 1, 0],
            "rotation_angle_rad": np.pi / 2,
        }
    )
    assert sample_target_region(moved, 7) == pytest.approx(points @ moved.rotation_matrix.T + moved.center_m)
    assert TargetRegion.from_dict(moved.to_dict()) == moved


@pytest.mark.parametrize(
    "spec",
    [
        {"shape": "sphere", "radius_m": 0},
        {"shape": "ellipsoid", "semi_axes_m": [0.1, -0.1, 0.2]},
        {"shape": "box", "size_m": [0.1, np.nan, 0.1]},
        {"shape": "cylinder", "radius_m": 0.1},
        {"shape": "sphere", "radius_m": True},
        {"shape": "sphere", "radius_m": 0.02, "center_m": [True, 0, 0]},
        {"shape": "sphere", "radius_m": 0.02, "rotation_axis": [0, 0, 0], "rotation_angle_rad": 1},
        {"shape": "box", "size_m": [1, 1, 1], "radius_m": 1},
        {"shape": []},
        {"shape": "enclosing_mesh", "center_m": [0.1, 0, 0]},
        {"shape": "sphere", "radius_m": 0.02, "sampling": "bbox"},
    ],
)
def test_invalid_specs_are_actionable(spec):
    with pytest.raises(ValueError):
        TargetRegion.from_dict(spec)


@pytest.mark.parametrize("resolution", [0, 1, True, 5.5])
def test_invalid_resolution_rejected(resolution):
    with pytest.raises(ValueError, match="integer >=2"):
        sample_target_region(REGIONS[0], resolution)


def test_axis_normalization_stays_stable_for_extreme_finite_scales():
    normal = TargetRegion.from_dict({**REGIONS[2], "rotation_axis": [1, 0, 0], "rotation_angle_rad": 0.7})
    for scale in (1e-300, 1e300):
        scaled = TargetRegion.from_dict({**normal.to_dict(), "rotation_axis": [scale, 0, 0]})
        assert scaled.rotation_matrix == pytest.approx(normal.rotation_matrix)
        assert np.linalg.det(scaled.rotation_matrix) == pytest.approx(1)


def test_legacy_sphere_and_mesh_points_are_bitwise_unchanged():
    config = CoilGenConfig(target_region_resolution=9, target_region_radius=0.02, target_gradient_normalization="none")
    axis = np.linspace(-0.04, 0.04, 9)
    grid = np.array(np.meshgrid(axis, axis, axis, indexing="xy")).reshape(3, -1).T
    expected = grid[np.linalg.norm(grid, axis=1) <= 0.02 + 32 * np.finfo(float).eps]
    actual = define_target_field([support()], None, None, config)
    np.testing.assert_array_equal(actual.coords, expected)
    mesh = tetra()
    axes = [np.linspace(mesh.vertices[:, i].min(), mesh.vertices[:, i].max(), 9) for i in range(3)]
    expected_mesh = np.vstack((mesh.vertices, np.array(np.meshgrid(*axes, indexing="xy")).reshape(3, -1).T))
    expected_mesh = np.array(list(dict.fromkeys(map(tuple, expected_mesh))))
    actual_mesh = define_target_field([support()], mesh, None, config)
    np.testing.assert_array_equal(actual_mesh.coords, expected_mesh)
    assert any(np.sum(actual_mesh.coords, axis=1) > 0.04)  # documented legacy bbox, not containment
    vertices_only = define_target_field([support()], mesh, None, replace(config, use_only_target_mesh_verts=True))
    np.testing.assert_array_equal(vertices_only.coords, mesh.vertices)


def test_world_expression_and_harmonic_phase_do_not_follow_region_rotation():
    config = CoilGenConfig(
        target_region_geometry={
            **REGIONS[2],
            "center_m": [0.03, -0.02, 0.01],
            "rotation_axis": [0, 0, 1],
            "rotation_angle_rad": np.pi / 2,
        },
        target_region_resolution=6,
        set_roi_into_mesh_center=True,
        field_shape_function="1e-6*(x+2*y)/0.04 + 2e-6*solid_harmonic(2,-1,x,y,z,0.04)",
        target_gradient_normalization="none",
    )
    result = define_target_field([support()], None, None, config)
    x, y, z = result.coords.T
    # Real SH2,m=-1 carries Condon-Shortley minus sign: -sqrt(15/4pi)*yz/R².
    expected = 1e-6 * (x + 2 * y) / 0.04 - 2e-6 * np.sqrt(15 / (4 * np.pi)) * y * z / 0.04**2
    assert result.b[:, 2] == pytest.approx(expected, abs=1e-19)
    assert result.coords.mean(axis=0) == pytest.approx([0.03, -0.02, 0.01])


def test_closed_mesh_uses_actual_concave_containment_and_excludes_boundary():
    mesh = concave_enclosure()
    points = sample_target_region({"shape": "enclosing_mesh"}, 8, mesh=mesh)
    assert len(points) == 8**3 * 3 // 4
    assert not np.any((points[:, 0] > 0.02) & (points[:, 1] > 0.02))
    assert np.all(points[:, 0] > 0) and np.all(points[:, 1] > 0)
    assert np.all(np.abs(points[:, 2]) < 0.02)
    simple = sample_target_region({"shape": "enclosing_mesh"}, 5, mesh=tetra())
    assert np.all(np.sum(simple, axis=1) < 0.04 - 1e-15)
    reverse = sample_target_region({"shape": "enclosing_mesh"}, 5, mesh=replace(tetra(), faces=tetra().faces[:, ::-1]))
    np.testing.assert_array_equal(simple, reverse)
    welded = Mesh(np.vstack((tetra().vertices, tetra().vertices)), tetra().faces + 4)
    assert validate_enclosing_mesh(welded).n_vertices == 4


def test_open_surface_supported_but_volume_requires_watertight_shell():
    open_mesh = replace(tetra(), faces=tetra().faces[:1])
    surface_spec = {"shape": "enclosing_mesh", "sampling": "surface"}
    points = sample_target_region(surface_spec, 7, mesh=open_mesh)
    assert len(points) == 6 * 7**2
    assert np.all(points[:, 2] == 0)
    assert np.all(points[:, :2] >= 0)
    assert np.all(np.sum(points[:, :2], axis=1) < 0.04)
    assert not coordinates(points) & coordinates(
        sample_target_region(surface_spec, 9, independent=True, mesh=open_mesh)
    )
    with pytest.raises(ValueError, match="watertight"):
        sample_target_region({"shape": "enclosing_mesh"}, 7, mesh=open_mesh)
    flipped = tetra()
    flipped.faces[0] = flipped.faces[0, ::-1]
    with pytest.raises(ValueError, match="orientation"):
        validate_enclosing_mesh(flipped)
    disconnected = Mesh(
        np.vstack((tetra().vertices, tetra().vertices + 0.1)), np.vstack((tetra().faces, tetra().faces + 4))
    )
    with pytest.raises(ValueError, match="connected"):
        validate_enclosing_mesh(disconnected)


def test_global_surface_budget_and_area_coverage_are_not_face_count_weighted():
    # Equal triangle counts, area ratio 4:1, spatially separated for auditing.
    mesh = Mesh(
        np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [4, 0, 0], [5, 0, 0], [4, 1, 0]], float),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    samples = sample_target_region({"shape": "enclosing_mesh", "sampling": "surface"}, 10, mesh=mesh)
    assert len(samples) == 600
    assert np.sum(samples[:, 0] < 3) == 480


def test_empty_thin_enclosure_and_degenerate_surface_fail_actionably():
    narrow = replace(tetra(), vertices=np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1.00001], [1, 1.00001, 1]], float))
    with pytest.raises(ValueError, match="no sample points"):
        sample_target_region({"shape": "enclosing_mesh"}, 2, mesh=narrow)
    degenerate = Mesh(np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]]), np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="degenerate"):
        sample_target_region({"shape": "enclosing_mesh", "sampling": "surface"}, 5, mesh=degenerate)


def test_owned_region_mesh_is_one_json_portable_and_independent_of_support_scale(tmp_path):
    project = ProjectDocument.new(
        config=CoilGenConfig(coil_mesh_file="create cylinder mesh", imported_mesh_length_scale_to_m=0.001)
    )
    project = project.with_results((), session={"selection": "target"}).update_study({"validation_resolution": 8})
    owned = project.with_mesh(
        tetra(), changes={"target_region_geometry": {"shape": "enclosing_mesh"}, "target_region_resolution": 7}
    )
    assert owned.id == project.id and owned.revision == project.revision + 1
    assert owned.study == project.study and owned.session == project.session and owned.saved_runs == project.saved_runs
    assert owned.values["target_mesh_file"].startswith("asset:")
    path = owned.save(tmp_path / "single.coilgen.json")
    relocated = ProjectDocument.from_dict(json.loads(path.read_text()), project_root=tmp_path / "other_location")
    points = sample_target_region_from_config(relocated.config)
    assert len(points) > 0 and np.max(points) > 0.01
    original = load_optional_target_mesh(relocated.config.target_mesh_file, relocated.config, tmp_path)
    assert np.max(original.vertices) == pytest.approx(0.04)
    assert validate_config(relocated.config).ok
    with pytest.raises(ValueError, match="Owned mesh field"):
        project.with_mesh(tetra(), field="research_physics")
    with pytest.raises(ValueError, match="Unknown CoilGenConfig"):
        project.with_mesh(tetra(), changes={"unknown": True})


def test_completed_regional_sheet_and_input_mesh_round_trip_in_one_json(tmp_path):
    from coilgen.application.bundles import RunArchive

    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.2, 0.12, 8, 4, 1, 0, 0, 0),
        target_region_geometry={"shape": "enclosing_mesh"},
        target_region_resolution=5,
        field_shape_function="y",
        target_gradient_strength=0.02,
        target_gradient_normalization="linear_fit",
        target_gradient_axis="y",
        target_error_tolerance_percent=40,
        research_physics={"objective": "power"},
        gauss_order=1,
        progress_flag=False,
    )
    project = ProjectDocument.new(config=config).with_mesh(tetra())
    directory = tmp_path / "regional_sheet"
    DesignService.run(project, directory, pipeline_stage="sheet", fabricate=False, validation_resolution=6)
    record = DesignService.load_run(directory)
    archive = RunArchive.capture(record)
    portable = ProjectDocument.from_dict(
        project.with_results((archive,)).to_dict(), project_root=tmp_path / "relocated"
    )
    assert portable.config.target_region_geometry == config.target_region_geometry
    assert portable.model_fingerprint == project.model_fingerprint
    restored = portable.saved_runs[0].materialize(tmp_path / "restored")
    assert restored.metadata["summary"]["construction_released"] is False
    for name, values in record.arrays.items():
        np.testing.assert_array_equal(restored.arrays[name], values)
    assert len(restored.arrays["validation_points_m"]) > 0


def test_old_json_migration_preserves_prior_fingerprints_and_stays_narrow():
    project = ProjectDocument.new()
    value = project.to_dict()
    value["config"].pop("target_region_geometry")

    def encoded(data):
        return sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    prior_fingerprint = encoded({"config": value["config"], "assets": {}, "study": value["study"]})
    prior_model = encoded(
        {
            "config": {
                key: item
                for key, item in value["config"].items()
                if key != "geometry_source_path" and config_field_stage(key) is not None
            },
            "assets": {},
        }
    )
    restored = ProjectDocument.from_dict(value)
    assert restored.values["target_region_geometry"] is None
    assert restored.fingerprint == prior_fingerprint == project.fingerprint
    assert restored.model_fingerprint == prior_model == project.model_fingerprint
    broken = deepcopy(value)
    broken["config"].pop("levels")
    with pytest.raises(ValueError, match="every resolved"):
        ProjectDocument.from_dict(broken)


def test_imported_map_authoritative_even_when_region_would_be_invalid():
    target = TargetField(
        np.array([[0.1, 0.2, 0.3]]), np.array([[1e-6, -2e-6, 3e-6]]), np.array([2.0]), np.array([False]), np.array([1])
    )
    config = CoilGenConfig(
        target_field_definition_file=register_in_memory_target_field(target),
        target_region_geometry={"shape": "not_a_shape"},
    )
    actual = define_target_field([support()], None, None, config)
    np.testing.assert_array_equal(actual.coords, target.coords)
    np.testing.assert_array_equal(actual.b, target.b)
    np.testing.assert_array_equal(actual.weights, target.weights)
    assert any("authoritative" in item.message for item in validate_config(config).warnings)
    independent = validate_on_independent_grid(SimpleNamespace(input_data=config))
    assert independent["available"] is False and independent["passed"] is None


def test_independent_sheet_keeps_saved_expression_scale_and_does_not_refit():
    config = CoilGenConfig(
        target_region_geometry={**REGIONS[2], "center_m": [0.01, 0.005, -0.01]},
        target_region_resolution=5,
        field_shape_function="x*y + z*z",
        target_gradient_normalization="none",
        progress_flag=False,
        gauss_order=1,
    )
    mesh = support()
    original_current = np.array([0, 0, 0, 0, 1.0])
    part = CoilPart(mesh, stream_function=original_current.copy())
    result = SimpleNamespace(
        input_data=config,
        combined_mesh=mesh,
        coil_parts=[part],
        target_field=define_target_field([mesh], None, None, config),
        regularization_diagnostics={"target_gradient_normalization_scale": 0.0073},
    )
    check = validate_on_independent_grid(result, resolution=6)
    assert check["available"] and check["source"] == "continuous_sheet_independent_grid_no_current_refit"
    assert not check["independent_quadrature_check"]
    assert not coordinates(check["points_m"]) & coordinates(result.target_field.coords)
    assert check["target_T"][:, 2] == pytest.approx(
        0.0073 * evaluate_field_expression(config.field_shape_function, check["points_m"])
    )
    np.testing.assert_array_equal(part.stream_function, original_current)
    assert part.sensitivity_matrix is None and part.basis_elements == []
    assert np.all(np.isfinite(check["field_T"]))


def test_active_region_invalidates_every_stage_and_summaries_expose_scope(tmp_path):
    original = CoilGenConfig()
    updated = replace(original, target_region_geometry=REGIONS[0])
    assert config_field_stage("target_region_geometry") is PipelineStage.PREPARED
    before, after = stage_fingerprints(original, tmp_path), stage_fingerprints(updated, tmp_path)
    assert all(before[stage.value] != after[stage.value] for stage in PipelineStage)
    assert "radius 20 mm" in target_region_summary(updated)
    assert "bounding-box" in target_region_summary(replace(original, target_mesh_file="mesh.stl"))
    assert "Imported mesh boundary" in target_region_summary(
        replace(updated, target_region_geometry={"shape": "enclosing_mesh", "sampling": "surface"})
    )
    assert "Closed imported enclosure" in target_region_summary(
        replace(updated, target_region_geometry={"shape": "enclosing_mesh"})
    )


def slice_result(config):
    target = define_target_field([support()], None, None, config)
    return SimpleNamespace(input_data=config, target_field=target, coil_parts=(), diagnostic_cache={})


@pytest.mark.parametrize("domain", ["dsv", "roi", "fov", "target", "primary", ""])
def test_spherical_slice_diagnostics_reject_explicit_regions_without_false_coverage(domain):
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        target_region_geometry=REGIONS[2],
        target_region_resolution=5,
        target_gradient_normalization="none",
        gradient_slice_maps_enabled=True,
        gradient_slice_map_domain=domain,
    )
    spec = GradientSliceMapSpec(enabled=True, domain=domain, source="target", grid_size=8, slice_count=1)
    with pytest.raises(ValueError, match="bounding-box diagnostic"):
        evaluate_gradient_slice_maps(slice_result(config), spec)
    assert any(
        item.field == "gradient_slice_map_domain" and "explicit target-region" in item.message
        for item in validate_config(config).warnings
    )
    assert target_region_slice_map_limitation(replace(config, target_region_geometry=None), domain) == ""
    assert (
        target_region_slice_map_limitation(replace(config, target_field_definition_file="saved_map.npz"), domain) == ""
    )


def test_full_slice_diagnostic_reports_sample_bounding_box_and_region_changes_invalidate_cache():
    config = CoilGenConfig(
        target_region_geometry=REGIONS[2],
        target_region_resolution=5,
        field_shape_function="y",
        target_gradient_normalization="none",
        progress_flag=False,
    )
    result = slice_result(config)
    spec = GradientSliceMapSpec(
        enabled=True,
        domain="full",
        source="target",
        grid_size=8,
        slice_count=2,
        field_components=("Bz",),
        gradient_components=(),
        concomitant_components=(),
    )
    evaluated = evaluate_gradient_slice_maps(result, spec)
    assert evaluated.summary["gradient_slice_map_sampling_scope"] == "saved_target_sample_bounding_box_diagnostic"
    assert evaluated.spec.domain == "all"
    assert all(item.dsv_outline_radius_m is None for item in evaluated.slices)
    assert all(np.all(np.isfinite(item.maps["Bz"])) for item in evaluated.slices)
    old_signature = _slice_map_source_signature(result, spec)
    # Keep numerical arrays fixed: the region contract itself must invalidate.
    result.input_data = replace(config, target_region_geometry=REGIONS[0])
    assert _slice_map_source_signature(result, spec) != old_signature
