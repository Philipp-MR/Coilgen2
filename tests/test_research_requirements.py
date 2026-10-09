"""Physical regional contracts preserve units, nodal correspondence and currents."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from coilgen.application.gradient_studies import MeasuredFieldMap, RigidTransform
from coilgen.application.research_requirements import (
    DesiredDataset,
    FieldSamplingError,
    PhysicalScenario,
    RegionalRequirement,
    ResearchRequirements,
    evaluate_requirements,
    nominal_requirements,
    prepare_requirement_constraints,
    requirements_target,
    _metrics,
    named_independent_sheet_validation,
    requirements_constraint_rows,
)
from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh
from coilgen.config import CoilGenConfig
from coilgen.geometry import compute_face_normals, compute_vertex_normals
from coilgen.models import CoilPart, Mesh, WirePath
from coilgen.optimization import StreamFunctionOptimizationProblem, solve_stream_function_problem
from coilgen.physics import biot_savart_wire_path, calculate_resistance_matrix, calculate_sensitivity_matrix
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint
from coilgen.target_regions import TargetRegion


def dataset(points=((0, 0, 0.04),), field=((0, 0, 1e-6),), *, independent=False):
    return DesiredDataset(MeasuredFieldMap(points, field, independent=independent))


def parts_config():
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 0]], float) * 0.05
    mesh = Mesh(
        vertices, np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]), boundaries=[np.array([0, 1, 2, 3, 0])]
    )
    mesh.face_normals = compute_face_normals(mesh)
    mesh.vertex_normals = compute_vertex_normals(mesh)
    parts = [CoilPart(mesh)]
    calculate_one_ring_by_mesh(parts)
    calculate_basis_functions(parts)
    config = CoilGenConfig(
        progress_flag=False, gauss_order=3, symmetry_reduction="off", constrained_optimization_tolerance=1e-10
    )
    return parts, config


def test_registration_rotates_vectors_and_coordinates_and_roundtrips_owned_units():
    rotation = RigidTransform(((0, -1, 0), (1, 0, 0), (0, 0, 1)), (0.2, 0, 0))
    value = DesiredDataset(MeasuredFieldMap([[10, 0, 0]], [[3, 0, 0]], "mm", "mT"), rotation)
    points, field = value.world_values()
    np.testing.assert_allclose(points, [[0.2, 0.01, 0]])
    np.testing.assert_allclose(field, [[0, 0.003, 0]])
    restored = DesiredDataset.from_dict(value.to_dict())
    assert restored.digest == value.digest
    assert not restored.data.coordinates.flags.writeable


def test_independence_rejects_reused_and_unit_converted_training_points():
    design = dataset()
    copied = DesiredDataset(replace(design.data, independent=True))
    with pytest.raises(ValueError, match="repeats"):
        ResearchRequirements((RegionalRequirement("A", design, peak_limit_T=1e-7, validation=copied),))
    copied = DesiredDataset(MeasuredFieldMap([[0, 0, 40]], [[0, 0, 2]], "mm", "uT", independent=True))
    with pytest.raises(ValueError, match="overlap"):
        ResearchRequirements((RegionalRequirement("A", design, peak_limit_T=1e-7, validation=copied),))


def test_finite_poses_cannot_repeat_nominal_or_training_as_heldout():
    region = RegionalRequirement("A", dataset(), peak_limit_T=1e-7)
    with pytest.raises(ValueError, match="differ"):
        ResearchRequirements((region,), (PhysicalScenario("Fake heldout", "validation"),))
    pose = PhysicalScenario("Moved", support_transform=RigidTransform(translation_m=(0, 0, 0.003)))
    with pytest.raises(ValueError, match="repeat training"):
        ResearchRequirements((region,), (pose, replace(pose, name="Heldout", membership="validation")))
    robust = ResearchRequirements((region,), (pose,))
    nominal = nominal_requirements(robust)
    assert nominal.scenarios[0].membership == "validation"
    assert nominal.digest != robust.digest
    assert nominal.evaluation_contract_digest == robust.evaluation_contract_digest


def test_separate_regional_budgets_do_not_change_with_fit_grid_weights():
    parts, config = parts_config()
    first = RegionalRequirement("Useful ROI", dataset(), peak_limit_T=1e-7, weight=1e-4)
    second = RegionalRequirement(
        "Protected sensor", dataset(((0, 0, 0.08),), ((0, 0, 0),)), peak_limit_T=1e-6, protected=True
    )
    requirements = ResearchRequirements((first, second))
    target = requirements_target(requirements)
    calculate_sensitivity_matrix(parts, target, config)
    calculate_resistance_matrix(parts, config)
    problem = StreamFunctionOptimizationProblem.from_parts(parts, target, config)
    physical = prepare_requirement_constraints(parts, config, requirements)
    solution = solve_stream_function_problem(problem, physical)
    actual = (
        problem.reducer.expand(solution.reduced_stream_function)
        if hasattr(problem.reducer, "expand")
        else problem.reducer.expansion_matrix @ solution.reduced_stream_function
    )
    fields = np.einsum("ctv,v->tc", parts[0].sensitivity_matrix, actual)
    assert fields[0, 2] == pytest.approx(0.9e-6, rel=2e-5)
    assert abs(fields[1, 2]) <= 1e-6
    constraints = physical.research_physics["field_constraints"]
    assert [row["tolerance"] for row in constraints] == [1e-7, 1e-6]
    np.testing.assert_array_equal(target.weights, [1e-4, 1])
    assert not physical.research_physics["include_default_field_constraints"]


def test_support_and_sample_joint_rotation_preserves_full_vector_operator():
    parts, config = parts_config()
    region = RegionalRequirement("Vector ROI", dataset(((0.01, 0.005, 0.04),)), "vector", peak_limit_T=1e-7)
    rotation = RigidTransform(((0, -1, 0), (1, 0, 0), (0, 0, 1)), (0.2, -0.3, 0.1))
    spec = ResearchRequirements(
        (region,), (PhysicalScenario("Together", support_transform=rotation, sample_transform=rotation),)
    )
    physical = prepare_requirement_constraints(parts, config, spec)
    rows = physical.research_physics["field_constraints"]
    original, moved = (np.asarray(row["matrix"]) for row in rows)
    np.testing.assert_allclose(moved, np.asarray(rotation.rotation) @ original, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(rows[1]["target"], np.asarray(rotation.rotation) @ np.array(rows[0]["target"]))
    assert rows[1]["group_size"] == 3


def test_sheet_surface_samples_are_rejected_instead_of_kernel_zero():
    parts, config = parts_config()
    spec = ResearchRequirements((RegionalRequirement("On support", dataset(((0.01, 0.01, 0),)), peak_limit_T=1e-7),))
    with pytest.raises(FieldSamplingError, match="singular sheet"):
        prepare_requirement_constraints(parts, config, spec)


def test_final_winding_scenarios_use_actual_path_and_original_signed_current():
    parts, config = parts_config()
    path = np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0], [-0.05, -0.05, 0]])
    parts[0].wire_path = WirePath(v=path)
    parts[0].contour_step = -2
    parts[0].stream_function = np.array([0, 0, 0, 0, 1.0])
    points = np.array([[0, 0, 0.04], [0.01, 0, 0.045]])
    exact = -2 * biot_savart_wire_path(path, points, worker_count=1)
    design = dataset(points, exact)
    validation = dataset([[0, 0.01, 0.05]], [[0, 0, 0]], independent=True)
    region = RegionalRequirement("ROI", design, "vector", peak_limit_T=1e-15, validation=validation)
    shift = RigidTransform(translation_m=(0, 0, 0.005))
    spec = ResearchRequirements((region,), (PhysicalScenario("Shift", "validation", support_transform=shift),))
    report, arrays = evaluate_requirements(parts, config, spec)
    wire_rows = [row for row in report["rows"] if row["scope"] == "saved_routed_centroids"]
    nominal = wire_rows[0]
    assert nominal["passed"]
    moved = next(row for row in wire_rows if row["scenario"] == "Shift" and row["sampling"] == "design")
    expected = -2 * biot_savart_wire_path(path + shift.translation_m, points, worker_count=1)
    np.testing.assert_allclose(arrays[moved["array_prefix"] + "_field_T"], expected, rtol=1e-14)
    assert moved["scenario_membership"] == "validation"
    assert report["current_refitted"] is False
    assert report["passed"] is False
    assert moved["failed_sample_indices"]
    assert not report["external_feeds_included"]


def test_finite_radial_tolerance_moves_sheet_and_saved_wire_about_authored_centre():
    parts, config = parts_config()
    centre = np.array((0.01, 0.02, 0))
    path = (
        np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0], [-0.05, -0.05, 0]]) + centre
    )
    parts[0].wire_path = WirePath(v=path)
    parts[0].contour_step = 1.25
    parts[0].stream_function = np.array([0, 0, 0, 0, 1.0])
    factors = (1.01, 1.01, 1.0)
    region = RegionalRequirement("ROI", dataset(((0.01, 0.02, 0.04),)), "vector", peak_limit_T=1e-3)
    pose = PhysicalScenario(
        "Radial manufacturing tolerance", "validation", support_scale_xyz=factors, support_scale_center_m=tuple(centre)
    )
    spec = ResearchRequirements((region,), (pose,))
    report, arrays = evaluate_requirements(parts, config, spec)
    row = next(
        row for row in report["rows"] if row["scenario"] == pose.name and row["scope"] == "saved_routed_centroids"
    )
    transformed = (path - centre) * factors + centre
    expected = 1.25 * biot_savart_wire_path(transformed, region.design.world_values()[0], worker_count=1)
    np.testing.assert_allclose(arrays[row["array_prefix"] + "_field_T"], expected, rtol=1e-14)
    assert row["support_scale_xyz"] == list(factors)
    assert not report["perturbed_resistance_power_assessed"]
    physical = prepare_requirement_constraints(
        parts, config, replace(spec, scenarios=(replace(pose, membership="design"),))
    )
    moved_mesh = parts[0].mesh.copy()
    moved_mesh.vertices = (moved_mesh.vertices - centre) * factors + centre
    moved_mesh.face_normals = compute_face_normals(moved_mesh)
    moved_mesh.vertex_normals = compute_vertex_normals(moved_mesh)
    moved_parts = [CoilPart(moved_mesh)]
    calculate_one_ring_by_mesh(moved_parts)
    calculate_basis_functions(moved_parts)
    calculate_sensitivity_matrix(moved_parts, requirements_target(spec), config)
    expected_operator = np.moveaxis(moved_parts[0].sensitivity_matrix, 0, 1).reshape(3, -1)
    np.testing.assert_allclose(physical.research_physics["field_constraints"][1]["matrix"], expected_operator)
    assert ResearchRequirements.from_dict(spec.to_dict()).digest == spec.digest


@pytest.mark.parametrize("scales", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (True, 1.0, 1.0), (np.inf, 1.0, 1.0)])
def test_dimensional_factors_require_positive_finite_real_values(scales):
    with pytest.raises(ValueError):
        PhysicalScenario("Invalid tolerance", support_scale_xyz=scales)


def test_partial_fabrication_evaluates_recorded_paths_but_cannot_pass_complete_circuit():
    parts, config = parts_config()
    early = np.array([[-0.05, 0, 0], [0.05, 0, 0]])
    recorded = np.array([[-0.05, 0, 0], [0.05, 0, 0], [0.05, 0.03, 0]])
    parts[0].wire_path = WirePath(v=early)
    parts[0].contour_step = 100
    points = np.array([[0, 0, 0.04]])
    current = -1.7
    field = current * biot_savart_wire_path(recorded, points, worker_count=1)
    spec = ResearchRequirements((RegionalRequirement("ROI", dataset(points, field), "vector", peak_limit_T=1e-15),))
    fabrication = SimpleNamespace(
        winding_steps=[{"body_id": "A", "electrical_points_m": recorded.tolist(), "circuit_complete": False}],
        electrical_metrics={"bodies": [{"body_id": "A", "current_A": current}]},
    )
    report, arrays = evaluate_requirements(parts, config, spec, fabrication=fabrication)
    row = next(row for row in report["rows"] if row["scope"] == "partial_electrical_paths")
    np.testing.assert_allclose(arrays[row["array_prefix"] + "_field_T"], field)
    assert row["currents_A"] == [current]
    assert row["residual_bounds_met"] is True
    assert row["passed"] is None
    assert report["passed"] is None


def test_unconnected_contours_do_not_certify_routed_evidence():
    parts, config = parts_config()
    parts[0].stream_function = np.zeros(5)
    spec = ResearchRequirements((RegionalRequirement("ROI", dataset(), peak_limit_T=1e-5),))
    report, _ = evaluate_requirements(parts, config, spec)
    assert report["rows"][0]["available"]
    assert not report["rows"][1]["available"]
    assert report["passed"] is None


def test_regional_gradient_fit_is_scaled_descriptive_and_reports_suppression():
    points = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]) * 1e-4
    points += [1e3, -2e3, 3e3]
    gradient = np.array([[0.1, 0.2, -0.3], [0.4, -0.1, 0.2], [0.3, -0.2, 0.5]])
    field = (points - points.mean(axis=0)) @ gradient.T + [1e-6, -2e-6, 3e-6]
    desired = np.zeros_like(field)
    region = RegionalRequirement("Sensor", dataset(points, desired), "vector", peak_limit_T=1e-3, protected=True)
    report = _metrics(field, desired, points, region, CoilGenConfig())
    np.testing.assert_allclose(report["field_gradient_T_per_m"], gradient, rtol=1e-13, atol=1e-15)
    assert report["regional_gradient_rank"] == 4
    assert report["regional_gradient_condition_number"] < 2
    assert report["regional_gradient_acceptance_claim"] is False
    assert report["suppression_peak_T"] == pytest.approx(np.linalg.norm(field, axis=1).max())
    planar = _metrics(field[:4], desired[:4], points[:4], region, CoilGenConfig())
    assert not planar["regional_gradient_available"]
    assert "field_gradient_T_per_m" not in planar


def test_named_acceptance_exposes_raw_effective_budgets_and_validation_margin():
    points = np.array([[0, 0, 0.04]])
    target = np.zeros_like(points)
    region = RegionalRequirement("ROI", dataset(points, target), peak_limit_T=1e-6)
    config = CoilGenConfig(
        research_physics={"acceptance_relative_tolerance": 0.01, "acceptance_absolute_by_unit": {"T": 2e-9}}
    )
    within_numerical_slack = _metrics([[0, 0, 1.005e-6]], target, points, region, config)
    assert within_numerical_slack["passed"]
    assert within_numerical_slack["peak_limit_T"] == 1e-6
    assert within_numerical_slack["effective_peak_limit_T"] == pytest.approx(1.012e-6)
    assert not _metrics([[0, 0, 1.3e-6]], target, points, region, config)["passed"]
    with_margin = _metrics([[0, 0, 1.3e-6]], target, points, region, config, design_margin=0.5)
    assert with_margin["passed"]
    assert with_margin["validation_peak_limit_T"] == 2e-6
    assert with_margin["effective_peak_limit_T"] == pytest.approx(2.022e-6)
    parts, _ = parts_config()
    spec = ResearchRequirements((region,))
    first, _ = evaluate_requirements(parts, config, spec)
    second, _ = evaluate_requirements(parts, config, spec, design_margin=0.5)
    assert first["evaluation_contract_sha256"] != second["evaluation_contract_sha256"]


def test_pipeline_checkpoint_owns_requirements_and_rejects_changed_contract(tmp_path):
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        symmetry_reduction="off",
        gauss_order=3,
        progress_flag=False,
        skip_postprocessing=True,
        skip_inductance_calculation=True,
        constrained_optimization_tolerance=1e-10,
    )
    region = TargetRegion("box", size_m=(0.02, 0.02, 0.02))
    spec = ResearchRequirements(
        (
            RegionalRequirement(
                "Constant ROI", DesiredDataset.analytic(region, ("0", "0", "1e-6"), resolution=2), peak_limit_T=2e-7
            ),
        )
    )
    prepared = CoilGenPipeline(config, requirements=spec).run_stage("prepared")
    restored = PipelineCheckpoint.load(prepared.save(tmp_path / "requirements.cgstage"))
    solved = CoilGenPipeline(config, requirements=spec).run_stage("sheet", restored)
    report, _ = evaluate_requirements(solved.parts, config, spec)
    assert report["rows"][0]["passed"]
    assert solved.config.research_physics is None  # Dense generated matrices are derived, not native inputs.
    changed = replace(spec, regions=(replace(spec.regions[0], peak_limit_T=1e-7),))
    with pytest.raises(ValueError, match="Cannot reuse prepared"):
        CoilGenPipeline(config, requirements=changed).run_stage("sheet", restored)
    with pytest.raises(ValueError, match="Cannot reuse prepared"):
        CoilGenPipeline(config).run_stage("sheet", restored)


@pytest.mark.parametrize("stage", ["sheet", "routed"])
def test_partial_research_run_keeps_owned_regional_evidence_and_scope(tmp_path, stage):
    from coilgen.research.application import run_design
    from coilgen.research.requests import ResearchRequest
    from coilgen.application.service import DesignService
    from coilgen.application.spatial_scene import ScenePart, SpatialScene

    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        field_shape_function="none",
        research_physics={"objective": "power"},
        symmetry_reduction="off",
        gauss_order=3,
        progress_flag=False,
        levels=3,
        min_point_loop_number=3,
        skip_postprocessing=False,
        skip_sweep=True,
        skip_inductance_calculation=True,
        constrained_optimization_tolerance=1e-10,
    )
    region = TargetRegion("box", size_m=(0.02, 0.02, 0.02))
    design = DesiredDataset.analytic(region, ("0", "0", "1e-6"), resolution=2)
    validation = DesiredDataset.analytic(region, ("0", "0", "1e-6"), resolution=2, independent=True)
    spec = ResearchRequirements(
        (RegionalRequirement("Owned constant ROI", design, peak_limit_T=3e-7, validation=validation),)
    )
    request = ResearchRequest.from_config(
        config,
        name="Partial regional evidence",
        pipeline_stage=stage,
        requirements=spec.to_dict(),
        scene=SpatialScene((ScenePart.box("Far obstacle", center_m=(1, 0, 0)),)).to_dict(),
    )
    run_design(request, tmp_path / stage)
    saved = DesignService.load_run(tmp_path / stage)
    metadata = saved.metadata
    assert metadata["validation"]["scope"] == "named_regional_independent_sheet"
    assert metadata["validation"]["available"]
    assert "validation_points_m" not in saved.array_names
    report = metadata["requirements_review"]
    independent = next(
        row for row in report["rows"] if row["scope"] == "continuous_sheet" and row["independent_sampling"]
    )
    np.testing.assert_array_equal(saved.array(independent["array_prefix"] + "_points_m"), validation.world_values()[0])
    np.testing.assert_array_equal(saved.array(independent["array_prefix"] + "_target_T"), validation.world_values()[1])
    assert any(row["name"].endswith("· peak") and row["limit"] == 3e-7 for row in metadata["constraints"])
    assert not metadata["summary"]["construction_released"]
    if stage == "sheet":
        assert all(not row.get("available") for row in report["rows"] if row["scope"] != "continuous_sheet")
        assert all(row["scope"] == "continuous_sheet" for row in metadata["constraints"])
    else:
        assert not metadata["summary"]["accepted"]
        wire = [row for row in report["rows"] if row["scope"] == "saved_routed_centroids"]
        assert wire and all(row["available"] for row in wire)
        assert not report["external_feeds_included"]
        assert metadata["scene_review"]["passed"] is True
        assert any(row["name"].startswith("Spatial: Far obstacle") for row in metadata["constraints"])


def test_regional_check_rows_separate_bounds_and_keep_partial_unknown():
    row = {
        "region": "ROI",
        "scenario": "Nominal",
        "sampling": "design",
        "scope": "continuous_sheet",
        "available": True,
        "passed": False,
        "rms_error_T": 1e-6,
        "peak_error_T": 3e-6,
        "rms_limit_T": 2e-6,
        "effective_rms_limit_T": 2.1e-6,
        "peak_limit_T": 2e-6,
        "effective_peak_limit_T": 2.1e-6,
    }
    report = {"rows": [row, {**row, "scope": "partial_electrical_paths", "passed": None}]}
    checks = requirements_constraint_rows(report)
    assert [check["passed"] for check in checks] == [True, False, None, None]
    assert len(requirements_constraint_rows(report, include_wire=False)) == 2
    config = CoilGenConfig()
    spec = ResearchRequirements((RegionalRequirement("ROI", dataset(), peak_limit_T=1e-6),))
    report, _ = evaluate_requirements(parts_config()[0], config, spec)
    assert not named_independent_sheet_validation(report, spec)["available"]
