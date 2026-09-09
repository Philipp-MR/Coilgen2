from dataclasses import replace
import json

import numpy as np
import pytest

from coilgen import CoilGenConfig, run_coilgen
from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh
from coilgen.geometry import compute_face_normals, compute_vertex_normals
from coilgen.models import CoilPart, Mesh, TargetField
from coilgen.optimization import StreamFunctionOptimizationProblem
from coilgen.physics import calculate_resistance_matrix, calculate_sensitivity_matrix
from coilgen.research.convex import (
    FieldScenario,
    LinearFieldConstraint,
    ParameterizedResearchProblem,
    ResearchPhysicsSpec,
)
from coilgen.research.physics import surface_operators


def planar_problem(target_T=1e-6):
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 0]], dtype=float) * 0.05
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    mesh = Mesh(vertices, faces, boundaries=[np.array([0, 1, 2, 3, 0])])
    mesh.face_normals = compute_face_normals(mesh)
    mesh.vertex_normals = compute_vertex_normals(mesh)
    parts = [CoilPart(mesh)]
    config = CoilGenConfig(
        sf_opt_method="constrained_quadratic",
        symmetry_reduction="off",
        progress_flag=False,
        gauss_order=3,
        target_error_constraint_metric="rms",
        target_error_rms_limit_T=target_T * 0.1,
        target_error_constraint_acceptance_absolute_T=0,
        constrained_optimization_tolerance=1e-9,
    )
    target = TargetField(
        np.array([[0, 0, 0.04]]),
        np.array([[0, 0, target_T]]),
        np.ones(1),
        np.zeros(1, dtype=bool),
        np.ones(1, dtype=int),
    )
    calculate_one_ring_by_mesh(parts)
    calculate_basis_functions(parts)
    calculate_resistance_matrix(parts, config)
    calculate_sensitivity_matrix(parts, target, config)
    return StreamFunctionOptimizationProblem.from_parts(parts, target, config), config


@pytest.mark.parametrize("objective", ["power", "magnetic_energy", "minimax", "mixture"])
@pytest.mark.parametrize("solver", ["CLARABEL", "SCS"])
def test_physical_objectives_have_known_single_dof_optimum(objective, solver):
    problem, config = planar_problem()
    config.constrained_optimization_solver = solver
    spec = ResearchPhysicsSpec(objective=objective, evaluate_energy=True)
    compiled = ParameterizedResearchProblem(problem, spec, config)
    result = compiled.solve()
    achieved = problem.reduced_sensitivity @ result.reduced_stream_function
    assert achieved == pytest.approx([0.9e-6], rel=2e-5)
    assert result.diagnostics["sheet_power_W"] > 0
    assert result.diagnostics["energy_J"] > 0
    assert result.diagnostics["constrained_solver_contract_met"]
    assert np.linalg.norm(result.diagnostics["force_N"]) < 1e-14
    assert compiled.cvxpy_problem.is_dcp(dpp=True)
    if objective == "mixture":
        assert compiled.reference_diagnostics["source"] == "minimum_power_feasible_at_tolerance_scale_1"
        assert set(compiled.objective_scales) == {"power", "magnetic_energy"}


def test_parameterized_tolerance_sweep_reuses_problem_and_increases_power_when_tightened():
    problem, config = planar_problem()
    compiled = ParameterizedResearchProblem(problem, ResearchPhysicsSpec(), config)
    identity = id(compiled.cvxpy_problem)
    loose = compiled.solve(tolerance_scale=2)
    tight = compiled.solve(tolerance_scale=0.5)
    assert id(compiled.cvxpy_problem) == identity
    assert compiled.solve_count == 2
    assert tight.diagnostics["sheet_power_W"] > loose.diagnostics["sheet_power_W"]
    assert problem.reduced_sensitivity @ tight.reduced_stream_function == pytest.approx([0.95e-6], rel=2e-5)


def test_mixture_sweep_freezes_explicit_reference_budget_independent_of_point_order():
    problem, config = planar_problem()
    spec = ResearchPhysicsSpec(objective="mixture", reference_tolerance_scale=2.0)
    model = ParameterizedResearchProblem(problem, spec, config)
    graph = model.cvxpy_problem
    scales = dict(model.objective_scales)
    forward = model.solve_many([0.5, 2.0])
    backward = model.solve_many([2.0, 0.5])
    assert model.cvxpy_problem is graph
    assert model.solve_count == 4
    assert model.reference_diagnostics["tolerance_scale"] == 2.0
    # Single physical degree of freedom: minimum-power reference at a 20%
    # field budget has 0.8/0.95 of the tight solution's current amplitude.
    assert scales["power"] / forward[0].diagnostics["sheet_power_W"] == pytest.approx((0.8 / 0.95) ** 2, rel=2e-5)
    for solution in (*forward, *backward):
        assert solution.diagnostics["research_objective_scales"] == scales
    assert forward[0].reduced_stream_function == pytest.approx(backward[1].reduced_stream_function, rel=2e-5)
    assert forward[1].reduced_stream_function == pytest.approx(backward[0].reduced_stream_function, rel=2e-5)
    forward[0].diagnostics["research_objective_scales"]["power"] = -1
    assert model.objective_scales == scales


def test_energy_budget_fails_before_building_surface_operators(monkeypatch):
    from coilgen.research import convex

    problem, config = planar_problem()

    def unexpected_assembly(_problem):
        raise AssertionError("Heavy operator assembly must follow preflight.")

    monkeypatch.setattr(convex, "surface_operators", unexpected_assembly)
    spec = ResearchPhysicsSpec.from_dict({"objective": "magnetic_energy", "energy": {"maximum_faces": 2}})
    with pytest.raises(ValueError, match="4 faces.*maximum_faces=2"):
        ParameterizedResearchProblem(problem, spec, config)
    with pytest.raises(ValueError, match="maximum_faces"):
        convex.preflight_research_physics(replace(spec, objective="power", evaluate_energy=True), 4)
    assert not convex.preflight_research_physics(replace(spec, objective="power"), 4)["energy_requested"]


def test_compiled_contract_owns_custom_operator_snapshot():
    problem, config = planar_problem()
    matrix = problem.sensitivity_matrix[2].copy()
    spec = ResearchPhysicsSpec(field_constraints=(LinearFieldConstraint("exact", matrix, [1e-6], norm="equality"),))
    model = ParameterizedResearchProblem(problem, spec, config)
    matrix[:] = 0
    result = model.solve()
    assert problem.reduced_sensitivity @ result.reduced_stream_function == pytest.approx([1e-6], rel=1e-7)


@pytest.mark.parametrize(
    "setting,metric,row_name,unit",
    [
        ("sheet_power_limit_W", "sheet_power_W", "sheet_power", "W"),
        ("magnetic_energy_limit_J", "energy_J", "magnetic_energy", "J"),
    ],
)
def test_convex_physical_cap_is_fixed_across_tolerance_sweep(setting, metric, row_name, unit):
    problem, config = planar_problem()
    baseline = ParameterizedResearchProblem(problem, ResearchPhysicsSpec(evaluate_energy=True), config).solve()
    maximum = baseline.diagnostics[metric] * 1.01
    spec = ResearchPhysicsSpec(**{setting: maximum})
    model = ParameterizedResearchProblem(problem, spec, config)
    loose, nominal = model.solve_many([2.0, 1.0])
    for result in (loose, nominal):
        row = next(row for row in result.diagnostics["research_constraint_diagnostics"] if row["name"] == row_name)
        assert row["unit"] == unit
        assert row["limit"] == maximum
        assert row["met"]
        assert result.diagnostics[metric] <= maximum * 1.0001
    # The tighter field now needs more current, so the unchanged physical cap
    # makes this point infeasible. A sweep must never relax that cap.
    with pytest.raises(RuntimeError, match="No research conic solve succeeded"):
        model.solve(tolerance_scale=0.1)


def test_quadratic_power_cap_is_recomputed_from_coefficients():
    problem, config = planar_problem()
    model = ParameterizedResearchProblem(problem, ResearchPhysicsSpec(sheet_power_limit_W=1e-3), config)
    rows = model._check_bounds(np.array([1e5]), 1.0)
    power = next(row for row in rows if row["name"] == "sheet_power")
    assert power["value"] > power["effective_limit"]
    assert not power["met"]


def test_numpy_spec_values_serialize_and_accept_with_builtin_booleans():
    problem, config = planar_problem()
    spec = ResearchPhysicsSpec(
        sheet_power_limit_W=np.float64(1.0),
        force_limit_N=np.float64(1.0),
        reference_tolerance_scale=np.float64(1.0),
        evaluate_energy=np.bool_(False),
    )
    json.dumps(spec.to_dict())
    solution = ParameterizedResearchProblem(problem, spec, config).solve()
    assert all(row["met"] is True for row in solution.diagnostics["research_constraint_diagnostics"])


def test_finite_robust_scenarios_enforce_every_independent_bound():
    problem, config = planar_problem()
    config.target_error_rms_limit_T = 0.2e-6
    field = problem.sensitivity_matrix[2]
    spec = ResearchPhysicsSpec(
        scenarios=(
            FieldScenario("weak_gain", 0.9 * field, [1e-6], rms_limit_T=0.05e-6),
            FieldScenario("strong_gain", 1.02 * field, [1e-6], peak_limit_T=0.15e-6),
        )
    )
    result = ParameterizedResearchProblem(problem, spec, config).solve()
    achieved = problem.reduced_sensitivity @ result.reduced_stream_function
    assert achieved == pytest.approx([0.95e-6 / 0.9], rel=2e-5)
    rows = result.diagnostics["research_constraint_diagnostics"]
    assert len([row for row in rows if row["name"].startswith("scenario:")]) == 2
    assert all(row["met"] for row in rows)


def test_general_field_equality_and_one_sided_bound_survive_tolerance_sweep():
    problem, config = planar_problem()
    matrix = problem.sensitivity_matrix[2]
    spec = ResearchPhysicsSpec(
        field_constraints=(
            LinearFieldConstraint("exact_Bz", matrix, [1e-6], norm="equality"),
            LinearFieldConstraint("upper_Bz", matrix, [1.01e-6], norm="upper"),
        )
    )
    compiled = ParameterizedResearchProblem(problem, spec, config)
    result = compiled.solve(tolerance_scale=3)
    assert problem.reduced_sensitivity @ result.reduced_stream_function == pytest.approx([1e-6], rel=1e-7)


def test_solver_false_optimal_is_rejected_by_physical_recomputation(monkeypatch):
    problem, config = planar_problem()
    compiled = ParameterizedResearchProblem(problem, ResearchPhysicsSpec(), config)

    def fake_optimal(_problem):
        compiled.variable.value = np.zeros(compiled.variable.shape)
        return "mock_false_optimal", []

    monkeypatch.setattr(compiled, "_run_backend", fake_optimal)
    with pytest.raises(RuntimeError, match="physical acceptance failed.*standard:dsv"):
        compiled.solve()


def test_infeasible_hardware_constraint_does_not_fall_back():
    problem, config = planar_problem()
    config.surface_current_density_limit_A_per_m = 1e-8
    with pytest.raises(RuntimeError, match="No research conic solve succeeded"):
        ParameterizedResearchProblem(problem, ResearchPhysicsSpec(), config).solve()


def test_zero_force_equality_preserves_closed_current_solution():
    problem, config = planar_problem()
    spec = ResearchPhysicsSpec(b0_T=(0, 3, 0), force_equality_N=(0, 0, 0))
    result = ParameterizedResearchProblem(problem, spec, config).solve()
    assert result.diagnostics["force_N"] == pytest.approx([0, 0, 0], abs=1e-15)
    assert np.linalg.norm(result.diagnostics["torque_Nm"]) > 0


def test_torque_constraint_is_independently_checked_after_false_optimal(monkeypatch):
    problem, config = planar_problem()
    baseline = ParameterizedResearchProblem(problem, ResearchPhysicsSpec(b0_T=(0, 3, 0)), config).solve()
    torque = np.linalg.norm(baseline.diagnostics["torque_Nm"])
    compiled = ParameterizedResearchProblem(
        problem, ResearchPhysicsSpec(b0_T=(0, 3, 0), torque_limit_Nm=torque / 2), config
    )

    def fake_optimal(_problem):
        compiled.variable.value = baseline.reduced_stream_function / compiled.current_scale
        return "mock_false_optimal", []

    monkeypatch.setattr(compiled, "_run_backend", fake_optimal)
    with pytest.raises(RuntimeError, match="physical acceptance failed.*torque"):
        compiled.solve()


def test_nonuniform_background_force_bound_is_real_and_not_ignored():
    problem, config = planar_problem()
    background = np.array([[0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]])
    spec = ResearchPhysicsSpec(background_B_per_face_T=background)
    baseline = ParameterizedResearchProblem(problem, spec, config).solve()
    force = np.linalg.norm(baseline.diagnostics["force_N"])
    assert force > 0
    with pytest.raises(RuntimeError, match="No research conic solve succeeded"):
        ParameterizedResearchProblem(problem, replace(spec, force_limit_N=force / 2), config).solve()


def test_current_operator_corruption_is_detected_independently():
    problem, _ = planar_problem()
    problem.part_current_density_mats = tuple(np.zeros_like(matrix) for matrix in problem.part_current_density_mats)
    with pytest.raises(ValueError, match="disagree with the triangle geometry"):
        surface_operators(problem)


def test_units_scaling_power_energy_and_peak_current():
    values = []
    for target in (1e-6, 1e-3):
        problem, config = planar_problem(target)
        result = ParameterizedResearchProblem(problem, ResearchPhysicsSpec(evaluate_energy=True), config).solve()
        values.append(result.diagnostics)
    assert values[1]["sheet_power_W"] / values[0]["sheet_power_W"] == pytest.approx(1e6, rel=2e-5)
    assert values[1]["energy_J"] / values[0]["energy_J"] == pytest.approx(1e6, rel=2e-5)
    assert values[1]["max_surface_current_density_A_per_m"] / values[0][
        "max_surface_current_density_A_per_m"
    ] == pytest.approx(1e3, rel=2e-5)


def test_serialized_spec_round_trip_and_validation():
    spec = ResearchPhysicsSpec(scenarios=(FieldScenario("offset", np.eye(3), np.zeros(3), rms_limit_T=1e-6),))
    payload = json.loads(json.dumps(spec.to_dict()))
    assert ResearchPhysicsSpec.from_dict(payload).to_dict() == payload
    with pytest.raises(ValueError, match="nonnegative"):
        replace(spec, objective="mixture", objective_weights={"power": -1})
    with pytest.raises(ValueError, match="strictly positive"):
        replace(spec, objective_scales={"power": 0})


def test_research_objective_flows_through_real_coil_pipeline():
    result = run_coilgen(
        CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.2, 0.12, 8, 4, 1, 0, 0, 0),
            field_shape_function="y",
            target_region_radius=0.035,
            target_region_resolution=5,
            levels=4,
            gauss_order=1,
            target_error_tolerance_percent=40,
            min_point_loop_number=3,
            contourability_validation="auto",
            contourability_repair_enabled=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance_calculation=True,
            progress_flag=False,
            research_physics={"objective": "power"},
        )
    )
    assert result.regularization_diagnostics["constrained_solver_contract_met"]
    assert result.regularization_diagnostics["sheet_power_W"] > 0
    assert not result.regularization_diagnostics["surface_current_density_scaling_applied"]
    assert result.regularization_factor == 0
    assert any(part.contour_lines for part in result.coil_parts)
    assert result.input_data.field_evaluation_current_source == "target_fit"
    assert result.regularization_diagnostics["physical_current_refit_disabled"]
    assert result.regularization_diagnostics["field_evaluation_current_source"] == "stream_function"
    assert result.needed_current_layout == pytest.approx(result.coil_parts[0].contour_step)
    assert result.field_by_layout == pytest.approx(result.field_layout_per1Amp * result.coil_parts[0].contour_step)
    assert result.regularization_diagnostics["contourability_target_current_source"] == "stream_function"
