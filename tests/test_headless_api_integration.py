"""Contracts at the typed-design, numerical-pipeline and optional API boundary."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import (
    CoilDesignProblem,
    CoilGenConfig,
    CoilSupportSpec,
    LayoutSpec,
    Mesh,
    OptimizationSpec,
    SupportMesh,
    TargetFieldSpec,
    run_coilgen,
)
from coilgen.research import EnergyQuadrature, ResearchPhysicsSpec


def test_typed_physics_and_surface_policy_preserve_base_and_override_explicitly():
    mesh = Mesh(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]), np.array([[0, 1, 2]]))
    problem = CoilDesignProblem(
        target_field=TargetFieldSpec.from_arrays(np.array([[0.0, 0.0, 0.02]]), np.array([1e-5])),
        support=CoilSupportSpec("support", (SupportMesh(mesh, role="primary"),)),
    )
    initial = ResearchPhysicsSpec(objective="power", torque_limit_Nm=0.2).to_dict()
    base = CoilGenConfig(research_physics=initial, sweep_surface_realization="mesh")
    inherited = problem.to_config(base_config=base)
    assert inherited.research_physics == initial
    assert inherited.sweep_surface_realization == "mesh"

    physical = ResearchPhysicsSpec(objective="magnetic_energy", torque_limit_Nm=0.1).to_dict()
    explicit = replace(
        problem,
        optimization=OptimizationSpec(research_physics=physical),
        layout=LayoutSpec(sweep_surface_realization="analytic"),
    ).to_config(base_config=base)
    assert ResearchPhysicsSpec.from_dict(explicit.research_physics).torque_limit_Nm == 0.1
    assert explicit.sweep_surface_realization == "analytic"
    assert base.research_physics == initial
    assert base.sweep_surface_realization == "mesh"


def test_energy_resource_guard_precedes_field_operator_assembly(monkeypatch):
    import coilgen.pipeline as pipeline

    def unexpected_assembly(*args, **kwargs):
        pytest.fail("The resource guard must run before field operators are assembled.")

    monkeypatch.setattr(pipeline, "calculate_sensitivity_matrix", unexpected_assembly)
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.25, 0.1, 12, 6, 1, 0, 0, 0),
        research_physics=ResearchPhysicsSpec(
            objective="magnetic_energy", energy=EnergyQuadrature(maximum_faces=8)
        ).to_dict(),
        progress_flag=False,
    )
    with pytest.raises(ValueError, match="maximum_faces"):
        run_coilgen(config)


def test_quality_linearity_cannot_refit_current_after_physical_optimization():
    from coilgen.reporting import _gradient_linearity_field_for_result

    per_amp = np.array([[0.0, 0.0, 1e-3]])
    result = SimpleNamespace(
        input_data=CoilGenConfig(target_gradient_strength=0.01, research_physics={"objective": "power"}),
        layout_gradient={"achieved_gradient": 0.001},
        needed_current_layout=2.0,
        field_layout_per1Amp=per_amp,
        field_by_layout=2 * per_amp,
    )
    # A gradient refit would use 10 A and change power by a factor of 25.
    np.testing.assert_allclose(_gradient_linearity_field_for_result(result), 2 * per_amp)
    result.input_data.research_physics = None
    np.testing.assert_allclose(_gradient_linearity_field_for_result(result), 10 * per_amp)
