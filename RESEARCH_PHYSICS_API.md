# Headless physical coil design

Install `python -m pip install -e ".[research,accelerate]"` from the main
repository. `research` supplies SciPy, CVXPY and native-thread control;
`accelerate` adds optional cached Numba geometry kernels. No server or GUI is
installed. The normal `coilgen` API remains available.

## Select physical objectives

Set `CoilGenConfig.research_physics` to a serialized `ResearchPhysicsSpec` and
call `run_coilgen(config)`. The typed API accepts the same dictionary through
`OptimizationSpec(research_physics=...)`. An unspecified typed field preserves
the base config. With `research_physics=None`, existing optimization remains
unchanged. Preoptimized stream functions cannot bypass this physical solver.

```python
from coilgen.research import ResearchPhysicsSpec

config.research_physics = ResearchPhysicsSpec(
    objective="magnetic_energy",
    b0_T=(0, 0, 3),
    torque_origin_m=(0, 0, 0),
    torque_limit_Nm=0.2,
    sheet_power_limit_W=20,
).to_dict()
```

All lengths, fields, currents and loads use SI units. The unknown expands as
`psi = E x`, preserving the existing boundary/gauge/symmetry/spectral reduction.
Unsupported positive-genus topology is rejected because independent harmonic
circulation modes are not implemented.

| Objective | Quantity |
|---|---|
| `power` | Continuous-sheet Joule loss `x.T R x`, W |
| `magnetic_energy` | Surface-current self and mutual magnetic energy, J |
| `minimax` | Maximum facewise vector current density magnitude, A/m |
| `mixture` | Weighted sum of the above divided by fixed physical reference scales |

`objective_weights` and `objective_scales` use keys `power`, `magnetic_energy`
and `minimax`. Scales must be positive, in W, J and A/m respectively. Missing
scales come from a feasible minimum-power reference at
`reference_tolerance_scale` (default 1). They stay fixed for the life of the
compiled model. Zero reference quantities require explicit scales. Smoothness
and spacing penalties must be zero in this physical objective path.

`evaluate_energy=True` evaluates energy even when the objective does not need
it. `sheet_power_limit_W` and `magnetic_energy_limit_J` impose convex upper
bounds independently of the selected objective. These are sheet-model limits;
they do not automatically certify routed-wire loss or circuit inductance.

Energy uses the current-current surface integral, including self and cross-part
terms. An analytic source-triangle potential and converged target quadrature
avoid an artificial self radius. `EnergyQuadrature` exposes convergence and PSD
tolerances and a face-count resource guard. Nonconvergence/significant negative
energy eigenvalues are errors. Roundoff correction and quadrature changes are
reported. The quadrature estimate does not establish mesh convergence.

This is a dense reference implementation, with approximately `O(F² q²)` work.
The default `maximum_faces=768` guard runs before field-operator assembly.
Energy requests are never silently disabled on fine meshes; increasing the
guard is an explicit resource choice, not a faster algorithm. The physical
formulation is consistent with the surface-current framework described by
[Mäkinen et al.](https://arxiv.org/abs/2005.10060).

## Field, force and torque constraints

`LinearFieldConstraint(name, matrix, target, tolerance, norm, unit)` acts on
the full nodal stream function, before reduction. Supported norms are
`equality`, `upper` (componentwise halfspace), `l2`, `rms`, and `linf`.
For `group_size=3`, consecutive rows form Cartesian vectors and magnitude
norms apply. The caller supplies physically valid coupling matrices in the
declared units. Existing default field/suppression/gradient/current-density/
contour-current/shield-sign constraints remain active unless explicitly replaced.

Each `FieldScenario` supplies another field operator/target and RMS or peak
budget. Feasibility covers those finite scenarios only. It is not a guarantee
over an unspecified continuous uncertainty set.

Loads use `integral K x B0 dS` and
`integral (r-origin) x (K x B0) dS`. Set force/torque norm limits or exact
vector equalities with `force_limit_N`, `torque_limit_Nm`, `force_equality_N`,
and `torque_equality_Nm`. `background_B_per_face_T` allows a prescribed
piecewise-constant nonuniform background. All vectors share the mesh world
frame. `b0_T` is a physical field vector; the legacy `b_0_direction` selects
field-component conventions independently and should be configured deliberately.

A complete closed current distribution has zero total force in uniform B0;
torque may remain. Net balance is not a local stress, vibration or transient
mechanical model. RF fields, loading/SAR, skin/proximity effects and eddy-current
dynamics are outside this magnetostatic solver.

CLARABEL, ECOS and SCS are capability-checked conic backends. Explicit solver
selection is honored. Every bound is recomputed in physical units, with
separate absolute allowances by unit and relative acceptance tolerance.
An optimal status alone does not establish acceptance; no scalar scan fallback
is substituted for the physical constraints.

## Repeated solves and durable studies

```python
from coilgen.research import ParameterizedResearchProblem

compiled = ParameterizedResearchProblem(problem, spec, config)
solutions = compiled.solve_many([0.5, 1.0, 1.5])
```

Only residual budgets change with the tolerance scale. Mechanical/current
limits, power/energy caps, equalities and halfspaces stay fixed. The model
reuses one CVXPY graph and warm starts, following
[CVXPY's DPP contract](https://www.cvxpy.org/tutorial/dpp/index.html).

`ToleranceStudy(directory, problem, spec, config).run(scales)` adds durable
results, checksummed operators/artifacts and restart support. Its acceptance
scope is the continuous sheet on the supplied sampled operators. It does not
claim independent-grid or fabrication validation.

For independent designs, use `ResearchRequest.from_config(config)`,
`run_design(request, output_dir)`, or `ResearchStudy(directory, policy).run(requests)`.
Independent mixture studies require explicit objective scales; use the same
scales to compare a common tradeoff. `RuntimePolicy` bounds processes, native
threads, pending jobs and execution time. Windows workers use spawn, so protect
script entrypoints with `if __name__ == "__main__":`. Process-local mesh/target
registry handles cannot cross worker boundaries; use files for those inputs.

Cancellation works with console progress disabled. OS directory locks prevent
another process from rewriting live job state. Resume requires matching code,
resolved configuration, inputs, environment and artifact checksums. Failed,
cancelled and unavailable results remain visible. Pareto selection admits only
accepted finite results within one physical validation scope.

## Material and fabrication interpretation

`WireSheetMaterial.round_wire(diameter_m, pitch_m).apply_to(config)` explicitly
sets the sheet material approximation. For wire area A and uniform track pitch
d, effective thickness is `A/d` and sheet resistance is `rho*d/A`. This matches
DC power along locally parallel tracks in one layer; it does not model variable
pitch, leads or temperature. The historical config field
`specific_conductivity_conductor` contains **resistivity**, in ohm metres.
Final wire loss remains `I² rho length/A` on the actual conductor geometry.

`compile_fabrication(result, ...)` retains the mounted centreline, frames and
cross-section as `FabricationPath`, evaluates the actual conductor field at the
original current, and checks local bends plus all-body clearance. It defaults
to mesh realization. Explicit `surface_realization="analytic"` allows idealized
support projection, with displacement and field-change diagnostics. The legacy
pipeline retains `sweep_surface_realization="analytic"`; select `"mesh"` to
avoid silently replacing finite-element support geometry with an ideal cylinder.

Bend radius is checked again after final mounting/repair. A failed radius or
clearance blocks strict fabrication export; choosing a radius is not proof that
the constructor achieved it. Winding guides contain grouped operations and
arclength checkpoints while retaining full sampled paths and provenance.

`validate_on_independent_grid` checks new field samples without current refit.
`validate_fabricated_constraints` uses realized electrical paths and explicit
returns. Missing wire couplings, unmodeled sheet-to-wire caps or incomplete
circuits remain unverified. Circuit self inductance is not a full multi-circuit
mutual-inductance matrix. Feed breaks, supply routing and manufacturing release
still need explicit engineering design.
