# Stream-function optimization: capabilities and engineering choices

Audit date: 2026-09-08. The core has a useful convex optimization toolkit, but it
does not yet implement every important physical objective. Backend count is not
a measure of mathematical coverage: first specify the physical problem, then
select a compatible numerical solver.

Main integration update (2026-09-09): physical energy/power/current-density
objectives, generalized field bounds, finite scenarios, force/torque constraints,
power/energy caps and reusable DPP tolerance studies are now available through
the opt-in [headless physical API](RESEARCH_PHYSICS_API.md). The numerical
formulations and original audit evidence below remain useful; the numbered
gaps section records the audit's original recommendations rather than current
implementation status. See [main integration](MAIN_CORE_IMPROVEMENTS.md) for
current limits and validation.

## Physical model

Let `psi = E x` be the nodal stream function, `A x` the sampled magnetic field,
and `K_f x` the vector surface current density on triangle `f`. Nodal stream
functions have units A; surface current density has units A/m. `E` incorporates
boundary conditions, gauge removal, and any symmetry or spectral reduction.

The existing FEM resistance matrix represents a continuous conducting sheet:

```text
P_sheet = x^T R x                         [W]
R = (resistivity / sheet_thickness) integral K_i dot K_j dS
```

This is not `||R x||^2`, not magnetic energy, and not the resistance of a routed
fixed-width wire. Minimum sheet power at a prescribed field does not directly
maximize field per wire ampere. `ConductorSpec.effective_sheet_thickness_m`
controls this sheet approximation independently of physical wire dimensions;
if omitted it preserves the configured thickness (default 5 mm). The final
wire uses its own conductor cross section for resistance.

Different boundary loops retain separate constant stream-function values.
This enforces no outward surface-current flux while preserving loop-to-loop
circulation. Only the constant gauge on each connected component is removed.
Setting every boundary loop to zero would discard valid currents. Single-valued
nodal stream functions alone are incomplete on closed surfaces of positive
genus: harmonic circulation modes need an explicit extension.

## What is implemented

| Formulation or control | Current coverage and interpretation |
|---|---|
| Tikhonov | Weighted least squares plus `lambda^2 x^T R x`; optional surface smoothness and adaptive spacing penalties. Fixed or selected regularization; normal-equation and transformed-SVD paths are available. |
| Fixed-error scalar search | Searches the Tikhonov family. Equivalent to constrained minimum power only for the matching single L2 residual model without extra objective penalties. Multiple independent budgets generally need independent multipliers. |
| Direct minimum quadratic cost | `min x^T Q x`, subject to separately modeled field and engineering bounds. `Q` includes sheet resistance and configured quadratic penalties. Non-PSD objectives are rejected beyond numerical roundoff. |
| Smoothness priority | Weights the surface smoothness cost with a positive resistance tie-break. This is a weighted objective, not an exact lexicographic solve or an energy objective. |
| **Minimax current density** | `min t`, subject to `||K_f x||_2 <= t` on every face and the same hard field/engineering bounds. Reports the independently evaluated physical peak in A/m. |
| Useful field | Bx, By, Bz, B projected on the B0 direction, and vector targets. Separate DSV norm, RMS, or mean-free STD budgets. STD permits a constant residual offset; use it only when that offset is acceptable. |
| Suppression | RMS and peak limits, Cartesian/vector/surface-normal components, and area weighting where configured. |
| Gradient constraints | Strength, cross-gradient, RMS/peak linearity, and optional shield-gradient opposition. |
| Engineering bounds | Facewise vector-current magnitude and stream-function range/contour-current limits. These remain continuous-sheet bounds. |
| Basis choice | Nodal basis, validated symmetry reduction, or mass-normalized cotangent spectral basis with a physical wavelength cutoff. A spectral cutoff changes the admissible current space. |
| Joint/cohort shimming | Bounded actuator/current fits, field-first/resistance-second joint optimization, resistance-weighted POD and population-average refinement. These have their own APIs. |
| MRI shim metric | Convex voxel-tail CVaR, residual RMS and spatial-gradient terms in the dedicated shim metric path. Voxel CVaR is not a worst-scenario uncertainty guarantee. |

These are magnetostatic formulations. RF conductor-current phase, impedance,
sample loading, electric field and SAR require an RF-specific electromagnetic
model; a magnetostatic stream-function optimum does not establish RF performance.

## Recommended use

For a reproducible comparison, specify absolute physical field limits, separate
useful-field and suppression regions, and select a direct objective explicitly.
Use a tolerance sweep to obtain a tradeoff curve rather than presenting one
regularization parameter as universally best.

```python
from coilgen import OptimizationSpec, conic_solver_capabilities

optimization = OptimizationSpec(
    method="constrained_quadratic",
    constrained_objective_mode="minimax_current_density",
    conic_solver="auto",
    conic_tolerance=1e-8,
    conic_fallback_policy="strict",
    target_error_mode="separate",
    target_error_metric="rms",
    target_error_rms_limit_T=1e-6,  # example design budget, not a universal setting
    target_error_constraint_acceptance_absolute_T=0.0,
)

for backend in conic_solver_capabilities():
    print(backend.name, backend.method, backend.installed)
```

Attach this `OptimizationSpec` to a `CoilDesignProblem`. New optional typed
controls default to `None`, preserving advanced `base_config` values. Metadata
overrides and explicit configuration overrides retain their existing priority.
For low-level configurations use `sf_opt_method="constrained_quadratic"` and
`constrained_quadratic_objective_mode="minimax_current_density"`.

Pure minimax requires zero explicit smoothness and spacing penalty factors; it
replaces the implicit engineering spacing penalty. Physical hard bounds remain
active. To trade peak current against sheet power, sweep a hard current-density
cap while minimizing power (`balanced`, zero extra penalties). This samples the
convex tradeoff using existing controls without an arbitrary mixed-unit weight.
Pure minimax can increase power and inductance, so compare the resulting coils.

## Solver and acceptance policy

The supported general conic backends are Clarabel and ECOS (interior-point
methods), and SCS (operator splitting). Automatic selection tries installed
backends in that order. The environment used for this audit had Clarabel and
SCS installed; ECOS availability must be checked on the deployment environment.
NumPy linear algebra already uses native compiled routines.

OSQP is appropriate for quadratic programs with linear constraints, but cannot
solve this general formulation with Euclidean residual/current cones. It is
therefore rejected by this adapter. MOSEK is a valid conic candidate for a future
optional adapter; this audit did not add an untested licensed integration.
Adding unrelated nonlinear or evolutionary solvers would not improve the convex
fixed-support problem's guarantees. Outer geometry/topology optimization is a
different problem and can remain nonconvex.

Solver `optimal` or `optimal_inaccurate` status is followed by independent checks
of residual blocks, every requested density bound, stream-function range and
shield opposition. Requested constraints with missing data fail the contract.
Nonfinite values cannot pass. Diagnostics include physical values, requested and
effective limits, units, availability, and failed constraints. Post-solve current
scaling triggers another check. Feasibility checks do not prove exact numerical
optimality or certify the continuum problem between sample points.

The existing absolute numerical field acceptance allowance defaults to `1e-7 T`.
It is separate from the requested error budget and may be too large for shim
work. Explicitly set it to zero or to a justified smaller value; inspect effective
accepted limits. The dimensionless relative allowance is derived from solver
tolerance unless `target_error_constraint_acceptance_tolerance > 0`. Hardware
checks use their own physical units, not a tesla-valued allowance.
Fitted gradient-strength and cross-gradient residuals are in T/m and therefore
also use relative acceptance, without adding the field-valued absolute slack.

An optional Tikhonov scan fallback is explicitly diagnostic and does not certify
the full direct constrained optimum. It cannot silently replace minimax or
requested hardware bounds. Use `strict` for studies requiring the direct model.
Independent validation targets, mesh refinement, contouring/routing reevaluation
and physical winding clearances remain necessary after a continuous solve.

## Original audit recommendations (current status linked above)

1. **Surface magnetic energy.** Assemble a symmetric PSD energy matrix including
   self terms and cross-part coupling, then expose `W = 0.5 x^T L x` and normalized
   power/energy mixtures. Validate quadrature and mesh convergence before enabling
   it. Existing post-layout inductance is not this matrix.
2. **General field-bound specification.** Expose reusable constraint blocks for
   componentwise DSV absolute/relative bounds, pointwise vector error and linear
   equalities. Internal residual blocks support more than the current DSV API
   exposes. Near zero target values, absolute bounds are essential.
3. **Scenario robustness.** Add shared-design constraints for every displacement,
   manufacturing or field-map scenario, or a worst-scenario epigraph. Make the
   distinction between fixed and adjustable scenario currents explicit. Existing
   average-case cohort refinement does not provide these guarantees.
4. **Force/torque and eddy-current objectives.** Fixed-background force/torque can
   become linear stream-function constraints; dynamic shielding requires validated
   conducting-environment response operators. These are not interchangeable with
   a static suppression target.
5. **Topology completeness.** Detect unsupported positive-genus circulation and
   add harmonic current modes where such supports are needed. Preserve the
   existing independent boundary constants.
6. **Repeated-solve performance.** Reuse parameterized conic models and warm starts
   across fixed-geometry tolerance/current-density sweeps. Benchmark memory,
   sparse assembly and repeated canonicalization before adding native extensions.
7. **Build-aware selection.** Rerank candidates using routed/swept field, discrete
   resistance, clearance, terminals and winding sequence. Locally
   `spacing ~ contour_current / |K|`; this does not certify connectors, folds,
   cross-layer clearance or insulation.

## Validation added in this audit

Final regression run: **735 passed, 1 skipped, 26 subtests passed**. The skipped
test requires unavailable `h5py`; 10 existing plotting deprecation warnings
remain. Repository Ruff checks pass.

- Analytic two-DOF problems distinguish minimum power from minimax and are solved
  with Clarabel and SCS, using both dense and local face operators.
- Rotation invariance, feasible/infeasible density caps and complete operator
  coverage are checked independently.
- Very large uniform and unequal basis rescalings preserve the physical minimax
  solution; variables are normalized by physical current gains and field drive.
- Regressions cover falsely reported optimal solutions, unavailable constraints,
  nonfinite peak residuals, unsupported fallbacks, and post-solve scaling.
- API tests check explicit zeros, inheritance and override precedence, insulation
  mapping, and separate sheet thickness; solver discovery is tested without
  requiring optional packages to be present.

### Generated-cylinder comparison

The reproducible [comparison example](../examples/optimization_minimax_comparison.py)
uses a 20 mT/m transverse gradient, a 60 mm spherical ROI, 544 mesh vertices,
1024 faces, 481 unknowns and 81 fitting points. Clarabel solved both objectives
at the same 3% fitted relative L2 field-error budget, with no post-solve scaling.

| Measured quantity | Minimum sheet power | Minimax current density |
|---|---:|---:|
| Peak surface current density | 51,971.74 A/m | 28,304.78 A/m |
| Continuous-sheet power | 377.79 W | 616.68 W |
| Fitting-grid relative field error | 3.000% | 3.000% |
| Independent 912-point relative field error | 2.233% | 2.349% |
| Independent peak absolute field error | 49.99 µT | 48.35 µT |
| Extracted contour components at the same level count | 20 | 36 |

In this case minimax reduces peak current density by 45.54% and increases sheet
power by 63.23%. Triangle-gradient current reconstruction agrees with the FEM
operators within `1.84e-11 A/m`; independent area integration of `rho/t |K|^2`
agrees with the matrix power. These are sampled continuous-sheet results, not
mesh-convergence, routed-wire, insulation-clearance or RF validations. This
comparison deliberately skips routing, sweeping and wire inductance.
The larger contour-component count also shows why lower peak current alone
does not guarantee a simpler winding. Compare power and contour topology across
current-density caps and validate the connected conductor before choosing a build.

Explicit diagnostic-only results and false solver contracts now receive `FAIL`
in reporting and are excluded from study `best_by` selection by default. Legacy
results without these flags preserve their existing status. Diagnostic
inspection can explicitly request `allowed_statuses=None`.

## Primary references

- [Poole et al., Minimax Current Density Coil Design](https://arxiv.org/pdf/1002.3413):
  established peak-current design objective and its physical tradeoffs. The
  epigraph SOC formulation above is our implementation of that convex objective.
- [bfieldtools physical principles](https://arxiv.org/pdf/2005.10060) and
  [optimization API](https://bfieldtools.github.io/generated/bfieldtools.coil_optimize.optimize_streamfunctions.html):
  surface-current power/energy models and field-constrained optimization.
- [CVXPY solver capabilities and warm starts](https://www.cvxpy.org/tutorial/solvers/index.html):
  formulation compatibility and repeated-solve behavior.
- [Open FUSION Toolkit ThinCurr topology](https://openfusiontoolkit.github.io/OpenFUSIONToolkit/docs/v1.0.0-beta6/doc_tw_main.html):
  circulation degrees of freedom on surfaces with holes/handles.
- [While et al., Accounting for discretization into coil windings](https://www.sciencedirect.com/science/article/pii/S1090780713001961):
  the distinction between continuous and fabricated-conductor optimization.
