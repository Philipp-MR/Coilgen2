# Architecture and compatibility

CoilGen separates numerical calculations from project/session management and
from desktop/browser interaction. The engine accepts explicit configuration
and data; it does not import either workbench.

## Starting API

Most scripts only need these entry points:

```python
from coilgen import CoilGenConfig, run_coilgen
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage
```

Use `run_coilgen(config)` for a complete calculation and `CoilGenPipeline` for
staged execution and checkpoint reuse. See [Getting Started](GETTING_STARTED.md)
for design objects and [Headless physical design](RESEARCH_PHYSICS_API.md) for
research requests, constrained objectives and result artifacts.

The historical package and `coilgen.api` exports remain available with their
existing object identities. They resolve on first access. Importing `coilgen`
alone does not import NumPy, SciPy, CVXPY, Qt or visualization runtimes. A wildcard
import still requests every exported object and therefore loads its dependencies.
New implementation code should import from the owning module rather than expand
the package-wide compatibility surface.

## Numerical ownership

| Responsibility | Owning module |
| --- | --- |
| Physical requirement operators, force/torque constraints and evaluation | `coilgen.research_requirements` |
| Measured field datasets, coordinate transforms and unit conversion | `coilgen.field_datasets` |
| Optimization assembly, reduced operators and objective terms | `coilgen.optimization_problem` |
| Residual constraints and acceptance diagnostics | `coilgen.optimization_constraints` |
| CVXPY/conic solver execution | `coilgen.optimization_cvxpy` |
| Optimization orchestration and regularization searches | `coilgen.optimization` |
| Intrinsic contour inclusion and grouping | `coilgen.loop_topology` |
| Segment distance, contact queries and spatial acceleration | `coilgen.wire_distance` |
| Route/path construction and interconnection | `coilgen.layout` |
| Physical path realization, frames and swept geometry | `coilgen.manufacturing` |
| Swept-mesh intersection inspection | `coilgen.mesh_inspection` |
| Manufacturing export and export admission | `coilgen.manufacturing_export` |
| Physical frame diagnostics shared with review clients | `coilgen.manufacturing_witness` |

Moved numerical functions remain ordinary functions. Their old module imports
are re-exported for compatibility; implementations and internal test patches
belong in the owning module. `application.research_requirements` and measured
field classes in `application.gradient_studies` likewise retain compatibility
imports without owning numerical implementations.

`persistence.numerical_source_digest` hashes numerical source names and contents,
including new engine modules by default. Application, presentation and job
adapters are excluded. Every checkpoint stage incorporates this identity;
full-source provenance remains a separate record. Moving numerical code changes
that identity, so checkpoints created before this refactor must be recomputed.
Do not exclude a module that constructs operators or changes numerical results.

## Configuration and sessions

`CoilGenConfig` remains the authoritative set of solver input fields and defaults,
including historical MATLAB-compatible names. Existing typed settings and design
objects are views/converters over those inputs; their override precedence is
unchanged. `config_semantics` owns optimization aliases, accepted spelling sets and
shared physical unit factors. Normalization preserves unknown draft values;
validation and solver entry points still reject unsupported values explicitly.
Coupled physical validation remains in `config_validation` and the numerical
checks, rather than being reduced to independent form-field rules.

`application.settings` owns shared field definitions, display units and conversion
accessors. `application.optimization_summary` owns framework-independent
optimization presentation. Desktop and browser adapters consume these shared
definitions. Specialized native widgets remain in `gui.settings_pages`;
`CoilGenWorkbench._node_changed` handles page selection and native coordination.

`application.session_model.SessionModel` owns the active assembly, channel and
project together. Both clients use its transitions and `ModelEdit` before/after
snapshots. Browser history uses `ModelHistory`; the desktop wraps the same records
in Qt undo commands to retain native signals and clean-state behavior. Widget
notifications, dialog choices and draft interaction remain adapter responsibilities.
The common single-design submission helper preserves revision identity and shared
job registration/reuse behavior while each client retains its display state.

## Shared helpers and verification

`field_source.shim_source_key` provides the field-source selection used by both
plotting implementations. Test builders and reusable fixtures live under
`tests/_coilgen_test_support`; collected `test_*.py` modules do not import one
another. Keep assertions in test modules and reusable setup in support modules.

The architecture regression tests check numerical checkpoint identity, import
boundaries and laziness, compatibility identities, test-module independence, and
desktop/browser revision and undo parity. Run the full suite with the repository's
CI capture mode, which also avoids Windows native-library issues with descriptor
capture:

```text
python -m pytest --capture=tee-sys
```

Numerical refactoring should additionally replay representative saved inputs or
benchmark cases and compare stream functions, fields, currents, contour geometry,
and physical paths. Differences in source identities and timings are expected;
changes to numerical outputs require an explicit engineering justification.
