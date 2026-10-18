# Quality and Extension API

This page summarizes the small APIs intended for physicists, MR scientists,
coil engineers, and application developers who need robust magnetostatic coil
studies without editing the pipeline internals.

## User Run Helper

Use `run_coil_design` when a script should run one design, save standard reports,
and optionally open one interactive plot book. `run_scientist_design` remains as
a compatibility wrapper for older scripts.

```python
from pathlib import Path
from coilgen import CoilDesignRunOptions, CylinderCoilDesign, run_coil_design

design = CylinderCoilDesign(target_bz_expression="y", radius_m=0.25, length_m=0.5)
artifacts = run_coil_design(
    design,
    CoilDesignRunOptions(
        project_root=Path(__file__).resolve().parents[1],
        output_prefix="cylinder_y_gradient",
        open_plot_book=True,
        plot_include=("wire_path", "field_pointclouds", "gradient_maps", "regularization"),
    ),
)

result = artifacts.result
quality = artifacts.quality
```

The returned `CoilDesignRunArtifacts` contains:

- `result`: the full `CoilGenResult`;
- `quality`: PASS/WARNING/FAIL plus warnings;
- `study_result`: typed scalar metrics for parameter sweeps;
- `files`: saved report paths;
- `figures`: active plot book objects.

## Regularization Diagnostics

Use `regularization_selection_report(result)` to inspect the L-curve choice.
It reports the selected factor, selected index, log dynamic range, selected
position on the normalized residual/resistance axes, a `knee_type`, and
warnings.

Important `knee_type` values:

- `first_tradeoff_knee`: preferred onset of the useful regularization tradeoff;
- `tradeoff_knee`: stable interior L-curve knee;
- `overregularized_collapse`: late current-norm collapse arm;
- `underregularized_plateau`: residual plateau before useful regularization;
- `fallback`: no reliable L-curve knee was available.

The design report includes these values automatically.

## Validation Reports

`write_design_report(result, path)` and `write_design_summary_csv(result, path)`
are the standard report writers. They separate DSV target metrics from
suppression-region fringe-field metrics.

`result_quality_status(result)` classifies a design with route, manufacturing,
regularization, contour-stability, and field-error warnings.

## Parameter Studies

Use the typed study objects from `coilgen.study`:

```python
from coilgen import DesignStudyTable, study_result_from_coilgen_result

rows = []
for gap in gaps:
    result = run_one_design(gap)
    rows.append(study_result_from_coilgen_result(f"gap={gap:.3g}", {"gap_m": gap}, result))

table = DesignStudyTable.from_results(rows)
best = table.best_by("layout_gradient_efficiency_mT_per_m_per_A", maximize=True)
```

These objects avoid fragile dictionary conventions in sweep plotting and CSV
export.

`best_by` excludes `FAIL` and `UNKNOWN` rows by default. Use
`allowed_statuses=("PASS",)` for strict screening; `allowed_statuses=None`
includes every finite candidate for diagnostic comparisons. Inspect warnings
before selecting a `WARNING` candidate; this ranking does not establish
manufacturing or hardware feasibility.

`ParameterSweepWorkflow` resumes successful case rows only, retries failed or
invalid cache entries, and preserves parameter order in tables even when some
cases are cached and others run again. Failed cases remain visible in the
tables. The case factory is called on resume to fingerprint the resolved
design/configuration, standard mesh/target/preoptimized input contents, and
core Python source. Old caches are consequently recomputed once.

Set `ParameterSweepSpec.cache_revision` when changing custom factory or metric
dependencies, external solver versions, or data not referenced by the standard
config inputs. Cache reuse restores scalar rows, not full result objects or
missing exported files; rerun with `resume_from_case_cache=False` to regenerate
artifacts. Compare finalists using the same target samples, operating current,
field-error constraints, and manufacturing checks.

## General Extension Priorities

CoilGen should stay application-neutral at the core. Conventional MRI gradients,
Halbach-system gradients, shim coils, local nonlinear gradients, active
shielding, arbitrary imported target maps, and project-specific studies should
all use the same public extension points where possible.

Near-term extension work should prefer these API shapes:

- vector or `B dot b0_hat` target-field components in addition to the current
  convenient `target_bz_expression` path;
- named target and diagnostic sample groups, including volumes, surfaces, and
  non-optimized diagnostic regions;
- optimization objectives registered through
  `available_stream_function_optimization_schemes()`, especially
  peak-current-density or minimax-current-density objectives;
- general MR-gradient reports such as concomitant-field estimates, without
  adding project-specific metrics to core CoilGen;
- report fields that separate static magnetostatic feasibility from downstream
  thermal, voltage/slew, PNS, acoustic, force, eddy-current, and terminal-route
  checks;
- exportable field, vector-potential, resistance, and inductance data for
  downstream MR-gradient waveform and safety tools.

## Geometry Factories

Generated mesh types are registered through `GeometryFactory`. This allows new
geometry APIs to be added without extending long example scripts.

```python
from coilgen import GeometryFactory, Mesh, register_geometry_factory

def build_my_surface(config):
    params = config.my_surface_parameter_list
    return Mesh(vertices=..., faces=...)

register_geometry_factory(
    GeometryFactory(
        name="create my surface mesh",
        description="Short user-facing description.",
        parameter_attribute="my_surface_parameter_list",
        builder=build_my_surface,
    )
)
```

Then set `coil_mesh_file = "create my surface mesh"` in the config. Factory
builders must return SI-unit vertices and triangular faces.

## Optimization Schemes

Use `available_stream_function_optimization_schemes()` to inspect the currently
implemented stream-function solvers and cost functions. New optimization
schemes should be added as explicit entries before they are wired into the
solver path, so reports can state exactly which mathematical objective was used.

## Plot Book

The plot book keeps one live Matplotlib canvas per page when Qt is available.
Users can rotate and zoom 3D pages with the normal Matplotlib toolbar.

Current features:

- compact right-side page selector;
- category filter;
- scrollable performance table page;
- export selected page or all pages as PNG files.

New plot types should normally be added through `build_result_plot_specs` in
`matplotlib_visualization.py`, not by subclassing the plot book.
