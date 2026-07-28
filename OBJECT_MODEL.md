# CoilGen Object Model

This note describes the high-level classes intended for future user-facing APIs
for physicists, MR scientists, coil engineers, and application developers. They
are additive and do not replace the existing `CoilGenConfig` pipeline yet.

## Main Classes

- `TargetFieldSpec`: DSV target-field samples plus explicit suppression regions.
  It converts to the internal `TargetField` model while preserving DSV and
  suppression masks.
- `SuppressionRegion`: fringe-field or magnet-interface sample points where the
  target field is usually zero. These points are weighted separately from DSV
  points and are evaluated with shielding metrics, not gradient-linearity
  metrics.
- `SupportMesh`: a mesh with a role: `primary`, `shield`, `suppression`, or
  `auxiliary`.
- `PipelineGeometrySpec`: an explicit mapping to one of the currently supported
  pipeline geometry generators, such as cylinder, biplanar, circular biplanar,
  shielded biplanar, flanged split bore, shielded flanged split bore, or
  imported STL mesh.
- `CoilSupportSpec`: the full support body, including primary and active-shield
  meshes. It can validate that active shields share topology with their paired
  primary surfaces.
- `ConductorSpec`: conductor dimensions, resistivity, insulation, and minimum
  turn spacing in SI units.
- `OptimizationSpec`: stream-function optimization choices such as Tikhonov,
  fixed-target-error optimization, regularization choice, turn spacing, and
  symmetry reduction.
- `LayoutSpec`: contour count and routing/postprocessing choices.
- `EvaluationSpec`: which field, resistance, inductance, and layout metrics are
  evaluated.
- `CoilDesignProblem`: the complete high-level design problem assembled from the
  objects above.
- `CoilChannelSpec` and `CoilProject`: user-facing project containers for
  multi-channel systems. A project owns shared design defaults and structured
  `project.parameters`, while each channel supplies only the target expression
  or other overrides that differ from the template.
- `CoilProjectParameters`: nested editable project defaults with sections such
  as `mesh`, `target`, `optimization`, `conductor`, `system`, and `numerics`.
- `CoilGenConfigDesign`: adapter that wraps an explicit `CoilGenConfig` so
  expert config-based scripts can still run as project channels.
- `CoilProjectRun` and `CoilChannelRun`: ordered run artifacts for a project,
  including per-channel results, summary rows, files, failures, and plot books.
- `MetricDefinition` and `MetricRegistry`: reusable metric labels, units, and
  interpretation metadata for reports, plot books, and paper-study figures.
- `ParameterAxis` and `ParameterStudySpec`: explicit parameter-sweep definitions
  that generate named `DesignStudyCase` objects before execution.
- `CoilDesignSolver`: protocol for solvers that consume a `CoilDesignProblem`.
- `ContourLine.turn_multiplicity`: positive integer ampere-turn weight of one
  nominal contour geometry.
- `ContourSparsificationResult`: original contours, merge provenance,
  final source-contour clusters, iterative-pass diagnostics, and any later
  physical turn realization for a sparsified part.

## Design Rule

Keep geometry, target fields, optimization settings, layout settings, and
evaluation metrics separate. This makes parameter studies cleaner: one can vary
the support body geometry while keeping DSV, active-shielding target points,
conductor, and optimization criterion fixed.

## Current Integration Status

The classes are public API and can be imported from `coilgen`. The production
pipeline still runs through `CoilGenConfig` and existing `CoilDesign` helper
classes.

`CoilDesignProblem.to_config()` and `coil_design_problem_to_config()` explicitly
convert high-level problem objects into the current `CoilGenConfig` input. Meshes
and target fields are registered in memory and referenced by same-process keys,
so conversion does not create temporary STL or CSV files. This bridge supports
all geometries that can be represented as `SupportMesh` objects.

If `CoilSupportSpec.pipeline_geometry` is set, conversion preserves the native
generated geometry and writes the current pipeline parameters directly, for
example `coil_mesh_file="create cylinder mesh"` with
`cylinder_mesh_parameter_list`. If no `PipelineGeometrySpec` is set, conversion
falls back to registering the provided `SupportMesh` objects as an in-memory
combined mesh. This gives both modes:

- generated/supported geometry: transparent, reproducible `CoilGenConfig`
  parameters;
- arbitrary mesh geometry: direct same-process execution without temporary mesh
  files.

`CoilDesignProblem.run()` and `PipelineCoilDesignSolver` use the same conversion
bridge before calling the current pipeline.

`CoilProject.solve()` is a higher-level orchestration layer. It resolves each
enabled `CoilChannelSpec` into an ordinary executable design object, runs the
existing single-design pipeline, writes project-level CSV/JSON/Markdown
summaries, and can open one combined project plot book. The executable design
can be a user-facing `CoilDesign` dataclass or a `CoilGenConfigDesign` wrapper
around a fully specified `CoilGenConfig`. This layer does not alter the
stream-function solver or replace `CoilDesignProblem`.

Project/channel parameter precedence is:

```text
template design -> project.parameters -> project design_overrides -> channel design_overrides -> channel target shortcuts
```

This makes shared settings explicit while preserving channel-specific override
behavior.

Example:

```python
from coilgen import CoilProject, CylinderCoilDesign

project = CoilProject("three_axis", template=CylinderCoilDesign())
project.parameters.mesh.mesh_size = 24
project.parameters.mesh.radius_m = 0.26
project.parameters.target.scale = 0.010
project.parameters.optimization.contour_levels = 8
project.parameters.system.main_field_strength_T = 1.5
project.parameters.system.b0_direction = (0.0, 0.0, 1.0)

project.channel("GX", target_bz_expression="x")
project.channel("GY", target_bz_expression="y", contour_levels=10)
project.channel("GZ", target_bz_expression="z")

project_run = project.solve()
figures = project.plot()
```

Here `mesh_size`, `radius_m`, `target.scale`, and the default contour count are
shared by all channels. `GY` still gets its channel-specific contour level.

## Plotting Classes

Plotting stays separate from the design-problem classes. A design problem should
define physics and geometry; plotting classes should consume a finished
`CoilGenResult`, study table, or external Matplotlib figures.

Use these extension points:

- `MatplotlibPlotSpec`: one lazy figure factory for result plots.
- `PlotBookEntry` and `PlotBookPage`: one live or rasterized page in a plot book.
- `MatplotlibPlotBookBuilder`: collects plot specs or existing figures into one
  interactive plot-book window.
- `QtMatplotlibPlotBook`: preferred live backend with normal Matplotlib
  rotation/zoom interaction.
- `ImageMatplotlibPlotBook`: raster fallback when Qt embedding is unavailable.

New scientific plot types should normally be added as figure factories and then
registered with `MatplotlibPlotBookBuilder`, not by subclassing the plot book.
Subclass the plot book only when changing the navigation/container behavior.

Project plot books use the same builder. They add project summary/comparison
pages first, then prefix the normal result plot pages with the channel label
such as `GX`, `GY`, or `GZ`.
