# Parameters

The opt-in `CoilGenConfig.research_physics` (also available on
`OptimizationSpec`) selects physical objectives, mechanical limits and finite
scenarios. `sweep_surface_realization` selects `"analytic"` or `"mesh"` support
realization and is also exposed on `LayoutSpec`. See the
[headless physical API](RESEARCH_PHYSICS_API.md) for units and defaults.

## User-Facing Design Fields

The recommended API is based on dataclasses such as `CylinderCoilDesign`,
`PlanarCoilDesign`, `BiplanarCoilDesign`, and `ImportedMeshCoilDesign`.

Common fields:

| Field | Unit | Meaning |
|---|---:|---|
| `target_bz_expression` | - | Requested Bz shape, e.g. `"x"`, `"y"`, `"z"`, `"1"`, or an expression. |
| `target_scale` | T or T/m with `target_field_units="T"` | Requested field scale for `"1"`; requested gradient scale for `"x"`, `"y"`, `"z"`. This is target amplitude, not the resulting coil efficiency per ampere. Other expressions follow the configured target normalization. |
| `target_region_radius_m` | m | Radius of spherical target point cloud. |
| `target_region_resolution` | - | Target sampling resolution; explicit regions use the shape-dependent definition below. |
| `target_field_units` | T or Hz | Unit used by generated or imported target-field values; default is `T`. |
| `field_display_units` | T or Hz | Unit for field maps and report display; default is `T`. |
| `contour_levels` | - | Requested stream-function contour count. |
| `min_turn_spacing_m` | m | Minimum requested spacing between neighboring turns. |
| `conductor_width_m` | m | Rectangular conductor width. |
| `conductor_height_m` | m | Rectangular conductor height. |
| `conductor_resistivity_ohm_m` | Ohm m | Conductor resistivity; examples use copper, `0.018e-6`. |
| `save_wire_trace_csv` | - | Save unswept contour STEP wires and, when routed, the final centerline as XYZ CSV/STEP; default is `False`. |
| `optimization_method` | - | `"tikhonov"`, `"target_error_constrained_efficiency"`, or `"constrained_quadratic"`. |
| `regularization_method` | - | `"l_curve"` or `"fixed"`. |
| `regularization_factor` | - | Tikhonov scan center or fixed factor. |
| `target_error_constraint_mode` | - | `"separate"` constrains DSV accuracy and active-shield suppression independently; `"combined"` keeps legacy weighted behavior. |
| `target_error_tolerance_percent` | % | Allowed DSV Bz residual for fixed-error efficiency optimization. |
| `suppression_field_rms_limit_T` | T | Preferred physical RMS Bz limit on active-shield suppression points. |
| `suppression_error_tolerance_percent` | % | Legacy active-shield suppression residual, referenced to the DSV target norm when no RMS limit is set. |
| `cpu_workers` | - | `0` uses all logical cores; `1` is serial; `-1` keeps one core free. |

Expert-only parameters can be passed through `advanced_parameters`.

## Native generated support surfaces

The GUI exposes 13 existing mesh generators. Cylinder, Biplanar and Conical
frustum are direct tiles; **More surfaces…** opens the remaining catalog. The
native `CoilGenConfig.coil_mesh_file` source and its parameter-list field remain
available to scripts:

| GUI surface | Native `coil_mesh_file` source | Parameter-list field |
|---|---|---|
| Cylinder | `create cylinder mesh` | `cylinder_mesh_parameter_list` |
| Biplanar | `create bi-planary mesh` | `biplanar_mesh_parameter_list` |
| Conical frustum | `create conical frustum mesh` | `conical_frustum_mesh_parameter_list` |
| Rectangle | `create planary mesh` | `planar_mesh_parameter_list` |
| Disc | `create circular mesh` | `circular_mesh_parameter_list` |
| Annulus | `create annular disc mesh` | `annular_disc_mesh_parameter_list` |
| Partial cylinder | `create partial cylinder mesh` | `partial_cylinder_mesh_parameter_list` |
| Elliptical cylinder | `create elliptical cylinder mesh` | `elliptical_cylinder_mesh_parameter_list` |
| Superellipse | `create superellipse cylinder mesh` | `superellipse_cylinder_mesh_parameter_list` |
| Circular biplanar | `create circular bi-planary mesh` | `circular_biplanar_mesh_parameter_list` |
| Spherical cap | `create spherical cap mesh` | `spherical_cap_mesh_parameter_list` |
| Ellipsoidal cap | `create ellipsoidal cap mesh` | `ellipsoidal_cap_mesh_parameter_list` |
| Toroidal segment | `create toroidal segment mesh` | `toroidal_segment_mesh_parameter_list` |

Native generator lengths are metres and generator rotation/sector/polar angles
are radians. Division counts are integers. The GUI converts lengths to mm and
angles to degrees, preserves unedited native parameters and reports each shape's
local frame. Cylinder-like supports have local axis z. Rectangle height is along
local x and width along local y. Biplanar separation is the full plane gap.
Elliptical/cap radii are semi-axes; superellipse half extents and exponent define
its profile (`n=2` ellipse, `n=4` rounded rectangle). Annulus and paired supports
use their specified world normal and center; rotated primitives translate their
center after rotation. The legacy Cylinder has no translation parameters.

The legacy Cylinder builder drops its final axial row. Its GUI Length control
therefore displays `native_height × axial_divisions / (axial_divisions + 1)`;
editing Length or axial divisions preserves the requested physical span. Other
cylindrical generators use their full native axial length. No generator formula
or script parameter-vector meaning is changed.

GUI validation guards known degenerate meshes: annulus inner radius must be
positive (choose Disc otherwise), open cap polar angle is below 180°, and open
partial-cylinder/toroidal major span is below 360°. A full 360° toroidal tube seam
is welded. Disc grid divisions across the diameter are at least four. Circular
biplanar grid-step target maps to approximately `ceil(2 × radius / step)` diameter
divisions, bounded to 512 in the GUI; it is not a guaranteed triangle edge length.

## Explicit target-region geometry

`CoilGenConfig.target_region_geometry` is an optional JSON-compatible SI
dictionary. `None` preserves the existing spherical point cloud or legacy
target-mesh vertex/bounding-box sampling. An explicit definition separates the
analytic target function from the geometry on which it is sampled:

| `shape` | Required dimensions | Convention |
|---|---|---|
| `"sphere"` | `radius_m` | Radius in m. |
| `"ellipsoid"` | `semi_axes_m: [rx, ry, rz]` | Positive local semi-axes in m; GUI displays full diameters in mm. |
| `"box"` | `size_m: [lx, ly, lz]` | Positive full local side lengths in m. |
| `"cylinder"` | `radius_m`, `length_m` | Positive radius and full local z length in m; closed sampling region. |
| `"enclosing_mesh"` | `target_mesh_file` on the config | Imported mesh uses its saved SI world coordinates. |

All definitions accept `sampling: "volume"` (default) or `"surface"`. Analytic
shapes accept `center_m: [x, y, z]` (default origin), `rotation_axis: [x, y, z]`
(default z) and `rotation_angle_rad` (default zero). Rotation is
right-handed; a nonzero angle requires a nonzero axis. Local samples are rotated
about the local origin and then translated to the world center. This changes
the samples only: expressions retain world x/y/z, and harmonic center/normalization
settings retain their own meaning. Explicit placement ignores the legacy
`set_roi_into_mesh_center` option.

```python
from math import radians
from coilgen import CoilGenConfig

config = CoilGenConfig(
    target_region_geometry={
        "shape": "ellipsoid",
        "sampling": "volume",
        "semi_axes_m": [0.020, 0.025, 0.030],
        "center_m": [0.0, 0.0, 0.010],
        "rotation_axis": [0.0, 1.0, 0.0],
        "rotation_angle_rad": radians(20),
    },
    target_region_resolution=10,
)
```

For `"enclosing_mesh"`, volume mode requires a watertight, consistently oriented,
connected, nonzero-volume triangular shell. Both inward and outward global face
orientation are accepted. Samples are filtered by actual enclosure containment,
including concavity. Surface mode accepts open valid triangular meshes. Mesh
center and rotation must remain zero; transform its world coordinates explicitly
before import. Closure checks do not prove absence of self-intersections;
ambiguous sampled containment fails with a repair message.

Explicit `target_region_resolution` is an integer at least two. For volumes it
is the number of Cartesian bins per local bounding-box axis; filtering changes
the retained count. Analytic surfaces use proportional angular/face sampling.
Sphere/ellipsoid surfaces and imported surfaces use a global `6 × resolution²`
points. Imported surface points are distributed in proportion to triangle area
and lie in triangle interiors. Every retained target point has equal objective
weight; these samples are not volume/surface integration quadratures.
Independent field evaluation reuses the exact region with a different sampling
phase and the study's explicit `validation_resolution`. That requested resolution
is authoritative and is not silently increased to exceed the optimization grid.

An imported target field (`target_field_definition_file`) uses its own saved
coordinates and values. The explicit region does not resample that map. With
`target_region_geometry=None`, the previous target mesh behavior remains exact;
choosing an explicit enclosure is required for actual interior sampling.

With an active explicit analytic region, the spherical slice-map domain
`gradient_slice_map_domain="dsv"` and its ROI aliases are rejected by the
slice-map diagnostic. Field optimization and independent validation still use
the actual region samples. Set the diagnostic domain to `"all"` or `"full"` to
request a separate bounding box derived from saved target-sample coordinates;
that diagnostic does not claim shaped-region coverage. Imported maps and legacy
`target_region_geometry=None` retain their previous diagnostic behavior.

## Closed-series interconnection

Controls are available on `CoilGenConfig` or through `advanced_parameters`:

| Field | Default | Meaning |
|---|---:|---|
| `routing_circuit_scope` | `"single"` | Assemble grouped supports into one closed circuit at a common current. `"per_part"` retains independent closed circuits. Cross-support gaps still require physical validation. |
| `joint_group_routing` | `True` | Search directed group orders, spiral states, cut alternatives and return curves before independent validation. |
| `joint_group_routing_search_mode` | `"progressive"` | Cheap complete-circuit ranking followed by bounded representative curves; broaden only after no accepted improvement. `"thorough"` retains the wider eager transition/curve search. Both use identical physical and fixed-current field gates; neither proves global optimality or infeasibility. |
| `joint_group_routing_field_acceptance` | `"absolute"` | Explicit `"non_regression"` preserves passing limits and admits inherited failures only when they do not worsen. Explicit `"budget_only"` permits measured gradient/layout scalar failures within the hard delta budgets; suppression remains hard. Absolute, new and regressed inherited failures remain reported. |
| `joint_group_routing_clearance_m` | `None` | Positive route-only centerline target in metres. An explicit value replaces both spacing aliases in an isolated routing/physical-preview config; original contour optimization settings remain unchanged. Conductor-envelope, insulation, tolerance and existing manufacturing-margin floors remain hard. `None` preserves existing settings. |
| `joint_group_routing_max_candidates` | `6` | Complete physical-validation budget, shared by all cut and shape alternatives; range 1–64. |
| `joint_group_routing_max_field_change_rel` | `0.001` | Hard pointwise vector-field change allowance at unchanged current, relative to peak active DSV target magnitude. |
| `joint_group_routing_max_longitudinal_field_change_rel` | `None` | Optional additional hard pointwise change along normalized B0, using the same target/current normalization and allocation across eligible parts as the vector budget. Positive finite numeric value when enabled. |
| `joint_group_routing_search_cuts` | `True` | Include bounded alternative cut rails; explicit forced cuts remain authoritative. |
| `joint_group_routing_port_search_count` | `8` | Regular repair context limit, integer 0–32. Zero disables new arc-length/return-lane proposals while retaining historical high/low alternatives. Physical finalist budgets remain unchanged. |
| `joint_group_routing_cut_span_max_factor` | `4.0` | Bound proposed removed contour arc by 1–16 times nominal cut width. Actual spans are recorded; source contours/current and physical acceptance limits remain unchanged. |
| `joint_group_routing_seam_search_extent_rel` | `0.1` | Maximum directed port relocation as a fraction of original contour perimeter, 0–0.25. Deterministic bounded samples of continuous arc positions; explicit forced cuts remain exact. |
| `joint_group_routing_b0_weight` | `1.0` | Nonnegative alignment weight for the provisional geometric search guide. It does not replace the final field objective. |
| `routing_jump_height_tolerance_m` | `1e-6` | Nominal-support departure threshold, after removing mounting/profile offsets. |
| `routing_jump_sample_spacing_m` | `0.001` | Maximum sampling interval for measuring continuous raised transfers. |

Among feasible candidates, selection minimizes continuous raised-transfer count,
then weighted DSV RMS longitudinal field change relative to the original
directed contours at fixed current, then fabricated length. A raised transfer
crossing several turns counts once, including across the stored cycle seam.
Physical and field limits remain hard constraints. This bounded search reports
the best validated route found; it does not certify global optimality. See
[joint routing](JOINT_ROUTING.md) for candidate families and validation scope.

Manufacturing sweep controls include:

| Field | Unit | Meaning |
|---|---:|---|
| `sweep_bend_mode` | - | `"fillet"` builds radius-controlled manufacturing-spine bends; `"miter"` retains the polyline for visualization. |
| `sweep_min_bend_radius` | m | Hard minimum centerline bend radius. In fillet mode CoilGen also enforces the cross-section envelope limit below. |
| `sweep_corner_cluster_max_source_deviation_m` | m | Default `0` disables compound-corner fallback. A positive value explicitly permits local intrinsic two-arc fillets for unresolved corners inside one protected contour on an analytic cylinder. The two-way corridor is measured against the original contour after its declared cylinder retraction, including the validation sampling margin. Raw contour arrays and current stay unchanged; endpoint/tangent, bend, support, body and fixed-current field checks still apply. Requires `sweep_bend_mode="fillet"`. |
| `sweep_corner_detection_angle_deg` | deg | Tangent-jump scale for isolated CAD-style vertices; default `15`. Detection also promotes locally undersized radii and uses physical arclength windows tied to the required bend radius, so a critical bend cannot be hidden by distributing its turn over several samples. |
| `sweep_max_curvature_envelope_ratio` | - | Maximum allowed `cross-section envelope radius / centerline radius`; default `0.5` and always less than `1`. |
| `sweep_bend_arc_angle_deg` | deg | Maximum angular sampling step for an accepted circular bend. |

Curved-support corners are not rounded by projecting a Euclidean arc onto the
mesh. They are reported as requiring intrinsic surface rerouting; planar,
offset-surface, free-3D, and surface/free transition bends can be constructed by
the manufacturing-spine resolver.

For shimming workflows, use `field_display_units="Hz"` and inspect the
`shim_layout_residual_*` summary fields. These are computed at the final fitted
current from `target - connected-wire Bz`, not from the per-ampere diagnostic
maps.
The preset `local_b0_shim` applies Hz input/display units and lightweight
defaults for imported local B0 shim maps.

## Optional Multi-Turn Contour Sparsification

Pairwise contour sparsification is an opt-in post-contouring operation. Each
pass replaces adjacent contours from the same topological family by one
on-surface representative whose `turn_multiplicity` is the sum of its sources.
For example, three passes can form `x1 -> x2 -> x4 -> x8` representatives. The
current per physical turn remains `contour_step`; field evaluation applies the
integer multiplicity separately.

```python
config = CoilGenConfig(
    contour_sparsification_mode="pairwise",
    contour_sparsification_positioning="midpoint_level",  # or "field_fit"
    contour_sparsification_max_rms_field_error_percent=1.0,
    contour_sparsification_max_peak_field_error_percent=5.0,
    contour_sparsification_max_merges=0,  # 0 means no count limit
    contour_sparsification_max_passes=3,
    contour_sparsification_max_turn_multiplicity=8,
    contour_sparsification_field_fit_candidate_count=9,
    contour_sparsification_preserve_symmetry=True,
    # Optional cumulative target-residual guard for merging and pruning.
    # None preserves the earlier field-error-only merge behavior.
    contour_sparsification_residual_relative_tolerance=0.0,
    contour_sparsification_residual_absolute_tolerance_T=0.0,
    contour_sparsification_pruning_mode="residual_budget",
    contour_sparsification_pruning_residual_relative_tolerance=0.0,
    contour_sparsification_pruning_residual_absolute_tolerance_T=0.0,
    contour_sparsification_pruning_max_removed_geometries=0,  # 0 means no count limit
    contour_sparsification_pruning_target_geometry_count=5,  # lower bound; symmetry orbits stay intact
    contour_sparsification_pruning_preserve_symmetry=True,
    contour_sparsification_pruning_search_mode="auto",  # exact for small orbit sets, beam otherwise
    contour_sparsification_pruning_exact_max_orbits=16,
    contour_sparsification_pruning_beam_width=128,
    contour_sparsification_post_pruning_merge_mode="off",  # off | retained_adjacent
    contour_sparsification_post_pruning_max_passes=1,
    skip_postprocessing=True,
)
```

The RMS and peak limits are cumulative relative to the original unsparsified
contour field; every later pass uses the same baseline rather than receiving a
new error budget. `max_merges` counts accepted pair operations across all
passes. `field_fit_candidate_count` controls the number of candidate levels
sampled around the multiplicity-weighted source level.

When either cumulative sparsification residual tolerance is not `None`, every
accepted merge and post-sparsification deletion is constrained against the same
unsparsified contour baseline. This permits a wider RMS/peak field envelope for
fewer geometries without allowing an unreported loss of target or shim
performance. Optional post-sparsification pruning then applies
multiplicity-aware global subset selection to the final weighted
representatives. Complete mirror orbits are the atomic decisions. The target
geometry count is a lower bound: if five geometries cannot be represented as
a union of complete orbits, a feasible six-geometry design is retained instead
of deleting one mirror partner. Small orbit sets are exhaustively searched and
report an optimality certificate; larger sets use deterministic beam search.
An orbit is removed only when both the residual and field-error envelopes
remain satisfied. The legacy
`contour_sparsification_pruning_residual_*` settings are used only when both
cumulative tolerances are `None`. Source-provenance clusters are filtered with
the active representatives, and every accepted removal records source indices,
turn count, field significance, residual change, and cumulative field error.
With `contour_sparsification_post_pruning_merge_mode="retained_adjacent"`, a
final pass may merge contours that became neighbors only after pruning. The
pair must remain in one intrinsic topology family and pass the same cumulative
field, residual, symmetry, spacing, and swept-manufacturing checks.

The present realization mode is `ideal_weighted`: it is suitable for
electromagnetic comparison and contour plotting, but it does not invent
separated copper tracks. For that reason configuration validation requires
`skip_postprocessing=True`. Reports distinguish nominal contour geometries
from physical turn count, and hardware compilation marks the design incomplete
until physical turn centerlines are supplied.

The sparsification result stored on each `CoilPart` contains the original
contours, final source-provenance clusters, pass-indexed accepted and rejected
merge records, pairwise field errors, and the global field deviation from the
unsparsified contour layout.

## Project And Channel Parameters

For multi-channel systems, use `CoilProject` instead of repeating whole design
objects by hand. The project owns shared geometry, conductor, optimization,
layout, output, and plotting defaults. Each `CoilChannelSpec` supplies only the
fields that differ for that channel.

Expert scripts that already build a full `CoilGenConfig` can wrap it with
`CoilGenConfigDesign` and still use the same project/channel runner:

```python
project = CoilProject("expert_config_project")
project.add_channel("main", design=CoilGenConfigDesign(name="main", config=config))
```

Resolution order is deliberate:

```text
template design values
  -> CoilProject.design_overrides       # shared by all channels
  -> CoilChannelSpec.design_overrides   # only this channel
  -> CoilChannelSpec.target_bz_expression / target_scale shortcuts
```

So a project-level `contour_levels=10` applies to every channel unless one
channel overrides `contour_levels` itself. Resolved channel designs are copied
before overrides are applied, so mutable settings such as `advanced_parameters`
do not leak edits from one channel into another.

Example:

```python
project = CoilProject(
    "standard_cylindrical_3axis_gradient",
    template=template,
    design_overrides={"contour_levels": 10},
)
project.add_channel("GX", target_bz_expression="x", color="tab:blue")
project.add_channel("GY", target_bz_expression="y", color="tab:green", contour_levels=12)
project.add_channel("GZ", target_bz_expression="z", color="tab:red")
```

Useful project methods:

- `project.describe()` returns an IDE-friendly summary of resolved channel
  setup.
- `project.parameter_table()` returns rows showing shared values plus
  per-channel overrides.
- `project.resolved_designs()` returns the exact executable `CoilDesign`
  objects for inspection before an expensive run.
- `project.run()` runs all enabled channels and writes project summary files.
- `project.show_plot_book(run)` opens one combined plot book with project
  summary pages and channel-prefixed result pages.

## Inductance And Resistance

The example scripts keep final electrical evaluation active by default.

For user-facing design classes:

```python
"conductor_width_m": 0.001,
"conductor_height_m": 0.001,
"conductor_resistivity_ohm_m": 0.018e-6,
"skip_inductance": False,
```

For direct `CoilGenConfig` dictionaries:

```python
"conductor_cross_section_width": 0.001,
"conductor_cross_section_height": 0.001,
"specific_conductivity_conductor": 0.018e-6,
"skip_inductance_calculation": False,
"inductance_source": "auto",
```

`specific_conductivity_conductor` is the historic MATLAB-compatible parameter
name; in the Python electrical estimate it is treated as conductor resistivity
in ohm metre. The default value is copper-like.

Final wire resistance is computed from

```text
R = rho * wire_length / conductor_cross_section_area
```

Final inductance uses the thin-wire Neumann/Biot-Savart line integral with an
equivalent round-wire radius derived from the configured cross-section area.
With `inductance_source="auto"`, routed examples use the connected final wire
path, while contour-only grid studies use the closed unconnected contour loops.

## Volumetric Sweep Controls

`cross_sectional_points` is a strictly convex polygon in the physical local
coordinates `(surface_transverse, surface_normal)`. Thus coordinate 0 is
conductor width across the route and coordinate 1 is conductor height normal to
the support. The default `((0.0, 0.0),)` is a sentinel that generates the
configured rectangle. Repeated, concave, self-intersecting, zero-area, or
non-finite custom profiles are rejected.

| Field | Unit | Meaning |
|---|---:|---|
| `contour_only_sweep_mode` | - | When true, bypass routing and sweep every closed contour into a separate volumetric mesh body. Requires `skip_sweep=False`; ideal-weighted contour sparsification must be disabled. |
| `sweep_max_segment_length` | m | Optional explicit upper bound for longitudinal sweep facets. Zero uses the triangle-quality-derived bound. |
| `sweep_max_chord_error` | m | Maximum support-following correction between a 3D chord and its projected midpoint; default `5e-5`. |
| `sweep_max_frame_rotation_deg` | deg | Maximum resolvable support-normal rotation between adjacent rings; default `10`. |
| `sweep_self_intersection_check` | - | `capsule_fast` uses the centerline BVH and is the default manufacturing gate. `off` makes the triangular body visualization-only. |
| `conductor_mounting_mode` | - | `centered`, `surface_outer`, or `surface_inner`. Asymmetric profiles use their actual normal minimum/maximum when offset to the support. |

Inserted support samples retain routed normal lift: CoilGen interpolates the
signed endpoint offsets and applies them to the closest support point. The
finished triangular boundary is additionally checked for directed-edge
orientation, positive volume, short edges, local folds, triangle aspect ratio,
triangle-BVH self-intersection, and support penetration. A failed check blocks
manufacturing STL/STEP export but leaves the mesh available for visualization
and diagnostics. This triangular backend remains a faceted representation, not
an analytic CAD sweep for robust Boolean subtraction.

Contour-only mode stores the authoritative meshes in
`CoilPart.swept_bodies`, one body per contour, while `CoilPart.swept_body`
remains a disconnected aggregate for legacy visualization. STL and STEP exports
use names such as `*_part1_contour0001.stl` and
`*_part1_contour0001.step`; every exported body must independently pass the
closed-shell, manifold, orientation, non-degeneracy, positive-volume, and
self-intersection gates.

## Fixed-Error Efficiency Optimization

Besides the default Tikhonov/L-curve mode, CoilGen supports:

```python
"sf_opt_method": "target_error_constrained_efficiency",
"target_error_tolerance_percent": 5.0,
```

or, with the user-facing dataclasses:

```python
optimization_method="target_error_constrained_efficiency",
target_error_tolerance_percent=5.0,
```

For geometry comparisons and active-shielded studies, the recommended mode is:

```python
"sf_opt_method": "constrained_quadratic",
"target_error_constraint_mode": "separate",
"target_error_tolerance_percent": 5.0,
"suppression_field_rms_limit_T": 0.005,
"stream_function_smoothness_factor": 0.05,
```

The mathematical problem is

```text
minimize_x  x^T R x + eta ||L psi||_2^2
subject to  ||A_dsv x - b_dsv||_2 <= eps_dsv ||b_dsv||_2 + delta_dsv
            ||A_sup x||_2 / sqrt(N_sup) <= B_sup,rms,max
```

where `x` is the reduced stream-function vector, `A_dsv` maps it to Bz on the
internal DSV target samples, `b_dsv` is the requested DSV Bz field, `A_sup`
maps it to Bz on the external suppression surface, `N_sup` is the number of
active suppression samples, `R` is the continuous-sheet resistance matrix, and `L` is a
surface smoothness operator. The suppression constraint is therefore a physical
RMS fringe-field limit in tesla. If `suppression_field_rms_limit_T` is left at
zero, CoilGen falls back to the older residual-norm budget
`||A_sup x - b_sup||_2 <= eps_sup ||b_dsv||_2 + delta_sup`.

With CVXPY installed, `constrained_quadratic` solves this problem directly as a
convex conic problem. The optional `target_error_constrained_efficiency` fallback
searches a restricted Tikhonov family; it does not generally solve the same
multiple-constraint optimum. Such fallback results are marked diagnostic.
Use `constrained_quadratic_fallback_policy="strict"` for comparisons that require
the requested direct formulation. Minimax and requested hardware bounds cannot
silently fall back to the scalar search.

Set `constrained_quadratic_objective_mode="minimax_current_density"` to minimize
the maximum vector surface current density, in A/m, over every mesh face. This
retains the field and engineering hard bounds and requires zero explicit
smoothness/spacing penalty factors. The implicit engineering spacing penalty is
disabled for this objective. For a power/spacing tradeoff, use `balanced` with a
hard `surface_current_density_limit_A_per_m` and sweep that limit.

The default absolute numerical acceptance slack is `1e-7 T`; it is separate from
the requested field budget and can be excessive for small shim targets. Set
`target_error_constraint_acceptance_absolute_T=0.0` to use relative acceptance
only. `target_error_constraint_acceptance_tolerance=0.0` derives that relative
tolerance from the solver setting; a positive value explicitly sets it.
See [Stream-function optimization](STREAM_FUNCTION_OPTIMIZATION.md) for the
mathematical inventory, typed API, solver capabilities, and remaining gaps.

Relevant low-level fields:

| Field | Unit | Meaning |
|---|---:|---|
| `target_error_constraint_mode` | - | `"separate"` for DSV plus suppression constraints; `"combined"` for legacy weighted residual. |
| `target_error_tolerance_percent` | % | Relative DSV residual budget. Must be between 0 and 100. |
| `target_error_tolerance_absolute_T` | T | Optional absolute DSV residual-norm allowance, normally `0.0`. |
| `suppression_field_rms_limit_T` | T | Preferred fixed RMS Bz limit on suppression points. Set >0 for paper studies. |
| `suppression_error_tolerance_percent` | % | Legacy relative suppression residual budget, referenced to `||b_dsv||_2`. Used only when `suppression_field_rms_limit_T` is zero. |
| `suppression_error_tolerance_absolute_T` | T | Legacy absolute suppression residual-norm allowance, normally `0.0`. |
| `target_error_constraint_factor_count` | - | Number of candidate multipliers before binary refinement. |
| `stream_function_smoothness_factor` | - | Optional dimensionless smoothness weight, internally scaled to the field normal-equation matrix. |

## Mesh-Coupled Contour Smoothing

Smoothing is optional and off by default. In the GUI, open **Winding and
routing → Contour smoothing → Smooth on surface**. The engine fits the closed
contours before grouping, cutting and interconnection; it does not cosmetically
smooth plotted wire paths. Changing these settings invalidates the contour
stage and downstream geometry while allowing the unchanged current sheet to
be reused.

`contour_path_regularization_mode="parameterized_bspline"` supports two
smoothing-scale policies. The default `contour_smoothing_scale_mode="legacy"`
uses the historical SciPy residual bound. The `"mesh_coupled"` policy estimates
an intrinsic element length from the triangles crossed by each contour and
calibrates a periodic second-difference filter to that physical wavelength.

| Field | Unit | Meaning |
|---|---:|---|
| `contour_smoothing_scale_mode` | - | `"legacy"` or `"mesh_coupled"`. |
| `contour_smoothing_mesh_gain` | - | Fraction of a one-mesh-length oscillation retained by mesh-coupled smoothing; `0.1` retains 10%. |
| `contour_smoothing_work_samples_per_mesh_length` | - | Uniform working samples per estimated mesh length; must be at least 4. |
| `contour_smoothing_output_samples_per_mesh_length` | - | Minimum output samples per estimated mesh length; must be at least 2. |
| `contour_resample_target_segment_length` | m | Independent upper bound on final contour segment length. |
| `contour_smoothing_max_displacement` | m | Maximum movement from the extracted contour. Set to `0.0` to disable this cap; turn-spacing preservation remains independent. |

The effective output segment length is the smaller of
`contour_resample_target_segment_length` and the estimated mesh length divided
by `contour_smoothing_output_samples_per_mesh_length`. Refining the support
mesh therefore shortens both the attenuation wavelength and, when needed, the
output sampling interval.

For a bounded first trial, use:

```python
config.contour_path_regularization_mode = "parameterized_bspline"
config.contour_smoothing_scale_mode = "mesh_coupled"
config.contour_smoothing_mesh_gain = 0.1
config.contour_smoothing_max_displacement = 0.001  # metres
config.contour_smoothing_preserve_min_turn_distance = True
```

A smaller mesh gain attenuates mesh-scale oscillations more strongly. The fit
backs off when the displacement cap or surface-patch constraints cannot be
met. Invalid candidates retain the original contour and record their reason.
Closure, circulation, contour potential, current orientation and turn
multiplicity are retained. Every output segment has face and barycentric
provenance on the original mesh. The spacing guard uses segment distances;
when a positive configured minimum was already met, it preserves that
requirement, otherwise it preserves the original minimum spacing. The field
must be evaluated again at the unchanged contour current because smoothing
changes the wire geometry. Surface attachment does not imply exact circular
geometry on a coarse faceted cylinder.

## Example Defaults

Scientist-facing example scripts define one editable `PARAMETERS` dictionary
near the top of the file. They no longer duplicate a second local defaults
dictionary. Missing advanced values are filled from:

```text
projects/general_examples/parameters/defaults.py
```

The parser prints every missing value that was taken from the defaults file.
This gives two intended edit levels:

- edit `PARAMETERS` in an example script for a specific design or study;
- edit `projects/general_examples/parameters/defaults.py` only when changing the
  shared fallback behavior of that example family.

## Validation

`validate_config(config)` checks common unit, range, file-format, enum, and
compatibility errors before the numerical pipeline runs. `design.run()` performs
this validation automatically.
