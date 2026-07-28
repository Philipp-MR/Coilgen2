# CoilGen Project Understanding

## Purpose

CoilGen generates MRI/NMR coil layouts on 3D support surfaces. The user defines:

- a coil surface mesh, usually an STL or a generated mesh;
- a target magnetic field, usually a scalar expression for Bz such as `x` or `y`;
- optional secondary/suppression surfaces where the field should be minimized;
- contour/layout parameters such as number of potential levels and interconnection settings.

The code optimizes a stream function over the mesh using a boundary-element-style basis. It then discretizes the optimized stream function into contour loops, connects those loops into manufacturable non-overlapping wire paths, and evaluates the resulting field.

## Main Pipeline

`CoilGen.m` is the driver. Its algorithm is deliberately linear:

1. Parse name-value inputs with `parse_input`.
2. Read or generate coil, target, and suppression meshes with `read_mesh`.
3. Split disconnected mesh components with `split_disconnected_mesh`.
4. Optionally refine the mesh with `refine_mesh`.
5. Parameterize each 3D mesh component into 2D UV space with `parameterize_mesh`.
6. Define target points and target B field with `define_target_field`.
7. Evaluate cache reuse through `temp_evaluation`.
8. Build one-ring neighborhoods with `calculate_one_ring_by_mesh`.
9. Build stream-function basis elements with `calculate_basis_functions`.
10. Build Biot-Savart sensitivity matrices with `calculate_sensitivity_matrix`.
11. Build the resistance regularization matrix with `calculate_resistance_matrix`.
12. Optimize the stream function with `stream_function_optimization`.
13. Convert stream-function values to contour levels with `calc_potential_levels`.
14. Extract contour loops in UV space with `calc_contours_by_triangular_potential_cuts`.
15. Map loops back to 3D, smooth/filter them, and evaluate loop significance with `process_raw_loops`.
16. If post-processing is enabled, group and interconnect loops, shift return paths, create PCB/swept bodies, and optionally estimate inductance.
17. Evaluate final wire-path fields and errors with `evaluate_field_errors`.
18. Calculate resulting gradient metrics with `calculate_gradient`.

## Data Model

MATLAB structs carry most state. Important containers:

- `input`: parsed configuration.
- `coil_parts`: array of disconnected mesh components. Each component accumulates mesh data, basis data, sensitivity/resistance matrices, optimized stream function, contours, groups, and final wire path.
- `coil_mesh`: uses mostly 3-by-N `vertices`, 3-by-M `faces`, 2-by-N `uv`, 3-by-N normals `n`, and cell-array `boundary` loops.
- `target_field`: contains `coords`, `b`, `weights`, suppression flags/groups, and sometimes symbolic gradient targets.
- `combined_mesh`: merged view of all coil parts for global optimization.
- `result_out`: final public result object containing coil parts, fields, errors, input data, layout gradient, and combined mesh.

MATLAB indexing is 1-based and many arrays are transposed when moving between MATLAB geometry APIs and CoilGen's internal convention. The Python port should normalize this early.

## Numerical Core

The stream function is represented with one basis function per mesh vertex. For each vertex, the basis function spans its one-ring triangles.

`calculate_basis_functions` computes, for each vertex-owned triangle:

- triangle area;
- face normal;
- triangle corner coordinates;
- piecewise-constant current vector, roughly `(point_c - point_b) / (2 * area)`.

`calculate_sensitivity_matrix` integrates the Biot-Savart contribution of each basis element at each target point using Gaussian integration over triangles. The output has shape equivalent to:

```text
B component x target point x basis node
```

`calculate_resistance_matrix` builds a local overlap matrix between basis functions sharing triangles, scaled by conductor material/thickness. This acts as the Tikhonov regularizer.

`stream_function_optimization` merges all coil parts, reduces boundary nodes so each boundary loop has a constant stream-function value, and solves:

```text
argmin_sf ||Cz * sf - target_Bz||^2 + ||lambda * R * sf||^2
```

The default method is Tikhonov via pseudo-inverse. An `fmincon` path exists but is secondary and MATLAB-specific.

## Geometry And Layout Core

`parameterize_mesh` is critical. It has three major paths:

- planar meshes: rotate into XY and use x/y as UV;
- cylindrical meshes: unwrap using angle and axial coordinate;
- custom flange/split-bore cylinder mode: detect the flange axis/split plane and embed half-disc flange geometry as a cylindrical extension in polar UV.

The local copy includes custom split-bore/flange support:

- `build_split_bore_flange_mesh`
- `create_split_cylinder_mesh`
- `build_two_coaxial_discs_mesh`
- shielded flanged and shielded biplanar mesh input options
- suppression-surface generation options

After optimization, `calc_contours_by_triangular_potential_cuts` finds mesh edges crossed by contour levels, interpolates UV cut points, stitches them into loops by edge/triangle connectivity, and orients each loop according to current direction.

`process_raw_loops` smooths loops, maps UV loops back to 3D through barycentric lookup (`uv_to_xyz`), deletes insignificant loops, closes loops, and computes loop length.

Post-processing then groups nested contours topologically, opens loops at cut positions, connects loops within groups, connects groups into one wire path, shifts return paths along surface normals to avoid overlap, creates PCB outputs, and can sweep a volumetric conductor body.

## Translation Priorities

Recommended Python phases:

1. Build stable dataclasses for input, mesh, coil part, target field, contour loop, group, and result.
2. Port mesh loading/generation and normalize shapes to NumPy arrays with 0-based faces.
3. Port target-field generation and simple sphere targets first.
4. Port one-ring, basis, sensitivity, resistance, and Tikhonov optimization.
5. Create parity tests against small MATLAB-generated fixtures where possible.
6. Port contour extraction and UV-to-XYZ mapping.
7. Port loop processing and field-error evaluation.
8. Port post-processing/interconnection in stages.
9. Port custom split-bore/flange geometry carefully, since it is local to this project and likely important.

Useful Python libraries to consider:

- `numpy`, `scipy` for arrays, sparse/block matrices, optimization, linear algebra;
- `trimesh` or `meshio` for STL mesh IO;
- `scipy.spatial` and possibly `shapely` for geometry/topology operations;
- `matplotlib`/`pyvista` for plotting and 3D inspection;
- `h5py`/`scipy.io` for `.mat` files, depending on MAT version.

## Known Risks

- MATLAB relies heavily on implicit shape conventions and transposes; Python should make array shapes explicit.
- Boundary reduction is subtle and affects the optimization degrees of freedom.
- UV parameterization and `uv_to_xyz` are numerically fragile near triangle edges.
- Post-processing is algorithmically stateful and less vectorized than the numerical core.
- The code has local custom mesh features that may not exist in upstream pyCoilGen.
- Some functions use duplicated helper implementations, especially Biot-Savart and smoothing utilities.
