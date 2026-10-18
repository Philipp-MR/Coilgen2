# Primitive Geometry Reference

CoilGen supports several generated support-surface primitives. Use the
user-facing design classes when possible; they produce the correct
`CoilGenConfig` fields and keep all dimensions in SI units.

## Public Design Classes

- `CylinderCoilDesign`: open circular cylinder.
- `PlanarCoilDesign`: rectangular planar surface.
- `CircularDiscCoilDesign`: single filled circular disc.
- `AnnularDiscCoilDesign`: circular disc with a central hole.
- `PartialCylinderCoilDesign`: open angular sector of a circular cylinder.
- `EllipticalCylinderCoilDesign`: open elliptical cylinder.
- `SuperellipseCylinderCoilDesign`: rounded-rectangle / superellipse cylinder.
- `SphericalCapCoilDesign`: spherical cap surface.
- `EllipsoidalCapCoilDesign`: ellipsoidal cap surface.
- `ConicalFrustumCoilDesign`: tapered cylindrical support.
- `ToroidalSegmentCoilDesign`: curved toroidal segment.
- `RuledSurfaceCoilDesign`: surface interpolated between two 3D boundary curves.
- `SurfaceOfRevolutionCoilDesign`: axisymmetric surface from `(z, radius)`
  samples.
- `ParametricSurfaceCoilDesign`: arbitrary sampled `surface(u, v)` Python
  callable. This uses an in-memory mesh key because callables are not
  serializable into a static `CoilGenConfig`.
- `ShieldedCoilDesign`: generic wrapper around any generated or in-memory
  primary design. It creates an active shield by offsetting the primary mesh
  along vertex normals with identical topology.
- `BiplanarCoilDesign`, `CircularBiplanarCoilDesign`,
  `ShieldedCircularBiplanarCoilDesign`: two-plane variants.
- `FlangedSplitBoreCoilDesign`, `ShieldedFlangedSplitBoreCoilDesign`:
  split-bore surfaces with radial and split-gap flange sections.

Example:

```python
from coilgen import AnnularDiscCoilDesign

design = AnnularDiscCoilDesign(
    target_bz_expression="y",
    inner_radius_m=0.07,
    outer_radius_m=0.25,
    radial_divisions=6,
    circular_divisions=48,
)
result = design.run()
```

## Low-Level Mesh Keys

The same primitives are available in `CoilGenConfig.coil_mesh_file`:

- `create cylinder mesh`
- `create planary mesh`
- `create circular mesh`
- `create annular disc mesh`
- `create partial cylinder mesh`
- `create elliptical cylinder mesh`
- `create superellipse cylinder mesh`
- `create spherical cap mesh`
- `create ellipsoidal cap mesh`
- `create conical frustum mesh`
- `create toroidal segment mesh`
- `create ruled surface mesh`
- `create surface of revolution mesh`
- `create circular bi-planary mesh`
- `create bi-planary mesh`
- `create shielded bi-planary mesh`
- `create flanged split bore mesh`
- `create shielded flanged split bore mesh`

The validation layer checks parameter counts, positive physical dimensions,
division counts, normal vectors, angular spans, ruled-surface boundary curves,
and surface-of-revolution profile ordering.

## Shield Copies

For primitives that need an active shield with the same topology as the primary
surface, use `coilgen.geometry.offset_mesh_along_vertex_normals(mesh,
separation_m)`. The returned mesh keeps the same face connectivity and shifts
each vertex by the requested outward separation along the vertex normals. This
is the preferred way to create paired primary/shield surfaces for exploratory
geometries, because the stream-function variables stay topology-compatible.

For normal design scripts, prefer the higher-level wrapper:

```python
from coilgen import AnnularDiscCoilDesign, ShieldedCoilDesign

design = ShieldedCoilDesign(
    primary_design=AnnularDiscCoilDesign(
        inner_radius_m=0.07,
        outer_radius_m=0.25,
    ),
    shield_separation_m=0.03,
)
```

The wrapper registers a combined in-memory mesh containing the primary and its
offset shield. The shield uses the same faces and vertex ordering as the
primary, shifted outward by `shield_separation_m`.

## Mesh Quality Diagnostics

Use `coilgen.mesh_quality_report(mesh)` before running a new exploratory
support surface. The report checks:

- invalid face indices, repeated vertices, zero-area faces;
- boundary-loop count and boundary-loop sizes;
- disconnected face components;
- non-manifold edges and orientability;
- supplied-normal consistency;
- optional obvious self-intersection checks;
- edge-length distribution and triangle aspect-ratio statistics.

Errors indicate surfaces that are not suitable for intrinsic stream-function
topology. Warnings identify meshes that can run but may produce unstable
contours or poorly conditioned optimization matrices.

## Mesh Metadata

Generated meshes carry a `metadata` dictionary with the geometry type,
SI-valued parameters, basic symmetry labels where known, and role information
for offset copies. This metadata is preserved through in-memory registration and
mesh combination, which makes later diagnostics and paper-study bookkeeping
less ambiguous.

## Notes For New Support Structures

Generated mesh faces should have consistent physical normals. For stream
function designs this matters because contour current orientation is derived
from the mesh normal and stream-function gradient. When adding a new primitive,
also add:

- a geometry builder in `coilgen.geometry`;
- a `GeometryFactory` registration;
- config validation in `coilgen.config_validation`;
- a user-facing `CoilDesign` subclass if the geometry is useful to users;
- `PipelineGeometrySpec` conversion if the primitive should be representable
  in high-level problem objects;
- at least one unit test that validates and builds the mesh.
