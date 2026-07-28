# Geodesic Topology Fork Plan

This fork removes the global surface UV parameterization from the core CoilGen
pipeline. The first target is replacing UV-based contour inclusion with
intrinsic topology on the triangulated coil surface. There should be no runtime
switch between UV topology and intrinsic topology in this fork.

## Core Mathematical Replacement

Current grouping builds an inclusion matrix with planar point-in-polygon tests:

```text
contains[outer, inner] = all points of inner loop lie inside outer loop in UV.
```

The UV-free version should treat each contour loop as a barrier on the surface.
For a surface mesh S and loop Gamma_j, remove crossing adjacency along Gamma_j.
The remaining surface S \ Gamma_j splits into connected components when Gamma_j
is separating. Another loop Gamma_i is inside Gamma_j if its samples lie in the
chosen bounded/interior component.

Distance-field view:

```text
d_j,p(x) = shortest path distance from seed p to x on S \ Gamma_j
```

The finite support of this field is exactly the reachable component containing
p. Distances are useful for robustness, but the inclusion decision comes from
reachability/component labels created with Gamma_j as an impenetrable wall.

## Discrete Distance-Topology Algorithm

Let the mesh be M = (V, E, F). A contour loop is an ordered set of surface
segments. Each segment has endpoints in barycentric coordinates of one triangle
and, in the generic case, connects two different triangle edges.

For each tested loop Gamma_j:

1. Build a virtual cut mesh M_j.
   - Do not just mark nearest vertices as blocked; that leaks on coarse meshes.
   - Split every crossed triangle by the contour segment inside that triangle.
   - Duplicate the contour edge so the two sides of Gamma_j are not adjacent.
   - The split can be virtual: store subface nodes and adjacencies rather than
     physically rewriting the production mesh.

2. Build the dual adjacency graph G_j of the subfaces of M_j.
   - Adjacent subfaces share a non-contour edge.
   - Subfaces across Gamma_j are deliberately not connected.

3. Compute region labels and optional geodesic distances.
   - A flood fill on G_j gives the same finite-support topology as an
     infinite-cost geodesic metric.
   - Dijkstra/FMM on G_j with edge weights equal to surface distances gives
     approximate geodesic distance values:

     ```text
     D_q(r) = shortest path distance from seed subface q to subface r in G_j
     ```

   - Separate connected components appear as infinite distance from each other.

4. Assign each other loop Gamma_i to a side.
   - For every sample point of Gamma_i, locate the subface/component containing
     it in M_j.
   - Use a tolerance vote, e.g. inside if at least 95% of valid samples lie in
     the selected interior component.

5. Select the interior component.
   - Prefer the component with smaller integrated surface area for ordinary
     nested stream-function contours.
   - On open surfaces, prefer the component that does not touch a mesh boundary.
   - If more than two components or ambiguous area occurs, use consistency over
     the whole contour family: inner loops should form a partial order, not
     cycles.

This is a distance-metric construction because the metric is evaluated on the
cut surface M_j, not on the original uncut surface M. On M, the shortest path
between two sides may cross Gamma_j. On M_j, crossing Gamma_j has infinite cost.

## Implementation Phases

1. Preserve the original project as the control case.
   - Keep the original sibling directory `coilgen-python` untouched.
   - In this fork, remove UV from the main pipeline rather than maintaining a
     dual UV/geodesic topology switch.

2. Introduce surface-intrinsic contour metadata.
   - Add segment provenance to contours: face index, crossed mesh edge, and
     barycentric coordinates for each contour point where possible.
   - Stop relying on UV endpoints for contour stitching; stitch isoline
     segments by mesh face/edge incidence and 3D coordinates.
   - Do not require UV fields for contour construction, topology, grouping, or
     postprocessing.

3. Build a barrier representation for one contour loop.
   - For every contour segment, identify the mesh face it crosses.
   - Mark crossed face-to-face adjacencies as blocked.
   - If a contour runs through a vertex or along an edge, use a tolerance band
     or split/duplicate local adjacency so the loop remains a closed wall.

4. Compute geodesic/reachability regions.
   - Start with graph Dijkstra/FMM-like propagation over mesh faces or vertices
     with blocked adjacencies removed.
   - Store component labels and optional distances.
   - Later, replace or augment with Fast Marching or Heat Method if smooth
     geodesic distances are needed for cut selection or tolerance margins.

5. Define the interior side of a loop.
   - If the barrier creates two components, choose the interior by:
     - smaller surface area for disk-like nested contours;
     - component not touching an exterior mesh boundary for open surfaces;
     - consistency with stream-function nesting when area is ambiguous.
   - If the loop is non-separating, record that no valid inside relation exists.

6. Replace `topological_loop_grouping`.
   - Build `loop_in_loop_matrix` from intrinsic component membership:

     ```text
     loop_i inside loop_j iff most samples of loop_i lie in Gamma_j's interior component.
     ```

   - Reuse only the existing group/level construction after the inclusion
     matrix is generated.
   - Delete the UV point-in-polygon inclusion path once the intrinsic matrix is
     in place.

7. Remove downstream UV dependencies from postprocessing.
   - `calculate_group_centers`: compute centers in 3D/surface terms, using
     interior-region centroids or medoids instead of UV polygon centers.
   - Opening remains 3D sphere based; preserve `open_loop_with_3d_sphere`.
   - Replace UV-based self-intersection/normal-shift logic with surface-aware
     checks or make it optional.
   - Keep cylindrical PCB unwrapping as a manufacturing-specific export, not a
     global mesh parameterization requirement.
   - Remove UV visualization/export calls from the default workflow.

8. Test progressively.
   - Planar nested squares should match the current UV grouping.
   - Cylindrical nested contour examples should match the old pipeline.
   - Split-bore/flange examples should verify disconnected surfaces and open
     boundaries.
   - Add a non-separating-loop case and assert it does not create false nesting.

## First Code Targets

- `src/coilgen/contours.py`
  - Decouple contour extraction/stitching from `mesh.uv`.
  - Preserve per-point surface provenance.

- `src/coilgen/layout.py`
  - Add `build_intrinsic_loop_inclusion_matrix`.
  - Refactor `topological_loop_grouping` so grouping consumes an inclusion
    matrix independent of how it was computed.

- `src/coilgen/geometry.py`
  - Stop calling `parameterize_mesh` in the main pipeline.
  - Remove global UV construction from core geometry preparation.

- `src/coilgen/models.py`
  - Remove UV requirements from `ContourLine`, `WirePath`, and connected group
    containers.
  - Keep any flat coordinates only in manufacturing/export-specific structures.

## Key Risk

The hard part is not computing distances; it is making each contour loop a
topologically correct barrier on a discrete triangle mesh. If the barrier leaks
through a vertex, touches a boundary, or crosses itself, component labels become
wrong. Solve this before optimizing the distance method.

## Progress

- First slice completed:
  - Main pipeline no longer calls global `parameterize_mesh`.
  - `extract_contours` no longer requires `mesh.uv`.
  - Isoline segments are extracted from triangle stream-function crossings in
    3D and stitched by shared mesh-edge interpolation keys.
  - `ContourLine` carries 3D coordinates plus surface provenance:
    `surface_faces`, `surface_barycentric`, and `segment_faces`.
  - A synthetic disc smoke test produced a closed UV-free contour with segment
    face provenance.
- Second slice completed:
  - Added `surface_topology.py` with face adjacency, barrier component labels,
    and a face-graph geodesic distance field with infinite contour barriers.
  - Added crossed-edge metadata to `ContourLine`:
    `surface_edge_vertices` and `surface_edge_t`.
  - `topological_loop_grouping` now builds `loop_in_loop_matrix` from intrinsic
    surface component membership instead of UV point-in-polygon.
  - Deleted the old UV mutual-inclusion helper.
  - `calculate_group_centers` no longer depends on UV and now produces 3D group
    centers only.
  - A nested-contour smoke test on a UV-free planar mesh produced the expected
    inclusion matrix.
- Third slice completed:
  - Replaced the conservative crossed-face removal barrier with a virtual cut
    graph.
  - Crossed triangles are now split into two side-cells, with adjacency across
    original mesh edges connected only for matching edge intervals.
  - Added `segment_barycentric` to `ContourLine`; each contour segment now has
    the correct endpoint coordinates in its own triangle.
  - Inclusion voting locates each tested contour segment in the virtual cut
    graph by face and segment-midpoint barycentric coordinates.
  - The geodesic distance field now runs on the virtual cut graph, returning
    finite distances only within the reachable side of the contour wall.
  - A nested-contour smoke test split the UV-free planar mesh into two virtual
    cut components and recovered the expected inclusion matrix.
- Fourth slice completed:
  - Within-group interconnection no longer requires UV rectangle opening.
    Cut positions are selected in 3D and opened with `open_loop_with_3d_sphere`.
  - `open_loop_with_3d_sphere` works from 3D points and returns 3D opening
    cut points.
  - `ConnectedGroup` and `WirePath` carry 3D paths through within-group and
    among-group handoff.
  - Existing return-path suppression now uses 3D return points instead of UV
    return points.
  - Cleanup and normal-shift code operate on 3D-only wire paths; the old flat
    self-intersection detector has been removed.
  - A UV-free nested-contour smoke test now reaches a cleaned 3D wire path.
- Fifth slice completed:
  - Removed the remaining path-level flat coordinate fields from `ContourLine`,
    `WirePath`, `ConnectedGroup`, `LoopGroup`, and `CoilPart`.
  - Removed `Mesh.uv`, `Mesh.uv_faces`, and the unused `parameterize_mesh`
    family from `geometry.py`.
  - Deleted the old 2D rectangle/cut-shape opening helpers and public exports.
  - Default visual exports now produce 3D-coordinate projections and field
    comparison only.
  - Replaced the old UV-oriented test file with intrinsic-topology and 3D path
    tests; `compileall` and the unit suite pass.
- Sixth slice completed:
  - Removed stale generated flat visualization outputs.
  - Removed unused parameterization config fields while preserving the PCB
    cylinder flag.
  - Added `validate_intrinsic_topology_inputs` and `TopologyValidationReport`.
  - `topological_loop_grouping` now records topology diagnostics and fails
    early for invalid meshes or contours.
  - Preflight now checks finite triangular meshes, non-manifold edges,
    orientability, contour closure, segment provenance, repeated face cuts,
    boundary-touching contours, vertex-touching contours, and non-separating
    contours.
  - README now describes the intrinsic topology fork and projection-only
    visualization outputs.
- Seventh slice completed:
  - Added `find_wire_segment_contacts_3d`, an intrinsic 3D segment proximity
    detector for non-adjacent wire path segments.
  - `shift_return_paths` now marks arbitrary 3D self-contact segments for
    normal shifting, preferring existing return-path markers when available.
  - `route_diagnostics` now reports `self_contact_count`.
  - Added a curved cylinder contour preflight test and a 3D crossing-shift test.
