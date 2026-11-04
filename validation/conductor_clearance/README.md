# Exported conductor clearance

The audit reads the fabrication review’s released or draft STL in world metres. It verifies the STL against the saved artifact digest and uses the matching reviewed ring order to identify each facet’s body, interval and owner. Reported distances are Euclidean copper-surface gaps. The required gap sums both insulation thicknesses, the manufacturing tolerance and the bend-clearance margin. For bodies with different settings the audit uses the larger tolerance and margin.

## Run

The run map is a JSON object mapping each GUI `template__channelIndex` to its completed job directory. Run these commands from the repository root with the project Python.

```text
python validation/conductor_clearance/audit_exports.py --runs jobs.json --output results
python validation/conductor_clearance/pack_evidence.py results reports --tag rerun
python -m pytest validation/conductor_clearance/test_surface_gap.py
```

Use `--case template__0` to inspect one channel or `--system template` to evaluate only the body pairs crossing channel boundaries in one assembly.

To illustrate a physical crossing, run `python validation/conductor_clearance/plot_witness.py --runs jobs.json --case biplanar_gz__0 --body 1 --output crossing.png`. The PNG shows the actual STL facets and a section through both closest points. Its companion JSON contains the gap, world coordinates, winding owners, facet and station indices and the STL digest. Automatic selection prefers opposing legs of the same cut when their projected intervals cross. Use `--faces I J` to reproduce a specific pair from the independent audit.

To check replacement channel exports in the original assembly geometry, use `python validation/conductor_clearance/assembly_context.py --runs jobs.json --system biplanar_gradient_shim --replacement-runs replacements.json --output candidate-assembly.json`. The replacement map accepts complete GUI jobs keyed by channel. Disjoint world bounding boxes provide a rigorous Euclidean lower bound. The tool measures actual STL facets if that bound cannot establish the required clearance. `--exact` measures every pair, and `--changed-only` limits the comparison to interfaces involving replaced channels.

To submit all templates through the Qt workbench without occupying the desktop, use a fresh job directory and keep the
checkout fixed for the duration of the run. The driver uses the Qt offscreen platform, a private VTK framebuffer and
four workers with one numerical thread per worker. From the repository root:

```text
python validation/conductor_clearance/offscreen_gui.py --output gallery-check --jobs gallery-check/jobs
python validation/conductor_clearance/audit_exports.py --runs gallery-check/new_runs.json --output gallery-check/cases
python validation/conductor_clearance/verify_viewport.py gallery-check
```

The last command reopens each submitted project revision, loads the result, and checks that the native viewport's
rendered winding coordinates are exactly those in the saved fabrication review arrays. The independent clearance
checks use the exported STL facets themselves, including in runs where other manufacturing checks keep the STL in draft.

## Geometry and interpretation

* **Remote intervals.** A same-body station pair is remote when the arclength of the intervening centerline exceeds four maximum profile diameters. The diameter is the largest corner-to-corner distance in the recorded profile. The arclength excludes both tested intervals and takes the shorter path around a closed winding. Every pair from distinct bodies is tested. A hierarchy of per-facet bounding boxes prunes only proven lower bounds. For separate bodies, a measured vertex pair supplies an initial upper bound. For self-clearance the search expands until it finds the global remote minimum. Solid-angle containment also catches a separate body wholly enclosed by another.
* **Short bends.** The short-fold screen also tests nonadjacent intervals within four diameters when the shorter intervening arc turns through a cumulative angle of at least 90°. This catches a close return immediately after a hairpin. A short-fold row with no witness certifies that this screen found no separation below the requested threshold. Directly joined intervals and locally continuing arcs can approach arbitrarily closely on one continuous copper skin. Shorter arcs turning less than 90° receive the penetration test below. Their positive clearance, curvature, sweep conditioning, feed design and interlayer wiring require separate qualification.
* **Contact at any station.** The skin check includes same-station and adjacent-station triangles. It discriminates the legitimate common vertex or edge from a segment crossing through both triangle interiors, a coplanar positive-area overlap, or extra line contact away from a single common vertex. A nonincident triangle must be clear even if its station lies nearby. Inspecting all station contacts catches local path folds and twisted profile rings.
* **Tolerances.** Copper contact uses a 1 pm numerical tolerance. The insulation requirement uses a 0.1 nm acceptance tolerance for the exported facets. Positive subminimum facet pairs are separated by more than 1 pm but below the required gap after the acceptance tolerance. Each witness records both closest XYZ points, their facet and station indices, and their interval owners. Facet-pair counts depend on tessellation. They do not count distinct physical crossing sites.
* **Assembly scope.** Interchannel checks use the exported body meshes in their common GUI world coordinates. Each body pair uses all facets with no station exclusions. Conductor geometry must exist to be measured. Paths omitted by the design, including unswept external feed or interlayer bridges, have no clearance result.
