# Literature-Backed Design Insights

CoilGen should be treated as a general magnetostatic coil-design tool. The useful
core is broad: solve inverse magnetostatic design
problems on arbitrary support surfaces, convert surface-current solutions to
inspectable conductor layouts, and report enough physics and manufacturing
diagnostics to decide what should be engineered next.

## What The Literature Suggests

### Keep The Stream-Function Mesh Core General

Modern open coil-design tools use stream functions on triangle meshes because
they handle arbitrary surfaces without relying on cylindrical or planar analytic
bases. bfieldtools is a useful external reference point: it presents a Python
surface-current toolkit built around stream-function discretization on
triangular meshes, with linear field maps and quadratic energy/power metrics.

Implications for CoilGen:

- keep `Mesh`, `TargetField`, and `CoilDesignProblem` application-neutral;
- support arbitrary generated and imported surfaces as first-class inputs;
- keep field maps, current density, resistance, inductance, and reports exposed
  through stable APIs;
- consider a future surface-harmonic or reduced-basis stream-function option for
  smoother current patterns and lower optimization dimension.

Primary source: Zetter et al., *Magnetic-field modeling with surface currents:
Implementation and usage of bfieldtools*, arXiv:2005.10056,
https://arxiv.org/abs/2005.10056.

### Optimize Current Density, Not Only Resistance

Gradient and shim coils are constrained by field accuracy, efficiency,
inductance, heat, and manufacturability. The minimax current-density literature
is especially relevant because limiting peak current density spreads winding
density, improves practical wire spacing, and reduces localized Joule heating.

Implications for CoilGen:

- add a peak-current-density or minimax-current-density optimization mode;
- report current density in `A/mm^2` everywhere current and conductor dimensions
  are known;
- add duty-cycle-aware thermal estimates: RMS current, average power, copper
  temperature-rise proxy, and warning thresholds;
- add plots that show local current-density hot spots before contour extraction.

Primary source: Poole et al., *Minimax Current Density Coil Design*, Journal of
Physics D: Applied Physics 43, 095001, 2010,
https://arxiv.org/abs/1002.3413.

### Prefer Constraint-Based Study Comparisons

For geometry sweeps, the design with the lowest current is not meaningful unless
field quality and secondary constraints are matched. Recent local-gradient work
uses constrained optimization to make design tradeoffs explicit, for example by
limiting gradient-magnitude variation while allowing other degrees of freedom.

Implications for CoilGen:

- keep `constrained_quadratic` and fixed-error efficiency modes as the default
  for fair geometry comparisons;
- generalize constraints beyond DSV Bz error and suppression RMS to support
  vector-field constraints, local-gradient constraints, and application-specific
  slice or volume constraints;
- report achieved constraint usage next to efficiency, current, resistance, and
  inductance, so users can see when a design is spending the full error budget.

Primary source: Jia et al., *Design of a non-linear human breast diffusion coil*,
Journal of Magnetic Resonance 331, 107052, 2021,
https://arxiv.org/abs/2012.12813.

### Support Nonstandard B0 Orientations And Field Components

Portable and open MR systems can have transverse or otherwise nonstandard static
field orientations. Halbach gradient-coil work shows that target fields and
usable gradient components must be defined relative to the actual B0 direction,
not assumed to be scanner-z by default.

Implications for CoilGen:

- keep the convenient `target_bz_expression` API for conventional MRI;
- add a more general `target_field_component` or `b0_direction` workflow where
  the useful field is `B dot b0_hat`;
- evaluate gradient linearity along the requested physical encoding direction,
  not only fixed Cartesian Bz;
- document when `Bz` is a modeling convention and when it is a physical scanner
  axis.

Primary source: de Vos et al., *Gradient Coil Design and Realization for a
Halbach-Based MRI System*, arXiv:1912.08500,
https://arxiv.org/abs/1912.08500.

### Separate Static Coil Design From Time-Domain Gradient Operation

CoilGen is currently magnetostatic: it designs conductors and evaluates fields,
resistance, and inductance. Finished MRI gradient design also needs time-domain
operation checks: amplifier voltage, slew rate, PNS, acoustic response,
mechanical resonance, and eddy currents. Recent waveform optimization work shows
that PNS, back-EMF, acoustic noise, and field oscillation are waveform-dependent
and should be evaluated after the coil geometry exists.

Implications for CoilGen:

- keep static coil design and waveform design as separate layers;
- export coil electrical parameters and field maps in a form usable by waveform,
  PNS, and acoustic tools;
- add approximate PNS/E-field screening hooks based on vector-potential maps;
- add explicit report language that `PASS` means a static CoilGen quality pass,
  not regulatory or clinical MRI safety approval.

Primary sources:

- Roemer et al., *Electric Field Calculation and PNS Prediction for Head and
  Body Gradient Coils*, arXiv:2012.08694,
  https://arxiv.org/abs/2012.08694.
- McCready et al., *OPTIKS: Optimized Gradient Properties Through Timing in
  K-Space*, IEEE Transactions on Medical Imaging, 2025,
  https://arxiv.org/abs/2505.07117.

## Current implementation and remaining work

General field components, current-density and concomitant-field reports, and
constrained minimax objectives are implemented. See
[Physical Assumptions](PHYSICAL_ASSUMPTIONS.md),
[Stream-function optimization](STREAM_FUNCTION_OPTIMIZATION.md) and
[Headless physical design](RESEARCH_PHYSICS_API.md) for their scope and limits.

Duty-cycle/cooling models, complete terminals and leads, winding order, and
validated downstream waveform/PNS/eddy-current analysis remain engineering
work. The [Engineering Roadmap](IMPROVEMENT_ROADMAP.md) keeps those open items
and their validation criteria.

## Documentation Rule

When documenting results, distinguish these levels:

- `magnetostatic feasibility`: field target can be generated on the support
  surface with reasonable current, resistance, inductance, and current density;
- `layout feasibility`: contours have acceptable spacing, smoothness,
  interconnects, and clearance diagnostics;
- `engineering feasibility`: thermal, voltage/slew, PNS, acoustic, force,
  eddy-current, and terminal-routing checks have been completed;
- `clinical/device readiness`: safety, regulatory, QA, and system integration
  evidence exists.

CoilGen currently aims to be strong at the first two levels and to export enough
diagnostics to support the third. It should not present a static field design as
a final MRI device design.
