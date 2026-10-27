# CoilGen2 validation — 18 October 2026

This audit tests the Python library, application services, native desktop
workbench, packaging, and the examples used on the GitHub welcome page.
Execution success and engineering acceptance are recorded separately.

## Regression and installation checks

The local full regression passed **5,887 tests plus 29 subtests**, with zero
failures, errors or skips, in 1,794.60 seconds (29 min 54 s). It reported eight
identical Matplotlib `PendingDeprecationWarning` messages for project colormap
styling; there were no numerical or solver warnings.

That run was collected before the final cylinder, efficiency and affine-fit
corrections. The cylinder changes passed 274 affected tests plus 26 subtests;
the final units/conditioning changes passed 276 affected tests plus seven
pipeline cases. These focused groups overlap and are not additive totals.

The final published-main CI run completed successfully on **both platforms**.
Installation, Ruff lint, formatting, and the full regression suite passed:

| Platform | Passed tests | Passed subtests | Skipped | Python warnings | Test duration |
| --- | ---: | ---: | ---: | ---: | --- |
| Linux / Python 3.10 | 5,997 | 29 | 26 | 0 | 3,422.87 s (57 min 02 s) |
| Windows / Python 3.12 | 6,004 | 29 | 19 | 16 | 5,574.01 s (1 h 32 min 54 s) |

There were no failed tests or test errors. All 19 Windows skips require local
measured CSV or STL inputs unavailable on the hosted runner.
Linux has those same skips, plus five Windows-launcher tests and two
Windows-font layout cases. The 16 Windows warnings comprise eight project
Matplotlib colormap-styling deprecations and eight upstream Matplotlib Qt
high-DPI attribute deprecations; no numerical, solver or thread-exception
warnings were reported.

[Published-main CI run](https://github.com/Philipp-MR/Coilgen2/actions/runs/39164628852)
tested commit
[`80f0361`](https://github.com/Philipp-MR/Coilgen2/commit/80f03616bb4ec4d4834090321342a98098a46ed8).
Its application, tests, projects, package configuration and CI workflow are
unchanged from the final numerical/application revision below. The earlier
[final-source CI run](https://github.com/Philipp-MR/Coilgen2/actions/runs/39159035609)
also completed successfully on both platforms. This final evidence update
changes only this report and replaces the Windows-pending note from the
initial audit closure. No additional source changes were needed.

The focused GUI audit passed **183 tests** in 56.20 seconds, including real
artifact/CAD publication, result ownership, asynchronous loading, and worker
shutdown. The CLI stage-override regressions passed with and without a saved
gradient-engineering study. Ruff lint and formatting checks passed for all
682 Python files under `src`, `tests`, and `projects`. All those files also
parsed under an isolated CPython 3.10.22 runtime, where the four failure-evidence
tests also passed without skips or warnings. Other local runtime testing used
Python 3.12.14 on Windows. The repository CI additionally specifies Python 3.10
on Linux.

The final numerical/application revision is
[`ff789ee`](https://github.com/Philipp-MR/Coilgen2/commit/ff789eef7975786963bb5a35b7af684ba81bb4af).
The wheel built and installed into a separate target directory; all **227**
packaged Python modules matched the source byte for byte. A headless
prepared-stage project executed from that installed package while an import
hook rejected Qt/VTK imports. The packaged GUI image assets were present.
`python -m pip check` passed. Documentation links were checked locally.
The audited wheel's SHA-256 was
`d8f6ec6afdba867d6c2e54b436de9247d195c1eeb45be7cd320390296a0530a9`.
The exact Python example in the welcome page completed in 141.0 seconds on
that revision; its realized support bounds were 240 × 240 × 300 mm and all
stream-function values were finite.

A separate CPython 3.10.22 environment containing NumPy 2.2.6 and pytest,
with no SciPy, CVXPY, Numba, Qt, VTK or Matplotlib available, completed a real
small public `CylinderCoilDesign` through contours using the reference kernel
and unconstrained Tikhonov optimization. Its 84-vertex/144-triangle support,
19 ROI samples and eight contours had finite stream functions, sheet fields
and contour coordinates; the requested 300 mm physical length was realized.
Repeating with the declared NumPy floor, **1.24.4**, also passed that contour
solve, 53 focused finite-wire/ROI/affine-fit/cylinder tests, and the independent
analytical physical checks, with no skips, errors or warnings.

## Defects corrected

- **Exterior finite-wire field conditioning.** The endpoint expression
  subtracted cosines close to ±1 near a segment's exterior axis. At a 10 nm
  radial offset, one reproduced case had about 3.6% error; at 0.1 nm the
  field collapsed to zero. An algebraically equivalent rationalization
  preserves the small nonzero field. Independent 60-digit decimal reference,
  reversed-current, and segment-subdivision checks pass. Interior and
  singularity conventions are retained. Chunk sizing accounts for the added
  intermediates; small-memory serial/parallel results agree to 5.4 × 10⁻¹⁶
  relative L2 in the recorded check.
- **Shielding comparison sample counts.** Unequal shielded/unshielded arrays
  previously truncated silently, allowing misleading suppression percentages.
  They now raise an error; the caller must supply the same ordered grid.
- **Uniform-field efficiency units.** Constant Bz targets previously displayed
  efficiency in gradient units. Their reports now use mT/A from the mean DSV
  field per ampere, excluding suppression samples. Target normalization does
  not change this physical efficiency; fitted currents and errors are retained.
  Gradient-only efficiency keys are unavailable for a uniform target.
  Reported gradient efficiencies use the physical DSV target's resolved
  affine gradient and fitted current, covering Hz and imported targets whose
  normalization differs from the requested input scale. A nonlinear target
  without a resolved affine gradient does not publish gradient efficiency.
  A spatial-rank check also omits derivatives that cannot be identified from
  the supplied sample geometry, while retaining identifiable in-plane slopes.
- **Affine-fit conditioning.** Centering coordinates and Bz separates the
  intercept from the target-gradient fit. A reproduced 1,000 km world-coordinate
  offset previously mixed a 0.03 T/m Gy target into spurious x/z components
  and a 0.02 T/m Gy estimate. The centered fit recovers the intended derivative.
  The affine model is unchanged; ill-conditioned target normalization can
  consequently produce a different, corrected scale.
- **Empty generated target regions.** Legacy spherical grids at resolutions
  1, 2 or 4 contain no points inside the ROI. They previously reached expression
  evaluation with a malformed empty array and raised `IndexError`. Target
  preparation now raises an actionable `ValueError` before evaluation, asking
  for a finer grid or an adjusted target region. Nonempty sample locations
  and all downstream numerical behavior are retained; no samples are invented.
- **Public cylinder dimensions.** `CylinderCoilDesign.length_m` and
  `PipelineGeometrySpec.cylinder` now realize the requested physical length
  independently of axial mesh resolution. Previously, a requested 300 mm
  cylinder spanned 240 mm at four axial divisions and 292.683 mm at forty.
  The conversion compensates the legacy generator's parameter convention;
  raw `CoilGenConfig` tuples and saved configurations retain their geometry.
  Rerunning an earlier high-level script therefore changes its cylinder to
  the requested length and can change its optimized winding and field.
- **CLI partial-stage execution.** `--stage prepared`, `sheet`, `contours`, or
  `routed` now disables the full fabrication and gradient-engineering reviews
  for that invocation. The saved full-design study is preserved.
- **GUI worker lifetime.** Background preview, array, export, and saved-settings
  tasks can finish after their Qt signal sources have been destroyed. Other
  runtime and numerical errors retain their existing handling.
- **Legacy result display.** Routing availability accepts earlier/injected
  records with an array mapping and retains the name-only path for current
  records, avoiding unnecessary array loading.
- **Portable legacy project references.** Windows drive, rooted and UNC paths
  were treated as relative filenames on POSIX. Asset and collection loading
  now apply both platforms' path rules before resolving and checking project
  containment. Valid relative references, including Windows separators, remain
  portable. The affected project/bundle tests passed on the local platform.
- **Qt binding compatibility.** Linux/Python 3.10 reproduced a native
  `none_dealloc: deallocating None` abort during Qt property/color updates with
  PySide6/shiboken 6.12.0. An identical-source comparison passed all 34 native
  workbench tests with 6.11.2. Qt extras therefore require a version below 6.12.
  [Controlled comparison](https://github.com/Philipp-MR/Coilgen2/actions/runs/39153508782).
- **Interpolation regression precision.** A Linux result differed from a
  theoretically zero field component by 1.39 × 10⁻¹⁷ T. The regression now
  checks the affine field at all nine valid samples with tight, scale-aware
  epsilon tolerances. Convex-hull, off-plane and unavailable-sample checks
  remain exact; the production interpolation is unchanged.
- **Optional spatial dependency.** Explicit refined geodesic routing reports
  how to install SciPy when missing. Ordinary graph distances and default
  routing retain their dependency-light behavior.
- **GitHub Actions startup.** `runner.temp` was referenced in workflow-level
  environment variables, where the runner context is unavailable. It is now
  evaluated in the test step's environment, following GitHub's
  [context-availability rules](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability).
- **Formatter reproducibility.** Development installs pin Ruff 0.15.22, the
  version used to verify the source. An unconstrained newer release introduced
  Markdown formatting into the CI command and disagreed with the local check.
- **Cross-platform CI environment.** Python checkouts explicitly use LF to
  match Ruff on Windows. Linux CI installs the EGL/OpenGL runtime libraries
  required to import the Qt/VTK stack during headless tests.
- **Portable regression expectations.** Failure-evidence tests retain the exact
  scientific exception on Python 3.10 and check exception notes when that API is
  available. Launcher tests verify the actual interpreter and corresponding
  diagnostic, covering both a checkout's `.venv` and Python on PATH. The full
  worker completion/evidence test allows 180 seconds plus observer grace on
  hosted runners, retaining every stage, geometry, archive and identity check.
  Numerical and engineering acceptance limits are unchanged.
- **Figure geometry and reproducibility.** The welcome-page generator verifies
  actual support dimensions and compensates the legacy cylinder builder's
  height convention. It records inputs, currents, solver status, source
  fingerprint, dependency versions, arrays, and artifact checksums. SVG output
  uses canonical LF line endings; recorded checksums match Git's staged blobs,
  including on Windows.

## Independent physical checks

Run `python validation/physical_invariants.py` after installing the package.
[The saved evidence](../validation/physical_invariants.json) includes SI scaling,
rigid transforms, constant stream-function gauge, surface-current direction,
resistive power, scalar/vectorized sensitivity, and analytical field checks.

| Check | Observed result |
| --- | --- |
| Circular-loop segment refinement, three axial locations | Approximately 16-fold error reduction for each 4-fold segment increase; final worst error below 0.000056%. |
| Uniform rectangular current sheet vs analytical solid angle | Relative field error below 7.4 × 10⁻¹³ across the tested scales and transforms. |
| Maxwell-pair fabrication reference | Strict geometry checks and fixed-current gradient validation passed on 913 independent ROI points. |
| Solenoid analytical circular-loop stack | Center-field error 0.0732%; inductance error 1.91%. |
| Welcome-page cylinder / biplanar examples | Independent sheet Bz relative RMS errors 3.404% / 3.014%; original-current unconnected contours 3.381% / 2.959%. |

The Maxwell example uses two independently closed turns, with no designed
drive feeds. The solenoid uses the example's deterministic ordered winding.
The welcome figures use unconnected contours. These scopes are distinct.
See [figure inputs and evidence](frontpage/README.md).

## Native workbench integration

A real Windows Qt/VTK `ScientificViewport` was exercised with a spawned
numerical worker. The audit loaded the cylindrical Gy template, computed
contours, rendered a nonuniform VTK image, saved/reopened a compact `.coilgen`
project with matching model identity, and compiled the exported Python script.
It then completed routing, conductor sweeping, and full evaluation in
124.5 seconds on the final code revision. No mock viewport or solver was used
for this check.

The exact exported Python script and JSON sidecar also executed headlessly
in 7.28 seconds with Qt/VTK imports blocked. The replay retained the saved
contours stage and matching project/model fingerprints. All 47 scientific
arrays were finite; its stream function, sheet field and independent validation
field matched the native run exactly (zero relative L2 difference). The
sidecar's input hash was unchanged.

The full run's continuous sheet passed its independent field check
(3.109% relative Bz error). **The final winding was rejected for fabrication.**
Recorded failures include route/sweep clearance, self-intersection, minimum
bend radius, curvature envelope, unresolved frame sampling, and support
penetration. The run remained `accepted=false` and `construction_released=false`.
This template is an engineering starting point; its successful execution
does not make the resulting winding suitable for construction. This audit
does not resolve that winding-design problem or relax its acceptance limits.

## Reproduce

From an activated environment in the repository root:

```bash
python -m pip install -e ".[dev,gui,plot-qt]"
python -m ruff check src tests projects
python -m ruff format --check src tests projects
python -m pytest -q
python validation/physical_invariants.py
python examples/research_valid_gradient_fabrication.py
python projects/general_examples/scripts/solenoid_constant_z_validation.py --headless
python docs/frontpage/generate_figures.py
```

For headless tests set `QT_QPA_PLATFORM=offscreen` and `MPLBACKEND=Agg`, and
limit native numerical thread pools to one for comparable timings. In this
Windows environment, actual VTK rendering required the desktop Qt platform;
the offscreen platform could not create a valid OpenGL context. Restricted
sandbox TEMP and hard-link permissions also caused environmental test errors;
the successful complete run used normal filesystem permissions and writable
temporary directories.

This audit does not establish optimized-mesh or quadrature convergence for
every design, reproduce external measured-data studies, or qualify thermal,
mechanical, RF, transient, PNS, or clinical behavior.
