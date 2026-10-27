# ISMRM 2027 abstract figures and evidence

These figures and results come from the CoilGen2 ISMRM 2027 abstract package.
Numerical results use **v26**; the gradient summaries below are derived from its
saved fields. They are not a benchmark of the current checkout. Both PNGs are
retained byte for byte; no field maps were redrawn.

| Local figure | Abstract source |
| --- | --- |
| [workspace.png](workspace.png) | v26 Figure 1: graphical workspace and processing sequence |
| [coil-examples.png](coil-examples.png) | Original v25 Figure 5 four-panel image, reused in v26; results are presented as a separate Markdown table |

## Field evaluation through the design stages

Values below are **relative RMS Bz error (%)**:
`100 × RMS(Bz − target Bz) / RMS(target Bz)`.
Each design uses the same 2,048 independent volume samples across all stages,
with its saved stream function and signed currents held fixed. No gain, offset
or current is fitted. Sampling uses a scrambled Sobol sequence (seed 20271017)
within an origin-centered sphere: radius 20 mm, or 50 mm for the conical shim.

| Design | Surface current | Extracted | Filtered | Smoothed | Connected | Mounted centerline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shielded y-gradient | 4.052 | 3.826 | 3.826 | 3.829 | 3.840 | 4.432 |
| Biplanar x-gradient | 4.212 | 4.188 | 4.188 | 4.189 | 4.193 | 11.766 |
| Cylindrical X²−Y² shim | 3.812 | 3.740 | 3.740 | 3.741 | 3.827 | 4.572 |
| Conical Z2 shim | 4.033 | 4.721 | 5.510 | 5.510 (off) | 5.712 | 10.577 |

“Mounted centerline” includes path realization and placement; the change from
the connected route cannot be attributed to mounting alone. Discrete fields use
thin-filament Biot–Savart evaluation. The surface-current stage uses order-4
triangle quadrature. Increasing volume samples to 4,096 changed these errors
by less than 0.005 percentage points.

## Gradient efficiency

The welcome-page table reports the achieved mounted-centerline gradient per
ampere, in **mT/m/A**. Both linear designs target **20 mT/m**. The achieved values
are derived from their saved fields at the original currents:

| Design | Achieved gradient (mT/m) | Gradient efficiency (mT/m/A) |
| --- | ---: | ---: |
| Shielded y-gradient | 19.471 | 1.759 |
| Biplanar x-gradient | 19.744 | 1.636 |

An equally weighted least-squares fit of `Bz = c + Gx*x + Gy*y + Gz*z` uses the
same 2,048 volume points and unchanged currents. Coordinates are divided by the
20 mm ROI radius before fitting, then coefficients converted back to T/m.
The table uses Gy or Gx, divided by the shared per-part current of 11.0714 A or
12.0707 A, respectively; currents are not summed across parts.
Increasing to 4,096 samples gives
19.471171 and 19.744088 mT/m. This fit only extracts gradient coefficients;
the original RMS errors above retain the full field, including its offset.
A first-order gradient does not characterize the two nonlinear shims.

## Second-order shim efficiency

For `Bz = C22*(x² − y²) + C20*(z² − (x²+y²)/2) + other modes`, coordinates are
in metres and C22/C20 have units T/m². Their efficiencies are coefficients per
ampere, in **mT/(m²·A)**; these are polynomial coefficients, not Hessian entries.

| Design | Target coefficient (mT/m²) | Achieved coefficient (mT/m²) | Efficiency (mT/(m²·A)) |
| --- | ---: | ---: | ---: |
| Cylindrical X²−Y² shim | 25 | 24.272 | 12.635 |
| Conical Z2 shim | 4 | 3.909 | 2.442 |

An equal-weight fit includes all nine real solid-harmonic modes through degree 2
on the same 2,048 volume points, using coordinates scaled by the ROI radius.
Saved currents are 1.9210534195 A and 1.6010144933 A. With 4,096 points, efficiencies
are 12.6348447733 and 2.4417779651 mT/(m²·A). These derived summaries preserve
the original fields and RMS errors. The stated Z2 polynomial fixes its normalization.

## Controlled contour smoothing

Meshes, contour levels, retained loops and currents were held fixed.

| Design | 3D RMS curvature reduction | RMS Bz change / target RMS |
| --- | ---: | ---: |
| Cylindrical X²−Y² shim | 61.3% | 0.00671% |
| Biplanar x-gradient | 39.8% | 0.00941% |
| Shielded y-gradient | 45.5% | 0.00737% |

Curvature is length-weighted 3D turning curvature at 0.5 mm arclength sampling;
it depends on sampling scale and mesh faceting. The last column measures the
field perturbation caused by smoothing, not the change in field-error percentage.

## Scope and provenance

These are numerical examples, not fabrication-qualified coils. The illustrated
elliptical shim sweep failed configured bend, clearance, frame and support
checks. Finite-cross-section current distributions and hardware performance
were not evaluated. Saved routing and placement are not a controlled routing trial.

[evidence.json](evidence.json) records exact saved metrics, derived gradients,
relative source paths, SHA-256 hashes and verification details. Paths are relative to the
supplied abstract package. Its large numerical archives, frozen software and
runtime are not duplicated here; this directory is a compact provenance record,
not a standalone reproduction package.
