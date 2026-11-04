# Winding results by channel

The updated templates fitted **1044 of 1049 extracted loops**, including 262 accepted local fits. Across 29 mounted windings, P95 turning at 3 mm spacing fell for 28, changed by at most 0.01° for 1, and rose for 0. Pass counts: independent sheet 29/29, pipeline sweep 0/29, fabrication review 0/29.

Each arrow compares the previous published gallery to the updated template at each run’s original current. Angles measure direction change at equal 3 mm arclength stations. Bz RMS is the relative design-DSV error of the mounted winding. A passing sheet does not qualify a conductor. The [before/after screenshot comparison](README.md) uses the user’s exact original job; it is distinct from the published reference used in this table.

[Contours and model fingerprints](gallery.csv) · [Final winding metrics and geometry hashes](full-gallery.csv) · [Saved mounted wires](reviewed_centerlines/)

## Gradients

| Channel | Fits / loops (local) | Before routing: P95 ° | Mounted: P95 ° | Mounted: Bz RMS % | Min reviewed bend, mm | Sweep / fabrication |
|---|---:|---:|---:|---:|---:|---|
| Cylindrical X gradient (Gx) | 32 / 32 (0) | 17.99 → 9.55 | 21.18 → 14.95 | 4.117 → 4.124 | 1.081 | fail / fail |
| Cylindrical Y gradient (Gy) | 32 / 32 (0) | 17.93 → 9.61 | 21.13 → 15.33 | 4.121 → 4.125 | 1.248 | fail / fail |
| Cylindrical Z gradient (Gz) | 32 / 32 (0) | 9.79 → 6.84 | 10.32 → 7.14 | 4.304 → 4.210 | 2.002 | fail / fail |
| Halbach transverse-field X gradient (Gx) | 32 / 32 (0) | 13.33 → 8.49 | 17.45 → 14.16 | 5.821 → 5.674 | 1.390 | fail / fail |
| Halbach transverse-field Y gradient (Gy) | 32 / 32 (0) | 16.52 → 9.85 | 20.30 → 14.63 | 4.419 → 4.349 | 0.882 | fail / fail |
| Halbach transverse-field Z gradient (Gz) | 32 / 32 (0) | 11.66 → 8.28 | 16.60 → 12.96 | 5.661 → 5.390 | 1.347 | fail / fail |
| Actively shielded Y gradient (Gy) | 40 / 40 (28) | 15.93 → 12.78 | 19.95 → 17.58 | 4.426 → 4.438 | 0.891 | fail / fail |
| Biplanar X gradient (Gx) | 32 / 32 (8) | 18.61 → 13.05 | 20.21 → 16.94 | 11.880 → 11.876 | 1.613 | fail / fail |
| Biplanar Y gradient (Gy) | 69 / 72 (25) | 27.04 → 20.08 | 27.10 → 23.95 | 4.191 → 4.097 | 0.306 | fail / fail |
| Biplanar Z gradient (Gz) | 34 / 34 (6) | 14.26 → 9.88 | 15.63 → 11.95 | 4.008 → 3.975 | 0.620 | fail / fail |

## Shim coils

| Channel | Fits / loops (local) | Before routing: P95 ° | Mounted: P95 ° | Mounted: Bz RMS % | Min reviewed bend, mm | Sweep / fabrication |
|---|---:|---:|---:|---:|---:|---|
| Uniform B0 shim (Z0) | 12 / 12 (0) | 9.77 → 5.26 | 10.34 → 5.50 | 4.215 → 4.133 | 1.771 | fail / fail |
| Second-order shim — Z2 | 32 / 32 (0) | 9.81 → 9.81 | 10.26 → 10.26 | 32.239 → 32.237 | 1.680 | fail / fail |
| Second-order shim — XZ | 12 / 12 (0) | 12.28 → 8.36 | 17.55 → 14.49 | 4.629 → 4.656 | 1.331 | fail / fail |
| Second-order shim — YZ | 12 / 12 (0) | 12.42 → 8.26 | 17.45 → 13.75 | 4.620 → 4.674 | 1.275 | fail / fail |
| Second-order shim — X²−Y² | 48 / 48 (0) | 25.19 → 11.23 | 28.62 → 19.81 | 5.044 → 5.019 | 2.724 | fail / fail |
| Second-order shim — XY | 48 / 48 (0) | 24.85 → 11.39 | 28.26 → 18.87 | 5.311 → 5.206 | 2.753 | fail / fail |
| Conical Z2 | 35 / 35 (7) | 9.22 → 7.16 | 10.07 → 7.61 | 21.130 → 5.461 | 0.273 | fail / fail |

## Coil systems

| Channel | Fits / loops (local) | Before routing: P95 ° | Mounted: P95 ° | Mounted: Bz RMS % | Min reviewed bend, mm | Sweep / fabrication |
|---|---:|---:|---:|---:|---:|---|
| Cylindrical XYZ gradient system (Gx, Gy, Gz): Cylindrical Gx | 32 / 32 (0) | 17.99 → 9.55 | 21.18 → 14.95 | 4.117 → 4.124 | 1.081 | fail / fail |
| Cylindrical XYZ gradient system (Gx, Gy, Gz): Cylindrical Gy | 32 / 32 (0) | 17.67 → 9.20 | 20.83 → 13.97 | 4.079 → 4.092 | 0.984 | fail / fail |
| Cylindrical XYZ gradient system (Gx, Gy, Gz): Cylindrical Gz | 32 / 32 (0) | 9.57 → 6.75 | 10.16 → 7.04 | 4.227 → 4.294 | 2.653 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar Gx | 32 / 32 (14) | 21.30 → 11.89 | 21.37 → 15.94 | 11.872 → 11.868 | 2.017 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar Gy | 62 / 64 (26) | 24.84 → 15.23 | 23.87 → 20.60 | 4.115 → 4.115 | 0.389 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar Gz | 34 / 34 (10) | 17.16 → 10.38 | 19.28 → 13.85 | 4.187 → 4.113 | 1.116 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar Z0 | 24 / 24 (10) | 17.50 → 10.01 | 21.08 → 13.17 | 4.144 → 4.033 | 1.651 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar Z2 | 52 / 52 (50) | 29.32 → 26.62 | 34.95 → 33.72 | 6.277 → 6.389 | 0.305 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar XZ | 26 / 26 (4) | 22.32 → 14.26 | 26.71 → 17.88 | 7.769 → 7.729 | 0.511 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar YZ | 32 / 32 (8) | 24.45 → 14.69 | 33.21 → 21.77 | 7.309 → 4.060 | 1.000 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar X²−Y² | 72 / 72 (58) | 28.01 → 24.82 | 33.00 → 30.89 | 13.244 → 10.883 | 0.206 | fail / fail |
| Biplanar gradients + second-order shims: Biplanar XY | 48 / 48 (8) | 23.29 → 13.18 | 26.88 → 19.65 | 7.503 → 7.571 | 0.606 | fail / fail |
