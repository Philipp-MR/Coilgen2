# CoilGen2

**Interactive design and field evaluation for MR gradient and shim coils.**

Define a support surface and target field, optimize the surface current, and
turn its contours into a winding. CoilGen2 brings these steps into a Python
library, a desktop workbench and a browser workbench with the same numerical engine.

[Installation](#installation) ·
[Desktop guide](../docs/NATIVE_WORKBENCH.md) · [Browser guide](../docs/BROWSER_WORKBENCH.md) ·
[Examples](../docs/EXAMPLES.md) ·
[Documentation](../docs/README.md)

![Previous and updated X²−Y² shim at the same scale, showing the recorded mounted winding and a developed view of one lobe.](../validation/gui_winding_quality_2026-11-03/x2y2-reviewed-windings.png)

[GUI gallery: winding layouts, field comparisons and fabrication checks](../validation/gui_winding_quality_2026-11-03/README.md).

The table below reports **earlier ISMRM 2027 abstract designs**, with fields
evaluated on their mounted conductor centerlines. Their configurations and
numbers are separate from the current GUI presets.

| Example | ROI radius (mm) | Mode efficiency | Relative RMS Bz error (%) |
| --- | ---: | ---: | ---: |
| A · Shielded y-gradient | 20 | 1.759 mT/m/A | 4.432 |
| B · Biplanar x-gradient | 20 | 1.636 mT/m/A | 11.766 |
| C · Cylindrical X²−Y² shim | 20 | 12.635 mT/(m²·A) | 4.572 |
| D · Conical Z2 shim | 50 | 2.442 mT/(m²·A) | 10.577 |

Efficiencies are fitted Bz-mode coefficients of the mounted winding per ampere,
using the same current in each part. The linear examples target **20 mT/m**.
Shim coefficients multiply **x²−y²** (C) and **z²−(x²+y²)/2** (D), with coordinates
in meters; their efficiencies therefore use **mT/(m²·A)**.
[Figure sources and numerical results](../docs/frontpage/ismrm2027/README.md).

## Installation

On Windows, run `Setup CoilGen2.cmd` from the checkout, then open
`Start CoilGen2 GUI.cmd`. Choose **Browse examples… → Gradients → Cylindrical Y gradient (Gy)**, then **Compute full design**.
When the job finishes, use **Jobs → View results → Run design choices** to
inspect the model that was submitted. **Engineering checks** shows the separate
stream function, sweep and fabrication verdicts. For the browser, use
`Start CoilGen2 Browser.cmd` and follow the [first-run guide](../docs/BROWSER_WORKBENCH.md).

Requires **Python 3.10+** and Git. In a terminal:

```bash
git clone https://github.com/Philipp-MR/Coilgen2.git
cd Coilgen2
python -m venv .venv
```

Activate the environment:

| Platform | Command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Linux / macOS | `source .venv/bin/activate` |

Install the workbench and scientific dependencies, then launch:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[gui,plot]"
python -m coilgen.gui
```

Choose **File → Load template** to start with a gradient or shim design.
New gallery designs fit contours on the support with a 0.5 mm displacement
cap and a turn-spacing guard. If a whole-loop fit fails, eligible corners can
still be rounded within those limits. Compare **Contours before routing** with
**Review conductor centerlines** after a run; **Fit** shows the whole coil.
The connectors and mounted conductor have their own field, clearance and bend checks.
On Windows, subsequent launches can also use **`Start CoilGen2 GUI.cmd`**.
For scripts without the GUI, install `python -m pip install -e ".[research,plot]"`.
[Full installation guide](../docs/INSTALL.md).

## Design and inspect

- **Optimize:** generated or imported support meshes, analytic or imported
  target fields, suppression regions, regularization and constrained objectives.
- **Build the winding:** extract and smooth contours, connect loops, route
  returns, and set conductor cross-section and placement.
- **Evaluate** the fields from the stream function, contours and conductor centerlines.
  Inspect spacing, bend and clearance diagnostics, resistance and inductance estimates.

![Browser workbench showing the X²−Y² shim. Model Builder is on the left, the reviewed winding and field metrics are in the center, and the five completed design steps are on the right. Engineering checks failed.](../validation/gui_winding_quality_2026-11-03/browser-workbench-results.jpg)

The workbench and Python scripts share the numerical engine. Saved projects
retain editable geometry, targets and processing settings.

### Conductor cross-sections and sweeping

Edit the cross-section in millimeters, sweep it along the winding, and choose
centered, inner or outer mounting on the support. Profiles can be rectangular
or custom convex polygons, including elliptical and trapezoidal profiles.

[Illustrative circular-profile sweep meshes and bend details](../docs/frontpage/swept_bodies.svg) ·
[Profile editor](../docs/NATIVE_WORKBENCH.md) ·
[Sweep and mounting controls](../docs/PARAMETERS.md#volumetric-sweep-controls).

### Check the field after each geometry change

In the abstract's three controlled smoothing examples, 3D RMS curvature fell
**39.8–61.3%** at 0.5 mm arclength sampling. The smoothing-induced RMS Bz change
stayed below **0.010% of target RMS**, with the original signed currents retained.

Later operations mattered more: the biplanar example's relative RMS Bz error
rose from **4.193%** on the connected route to **11.766%** after path realization
and mounting. These comparisons use 2,048 independent volume samples per design,
without gain or offset fitting. [Methods, stage comparisons and provenance](../docs/frontpage/ismrm2027/README.md).

These are numerical examples. Discrete winding fields use thin-filament
Biot–Savart evaluation; the abstract's profile sweeps failed
configured geometric checks and are not ready for fabrication.
[Physical assumptions](../docs/PHYSICAL_ASSUMPTIONS.md) ·
[Interpreting results](../docs/RESULT_INTERPRETATION.md).

## Python workflow

```python
from coilgen import CylinderCoilDesign

design = CylinderCoilDesign(
    target_bz_expression="y",
    target_scale=0.01,             # 10 mT/m
    radius_m=0.12,
    length_m=0.30,
    target_region_radius_m=0.045,
    contour_levels=16,
)
result = design.run()
```

Geometry uses meters and fields use tesla. This is a starter design;
the abstract figures use separately saved configurations.
See [Getting started](../docs/GETTING_STARTED.md) for plots, reports and exports.

## Citation

Created and maintained by **Philipp Amrein**. CoilGen2 continues the original
CoilGen work by Philipp Amrein, Feng Jia, Maxim Zaitsev and Sebastian Littin.
Please cite the original paper and record the CoilGen2 version or commit:

> Amrein P, Jia F, Zaitsev M, Littin S. **CoilGen: Open-source MR coil layout
> generator.** *Magnetic Resonance in Medicine.* 2022;88(3):1465–1479.
> [doi:10.1002/mrm.29294](https://doi.org/10.1002/mrm.29294).

[Citation metadata](../CITATION.cff) ·
[Contributing](../docs/CONTRIBUTING.md) ·
[Contact](../docs/CONTACT.md) ·
[Report an issue](https://github.com/Philipp-MR/Coilgen2/issues)
