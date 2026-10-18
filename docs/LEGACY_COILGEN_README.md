> **CoilGen2 is now available: [github.com/Philipp-MR/Coilgen2](https://github.com/Philipp-MR/Coilgen2).**
> **This legacy CoilGen repository and its legacy versions are no longer maintained.**
> Please use CoilGen2 for new work. This repository remains available for
> reproducing earlier MATLAB-based studies.

# CoilGen — legacy MATLAB version

CoilGen generates coil winding patterns for MRI/NMR and magnetostatic
field-shaping studies. Specify a target field, such as `Bz(x,y,z) = y` for
a constant gradient in the y direction, and a triangular support surface,
such as a cylindrical STL mesh. The boundary-element stream-function method
produces a current distribution and postprocesses its contours into a
coil layout.

## Continue with CoilGen2

[CoilGen2](https://github.com/Philipp-MR/Coilgen2) is the current Python
development, with a native desktop workbench, a class-based scripting API,
generated and imported support surfaces, constrained optimization, and field
and engineering diagnostics.

- [Installation](https://github.com/Philipp-MR/Coilgen2/blob/main/docs/INSTALL.md)
- [Getting started](https://github.com/Philipp-MR/Coilgen2/blob/main/docs/GETTING_STARTED.md)
- [Desktop workbench](https://github.com/Philipp-MR/Coilgen2/blob/main/docs/NATIVE_WORKBENCH.md)
- [Examples](https://github.com/Philipp-MR/Coilgen2/blob/main/docs/EXAMPLES.md)

CoilGen2 has its own implementation and validation workflow. When reproducing
a historical result, record the exact legacy revision, MATLAB version,
parameters and inputs used.

## Run a legacy example

1. Download or clone this repository.
2. Open the checkout in **MATLAB R2020a or newer**.
3. Run an example from the `Examples` folder and edit its target field,
   geometry and design parameters as needed.

**FastHenry2** is optional for the legacy inductance calculation. The original
CoilGen publication describes the method and its postprocessing workflow.

The legacy README flagged a field-scale correction in
`plot_various_error_metrics`. Take care when reproducing figures from older
revisions and retain the code version with the reported results.

## Related Python implementation

The original author also supported **pyCoilGen**, a separate Python translation
of the legacy project: [source](https://github.com/kev-m/pyCoilGen) and
[documentation](https://pycoilgen.readthedocs.io/). Its release and maintenance
status are managed by that project.

## Citation

Please cite the original work when using CoilGen:

> Amrein P, Jia F, Zaitsev M, Littin S. **CoilGen: Open-source MR coil layout
> generator.** *Magnetic Resonance in Medicine.* 2022;88(3):1465–1479.
> [doi:10.1002/mrm.29294](https://doi.org/10.1002/mrm.29294).

## License and contact

See `LICENSE.txt` in this legacy repository for its license.

Philipp Amrein · [GitHub profile](https://github.com/Philipp-MR)
· [CoilGen2 project](https://github.com/Philipp-MR/Coilgen2)
