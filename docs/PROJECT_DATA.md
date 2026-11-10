# Project Data Layout

Large or machine-local input assets are stored outside the source tree under
the repository-level `project_data/` directory. The complete directory is
excluded from Git.

```text
project_data/
├── general_examples/
├── primitive_support_structures/
├── standard_cylindrical_3axis_gradient/
├── halbach_cylindrical_3axis_gradient/
└── _shared/
    └── legacy_coilgen/
        ├── Geometry_Data/
        ├── Documentation/
        ├── Pre_Optimized_Solutions/
        └── target_fields/
```

Project configuration paths beginning with `input/`, `inputs/`, or `data/`
are resolved below that project's data directory. For example,
`inputs/support.stl` in the general examples resolves to
`project_data/general_examples/support.stl`.

Absolute input paths are accepted when a dataset must live outside the
repository. Generated files belong under the separate root `outputs/`
directory and must not be used as persistent source data unless a workflow
explicitly generates them as an intermediate result.

Every project execution writes to a unique run directory:

```text
outputs/<project-name>/<YYYY-MM-DD_HH-MM-SS-microseconds_timezone_pPID>/...
```

All artifacts from one execution share that directory, so repeated and
concurrent runs do not overwrite one another. A relative `outputs/...` path
means the active run. A relative `latest_outputs/...` path selects the newest
matching artifact from an earlier run and is reserved for staged workflows.
Both `outputs/` and `project_data/` are excluded from Git.

Because `project_data/` is not cloned from GitHub, datasets must be transferred
separately and restored with the same project-relative filenames.

## Restored runnable inputs (2 November 2026)

The Dropbox backup lacked derived inputs that the default workflows load. They
were copied unchanged from the working checkout
`C:\Users\Flipp\Documents\Codex\CoilGen2\project_data` (commit `6e3c2d9`).
No generator exists in the repository for the files marked *supplier remesh*;
keep them together with the original supplier surfaces.

| File (below `project_data/`) | SHA-256 | Origin |
|---|---|---|
| `halbach_cylindrical_3axis_gradient/Octagonal Gradients - Gx Surface_remeshed.stl` | `5a57bd10…063849` | supplier remesh of `… Gx Surface.stl` |
| `halbach_cylindrical_3axis_gradient/Octagonal Gradients - Gy Surface_remeshed.stl` | `a80f6e73…d134d` | supplier remesh of `… Gy Surface.stl` |
| `halbach_cylindrical_3axis_gradient/Octagonal Gradients - Gz Surface_remeshed.stl` | `e07389cc…f374db` | supplier remesh of `… Gz Surface.stl` |
| `dental_b0_local_shim/dental_shim_mask_onshape_8_12_2026_remeshed.stl` | `936c0dcc…24d335fc` | supplier remesh |
| `dental_b0_local_shim/dental_shim_surface 2_remeshed.stl` | `96b6f4bb…c46d132` | supplier remesh |
| `dental_b0_local_shim/*_symmetric_regular_3p8mm.stl` | see files | `symmetrize_phantom_mask.py` (gmsh) |
| `dental_b0_local_shim/phantom_0A_20260907/current_surface_gmsh_harmonized*_3p8mm.stl` | see files | `harmonize_phantom_mask.py` / `symmetrize_phantom_mask.py` (gmsh) |
| `dental_b0_local_shim/phantom_0A_20260907/current_surface_blender_decimated_ratio_0p15.stl` | `33ac3966…fba7f` | `decimate_phantom_surface_blender.py` (Blender) |
| `dental_b0_local_shim/phantom_0A_20260907/phantom_0A_registered.npz`, `phantom_0A_head_roi_*.csv`, `*validation.json`, `registered_*.png` | see files | `prepare_phantom_field.py` and `build_*_target.py`; the upstream `.mat` field export is **not** present in either copy |
| `dental_b0_multichannel/single_channel_contour_dictionary.npz` | `e403180b…fa11d6279` | no generator found |
