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
