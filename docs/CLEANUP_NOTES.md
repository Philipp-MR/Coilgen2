# CoilGen Cleanup Notes

## Completed stabilization baseline

- Root test discovery is isolated from independently packaged project tests.
- [Quality CI](../.github/workflows/quality.yml) runs Python 3.10 on Linux and 3.12
  on Windows, installing the `dev`, `gui` and `plot-qt` extras. It checks Ruff
  lint/formatting for `src`, `tests` and `projects`, then runs root pytest
  discovery with headless plotting.
- Matplotlib backend selection has an independent module, removing the
  plot-book/result-visualization import cycle.
- SciPy-dependent MRI metrics fail only when requested; the dependency-light
  core remains importable.
- Constant-field targets cannot amplify floating-point linear-fit noise into an
  unphysical gradient normalization.

This repository now exposes `CoilProject` and project channels as the preferred
user-facing structure for multi-channel designs. The old generic workflow types
are no longer exported from `coilgen`; import them from `coilgen.workflow` only
for legacy batch/sweep tests or maintenance.

## Compactness cleanup

- Removed the unused six-button GUI workflow strip and its dedicated tests;
  current workbench navigation remains covered separately.
- Removed the generated-mesh dispatch duplicated after the factory registry,
  plus five unreferenced private helpers in routing, manufacturing and plotting.
  Public aliases, old-file readers and numerical legacy modes remain available.
- Consolidated script selection into [Examples](EXAMPLES.md), retained current
  array/basis conventions in [Core Concepts](CORE_CONCEPTS.md) and
  [Object Model](OBJECT_MODEL.md), and removed the obsolete MATLAB port plan.
- Reduced the [Engineering Roadmap](IMPROVEMENT_ROADMAP.md) to unresolved physical
  and validation work; implemented features link to their maintained API docs.

The pre-cleanup source/documentation snapshot is retained on
`codex/backup-before-application-cleanup-20261007`. MATLAB reference material,
topology studies, measured inputs and scientific results are preserved.

## Remaining cleanup candidates

- Retire `src/coilgen/workflow.py` only after its active batch/sweep callers are
  replaced with `CoilProject` plus explicit study-table utilities.
- Replace the shared `object_parameters` dictionary bridge with typed parameter
  dataclasses where this removes duplicate mappings and preserves script inputs.
- Keep the generated repository-level `outputs/` tree and `__pycache__/`
  directories out of version control; plot SVG galleries are now disabled by
  default.
