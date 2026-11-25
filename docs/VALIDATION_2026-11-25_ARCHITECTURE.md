# Architecture validation — 25 November 2026

This change preserves the numerical model, optimization objectives, tolerances,
candidate order, acceptance limits, and routing budgets. It changes ownership,
cache identity, application boundaries, configuration metadata, and display access.

## Implemented changes

1. Automatic mesh and target references own deep snapshots with unique identities.
   Equal display names cannot overwrite a different design; snapshots are released
   when no configuration references them. Explicit named registry entries retain
   their public replacement and clearing behavior.
2. Completed results include acceptance/orchestration code in their cache identity.
   Numerical checkpoints retain a separate engine identity. Both cache keys include
   numerical and geometry backend versions; workers reject changed implementations.
3. Native document, execution, and result controllers coordinate the existing
   session model. Qt owns delayed editor callbacks so deleted editors cancel them.
4. Joint routing has explicit preparation, proposal, validation, adoption, and
   reporting methods within one transaction. Rejected proposals remain detached.
5. A shared configuration catalog owns group membership and stage dependencies;
   versioned migrations record historical defaults explicitly.
6. Structural result/job interfaces keep adapters independent of concrete clients.
   Checkpoints expose selected read-only display arrays while mutable state and
   resumed calculations retain independent ownership. PCB availability also accepts
   legacy metadata-only adapters without copying native read-only metadata.

Desktop and browser Python exports remain editable standalone scripts containing
resolved configuration, study settings, and owned input bytes. Importing a script
does not run it; execution uses the same application service as the interfaces.

## Numerical comparisons

The implementation was compared with a source snapshot taken before these six
changes, using the same Python environment and a single numerical thread.

| Comparison | Result |
| --- | --- |
| Cylindrical Gy Tikhonov, constrained power optimization, and a free-space physical sweep | 68 arrays identical in values, dtype, shape, and bytes |
| Complete cylindrical Gy pipeline including routing, realization, and sweep | 60 arrays identical in values, dtype, shape, and bytes |
| Stage dependencies across 381 fields and four configurations | 1,524 classifications unchanged |
| Extracted native controller methods, after reversing receiver renaming | 22 executable statement/callback ASTs unchanged |

Compared arrays include stream functions, fields, currents, contour/path geometry,
and physical sweep data. The routed fixture's existing negative manufacturing
verdict was preserved; numerical equivalence is not a manufacturing approval.
Timings from individual replay runs are not used as evidence of a speed improvement.

## Ownership and memory

Tests cover independent same-name designs, deep input copies including nested
metadata, lifetime through configuration copies and pickling, release after
repeated conversion, source/backend invalidation, legacy migrations, checkpoint
save/resume, read-only views, and native/browser session behavior.

For a synthetic float64 sensitivity operator with 2,000 nodes and 2,000 target
samples, `tracemalloc` peak access allocations after snapshot construction were:

| Access | Peak allocated bytes |
| --- | ---: |
| Detached full state | 96,240,256 |
| Detached display arrays | 233,740 |
| One copied field | 48,208 |
| One borrowed read-only field | 864 |

This measures access allocations, not solver memory or whole-process RSS. Snapshot
construction continues to own an independent full state. Read-only display views
are an API ownership contract, not a private-memory security boundary.

## Regression execution

The local workspace aggregate across split runs and focused rechecks is:

| Outcome | Count |
| --- | ---: |
| Passed | 6,773 |
| Skipped: native 3D drag requires an interactive OpenGL session | 1 |
| Not completed: standard-example biplanar physical sweep | 1 |

The full-suite process was stopped after 58 minutes while its
`test_standard_example_configs_smoke` case was still evaluating the biplanar
conductor. This is the suite process duration, not a measured duration of that
individual case. That benchmark is **not validated** by this run.

All eight standalone Python tests passed, including cylinder/biplanar numerical
parity, the object API, import without execution, CLI export, and relocation after
deleting original input files. These subprocesses actively reject GUI imports.
Desktop/browser project, export, revision, ownership, and worker tests passed.
The final source also passed 14 focused interface/source-identity/resume checks.

One Windows access violation occurred during a long physical test while timeout
traceback diagnostics were enabled; that test passed when rerun without the timeout
diagnostic. Windows directory locks in the Dropbox test location affected removal
of original bundle directories; both bundle relocation cases passed using an
unsynced temporary directory. The Python portability fixture now verifies deletion
of every input file without requiring deletion of its empty parent directory.

Validation used Windows 11, Python 3.12.16, NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3,
PySide6 6.11.3, `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`, and one numerical
thread. Detailed package versions, logs, source hashes, before/after NPZ data, and
reproducible comparison scripts are under `tmp/architecture_cleanup_20261125/`.

All changed Python files pass Ruff checks and formatting. Existing numerical
checkpoints need recomputation because the numerical source layout changed;
historical artifacts remain inspectable and projects remain loadable.
