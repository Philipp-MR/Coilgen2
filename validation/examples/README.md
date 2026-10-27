# Example validation records

The [current inventory](current_inventory.json) is generated without running or
importing examples. Refresh it with `python validation/examples/inventory.py`.
Its hashes describe current source, not a passing test claim.

The 2026-10-11 existing-example audit is saved locally at
`outputs/example_validation_20261011/report.html`, with machine-readable scopes,
commands, overrides, failure evidence and environment hashes beside it. It covered
71 entrypoints before the subsequent fixes: 44 original defaults and 10 reduced
runs finished, 7 failed, 8 were time-limited and 2 partial. The 27 native GUI
workflows passed 1,007 assertions, while 26 strict fabrication reviews failed.
Fix-validation evidence is retained separately under `outputs/fix_validation_20261011/`.
The implementation report records 5,232 broad-suite passes and explicit passing
follow-ups for all 24 initial failures, including native filesystem retries and
final-source GUI regressions. Focused test counts overlap. The corrected biplanar
defaults now generate swept bodies and reach actual assembly rejection; failure
geometry remains available, and the standard runner continues to later cases.
These records do not qualify a full research grid, a fabricated clinical fixture
or a released driven coil.

## Historical cleanup audit

Validated on Windows with Python 3.12.14 on 2026-09-09. Exact dependency versions,
source fingerprints and the retained public runnable entrypoints are in [manifest.json](manifest.json).
Compact numerical results are in [numerical_checks.json](numerical_checks.json).

| Coverage | Outcome |
|---|---|
| 28 general/physical examples and benchmarks | Full demonstration defaults executed successfully from unrelated working directories. |
| Primitive annular disc, cylinder-smoothing sandbox, standard GX/GY/GZ project | Default numerical cases executed, with interactive plots disabled and bounded native threads. |
| Halbach GX | Actual imported support coarsened to 392 vertices; workflow completed, but manufacturing checks rejected the geometry. Visualization STL export is not fabrication acceptance. |

The reduced configurations and remaining limitations are recorded per entrypoint.
The scratch Halbach harness failed after the workflow completed because it called
a tuple property; its status is retained separately from the recovered project
summary. Timings are workload-specific and include normal run-to-run variation.
Large generated artifacts and full run logs remain locally under `outputs/` and
`tmp/`; they are not committed.

The final frozen-source regression suite passed **957 tests and 26 subtests** in
95.88 s with `MPLBACKEND=Agg`; 10 existing Matplotlib deprecation warnings remain.
The complete test output is in [pytest.txt](pytest.txt). Ruff passes across
`src`, `tests`, `examples` and `projects`. The wheel builds, dependency checks pass,
and wheel metadata includes the conditional Python 3.10 TOML fallback. Python 3.10
itself was not executed in this environment.

The tolerance and parallel examples also passed verified warm restart: 3/3 and
2/2 candidates reused, respectively, with unchanged candidate checksums. The
tolerance restart performed zero new operator/model assemblies. Cold/warm cache
benchmarks now distinguish actual computation from reuse.

For launch commands, dependencies and output interpretation, see [EXAMPLES.md](../../docs/EXAMPLES.md).
