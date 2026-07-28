# Contributing to CoilGen

CoilGen changes are evaluated first for physical and numerical correctness, then
for software quality. Refactors must preserve field maps, stream-function
solutions, contour topology, and hardware metrics unless the change explicitly
documents a new mathematical contract.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the same checks used by continuous integration:

```powershell
python -m ruff check src tests projects
python -m ruff format --check src tests projects
python -m pytest -q
```

Use `python -m ruff format src tests projects` to format edited Python files.

## Numerical change requirements

- State all coordinate frames and use SI units at module boundaries.
- Document objective terms, scaling, constraints, solver tolerances, and
  regularization choices.
- Add a focused regression test for every corrected numerical failure.
- Compare field error, current density, resistance/inductance, contour count,
  minimum spacing, and routed-wire residuals where applicable.
- Treat changes to mesh orientation, boundary handling, symmetry reduction, or
  stream-function gauge selection as mathematical changes, not formatting.
- Keep solver-independent baselines separate from joint or constrained
  solutions so comparisons remain fair.

Generated plot books and other run artifacts belong in the repository-level
`outputs/<project-name>/<date-time-run>/` tree and are not source files. Start
a new programmatic run with `begin_project_run(project_name)` when several runs
share one Python process; normal project-script and Spyder executions are
detected automatically. Use `latest_outputs/...` only for explicit dependencies
on an artifact from an earlier run.

## Project scripts

Runnable scripts under `projects/` are engineering control panels. Group related
settings, include units in names, and delegate implementation to reusable
project-support or `coilgen` modules. A new study variant should override a
canonical workflow configuration instead of copying the workflow.
