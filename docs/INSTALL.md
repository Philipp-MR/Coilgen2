# Installation

CoilGen requires Python 3.10 or newer.

## Browser workbench on Windows

1. From the CoilGen2 folder, run `Setup CoilGen2.cmd`. It creates or updates the shared environment at `%USERPROFILE%\.coilgen2\venv`. To choose another location, set `COILGEN_VENV` before running setup and use the same value when launching the workbench.
2. Open `Start CoilGen2 Browser.cmd` and leave its console open. The browser opens the local workbench on port 8765. The first-run walkthrough takes the cylindrical Y gradient from example selection through computation, wire inspection and project download. [Run your first coil](BROWSER_WORKBENCH.md#run-your-first-coil)
3. After moving or updating the checkout, close the hub and rerun setup from the new folder to refresh the source link and dependencies. To inspect setup options without installing anything, run `& ".\Setup CoilGen2.cmd" --help` in PowerShell.

The setup launcher checks the existing shared environment's base interpreter, the Windows `py` launcher, `python` on PATH and the usual Anaconda, Miniconda and Miniforge directories in your profile. If none supplies Python 3.10 or newer, install a supported Python and run setup again. [Python for Windows](https://www.python.org/downloads/windows/)

## Standard Install

Use this for the example scripts, plots and scientific solver stack.

```powershell
git clone https://github.com/Philipp-MR/Coilgen2.git
cd Coilgen2

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs CoilGen in editable mode using `requirements.txt`, which points
to the `full` dependency set in `pyproject.toml`.

In an activated environment, use the `web` extra for a manual browser installation.

```powershell
python -m pip install -e ".[web]"
python -m coilgen.web
```

## Minimal Install

Use this for dependency-light generated examples or non-interactive runs:

```powershell
python -m pip install -e .
```

The minimal install includes the CoilGen package and NumPy only. Matplotlib,
CVXPY, SciPy, and image-preview support are optional extras.

## Optional Extras

Install only the pieces you need:

```powershell
python -m pip install -e ".[plot]"     # Matplotlib/IPython plotting support
python -m pip install -e ".[gui]"      # Optional native Qt/VTK scientific workbench
python -m pip install -e ".[web]"      # Browser workbench, WebSocket hub and scientific workers
python -m pip install -e ".[solver]"   # CVXPY constrained optimization
python -m pip install -e ".[research]" # Physical objectives and headless studies
python -m pip install -e ".[accelerate]" # Cached compiled geometry kernels
python -m pip install -e ".[spatial]"  # SciPy spatial matching acceleration
python -m pip install -e ".[dev]"      # Tests and all common extras
```

The Qt extras select bindings below 6.12. Qt 6.12.0 reproduced a native
reference-counting abort on Python 3.10; the same Linux workbench tests pass
with 6.11.2. See the [validation record](VALIDATION_2026-10-18.md).

## Verify

From the repository root:

```powershell
python -c "import coilgen; print(coilgen.__name__)"
python projects/general_examples/scripts/conventional_cylindrical_mri_gradient.py
```

After installing the `web` extra, `python -m coilgen.web --help` lists local and team options.
After installing the `gui` extra, `python -m coilgen.gui` starts the desktop interface.
For a headless installation, `python -m coilgen --help` lists the command-line options.
[Browser workbench](BROWSER_WORKBENCH.md) · [Native workbench](NATIVE_WORKBENCH.md)
