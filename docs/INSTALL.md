# Installation

CoilGen requires Python 3.10 or newer.

## Standard Install

Use this when you want to run the example scripts with plots and the optional
scientific solver stack:

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

Start the optional desktop interface with `python -m coilgen.gui` after installing
the `gui` extra. See [the native workbench guide](NATIVE_WORKBENCH.md) for project
save/reopen, independent channels and reproducible script export. Headless users
can run `python -m coilgen --help` with no Qt or VTK installation.

If `python` is not on your PATH on Windows, install Python from
https://www.python.org/downloads/ and enable the installer option that adds
Python to PATH.
