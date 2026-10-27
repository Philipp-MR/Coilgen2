from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECT_SCRIPTS = ("primitive_support_structures/scripts/run_primitive_support_designs.py",)


@pytest.mark.parametrize("relative_path", DIRECT_SCRIPTS)
def test_direct_example_imports_from_an_unrelated_working_directory(tmp_path, relative_path):
    # run_name suppresses the expensive research solve, while preserving the
    # empty __package__ and sys.path behavior of a direct-file entrypoint.
    code = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='_entrypoint_import_')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "projects" / relative_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_halbach_preflight_checks_every_axis_before_starting_a_solve(tmp_path, monkeypatch):
    from projects.halbach_cylindrical_3axis_gradient import run_halbach_cylindrical_3axis_gradient as example

    project = deepcopy(example.PROJECT)
    for channel in project.channels:
        channel.design_overrides["mesh_file"] = str(tmp_path / f"missing_{channel.name}.stl")
    attempted = []
    monkeypatch.setattr(type(project), "solve", lambda *_args, **_kwargs: attempted.append(True))
    with pytest.raises(ValueError, match="preflight failed") as caught:
        example.main(project)
    for channel in project.channels:
        assert channel.name in str(caught.value)
    assert not attempted


@pytest.mark.parametrize("folder", ["standard_cylindrical_3axis_gradient", "halbach_cylindrical_3axis_gradient"])
def test_headless_project_run_preserves_requested_plot_setting(tmp_path, monkeypatch, folder):
    from importlib import import_module

    example = import_module(f"projects.{folder}.run_{folder}")
    project = deepcopy(example.PROJECT)
    project.open_plot_book = True
    project.channels = []
    observed = []
    monkeypatch.setattr(type(project), "solve", lambda *_args, **kwargs: observed.append(kwargs))
    monkeypatch.setattr(example, "export_project_stls", lambda _run: ())
    monkeypatch.setenv("MPLBACKEND", "Agg")
    example.main(project)
    assert observed == [{"open_plot_book": False}]
    assert project.open_plot_book is True
