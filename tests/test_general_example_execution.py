"""Regression checks for standalone launches and truthful example outputs."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from projects._shared.example_utils import (
    interactive_plots_enabled,
    native_thread_budget,
    open_project_plot_book_after_run,
)


ROOT = Path(__file__).resolve().parents[1]


def test_general_scripts_import_without_checkout_on_pythonpath(tmp_path):
    # A direct script launch sees its own directory, not the repository root.
    # Run imports in one clean child so tests cannot mask broken bootstrapping.
    scripts = sorted((ROOT / "projects/general_examples/scripts").glob("[!_]*.py"))
    code = """
import runpy, sys
from pathlib import Path
for name in sys.argv[1:]:
    script = Path(name)
    for module in list(sys.modules):
        if module == 'projects' or module.startswith('projects.'):
            del sys.modules[module]
    sys.path[:] = [entry for entry in sys.path if entry not in {str(script.parents[3]), str(script.parents[3] / 'src')}]
    namespace = runpy.run_path(str(script))
    assert 'main' in namespace, script
print('Imported', len(sys.argv)-1, 'general examples without executing designs.')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, *(str(path) for path in scripts)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())


def test_headless_or_disabled_examples_do_not_open_plot_books(monkeypatch):
    def unexpected_plot(**kwargs):
        raise AssertionError("A disabled plot book must not be created.")

    run = SimpleNamespace(show_plot_book=unexpected_plot)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert open_project_plot_book_after_run(run) is None
    monkeypatch.setenv("MPLBACKEND", "qtagg")
    assert open_project_plot_book_after_run(run, enabled=False) is None
    monkeypatch.setattr(sys, "argv", ["example.py", "--headless"])
    assert open_project_plot_book_after_run(run) is None
    assert not interactive_plots_enabled(True)


def test_native_thread_budget_restores_existing_pools():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    before = [(pool["filepath"], pool["num_threads"]) for pool in threadpoolctl.threadpool_info()]
    with native_thread_budget(1):
        assert all(pool["num_threads"] == 1 for pool in threadpoolctl.threadpool_info())
    assert [(pool["filepath"], pool["num_threads"]) for pool in threadpoolctl.threadpool_info()] == before


def test_simple_gradient_and_gallery_examples_match_their_names():
    from projects.general_examples.scripts import export_visualizations, simple_ygradient

    config = simple_ygradient.build_config()
    assert config.field_shape_function == "y"
    assert config.target_region_resolution <= 9
    assert config.cylinder_mesh_parameter_list[2] * config.cylinder_mesh_parameter_list[3] <= 1000
    assert export_visualizations.get_parameters()["output"]["save_svg_gallery"]


def test_swept_export_default_produces_valid_independent_solids(tmp_path):
    from coilgen import CoilGenPipeline
    from coilgen.manufacturing import save_swept_bodies_as_step, save_swept_bodies_as_stl
    from projects.general_examples.scripts import export_swept_body

    config = export_swept_body.build_config()
    config.progress_flag = False
    with native_thread_budget():
        result = CoilGenPipeline(config).run()
    bodies = [body for part in result.coil_parts for body in part.swept_bodies]
    assert len(bodies) == 4
    assert all(body.diagnostics["manufacturing_ok"] for body in bodies)
    assert all(body.contour_index is not None for body in bodies)
    files = save_swept_bodies_as_stl(result, tmp_path)
    files += save_swept_bodies_as_step(result, tmp_path)
    assert len([path for path in files if path.suffix == ".stl" and "contour" in path.stem]) == 4
    assert len([path for path in files if path.suffix == ".step"]) == 4
    assert all(path.stat().st_size > 0 for path in files)
