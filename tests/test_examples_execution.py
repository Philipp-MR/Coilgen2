"""Standalone example contracts: inert imports, explicit work and fixed operators."""

from dataclasses import replace
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "research_basic_designs",
    "research_tolerance_study",
    "research_parallel_study",
    "research_valid_fabrication",
    "research_geometry_benchmark",
    "optimization_minimax_comparison",
)


def _example(name):
    # Examples are checkout scripts, intentionally outside the installed package.
    sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module(f"examples.{name}")
    finally:
        sys.path.pop(0)


def test_all_examples_import_without_numerical_work_or_environment_changes(tmp_path):
    script = f"""
import importlib, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, {str(ROOT)!r})
environment, paths = dict(os.environ), list(sys.path)
for name in {EXAMPLES!r}:
    module = importlib.import_module('examples.' + name)
    assert callable(module.main)
assert dict(os.environ) == environment
assert sys.path == paths
assert not ({{'numpy', 'coilgen', 'numba', 'cvxpy'}} & set(sys.modules))
assert not list(Path.cwd().iterdir())
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("name", EXAMPLES)
def test_help_needs_no_numerical_dependencies(name, monkeypatch, capsys):
    module = _example(name)

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("--help must not initialize numerical dependencies")

    monkeypatch.setattr(module, "example_runtime", unexpected_runtime)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert "--output-dir" in capsys.readouterr().out


def test_example_paths_and_missing_dependency_guidance_restore_runtime(monkeypatch, tmp_path):
    support = _example("_example_support")
    monkeypatch.chdir(tmp_path)
    assert support.example_options("outputs/example", []).output_dir == ROOT / "outputs/example"
    assert support.example_options("outputs/example", ["--output-dir", "custom"]).output_dir == tmp_path / "custom"
    assert not support.example_options("outputs/example", ["--no-resume"], study=True).resume
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    environment, paths = dict(os.environ), list(sys.path)
    monkeypatch.setattr(support, "find_spec", lambda _name: None)
    with pytest.raises(SystemExit, match="pip install -e") as exc:
        with support.example_runtime(extra="research", required=("cvxpy",)):
            pytest.fail("Missing dependencies must stop before running")
    assert "[research]" in str(exc.value)
    assert "cvxpy" in str(exc.value)
    assert dict(os.environ) == environment
    assert sys.path == paths


def test_tolerance_example_assembly_matches_existing_pipeline_operators():
    pytest.importorskip("cvxpy")
    from threadpoolctl import threadpool_limits

    from coilgen import run_coilgen
    from coilgen.optimization import StreamFunctionOptimizationProblem

    example = _example("research_tolerance_study")
    config = replace(
        example.cylinder_config(), retain_heavy_fields=True, low_memory_mode=False, parallel_worker_count=1
    )
    with threadpool_limits(1):
        direct = example.assemble_cylinder_problem(config, ROOT)
        seed = run_coilgen(config, project_root=ROOT)
        previous = StreamFunctionOptimizationProblem.from_parts(seed.coil_parts, seed.target_field, config)
    for name in ("reduced_sensitivity", "reduced_resistance", "target_z"):
        np.testing.assert_array_equal(getattr(direct, name), getattr(previous, name))
    np.testing.assert_array_equal(direct.reducer.expansion_matrix, previous.reducer.expansion_matrix)
