"""Numerical provenance and import boundaries survive architectural refactoring."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from coilgen.config import CoilGenConfig
from coilgen.persistence import numerical_source_digest, source_digest
from coilgen import pipeline_stages


@pytest.mark.parametrize("changed", ["research_requirements.py", "field_datasets.py", "manufacturing_witness.py"])
def test_engine_implementation_changes_invalidate_every_checkpoint_stage(tmp_path, monkeypatch, changed):
    root = Path(__file__).resolve().parents[1] / "src/coilgen"
    for name in ("research_requirements.py", "field_datasets.py", "manufacturing_witness.py"):
        (tmp_path / name).write_bytes((root / name).read_bytes())
    monkeypatch.setattr(pipeline_stages, "numerical_source_digest", lambda _root: numerical_source_digest(tmp_path))
    config = CoilGenConfig()
    before = pipeline_stages.stage_fingerprints(config, tmp_path)
    path = tmp_path / changed
    path.write_text(path.read_text(encoding="utf-8") + "\n# Changed implementation.\n", encoding="utf-8")
    after = pipeline_stages.stage_fingerprints(config, tmp_path)
    assert before.keys() == after.keys()
    assert all(before[stage] != after[stage] for stage in before)


def test_application_facade_is_provenance_only(tmp_path):
    package = tmp_path / "application"
    package.mkdir()
    facade = package / "research_requirements.py"
    facade.write_text("# Compatibility imports\n", encoding="utf-8")
    before = numerical_source_digest(tmp_path), source_digest(tmp_path)
    facade.write_text("# Updated compatibility documentation\n", encoding="utf-8")
    assert numerical_source_digest(tmp_path) == before[0]
    assert source_digest(tmp_path) != before[1]


@pytest.mark.parametrize(
    "module",
    [
        "coilgen",
        "coilgen.api",
        "coilgen.application",
        "coilgen.application.optimization_summary",
        "coilgen.application.interfaces",
    ],
)
def test_public_imports_do_not_load_optional_numerical_or_gui_packages(module):
    code = f"""
import importlib, sys
importlib.import_module({module!r})
blocked = {{'numpy', 'scipy', 'cvxpy', 'numba', 'PySide6', 'pyvista', 'vtkmodules', 'trame'}}
assert not (blocked & {{name.split('.')[0] for name in sys.modules}})
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_engine_requirements_import_no_application_adapters():
    code = """
import sys
from coilgen.research_requirements import ResearchRequirements, prepare_requirement_constraints
assert not any(name.startswith(('coilgen.application', 'coilgen.gui', 'coilgen.web')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_legacy_requirement_and_dataset_classes_retain_identity():
    from coilgen.application.gradient_studies import MeasuredFieldMap as legacy_map, RigidTransform as legacy_pose
    from coilgen.application.research_requirements import ResearchRequirements as legacy_requirements
    from coilgen.field_datasets import MeasuredFieldMap, RigidTransform
    from coilgen.research_requirements import ResearchRequirements

    assert (legacy_map, legacy_pose, legacy_requirements) == (MeasuredFieldMap, RigidTransform, ResearchRequirements)


def test_collected_tests_do_not_import_other_collected_test_modules():
    for path in Path(__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("test_"), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("test_") for alias in node.names), path
