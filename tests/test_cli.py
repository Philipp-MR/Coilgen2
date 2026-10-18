"""Project CLI compatibility and dependency isolation checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from coilgen.cli import main


def test_project_cli_roundtrip_and_export(tmp_path, capsys):
    project_path = tmp_path / "gradient.coilgen.json"
    assert main(["new", "--preset", "cylinder", "--name", "Gradient", "--output", str(project_path)]) == 0
    capsys.readouterr()
    assert main(["inspect", str(project_path)]) == 0
    values = json.loads(capsys.readouterr().out)
    assert values["name"] == "Gradient"
    assert values["config"]["target_gradient_strength"] == 0.02
    assert values["study"]["pipeline_stage"] == "evaluated"
    script = tmp_path / "replay.py"
    assert main(["export-script", str(project_path), str(script)]) == 0
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    from coilgen.application import ProjectDocument

    assert ProjectDocument.load(script.with_suffix(".coilgen.json")).fingerprint == values["fingerprint"]


def test_malformed_project_returns_actionable_error(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"schema":"unknown"}', encoding="utf-8")
    assert main(["inspect", str(path)]) == 1
    assert "schema" in capsys.readouterr().err


@pytest.mark.parametrize("with_engineering", [False, True])
def test_stage_override_runs_full_design_preset_without_fabrication(tmp_path, capsys, with_engineering):
    from coilgen.application import ProjectDocument
    from coilgen.gradient_engineering import GradientEngineeringSpec

    path = tmp_path / "gradient.json"
    assert main(["new", "--preset", "cylinder", "--output", str(path)]) == 0
    original = ProjectDocument.load(path)
    if with_engineering:
        original = original.update_study({"engineering": GradientEngineeringSpec().to_dict()})
        original.save(path)
    assert original.study["fabricate"] is True
    output = tmp_path / "prepared"
    assert main(["run", str(path), "--stage", "prepared", "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert summary["pipeline_stage"] == "prepared"
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    assert request["fabricate"] is False
    assert request.get("engineering") is None
    assert ProjectDocument.load(path).fingerprint == original.fingerprint
    assert ProjectDocument.load(path).study["fabricate"] is True
    assert ProjectDocument.load(path).study.get("engineering") == original.study.get("engineering")


def test_headless_entrypoints_never_import_gui_dependencies(tmp_path):
    code = """
import importlib.abc
import sys
class BlockGUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'pyvista', 'pyvistaqt', 'vtk', 'vtkmodules'}:
            raise AssertionError('Headless operation imported ' + fullname)
sys.meta_path.insert(0, BlockGUI())
from coilgen import CoilGenConfig, run_coilgen
from coilgen.cli import main
from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from pathlib import Path
root = Path(sys.argv[1])
project = ProjectDocument.new('Headless', preset_config())
project.save(root / 'project.json')
assert main(['inspect', str(root / 'project.json')]) == 0
project.export_python(root / 'replay.py')
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], env=environment, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
