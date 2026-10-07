"""Project CLI compatibility and dependency isolation checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

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
