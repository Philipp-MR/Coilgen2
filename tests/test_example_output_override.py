"""Direct project launches can isolate outputs without moving their input data."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", ["", "isolated_runs", "absolute"])
def test_output_override_preserves_lazy_run_isolation_and_input_root(tmp_path, value):
    requested = str(tmp_path / "absolute_runs") if value == "absolute" else value
    env = {**os.environ, "COILGEN_OUTPUT_ROOT": requested, "PYTHONDONTWRITEBYTECODE": "1"}
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from projects._shared.paths import OUTPUT_ROOT, PROJECT_DATA_ROOT, begin_project_run
first = begin_project_run('general_examples', create=False)
second = begin_project_run('general_examples', create=False)
assert first != second
assert not first.exists() and not second.exists()
print(json.dumps([str(OUTPUT_ROOT), str(PROJECT_DATA_ROOT), str(first.parent)]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    output_root, data_root, project_base = map(Path, json.loads(result.stdout))
    expected = (tmp_path / requested).resolve() if requested else ROOT / "outputs"
    assert output_root == expected
    assert project_base == expected / "general_examples"
    assert data_root == ROOT / "project_data"
    assert not list(tmp_path.iterdir())
