import builtins
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock
import venv

import pytest

from tools import setup_environment


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher")
@pytest.mark.parametrize("configured", [False, True])
def test_setup_help_uses_existing_environment_without_python_on_path(tmp_path, configured):
    checkout = tmp_path / "CoilGen (local) & examples"
    (checkout / "tools").mkdir(parents=True)
    shutil.copy2(setup_environment.REPOSITORY_ROOT / "Setup CoilGen2.cmd", checkout)
    shutil.copy2(setup_environment.__file__, checkout / "tools")
    profile = tmp_path / "test profile"
    environment = tmp_path / "user environment" if configured else profile / ".coilgen2" / "venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    environ = {**os.environ, "USERPROFILE": str(profile), "PATH": str(Path(os.environ["SYSTEMROOT"]) / "System32")}
    environ.pop("COILGEN_VENV", None)
    if configured:
        environ["COILGEN_VENV"] = str(environment)
    command = f'"{os.environ["ComSpec"]}" /d /s /c ""{checkout / "Setup CoilGen2.cmd"}" --help"'
    result = subprocess.run(command, cwd=tmp_path, env=environ, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage: setup_environment.py" in result.stdout and "--extras" in result.stdout
    assert "Installing dependencies" not in result.stdout and "Setup finished" not in result.stdout


def test_old_host_uses_target_environment_for_toml(monkeypatch):
    real_import = builtins.__import__

    def without_tomllib(name, *args, **kwargs):
        if name == "tomllib":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tomllib)
    # Preserve check_output's real subprocess to exercise the isolated reader.
    run = subprocess.run
    pip_call = Mock()

    def verify_run(args, **kwargs):
        if args[1:3] == ["-m", "pip"]:
            return pip_call(args, **kwargs)
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", verify_run)
    python = Path(sys.executable)
    project = setup_environment._load_pyproject(python)["project"]
    pip_call.assert_called_once_with(
        [str(python), "-m", "pip", "install", "tomli>=2; python_version < '3.11'"], check=True
    )
    assert "web" in project["optional-dependencies"]
