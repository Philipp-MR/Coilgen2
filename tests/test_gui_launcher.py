"""Launch the Windows entry point from another directory, including a Qt startup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher")
ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "Start CoilGen2 GUI.cmd"


def _run_launcher(launcher, arguments, cwd, environment):
    command = f'cmd.exe /d /s /c ""{launcher}" {arguments}"'
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        input="\n",
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_launcher_resolves_local_source_forwards_arguments_and_preserves_exit(tmp_path, exit_code):
    checkout = tmp_path / "local checkout with spaces"
    module = checkout / "src" / "coilgen" / "gui"
    module.mkdir(parents=True)
    (module.parent / "__init__.py").write_text("")
    (module / "__init__.py").write_text("")
    (module / "__main__.py").write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import sys
            from pathlib import Path
            parser = argparse.ArgumentParser()
            parser.add_argument('--output', type=Path)
            parser.add_argument('--exit-code', type=int)
            parser.add_argument('project')
            args = parser.parse_args()
            args.output.write_text(json.dumps({
                'cwd': str(Path.cwd()),
                'module': __file__,
                'project': args.project,
                'stdout_unbuffered': sys.stdout.write_through,
            }))
            raise SystemExit(args.exit_code)
            """
        ),
        encoding="utf-8",
    )
    launcher = checkout / LAUNCHER.name
    launcher.write_bytes(LAUNCHER.read_bytes())
    conflicting = tmp_path / "other checkout"
    (conflicting / "coilgen").mkdir(parents=True)
    (conflicting / "coilgen" / "__init__.py").write_text("raise RuntimeError('wrong checkout')")
    report = tmp_path / "arguments.json"
    environment = os.environ.copy()
    environment.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", ""),
        PYTHONPATH=str(conflicting),
    )
    completed = _run_launcher(
        launcher,
        f'--output "{report}" --exit-code {exit_code} "model input.coilgen.json"',
        tmp_path,
        environment,
    )
    assert completed.returncode == exit_code, completed.stderr
    result = json.loads(report.read_text())
    assert Path(result["cwd"]) == checkout
    assert Path(result["module"]) == module / "__main__.py"
    assert result["project"] == "model input.coilgen.json"
    assert result["stdout_unbuffered"]
    assert "Starting CoilGen2..." in completed.stdout
    assert "Python: available Python on PATH" in completed.stdout
    if exit_code:
        assert "could not start (exit code 7)" in completed.stdout


def test_invalid_project_reports_loading_and_failure_without_ready(tmp_path):
    project_path = tmp_path / "Broken project.coilgen.json"
    project_path.write_text("not JSON", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(TEMP=str(tmp_path), TMP=str(tmp_path), QT_QPA_PLATFORM="offscreen")
    completed = _run_launcher(LAUNCHER, f'"{project_path}"', tmp_path, environment)
    assert completed.returncode == 2
    assert "Loading project: Broken project.coilgen.json" in completed.stdout
    assert "Workspace ready." not in completed.stdout
    assert "could not start (exit code 2)" in completed.stdout
    assert "Expecting value" in completed.stderr


@pytest.mark.parametrize("load_project", [False, True], ids=["empty-workspace", "saved-project"])
def test_repository_launcher_enters_qt_event_loop_without_live_jobs(tmp_path, load_project):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    report = tmp_path / "startup.json"
    jobs = tmp_path / "isolated" / "jobs"
    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication
            import pyvistaqt
            from test_native_workbench import RenderlessPlotter

            # Windows' offscreen Qt platform has no OpenGL pixel format. Keep
            # the real scientific viewport and datasets; replace its renderer.
            pyvistaqt.QtInteractor = RenderlessPlotter
            original_exec = QApplication.exec

            def probe_exec(self):
                def finish():
                    result = {}
                    try:
                        window = self._coilgen_workbenches[0]
                        result = {
                            'event_loop_entered': True,
                            'window_visible': window.isVisible(),
                            'compute_visible': window.compute_button.isVisible(),
                            'viewport': type(window.viewport).__name__,
                            'jobs_directory': str(window.jobs_directory),
                            'no_job_manager': window.jobs is None,
                            'project': window.project.name if window.project is not None else None,
                            'window_size': [window.width(), window.height()],
                            'compute_size': [window.compute_button.width(), window.compute_button.minimumSizeHint().width()],
                            'compute_text': window.compute_button.text(),
                            'toolbar_width': window.compute_button.parentWidget().width(),
                        }
                        window.grab().save(str(Path(os.environ['COILGEN_STARTUP_REPORT']).with_suffix('.png')))
                        window.undo_stack.setClean()
                        window._draft_dirty = False
                        window.close()
                    except Exception as exc:
                        result['error'] = repr(exc)
                    finally:
                        Path(os.environ['COILGEN_STARTUP_REPORT']).write_text(json.dumps(result))
                        self.quit()
                QTimer.singleShot(1500, finish)
                return original_exec()

            QApplication.exec = probe_exec
            """
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        QT_QPA_PLATFORM="offscreen",
        COILGEN_STARTUP_REPORT=str(report),
        PYTHONPATH=str(tmp_path) + os.pathsep + str(ROOT / "tests"),
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
    )
    arguments = f'--background --jobs-directory "{jobs}"'
    if load_project:
        from coilgen.application import ProjectDocument
        from coilgen.application.presets import preset_config

        project_path = tmp_path / "Saved design.coilgen.json"
        ProjectDocument.from_config(preset_config(), name="Saved launch design").save(project_path)
        arguments = f'"{project_path}" ' + arguments
    completed = _run_launcher(LAUNCHER, arguments, tmp_path, environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Traceback (most recent call last)" not in completed.stderr
    assert report.is_file(), completed.stderr
    result = json.loads(report.read_text())
    assert result.get("event_loop_entered"), result
    assert result["window_visible"]
    if load_project:
        assert result["compute_visible"]
    assert result["viewport"] == "ScientificViewport"
    assert result["no_job_manager"]
    assert Path(result["jobs_directory"]) == jobs
    assert result["project"] == ("Saved launch design" if load_project else None)
    messages = (
        "Starting CoilGen2...",
        "Python: project virtual environment (.venv)",
        "Loading GUI and scientific libraries...",
        "Initializing Qt...",
        "Initializing 3D rendering...",
        "Building the workspace...",
        "Workspace ready.",
    )
    positions = [completed.stdout.index(message) for message in messages]
    assert positions == sorted(positions)
    if load_project:
        assert completed.stdout.index("Loading project: Saved design.coilgen.json") < positions[2]
