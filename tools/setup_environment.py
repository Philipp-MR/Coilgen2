"""Create or update the CoilGen2 Python environment outside the source folder.

The environment lives in ``%USERPROFILE%\\.coilgen2\\venv`` (or ``$COILGEN_VENV``)
so synchronised folders such as Dropbox never receive thousands of package
files. Dependencies come from ``pyproject.toml``; the checkout's ``src`` is
linked through a ``.pth`` file instead of an editable install, so no
``*.egg-info`` or copied package is written next to the sources.

Usage::

    python tools/setup_environment.py [--extras dev,gui,plot-qt,web] [--venv PATH]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRAS = ("full", "gui", "plot-qt", "web", "dev")
# The Windows CI interpreter version; also used for uv-managed installs.
PYTHON_VERSION = "3.12"
PTH_NAME = "coilgen2-source.pth"


def default_venv() -> Path:
    override = os.environ.get("COILGEN_VENV", "").strip()
    if override:
        return Path(override).expanduser()
    # Not %LOCALAPPDATA%: packaged (MSIX) host applications see a private,
    # redirected copy of AppData, so an environment created there from such a
    # host would be invisible to the double-clicked launchers.
    return Path.home() / ".coilgen2" / "venv"


def _load_pyproject(python: Path) -> dict:
    path = REPOSITORY_ROOT / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:  # A fresh Python 3.10 host may have no TOML reader.
        subprocess.run([str(python), "-m", "pip", "install", "tomli>=2; python_version < '3.11'"], check=True)
        reader = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "try: import tomllib\n"
            "except ModuleNotFoundError: import tomli as tomllib\n"
            "print(json.dumps(tomllib.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))))"
        )
        return json.loads(subprocess.check_output([str(python), "-I", "-c", reader, str(path)], text=True))
    with path.open("rb") as handle:
        return tomllib.load(handle)


def requirements(project, extras: tuple[str, ...]) -> list[str]:
    optional = project.get("optional-dependencies", {})
    unknown = sorted(set(extras) - set(optional))
    if unknown:
        raise SystemExit(f"Unknown extras {unknown}; available: {sorted(optional)}")
    selected = list(project.get("dependencies", ()))
    for extra in extras:
        selected.extend(optional[extra])
    return list(dict.fromkeys(selected))


def venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _is_conda(prefix: str | Path) -> bool:
    return (Path(prefix) / "conda-meta").is_dir()


def _venv_base_is_conda(root: Path) -> bool:
    config = root / "pyvenv.cfg"
    if not config.is_file():
        return False
    for line in config.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home":
            return _is_conda(Path(value.strip()).parent) or _is_conda(Path(value.strip()))
    return False


def standalone_python(state: Path) -> Path:
    """Fetch a standalone CPython with uv into ``state``.

    Conda interpreters put their own Qt DLLs (Library\\bin) on the DLL search
    path; PySide6 wheels then fail with "procedure could not be found".
    """
    bootstrap = state / "bootstrap"
    uv = bootstrap / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    if not uv.exists():
        print("Fetching uv to install a standalone Python (the base Python is a conda build).")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--target", str(bootstrap), "uv"],
            check=True,
        )
    environment = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(state / "python")}
    subprocess.run([str(uv), "python", "install", PYTHON_VERSION], check=True, env=environment)
    found = subprocess.run(
        [str(uv), "python", "find", "--managed-python", PYTHON_VERSION],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return Path(found.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--venv", type=Path, default=default_venv())
    parser.add_argument("--extras", default=",".join(DEFAULT_EXTRAS))
    options = parser.parse_args(argv)
    extras = tuple(item.strip() for item in options.extras.split(",") if item.strip())
    root = options.venv.resolve()
    python = venv_python(root)
    if python.exists() and _venv_base_is_conda(root):
        print("Rebuilding the environment: it was created from a conda Python, whose Qt DLLs break PySide6.")
        shutil.rmtree(root)
    if not python.exists():
        print(f"Creating CoilGen2 environment: {root}")
        if _is_conda(sys.base_prefix):
            base = standalone_python(root.parent)
            subprocess.run([str(base), "-m", "venv", str(root)], check=True)
        else:
            venv.EnvBuilder(with_pip=True, clear=False).create(root)
    print(f"Installing dependencies for extras: {', '.join(extras)}")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    project = _load_pyproject(python)["project"]
    subprocess.run([str(python), "-m", "pip", "install", *requirements(project, extras)], check=True)
    site_packages = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pth = Path(site_packages) / PTH_NAME
    pth.write_text(f"{REPOSITORY_ROOT / 'src'}\n{REPOSITORY_ROOT}\n", encoding="utf-8")
    # Metadata only, so run provenance records the CoilGen2 version.
    for stale in Path(site_packages).glob("coilgen_python-*.dist-info"):
        for item in stale.iterdir():
            item.unlink()
        stale.rmdir()
    dist_info = Path(site_packages) / f"coilgen_python-{project['version']}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {project['version']}\n", encoding="utf-8"
    )
    (dist_info / "INSTALLER").write_text("coilgen2-setup\n", encoding="utf-8")
    subprocess.run([str(python), "-c", "import coilgen; print('coilgen from', coilgen.__file__)"], check=True)
    print(f"CoilGen2 environment ready: {python}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
