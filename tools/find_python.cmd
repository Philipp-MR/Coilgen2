@echo off
rem Resolve the Python used by the CoilGen2 launchers.
rem Sets COILGEN_PYTHON and COILGEN_PYTHON_CONTEXT, or exits with 9009.
rem Order: %COILGEN_VENV%, else %USERPROFILE%\.coilgen2\venv; then the
rem checkout's .venv; then a working python on PATH (not the Store alias).
set "COILGEN_ROOT=%~dp0.."
if defined COILGEN_VENV (
  set "COILGEN_PYTHON=%COILGEN_VENV%\Scripts\python.exe"
  set "COILGEN_PYTHON_CONTEXT=configured environment ^(COILGEN_VENV^)"
) else (
  set "COILGEN_PYTHON=%USERPROFILE%\.coilgen2\venv\Scripts\python.exe"
  set "COILGEN_PYTHON_CONTEXT=shared CoilGen2 environment"
)
if exist "%COILGEN_PYTHON%" exit /b 0
set "COILGEN_PYTHON=%COILGEN_ROOT%\.venv\Scripts\python.exe"
set "COILGEN_PYTHON_CONTEXT=project virtual environment ^(.venv^)"
if exist "%COILGEN_PYTHON%" exit /b 0
set "COILGEN_PYTHON=python"
set "COILGEN_PYTHON_CONTEXT=available Python on PATH"
rem The Microsoft Store alias is found by "where" but cannot run scripts.
python -c "import sys" >nul 2>nul
if not errorlevel 1 exit /b 0
set "COILGEN_PYTHON="
echo CoilGen2 needs Python. Run "Setup CoilGen2.cmd" in the CoilGen2 folder first.
exit /b 9009
