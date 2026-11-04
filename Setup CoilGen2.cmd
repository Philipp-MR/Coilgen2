@echo off
setlocal
rem One-time (and repeatable) installation of the CoilGen2 Python environment.
rem The environment is created outside this folder, in %USERPROFILE%\.coilgen2\venv,
rem unless COILGEN_VENV names another location. A conda base Python is replaced
rem by a standalone CPython, because conda's Qt DLLs break the PySide6 wheels.
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "COILGEN_BASE_PYTHON="
set "COILGEN_SETUP_VENV=%USERPROFILE%\.coilgen2\venv"
if defined COILGEN_VENV set "COILGEN_SETUP_VENV=%COILGEN_VENV%"
rem Use the shared environment's base interpreter, so setup can rebuild a
rem conda-based venv without holding its python.exe open on Windows.
if exist "%COILGEN_SETUP_VENV%\pyvenv.cfg" (
  for /f "usebackq tokens=1,* delims==" %%K in ("%COILGEN_SETUP_VENV%\pyvenv.cfg") do for /f "tokens=1" %%N in ("%%K") do if /i "%%N"=="home" (
    for /f "tokens=*" %%H in ("%%L") do if exist "%%H\python.exe" set "COILGEN_BASE_PYTHON=%%H\python.exe"
  )
)
if defined COILGEN_BASE_PYTHON (
  "%COILGEN_BASE_PYTHON%" -I -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
  if errorlevel 1 set "COILGEN_BASE_PYTHON="
)
if not defined COILGEN_BASE_PYTHON (
  py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "COILGEN_BASE_PYTHON=py -3"
)
if not defined COILGEN_BASE_PYTHON (
  python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "COILGEN_BASE_PYTHON=python"
)
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%USERPROFILE%\miniforge3\python.exe") do (
  if not defined COILGEN_BASE_PYTHON if exist %%P (
    "%%~P" -I -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "COILGEN_BASE_PYTHON=%%~P"
  )
)
if not defined COILGEN_BASE_PYTHON (
  echo No Python 3.10 or newer was found. Install Python from https://www.python.org/
  echo ^(tick "Add python.exe to PATH"^) or Anaconda, then run this setup again.
  pause
  exit /b 9009
)
echo Using base Python: "%COILGEN_BASE_PYTHON%"
if "%COILGEN_BASE_PYTHON%"=="py -3" (
  py -3 -I "%~dp0tools\setup_environment.py" %*
) else if "%COILGEN_BASE_PYTHON%"=="python" (
  python -I "%~dp0tools\setup_environment.py" %*
) else (
  "%COILGEN_BASE_PYTHON%" -I "%~dp0tools\setup_environment.py" %*
)
set "COILGEN_EXIT=%ERRORLEVEL%"
if /i "%~1"=="--help" exit /b %COILGEN_EXIT%
if /i "%~1"=="-h" exit /b %COILGEN_EXIT%
if "%COILGEN_EXIT%"=="0" (
  echo.
  echo Setup finished. Start "Start CoilGen2 GUI.cmd" or "Start CoilGen2 Browser.cmd".
) else (
  echo.
  echo Setup failed ^(exit code %COILGEN_EXIT%^). See the messages above.
)
pause
exit /b %COILGEN_EXIT%
