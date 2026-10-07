@echo off
setlocal
echo Starting CoilGen2...
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "COILGEN_PYTHON=%~dp0.venv\Scripts\python.exe"
set "COILGEN_PYTHON_CONTEXT=project virtual environment ^(.venv^)"
if exist "%COILGEN_PYTHON%" goto launch
set "COILGEN_PYTHON=python"
set "COILGEN_PYTHON_CONTEXT=available Python on PATH"
where python >nul 2>nul
if errorlevel 1 (
  echo CoilGen2 needs Python. Install Python and the GUI dependencies, then try again.
  pause
  exit /b 9009
)

:launch
echo Python: %COILGEN_PYTHON_CONTEXT%
"%COILGEN_PYTHON%" -u -m coilgen.gui %*
set "COILGEN_EXIT=%ERRORLEVEL%"
if not "%COILGEN_EXIT%"=="0" (
  echo.
  echo CoilGen2 could not start ^(exit code %COILGEN_EXIT%^). See the error above.
  pause
)
exit /b %COILGEN_EXIT%
