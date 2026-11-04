@echo off
setlocal
rem Browser workbench: the same design core as "Start CoilGen2 GUI.cmd".
rem   Start CoilGen2 Browser.cmd           this computer only, opens the browser
rem   Start CoilGen2 Browser.cmd --team    team server on the local network (prints URL and password)
rem Further options: --port 8765 --password TEXT --max-users 8 --data-root DIR
echo Starting CoilGen2 browser workbench...
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "PYTHONDONTWRITEBYTECODE=1"
call "%~dp0tools\find_python.cmd"
if errorlevel 1 (
  pause
  exit /b 9009
)
echo Python: %COILGEN_PYTHON_CONTEXT%
"%COILGEN_PYTHON%" -u -m coilgen.web %*
set "COILGEN_EXIT=%ERRORLEVEL%"
if not "%COILGEN_EXIT%"=="0" (
  echo.
  echo The CoilGen2 browser workbench stopped ^(exit code %COILGEN_EXIT%^). See the messages above.
  pause
)
exit /b %COILGEN_EXIT%
