@echo off
setlocal
echo Starting CoilGen2...
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
rem Keep bytecode caches out of synchronised folders such as Dropbox.
set "PYTHONDONTWRITEBYTECODE=1"
call "%~dp0tools\find_python.cmd"
if errorlevel 1 (
  pause
  exit /b 9009
)
echo Python: %COILGEN_PYTHON_CONTEXT%
"%COILGEN_PYTHON%" -u -m coilgen.gui %*
set "COILGEN_EXIT=%ERRORLEVEL%"
if not "%COILGEN_EXIT%"=="0" (
  echo.
  echo CoilGen2 could not start ^(exit code %COILGEN_EXIT%^). See the error above.
  pause
)
exit /b %COILGEN_EXIT%
