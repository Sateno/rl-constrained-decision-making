@echo off
setlocal

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

python -m experiments.calibrate_core_layouts %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
