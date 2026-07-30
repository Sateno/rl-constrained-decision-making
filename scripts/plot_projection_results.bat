@echo off
setlocal

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

python -m analysis.build_projection_results %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
