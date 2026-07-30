@echo off
setlocal

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

python -m evaluation.inspect_pre_experiment_validation %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
