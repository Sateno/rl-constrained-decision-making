@echo off
setlocal

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

python -m experiments.run_clean_ppo_baseline --replace-runs %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
