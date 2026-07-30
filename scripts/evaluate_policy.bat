@echo off
setlocal

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

python -m evaluation.evaluate_policy --episodes 5 --seed 0 --max-episode-steps 200 --output runs\evaluation\random_policy_evaluation.csv %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
