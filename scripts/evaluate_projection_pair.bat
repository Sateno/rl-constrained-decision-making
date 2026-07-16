@echo off
setlocal

cd /d "%~dp0.."

set CHECKPOINT=runs\checkpoints\ppo_baseline_51200_seed1.pt
set OUTPUT_PREFIX=runs\evaluation\ppo_baseline_51200_seed1_projection_pair

if not exist "%CHECKPOINT%" (
    echo ERROR: checkpoint not found: %CHECKPOINT%
    exit /b 1
)

if not exist runs\evaluation mkdir runs\evaluation

python -m evaluation.evaluate_projection_pair ^
  --checkpoint "%CHECKPOINT%" ^
  --episodes 20 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --max-obstacles 3 ^
  --num-active-obstacles 3 ^
  --projection-lookahead-distance 0.25 ^
  --projection-alpha 2.0 ^
  --projection-slack-penalty 1000.0 ^
  --projection-extra-clearance 0.0 ^
  --no-cuda ^
  --output-prefix "%OUTPUT_PREFIX%"
if errorlevel 1 exit /b 1

echo.
echo Paired projection evaluation completed successfully.
echo Outputs:
echo   %OUTPUT_PREFIX%_projection_disabled.csv
echo   %OUTPUT_PREFIX%_projection_enabled.csv
echo   %OUTPUT_PREFIX%_paired_episodes.csv
echo   %OUTPUT_PREFIX%_paired_summary.csv

endlocal
