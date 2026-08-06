@echo off
setlocal

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

python -m evaluation.evaluate_projection_pair --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt --method ppo_baseline --train-seed 1 --episodes 20 --seed 1000 --max-episode-steps 200 --max-obstacles 3 --num-active-obstacles 3 --collision-penalty 10.0 --projection-lookahead-distance 0.25 --projection-alpha 2.0 --projection-slack-penalty 1000.0 --projection-extra-clearance 0.0 --no-cuda --output-prefix runs\evaluation\ppo_baseline_51200_seed1_projection_pair %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
