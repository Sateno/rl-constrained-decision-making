@echo off
setlocal

cd /d "%~dp0.."

python -m evaluation.evaluate_policy ^
  --episodes 5 ^
  --seed 0 ^
  --max-episode-steps 200 ^
  --output runs\evaluation\random_policy_evaluation.csv

endlocal
