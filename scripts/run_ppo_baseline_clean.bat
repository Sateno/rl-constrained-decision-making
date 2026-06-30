@echo off
setlocal

cd /d "%~dp0.."

echo ============================================================
echo Clean PPO baseline run
echo Repository: %CD%
echo ============================================================

echo.
echo [1/9] Removing old generated runs directory...
if exist runs rmdir /s /q runs
if exist runs (
    echo ERROR: runs directory still exists. Stop TensorBoard or any process using runs/.
    exit /b 1
)

mkdir runs
mkdir runs\checkpoints
mkdir runs\evaluation

echo.
echo [2/9] Python syntax check...
python -m py_compile environments\constrained_navigation.py environments\factory.py evaluation\evaluate_policy.py algorithms\ppo\agent.py algorithms\ppo\ppo_continuous_action.py
if errorlevel 1 exit /b 1

echo.
echo [3/9] Test suite...
python -m pytest -q
if errorlevel 1 exit /b 1

echo.
echo [4/9] Random-policy evaluation...
python -m evaluation.evaluate_policy --policy random --episodes 20 --seed 1000 --output runs\evaluation\random_policy_seed1000.csv --max-episode-steps 200
if errorlevel 1 exit /b 1

echo.
echo [5/9] PPO baseline training...
python -m algorithms.ppo.ppo_continuous_action --exp-name ppo_baseline_51200_seed1_clean --env-id ConstrainedNavigation-v0 --total-timesteps 51200 --num-envs 4 --num-steps 256 --num-minibatches 8 --update-epochs 4 --max-episode-steps 200 --seed 1 --save-model --checkpoint-path runs\checkpoints\ppo_baseline_51200_seed1.pt
if errorlevel 1 exit /b 1

echo.
echo [6/9] Checkpoint existence check...
if not exist runs\checkpoints\ppo_baseline_51200_seed1.pt (
    echo ERROR: checkpoint was not created.
    exit /b 1
)
dir runs\checkpoints\ppo_baseline_51200_seed1.pt

echo.
echo [7/9] Deterministic PPO checkpoint evaluation...
python -m evaluation.evaluate_policy --policy ppo --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt --episodes 20 --seed 1000 --output runs\evaluation\ppo_baseline_51200_seed1_eval.csv --max-episode-steps 200
if errorlevel 1 exit /b 1

echo.
echo [8/9] Stochastic PPO checkpoint evaluation...
python -m evaluation.evaluate_policy --policy ppo --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt --episodes 20 --seed 1000 --output runs\evaluation\ppo_baseline_51200_seed1_eval_stochastic.csv --max-episode-steps 200 --stochastic
if errorlevel 1 exit /b 1

echo.
echo [9/9] Summary comparison...
python -c "import pandas as pd; cols=['episode_return','episode_length','success','collision','final_distance_to_goal','min_obstacle_distance']; r=pd.read_csv(r'runs\evaluation\random_policy_seed1000.csv'); p=pd.read_csv(r'runs\evaluation\ppo_baseline_51200_seed1_eval.csv'); s=pd.read_csv(r'runs\evaluation\ppo_baseline_51200_seed1_eval_stochastic.csv'); print('random'); print(r[cols].mean(numeric_only=True)); print(); print('ppo deterministic'); print(p[cols].mean(numeric_only=True)); print(); print('ppo stochastic'); print(s[cols].mean(numeric_only=True))"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Clean PPO baseline run completed successfully.
echo Outputs:
echo   runs\checkpoints\ppo_baseline_51200_seed1.pt
echo   runs\evaluation\random_policy_seed1000.csv
echo   runs\evaluation\ppo_baseline_51200_seed1_eval.csv
echo   runs\evaluation\ppo_baseline_51200_seed1_eval_stochastic.csv
echo TensorBoard:
echo   tensorboard --logdir runs
echo ============================================================

endlocal