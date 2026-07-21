@echo off
setlocal

cd /d "%~dp0.."

set "SEED=%~1"
if not defined SEED set "SEED=1"

set "TOTAL_TIMESTEPS=%~2"
if not defined TOTAL_TIMESTEPS set "TOTAL_TIMESTEPS=51200"

set "CHECKPOINT_PATH=%~3"
if not defined CHECKPOINT_PATH set "CHECKPOINT_PATH=runs\checkpoints\ppo_train_projection_%TOTAL_TIMESTEPS%_seed%SEED%.pt"

if exist "%CHECKPOINT_PATH%" (
    echo ERROR: checkpoint already exists: %CHECKPOINT_PATH%
    exit /b 1
)

for %%I in ("%CHECKPOINT_PATH%") do if not exist "%%~dpI" mkdir "%%~dpI"

python -m algorithms.ppo.ppo_continuous_action ^
  --method ppo_train_projection ^
  --exp-name ppo_train_projection_%TOTAL_TIMESTEPS%_seed%SEED% ^
  --env-id ConstrainedNavigation-v0 ^
  --total-timesteps %TOTAL_TIMESTEPS% ^
  --num-envs 4 ^
  --num-steps 256 ^
  --num-minibatches 8 ^
  --update-epochs 4 ^
  --max-episode-steps 200 ^
  --max-obstacles 3 ^
  --num-active-obstacles 3 ^
  --collision-penalty 10.0 ^
  --seed %SEED% ^
  --save-model ^
  --enable-projection ^
  --projection-lookahead-distance 0.25 ^
  --projection-alpha 2.0 ^
  --projection-slack-penalty 1000.0 ^
  --projection-extra-clearance 0.0 ^
  --checkpoint-path "%CHECKPOINT_PATH%"

if errorlevel 1 exit /b 1

if not exist "%CHECKPOINT_PATH%" (
    echo ERROR: checkpoint was not created: %CHECKPOINT_PATH%
    exit /b 1
)

echo Training completed successfully.
echo Checkpoint: %CHECKPOINT_PATH%

endlocal
