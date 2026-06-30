@echo off
setlocal

cd /d "%~dp0.."

if not exist runs\checkpoints mkdir runs\checkpoints

python -m algorithms.ppo.ppo_continuous_action --env-id ConstrainedNavigation-v0 --total-timesteps 2048 --num-envs 2 --num-steps 64 --num-minibatches 4 --update-epochs 2 --max-episode-steps 200 --save-model True --checkpoint-path runs\checkpoints\ppo_baseline_smoke.pt

endlocal