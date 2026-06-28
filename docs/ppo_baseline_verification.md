PPO Baseline and Evaluation Foundation Verification
Scope
Components:
`environments.factory.make_env`
`evaluation.evaluate_policy`
`algorithms.ppo.ppo_continuous_action`
Environment: `RL_PROJECTS`  
Repository root: `C:\rl_projects\src\repos\rl-constrained-decision-making`
This document records only canonical verification commands for the PPO baseline and evaluation foundation. It excludes repository setup, directory creation, file creation, Git operations, and one-off debugging commands.
Setup
```cmd
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
conda activate RL_PROJECTS
```
Syntax Check
```cmd
python -m py_compile environments\factory.py evaluation\evaluate_policy.py algorithms\ppo\ppo_continuous_action.py
```
Expected result:
```text
no output
```
Environment Factory Check
```cmd
python -c "from environments.factory import make_env; env = make_env(env_index=0, env_kwargs={'max_episode_steps': 10}, record_episode_statistics=False)(); obs, info = env.reset(seed=0); print(obs.shape, obs.dtype, env.action_space.shape, info); env.close()"
```
Expected properties:
```text
observation shape is (21,)
observation dtype is float32
action shape is (2,)
info dictionary contains environment diagnostics
```
Vector Environment Compatibility Check
```cmd
python -c "import gymnasium as gym, numpy as np; from environments.factory import make_env; envs = gym.vector.SyncVectorEnv([make_env(env_index=i, env_kwargs={'max_episode_steps': 10}, record_episode_statistics=False) for i in range(2)]); obs, infos = envs.reset(seed=0); actions = np.stack([envs.single_action_space.sample() for _ in range(2)]); next_obs, rewards, terms, truncs, infos = envs.step(actions); print(obs.shape, next_obs.shape, rewards.shape, terms.shape, truncs.shape); envs.close()"
```
Expected result:
```text
(2, 21) (2, 21) (2,) (2,) (2,)
```
Random-Policy Evaluator Run
```cmd
python -m evaluation.evaluate_policy --episodes 5 --seed 0 --output runs\evaluation\random_policy_evaluation.csv --max-episode-steps 200
```
Expected terminal output:
```text
Evaluation summary
------------------
episodes:                  5
mean_return:               ...
mean_length:               ...
success_rate:              ...
collision_rate:            ...
mean_min_obstacle_distance: ...
csv_output:                results\tables\random_policy_evaluation.csv
```
CSV Output Check
```cmd
python -c "import pandas as pd; df = pd.read_csv('results/tables/random_policy_evaluation.csv'); print(df.shape); print(df.columns.tolist()); print(df.head())"
```
Expected properties:
```text
5 rows
columns include episode, seed, episode_return, episode_length, success, collision, terminated, truncated, final_distance_to_goal, min_obstacle_distance
```
BAT Wrapper Check
```cmd
scripts\evaluate_policy.bat
```
Expected property:
```text
The wrapper runs the evaluator and writes results\tables\random_policy_evaluation.csv.
```
Existing Environment Tests
```cmd
python -m pytest -q tests\test_constrained_navigation.py
```
Expected result:
```text
5 passed
```
PPO Smoke Run
Use this command after the adapted PPO file is wired to the constrained-navigation environment.
```cmd
python -m algorithms.ppo.ppo_continuous_action --total-timesteps 256 --num-envs 2 --num-steps 32 --max-episode-steps 200 --env-id ConstrainedNavigation-v0
```
Expected properties:
```text
PPO script starts without import errors
vector environments construct successfully
rollout collection begins
no observation/action shape mismatch occurs
projection remains disabled
```
Exit Criteria
The PPO baseline and evaluation foundation is considered verified when:
the factory and evaluator modules compile;
the factory creates valid single environments;
`SyncVectorEnv` works with the factory;
the evaluator runs random-policy episodes and writes a CSV file;
the BAT wrapper launches the evaluator;
existing environment tests still pass;
the adapted PPO script completes a small smoke run without import or shape errors.