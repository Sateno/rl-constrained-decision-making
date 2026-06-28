Constrained Navigation Verification
Scope
Component: `environments.constrained_navigation.ConstrainedNavigationEnv`  
Environment: `RL_PROJECTS`  
Repository root: `C:\rl_projects\src\repos\rl-constrained-decision-making`
This document records only canonical verification commands for the constrained-navigation environment. It excludes repository setup, directory creation, file creation, Git operations, and one-off debugging commands.
Setup
```cmd
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
conda activate RL_PROJECTS
```
Syntax Check
```cmd
python -m py_compile environments\constrained_navigation.py
```
Expected result:
```text
no output
```
Environment Space Check
```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; env=ConstrainedNavigationEnv(); print(env.observation_space.shape, env.action_space.shape)"
```
Expected result:
```text
(21,) (2,)
```
Reset and Step API Check
```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(); obs,info=env.reset(seed=0); out=env.step(np.array([0.1,0.0], dtype=np.float32)); print(len(out), out[0].shape, out[0].dtype, type(out[1]), type(out[2]), type(out[3]), out[4])"
```
Expected properties:
```text
5-tuple returned
observation shape is (21,)
observation dtype is float32
reward is float
terminated is bool
truncated is bool
info dictionary contains environment diagnostics
```
Unicycle Dynamics Sanity Check
```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(); env.reset(seed=0, options={'start': np.array([0.0,0.0]), 'theta': 0.0, 'goal': np.array([10.0,0.0]), 'obstacle_mask': np.array([False,False,False])}); env.step(np.array([1.0,0.0], dtype=np.float32)); print(env.position, env.theta)"
```
Expected result for `dt = 0.1`:
```text
position approximately [0.1, 0.0]
theta approximately 0.0
```
Lightweight Environment Tests
```cmd
python -m pytest -q tests\test_constrained_navigation.py
```
Expected result:
```text
5 passed
```
Random Rollout Smoke Check
```cmd
python -c "exec('from environments.constrained_navigation import ConstrainedNavigationEnv\nimport numpy as np\nenv = ConstrainedNavigationEnv(max_episode_steps=10)\nenv.action_space.seed(0)\nobs, info = env.reset(seed=0)\nsteps = 0\ndone = False\nok = np.isfinite(obs).all()\nwhile not done:\n    action = env.action_space.sample()\n    obs, reward, terminated, truncated, info = env.step(action)\n    ok = ok and np.isfinite(obs).all() and np.isfinite(reward)\n    steps += 1\n    done = terminated or truncated\nprint(steps, ok, info)')"
```
Expected properties:
```text
step count is positive
finite-value check is True
final info dictionary is valid
episode ends by termination or truncation
```
Exit Criteria
The constrained-navigation environment is considered verified when:
the module compiles;
the environment constructs successfully;
observation and action spaces have the expected shapes;
`reset` and `step` satisfy the Gymnasium API;
the direct dynamics sanity check matches the expected forward step;
lightweight environment tests pass;
a short random rollout produces finite observations and rewards.