# P1.1 Environment Verification Commands

## Context

Milestone: P1.1 Environment  
Branch: `p1-environment`  
Conda environment: `RL_PROJECTS`  
Repository root: `C:\rl_projects\src\repos\rl-constrained-decision-making`

This file records the canonical verification commands used to close the P1.1 environment milestone. It is a lightweight research verification log, not a full test report.

## Repository and environment activation

```cmd
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
conda activate RL_PROJECTS
```

Purpose: ensure all commands run from the repository root with the intended Miniforge environment active.

## Pytest import-path configuration

Create `pytest.ini` at the repository root:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

Verify that the package marker exists:

```cmd
dir envs
```

Expected file:

```text
envs\__init__.py
```

If missing, create it:

```cmd
type nul > envs\__init__.py
```

Purpose: allow tests to import:

```python
from envs.constrained_navigation import ConstrainedNavigationEnv
```

## Environment class and space smoke check

```cmd
python -c "from envs.constrained_navigation import ConstrainedNavigationEnv; env=ConstrainedNavigationEnv(); print(env.observation_space.shape, env.action_space.shape)"
```

Expected properties:

```text
observation_space.shape == (21,)
action_space.shape == (2,)
```

Purpose: verify that the Gymnasium environment class imports, constructs, and exposes fixed spaces.

## Reset reproducibility smoke check

```cmd
python -c "from envs.constrained_navigation import ConstrainedNavigationEnv; env=ConstrainedNavigationEnv(); o1,_=env.reset(seed=7); o2,_=env.reset(seed=7); import numpy as np; print(o1.shape, o1.dtype, np.allclose(o1,o2))"
```

Expected properties:

```text
obs.shape == (21,)
obs.dtype == float32
same-seed reset comparison prints True
```

Purpose: verify `reset(seed=...)` produces repeatable initial observations.

## Observation and info smoke check

```cmd
python -c "from envs.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(); obs,info=env.reset(seed=0); print(obs.shape, obs.dtype, np.isfinite(obs).all(), info.keys())"
```

Expected properties:

```text
obs.shape == (21,)
obs.dtype == float32
all observation values are finite
info contains success, collision, distance_to_goal, min_obstacle_distance, step_count
```

Purpose: verify fixed observation encoding and required diagnostic fields.

## One-step Gymnasium API check

```cmd
python -c "from envs.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(); obs,info=env.reset(seed=0); out=env.step(np.array([0.1,0.0], dtype=np.float32)); print(len(out), out[0].shape, out[0].dtype, type(out[1]), type(out[2]), type(out[3]), out[4])"
```

Expected properties:

```text
step returns a 5-tuple
observation shape is (21,)
observation dtype is float32
reward is a Python float
terminated is a Python bool
truncated is a Python bool
info dictionary is present
```

Purpose: verify the modern Gymnasium `step` API.

## Dynamics sanity check

```cmd
python -c "from envs.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(); env.reset(seed=0, options={'start': np.array([0.0,0.0]), 'theta': 0.0, 'goal': np.array([10.0,0.0]), 'obstacle_mask': np.array([False,False,False])}); env.step(np.array([1.0,0.0], dtype=np.float32)); print(env.position, env.theta)"
```

Expected result if `dt = 0.1`:

```text
position approximately [0.1, 0.0]
theta approximately 0.0
```

Purpose: verify the direct math-to-code conversion for the unicycle update:

```text
x_next = x + v cos(theta) dt
y_next = y + v sin(theta) dt
theta_next = wrap(theta + omega dt)
```

## Lightweight smoke-test suite

```cmd
pytest -q tests\test_env.py
```

Actual result reported:

```text
..... [100%]
5 passed in 0.33s
```

Purpose: verify reset/step API, fixed shapes, finite values, seeded reset, forced success, forced collision, and short random rollout behavior.

## Random rollout smoke check

Canonical quick version:

```cmd
python -c "from envs.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(max_episode_steps=10); obs,info=env.reset(seed=0); outs=[env.step(env.action_space.sample()) for _ in range(10)]; print(len(outs), all(np.isfinite(o[0]).all() and np.isfinite(o[1]) for o in outs), outs[-1][4])"
```

Expected properties:

```text
first printed value is 10
second printed value is True
final printed object is an info dictionary
```

Cleaner done-aware version:

```cmd
python -c "exec('from envs.constrained_navigation import ConstrainedNavigationEnv\nimport numpy as np\nenv = ConstrainedNavigationEnv(max_episode_steps=10)\nenv.action_space.seed(0)\nobs, info = env.reset(seed=0)\nsteps = 0\ndone = False\nok = np.isfinite(obs).all()\nwhile not done:\n    action = env.action_space.sample()\n    obs, reward, terminated, truncated, info = env.step(action)\n    ok = ok and np.isfinite(obs).all() and np.isfinite(reward)\n    steps += 1\n    done = terminated or truncated\nprint(steps, ok, info)')"
```

Expected properties:

```text
ok is True
steps is positive
episode ends by termination or truncation
final info dictionary is valid
```

Purpose: verify repeated random interaction without invalid numerical values.

## Full current test suite

```cmd
python -m pytest -q
```

Purpose: run all tests discovered under `tests/` using the active `RL_PROJECTS` interpreter.

## Git closeout commands

Run after verification passes:

```cmd
git status
git diff
python -m pytest -q
git add envs\constrained_navigation.py tests\test_env.py pytest.ini docs\p1_1_environment_verification.md
git commit -m "Add constrained navigation environment"
git push -u origin p1-environment
```

Commit hash record after commit:

```cmd
git rev-parse --short HEAD
```

Commit Hash: `d0f07ec`

## Exit criteria confirmed

- `ConstrainedNavigationEnv` imports and constructs.
- `observation_space.shape == (21,)`.
- `action_space.shape == (2,)`.
- `reset(seed=...)` returns `(obs, info)`.
- `step(action)` returns `(obs, reward, terminated, truncated, info)`.
- Observations are finite `np.float32` arrays.
- Rewards are finite Python floats.
- Termination and truncation flags are Python booleans.
- Forced success and forced collision are covered by smoke checks.
- A short random rollout produces finite observations and rewards.

