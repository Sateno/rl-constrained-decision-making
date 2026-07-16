# Constrained Navigation Verification

## Scope

Component: `environments.constrained_navigation.ConstrainedNavigationEnv`  
Environment: `RL_PROJECTS`  
Repository root: `C:\rl_projects\src\repos\rl-constrained-decision-making`

This record contains canonical verification commands for the base constrained-navigation environment. Repository setup, directory creation, file creation, Git operations, and one-off debugging commands are excluded.

This document verifies the base environment with physical action bounds. Factory-level wrappers, including normalized PPO action mapping, are verified in the PPO baseline verification record.

## Setup

```cmd
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
conda activate RL_PROJECTS
```

## Syntax Check

```cmd
python -m py_compile environments\constrained_navigation.py
```

Expected result:

```text
no output
```

## Environment Space and Bound Check

```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; env=ConstrainedNavigationEnv(); print(env.observation_space.shape, env.action_space.shape); print(env.action_space.low, env.action_space.high); env.close()"
```

Expected result:

```text
(21,) (2,)
[ 0. -2.] [1. 2.]
```

## Reset and Step API Check

```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(); obs,info=env.reset(seed=0); out=env.step(np.array([0.1,0.0], dtype=np.float32)); print(len(out), out[0].shape, out[0].dtype, type(out[1]), type(out[2]), type(out[3]), out[4]); env.close()"
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

## Unicycle Dynamics Sanity Check

```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(); env.reset(seed=0, options={'start': np.array([0.0,0.0]), 'theta': 0.0, 'goal': np.array([10.0,0.0]), 'obstacle_mask': np.array([False,False,False])}); env.step(np.array([1.0,0.0], dtype=np.float32)); print(env.position, env.theta); env.close()"
```

Expected result for `dt = 0.1`:

```text
position approximately [0.1, 0.0]
theta approximately 0.0
```

## Obstacle Capacity and Active Count Check

```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; env=ConstrainedNavigationEnv(max_obstacles=5, num_active_obstacles=2); obs,info=env.reset(seed=0); print(env.observation_space.shape, obs.shape); print(env.obstacle_mask, int(env.obstacle_mask.sum())); env.close()"
```

Expected result:

```text
(31,) (31,)
[ True  True False False False] 2
```

The default configuration remains `max_obstacles=3` with three active obstacles and observation shape `(21,)`.

## Obstacle Clearance Metric Check

The canonical safety metric is:

```text
min_obstacle_clearance
    = min over active obstacles of
      center_distance - (obstacle_radius + agent_radius)
```

Positive values indicate separation, zero indicates contact with the collision boundary, and negative values indicate penetration.

```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(agent_radius=0.10); centers=np.zeros((3,2)); radii=np.zeros(3); mask=np.zeros(3,dtype=bool); centers[0]=[1.0,0.0]; radii[0]=0.25; mask[0]=True; _,info=env.reset(seed=0, options={'start':np.array([0.0,0.0]),'theta':0.0,'goal':np.array([5.0,0.0]),'obstacle_centers':centers,'obstacle_radii':radii,'obstacle_mask':mask}); print(info['min_obstacle_clearance']); assert abs(info['min_obstacle_clearance']-0.65)<1e-12; env.close()"
```

Expected result:

```text
0.65
```

When no obstacles are active, the clearance metric is undefined and is reported as `NaN`; collision remains `False`.

```cmd
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(num_active_obstacles=0); _,info=env.reset(seed=0); print(info['min_obstacle_clearance'], info['collision']); assert np.isnan(info['min_obstacle_clearance']); assert info['collision'] is False; env.close()"
```

Expected result:

```text
nan False
```

## Lightweight Environment Tests

```cmd
python -m pytest -q tests\test_constrained_navigation.py
```

Expected result:

```text
all current environment tests pass
```

## Random Rollout Smoke Check

```cmd
python -c "exec('from environments.constrained_navigation import ConstrainedNavigationEnv\nimport numpy as np\nenv = ConstrainedNavigationEnv(max_episode_steps=10)\nenv.action_space.seed(0)\nobs, info = env.reset(seed=0)\nsteps = 0\ndone = False\nok = np.isfinite(obs).all()\nwhile not done:\n    action = env.action_space.sample()\n    obs, reward, terminated, truncated, info = env.step(action)\n    ok = ok and np.isfinite(obs).all() and np.isfinite(reward)\n    steps += 1\n    done = terminated or truncated\nprint(steps, ok, info)\nenv.close()')"
```

Expected properties:

```text
step count is positive
finite-value check is True
final info dictionary is valid
episode ends by termination or truncation
```

## Exit Criteria

The constrained-navigation environment is verified when:

- The module compiles.
- The environment constructs successfully.
- Observation and action spaces have the expected shapes.
- The base physical action bounds are `[0, -omega_max]` and `[v_max, omega_max]`.
- `reset` and `step` satisfy the Gymnasium API.
- The direct dynamics sanity check matches the expected forward step.
- The obstacle metric reports signed collision-boundary clearance and uses `NaN` when no obstacle is active.
- Lightweight environment tests pass.
- A short random rollout produces finite observations and rewards.
