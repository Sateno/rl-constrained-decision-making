# Environment and Projection Contract

## Scope

This document defines the stable environment, action, reward, geometry, projection, and PPO-interaction contracts implemented in:

```text
environments/constrained_navigation.py
environments/action_wrappers.py
environments/factory.py
projection/cbf_qp_projection.py
projection/cbf_qp_wrapper.py
algorithms/ppo/ppo_continuous_action.py
```

## Environment state and observation

`ConstrainedNavigationEnv` owns position, heading, goal, obstacle geometry, active-obstacle mask, and episode step count.

Observation dimension:

\[
d_{\mathrm{obs}} = 6 + 5N_{\max}.
\]

Global features:

```text
x, y, cos(theta), sin(theta), goal_x - x, goal_y - y
```

Per-obstacle slot:

```text
relative_x, relative_y, signed collision clearance, radius, active mask
```

Inactive slots are zeros. `max_obstacles` fixes neural-network input capacity; `num_active_obstacles` changes only the built-in mask.

Default contract:

```text
max_obstacles=3
num_active_obstacles=3
observation shape=(21,)
dtype=float32
```

Changing `max_obstacles` changes checkpoint compatibility. Changing only the active count does not.

## Action interfaces

Physical control:

\[
u=[v,\omega]^\top,
\qquad
0\leq v\leq v_{\max},
\qquad
-\omega_{\max}\leq\omega\leq\omega_{\max}.
\]

Defaults:

```text
v_max=1.0
omega_max=2.0
```

The policy-facing normalized action is `a in [-1, 1]^2`:

\[
v=\tfrac12 v_{\max}(a_v+1),
\qquad
\omega=\omega_{\max}a_\omega.
\]

Thus a policy mean near zero maps to forward motion `[0.5 * v_max, 0]`.

## Dynamics, reward, and episode flags

Unicycle step:

\[
\begin{aligned}
x_{t+1}&=x_t+v_t\cos\theta_t\Delta t,\\
y_{t+1}&=y_t+v_t\sin\theta_t\Delta t,\\
\theta_{t+1}&=\operatorname{wrap}(\theta_t+\omega_t\Delta t).
\end{aligned}
\]

Default `dt=0.1`.

Reward:

\[
r_t=c_p(d_{t-1}-d_t)-c_u(v_t^2+\omega_t^2)-c_{\mathrm{time}}
+r_{\mathrm{goal}}\mathbf1_{\mathrm{success}}
-c_{\mathrm{collision}}\mathbf1_{\mathrm{collision}},
\]

plus a distance-weighted timeout penalty on truncation.

| Parameter | Default |
|---|---:|
| progress weight | `1.0` |
| action penalty | `0.01` |
| time penalty | `0.01` |
| goal reward | `10.0` |
| collision penalty | `10.0` |
| timeout-distance penalty | `1.0` |
| goal radius | `0.25` |
| episode limit | `200` |

The collision penalty is the only reward term intentionally varied by the high-penalty method.

```text
terminated = success or collision
truncated  = episode limit reached without termination
```

## Collision clearance

Collision radius:

\[
R_i^{\mathrm{collision}}=r_i+r_{\mathrm{agent}}.
\]

Signed clearance:

\[
d_{i,t}^{\mathrm{clear}}=\|p_t-o_i\|_2-R_i^{\mathrm{collision}}.
\]

```text
positive   separated
zero       boundary contact
negative   penetration
```

`min_obstacle_clearance` is the minimum signed clearance over active obstacles. It is not center distance.

No active obstacles:

```text
min_obstacle_clearance=NaN
collision=False
```

Evaluation averages only finite clearances.

## Wrapper and action ownership

Construction order:

```text
ConstrainedNavigationEnv
-> CbfQpProjectionWrapper, optional
-> NormalizedActionWrapper, optional
-> RecordEpisodeStatistics, optional
```

Action path:

```text
raw normalized policy action
-> bounded physical raw action
-> physical executed action after optional projection
-> environment transition
```

Ownership:

```text
policy distribution        raw normalized action
normalized wrapper         normalized-to-physical mapping
projection wrapper         raw-to-executed physical action
environment                state, reward, termination
```

## Projection geometry and QP

Lookahead point and action Jacobian:

\[
p_L=p+L[\cos\theta,\sin\theta]^\top,
\]

\[
J_L(\theta)=
\begin{bmatrix}
\cos\theta&-L\sin\theta\\
\sin\theta&L\cos\theta
\end{bmatrix}.
\]

Projection radius:

\[
R_i^{\mathrm{projection}}
=r_i+r_{\mathrm{agent}}+d_{\mathrm{extra}}.
\]

`extra_clearance` affects projection only; it does not alter collision detection.

Barrier and softened constraint:

\[
h_i=\|p_L-o_i\|_2^2-(R_i^{\mathrm{projection}})^2,
\]

\[
\nabla h_i^\top J_L(\theta)u+\xi_i\geq-\alpha h_i,
\qquad \xi_i\geq0.
\]

Projection objective:

\[
\min_{u,\xi}
\frac12\|u-u_{\mathrm{raw}}\|_2^2
+
\frac{\rho}{2}\|\xi\|_2^2
\]

subject to physical action bounds and one softened CBF inequality per active obstacle.

| Parameter | Default | Constraint |
|---|---:|---|
| lookahead distance | `0.25` | nonnegative |
| alpha | `2.0` | positive |
| slack penalty | `1000.0` | positive |
| extra clearance | `0.0` | nonnegative |
| correction tolerance | `1e-6` | nonnegative |
| solver | `OSQP` | nonempty name |

## Projection diagnostics

`ProjectionResult` records:

```text
action_raw, action_exec, correction, correction_norm, intervened
slack_values, slack_sum, slack_max, active_constraint_count
solver_status, success, objective_value
```

The wrapper expands active slack values into a fixed-capacity slot-aligned vector.

An intervention occurs when correction norm exceeds `correction_tolerance`.

## Failure behavior

No active constraints:

```text
action_exec=clipped raw action
solver_status=no_active_constraints
success=True
```

Successful solve:

```text
bounded finite action
finite nonnegative slack
status optimal or optimal_inaccurate
success=True
```

Failed or invalid solve:

```text
action_exec=[0.0, 0.0]
active slack=NaN
slack_sum=NaN
slack_max=NaN
success=False
```

A failed solve is never represented as successful zero slack. Projection-enabled training aborts on solver failure.

## PPO interaction invariant

PPO stores and scores the sampled raw normalized action. The wrapper stack may execute a different physical action, and reward/next state arise from that executed action.

```text
PPO likelihood action = raw normalized action
environment action    = executed physical action
```

This is an empirical projected-environment formulation, not a corrected projected policy-gradient estimator.

The vector trainer uses Gymnasium same-step autoreset. Terminal episode and projection data are read from `final_info`, preventing reset-only calls from entering the rollout as ordinary transitions.

## Change-control boundary

Contract review and regression validation are required for changes to observation ordering, action mapping, dynamics, reward semantics, clearance meaning, projection geometry, QP formulation, stationary fallback, or raw-action PPO likelihood semantics.

Geometry references:

- [Original training layout](../assets/layouts/original_training_layout.pdf)
- [Frozen core layout suite](../assets/layouts/core_navigation_layouts_visual_reference.pdf)
