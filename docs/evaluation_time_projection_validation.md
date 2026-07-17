# Evaluation-Time Predictive Action Projection Validation

## 1. Purpose and Authority

This document is the canonical validation and regression entry point for the evaluation-time predictive action-projection integration.

For integration closure, follow the command order and acceptance criteria in this document. The component-specific verification records under `docs/` remain authoritative technical references for individual mechanisms, but they are not separate validation sequences that must all be rerun in full.

The validated scope is evaluation-time predictive action projection. Training-time projection and comparative training experiments remain explicitly out of scope.

## 2. Validation Status

```text
Canonical Windows validation: PASS
Functional evaluation-time projection changes remaining: none known
Canonical validation record: complete
Ready for Git staging and repository commit: yes
Final integration commit recorded: pending repository commit
Trajectory archive metadata refresh: required for existing local NPZ artifacts
Merged to the target branch: pending repository workflow
```

All canonical validation commands in this record were reported as passing in the local Windows `RL_PROJECTS` Conda environment.

The descriptive naming update changed only repository-facing labels and the trajectory archive
format identifier. The complete lightweight suite passed after the update. Existing local trajectory
archives created before this update must be regenerated or have their metadata refreshed before
final repository closure so that they report `evaluation_trajectory_v1`.

## 3. Validated Source Identity

| Item | Value |
|---|---|
| Immediate modification baseline | Closeout-validation source tree |
| Immediate modification baseline SHA-256 | `b4283cfcdeb6b65e61baef36e79141907e661e5b491cd20e63b8be69e2f455b1` |
| Repository branch | `predictive-action-projection` |
| Embedded pre-integration Git `HEAD` | `d682a35a2606f658f3e2fc19be87f9ab7da713e8` |
| Validation date | `2026-07-16` |
| Validation host | Windows local development host |
| Conda environment | `RL_PROJECTS` |
| Final integration commit | Record after the repository commit |
| Merge or pull-request identifier | Record after the repository merge workflow |

The embedded Git `HEAD` predates the evaluation-time projection integration. The validated changes exist in the working tree and must be committed during repository closure. That pre-integration commit must not be presented as the final integration commit.

## 4. Validation Environment Evidence

The local environment evidence is stored outside normal Git under:

```text
runs\validation\projection_environment.txt
runs\validation\projection_conda_list.txt
```

The first file records the Python version, NumPy, pandas, PyTorch, Gymnasium, CVXPY, OSQP, and the CVXPY installed-solver list. The second file records the complete Conda package list.

Environment acceptance status:

```text
RL_PROJECTS Python selected: PASS
Required imports: PASS
CVXPY detects OSQP: PASS
```

The exact Windows build, Python executable path, Python version, and package versions remain in the local validation evidence and should be preserved with the run artifacts. They are not duplicated here because those values were not supplied with the source archive.

## 5. Checkpoint Identity

The canonical comparison uses the established PPO baseline checkpoint:

```text
runs\checkpoints\ppo_baseline_51200_seed1.pt
```

Expected and validated SHA-256:

```text
3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3
```

Checkpoint identity gate:

```text
Checkpoint exists: PASS
Checkpoint SHA-256 matches the established PPO baseline artifact: PASS
```

## 6. Canonical Validation Order

The following sequence is the complete evaluation-time projection validation. Component-specific commands in other documents are diagnostic references and are not additional mandatory steps after this sequence passes.

### 6.1 Activate the Repository Environment

```bat
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
conda activate RL_PROJECTS
where python
python -c "import sys; print(sys.executable); print(sys.version)"
```

Result:

```text
PASS
```

### 6.2 Record the Software Environment

```bat
if not exist runs\validation mkdir runs\validation

python -c "import sys, numpy, pandas, torch, gymnasium, cvxpy as cp, osqp; print('python',sys.version); print('numpy',numpy.__version__); print('pandas',pandas.__version__); print('torch',torch.__version__); print('gymnasium',gymnasium.__version__); print('cvxpy',cp.__version__); print('osqp',osqp.__version__); print('installed_solvers',cp.installed_solvers()); assert 'OSQP' in cp.installed_solvers()" > runs\validation\projection_environment.txt

conda list > runs\validation\projection_conda_list.txt
```

Result:

```text
PASS
```

### 6.3 Compile the Active Source Tree

```bat
python -m compileall -q environments projection evaluation algorithms tests
```

Result:

```text
PASS
No syntax errors were reported.
```

### 6.4 Run the Complete Lightweight Regression Suite

```bat
python -m pytest -q -rs
```

Result:

```text
19 passed
0 failed
0 errors
0 unexpected skips
```

This full suite supersedes the repeated focused test commands distributed across the component verification records.

### 6.5 Verify Checkpoint Compatibility with a Different Active-Obstacle Count

```bat
python -m evaluation.evaluate_policy ^
  --policy ppo ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --episodes 1 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --max-obstacles 3 ^
  --num-active-obstacles 2 ^
  --no-cuda ^
  --output runs\evaluation\checkpoint_capacity3_active2_smoke.csv
```

Result:

```text
PASS
Observation capacity remained 3.
Observation dimension remained 21.
The existing checkpoint loaded successfully.
The evaluation CSV was created.
```

### 6.6 Verify Expected Rejection When Observation Capacity Changes

```bat
del /q runs\evaluation\checkpoint_capacity_mismatch_should_not_exist.csv 2>nul

python -m evaluation.evaluate_policy ^
  --policy ppo ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --episodes 1 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --max-obstacles 5 ^
  --num-active-obstacles 2 ^
  --no-cuda ^
  --output runs\evaluation\checkpoint_capacity_mismatch_should_not_exist.csv
```

Expected diagnostic:

```text
Checkpoint obs_dim=21 does not match environment obs_dim=31.
```

Result:

```text
PASS — expected rejection occurred.
The incompatible output CSV was not created.
```

This is an intentional negative test and is not a validation failure.

### 6.7 Run the Canonical Deterministic Paired Evaluation

```bat
del /q runs\evaluation\ppo_baseline_51200_seed1_projection_pair_* 2>nul
scripts\evaluate_projection_pair.bat
```

Result:

```text
PASS
Paired projection evaluation completed successfully.
```

The command uses:

```text
checkpoint:                    runs\checkpoints\ppo_baseline_51200_seed1.pt
episodes per mode:             20
seeds:                         1000 through 1019
maximum episode steps:         200
maximum obstacle capacity:     3
active obstacles:              3
policy evaluation:             deterministic actor mean
projection lookahead distance: 0.25
projection alpha:              2.0
projection slack penalty:      1000.0
projection extra clearance:    0.0
device:                        CPU
```

The command creates six deterministic artifacts:

```text
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_disabled.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_enabled.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_disabled_trajectories.npz
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_enabled_trajectories.npz
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_paired_episodes.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_paired_summary.csv
```

### 6.8 Audit the Deterministic CSV Artifacts

```bat
python -c "import numpy as np, pandas as pd; p=r'runs\evaluation\ppo_baseline_51200_seed1_projection_pair'; off=pd.read_csv(p+'_projection_disabled.csv'); on=pd.read_csv(p+'_projection_enabled.csv'); pair=pd.read_csv(p+'_paired_episodes.csv'); s=pd.read_csv(p+'_paired_summary.csv').iloc[0]; expected=np.arange(1000,1020); assert len(off)==len(on)==len(pair)==20; np.testing.assert_array_equal(off['seed'].to_numpy(),expected); np.testing.assert_array_equal(on['seed'].to_numpy(),expected); assert off[['episode','seed']].equals(on[['episode','seed']]); assert pair[['episode','seed']].equals(off[['episode','seed']]); assert not off['projection_enabled'].any(); assert on['projection_enabled'].all(); np.testing.assert_allclose(off['episode_return'],on['episode_return'],rtol=0.0,atol=1e-5); assert off['success'].all() and on['success'].all(); assert not off['collision'].any() and not on['collision'].any(); assert int(on['projection_intervention_count'].sum())==0; assert int(on['projection_solver_failure_count'].sum())==0; assert float(on['max_projection_correction_norm'].max())<=1e-6; assert s['checkpoint_sha256']=='3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3'; assert abs(float(s['without_projection_mean_return'])-12.319942)<1e-3; assert abs(float(s['with_projection_mean_return'])-12.319942)<1e-3; assert float(s['without_projection_success_rate'])==1.0 and float(s['with_projection_success_rate'])==1.0; assert float(s['without_projection_collision_rate'])==0.0 and float(s['with_projection_collision_rate'])==0.0; print('deterministic paired CSV audit passed'); print(s.to_string())"
```

Result:

```text
PASS
Episode indices and seeds aligned across modes.
Checkpoint identity matched.
Projection flags were correct.
All deterministic episode returns matched within the audit tolerance.
No deterministic intervention or solver failure occurred.
```

Accepted deterministic reference values:

| Metric | Projection disabled | Projection enabled | Enabled minus disabled |
|---|---:|---:|---:|
| Mean return | `12.319941656945442` | `12.319941656945442` | `0.0` |
| Mean episode length | `96.0` | `96.0` | `0.0` |
| Success rate | `1.0` | `1.0` | `0.0` |
| Collision rate | `0.0` | `0.0` | `0.0` |
| Mean minimum obstacle clearance | `0.5181634403654055` | `0.5181634403654055` | `0.0` |
| Total projection interventions | not applicable | `0` | not applicable |
| Mean projection slack sum | not applicable | `0.0` | not applicable |
| Maximum projection slack | not applicable | `0.0` | not applicable |
| Solver failures | `0` | `0` | not applicable |

The generated CSV files are the authoritative Windows numerical record. The values above are the established deterministic reference values used by the audit.

Zero interventions are the expected noninterference result for the already-safe deterministic policy on the fixed default layout.

### 6.9 Audit the Deterministic Trajectory Archives

```bat
python -c "import numpy as np; p=r'runs\evaluation\ppo_baseline_51200_seed1_projection_pair'; off=np.load(p+'_projection_disabled_trajectories.npz',allow_pickle=False); on=np.load(p+'_projection_enabled_trajectories.npz',allow_pickle=False); keys=off['episode_keys'].tolist(); assert off['trajectory_archive_version'].item()=='evaluation_trajectory_v1'; assert on['trajectory_archive_version'].item()=='evaluation_trajectory_v1'; assert int(off['episode_count'])==int(on['episode_count'])==20; assert keys==on['episode_keys'].tolist(); assert all(not off[name].dtype.hasobject for name in off.files); assert all(not on[name].dtype.hasobject for name in on.files); assert all(off[k+'_positions'].shape[0]==off[k+'_action_raw_physical'].shape[0]+1 for k in keys); assert all(on[k+'_positions'].shape[0]==on[k+'_action_exec_physical'].shape[0]+1 for k in keys); assert all(np.array_equal(off[k+'_action_raw_physical'],off[k+'_action_exec_physical']) for k in keys); assert all(np.allclose(on[k+'_action_exec_physical']-on[k+'_action_raw_physical'],on[k+'_action_correction_physical'],rtol=0.0,atol=1e-12) for k in keys); assert all(set(off[k+'_projection_solver_status'].tolist())=={'disabled'} for k in keys); assert all(set(on[k+'_projection_solver_status'].tolist()).issubset({'optimal','optimal_inaccurate'}) for k in keys); assert all(on[k+'_projection_success'].all() for k in keys); print('trajectory/action audit passed for',len(keys),'episodes')"
```

Result:

```text
PASS
Both archives loaded with allow_pickle=False.
Both archives contained 20 aligned episodes.
Every episode contained T + 1 state samples and T transition samples.
Projection-disabled physical raw and executed actions were identical.
Projection-enabled corrections equaled executed actions minus raw actions.
All projection-enabled deterministic solves succeeded.
```

### 6.10 Run the Mandatory Active-Projection Diagnostic

```bat
del /q runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic_* 2>nul

python -m evaluation.evaluate_projection_pair ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --episodes 20 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --max-obstacles 3 ^
  --num-active-obstacles 3 ^
  --projection-lookahead-distance 0.25 ^
  --projection-alpha 2.0 ^
  --projection-slack-penalty 1000.0 ^
  --projection-extra-clearance 0.0 ^
  --stochastic ^
  --no-cuda ^
  --output-prefix runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic
```

Result:

```text
PASS
```

This diagnostic creates the corresponding six stochastic paired artifacts under the specified output prefix.

### 6.11 Audit the Active-Projection Diagnostic

```bat
python -c "import numpy as np, pandas as pd; p=r'runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic'; s=pd.read_csv(p+'_paired_summary.csv').iloc[0]; on=pd.read_csv(p+'_projection_enabled.csv'); z=np.load(p+'_projection_enabled_trajectories.npz',allow_pickle=False); keys=z['episode_keys'].tolist(); assert int(s['with_projection_total_interventions'])>0; assert float(s['with_projection_mean_intervention_rate'])>0.0; assert float(s['with_projection_mean_correction_norm'])>0.0; assert int(s['with_projection_total_solver_failures'])==0; assert np.isfinite(float(s['with_projection_mean_slack_sum'])); assert np.isfinite(float(s['with_projection_max_slack'])); assert any(z[k+'_projection_intervened'].any() for k in keys); assert all(np.allclose(z[k+'_action_exec_physical']-z[k+'_action_raw_physical'],z[k+'_action_correction_physical'],rtol=0.0,atol=1e-12) for k in keys); print('active projection diagnostic passed'); print(s.to_string())"
```

Result:

| Diagnostic | Acceptance result |
|---|---|
| At least one projection intervention | PASS |
| Positive mean intervention rate | PASS |
| Positive mean correction norm | PASS |
| Finite mean summed slack | PASS |
| Finite maximum slack | PASS |
| Projection solver failures | `0` |
| At least one intervened transition in the NPZ archive | PASS |
| Executed-minus-raw action identity | PASS |

The stochastic paired-summary CSV is the authoritative numerical record for the local Windows run. Exact stochastic metric magnitudes are intentionally not treated as cross-platform regression constants.

This run is an active projection-path diagnostic, not a comparative training experiment or statistical research claim.

### 6.12 Measure Projection Runtime

```bat
python -c "from time import perf_counter; import numpy as np; from projection.cbf_qp_projection import ProjectionParams, project_physical_action; params=ProjectionParams(); kw=dict(position=np.array([0.0,0.0]),heading=0.0,obstacle_centers=np.array([[0.75,0.0],[1.4,0.6],[2.0,-0.5]]),obstacle_radii=np.array([0.25,0.30,0.25]),obstacle_mask=np.array([True,True,True]),agent_radius=0.10,raw_action=np.array([1.0,0.0]),params=params); [project_physical_action(**kw) for _ in range(10)]; n=300; t0=perf_counter(); results=[project_physical_action(**kw) for _ in range(n)]; elapsed=perf_counter()-t0; assert all(r.success for r in results); print('calls=',n); print('elapsed_seconds=',elapsed); print('milliseconds_per_call=',1000.0*elapsed/n); print('calls_per_second=',n/elapsed); print('last_status=',results[-1].solver_status)"
```

Result:

```text
PASS
300 of 300 projection calls succeeded.
Measured runtime satisfied the green runtime gate: less than 20 ms per call.
```

Runtime interpretation:

| Result | Interpretation |
|---|---|
| `< 20 ms/call` with no failures | Green; proceed to training-time projection work after repository closure |
| `20–50 ms/call` with no failures | Evaluation usable; record an optimization warning before large training runs |
| `> 50 ms/call` or any failure | Investigate before training-time projection |

The exact local Windows timing remains in the command output and may be copied into this record if a point estimate is required for later reporting.

## 7. Consolidated Validation Result

The canonical validation establishes the following:

1. The projection core maps raw physical actions to bounded executed physical actions.
2. Safe deterministic policy behavior is preserved without material correction.
3. The complete stochastic evaluator path produces real interventions and nonzero corrections.
4. Solver failures are explicitly represented and use the stationary fallback.
5. Mean and maximum slack diagnostics are propagated without converting failure values to false zeros.
6. Obstacle capacity and active obstacle count remain distinct.
7. The existing 21-dimensional PPO checkpoint remains compatible when only the active count changes.
8. The checkpoint is correctly rejected when observation capacity changes to five obstacles.
9. Projection parameters are explicit and persisted in evaluation outputs.
10. Obstacle proximity is reported as collision-boundary clearance.
11. Projection-disabled and projection-enabled runs use one checkpoint, aligned seeds, and one common configuration.
12. CSV and NPZ artifacts preserve task outcomes, raw actions, executed actions, correction vectors, slack, and solver status.
13. Projection runtime is acceptable for beginning training-time projection work after repository closure.

## 8. Required Raw Validation Artifacts

The following files are first-class validation artifacts and must remain under `runs/` rather than normal Git:

```text
runs\validation\projection_environment.txt
runs\validation\projection_conda_list.txt
runs\evaluation\checkpoint_capacity3_active2_smoke.csv

runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_disabled.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_enabled.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_disabled_trajectories.npz
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_projection_enabled_trajectories.npz
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_paired_episodes.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_paired_summary.csv

runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic_projection_disabled.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic_projection_enabled.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic_projection_disabled_trajectories.npz
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic_projection_enabled_trajectories.npz
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic_paired_episodes.csv
runs\evaluation\ppo_baseline_51200_seed1_projection_pair_stochastic_paired_summary.csv
```

The deterministic paired-summary CSV is the primary compact evaluation-time projection comparison artifact. The two deterministic NPZ archives are the primary raw/executed-action audit artifacts. The stochastic paired-summary and projection-enabled NPZ archive are the primary active-intervention diagnostic artifacts.

## 9. Known Limitations and Claim Boundaries

1. The default deterministic environment layout is fixed.
2. The deterministic PPO policy is already collision-free on that layout and therefore produces zero projection interventions.
3. The stochastic run exists to exercise and audit the projection path; it is not a final comparative training result.
4. Projection performance is local to the simulated benchmark and current parameterization.
5. The CBF-QP encourages local safety but does not constitute a global safety guarantee.
6. Training-time projection has not been executed.
7. No training-time projection run, multi-seed comparative experiment, or final paper claim is included in this validation.
8. Raw CSV, NPZ, checkpoint, and package-manifest artifacts remain under `runs/` and are not normal source-control content.

## 10. Git Closure Requirements

Validation completion does not by itself complete repository closure. The following actions remain:

```text
[ ] Review semantic changes with end-of-line noise ignored.
[ ] Resolve the tracked Code.zip deletion deliberately.
[ ] Stage only intended projection source, tests, scripts, and documentation.
[ ] Exclude runs, checkpoints, CSVs, NPZ files, caches, and IDE metadata.
[ ] Run git diff --cached --check.
[ ] Commit the evaluation-time projection integration.
[ ] Record the final integration commit hash in this document or merge metadata.
[ ] Rerun python -m pytest -q -rs after the commit.
[ ] Confirm 19 passed after the commit.
[ ] Confirm git status --short is clean.
[ ] Merge through the normal repository workflow.
[ ] Rerun the complete lightweight suite on the merged tree.
```

Suggested commit message:

```text
Complete evaluation-time predictive action projection
```

The evaluation-time projection integration is formally closed when the canonical validation remains passing, the intended changes are committed, the working tree is clean, and the feature branch is merged.

## 11. Component Reference Index

The following records provide mechanism-specific detail. They are subordinate references for diagnosis and audit; this document defines the canonical validation command order.

| Record | Scope |
|---|---|
| `docs/constrained_navigation_verification.md` | Environment mechanics and obstacle-capacity behavior |
| `docs/ppo_baseline_verification.md` | PPO baseline checkpoint and behavior |
| `docs/cbf_qp_projection_verification.md` | Numerical projection geometry, bounds, intervention, and fallback |
| `docs/projection_evaluation_parameters_verification.md` | Projection CLI parameters and persisted metadata |
| `docs/obstacle_clearance_metric_verification.md` | Collision-boundary clearance semantics |
| `docs/projection_failure_diagnostics_verification.md` | Failed-solve status and undefined slack behavior |
| `docs/projection_slack_metrics_verification.md` | Mean and maximum slack aggregation |
| `docs/paired_projection_evaluation_verification.md` | Same-checkpoint paired evaluation and paired CSVs |
| `docs/trajectory_audit_verification.md` | NPZ schema and raw/executed-action audit |

## 12. Final Gate

```text
[PASS] Source archive identity verified.
[PASS] RL_PROJECTS validation environment used.
[PASS] OSQP available through CVXPY.
[PASS] Checkpoint SHA-256 verified.
[PASS] Complete source compilation passed.
[PASS] Full lightweight suite reported 19 passed.
[PASS] Active-obstacle-count compatibility preserved.
[PASS] Observation-capacity mismatch rejected.
[PASS] Deterministic paired evaluation created six artifacts.
[PASS] Deterministic seeds and episode keys aligned.
[PASS] Deterministic PPO baseline behavior was preserved.
[PASS] Deterministic projection reported zero interventions and zero failures.
[PASS] Both deterministic trajectory archives passed the action/state audit.
[PASS] Stochastic diagnostic produced real projection intervention.
[PASS] Stochastic diagnostic reported zero solver failures.
[PASS] Projection runtime satisfied the green runtime gate.
[PASS] Canonical evaluation-time projection validation record created.
[PENDING] Existing local trajectory archives regenerated or metadata-refreshed.
[PENDING] Final integration commit recorded.
[PENDING] Clean working tree confirmed after commit.
[PENDING] Evaluation-time projection integration merged through the repository workflow.
```

No training-time projection work should begin until the three pending repository-closure items are complete.
