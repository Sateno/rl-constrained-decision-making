# Predictive Action Projection with PPO: Analysis Command Record

> **Completed-study context, 7 September 2026:** This is a chronological record.
> Statements about pending analysis, historical commands, temporary outputs, or
> retired paper assets describe the stage at which they were written. Current
> findings are in the [final study record](final_study_record.md), and the merge
> checks are in the [release verification guide](../validation/release_verification.md).


**Author:** Salvador Tenorio\
**Status:** Living command record\
**Frozen source:** `ba64926aed98b08b7b285266cf85989d466f9f1c`\
**Repository path:** `docs/records/Predictive_Action_Projection_Analysis_Command_Record.md`\
**Started:** 2026-08-22

## Purpose and scope

This record collects the exact commands used to build, verify, audit, and inspect the frozen study results across the implementation and scientific-interpretation workflows. It is intended to accompany `Predictive_Action_Projection_Analysis_Record.md`.

The historical build commands consume frozen checkpoints, evaluation CSV/NPZ artifacts, trajectory archives, and saved TensorBoard records. They perform no training or evaluation. References in the older sections to implementation Steps 5 and 6 use that historical workflow's numbering; the later scientific-interpretation sections explicitly identify their own phase and step.

Run all commands from the repository root with the `RL_PROJECTS` environment active. Result builders refuse nonempty output directories; the build commands are therefore for a fresh result branch or fresh output tree, not for overwriting an existing build.

## 1. Source identity precheck

```bat
conda activate RL_PROJECTS
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making

git branch --show-current
git status --short
git rev-parse HEAD
git rev-list -n 1 predictive-action-projection-protocol-v1
```

Expected frozen commit and tag target:

```text
ba64926aed98b08b7b285266cf85989d466f9f1c
```

The working tree should be clean before generating results. The result branch used for the campaign is `final_evaluation_runs`.

## 2. Primary fixed-training-geometry result build

### 2.1 Generate primary tables

```bat
if exist "results\tables\fixed_training_geometry" (echo STOP: output directory exists) else echo READY
```

If the result is `READY`:

```bat
python -m analysis.aggregate_projection_results ^
  --protocol experiments\fixed_training_geometry_analysis_protocol.json ^
  --evaluation-dir runs\evaluation\final\fixed_training_geometry ^
  --output-dir results\tables\fixed_training_geometry
```

### 2.2 Generate primary figures and training diagnostics

```bat
if exist "results\figures\fixed_training_geometry" (echo STOP: output directory exists) else echo READY
```

If the result is `READY`:

```bat
python -m analysis.plot_projection_results ^
  --protocol experiments\fixed_training_geometry_analysis_protocol.json ^
  --tables-dir results\tables\fixed_training_geometry ^
  --evaluation-dir runs\evaluation\final\fixed_training_geometry ^
  --figures-dir results\figures\fixed_training_geometry ^
  --runs-dir runs
```

## 3. Secondary core-layout-transfer result build

### 3.1 Generate transfer tables

```bat
if exist "results\tables\core_layout_transfer" (echo STOP: output directory exists) else echo READY
```

If the result is `READY`:

```bat
python -m analysis.aggregate_projection_results ^
  --protocol experiments\projection_analysis_protocol.json ^
  --evaluation-dir runs\evaluation\final\core_layout_transfer ^
  --output-dir results\tables\core_layout_transfer
```

### 3.2 Generate transfer figures and training diagnostics

```bat
if exist "results\figures\core_layout_transfer" (echo STOP: output directory exists) else echo READY
```

If the result is `READY`:

```bat
python -m analysis.plot_projection_results ^
  --protocol experiments\projection_analysis_protocol.json ^
  --tables-dir results\tables\core_layout_transfer ^
  --evaluation-dir runs\evaluation\final\core_layout_transfer ^
  --figures-dir results\figures\core_layout_transfer ^
  --runs-dir runs
```

## 4. Verify generated build audits

### 4.1 Primary build

```bat
python -c "import json; from pathlib import Path; t=json.loads(Path(r'results\tables\fixed_training_geometry\result_build_audit.json').read_text()); f=json.loads(Path(r'results\figures\fixed_training_geometry\figure_build_audit.json').read_text()); assert t['status']=='PASS'; assert t['layout_count']==1; assert t['selected_csv_count']==30; assert t['episode_row_count']==3000; assert t['checkpoint_row_count']==30; assert t['method_row_count']==6; assert t['projection_solver_failure_count']==0; assert f['status']=='PASS'; assert len(f['generated'])==27; assert not f['skipped']; print('PASS: primary tables and figures verified')"
```

### 4.2 Transfer build

```bat
python -c "import json; from pathlib import Path; t=json.loads(Path(r'results\tables\core_layout_transfer\result_build_audit.json').read_text()); f=json.loads(Path(r'results\figures\core_layout_transfer\figure_build_audit.json').read_text()); assert t['status']=='PASS'; assert t['layout_count']==24; assert t['selected_csv_count']==30; assert t['episode_row_count']==720; assert t['checkpoint_row_count']==30; assert t['method_row_count']==6; assert t['projection_solver_failure_count']==0; assert f['status']=='PASS'; assert len(f['generated'])==27; assert not f['skipped']; print('PASS: transfer tables and figures verified')"
```

## 5. Inspect canonical method and paired summaries

### 5.1 Primary method summary

```bat
type results\tables\fixed_training_geometry\generated_method_summary.tex
```

### 5.2 Primary paired projection effects

```bat
type results\tables\fixed_training_geometry\generated_paired_projection_deltas.tex
```

### 5.3 Transfer method summary

```bat
type results\tables\core_layout_transfer\generated_method_summary.tex
```

### 5.4 Transfer paired projection effects

```bat
type results\tables\core_layout_transfer\generated_paired_projection_deltas.tex
```

## 6. Inspect checkpoint-level outcomes

### 6.1 Primary stochastic evaluation

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\checkpoint_summary.csv'); p['timeout_rate']=1.0-p['success_rate']-p['collision_rate']; c=['display_name','train_seed','projection_mode','episode_return','success_rate','collision_rate','timeout_rate','projection_intervention_rate']; print(p[c].round(3).to_string(index=False))"
```

### 6.2 Deterministic core-layout transfer

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\core_layout_transfer\checkpoint_summary.csv'); p['timeout_rate']=1.0-p['success_rate']-p['collision_rate']; c=['display_name','train_seed','projection_mode','episode_return','success_rate','collision_rate','timeout_rate','projection_intervention_rate']; print(p[c].round(3).to_string(index=False))"
```

## 7. Inspect primary action-bound clipping

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\method_summary.csv'); c=['display_name','projection_mode','action_bound_clipping_rate_mean','action_bound_clipping_rate_std','speed_action_bound_clipping_rate_mean','speed_action_bound_clipping_rate_std','turn_rate_action_bound_clipping_rate_mean','turn_rate_action_bound_clipping_rate_std','action_bound_clipping_norm_mean','action_bound_clipping_norm_std']; print(p[c].round(3).to_string(index=False))"
```

## 8. Inspect transfer action-bound clipping

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\core_layout_transfer\method_summary.csv'); c=['display_name','projection_mode','action_bound_clipping_rate_mean','action_bound_clipping_rate_std','speed_action_bound_clipping_rate_mean','speed_action_bound_clipping_rate_std','turn_rate_action_bound_clipping_rate_mean','turn_rate_action_bound_clipping_rate_std','action_bound_clipping_norm_mean','action_bound_clipping_norm_std']; print(p[c].round(3).to_string(index=False))"
```

## 9. Inspect checkpoint-level transfer clipping

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\core_layout_transfer\checkpoint_summary.csv'); c=['display_name','train_seed','projection_mode','action_bound_clipping_rate','speed_action_bound_clipping_rate','turn_rate_action_bound_clipping_rate','action_bound_clipping_norm']; print(p[c].round(3).to_string(index=False))"
```

## 10. Inspect checkpoint-level primary clipping

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\checkpoint_summary.csv'); c=['display_name','train_seed','projection_mode','action_bound_clipping_rate','speed_action_bound_clipping_rate','turn_rate_action_bound_clipping_rate','action_bound_clipping_norm']; print(p[c].round(3).to_string(index=False))"
```

## 11. Inventory generated training scalar tags

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\training_scalar_events.csv'); print('columns:',p.columns.tolist()); c=next((x for x in p.columns if x.lower() in {'tag','metric','scalar','name'}),None); print('metric column:',c); print('\n'.join(sorted(p[c].dropna().astype(str).unique())) if c else p.head(10).to_string(index=False))"
```

The table includes schema version, event index, step, method, display name, training seed, checkpoint SHA-256, training-projection flag, and run-directory provenance columns. It contains no policy standard-deviation or log-standard-deviation tag. Final policy variance must therefore be inspected from the frozen checkpoint parameters.

## 12. Inspect one frozen checkpoint's structure

```bat
python -c "import torch; p=r'runs\checkpoints\final\ppo_baseline_51200_seed1.pt'; x=torch.load(p,map_location='cpu',weights_only=True); print('type:',type(x).__name__); [print(k,type(v).__name__,list(v.keys()) if isinstance(v,dict) else (tuple(v.shape) if hasattr(v,'shape') else repr(v)[:100])) for k,v in x.items()]"
```

The checkpoint is a dictionary, and its `agent_state_dict` contains the state-independent `actor_logstd` parameter required for the final policy-variance diagnostic.

## 13. Extract final policy standard deviations from all checkpoints

```bat
python -c "from pathlib import Path; import torch,pandas as pd; root=Path(r'runs\checkpoints\final'); ps=sorted(root.glob('ppo_*_51200_seed*.pt')); assert len(ps)==15,f'expected 15 checkpoints, found {len(ps)}'; loaded=[(p,torch.load(p,map_location='cpu',weights_only=True)) for p in ps]; params=[(p,x,x['agent_state_dict']['actor_logstd'].detach().cpu().reshape(-1)) for p,x in loaded]; assert all(len(a)==2 and x['global_step']==51200 and x['action_dim']==2 for p,x,a in params); rows=[{'method':x['args']['method'],'seed':x['args']['seed'],'logstd_speed':float(a[0]),'std_speed':float(a[0].exp()),'logstd_turn':float(a[1]),'std_turn':float(a[1].exp()),'checkpoint':p.name} for p,x,a in params]; d=pd.DataFrame(rows).sort_values(['method','seed']); print(d.round(6).to_string(index=False)); print('PASS:',len(d),'frozen checkpoints')"
```

The command verified all 15 frozen checkpoints. Final speed σ spans 0.978563–1.015498, and final turn-rate σ spans 0.922107–0.989746.

## 14. Inspect the aggregated training-curve table schema

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\training_curve_points.csv'); print('columns:',p.columns.tolist()); print('rows:',len(p)); print(p.head(12).to_string(index=False))"
```

The table uses schema `training_diagnostics_v1`, contains 14,811 rows, and provides method/tag curves with aligned step coordinates, across-seed mean and sample standard deviation, and contributing seed count.

## 15. Inventory training-curve coverage

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\training_curve_points.csv'); q=p.groupby(['display_name','tag'],as_index=False).agg(points=('step','size'),step_min=('step','min'),step_max=('step','max'),seed_min=('seed_count','min'),seed_max=('seed_count','max')); print(q.round(3).to_string(index=False))"
```

Every curve includes all five seeds. Rollout-level clipping and projection curves contain 50 points through step 51,200; episodic and rolling-outcome curves extend to each method's final completed episode near the same boundary.

## 16. Compare the first and last 20% of core training curves

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\training_curve_points.csv'); tags=['charts/episodic_return','training/rolling_success_rate','training/rolling_collision_rate','action_bounds/clipping_frequency']; f=p[p['tag'].isin(tags)].copy(); f['max_step']=f.groupby(['display_name','tag'])['step'].transform('max'); w=f[(f['step']<=0.2*f['max_step'])|(f['step']>=0.8*f['max_step'])].copy(); w['window']=w.apply(lambda r:'early' if r['step']<=0.2*r['max_step'] else 'late',axis=1); q=w.groupby(['display_name','tag','window'])['value_mean'].mean().unstack('window'); q['late_minus_early']=q['late']-q['early']; q['final']=f.sort_values('step').groupby(['display_name','tag']).tail(1).set_index(['display_name','tag'])['value_mean']; print(q.reset_index().round(3).to_string(index=False))"
```

This descriptive comparison established partial baseline learning, conservative non-completion under the high collision penalty, and strong protected-task learning for projection-trained PPO. Action-bound clipping increased for every method and therefore does not explain their competence differences. The comparison also showed that a single method-level budget verdict is not supported.

## 17. Compare the final two training deciles by seed

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\training_scalar_events.csv'); f=p[p['tag'].isin(['charts/episodic_return','safety/success'])].copy(); f['max_step']=f.groupby(['display_name','train_seed','tag'])['step'].transform('max'); f=f[f['step']>=0.8*f['max_step']].copy(); f['window']=f.apply(lambda r:'80-90%' if r['step']<0.9*r['max_step'] else '90-100%',axis=1); q=f.groupby(['display_name','train_seed','tag','window'])['value'].mean().unstack('window'); q['second_minus_first']=q['90-100%']-q['80-90%']; print(q.reset_index().round(3).to_string(index=False))"
```

The baseline tail is heterogeneous: seed 3 improves strongly, seed 4 shows weak emergence, and the other three checkpoints do not gain success. Every high-penalty checkpoint has zero success in both windows. Projection-trained PPO improves in four seeds, while seed 4 remains near its approximately 97% ceiling. This completes the general convergence inspection.

## 18. Summarize projection burden during training

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\training_curve_points.csv'); tags=['projection/intervention_frequency','projection/correction_norm','projection/correction_norm_max','projection/slack_sum','projection/slack_max']; f=p[(p['method']=='ppo_train_projection')&p['tag'].isin(tags)].copy(); f['max_step']=f.groupby('tag')['step'].transform('max'); w=f[(f['step']<=0.2*f['max_step'])|(f['step']>=0.8*f['max_step'])].copy(); w['window']=w.apply(lambda r:'early' if r['step']<=0.2*r['max_step'] else 'late',axis=1); q=w.groupby(['tag','window'])['value_mean'].mean().unstack('window'); q['late_minus_early']=q['late']-q['early']; q['final']=f.sort_values('step').groupby('tag').tail(1).set_index('tag')['value_mean']; print(q.reset_index().round(6).to_string(index=False)); e=pd.read_csv(r'results\tables\fixed_training_geometry\training_scalar_events.csv'); s=e[e['tag']=='projection/solver_failure_count']; print('solver_failure_sum:',int(s['value'].sum()),'events:',len(s),'seeds:',s['train_seed'].nunique())"
```

Intervention frequency rises from 0.139 early to 0.393 late and finishes at 0.411; correction norm rises from 0.074 to 0.209 and finishes at 0.217. Maximum correction and maximum slack remain comparatively stable. The summed solver-failure count is zero across all 250 five-seed rollout records. Numerical training-curve inspection is complete.

## 19. Inspect the primary representative-trajectory selection manifest

```bat
python -c "import pandas as pd; p=pd.read_csv(r'results\tables\fixed_training_geometry\representative_trajectory_selection.csv'); print('columns:',p.columns.tolist()); print('rows:',len(p)); print(p.to_string(index=False))"
```

The manifest contains six fixed-geometry selections: projection off/on for each method. All use training seed 1, evaluation seed 10000, and episode 0; checkpoint hashes match within every off/on pair. The corresponding plot is a reproducible illustration, not a statistical representation of the evaluation distribution.

## 20. Package tracked tables and figures for external review

First ensure the temporary handoff archive does not already exist:

```bat
if exist "predictive_action_projection_results_bundle.zip" (echo STOP: bundle already exists) else echo READY
```

If the result is `READY`:

```bat
powershell -NoProfile -Command "Compress-Archive -Path 'results\tables','results\figures' -DestinationPath 'predictive_action_projection_results_bundle.zip' -CompressionLevel Optimal"
```

This compact archive contains the generated tables, LaTeX fragments, audits, CSV diagnostics, and PDFs for both evaluation suites. It excludes checkpoints, raw `runs\evaluation` evidence, trajectory NPZ archives, and calibration evidence. The archive is a temporary review handoff and should not be committed.

## 21. Join each trajectory-selection manifest to its exact episode evidence

Primary fixed geometry:

```bat
python -c "import pandas as pd; s=pd.read_csv(r'results\tables\fixed_training_geometry\representative_trajectory_selection.csv'); e=pd.read_csv(r'results\tables\fixed_training_geometry\evaluation_episode_results.csv'); k=['method','train_seed','projection_mode','layout_id','evaluation_seed','checkpoint_sha256','episode']; q=s.merge(e,on=k,how='left',validate='one_to_one'); q['outcome']=q.apply(lambda r:'success' if r['success'] else ('collision' if r['collision'] else 'timeout'),axis=1); c=['method','train_seed','projection_mode','layout_id','evaluation_seed','episode','outcome','episode_length','episode_return','final_distance_to_goal','min_obstacle_clearance','action_bound_clipping_rate','projection_intervention_rate','mean_projection_correction_norm','max_projection_correction_norm','mean_projection_slack_sum','max_projection_slack','projection_solver_failure_count']; print('selection duplicates:',int(s.duplicated(k).sum()),'episode duplicates:',int(e.duplicated(k).sum()),'matches:',len(q)); print(q[c].round(6).to_string(index=False))"
```

Core-layout transfer:

```bat
python -c "import pandas as pd; s=pd.read_csv(r'results\tables\core_layout_transfer\representative_trajectory_selection.csv'); e=pd.read_csv(r'results\tables\core_layout_transfer\evaluation_episode_results.csv'); k=['method','train_seed','projection_mode','layout_id','evaluation_seed','checkpoint_sha256','episode']; q=s.merge(e,on=k,how='left',validate='one_to_one'); q['outcome']=q.apply(lambda r:'success' if r['success'] else ('collision' if r['collision'] else 'timeout'),axis=1); c=['method','train_seed','projection_mode','layout_id','evaluation_seed','episode','outcome','episode_length','episode_return','final_distance_to_goal','min_obstacle_clearance','action_bound_clipping_rate','projection_intervention_rate','mean_projection_correction_norm','max_projection_correction_norm','mean_projection_slack_sum','max_projection_slack','projection_solver_failure_count']; print('selection duplicates:',int(s.duplicated(k).sum()),'episode duplicates:',int(e.duplicated(k).sum()),'matches:',len(q)); print(q[c].round(6).to_string(index=False))"
```

Both joins are unique and complete (`6/6`). In both suites, the selected baseline and high-penalty pairs are identical timeouts with zero intervention. The selected projection-trained pair changes from collision to timeout under projection; in transfer, the protected projector intervenes on every step.

## 22. Audit PDF authorship metadata

Run in an environment containing `pypdf`:

```bat
python -c "from pathlib import Path; from pypdf import PdfReader; paths=sorted(Path(r'results\figures').rglob('*.pdf')); bad=[]; [(bad.append((str(p),dict(PdfReader(str(p)).metadata or {}).get('/Author'))) if dict(PdfReader(str(p)).metadata or {}).get('/Author')!='Salvador Tenorio' else None) for p in paths]; print('pdfs:',len(paths),'author failures:',len(bad)); print('\n'.join(f'{p}: {a}' for p,a in bad))"
```

The uploaded build contains 44 PDFs and all 44 fail the requirement because `/Author` is absent. Creator and producer identify Matplotlib; no OpenAI or ChatGPT attribution is present.

The PDFs were also inspected with Poppler's `pdffonts`. They embed Type 3 DejaVu Sans fonts without Unicode mapping. Regenerate with `matplotlib.rcParams['pdf.fonttype'] = 42`.

## 23. Detect duplicated training outputs across result suites

```bat
python -c "from pathlib import Path; import hashlib; root=Path(r'results'); names=['training_scalar_events.csv','training_episode_diagnostics.csv','training_rollout_diagnostics.csv','training_curve_points.csv']; h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); [(print(n,h(root/'tables'/'fixed_training_geometry'/n)==h(root/'tables'/'core_layout_transfer'/n))) for n in names]"
```

All four comparisons are `True`. The ten transfer `training_*.pdf` files were separately rasterized and compared; every rendered page is pixel-identical to the fixed-geometry counterpart. These are shared training diagnostics, not transfer-specific evidence, and should be emitted only once.

## 24. Review bundle completeness and visual rendering

The temporary bundle was safely extracted outside the repository and inventoried before inspection. It contained 80 archive members, 74 regular files, and 44 one-page PDFs. Both result-table audits and both figure-build audits report `PASS`; all PDFs render without corruption, clipped panels, blank pages, or missing visible glyphs.

This external review did not modify repository files. Its verified scientific and presentation findings are recorded in the living analysis record.

## 25. Validate the presentation-remediation patch before applying it

Save the supplied patch outside the repository as:

```text
C:\rl_projects\tools\Predictive_Action_Projection_Figure_Remediation.patch
```

From the repository root, run only the non-mutating preflight:

```bat
git apply --check "C:\rl_projects\tools\Predictive_Action_Projection_Figure_Remediation.patch"
```

Expected output: none, with exit code zero. Do not apply the patch if this command prints an error.

The reviewed patch SHA-256 is:

```text
f7b5ba074c2844468a764c31d411afb300050b28a62b486aef108773815051ae
```

The patch modifies only `analysis\plot_projection_results.py` and `tests\test_result_aggregation.py`. Application, testing, clean result regeneration, and final QA remain separate subsequent steps.

An independent blocker-only review of the final patch found no remaining blocker after strict timeout classification, endpoint draw order, stale-table refusal, and audit-scope handling were verified.

## 26. Install the reviewed complete files

The preferred handoff is now:

```text
Predictive_Action_Projection_Figure_Remediation_Full_Files.zip
```

It contains exactly:

```text
analysis\plot_projection_results.py
tests\test_result_aggregation.py
```

Extract the archive into the repository root and allow only those two files to replace their existing counterparts. The archive has no other members. Its extracted files are byte-identical to the independently reviewed patch result and both compile successfully.

Archive SHA-256:

```text
aa583ff244f101c2f7ddea51da8c9636526bc63be2a461e1e36c8d8896f37c06
```

After replacement, the first non-mutating verification command is:

```bat
git diff --check -- analysis\plot_projection_results.py tests\test_result_aggregation.py
```

Expected output: none. Testing and result regeneration remain subsequent steps.

The local check completed with only these informational Windows line-ending notices:

```text
warning: in the working copy of 'analysis/plot_projection_results.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_result_aggregation.py', LF will be replaced by CRLF the next time Git touches it
```

This is accepted as a pass. Do not change `core.autocrlf`, renormalize the repository, or manually rewrite line endings.

## 27. Run the targeted plotting and aggregation tests

```bat
python -m pytest tests\test_result_aggregation.py -q
```

This is the next command. Stop and preserve the complete output if it fails; do not regenerate results until it passes.

Result:

```text
19 passed
```

Targeted testing passes.

## 28. Run the complete repository test suite

```bat
python -m pytest -q
```

This is the next command. Do not regenerate results unless the full suite passes. If it fails, preserve and report the complete output.

The first run aborted after 47 completed cases during Matplotlib PDF rendering in `test_result_figure_pdf_metadata_and_font_type`. The same module had already passed `19 passed` alone. Review identified a process-dependent Windows native-runtime interaction after PyTorch execution, not a plotting-code or evidence failure.

## 29. Install the isolated PDF smoke test

Replace only:

```text
tests\test_result_aggregation.py
```

with the revised complete file whose SHA-256 is:

```text
f6a2502d2c0ea352a2a74fdc47d303afe96f878daddb9d9194af595a1ef5d1cc
```

The production file `analysis\plot_projection_results.py` is unchanged. The revised test generates its tiny metadata/font PDF in a fresh subprocess and inspects it from the parent pytest process.

Then rerun the targeted module:

```bat
python -m pytest tests\test_result_aggregation.py -q
```

Result:

```text
19 passed
```

The isolated PDF smoke test passes. Do not set `KMP_DUPLICATE_LIB_OK` and do not regenerate results yet.

## 30. Rerun the complete repository test suite

```bat
python -m pytest -q
```

This is the next command. Stop and preserve the complete output if it fails. Do not regenerate results until the complete suite passes.

Result:

```text
65 passed
```

The complete suite passes. Source validation is complete and result regeneration may proceed.

## 31. Regenerate the primary fixed-training-geometry figures

```bat
python -m analysis.plot_projection_results ^
  --protocol experiments\fixed_training_geometry_analysis_protocol.json ^
  --tables-dir results\tables\fixed_training_geometry ^
  --evaluation-dir runs\evaluation\final\fixed_training_geometry ^
  --figures-dir results\figures\fixed_training_geometry ^
  --runs-dir runs
```

This rebuilds presentation artifacts only from the already frozen and audited evidence. Stop and preserve the complete output if it fails. Do not retrain or reevaluate.

Result: the command stopped before writing because the revised builder requires a new or empty figure directory:

```text
FileExistsError: Result figure directory already exists and is not empty: results\figures\fixed_training_geometry
```

This is an intentional clean-output safeguard.

## 32. Remove only the superseded primary figure directory

```bat
rmdir /s /q "results\figures\fixed_training_geometry"
```

This removes only generated primary figures. It does not touch the frozen evaluation evidence, checkpoints, source tables, calibration artifacts, or source code. The directory will be recreated by the figure builder. Do not remove `results\tables\fixed_training_geometry`.

Result: completed. The obsolete primary figure directory is removed.

## 33. Rerun the primary fixed-training-geometry figure build

```bat
python -m analysis.plot_projection_results ^
  --protocol experiments\fixed_training_geometry_analysis_protocol.json ^
  --tables-dir results\tables\fixed_training_geometry ^
  --evaluation-dir runs\evaluation\final\fixed_training_geometry ^
  --figures-dir results\figures\fixed_training_geometry ^
  --runs-dir runs
```

The builder will recreate the clean output directory. Stop and preserve the complete output if it fails.

Result: completed successfully. The build emitted 23 PDFs, five supporting CSVs, and `figure_build_audit.json`. The evaluation set includes the new timeout-rate figure.

## 34. Audit the regenerated primary figure set

```bat
python -c "from pathlib import Path; import json; r=Path(r'results\figures\fixed_training_geometry'); a=json.loads((r/'figure_build_audit.json').read_text(encoding='utf-8')); ps=sorted(r.glob('*.pdf')); bad=[p.name for p in ps if not (lambda b: b'/Author (Salvador Tenorio)' in b and b'/Title (' in b and b'/Subject (Predictive action projection with PPO)' in b and b'/Subtype /Type3' not in b and b'/Subtype /Type0' in b and b'/FontFile2' in b)(p.read_bytes())]; assert a['status']=='PASS'; assert a['artifact_scope']=='evaluation_and_training'; assert a['training_diagnostics_included'] is True; assert not a['skipped'], a['skipped']; assert len(a['generated'])==28, len(a['generated']); assert len(ps)==23, len(ps); assert (r/'evaluation_timeout_rate.pdf').is_file(); assert not bad, bad; print('PASS: primary audit, 23 PDFs, 28 generated artifacts, metadata/fonts valid, no skips')"
```

This is read-only. Stop and preserve the complete output if any assertion fails.

Result:

```text
PASS: primary audit, 23 PDFs, 28 generated artifacts, metadata/fonts valid, no skips
```

The regenerated primary set passes its structural post-build audit.

## 35. Remove only the superseded transfer figure directory

```bat
rmdir /s /q "results\figures\core_layout_transfer"
```

This removes only the old generated transfer figures. It does not touch transfer tables or frozen transfer evidence. The evaluation-only transfer builder will recreate this directory.

Result: completed. The superseded transfer figure directory is removed.

## 36. Remove duplicated training tables from the transfer result folder

```bat
del /q ^
  "results\tables\core_layout_transfer\training_scalar_events.csv" ^
  "results\tables\core_layout_transfer\training_episode_diagnostics.csv" ^
  "results\tables\core_layout_transfer\training_rollout_diagnostics.csv" ^
  "results\tables\core_layout_transfer\training_curve_points.csv"
```

These four files are generated duplicates of the shared training diagnostics retained under `results\tables\fixed_training_geometry`. Do not delete any other transfer table.

Result: completed. Only the four duplicated training-diagnostic CSVs were removed.

## 37. Build the evaluation-only transfer figures

```bat
python -m analysis.plot_projection_results ^
  --protocol experiments\projection_analysis_protocol.json ^
  --tables-dir results\tables\core_layout_transfer ^
  --evaluation-dir runs\evaluation\final\core_layout_transfer ^
  --figures-dir results\figures\core_layout_transfer ^
  --runs-dir runs ^
  --skip-training-diagnostics
```

This rebuilds transfer evaluation and trajectory figures only. Shared training diagnostics remain solely in the fixed-training-geometry result set. Stop and preserve the complete output if it fails.

Result: completed successfully. The build emitted 13 PDFs, the trajectory-selection CSV, and `figure_build_audit.json`, with no duplicated transfer training artifacts.

## 38. Audit the regenerated transfer figure set

```bat
python -c "from pathlib import Path; import json; f=Path(r'results\figures\core_layout_transfer'); t=Path(r'results\tables\core_layout_transfer'); a=json.loads((f/'figure_build_audit.json').read_text(encoding='utf-8')); ps=sorted(f.glob('*.pdf')); bad=[p.name for p in ps if not (lambda b: b'/Author (Salvador Tenorio)' in b and b'/Title (' in b and b'/Subject (Predictive action projection with PPO)' in b and b'/Subtype /Type3' not in b and b'/Subtype /Type0' in b and b'/FontFile2' in b)(p.read_bytes())]; stale=[p.name for p in t.glob('training_*.csv')]+[p.name for p in f.glob('training_*.pdf')]; assert a['status']=='PASS'; assert a['artifact_scope']=='evaluation_only'; assert a['training_diagnostics_included'] is False; assert a['intentional_omissions']==[{'category':'shared_training_diagnostics','reason':'disabled_by_command'}], a['intentional_omissions']; assert not a['skipped'], a['skipped']; assert len(a['generated'])==14, len(a['generated']); assert len(ps)==13, len(ps); assert (f/'evaluation_timeout_rate.pdf').is_file(); assert not stale, stale; assert not bad, bad; print('PASS: transfer audit, 13 PDFs, 14 generated artifacts, evaluation-only, metadata/fonts valid, no duplicates or skips')"
```

This is read-only. Stop and preserve the complete output if any assertion fails.

Result:

```text
PASS: transfer audit, 13 PDFs, 14 generated artifacts, evaluation-only, metadata/fonts valid, no duplicates or skips
```

Both regenerated suites now pass structural post-build validation.

## 39. Package regenerated tables and figures for visual QA

```bat
powershell -NoProfile -Command "$out='runs\final_result_visual_review.zip'; if (Test-Path -LiteralPath $out) { throw 'Review archive already exists; do not overwrite it.' }; Compress-Archive -Path 'results\figures','results\tables' -DestinationPath $out -CompressionLevel Optimal; Write-Host ('Created: '+$out)"
```

Upload `runs\final_result_visual_review.zip` for direct inspection. This archive is a review copy outside the tracked `results` tree; it does not modify the generated artifacts.

Result: the archive was uploaded and all 36 PDFs were rendered and checked against their source tables. Numerical and trajectory integrity passed. Visual QA found 13 presentation-only failures: ten clipped rightmost labels in projection-only evaluation plots, two bounded-rate training bands extending outside `[0,1]`, and a missing cross-method reward-scale warning in `training_return.pdf`. No local command can repair these safely until the plotting source and tests are updated.

## 40. Prepare the narrow visual-QA remediation

No local command yet. Install the reviewed full replacement files for:

```text
analysis\plot_projection_results.py
tests\test_result_aggregation.py
```

The revision must only add right-edge accommodation for projection-only method labels, clip displayed rate uncertainty bands to `[0,1]`, add the training-return reward-scale caveat, and test those behaviors. It must not change frozen evidence, numerical aggregation, trajectory selections, protocols, or evaluation.

Result: completed. The reviewed archive contains exactly the two required repository-relative files. Compilation, real-data smoke rendering, all-ten PDF text-bound checks, all-four bounded-rate checks, metadata-derived reward-warning checks, archive integrity, and independent source review pass. Archive SHA-256:

```text
047a481248b46c721a2afb2ada082765316efb9429a528243fd5026957b9d522
```

## 41. Install the narrow visual-QA remediation

After downloading the archive to the normal Downloads folder, run from the repository root:

```bat
powershell -NoProfile -Command "$z=Join-Path $env:USERPROFILE 'Downloads\Predictive_Action_Projection_Visual_QA_Remediation_Full_Files.zip'; $h=(Get-FileHash -LiteralPath $z -Algorithm SHA256).Hash.ToLowerInvariant(); if($h -ne '047a481248b46c721a2afb2ada082765316efb9429a528243fd5026957b9d522'){throw 'Archive SHA-256 mismatch'}; Expand-Archive -LiteralPath $z -DestinationPath '.' -Force; $a=(Get-FileHash -LiteralPath 'analysis\plot_projection_results.py' -Algorithm SHA256).Hash.ToLowerInvariant(); $t=(Get-FileHash -LiteralPath 'tests\test_result_aggregation.py' -Algorithm SHA256).Hash.ToLowerInvariant(); if($a -ne '5d0abf9e3207d4f01dbc7947094f65ef49cc109d7461491d814b6c492550f8bc' -or $t -ne '194c3a553ec614379168989a66d57be8ae7e001d049fff29898fff28ed41cb2e'){throw 'Installed file hash mismatch'}; Write-Host 'PASS: visual-QA remediation installed and verified'"
```

This overwrites only the plotting source and its test module. Do not remove or regenerate result directories until targeted and full-suite tests pass.

Result: the two files were installed. The subsequent complete-suite run reached the new label-layout test but aborted the Python process inside its in-process `figure.canvas.draw()` call. This is a test-isolation defect, not a plotting or result failure.

## 42. Full-suite attempt exposing the label-test isolation defect

```bat
python -m pytest -q
```

Result: fatal native abort in `test_projection_only_figure_reserves_right_label_margin` at the direct Matplotlib canvas draw. Do not regenerate figures.

## 43. Install the isolated label-layout test

Download `Predictive_Action_Projection_Label_Test_Subprocess_Fix.zip` to the normal Downloads folder, then run from the repository root:

```bat
powershell -NoProfile -Command "$z=Join-Path $env:USERPROFILE 'Downloads\Predictive_Action_Projection_Label_Test_Subprocess_Fix.zip'; $h=(Get-FileHash -LiteralPath $z -Algorithm SHA256).Hash.ToLowerInvariant(); if($h -ne '1bb1d5fe9c7b80cce0e9323dadb67f121a1de0ef20771eb9532193cf5530f2c4'){throw 'Archive SHA-256 mismatch'}; Expand-Archive -LiteralPath $z -DestinationPath '.' -Force; $t=(Get-FileHash -LiteralPath 'tests\test_result_aggregation.py' -Algorithm SHA256).Hash.ToLowerInvariant(); if($t -ne '13202dac910c7e797b89fcef30a86db4e6a8b27584e569bfe4f281cc5b92e88d'){throw 'Installed test-file hash mismatch'}; Write-Host 'PASS: isolated label-layout test installed and verified'"
```

This overwrites only `tests\test_result_aggregation.py`. The plotting source remains unchanged.

Result: installed and hash-verified successfully.

## 44. Rerun the targeted plotting and aggregation tests

```bat
python -m pytest tests\test_result_aggregation.py -q
```

Result: `25 passed`.

## 45. Rerun the complete repository suite

```bat
python -m pytest -q
```

Result: `71 passed`. Final figure regeneration is cleared to proceed.

## 46. Remove only the superseded primary figure directory

```bat
rmdir /s /q "results\figures\fixed_training_geometry"
```

Result: completed. Only the generated primary figure directory was removed. Primary tables, frozen evaluation evidence, checkpoints, calibration artifacts, and source files were preserved.

## 47. Regenerate the final primary figures

```bat
python -m analysis.plot_projection_results ^
  --protocol experiments\fixed_training_geometry_analysis_protocol.json ^
  --tables-dir results\tables\fixed_training_geometry ^
  --evaluation-dir runs\evaluation\final\fixed_training_geometry ^
  --figures-dir results\figures\fixed_training_geometry ^
  --runs-dir runs
```

Result: completed successfully. The builder produced 23 PDFs, five supporting CSV files, and `figure_build_audit.json`, including the repaired timeout, label-margin, bounded-rate-band, and training-return presentation outputs. The compact structural audit remains the next gate.

## 48. Audit the final primary figure set

```bat
python -c "from pathlib import Path; import json; r=Path(r'results\figures\fixed_training_geometry'); a=json.loads((r/'figure_build_audit.json').read_text(encoding='utf-8')); ps=sorted(r.glob('*.pdf')); bad=[p.name for p in ps if not (lambda b: b'/Author (Salvador Tenorio)' in b and b'/Title (' in b and b'/Subject (Predictive action projection with PPO)' in b and b'/Subtype /Type3' not in b and b'/Subtype /Type0' in b and b'/FontFile2' in b)(p.read_bytes())]; assert a['status']=='PASS'; assert a['artifact_scope']=='evaluation_and_training'; assert a['training_diagnostics_included'] is True; assert not a['skipped'], a['skipped']; assert len(a['generated'])==28, len(a['generated']); assert len(ps)==23, len(ps); assert (r/'evaluation_timeout_rate.pdf').is_file(); assert not bad, bad; print('PASS: primary audit, 23 PDFs, 28 generated artifacts, metadata/fonts valid, no skips')"
```

Result:

```text
PASS: primary audit, 23 PDFs, 28 generated artifacts, metadata/fonts valid, no skips
```

The final primary output is structurally validated.

## 49. Remove only the superseded transfer figure directory

```bat
rmdir /s /q "results\figures\core_layout_transfer"
```

This is the next cleanup command. It removes only generated transfer figures. It must not touch `results\tables\core_layout_transfer` or `runs\evaluation\final\core_layout_transfer`.

Result: completed. Only the superseded transfer figure directory was removed.

## 50. Regenerate the final evaluation-only transfer figures

```bat
python -m analysis.plot_projection_results ^
  --protocol experiments\projection_analysis_protocol.json ^
  --tables-dir results\tables\core_layout_transfer ^
  --evaluation-dir runs\evaluation\final\core_layout_transfer ^
  --figures-dir results\figures\core_layout_transfer ^
  --runs-dir runs ^
  --skip-training-diagnostics
```

Run this immediately after command 49. Stop and preserve the complete output if it fails. The expected build contains 13 evaluation/trajectory PDFs, `representative_trajectory_selection.csv`, and `figure_build_audit.json`, with no duplicated training outputs.

Result: completed successfully. The final transfer build produced the expected 13 PDFs, `representative_trajectory_selection.csv`, and `figure_build_audit.json`; no training-diagnostic outputs were generated. The compact transfer audit remains pending.

## 51. Audit the final transfer figure set

```bat
python -c "from pathlib import Path; import json; f=Path(r'results\figures\core_layout_transfer'); t=Path(r'results\tables\core_layout_transfer'); a=json.loads((f/'figure_build_audit.json').read_text(encoding='utf-8')); ps=sorted(f.glob('*.pdf')); bad=[p.name for p in ps if not (lambda b: b'/Author (Salvador Tenorio)' in b and b'/Title (' in b and b'/Subject (Predictive action projection with PPO)' in b and b'/Subtype /Type3' not in b and b'/Subtype /Type0' in b and b'/FontFile2' in b)(p.read_bytes())]; stale=[p.name for p in t.glob('training_*.csv')]+[p.name for p in f.glob('training_*.pdf')]; assert a['status']=='PASS'; assert a['artifact_scope']=='evaluation_only'; assert a['training_diagnostics_included'] is False; assert a['intentional_omissions']==[{'category':'shared_training_diagnostics','reason':'disabled_by_command'}], a['intentional_omissions']; assert not a['skipped'], a['skipped']; assert len(a['generated'])==14, len(a['generated']); assert len(ps)==13, len(ps); assert (f/'evaluation_timeout_rate.pdf').is_file(); assert not stale, stale; assert not bad, bad; print('PASS: transfer audit, 13 PDFs, 14 generated artifacts, evaluation-only, metadata/fonts valid, no duplicates or skips')"
```

Result:

```text
PASS: transfer audit, 13 PDFs, 14 generated artifacts, evaluation-only, metadata/fonts valid, no duplicates or skips
```

The final evaluation-only transfer output is structurally validated. Both result suites are now ready for combined final visual review.

## 52. Package the final regenerated results for visual QA

```bat
powershell -NoProfile -Command "$out='runs\final_result_visual_review_v2.zip'; if (Test-Path -LiteralPath $out) { throw 'Review archive already exists; do not overwrite it.' }; Compress-Archive -Path 'results\figures','results\tables' -DestinationPath $out -CompressionLevel Optimal; Write-Host ('Created: '+$out)"
```

Upload `runs\final_result_visual_review_v2.zip` for direct inspection. Do not commit before the final visual and numerical review passes.

Result: completed. The archive passed compressed-data integrity testing and contained the exact expected final figures, tables, and audits.

## 53. Final direct visual and numerical review

No repository command was required. The uploaded archive was rendered and independently reconciled against its CSV and JSON sources.

Result: PASS.

- All 36 PDFs are visually clean and publication-ready.
- All ten repaired projection-only method labels are complete, with 14.76 points of right-page clearance.
- Both bounded rolling-rate figures remain within `[0,1]`.
- The training-return reward-scale caveat matches collision-penalty metadata.
- All 3,720 evaluation episodes reconcile exactly to checkpoint, method, paired-difference, and trajectory-selection tables.
- All training-curve source counts and plotted aggregates reconcile; no solver failure is present.
- Metadata, fonts, extractable text, artifact scopes, and duplicate-training checks pass.

The final presentation set is cleared for Git review. Do not stage or commit until command 54 is reviewed.

## 54. Capture the complete pre-commit Git inventory

Run from the repository root:

```bat
git status --short --untracked-files=all
git diff --check
git diff --stat
```

Preserve and return all three outputs. Do not run `git add`, `git commit`, or `git push` yet; the exact staging scope must be derived from the actual inventory.

Result: PASS with expected line-ending warnings only.

The inventory contains two modified reviewed Python files, two new living repository records, 38 generated figure artifacts, and 22 generated table artifacts. No unrelated file, review ZIP, checkpoint, raw evaluation archive, or duplicated transfer training diagnostic appears. `git diff --check` reports no whitespace error; the two LF-to-CRLF messages are expected `core.autocrlf` warnings. The tracked diff is 904 insertions and 53 deletions across the plotting and test modules; untracked generated artifacts are intentionally absent from that statistic until staged.

Before staging, replace the two local `docs\records` files with the newest synchronized versions containing this inventory result. Then stage only the exact paths that will be supplied in the next command.

## 55. Stage the approved result scope

```bat
git add -- ^
  analysis\plot_projection_results.py ^
  tests\test_result_aggregation.py ^
  docs\records\Predictive_Action_Projection_Analysis_Record.md ^
  docs\records\Predictive_Action_Projection_Analysis_Command_Record.md ^
  results\figures\core_layout_transfer ^
  results\figures\fixed_training_geometry ^
  results\tables\core_layout_transfer ^
  results\tables\fixed_training_geometry
```

Result: the exact 64-file scope was staged successfully. The staged statistic is 64,094 insertions and 53 deletions.

## 56. Check the staged diff for whitespace errors

```bat
git diff --cached --check
```

Result: nine trailing-whitespace findings were reported, all in lines 3--7 of the two Markdown record headers. The spaces were Markdown hard-line breaks; no source or generated-result artifact failed. The correction replaces them with explicit backslash line breaks and also makes the documented repository-path capitalization match the actual filenames.

Install the corrected two-record full-file handoff, restage those two files, and rerun this command. Do not commit until it produces no output.

Result: PASS after the corrected records were installed and restaged. The command produced no output. The final staged inventory contains exactly the approved 64 files, with 64,128 insertions and 53 deletions.

## 57. Run the public-release privacy and scope gate

The staged records were reviewed for machine-specific paths and sensitive strings. No credential, token, private key, password, or email address was found. Two historical commands contained a literal local Windows user-profile path; both now resolve the Downloads directory through `$env:USERPROFILE`.

After installing and restaging the privacy-clean records, run:

```bat
git diff --cached --check
powershell -NoProfile -Command "$needle='C:' + [char]92 + 'Users'; $hits=& git grep --cached -n -I -F $needle -- docs/records; if($LASTEXITCODE -eq 0){$hits; throw 'Machine-specific Windows user path found'}; if($LASTEXITCODE -ne 1){throw 'git grep privacy scan failed'}; Write-Host 'PASS: no machine-specific Windows user path in staged records'"
git status --short
git diff --cached --stat
```

The whitespace command must produce no output, and the privacy command must print its `PASS` message. The status must contain only the approved staged files, with no unstaged or untracked entry. The staged statistic must report 64 files. Do not commit until all four conditions pass.

## 58. Commit the audited result set

```bat
git commit -m "Add audited predictive action projection results"
```

Result: PASS. Commit `b005123cb2c6c754a991d1e7fdc709437b90e915` was created with exactly 64 files.

## 59. Verify the local commit

```bat
git status --short
git show --no-patch --format="%H%n%an <%ae>%n%ad%n%s" --date=iso-strict HEAD
git diff-tree --no-commit-id --name-only -r HEAD | find /c /v ""
```

Result: PASS. The working tree was clean, the commit identity and message were correct, and the file count was `64`.

## 60. Push the final result branch privately

```bat
git push -u origin final_evaluation_runs
```

Result: PASS. The new remote branch was created and configured as the upstream tracking branch. The repository remained private.

Direct remote inspection then confirmed that `final_evaluation_runs` is one commit ahead of and zero commits behind `main`, with `b005123c` at its tip. No CI workflow or commit status is configured.

## 61. Run the full-history public-release audit

The audit traversed all reachable textual patch history and Git objects for high-confidence credential signatures, literal Windows user-profile paths, suspicious credential-related filenames, and blobs larger than 20 MB.

The exact detector expressions are intentionally not copied into this versioned record because embedding credential signatures in the repository would cause future scans to match the scanner's own rule definitions.

Result:

```text
PASS: full-history public-release audit
```

## 62. Install and validate the documentation-only release patch

The reviewed patch contains exactly these repository-root files:

```text
README.md
LICENSE
THIRD_PARTY_NOTICES.md
docs/records/Predictive_Action_Projection_Analysis_Record.md
docs/records/Predictive_Action_Projection_Analysis_Command_Record.md
```

After installing the full files, verify the documentation-only scope:

```bat
git status --short
git diff --check
git diff --stat
```

Expected scope: three new or modified root documentation files and the two updated living repository records. No source, test, result, protocol, or evidence file may change. Do not commit until this exact scope and a silent whitespace check are confirmed.

## 63. Freeze the scientific-interpretation workspace

**Completed:** 2026-08-27\
**Shell for commands below:** Bash\
**Repository effect:** none; all operations were read-only or copied committed
bytes into a temporary external workspace.

### 63.1 Inspect the handoff package and printable guide

```bash
HANDOFF_ZIP='Predictive_Action_Projection_Interpretation_Handoff_Package(1).zip'
HANDOFF_DIR='handoff'

zipinfo -1 "$HANDOFF_ZIP"
sha256sum "$HANDOFF_ZIP"
mkdir -p "$HANDOFF_DIR"
unzip -q -o "$HANDOFF_ZIP" -d "$HANDOFF_DIR"

pdfinfo "$HANDOFF_DIR/reference/Predictive_Action_Projection_Scientific_Interpretation_Guide.pdf"
pdftotext -layout \
  "$HANDOFF_DIR/reference/Predictive_Action_Projection_Scientific_Interpretation_Guide.pdf" \
  interpretation_guide.txt
pdftoppm -f 1 -l 1 -r 150 -png -singlefile \
  "$HANDOFF_DIR/reference/Predictive_Action_Projection_Scientific_Interpretation_Guide.pdf" \
  guide_title
pdftoppm -f 2 -l 2 -r 150 -png -singlefile \
  "$HANDOFF_DIR/reference/Predictive_Action_Projection_Scientific_Interpretation_Guide.pdf" \
  guide_status
pdftoppm -f 34 -l 34 -r 150 -png -singlefile \
  "$HANDOFF_DIR/reference/Predictive_Action_Projection_Scientific_Interpretation_Guide.pdf" \
  guide_step1
```

Result: the ZIP member list matches the manifest and its SHA-256 is
`74450ef553ea74438536518f45ac774bd8dc7791d19bb4bc019cd1800c8f3279`.
The PDF has 43 US-Letter pages, author `Salvador Tenorio`, the expected title
and subject, no encryption, and readable rendered title, status, and Step 1
pages. `README_FIRST.md` was read before the continuation prompt, blueprint,
and guide.

### 63.2 Resolve public Git identities without checking out or changing a branch

```bash
PUBLIC_REPO_URL='https://github.com/Sateno/rl-constrained-decision-making.git'
PUBLIC_REPO='public_repo'
MAIN_TIP='5a5cc2041ad5b6194a86aff9e61460873ce185c9'
EVIDENCE_COMMIT='b005123cb2c6c754a991d1e7fdc709437b90e915'
RELEASE_DOC_COMMIT='97f375a5ecc22db16d1cfc641dbff0e7f7ac9eed'
PROTOCOL_COMMIT='ba64926aed98b08b7b285266cf85989d466f9f1c'

git ls-remote "$PUBLIC_REPO_URL" refs/heads/main
git clone --filter=blob:none --no-checkout "$PUBLIC_REPO_URL" "$PUBLIC_REPO"
cd "$PUBLIC_REPO"

git rev-parse refs/remotes/origin/main
git show -s --format='%H%n%P%n%ad%n%s' --date=iso-strict refs/remotes/origin/main
git log --graph --decorate --oneline --all --max-count=30
git merge-base --is-ancestor "$EVIDENCE_COMMIT" "$MAIN_TIP"
git merge-base --is-ancestor "$RELEASE_DOC_COMMIT" "$MAIN_TIP"
git merge-base --is-ancestor "$PROTOCOL_COMMIT" "$EVIDENCE_COMMIT"
```

Result: `refs/heads/main` resolved to `$MAIN_TIP`; all three ancestry checks
returned exit status zero. The evidence commit is the audited-result commit,
the later release commit changes documentation, and the current tip adds the
README guide-link correction.

### 63.3 Prove that later public commits do not redefine the evidence tree

```bash
git diff --name-status "$EVIDENCE_COMMIT".."$MAIN_TIP"
git diff --quiet "$EVIDENCE_COMMIT".."$MAIN_TIP" -- \
  experiments/fixed_training_geometry_analysis_protocol.json \
  experiments/projection_analysis_protocol.json \
  analysis/aggregate_projection_results.py \
  analysis/plot_projection_results.py \
  results/tables/fixed_training_geometry \
  results/tables/core_layout_transfer \
  results/figures/fixed_training_geometry \
  results/figures/core_layout_transfer

for commit in "$EVIDENCE_COMMIT" "$RELEASE_DOC_COMMIT" "$MAIN_TIP"; do
  git rev-parse "$commit:results/tables/fixed_training_geometry"
  git rev-parse "$commit:results/tables/core_layout_transfer"
  git rev-parse "$commit:results/figures/fixed_training_geometry/figure_build_audit.json"
  git rev-parse "$commit:results/figures/core_layout_transfer/figure_build_audit.json"
  git rev-parse "$commit:analysis/plot_projection_results.py"
  git rev-parse "$commit:experiments/fixed_training_geometry_analysis_protocol.json"
  git rev-parse "$commit:experiments/projection_analysis_protocol.json"
done

git diff-tree --no-commit-id --name-only -r "$EVIDENCE_COMMIT" | wc -l
```

Result: the path-restricted diff returned exit status zero. All listed evidence
objects have identical Git object IDs at the three commits. The full diff after
the evidence commit contains only release documentation, licensing,
attribution, and the two record files. The audited-result commit contains 64
changed files.

### 63.4 Export committed evidence bytes outside the repository

```bash
EVIDENCE_DIR='../evidence_snapshot'
mkdir -p "$EVIDENCE_DIR"
git archive "$EVIDENCE_COMMIT" | tar -x -C "$EVIDENCE_DIR"
cd "$EVIDENCE_DIR"
```

Result: the audited evidence set was inspected outside the Git working tree. No
checkout, branch, index, tracked file, Git setting, or remote reference was
changed.

### 63.5 Reconstruct the evidence counts and audit inventory

```bash
jq '{method_count:(.methods|length), seed_count:(.expected_train_seeds|length), repeats:.expected_repeats_per_layout, modes:[.methods[].required_projection_modes|length]}' \
  experiments/fixed_training_geometry_analysis_protocol.json
jq '{method_count:(.methods|length), seed_count:(.expected_train_seeds|length), repeats:.expected_repeats_per_layout, modes:[.methods[].required_projection_modes|length]}' \
  experiments/projection_analysis_protocol.json

jq '{status, layout_count, episode_row_count, checkpoint_row_count, method_row_count, paired_checkpoint_row_count, paired_method_row_count, projection_solver_failure_count}' \
  results/tables/fixed_training_geometry/result_build_audit.json
jq '{status, layout_count, episode_row_count, checkpoint_row_count, method_row_count, paired_checkpoint_row_count, paired_method_row_count, projection_solver_failure_count}' \
  results/tables/core_layout_transfer/result_build_audit.json

jq '{status, artifact_scope, pdf_count:([.generated|keys[]|select(endswith(".pdf"))]|length), generated_pre_audit:(.generated|length), skipped_count:(.skipped|length)}' \
  results/figures/fixed_training_geometry/figure_build_audit.json
jq '{status, artifact_scope, pdf_count:([.generated|keys[]|select(endswith(".pdf"))]|length), generated_pre_audit:(.generated|length), skipped_count:(.skipped|length), intentional_omissions}' \
  results/figures/core_layout_transfer/figure_build_audit.json

wc -l \
  results/tables/fixed_training_geometry/evaluation_episode_results.csv \
  results/tables/core_layout_transfer/evaluation_episode_results.csv
```

Calculation:

```text
fixed   = 3 x 5 x 2 x 1 x 100 = 3,000
transfer = 3 x 5 x 2 x 24 x 1 =   720
total   = 3,000 + 720          = 3,720 episodes
```

Result: both result-table and figure-build audits report `PASS`. The committed
CSV files contain 3,001 and 721 physical lines; after one header line per file,
these reconcile to 3,000 and 720 episode rows. The fixed figure audit records
23 PDFs and 28 pre-audit generated artifacts. The transfer audit records 13
PDFs and 14 pre-audit generated artifacts, with shared training diagnostics
intentionally omitted and no skipped artifact.

### 63.6 Recover the evaluation exposure for recorded projection failures

```bash
awk -F, '
  NR==1 {for (i=1;i<=NF;i++) idx[$i]=i; next}
  $idx["projection_mode"]=="enabled" {
    episodes++
    steps+=$idx["episode_length"]
    failures+=$idx["projection_solver_failure_count"]
  }
  END {
    printf "enabled_episodes=%d enabled_steps=%d solver_failures=%d\n", episodes, steps, failures
  }
' results/tables/fixed_training_geometry/evaluation_episode_results.csv

awk -F, '
  NR==1 {for (i=1;i<=NF;i++) idx[$i]=i; next}
  $idx["projection_mode"]=="enabled" {
    episodes++
    steps+=$idx["episode_length"]
    failures+=$idx["projection_solver_failure_count"]
  }
  END {
    printf "enabled_episodes=%d enabled_steps=%d solver_failures=%d\n", episodes, steps, failures
  }
' results/tables/core_layout_transfer/evaluation_episode_results.csv
```

Result:

```text
fixed:    1,500 enabled episodes; 236,084 enabled steps; 0 failures
transfer:   360 enabled episodes;  62,900 enabled steps; 0 failures
combined: 1,860 enabled episodes; 298,984 enabled steps; 0 failures
```

The denominator is recorded as enabled steps, not solver calls, until the
implementation separately verifies exactly one solve per enabled step.

### 63.7 Preserve living-record identity

The existing Library identities were resolved before editing: Analysis Record
version 50, Analysis Command Record version 44, and continuation prompt version
2. Their current bytes were materialized, edited locally, validated as text,
and written back with optimistic version guards. The repository copies were not
edited. Three new, uniquely named working sidecars were created only after an
exact-title and simplified-content search found no existing sidecar identity.

## 64. Phase I, Step 2 — reconstruct the frozen experimental design

All commands in this section were read-only with respect to the public
repository. The exact relational audit is preserved as
`Predictive_Action_Projection_Step2_Design_Audit.py`, SHA-256
`bda34a23e6958421f3484736284b935c6b660457694e18bc37346f404810122e`.

### 64.1 Recover the Step 2 requirements from the handoff guide

```bash
rg -n -i 'Step 2|reconstruct methods|methods, checkpoints|teach-back' \
  tmp/pdfs/interpretation_guide.txt \
  handoff/reference/Predictive_Action_Projection_Guide_Regeneration_Blueprint.md

sed -n '1485,1535p' tmp/pdfs/interpretation_guide.txt
sed -n '1715,1770p' tmp/pdfs/interpretation_guide.txt
```

Result: Step 2 requires mapping methods, checkpoints, projection modes,
episodes, layouts, seeds, and totals. The applicable teach-back distinguishes
evaluation coverage from independent training replication.

### 64.2 Inspect the authoritative design fields

```bash
jq '.' experiments/fixed_training_geometry_analysis_protocol.json
jq '.' experiments/projection_analysis_protocol.json
jq '.' evaluation/layouts/fixed_training_geometry.json
jq '{schema_version, suite_id, layout_count:(.layouts|length), layouts}' \
  evaluation/layouts/core_navigation_layouts.json

head -n 1 results/tables/fixed_training_geometry/evaluation_episode_results.csv
head -n 1 results/tables/core_layout_transfer/evaluation_episode_results.csv
head -n 4 results/tables/fixed_training_geometry/checkpoint_summary.csv
head -n 4 results/tables/core_layout_transfer/checkpoint_summary.csv
```

Fields used: protocol `methods`, `expected_train_seeds`,
`expected_training_timesteps`, `expected_repeats_per_layout`,
`evaluation_policy_mode`, `evaluation_device`,
`evaluation_collision_penalty`, `evaluation_base_seed`,
`evaluation_last_seed`, `max_episode_steps`, `layout_suite`, and each method's
`required_projection_modes`; layout-suite `suite_id` and `layouts[].layout_id`;
episode-table `method`, `train_seed`, training fields,
`checkpoint_sha256`, layout identity fields, `layout_id`, `layout_repeat`,
`evaluation_seed`, evaluation fields, `projection_mode`, `checkpoint`,
`episode`, `seed`, and `projection_enabled`.

### 64.3 Detect and resolve the transfer-protocol canonical-hash discrepancy

The human-readable frozen protocol record declares `dfc0e1...`, whereas the
transfer result audit declares `f0f853...`. Canonical hashes were recalculated
with the frozen implementation:

```bash
python -c "import json; from pathlib import Path; from evaluation.layout_suite import canonical_json_sha256; [print(p,canonical_json_sha256(json.loads(Path(p).read_text()))) for p in ['experiments/fixed_training_geometry_analysis_protocol.json','experiments/projection_analysis_protocol.json']]"
```

Result:

```text
experiments/fixed_training_geometry_analysis_protocol.json
89ea5ec0d2329bfe99ea3a1cb7a5c1339478b0ce2d8304f5b1b810f51f1f0d14

experiments/projection_analysis_protocol.json
f0f853fb53b910cdd9227e1562fc227201b08a6926d8895d430e507944b659d1
```

The protocol files were compared from the protocol source through the evidence
commit:

```bash
git diff --quiet \
  ba64926aed98b08b7b285266cf85989d466f9f1c \
  b005123cb2c6c754a991d1e7fdc709437b90e915 -- \
  experiments/fixed_training_geometry_analysis_protocol.json \
  experiments/projection_analysis_protocol.json
echo "protocol_path_diff_exit=$?"

git log --format='%H %s' --all -- \
  experiments/projection_analysis_protocol.json
```

Result: the diff exit status was zero. The transfer JSON is unchanged from the
protocol commit through the evidence commit and current public tip. No reachable
historical version of the transfer JSON has canonical hash `dfc0e1...`.

The declared hash was then reproduced:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

path = Path('experiments/projection_analysis_protocol.json')
actual = json.loads(path.read_text(encoding='utf-8'))

def canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

print('actual', canonical_hash(actual))
with_provenance = dict(actual)
with_provenance['validated_implementation_base_commit'] = (
    'd0548b3e6729571113675b6f4ccad6401f27f167'
)
print('with_provenance', canonical_hash(with_provenance))
PY
```

Result:

```text
actual          f0f853fb53b910cdd9227e1562fc227201b08a6926d8895d430e507944b659d1
with_provenance dfc0e1e3de29c0f63eb6152a3f063ad1d17c4461430855f33222f93e827c6e90
```

Repository-wide search found
`validated_implementation_base_commit` only in the fixed protocol and the
human-readable freeze record, not in runtime or aggregation logic. Adding that
operationally unused field explains the two hash values exactly, but the
calculation does not prove the historical cause of the discrepancy. The
executed identity remains the actual committed and audited `f0f853...`; the
repository was not changed.

An initial raw-file SHA-256 diagnostic was not used because the audits record
the platform-independent canonical JSON digest, not the raw-byte digest.

### 64.4 Execute the complete relational design audit

```bash
python sidecars/Predictive_Action_Projection_Step2_Design_Audit.py
```

The script uses only Python's standard library. It performs these exact checks:

1. recompute both protocol and layout-suite canonical hashes and require
   equality with each result audit;
2. require audit `status = PASS`, audited layout counts, and audited episode
   row counts;
3. construct the method map and require every row's training penalty,
   training-projection flag, evaluation mode, evaluation penalty, policy mode,
   maximum episode steps, suite identity, and projection-enabled flag to match
   its frozen declaration;
4. require one invariant `(checkpoint, checkpoint_sha256)` for every
   `(method, train_seed)` key;
5. require 30 complete method-seed-mode cells per suite;
6. require 100 rows per fixed cell and the exact mapping
   `(layout_repeat, evaluation_seed, episode) = (i, 10000+i, i)` for
   `i = 0,...,99`;
7. require 24 rows per transfer cell and the exact frozen layout-array mapping
   `(layout_id[j], evaluation_seed, episode, layout_repeat) =
   (layout_id[j], 1000+j, j, 0)` for `j = 0,...,23`;
8. require the episode-table `seed` field to equal `evaluation_seed` in every
   row;
9. require the same 15 checkpoint identities, method declarations, training
   seeds, and training budget in both suites; and
10. derive and print training totals, suite totals, per-method totals,
    per-mode totals, the complete checkpoint map, and the transfer layout-seed
    map.

Result:

```text
STATUS=PASS
METHODS=3
TRAIN_SEEDS=1,2,3,4,5
CHECKPOINTS=15
TRANSITIONS_PER_RUN=51200
TRANSITIONS_PER_METHOD=256000
TOTAL_TRAINING_TRANSITIONS=768000

fixed_training_geometry:
  30 method-seed-mode cells
  100 rows/cell
  3,000 expected rows
  3,000 observed rows
  1,000 rows/method
  1,500 rows/projection mode

core_layout_transfer:
  30 method-seed-mode cells
  24 rows/cell
  720 expected rows
  720 observed rows
  240 rows/method
  360 rows/projection mode

TOTAL_EVALUATION_ROWS=3720
```

The script printed the full 15-checkpoint SHA-256 map and 24-layout seed map;
both are preserved in the Analysis Record.

### 64.5 Derive rollout and optimizer-update totals

```bash
num_envs=4
num_steps=256
transitions_per_run=51200
num_minibatches=8
update_epochs=4
batch_size=$((num_envs*num_steps))
rollouts=$((transitions_per_run/batch_size))
updates_per_run=$((rollouts*num_minibatches*update_epochs))
runs=$((3*5))
printf 'batch_size=%d\nrollouts_per_run=%d\ntotal_rollout_iterations=%d\nminibatch_optimizer_steps_per_run=%d\ntotal_minibatch_optimizer_steps=%d\n' \
  "$batch_size" "$rollouts" "$((runs*rollouts))" "$updates_per_run" \
  "$((runs*updates_per_run))"
```

Result:

```text
batch_size=1024
rollouts_per_run=50
total_rollout_iterations=750
minibatch_optimizer_steps_per_run=1600
total_minibatch_optimizer_steps=24000
```

### 64.6 Step 2 completion result

The prescribed products, observed rows, exact cell coverage, checkpoint
summaries, and result audits all reconcile. The transfer-protocol hash mismatch
is bounded to one provenance-only field and does not change any executed design
factor. Step 2 therefore passes without reopening or modifying the experiment.
The continuation state advances to Phase I, Step 3.

## 65. Independent continuity audit and terminology correction

This follow-up audit was read-only with respect to the public repository. It
checked the Git topology, evidence-path identity, CSV design grid, training-loop
terminology, and transfer-protocol hash reconstruction before this chat became
the sole authoritative interpretation thread.

### 65.1 Confirm one current remote branch and the audit-anchor relationship

```bash
git fetch --unshallow origin
git merge-base --is-ancestor \
  b005123cb2c6c754a991d1e7fdc709437b90e915 \
  5a5cc2041ad5b6194a86aff9e61460873ce185c9
git branch -a
git ls-remote --heads origin
git diff --name-status \
  b005123cb2c6c754a991d1e7fdc709437b90e915..\
  5a5cc2041ad5b6194a86aff9e61460873ce185c9
```

Result: `b005123...` is a reachable ancestor of the recorded current `main`
revision. The remote exposes only `refs/heads/main`. Later changes are limited
to `README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and the two narrative
repository records; the audited evidence paths are unchanged.

### 65.2 Verify the optimizer-step terminology from executable code

```bash
rg -n 'batch_size|num_iterations|update_epochs|num_minibatches|optimizer.step|target_kl' \
  algorithms/ppo/ppo_continuous_action.py \
  experiments/train_ppo_variant.py \
  docs/records/predictive_action_projection_experimental_protocol.md
```

Result: the training command fixes four environments, 256 steps, eight
minibatches, and four epochs; `target_kl=None`; and the PPO loop calls
`optimizer.step()` once per minibatch. Therefore the exact totals are 750 PPO
rollout/update iterations and 24,000 minibatch optimizer steps across 15 runs.

### 65.3 Continuity-audit judgment

The Step 2 design reconstruction remains PASS. Protocol-identity consistency
is qualified by the one bounded transfer-protocol hash discrepancy. The living
records now describe `b005123...` as the audit anchor for one unchanged audited
evidence set rather than implying a second branch or dataset. No outcome value
was inspected.

## 66. Phase I, Step 3 — verify and lock the research questions

All repository operations in this section were read-only. No result CSV,
summary table, or figure was used for this step. The verification used only
protocol ancestry, design declarations, predeclared comparisons, and the
corrected interpretation specification.

### 66.1 Establish that the comparison record predates the results

```bash
git rev-parse HEAD
git merge-base --is-ancestor \
  ba64926aed98b08b7b285266cf85989d466f9f1c \
  b005123cb2c6c754a991d1e7fdc709437b90e915
echo "protocol_before_results_exit=$?"

git diff --quiet \
  ba64926aed98b08b7b285266cf85989d466f9f1c..\
  b005123cb2c6c754a991d1e7fdc709437b90e915 -- \
  docs/records/predictive_action_projection_experimental_protocol.md
echo "protocol_record_changed_before_results_exit=$?"

git diff --quiet \
  b005123cb2c6c754a991d1e7fdc709437b90e915..HEAD -- \
  docs/records/predictive_action_projection_experimental_protocol.md \
  experiments/fixed_training_geometry_analysis_protocol.json \
  experiments/projection_analysis_protocol.json
echo "design_sources_changed_after_results_exit=$?"
```

Result:

```text
HEAD=5a5cc2041ad5b6194a86aff9e61460873ce185c9
protocol_before_results_exit=0
protocol_record_changed_before_results_exit=0
design_sources_changed_after_results_exit=0
```

Exit zero from `merge-base --is-ancestor` establishes that the protocol commit
precedes the audited-results commit. Exit zero from both path-restricted diffs
establishes that the comparison record and machine-readable design sources are
unchanged across the stated ranges.

### 66.2 Read only the frozen design and comparison clauses

```bash
git show \
  ba64926aed98b08b7b285266cf85989d466f9f1c:\
docs/records/predictive_action_projection_experimental_protocol.md \
  | sed -n '145,270p'

jq '{study_id,protocol_role,evaluation_policy_mode,expected_train_seeds,expected_training_timesteps,expected_repeats_per_layout,methods}' \
  experiments/fixed_training_geometry_analysis_protocol.json

jq '{study_id,protocol_role,evaluation_policy_mode,expected_train_seeds,expected_training_timesteps,expected_repeats_per_layout,methods}' \
  experiments/projection_analysis_protocol.json
```

The extracted protocol section declares the 3-by-2 evaluation matrix, primary
and supporting outcome roles, six direct comparisons, descriptive transition
tables, checkpoint-level replicate rule, and separate primary fixed-geometry
and secondary deterministic-transfer roles.

### 66.3 Map the six comparisons to the five questions

The comparison-to-question mapping is deterministic:

```text
baseline off/on             -> RQ1
baseline off/high-penalty off -> RQ3
baseline off/projection-trained off -> RQ2
projection-trained off/on   -> RQ1 and supporting RQ5
baseline on/projection-trained on -> RQ2 and supporting RQ5
high-penalty off/on         -> RQ1; supporting context for RQ3 and RQ5
```

Two independent methodological reviews identified the same necessary
qualification: high-penalty on versus baseline on is not one of the six
predeclared direct comparisons. The locked RQ3 therefore contains only the
baseline-off versus high-penalty-off training contrast. High-penalty off/on is
reported under RQ1, while any high-penalty-on versus baseline-on result is
exploratory.

The reviews also classified formal difference-in-differences interactions,
pooled or formal cross-suite effect differences, post hoc layout groups,
intervention-outcome associations, and substitution/equivalence comparisons as
exploratory because they are not direct frozen comparisons.

### 66.4 Step 3 completion result

Five research questions are locked: three primary fixed-geometry questions,
one secondary prespecified transfer question, and one supporting filter-use
question. Every question now maps to frozen comparisons, prespecified metrics,
or the secondary-suite role. No outcome value was used. The gate passes and the
continuation state advances to Phase I, Step 4.

## Commands still to be added

- Phase I, Step 5 metric-dictionary verification procedures after Salvador
  explicitly authorizes that step.

## 67. Phase I, Step 4 — verify units, pairing, aggregation, and uncertainty

Step 4 was authorized on 28 August 2026 and executed read-only against the
audited repository. No repository file was modified.

### 67.1 Verify raw identities, pairing keys, and outcome exhaustiveness

Both canonical episode tables were loaded directly. The audit grouped by
`method`, `train_seed`, and `projection_mode`; checked the unique key
`method, train_seed, checkpoint_sha256, projection_mode, layout_id,
layout_repeat, evaluation_seed`; compared sorted disabled/enabled pairing keys;
and verified that integer-cast `success + collision + truncated = 1` on every
row.

Result:

```text
SUITE fixed_training_geometry
rows 3000; methods 3; checkpoints 15; modes disabled/enabled
layouts 1; evaluation seeds 10000..10099
duplicate_keys 0
cell_size_unique [100]; cells 30
paired_rows 1500 1500; pair_keys_equal True
outcomes_sum1 True; bad_outcome_rows 0
one_checkpoint_per_method_seed True

SUITE core_layout_transfer
rows 720; methods 3; checkpoints 15; modes disabled/enabled
layouts 24; evaluation seeds 1000..1023
duplicate_keys 0
cell_size_unique [24]; cells 30
paired_rows 360 360; pair_keys_equal True
outcomes_sum1 True; bad_outcome_rows 0
one_checkpoint_per_method_seed True
```

### 67.2 Verify table hierarchy and reproduce summary statistics

The checkpoint, method, paired-delta, and paired-summary CSVs were loaded for
each suite. Method means and sample SDs were recomputed from checkpoint rows;
paired means and sample SDs were recomputed from the five checkpoint deltas.

Result:

```text
fixed_training_geometry:
checkpoint rows 30; paired rows 15; method rows 6; paired-summary rows 3
checkpoint episode counts [100]; layouts [1]
five checkpoint rows per method/mode; five paired rows per method
maximum method-summary recomputation error 8.881784197001252e-16
maximum paired-summary recomputation error 7.105427357601002e-15

core_layout_transfer:
checkpoint rows 30; paired rows 15; method rows 6; paired-summary rows 3
checkpoint episode counts [24]; layouts [24]
five checkpoint rows per method/mode; five paired rows per method
maximum method-summary recomputation error 4.440892098500626e-16
maximum paired-summary recomputation error 1.4210854715202004e-14
```

### 67.3 Inspect the aggregation implementation and protocol

The following frozen sources were read directly:

```text
analysis/aggregate_projection_results.py
experiments/fixed_training_geometry_analysis_protocol.json
experiments/projection_analysis_protocol.json
docs/records/predictive_action_projection_experimental_protocol.md
```

`checkpoint_summary()` averages episode/layout rows within checkpoint and
mode. `method_summary()` gives equal weight to five checkpoint summaries and
uses `std(ddof=1)`. `paired_deltas()` performs a one-to-one off/on merge within
method and checkpoint on layout, repeat, and evaluation seed.
`paired_summary()` then averages the five checkpoint effects and uses sample
SD. The two suites use separate protocols and output directories.

The protocol explicitly identifies the independently trained checkpoint as the
empirical replicate, requires separate suite reporting, and predeclares no
formal significance test. It does not declare numeric training seeds as
cross-method blocks; no cross-method pairing exists in the aggregation code.

### 67.4 Step 4 qualifications

The audit also established:

```text
paired_layout_count=100 in fixed geometry means paired evaluation episodes
transfer clearance has 23 defined layouts because control_open_route has no obstacles
disabled projection metrics are not applicable rather than zero burden
episode/layout averages and pooled step rates are different estimands
paired maximum-clipping deltas differ from differences of checkpoint-wide maxima
timeout is exactly 1-success-collision
```

### 67.5 Step 4 completion result

The Step 4 gate passes. The continuation state advances to Phase I, Step 5:
build the exact metric and evidence dictionary before interpreting supporting
metrics.

## 68. Phase I, Step 5 — verify the metric and evidence dictionary

Step 5 was authorized and executed read-only on 28--29 August 2026. The
repository was not modified.

### 68.1 Verify the frozen source scope

```bat
git status --short
git rev-parse HEAD
git diff --name-status b005123cb2c6c754a991d1e7fdc709437b90e915..HEAD
```

The working tree was clean. The only paths changed after the audited result
commit were public-release documentation and licensing paths; metric-generating
code, protocols, tables, and figures were unchanged.

The following definition and reduction sources were inspected with numbered
lines:

```text
environments/constrained_navigation.py
environments/action_wrappers.py
environments/factory.py
algorithms/ppo/action_bound_diagnostics.py
algorithms/ppo/projection_training.py
algorithms/ppo/ppo_continuous_action.py
projection/cbf_qp_projection.py
projection/cbf_qp_wrapper.py
evaluation/evaluate_policy.py
evaluation/evaluate_layout_suite.py
evaluation/trajectory_recording.py
analysis/aggregate_projection_results.py
analysis/training_diagnostics.py
analysis/plot_projection_results.py
docs/contracts/environment_and_projection.md
docs/contracts/evaluation_and_artifacts.md
docs/contracts/trajectory_archive.md
```

### 68.2 Audit the committed episode schemas

Both committed episode tables were loaded with pandas. The audit compared
column order, inferred data types, missingness, unique method/checkpoint/layout
identities, and numeric ranges.

Result:

```text
fixed_training_geometry: 3000 rows x 53 columns
core_layout_transfer:     720 rows x 53 columns
identical column order and inferred dtypes
all fields complete except transfer min_obstacle_clearance
transfer clearance structural NA: 30 rows, all control_open_route
```

The two builder-added fields are `source_csv` and
`result_build_schema_version`. The original evaluator shards and full
trajectory archives are not committed; the canonical episode tables are the
lowest-level public numerical evidence.

### 68.3 Verify outcome, goal-distance, clearance, and reward identities

The direct row checks were equivalent to:

```python
active = episodes["min_obstacle_clearance"].notna()

assert (
    episodes.loc[episodes["success"], "final_distance_to_goal"]
    <= episodes.loc[episodes["success"], "layout_goal_radius"] + 1e-12
).all()

assert (
    episodes.loc[~episodes["success"], "final_distance_to_goal"]
    > episodes.loc[~episodes["success"], "layout_goal_radius"]
).all()

assert (
    episodes.loc[episodes["collision"] & active, "min_obstacle_clearance"]
    <= 1e-12
).all()

assert (
    episodes.loc[(~episodes["collision"]) & active, "min_obstacle_clearance"]
    > 0.0
).all()

assert (
    episodes.loc[episodes["truncated"], "episode_length"]
    == episodes.loc[episodes["truncated"], "max_episode_steps"]
).all()

assert (
    episodes["terminated"]
    == (episodes["success"] | episodes["collision"])
).all()

assert (
    episodes["success"].astype(int)
    + episodes["collision"].astype(int)
    + episodes["truncated"].astype(int)
    == 1
).all()
```

All assertions passed in both suites. Every evaluation collision penalty was
10. The per-step reward formula and its telescoped episode form were derived
directly from `ConstrainedNavigationEnv.step()` and `run_episode()`.

### 68.4 Verify clipping formulas

For each of overall, speed, and turn clipping, the audit recomputed

```python
rate = count / episode_length
```

and checked finite integer counts, bounds, mean norm not exceeding episode
maximum, zero consistency, and

```text
max(speed_count, turn_count) <= overall_count
overall_count <= speed_count + turn_count
```

Maximum absolute rate error across all 3,720 rows was below
`8.4e-17`. No field was missing. The wrapper/source inspection established that
the clipping norm is in raw normalized-action coordinates and precedes the
physical-action projector.

### 68.5 Verify projector metrics, structural values, and exposure

The audit checked the raw intervention identity:

```python
projection_intervention_rate
== projection_intervention_count / episode_length
```

Maximum absolute error was below `1.12e-16`. All enabled correction/slack
values were finite and nonnegative; all failures were zero. Disabled raw
burden placeholders were exact zero, and every disabled checkpoint burden
field was `NaN` after canonical aggregation.

Enabled transition exposures were computed by summing `episode_length`; active
QP exposure excluded the obstacle-free transfer layout:

```text
training:            256000 enabled; 256000 active-QP; 0 failures
fixed evaluation:    236084 enabled; 236084 active-QP; 0 failures
transfer evaluation:  62900 enabled;  60185 active-QP; 0 failures
combined:             554984 enabled; 552269 active-QP; 0 failures
```

The remaining 2,715 transfer steps used `no_active_constraints` and did not
invoke OSQP.

### 68.6 Verify aggregation and plotting semantics

`checkpoint_summary()` was confirmed to:

- mean ordinary values and rates over episodes/layouts;
- exclude structural clearance `NaN`s;
- convert disabled projector burdens to `NaN`;
- maximize episode maximum clipping, correction, and slack values.

`method_summary()` then computes the equal-checkpoint mean and sample SD with
`ddof=1`. `paired_deltas()` pairs task and clipping metrics, but correctly omits
projector burdens because the disabled value is not applicable.

The difference between equal-observation and pooled-step rates was recomputed.
For projection-trained transfer evaluation:

```text
canonical mean layout-level intervention rate = 0.567006
pooled transition intervention fraction       = 0.690070
```

All 12 evaluation figures per suite read checkpoint summaries. Timeout is
derived at plot time. The generated LaTeX outcome table omits timeout and uses
fixed three-decimal formatting; it is not approved for unchanged report use.

### 68.7 Verify training-diagnostic transformations

The structured training tables and figure audit reconcile to:

```text
training_scalar_events.csv:       36536 rows
training_episode_diagnostics.csv:  5131 rows
training_rollout_diagnostics.csv:   750 rows
training_curve_points.csv:        14811 rows
```

Each rollout represents `4 environments x 256 steps = 1024 transitions`; 50
rollouts per seed give 768,000 total training transitions. Projector rollout
fields are structurally absent for baseline/high-penalty methods and cover
256,000 projection-trained transitions.

Episode rolling metrics use 20 completed episodes with `min_periods=1`.
Direct curves use within-seed linear interpolation to a common overlapping
grid; derived episode curves use right-continuous carry-forward. The across-seed
display is mean plus/minus one sample SD, not a confidence interval.

### 68.8 Step 5 qualifications and result

The complete per-step evaluation archives are not public. Counts/rates and
row-level identities are independently reproducible from committed fields;
mean/max norms and returns have exact generating definitions but cannot all be
re-summed from absent per-step public values.

Two nonblocking builder gaps were recorded:

```text
projection_intervention_count is not a declared REQUIRED_COLUMNS field
the builder does not check intervention rate = count / episode length
the builder does not assert disabled burden placeholders are zero before N/A conversion
```

The frozen data independently pass all three checks. No numerical mismatch was
found, and no Step 5 blocker remains.

The Step 5 gate passes. Phase I is complete. The continuation state advances
to Phase II, Step 6: fixed-geometry absolute performance.

## 69. Phase II, Step 6 — fixed-geometry absolute performance

Step 6 was executed read-only on 29 August 2026. The verification routine read:

```text
results/tables/fixed_training_geometry/evaluation_episode_results.csv
results/tables/fixed_training_geometry/checkpoint_summary.csv
results/tables/fixed_training_geometry/method_summary.csv
```

The local audit command was:

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" tmp/step6_fixed_absolute_audit.py
```

The audit performed these explicit operations:

```text
1. Require 3,000 raw rows.
2. Group by method, train_seed, and projection_mode.
3. Require 30 groups and exactly 100 episodes per group.
4. Derive timeout = 1 - success - collision for every episode.
5. Require timeout to match truncated and S + C + T = 1 in every row.
6. Recompute checkpoint means for return, length, success, collision, timeout,
   and minimum clearance.
7. Join reconstructed and committed checkpoint rows one-to-one.
8. Average the five checkpoint values equally by method and mode.
9. Compute sample SD across checkpoints with ddof=1.
10. Join reconstructed and committed method rows one-to-one.
11. Inventory outcome counts and checkpoint ranges.
12. Expand the projection-trained enabled S/C/T means and SDs manually.
```

The reconciliation output was:

```text
PASS raw inventory: 3,000 unique rows, 30 groups, 100 episodes/group
PASS outcome identity: success + collision + timeout = 1 for every row
checkpoint return maximum error:       1.776e-15
checkpoint length maximum error:       0
checkpoint success maximum error:      0
checkpoint collision maximum error:    0
checkpoint clearance maximum error:    9.714e-17
method-level maximum error:             2.842e-14
PASS Phase II Step 6 fixed-geometry absolute reconstruction
```

The manual projection-trained enabled success check was:

```text
values = 0.78, 0.96, 0.97, 0.99, 0.98
mean = 4.68 / 5 = 0.936000
squared deviations = 0.024336, 0.000576, 0.001156, 0.002916, 0.001936
sample variance = 0.030920 / 4 = 0.007730
sample SD = sqrt(0.007730) = 0.087920419
```

No frozen evidence file was modified. Step 6 passes. The fixed-geometry paired
analysis and transfer analysis are recorded below.

## 70. Phase II, Step 7 - fixed-geometry paired projection effects

The paired reconstruction read the fixed-suite episodes and the frozen paired
tables, matched disabled and enabled rows within checkpoint and evaluation key,
and calculated one enabled-minus-disabled effect per trained checkpoint before
the five-checkpoint summary.

The reusable preservation command is:

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" \
  analysis/interpretation/summarize_paired_projection_effects.py \
  --episodes results/tables/fixed_training_geometry/evaluation_episode_results.csv \
  --protocol experiments/fixed_training_geometry_analysis_protocol.json \
  --layout-suite evaluation/layouts/fixed_training_geometry.json \
  --paired-deltas results/tables/fixed_training_geometry/paired_projection_deltas.csv \
  --paired-summary results/tables/fixed_training_geometry/paired_projection_summary.csv \
  --worked-example-method ppo_baseline \
  --worked-example-train-seed 1 \
  --output tmp/interpretation/fixed_geometry_paired_projection_effects.json \
  --label fixed-geometry-paired-projection-effects
```

The source implementation used for preservation has SHA-256:

```text
summarize_paired_projection_effects.py
d584d008d1170336ebc153468955d875b52f10800d80499ae4d89e2ea73b64df
```

A previously verified canonical Step 7 JSON has SHA-256:

```text
e81ea36deeb496e69cb3034e2de34d351954b42d4d555e09bf8092bab3b3a19d
```

Its temporary output pathname was not preserved in the verified record and is
therefore unknown. The repository command above defines the intended future
path, but that path is not represented as though it were the historical one.
Its output hash must be checked after execution because caller-supplied path
strings and labels are part of the deterministic result.

The verified paired results were:

```text
baseline S/C/T:           +2.6 / -13.8 / +11.2 pp
high-penalty S/C/T:       +0.2 / -12.8 / +12.6 pp
projection-trained S/C/T: +76.2 / -80.4 / +4.2 pp
```

All means use five checkpoint-level paired effects and sample SD uses
`ddof=1`. The Step 7 gate passed.

## 71. Phase II, Step 8 - transfer-suite performance

### 71.1 Exact read-only reconstruction command used for the completed analysis

The descriptive transfer analysis is preserved by a script whose name states
its operation rather than a workflow number. Every source path is supplied
explicitly:

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" \
  analysis/interpretation/summarize_layout_transfer_performance.py \
  --episodes results/tables/core_layout_transfer/evaluation_episode_results.csv \
  --protocol experiments/projection_analysis_protocol.json \
  --layout-suite evaluation/layouts/core_navigation_layouts.json \
  --checkpoint-summary results/tables/core_layout_transfer/checkpoint_summary.csv \
  --method-summary results/tables/core_layout_transfer/method_summary.csv \
  --paired-deltas results/tables/core_layout_transfer/paired_projection_deltas.csv \
  --paired-summary results/tables/core_layout_transfer/paired_projection_summary.csv \
  --build-audit results/tables/core_layout_transfer/result_build_audit.json \
  --worked-example-method ppo_train_projection \
  --worked-example-train-seed 2 \
  --worked-example-layout control_open_route \
  --output tmp/interpretation/core_layout_transfer_performance.json \
  --label phase_ii_transfer_suite_performance
```

The command reconstructs checkpoint and method absolute summaries, paired
checkpoint effects, descriptive layout effects, outcome breadth, sign counts,
and leave-one-checkpoint-out means. It validates the supplied protocol,
layout suite, build audit, and committed summary tables against the 720 raw
episode rows. It does not rerun a policy or alter frozen evidence.

This command has already been executed. It is preserved here as provenance and
is not a required script-installation or post-installation verification step.
Generic command-line usage belongs in `analysis/interpretation/README.md`.

### 71.2 Input and implementation identities

```text
6616c00116b40c7456d92c142b82c006379f44e095fac9a0e19674d6b79cfc08  analysis/interpretation/summarize_layout_transfer_performance.py
250789508fba8850e07bc19d1aef0b93ea8348b6b14eb09594469a6c62b53d5f  analysis/interpretation/_evaluation_evidence.py
8091681f988009657e2951089fcc520e3960143f48ef8dbda5c2fcfce6e7f4ae  results/tables/core_layout_transfer/evaluation_episode_results.csv
8f7697c658089fe80145c0c996450070980a8ce615e549a7f5266203aa8c49df  results/tables/core_layout_transfer/checkpoint_summary.csv
bfa713faaf170c7585fac4bdcdb5ee592db68d15b3124800ee299f2a204733cd  results/tables/core_layout_transfer/method_summary.csv
75d4a7c778d9ff5412603d6e4a0b27872120d38e10c0f763b9126f3c06ab8848  results/tables/core_layout_transfer/paired_projection_deltas.csv
9408e896a270965b6e4dbf69df955786c4ae4421880413b47c6aa423bea89159  results/tables/core_layout_transfer/paired_projection_summary.csv
acb1fa12ff5d24491719def11f049f542828f05f85e3e259f6edbbb390c4d82d  results/tables/core_layout_transfer/result_build_audit.json
8ba2cced1feb13207ca4f594ee4dfdf1930ae5da486e376d040c54491c586d28  experiments/projection_analysis_protocol.json
b4dfaf5589a57510735d6ce3ffae60118852145194b250c756ed604ecf72c385  evaluation/layouts/core_navigation_layouts.json
```

The recorded command completed successfully. Its deterministic output
identity is:

```text
cbd3bef76db967b9436f12c330698d7ec0228bf0c8704a42363353e67b5bf30a  tmp/interpretation/core_layout_transfer_performance.json
```

All input reconciliation and aggregation checks passed. The exact unrounded
projection-trained terminal-outcome mean deltas were +5.833333, -50.833333,
and +45.000000 percentage points for success, collision, and timeout.

### 71.3 Verified reductions

The raw design reconciliation was:

```text
3 methods x 5 checkpoints x 2 projection modes x 24 layouts = 720 rows
24 deterministic actor-mean episodes per checkpoint-mode cell
5 independent trained checkpoints per method
```

Checkpoint absolute outcomes were averaged equally within method and sample SD
was calculated with `ddof=1`. Enabled-minus-disabled effects were calculated
within checkpoint before method aggregation. For projection-trained PPO:

```text
disabled S/C/T: 27.50 +/- 3.73 / 63.33 +/- 3.49 / 9.17 +/- 5.43 percent
enabled S/C/T:  33.33 +/- 4.17 / 12.50 +/- 6.59 / 54.17 +/- 7.80 percent
paired S/C/T:   +5.83 +/- 2.28 / -50.83 +/- 9.50 / +45.00 +/- 9.95 pp
```

All five projection-trained checkpoint effects had positive success, negative
collision, and positive timeout. Leave-one-checkpoint-out means retained those
directions. Layout-mean collision effects were negative in 20 of 24 layouts;
success effects were positive in six. The open-route and upper/lower
observations were recorded only as exploratory.

The fixed and transfer suites were not pooled. The cross-suite descriptive
comparison was explicitly bounded because action-selection mode, repetition
structure, and geometry all differ. The Step 8 gate passed. The immediate next
action is Phase II, Step 9: paired terminal-outcome transition analysis.

## 72. Correct the interpretation-package execution boundary

The superseded package placed a complete transfer-analysis command immediately
after its installation checks without identifying it as optional. The
maintainer subsequently clarified that only the ZIP contents were copied. The
long transfer command was not executed. There was therefore no redundant
analysis execution, no new analysis output, and no second result identity to
record. The successful transfer analysis remains the single execution recorded
in Section 71.

The documentation correction made no change to any Python implementation or
test module. The corrected cumulative archive was extracted into an isolated
directory and verified with:

```bat
python -m tests.test_interpretation_analysis -v
```

Result: all 15 focused tests passed. The corrected installation guide now
contains only post-copy verification commands. Generic interface examples
remain in `analysis/interpretation/README.md`; exact study executions remain in
this established Analysis Command Record. No second interpretation command log
is used.

## 73. Phase II, Step 9 - paired terminal-outcome correspondences

### 73.1 Package-lineage verification before Step 9

The archive Salvador copied was:

```text
64721aa77e00fe5a7565b981442dde91b8b875f497ffd79b9446f883bbba6350  Predictive_Action_Projection_Interpretation_Scripts_Through_Transfer.zip
```

Its optional-command and documentation ambiguity is resolved in Section 72.
The later cumulative archive with the corrected installation material was:

```text
e74093522462d7fbb1e3e793513789d9fab275a8f01ab58beaf34300df685430  Predictive_Action_Projection_Interpretation_Scripts_Through_Transfer_Corrected.zip
```

Direct inspection shows that both archives contain the same correct
`analysis.interpretation` test import and the same test-module SHA-256:

```text
380ce9016f7edd819d8546b1ccafc2a9a8401fea41567048a4ecf45111d729e0  tests/test_interpretation_analysis.py
```

The corrected cumulative archive passes all 15 tests in a clean repository
installation. The cumulative Step 9 archive supersedes both through-transfer
archives only because it adds the Step 9 program, expanded tests, updated
records, and canonical reference outputs. This lineage verification changes no
scientific implementation, frozen evidence file, result table, or Salvador
worktree file.

### 73.2 Read-only commands

The Step 9 program pairs disabled and enabled rows within method, trained
checkpoint, layout, repeat, and evaluation seed. It derives the exclusive
terminal outcome from `success`, `collision`, and `truncated`, then emits all
nine cells with disabled outcomes as rows and enabled outcomes as columns.
The exact study arguments are shown below in Windows Command Prompt syntax.

Fixed-training-geometry command:

```bat
python -m analysis.interpretation.summarize_paired_terminal_outcome_transitions ^
  --episodes results/tables/fixed_training_geometry/evaluation_episode_results.csv ^
  --protocol experiments/fixed_training_geometry_analysis_protocol.json ^
  --layout-suite evaluation/layouts/fixed_training_geometry.json ^
  --checkpoint-summary results/tables/fixed_training_geometry/checkpoint_summary.csv ^
  --method-summary results/tables/fixed_training_geometry/method_summary.csv ^
  --paired-deltas results/tables/fixed_training_geometry/paired_projection_deltas.csv ^
  --paired-summary results/tables/fixed_training_geometry/paired_projection_summary.csv ^
  --build-audit results/tables/fixed_training_geometry/result_build_audit.json ^
  --output tmp/interpretation/fixed_geometry_terminal_outcome_transitions.json ^
  --label fixed-geometry-paired-terminal-outcome-transitions
```

Core-layout-transfer command:

```bat
python -m analysis.interpretation.summarize_paired_terminal_outcome_transitions ^
  --episodes results/tables/core_layout_transfer/evaluation_episode_results.csv ^
  --protocol experiments/projection_analysis_protocol.json ^
  --layout-suite evaluation/layouts/core_navigation_layouts.json ^
  --checkpoint-summary results/tables/core_layout_transfer/checkpoint_summary.csv ^
  --method-summary results/tables/core_layout_transfer/method_summary.csv ^
  --paired-deltas results/tables/core_layout_transfer/paired_projection_deltas.csv ^
  --paired-summary results/tables/core_layout_transfer/paired_projection_summary.csv ^
  --build-audit results/tables/core_layout_transfer/result_build_audit.json ^
  --output tmp/interpretation/core_layout_transfer_terminal_outcome_transitions.json ^
  --label core-layout-transfer-paired-terminal-outcome-transitions
```

These commands only read the supplied evidence families and create new JSON
files exclusively. They do not rerun evaluation, regenerate a frozen result,
or overwrite an existing output.

### 73.3 Implementation, input, and output identities

```text
751d3336863f772c0b700f1ec52c60ff34e9b6989abacd68cdd604c035c64825  analysis/interpretation/summarize_paired_terminal_outcome_transitions.py
16c3c3da1eefc44619a58ffaac4a8bae6886180d8587f8e0128e33f22801d449  tests/test_interpretation_analysis.py

4eafae0edfc0ff75e78a884cec2dbcd5cbec4408c772bb4c9f305d183645d750  results/tables/fixed_training_geometry/evaluation_episode_results.csv
1730f793b7a8b7eaa985cfe2850578c62096610851eb492f926bee45f29184bc  experiments/fixed_training_geometry_analysis_protocol.json
ce0f52bcdbe77259d5e0c5ad38ab42fc6010590e17faf2487efa692ac2e64f7e  evaluation/layouts/fixed_training_geometry.json
f00c16e40f590fe00d8266d729b57260a11562d8b7384234145d66288ae4dd53  results/tables/fixed_training_geometry/checkpoint_summary.csv
d3ac057c51a6908bc6693d9710bb594611e6043fb0d4310fca2b7ba1c3902e62  results/tables/fixed_training_geometry/method_summary.csv
39f99acc31189315cfc66fb51526f56ac4dce00b765644eab3573614fe93e0e6  results/tables/fixed_training_geometry/paired_projection_deltas.csv
73e8edba2834ea7db7ef9e7f984d86e17779b01c81c725d25fa0ff4cd735ee51  results/tables/fixed_training_geometry/paired_projection_summary.csv
e0fa01617c4581ea172f18f8a08c4a1733acbbf8fee435e62e39b6e56ced14fc  results/tables/fixed_training_geometry/result_build_audit.json
890b3c031ed4e1315185bd4444237523326cee251822d21ea87089bdd0f9000b  tmp/interpretation/fixed_geometry_terminal_outcome_transitions.json

8091681f988009657e2951089fcc520e3960143f48ef8dbda5c2fcfce6e7f4ae  results/tables/core_layout_transfer/evaluation_episode_results.csv
8ba2cced1feb13207ca4f594ee4dfdf1930ae5da486e376d040c54491c586d28  experiments/projection_analysis_protocol.json
b4dfaf5589a57510735d6ce3ffae60118852145194b250c756ed604ecf72c385  evaluation/layouts/core_navigation_layouts.json
8f7697c658089fe80145c0c996450070980a8ce615e549a7f5266203aa8c49df  results/tables/core_layout_transfer/checkpoint_summary.csv
bfa713faaf170c7585fac4bdcdb5ee592db68d15b3124800ee299f2a204733cd  results/tables/core_layout_transfer/method_summary.csv
75d4a7c778d9ff5412603d6e4a0b27872120d38e10c0f763b9126f3c06ab8848  results/tables/core_layout_transfer/paired_projection_deltas.csv
9408e896a270965b6e4dbf69df955786c4ae4421880413b47c6aa423bea89159  results/tables/core_layout_transfer/paired_projection_summary.csv
acb1fa12ff5d24491719def11f049f542828f05f85e3e259f6edbbb390c4d82d  results/tables/core_layout_transfer/result_build_audit.json
95cf84fe6b2fc6e76814bd88d2dd1efb573afc4d77544b80a57b92eb9741122c  tmp/interpretation/core_layout_transfer_terminal_outcome_transitions.json
```

### 73.4 Validation and reconciliation

The focused regression command was:

```bat
python -m tests.test_interpretation_analysis -v
```

Result: all 21 tests passed. The tests cover arbitrary input paths and working
directories, a hand-calculated sparse 3 by 3 matrix, explicit zero cells,
checkpoint aggregation with sample SD using `ddof=1`, malformed terminal
outcomes, missing and duplicate pairs, tampered checkpoint/method/paired/audit
tables, deterministic output, and refusal to overwrite an existing artifact.

Both real executions returned `status: PASS`. Fixed geometry contained 1,500
matched observation pairs, 100 per checkpoint and 500 per method. Transfer
contained 360 matched pairs, 24 per checkpoint and 120 per method. Each method
retained five trained checkpoints as the independent units. For each suite,
270 transition-margin and delta comparisons reconciled. All integer margins
and zero-sum outcome deltas were exact. The maximum numerical error was
`1.1102230246251565e-16`, below the declared `1e-12` tolerance.

### 73.5 Pooled descriptive matrices

Rows are projection-disabled outcomes and columns are projection-enabled
outcomes, both ordered success, collision, timeout. Pooled counts describe
matched evaluation coverage, not 500 or 120 independent policy replicates.

Fixed training geometry:

```text
PPO baseline             [[73, 0, 3], [12, 1, 57], [4, 0, 350]]
PPO high penalty         [[ 0, 0, 0], [ 0, 3, 64], [1, 0, 432]]
PPO trained projection   [[84, 1, 2], [382, 6, 21], [2, 0, 2]]
```

Transfer layouts:

```text
PPO baseline             [[ 9, 0, 0], [0,  4,  2], [0, 0, 105]]
PPO high penalty         [[ 0, 0, 0], [0,  0,  2], [0, 0, 118]]
PPO trained projection   [[33, 0, 0], [7, 15, 54], [0, 0,  11]]
```

In fixed geometry, 382 of 409 projection-trained disabled collisions
corresponded to enabled successes, 21 to timeouts, and six remained
collisions. The collision-to-success counts by checkpoint were 73, 84, 75,
75, and 75. For baseline, disabled collisions corresponded most often to
timeout, 57 of 70, and only 12 of 70 corresponded to success. For high-penalty
PPO, 64 of 67 disabled collisions corresponded to timeout and none to success.

In transfer, all off-diagonal cells originated from disabled collisions.
Projection-trained disabled collisions corresponded to success in 7 of 76
cases, timeout in 54 of 76, and collision in 15 of 76. The checkpoint
collision-to-success counts were 1, 2, 1, 1, and 2; the corresponding
collision-to-timeout counts were 9, 8, 11, 14, and 12. Baseline and
high-penalty collision changes again produced no new successes.

These are matched terminal-outcome correspondences between two controller
executions. They are not temporal transitions within one trajectory and do
not prove deliberate projector reliance, a causal policy-projector learning
mechanism, formal safety, performance under another projector, or arbitrary
geometric generalization.

### 73.6 Step gate

Step 9 passes. All checkpoint profiles, all nine cells, pooled method counts,
row and column margins, sample summaries, and supplied-table reconciliations
were verified without modifying frozen evidence. The next scientific step is
Phase II, Step 10: attribute results carefully to nominal-policy behavior and
policy-plus-projector composite-controller behavior. Step 10 is not authorized
by completion of this record and requires its own explanation and approval.


## Completed-repository handoff, 7 September 2026

The final source, evidence-inclusion, documentation, and verification decisions
are recorded in [repository release audit](repository_release_audit.md). The
completed empirical findings and inferential boundaries are consolidated in the
[final study record](final_study_record.md). Read-only interpretation outputs and
the exact reconstruction argument arrays are retained under
`results/interpretation/`. These reconstructions use the same frozen input data;
no final training, policy evaluation, calibration, or benchmark was repeated.

For the completed release, run the full tests and
`python -m evaluation.verify_repository_release --base-ref main`. After staging,
add `--require-tracked` to verify the exact file inventory and staged content.
The historical pre-experiment runner remains available for future development
and is not the merge gate for this completed study.
