# Predictive Action Projection with PPO: Analysis Command Record

**Author:** Salvador Tenorio\
**Status:** Living command record\
**Frozen source:** `ba64926aed98b08b7b285266cf85989d466f9f1c`\
**Repository path:** `docs/records/Predictive_Action_Projection_Analysis_Command_Record.md`\
**Started:** 2026-08-22

## Purpose and scope

This record collects the exact Windows Command Prompt commands used during Step 6 to build, verify, and inspect the frozen study results. It is intended to accompany `predictive_action_projection_analysis_record.md`.

The commands consume frozen checkpoints, evaluation CSV/NPZ artifacts, trajectory archives, and saved TensorBoard records. They perform no training or evaluation. The raw dataset audit, checksum creation, and external backup belong to Step 5 and are not repeated here.

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

## Commands still to be added

- Exact commit, push, and public-release verification commands after command 57 passes.
