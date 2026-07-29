@echo off
setlocal EnableExtensions

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

set "REQUIRED_CONDA_ENV=RL_PROJECTS"
set "VALIDATION_ROOT=runs\validation\pre_experiment_codebase"
set "CHECKPOINT_DIR=%VALIDATION_ROOT%\checkpoints"
set "LAYOUT_DIR=%VALIDATION_ROOT%\layout_evaluation"
set "RESULT_DIR=%VALIDATION_ROOT%\result_build"
set "SUMMARY=%VALIDATION_ROOT%\pre_experiment_validation_summary.txt"
set "SEED=9901"
set "TIMESTEPS=2048"

if /I not "%CONDA_DEFAULT_ENV%"=="%REQUIRED_CONDA_ENV%" (
    where conda >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Conda is not available on PATH.
        exit /b 1
    )
    call conda activate "%REQUIRED_CONDA_ENV%"
    if errorlevel 1 exit /b 1
)

if /I not "%CONDA_DEFAULT_ENV%"=="%REQUIRED_CONDA_ENV%" (
    echo ERROR: expected Conda environment %REQUIRED_CONDA_ENV%.
    exit /b 1
)

rem Remove only artifacts reserved for this repeatable validation.
if exist "%VALIDATION_ROOT%" rmdir /s /q "%VALIDATION_ROOT%"
for /d %%D in (runs\ConstrainedNavigation-v0__ppo_baseline_%TIMESTEPS%_seed%SEED%__%SEED%__*) do if exist "%%D" rmdir /s /q "%%D"
for /d %%D in (runs\ConstrainedNavigation-v0__ppo_high_penalty_%TIMESTEPS%_seed%SEED%__%SEED%__*) do if exist "%%D" rmdir /s /q "%%D"
for /d %%D in (runs\ConstrainedNavigation-v0__ppo_train_projection_%TIMESTEPS%_seed%SEED%__%SEED%__*) do if exist "%%D" rmdir /s /q "%%D"
mkdir "%CHECKPOINT_DIR%" >nul
mkdir "%LAYOUT_DIR%" >nul

call :heading "Preflight"
python -m evaluation.validate_pre_experiment_artifacts preflight
if errorlevel 1 goto :failed

call :heading "Compile active source"
python -m compileall -q algorithms environments evaluation analysis projection tests
if errorlevel 1 goto :failed

call :heading "Canonical evaluation-time regression"
call scripts\validate_evaluation_time_projection.bat
if errorlevel 1 goto :failed

call :heading "Canonical summary audit"
python -m evaluation.validate_pre_experiment_artifacts canonical
if errorlevel 1 goto :failed

call :heading "Baseline training smoke"
call scripts\train_ppo_baseline.bat %SEED% %TIMESTEPS% "%CHECKPOINT_DIR%\baseline.pt"
if errorlevel 1 goto :failed

call :heading "High-penalty training smoke"
call scripts\train_ppo_high_penalty.bat %SEED% %TIMESTEPS% "%CHECKPOINT_DIR%\high_penalty.pt"
if errorlevel 1 goto :failed

call :heading "Projection training smoke"
call scripts\train_ppo_projection.bat %SEED% %TIMESTEPS% "%CHECKPOINT_DIR%\train_projection.pt"
if errorlevel 1 goto :failed

call :heading "Training artifact audit"
python -m evaluation.validate_pre_experiment_artifacts training --checkpoint-dir "%CHECKPOINT_DIR%" --output "%VALIDATION_ROOT%\training_smoke_audit.json"
if errorlevel 1 goto :failed

call :heading "Development layouts without projection"
python -m evaluation.evaluate_layout_suite ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --layout-suite evaluation\layouts\development_navigation_layouts.json ^
  --method ppo_baseline ^
  --train-seed 1 ^
  --projection-mode disabled ^
  --repeats-per-layout 1 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --collision-penalty 10.0 ^
  --no-cuda ^
  --output "%LAYOUT_DIR%\baseline_projection_disabled.csv" ^
  --trajectory-output "%LAYOUT_DIR%\baseline_projection_disabled_trajectories.npz"
if errorlevel 1 goto :failed

call :heading "Development layouts with projection"
python -m evaluation.evaluate_layout_suite ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --layout-suite evaluation\layouts\development_navigation_layouts.json ^
  --method ppo_baseline ^
  --train-seed 1 ^
  --projection-mode enabled ^
  --repeats-per-layout 1 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --collision-penalty 10.0 ^
  --projection-lookahead-distance 0.25 ^
  --projection-alpha 2.0 ^
  --projection-slack-penalty 1000.0 ^
  --projection-extra-clearance 0.0 ^
  --no-cuda ^
  --output "%LAYOUT_DIR%\baseline_projection_enabled.csv" ^
  --trajectory-output "%LAYOUT_DIR%\baseline_projection_enabled_trajectories.npz"
if errorlevel 1 goto :failed

call :heading "Common-layout artifact audit"
python -m evaluation.validate_pre_experiment_artifacts layouts --evaluation-dir "%LAYOUT_DIR%" --output "%VALIDATION_ROOT%\layout_evaluation_audit.json"
if errorlevel 1 goto :failed

call :heading "Saved-result table and figure build"
call scripts\plot_projection_results.bat experiments\development_projection_analysis_protocol.json "%LAYOUT_DIR%" "%RESULT_DIR%" runs
if errorlevel 1 goto :failed

call :heading "Saved-result artifact audit"
python -m evaluation.validate_pre_experiment_artifacts results --result-dir "%RESULT_DIR%" --output "%VALIDATION_ROOT%\result_build_audit.json"
if errorlevel 1 goto :failed

(
    echo status=PASS
    echo completed_at=%DATE% %TIME%
    echo canonical_summary=runs\validation\evaluation_time_projection_validation_summary.txt
    echo training_audit=%VALIDATION_ROOT%\training_smoke_audit.json
    echo layout_audit=%VALIDATION_ROOT%\layout_evaluation_audit.json
    echo result_audit=%VALIDATION_ROOT%\result_build_audit.json
    echo figures=%RESULT_DIR%\figures
    echo tables=%RESULT_DIR%\tables
    echo manual_review=%VALIDATION_ROOT%\manual_review.txt
    echo manual_review_required=true
) > "%SUMMARY%"

echo.
echo ============================================================
echo PRE-EXPERIMENT VALIDATION PASSED
echo ============================================================
echo Summary: %SUMMARY%
echo Next:    scripts\inspect_pre_experiment_validation.bat
exit /b 0

:heading
echo.
echo ============================================================
echo %~1
echo ============================================================
exit /b 0

:failed
echo.
echo ============================================================
echo PRE-EXPERIMENT VALIDATION FAILED
echo ============================================================
echo Review the last phase output above.
echo The PASS summary was not created.
exit /b 1
