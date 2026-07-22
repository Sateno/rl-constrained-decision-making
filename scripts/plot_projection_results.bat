@echo off
setlocal

cd /d "%~dp0.."

if "%~3"=="" (
    echo Usage: scripts\plot_projection_results.bat protocol.json evaluation_dir output_dir [runs_dir]
    exit /b 2
)

set "PROTOCOL=%~1"
set "EVALUATION_DIR=%~2"
set "OUTPUT_DIR=%~3"
set "RUNS_DIR=%~4"
set "TABLES_DIR=%OUTPUT_DIR%\tables"
set "FIGURES_DIR=%OUTPUT_DIR%\figures"
set "SUMMARY_PATH=%OUTPUT_DIR%\result_build_summary.txt"

if exist "%SUMMARY_PATH%" del /q "%SUMMARY_PATH%"

python -m analysis.aggregate_projection_results ^
  --protocol "%PROTOCOL%" ^
  --evaluation-dir "%EVALUATION_DIR%" ^
  --output-dir "%TABLES_DIR%"

if errorlevel 1 exit /b 1

if defined RUNS_DIR (
    python -m analysis.plot_projection_results ^
      --protocol "%PROTOCOL%" ^
      --tables-dir "%TABLES_DIR%" ^
      --evaluation-dir "%EVALUATION_DIR%" ^
      --figures-dir "%FIGURES_DIR%" ^
      --runs-dir "%RUNS_DIR%"
) else (
    python -m analysis.plot_projection_results ^
      --protocol "%PROTOCOL%" ^
      --tables-dir "%TABLES_DIR%" ^
      --evaluation-dir "%EVALUATION_DIR%" ^
      --figures-dir "%FIGURES_DIR%"
)

if errorlevel 1 exit /b 1

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
(
    echo status=PASS
    echo protocol=%PROTOCOL%
    echo evaluation_dir=%EVALUATION_DIR%
    echo output_dir=%OUTPUT_DIR%
) > "%SUMMARY_PATH%"

echo Result build completed successfully.
echo Summary: %SUMMARY_PATH%

endlocal
