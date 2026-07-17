@echo off
setlocal EnableExtensions

cd /d "%~dp0.."
if errorlevel 1 (
    echo ERROR: could not enter the repository root.
    exit /b 1
)

set "REQUIRED_CONDA_ENV=RL_PROJECTS"

if /I not "%CONDA_DEFAULT_ENV%"=="%REQUIRED_CONDA_ENV%" (
    where conda >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Conda is not available on PATH.
        echo Open an Anaconda or Miniforge prompt and run this script again.
        exit /b 1
    )

    echo Activating Conda environment %REQUIRED_CONDA_ENV%...
    call conda activate "%REQUIRED_CONDA_ENV%"
    if errorlevel 1 (
        echo ERROR: could not activate Conda environment %REQUIRED_CONDA_ENV%.
        exit /b 1
    )
)

if /I not "%CONDA_DEFAULT_ENV%"=="%REQUIRED_CONDA_ENV%" (
    echo ERROR: active Conda environment is "%CONDA_DEFAULT_ENV%".
    echo Expected "%REQUIRED_CONDA_ENV%".
    exit /b 1
)

python -m evaluation.validate_evaluation_time_projection
set "VALIDATION_EXIT_CODE=%ERRORLEVEL%"

if not "%VALIDATION_EXIT_CODE%"=="0" (
    echo.
    echo Evaluation-time projection runtime validation failed.
    if exist runs\validation\evaluation_time_projection_validation.log (
        echo Review runs\validation\evaluation_time_projection_validation.log.
    ) else (
        echo The Python validation runner exited before creating its log.
    )
    endlocal & exit /b %VALIDATION_EXIT_CODE%
)

echo.
echo Evaluation-time projection runtime validation passed.
echo Summary: runs\validation\evaluation_time_projection_validation_summary.txt
echo Log:     runs\validation\evaluation_time_projection_validation.log

endlocal
exit /b 0
