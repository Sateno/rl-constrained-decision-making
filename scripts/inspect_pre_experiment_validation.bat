@echo off
setlocal EnableExtensions

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

set "ROOT=runs\validation\pre_experiment_codebase"
set "SUMMARY=%ROOT%\pre_experiment_validation_summary.txt"
set "FIGURES=%ROOT%\result_build\figures"
set "TABLES=%ROOT%\result_build\tables"

if not exist "%SUMMARY%" (
    echo ERROR: validation summary not found.
    echo Run scripts\validate_pre_experiment_codebase.bat first.
    exit /b 1
)

echo.
echo Automated validation summary
echo ----------------------------
type "%SUMMARY%"

echo.
echo Manual review
echo -------------
type "%ROOT%\manual_review.txt"

echo.
echo Generated PDFs
echo --------------
dir /b "%FIGURES%\*.pdf"

echo.
echo Method table: %CD%\%TABLES%\method_summary.csv
echo Figures:      %CD%\%FIGURES%
echo.

start "" explorer.exe "%CD%\%FIGURES%"

endlocal
exit /b 0
