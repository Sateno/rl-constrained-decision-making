@echo off
python "%~dp0viewer.py" %*
if errorlevel 1 pause
