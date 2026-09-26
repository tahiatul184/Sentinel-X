@echo off
setlocal
cd /d "%~dp0"
title AEROSENTINEL Setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\setup_windows.ps1"
if errorlevel 1 pause
endlocal
