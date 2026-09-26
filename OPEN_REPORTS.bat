@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0app\data\reports\auto" mkdir "%~dp0app\data\reports\auto"
start "" "%~dp0app\data\reports\auto"
endlocal
