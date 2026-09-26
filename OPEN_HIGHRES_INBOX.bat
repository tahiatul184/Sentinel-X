@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0app\highres_inbox" mkdir "%~dp0app\highres_inbox"
start "" "%~dp0app\highres_inbox"
endlocal
