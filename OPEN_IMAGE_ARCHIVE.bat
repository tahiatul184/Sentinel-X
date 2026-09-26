@echo off
if not exist "%~dp0app\data\imagery" mkdir "%~dp0app\data\imagery"
start "" "%~dp0app\data\imagery"
