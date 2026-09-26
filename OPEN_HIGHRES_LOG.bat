@echo off
if not exist "%~dp0app\data\highres" mkdir "%~dp0app\data\highres"
start "" "%~dp0app\data\highres"
