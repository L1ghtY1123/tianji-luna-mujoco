@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open-browser.ps1"
exit /b %ERRORLEVEL%
