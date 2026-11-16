@echo off
rem Starts the Retirement Stress Test Shiny app at http://127.0.0.1:8001 in a minimized window (close that window to stop it)
cd /d "%~dp0"
rem stop any older copy of the app (only Python from the MGT4170 environment)
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*envs\MGT4170*' } | Stop-Process -Force"
start "Retirement app (leave open)" /min cmd /c ""%USERPROFILE%\anaconda3\envs\MGT4170\python.exe" -m shiny run app.py --port 8001 > app_log.txt 2>&1"
