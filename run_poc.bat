@echo off
rem Problem 4 proof of concept: runs the model, the checks and the GenAI step in the MGT4170 conda env. Output: poc_log.txt
cd /d "%~dp0"
set PY=%USERPROFILE%\anaconda3\envs\MGT4170\python.exe
if not exist "%PY%" set PY=%USERPROFILE%\anaconda3\python.exe
echo START %DATE% %TIME% > poc_log.txt
for %%S in (simulate verify llm_inputs) do (
  echo ===== %%S >> poc_log.txt
  "%PY%" %%S.py >> poc_log.txt 2>&1
  call echo exit code %ERRORLEVEL% >> poc_log.txt
)
echo DONE %DATE% %TIME% >> poc_log.txt
