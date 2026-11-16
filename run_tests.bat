@echo off
rem Runs the model checks (verify.py) and the AI accuracy tests (eval_ai.py); output saved to tests_log.txt
cd /d "%~dp0"
set PY=%USERPROFILE%\anaconda3\envs\MGT4170\python.exe
set PYTHONIOENCODING=utf-8
echo ===== verify.py ===== > tests_log.txt
"%PY%" verify.py >> tests_log.txt 2>&1
echo. >> tests_log.txt
echo ===== eval_ai.py ===== >> tests_log.txt
"%PY%" eval_ai.py >> tests_log.txt 2>&1
echo DONE >> tests_log.txt
