@echo off
rem Runs only the GenAI step (Groq). Output: llm_log.txt and llm_demo_output.txt
cd /d "%~dp0"
set PY=%USERPROFILE%\anaconda3\envs\MGT4170\python.exe
"%PY%" llm_inputs.py > llm_log.txt 2>&1
echo DONE >> llm_log.txt
