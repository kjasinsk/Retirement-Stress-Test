@echo off
rem Double-click to save the app to https://github.com/kjasinsk/Retirement-Stress-Test
rem (Posit Connect Cloud republishes automatically after each push)
cd /d "%~dp0"
set GIT=git
where git >nul 2>&1 || set GIT="C:\Program Files\Git\cmd\git.exe"
set URL=https://github.com/kjasinsk/Retirement-Stress-Test.git
set LOG=%~dp0github_log.txt
echo ===== %date% %time% ===== > "%LOG%"
if not exist ".git" (
  %GIT% init >> "%LOG%" 2>&1
  %GIT% branch -M main >> "%LOG%" 2>&1
  %GIT% remote add origin %URL% >> "%LOG%" 2>&1
)
%GIT% config user.name "Kaitlyn Jasinski"
%GIT% config user.email "kjasinsk@villanova.edu"
%GIT% add . >> "%LOG%" 2>&1
%GIT% commit -m "Retirement Stress Test app" >> "%LOG%" 2>&1
echo --- push --- >> "%LOG%"
%GIT% push -u origin main >> "%LOG%" 2>&1 && goto done
echo --- GitHub already had files; merging them in (keeping ours if both changed), then pushing again --- >> "%LOG%"
%GIT% pull origin main --allow-unrelated-histories --no-edit -X ours >> "%LOG%" 2>&1
%GIT% push -u origin main >> "%LOG%" 2>&1
:done
echo --- files on GitHub --- >> "%LOG%"
%GIT% ls-tree --name-only origin/main >> "%LOG%" 2>&1
%GIT% status -sb >> "%LOG%" 2>&1
echo FINISHED >> "%LOG%"
