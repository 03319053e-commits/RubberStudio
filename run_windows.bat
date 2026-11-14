@echo off
setlocal enableextensions
cd /d "%~dp0"
title RUBBER STUDIO

if exist ".venv\Scripts\python.exe" goto launch

set "PY="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V --version >nul 2>nul
    if not errorlevel 1 set "PY=py -%%V"
  )
)
if not defined PY (
  python --version >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY goto nopython

echo First start: creating a virtual environment and installing the dependencies...
%PY% -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

:launch
".venv\Scripts\python.exe" -m rubberstudio %*
exit /b %errorlevel%

:nopython
echo Python 3.10 or newer was not found.
echo Install Python 3.12 from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
echo then run this file again.
pause
exit /b 1

:fail
echo.
echo *** Setup failed - see the messages above. ***
rmdir /s /q ".venv" 2>nul
pause
exit /b 1
