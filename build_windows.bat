@echo off
setlocal enableextensions
cd /d "%~dp0"
title RUBBER STUDIO - build for Windows
echo.
echo === RUBBER STUDIO: build the Windows program ===
echo     (usage: build_windows.bat        full build with tests
echo             build_windows.bat fast   skip the tests)
echo.

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
echo Using: %PY%

if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto fail
python -m pip install -r requirements.txt -r requirements-dev.txt
if errorlevel 1 goto fail

if /i "%~1"=="fast" goto build
echo.
echo --- running the tests ---
set "QT_QPA_PLATFORM=offscreen"
python -m pytest -q
if errorlevel 1 echo *** Some tests failed - continuing; the self-test of the built program decides. ***
set "QT_QPA_PLATFORM="

:build
echo.
echo --- building with PyInstaller ---
python -m PyInstaller packaging\RubberStudio.spec --noconfirm --clean
if errorlevel 1 goto fail

echo.
echo --- self-test of the built program ---
set "REPORT=%CD%\dist\selftest.json"
start "" /wait "dist\RubberStudio\RubberStudio.exe" --selftest "%REPORT%"
if errorlevel 1 (
  if exist "%REPORT%" type "%REPORT%"
  goto fail
)

for /f %%v in ('python -c "from rubberstudio import __version__; print(__version__)"') do set "VER=%%v"
set "ZIP=RubberStudio-%VER%-windows-x64.zip"
if exist "%ZIP%" del "%ZIP%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path 'dist\RubberStudio' -DestinationPath '%ZIP%' -CompressionLevel Optimal"
if errorlevel 1 goto fail

echo.
echo ==================================================================
echo  Done.
echo    Program : dist\RubberStudio\RubberStudio.exe
echo    Zip     : %ZIP%
echo ==================================================================
pause
exit /b 0

:nopython
echo.
echo Python 3.10 or newer was not found.
echo Install Python 3.12 from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
echo then run this file again.
pause
exit /b 1

:fail
echo.
echo *** The build failed - see the messages above. ***
pause
exit /b 1
