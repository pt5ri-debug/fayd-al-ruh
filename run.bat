@echo off
rem Portable launcher for Fayd Al-Ruh (Windows).
rem
rem Prefers a frozen executable sitting next to this script (a built bundle);
rem otherwise falls back to the source checkout via "python -m cli".
rem
rem When double-clicked with no arguments the window stays open so any error
rem is readable. Pass a command (e.g. "run.bat demo") for one-shot use in a
rem shell, where the window closing immediately is expected.
setlocal

set "HERE=%~dp0"

if exist "%HERE%fayd-al-ruh.exe" goto :frozen

where py >nul 2>nul
if not errorlevel 1 goto :source_py

where python >nul 2>nul
if not errorlevel 1 goto :source_python

echo Python was not found on PATH.
echo Install Python 3.12 or newer, or use the packaged build ^(fayd-al-ruh.exe^).
set "CODE=1"
goto :finish

:frozen
"%HERE%fayd-al-ruh.exe" %*
set "CODE=%ERRORLEVEL%"
goto :finish

:source_py
py -3 -m cli %*
set "CODE=%ERRORLEVEL%"
goto :finish

:source_python
python -m cli %*
set "CODE=%ERRORLEVEL%"

:finish
if "%~1"=="" (
    echo.
    echo Done ^(exit code %CODE%^). Press any key to close.
    pause >nul
)

endlocal & exit /b %CODE%
