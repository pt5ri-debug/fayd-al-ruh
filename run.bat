@echo off
rem Portable launcher for Fayd Al-Ruh (Windows).
rem
rem Prefers a frozen executable sitting next to this script (a built bundle);
rem otherwise falls back to running the source checkout.
setlocal

set "HERE=%~dp0"

if exist "%HERE%fayd-al-ruh.exe" (
    "%HERE%fayd-al-ruh.exe" %*
    exit /b %ERRORLEVEL%
)

python -m cli %*
exit /b %ERRORLEVEL%
