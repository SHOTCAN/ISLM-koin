@echo off
title ISLM Monitor - Standalone Bot
echo ================================================
echo  ISLM Monitor - Standalone Telegram AI Bot
echo  Ctrl+C to stop
echo ================================================
echo.

cd /d "%~dp0"
python bot_standalone.py

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Bot crashed. Restarting in 10 seconds...
    timeout /t 10 /nobreak
    goto :0
)

pause
