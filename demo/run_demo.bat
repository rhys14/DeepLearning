@echo off
REM Double-click to start the Galaxy10 model comparison demo, then it opens in your browser.
cd /d "%~dp0.."
python demo\app.py %*
pause
