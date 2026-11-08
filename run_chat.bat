@echo off
setlocal
chcp 65001 >nul
set ROOT=%~dp0
set PYTHONPATH=%ROOT%.pylib;%ROOT%scripts
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%ROOT%.cache\hf
set HF_HUB_DISABLE_XET=1
set TEMP=%ROOT%.tmp
set TMP=%ROOT%.tmp
set PYTHONIOENCODING=utf-8

python "%ROOT%scripts\chat.py" %*
if errorlevel 1 pause
