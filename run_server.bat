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
cd /d "%ROOT%"

echo.
echo   GOETY CODEX
echo   界面: http://127.0.0.1:8077
echo   关闭本窗口即停止服务
echo.

start "" http://127.0.0.1:8077
python -m uvicorn server:app --host 127.0.0.1 --port 8077
if errorlevel 1 pause
