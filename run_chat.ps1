# 启动交互式问答
# 用法：右键「使用 PowerShell 运行」，或在终端执行 .\run_chat.ps1
# 若被执行策略拦截：powershell -ExecutionPolicy Bypass -File .\run_chat.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$root\.pylib;$root\scripts"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "$root\.cache\hf"
$env:HF_HUB_DISABLE_XET = "1"
$env:TEMP = "$root\.tmp"
$env:TMP = "$root\.tmp"
$env:PYTHONIOENCODING = "utf-8"

python "$root\scripts\chat.py" @args
