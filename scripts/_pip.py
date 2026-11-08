"""pip 兼容启动器：绕开 Windows 沙箱下的 tempfile 权限问题。

背景
----
Python 3.12+ 在 Windows 上开始把 os.mkdir 的 mode 参数转成显式权限项。
tempfile.mkdtemp() 用 os.mkdir(name, 0o700) 创建临时目录，生成的目录
**不继承父目录的权限**，因此丢失了沙箱工作区授予的写权限。

pip 下载 wheel 时会先 mkdtemp 建临时目录，再在其中 open(path, "wb")，
于是直接 PermissionError: [Errno 13]。

做法
----
拦截 os.mkdir，丢弃 mode 参数，让新目录走默认创建流程、正常继承父目录权限。
临时目录因此变为可写，pip 恢复正常。

用法
----
  python scripts/_pip.py install --target .pylib beautifulsoup4
  python scripts/_pip.py download -d .tmp/dl --no-deps jieba
"""

from __future__ import annotations

import os
import runpy
import sys

_orig_mkdir = os.mkdir


def _mkdir_inherit(path, mode=0o777, *, dir_fd=None):
    """与 os.mkdir 同签名，但忽略 mode，保留父目录权限继承。"""
    if dir_fd is not None:
        return _orig_mkdir(path, mode, dir_fd=dir_fd)
    return _orig_mkdir(path)


os.mkdir = _mkdir_inherit

if __name__ == "__main__":
    sys.argv = ["pip", *sys.argv[1:]]
    runpy.run_module("pip", run_name="__main__")
