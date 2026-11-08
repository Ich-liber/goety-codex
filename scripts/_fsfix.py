"""文件系统兼容补丁：修复 tempfile 在 Windows 沙箱下的权限继承问题。

问题
----
`tempfile.mkdtemp()` 内部是 `os.mkdir(name, 0o700)`。当前 Python 在 Windows 上
会把 `mode` 转成显式权限项，而显式权限**不继承父目录**，于是新建的临时目录
丢失了工作区授予的写权限。任何「先 mkdtemp 再写文件」的库都会中招：
  - pip 下载 wheel        （见 _pip.py）
  - huggingface_hub 下载模型（本项目下载 embedding 模型时踩到）

表现都是 PermissionError / WinError 5，且路径指向 `tmpXXXX` 目录。

做法
----
导入本模块即生效：拦截 os.mkdir，丢弃 mode，恢复默认继承行为。
必须在使用相关库**之前** import。

用法：
  import _fsfix  # noqa: F401  —— 放在其他第三方 import 之前
"""

from __future__ import annotations

import os

_orig_mkdir = os.mkdir


def _mkdir_inherit(path, mode=0o777, *, dir_fd=None):
    if dir_fd is not None:
        return _orig_mkdir(path, mode, dir_fd=dir_fd)
    return _orig_mkdir(path)


os.mkdir = _mkdir_inherit
