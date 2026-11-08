"""下载 Goety 1.20.1 模组 jar 到 raw/。

支持：
  - raw/ 下已有校验通过的同版本 jar 则跳过
  - 流式下载并显示进度
  - sha1 校验，防止下到半截文件

用法：
  python scripts/fetch_mod.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"

# Modrinth: project goety (4ZVIxU8x), version 2.5.60.6, MC 1.20.1, Forge/NeoForge
URL = "https://cdn.modrinth.com/data/4ZVIxU8x/versions/XIafvWQz/goety-2.5.60.6.jar"
FILENAME = "goety-2.5.60.6.jar"
SHA1 = "66c32ec52d06a64342f7ac09afe5b454b8db87af"
SIZE = 95_868_671


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_valid() -> Path | None:
    """raw/ 下任意 jar，只要大小和 sha1 对上就认为已就绪。"""
    for p in sorted(RAW.glob("*.jar")):
        if p.stat().st_size == SIZE and sha1_of(p) == SHA1:
            return p
    return None


def download() -> Path | None:
    RAW.mkdir(parents=True, exist_ok=True)

    ok = existing_valid()
    if ok is not None:
        print(f"[skip] 已存在且校验通过: {ok.name} "
              f"({ok.stat().st_size / 1048576:.1f} MB)")
        return ok

    target = RAW / FILENAME
    if target.exists():
        print(f"[warn] 同名文件校验不符，重新下载: {target.name}")

    print(f"[get ] {URL}")
    with requests.get(URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or SIZE)
        done = 0
        next_mark = 10 << 20
        with open(target, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if done >= next_mark:
                    print(f"  {done / 1048576:5.1f} / {total / 1048576:.1f} MB", flush=True)
                    next_mark += 10 << 20

    got = sha1_of(target)
    if got != SHA1:
        print(f"[FAIL] sha1 不匹配\n  expect {SHA1}\n  got    {got}")
        return None

    print(f"[ok  ] {target.name}  {target.stat().st_size / 1048576:.1f} MB  sha1 校验通过")
    return target


if __name__ == "__main__":
    sys.exit(0 if download() else 1)
