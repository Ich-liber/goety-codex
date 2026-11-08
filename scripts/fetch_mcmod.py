"""抓取 MCMod 物品页，落盘为 HTML 供解析。

礼貌性约束：
  - 单线程、固定间隔（默认 1.5s）
  - 自定义 UA 标识用途
  - 已存在文件默认跳过，支持重复运行

用法：
  python scripts/fetch_mcmod.py --ids 935700
  python scripts/fetch_mcmod.py --ids 935700,751555,753098
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "raw" / "mcmod"
BASE = "https://www.mcmod.cn/item/{}.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "goety-rag-research/0.1 (personal non-commercial study)"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def fetch_one(item_id: str, delay: float) -> bool:
    dst = OUT / f"{item_id}.html"
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"[skip] {item_id} 已存在 ({dst.stat().st_size:,} B)")
        return True

    url = BASE.format(item_id)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {item_id} 请求异常: {exc}")
        return False

    if r.status_code != 200:
        print(f"[FAIL] {item_id} HTTP {r.status_code}")
        return False

    r.encoding = "utf-8"
    dst.write_bytes(r.content)

    # 快速判断是否拿到真实正文
    has_title = "MC百科" in r.text
    has_body = "资料分类" in r.text or "物品命令" in r.text
    flag = "ok" if (has_title and has_body) else "suspect"
    print(f"[{flag:<7}] {item_id}  HTTP {r.status_code}  "
          f"{len(r.content):>8,} B  title={has_title} body={has_body}")
    time.sleep(delay)
    return has_title and has_body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="逗号分隔的 MCMod 物品 id")
    ap.add_argument("--delay", type=float, default=1.5, help="每次请求间隔秒数")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    print(f"目标 {len(ids)} 个页面，间隔 {args.delay}s\n")

    ok = sum(1 for i in ids if fetch_one(i, args.delay))
    print(f"\n完成: {ok}/{len(ids)} 成功")
    return 0 if ok == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
