"""探索 MCMod 物品页的 HTML 结构，确定解析锚点。

只打印结构，不产出数据，用于确认选择器。

用法：
  python scripts/explore_mcmod.py --file raw/mcmod/935700.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bs4 import BeautifulSoup

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="raw/mcmod/935700.html")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / path
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    print(f"文件: {path.name}  {len(html):,} 字符\n")

    print("=" * 62)
    print("标题")
    print("=" * 62)
    print("  <title>:", (soup.title.string or "").strip()[:100])

    print("\n" + "=" * 62)
    print("各级标题（h1-h5）")
    print("=" * 62)
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        text = tag.get_text(strip=True)
        if text:
            classes = " ".join(tag.get("class") or [])
            print(f"  <{tag.name} class='{classes}'> {text[:60]}")

    print("\n" + "=" * 62)
    print("关键字段所在元素")
    print("=" * 62)
    for kw in ("资料分类", "物品命令", "矿物词典", "最大堆叠", "主要名称", "次要名称"):
        hits = soup.find_all(string=lambda s: s and kw in s)
        print(f"\n  [{kw}] 命中 {len(hits)} 处")
        for node in hits[:2]:
            parent = node.parent
            print(f"    直接父标签 : <{parent.name} class='{' '.join(parent.get('class') or [])}'>")
            gp = parent.parent
            print(f"    祖父标签   : <{gp.name} class='{' '.join(gp.get('class') or [])}>")
            row = parent.find_parent("tr")
            if row is not None:
                print(f"    所在 tr    : {row.get_text(' | ', strip=True)[:110]}")

    print("\n" + "=" * 62)
    print("正文候选容器（含较长纯文本的 div）")
    print("=" * 62)
    seen: dict[str, int] = {}
    for div in soup.find_all("div"):
        text = div.get_text(" ", strip=True)
        if 150 < len(text) < 4000:
            classes = " ".join(div.get("class") or [])
            if classes and classes not in seen:
                seen[classes] = len(text)
    for cls, size in sorted(seen.items(), key=lambda x: -x[1])[:12]:
        print(f"  {size:>6} 字  class='{cls}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
