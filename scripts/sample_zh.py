"""抽样检查 jar 内中文语料的真实质量。

回答三个问题：
  1. lang/zh_cn.json 有多少条目、按类型如何分布
  2. Patchouli zh_cn 条目是否真的翻译了（而不是复制英文）
  3. 翻译文本里有没有明显的英文 / 富文本残留

用法：
  python scripts/sample_zh.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r"^assets/goety/patchouli_books/([^/]+)/zh_cn/entries/(.+\.json)$")


def find_jar() -> Path:
    jars = sorted((ROOT / "raw").glob("*.jar"))
    if not jars:
        raise SystemExit("raw/ 下没有 jar，请先运行 scripts/fetch_mod.py")
    return jars[-1]


def pct_cjk(text: str) -> float:
    """中日韩统一表意文字占比，用来判断是否真翻译。"""
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(text)


def strip_rt(text: str) -> str:
    """去掉 Patchouli 富文本标记，便于阅读和统计。"""
    text = re.sub(r"\$\([^)]*\)", "", text)   # $(l:intro/x) $(l) $(br) ...
    text = text.replace("$()", "")
    return re.sub(r"\s+", " ", text).strip()


def show_lang(zf: zipfile.ZipFile) -> None:
    data = json.loads(zf.read("assets/goety/lang/zh_cn.json").decode("utf-8"))
    print("=" * 62)
    print(f"lang/zh_cn.json   共 {len(data):,} 个键")
    print("=" * 62)
    kinds: Counter = Counter(k.split(".")[0] for k in data)
    for k, v in kinds.most_common():
        print(f"  {k:<24} {v:>5}")

    samples = [k for k in data if k.startswith("item.goety.dark")]
    print("\n  样例（item.goety.dark*）：")
    for k in samples[:6]:
        print(f"    {k:<46} = {data[k][:40]}")

    # 未翻译检测：值是纯 ASCII 且看起来像英文句子
    untranslated = [
        k for k, v in data.items()
        if isinstance(v, str) and len(v) > 8
        and pct_cjk(v) == 0
        and re.search(r"[a-zA-Z]{4,}\s+[a-zA-Z]{3,}", v)
    ]
    print(f"\n  疑似未翻译（中文文件里仍为英文）: {len(untranslated)} 条")
    for k in untranslated[:5]:
        print(f"    {k} = {data[k][:50]}")
    print()


def show_entries(zf: zipfile.ZipFile) -> None:
    names = zf.namelist()
    entries = sorted(n for n in names if PAT.match(n))
    print("=" * 62)
    print(f"Patchouli zh_cn 条目共 {len(entries)} 条，抽取样例")
    print("=" * 62)

    # 每个顶级分类各取一条，最多 4 条
    picked: dict[str, str] = {}
    for n in entries:
        top = PAT.match(n).group(2).split("/")[0]
        if top not in picked:
            picked[top] = n
        if len(picked) >= 4:
            break

    for n in picked.values():
        raw = json.loads(zf.read(n).decode("utf-8"))
        top = PAT.match(n).group(2)
        print(f"\n--- [{top}] {raw.get('name', '(无 name)')} ---")
        print(f"    category : {raw.get('category')}")
        texts = []
        for page in raw.get("pages", []):
            if page.get("type", "").endswith(":text") and page.get("text"):
                texts.append(strip_rt(page["text"]))
        joined = " ".join(texts)
        print(f"    正文     : {joined[:220]}{'...' if len(joined) > 220 else ''}")
        print(f"    汉字占比 : {pct_cjk(joined):.0%}   段落数: {len(texts)}")
        kinds = Counter(p.get("type", "?").split(":")[-1] for p in raw.get("pages", []))
        print(f"    页面类型 : {dict(kinds)}")
    print()


def main() -> int:
    jar = find_jar()
    with zipfile.ZipFile(jar) as zf:
        show_lang(zf)
        show_entries(zf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
