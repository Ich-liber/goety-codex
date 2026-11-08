"""对比 Patchouli 中英文条目的 name 字段，找翻译异常。

用途：
  - 找出「中文文件名未翻译」的条目
  - 找出「name 与英文原意不符 / 疑似错位」的条目
  - 为后续数据质量分级提供依据

用法：
  python scripts/check_names.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r"^assets/goety/patchouli_books/([^/]+)/(en_us|zh_cn)/entries/(.+\.json)$")


def find_jar() -> Path:
    jars = sorted((ROOT / "raw").glob("*.jar"))
    if not jars:
        raise SystemExit("raw/ 下没有 jar，请先运行 scripts/fetch_mod.py")
    return jars[-1]


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff") / len(text)


def main() -> int:
    jar = find_jar()
    with zipfile.ZipFile(jar) as zf:
        names = zf.namelist()
        index: dict[tuple[str, str], str] = {}
        for n in names:
            m = PAT.match(n)
            if m:
                index[(m.group(1), m.group(3))] = n

        # 按 (手册, 相对路径) 配对同一份文档的中英文版本
        seen: dict[tuple[str, str], dict[str, str]] = {}
        for n in names:
            m = PAT.match(n)
            if not m:
                continue
            book, lang, rel = m.groups()
            key = (book, rel)
            try:
                obj = json.loads(zf.read(n).decode("utf-8"))
            except json.JSONDecodeError:
                continue
            seen.setdefault(key, {})[lang] = obj.get("name") or ""

        total = 0
        same = []
        low_cjk = []
        for (book, rel), langs in sorted(seen.items()):
            if "zh_cn" not in langs or "en_us" not in langs:
                continue
            total += 1
            zh, en = langs["zh_cn"], langs["en_us"]
            if zh == en and en:
                same.append((book, rel, en))
            elif en and cjk_ratio(zh) < 0.15:
                low_cjk.append((book, rel, en, zh))

        print("=" * 62)
        print(f"可配对的条目: {total}")
        print(f"  中文名与英文名完全相同（疑似未翻译）: {len(same)}")
        print(f"  中文名汉字占比 < 15%（疑似非中文）  : {len(low_cjk)}")
        print("=" * 62)

        if same:
            print("\n[未翻译清单]")
            for book, rel, en in same[:25]:
                print(f"  {rel:<44} en={en[:40]}")
            if len(same) > 25:
                print(f"  ... 另有 {len(same) - 25} 条")

        if low_cjk:
            print("\n[非中文清单]")
            for book, rel, en, zh in low_cjk[:25]:
                print(f"  {rel:<44} zh={zh[:30]!r}  en={en[:30]!r}")

        # 重点人工核对：intro/cursed_cage
        print("\n[人工核对] intro/cursed_cage")
        for (book, rel), langs in seen.items():
            if rel == "intro/cursed_cage.json":
                for lang, name in langs.items():
                    print(f"  {lang}: {name!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
