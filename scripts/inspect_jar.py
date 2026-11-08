"""盘点 Goety jar 内部可用的知识库素材，只读不写。

输出：版本信息、语言文件、Patchouli 手册条目分布、配方 / 进度 / 标签数量。

用法：
  python scripts/inspect_jar.py
"""

from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def find_jar() -> Path | None:
    jars = sorted((ROOT / "raw").glob("*.jar"))
    if not jars:
        print("raw/ 下没有找到 .jar 文件，请先运行 scripts/fetch_mod.py")
        return None
    if len(jars) > 1:
        print(f"[warn] raw/ 下有 {len(jars)} 个 jar，使用最后一个: {jars[-1].name}\n")
    return jars[-1]


def show_version(zf: zipfile.ZipFile) -> None:
    print("=" * 60)
    print("版本信息")
    print("=" * 60)
    for meta in ("META-INF/mods.toml", "META-INF/neoforge.mods.toml"):
        if meta not in zf.namelist():
            continue
        text = zf.read(meta).decode("utf-8", "replace")
        for line in text.splitlines():
            s = line.strip()
            if re.match(r"^(modId|version|displayName|description)\s*=", s):
                print("  " + s[:150])
            elif "versionRange" in s:
                print("    " + s[:150])
        break
    print()


def show_lang(zf: zipfile.ZipFile, names: list[str]) -> None:
    langs = [n for n in names if n.startswith("assets/goety/lang/") and n.endswith(".json")]
    print("=" * 60)
    print(f"语言文件  ({len(langs)} 个)")
    print("=" * 60)
    for n in sorted(langs):
        info = zf.getinfo(n)
        mark = "  <- 主力" if n.endswith("zh_cn.json") else ""
        print(f"  {n.split('/')[-1]:<12} {info.file_size:>9,} B{mark}")
    print()


def show_patchouli(zf: zipfile.ZipFile, names: list[str]) -> None:
    pat = re.compile(r"^assets/goety/patchouli_books/([^/]+)/([^/]+)/entries/(.+\.json)$")
    per_book: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for n in names:
        m = pat.match(n)
        if m:
            book, lang, rest = m.groups()
            top = rest.split("/")[0] if "/" in rest else "(根目录)"
            per_book[(book, lang)][top] += 1

    print("=" * 60)
    print("Patchouli 官方手册条目")
    print("=" * 60)
    if not per_book:
        print("  (未发现条目)")
    for (book, lang), counter in sorted(per_book.items()):
        total = sum(counter.values())
        mark = "  <- 主力" if lang == "zh_cn" else ""
        print(f"  {book}/{lang}: {total} 条{mark}")
        for top, cnt in counter.most_common():
            print(f"      {top:<12} {cnt:>4}")
    print()


def count_pattern(names: list[str], pattern: str) -> int:
    rx = re.compile(pattern)
    return sum(1 for n in names if rx.match(n))


def show_data(zf: zipfile.ZipFile, names: list[str]) -> None:
    print("=" * 60)
    print("数据文件")
    print("=" * 60)
    for label, pattern in (
        ("recipes（配方）", r"^data/goety/recipes/.+\.json$"),
        ("advancements（进度）", r"^data/goety/advancements/.+\.json$"),
        ("tags（标签）", r"^data/goety/tags/.+\.json$"),
        ("soul_taken（灵魂值）", r"^data/goety/soul_taken/.+\.json$"),
        ("loot_tables（掉落）", r"^data/goety/loot_tables/.+\.json$"),
        ("structures（结构）", r"^data/goety/structures/.+$"),
    ):
        print(f"  {label:<28} {count_pattern(names, pattern):>5}")
    print()

    others: Counter = Counter()
    for n in names:
        m = re.match(r"^data/goety/([^/]+)/", n)
        if m:
            others[m.group(1)] += 1
    print("  data/goety 各子目录文件数：")
    for k, v in others.most_common():
        print(f"    {k:<16} {v}")
    print()


def main() -> int:
    jar = find_jar()
    if jar is None:
        return 1

    print(f"JAR: {jar.name}   {jar.stat().st_size / 1048576:.1f} MB\n")
    with zipfile.ZipFile(jar) as zf:
        names = zf.namelist()
        print(f"zip 条目总数: {len(names):,}\n")
        show_version(zf)
        show_lang(zf, names)
        show_patchouli(zf, names)
        show_data(zf, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
