"""按分类批量抓取 MCMod 页面。

分类编号（来自 MCMod 物品列表页）：
  1  物品/方块      561
  4  生物/实体      165
  6  BUFF/DEBUFF     76
  7  多方块结构      15
  10 游戏设定        18

默认抓 4,7,10 —— 这三类全是「机制」「生物」「仪式结构」，
对答疑的边际价值远高于装饰方块。

用法：
  python scripts/fetch_batch.py --cats 4,7,10
  python scripts/fetch_batch.py --cats 1 --limit 50 --delay 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_mcmod import OUT, fetch_one  # noqa: E402

INDEX = ROOT / "kb" / "item_index.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="4,7,10")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not INDEX.exists():
        print(f"缺少 {INDEX.name}，先跑列表抓取生成索引")
        return 1

    idx = json.loads(INDEX.read_text("utf-8"))
    ids: list[str] = []
    for c in args.cats.split(","):
        rows = idx.get(c.strip(), [])
        print(f"  分类 {c.strip():<3} {len(rows):>4} 条")
        ids.extend(r["mcmod_id"] for r in rows)

    seen: set[str] = set()
    uniq = [i for i in ids if not (i in seen or seen.add(i))]
    if args.limit:
        uniq = uniq[: args.limit]

    todo = [i for i in uniq if not (OUT / f"{i}.html").exists()]
    print(f"\n合计 {len(uniq)} 个，已有 {len(uniq) - len(todo)}，待抓 {len(todo)}")
    print(f"预计耗时 {len(todo) * (args.delay + 0.4) / 60:.1f} 分钟\n")

    ok = 0
    for n, mid in enumerate(todo, 1):
        if fetch_one(mid, args.delay):
            ok += 1
        if n % 25 == 0:
            print(f"  ---- 进度 {n}/{len(todo)} ----", flush=True)

    print(f"\n完成 {ok}/{len(todo)}")
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
