"""检索评测：Hit@k / MRR，分类型对比 raw 与 rewrite。

评测集在 eval/qa_set.jsonl：
  gold 非空 -> 判定为「是否命中」；
  gold 为空 -> 拒答型，不参与命中率，只输出 top1 供人工判断是否该拒答。

用法：
  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py --mode rewrite
  python scripts/eval_retrieval.py --show-fail
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from query_rewrite import analyze          # noqa: E402
from retrieve_bm25 import BM25, build_docs, load_chunks, search_with_entities  # noqa: E402

QA = ROOT / "eval" / "qa_set.jsonl"
TYPE_ZH = {
    "concept": "概念", "howto": "操作", "troubleshoot": "排错",
    "variant": "变体", "progress": "进度", "refuse": "拒答",
}


def load_qa() -> list[dict]:
    return [json.loads(l) for l in QA.read_text("utf-8").splitlines() if l.strip()]


def hit_rank(chunks: list[dict], results, golds: list[str]) -> int | None:
    for rank, (i, _) in enumerate(results, 1):
        bc = chunks[i]["breadcrumb"]
        if any(g in bc for g in golds):
            return rank
    return None


def run(mode: str, chunks: list[dict], bm25: BM25, k: int) -> list[dict]:
    rows = []
    for qa in load_qa():
        info = analyze(qa["q"]) if mode == "rewrite" else {"query": qa["q"], "entities": []}
        res = search_with_entities(bm25, chunks, info["query"], info["entities"], k)
        rank = hit_rank(chunks, res, qa["gold"]) if qa["gold"] else None
        top = chunks[res[0][0]] if res else None
        rows.append({
            **qa, "query_used": info["query"], "rank": rank,
            "top1_bc": top["breadcrumb"] if top else "—",
            "top1_score": res[0][1] if res else 0.0,
        })
    return rows


def summarize(rows: list[dict], k: int) -> dict:
    scored = [r for r in rows if r["gold"]]
    n = len(scored)
    h1 = sum(1 for r in scored if r["rank"] == 1)
    hk = sum(1 for r in scored if r["rank"] and r["rank"] <= k)
    mrr = sum(1 / r["rank"] for r in scored if r["rank"]) / n if n else 0.0
    return {"n": n, "hit1": h1, "hitk": hk, "mrr": mrr}


def by_type(rows: list[dict], k: int) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["gold"]:
            groups[r["type"]].append(r)
    out = {}
    for t, items in groups.items():
        n = len(items)
        out[t] = {
            "n": n,
            "hit1": sum(1 for r in items if r["rank"] == 1),
            "hitk": sum(1 for r in items if r["rank"] and r["rank"] <= k),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "rewrite", "both"], default="both")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show-fail", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    bm25 = BM25(build_docs(chunks, 0))  # 标题加权已验证无效，固定关闭
    modes = ["raw", "rewrite"] if args.mode == "both" else [args.mode]

    results = {m: run(m, chunks, bm25, args.k) for m in modes}

    print(f"chunk={len(chunks)}  k={args.k}  评测集={len(load_qa())} 条\n")
    print(f"{'模式':<10}{'Hit@1':<12}{'Hit@'+str(args.k):<12}{'MRR':<8}")
    print("-" * 44)
    for m in modes:
        s = summarize(results[m], args.k)
        print(f"{m:<10}{s['hit1']}/{s['n']:<9}{s['hitk']}/{s['n']:<9}{s['mrr']:.3f}")
    print()

    print("分类型（Hit@1 / 条数）")
    print(f"{'类型':<8}" + "".join(f"{m:<12}" for m in modes))
    print("-" * 44)
    per = {m: by_type(results[m], args.k) for m in modes}
    for t in ["concept", "howto", "troubleshoot", "variant", "progress"]:
        cells = []
        for m in modes:
            v = per[m].get(t)
            cells.append(f"{v['hit1']}/{v['n']}" if v else "—")
        print(f"{TYPE_ZH[t]:<8}" + "".join(f"{c:<12}" for c in cells))
    print()

    # 拒答型：人工判断用
    print("拒答型（gold 为空，看 top1 分数是否异常高）")
    for r in results[modes[-1]]:
        if r["type"] == "refuse":
            flag = "⚠ 高分" if r["top1_score"] > 10 else "ok"
            print(f"  [{flag}] {r['q']:<26} {r['top1_score']:6.2f}  {r['top1_bc'][:40]}")

    if args.show_fail:
        for m in modes:
            print(f"\n===== {m} 未命中 =====")
            for r in results[m]:
                if r["gold"] and (r["rank"] is None or r["rank"] > args.k):
                    print(f"  {r['q']:<24} 期望={r['gold'][0]}")
                    print(f"      实际={r['top1_bc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
