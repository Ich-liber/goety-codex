"""BM25 检索 + 消融对比。

中文用字符 bigram（不依赖分词器，避免把「黑暗祭坛」切成「黑暗/祭坛」），
英文用长度 >= 2 的小写单词。

两种模式：
  raw     —— 原查询直接检索（基线）
  rewrite —— 先做查询理解（清洗 + 实体 + 症状扩展）再检索

用法：
  python scripts/retrieve_bm25.py --q "灵魂能量怎么获得"
  python scripts/retrieve_bm25.py --compare
  python scripts/retrieve_bm25.py --eval --mode rewrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "kb" / "chunks.jsonl"
sys.path.insert(0, str(ROOT / "scripts"))

RE_ZH = re.compile(r"[\u4e00-\u9fff]+")
RE_EN = re.compile(r"[a-z]{2,}")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for seg in RE_ZH.findall(text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    tokens.extend(RE_EN.findall(text))
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        if not q:
            return 0.0
        tf, dl = self.tf[idx], len(self.docs[idx])
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f or t not in self.idf:
                continue
            s += self.idf[t] * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.n)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def search_with_entities(bm25: "BM25", chunks: list[dict], query: str,
                         entities: list[str], k: int = 5):
    """识别到实体时，先把候选限定在该实体的 chunk 里再排序。

    没有这层过滤时，查询「聚晶怎么装上」会被「黑暗魔杖 > 概述」抢走——
    因为讲施法的那段通用文本和「装上」这类词更贴近，而它跟聚晶毫无关系。
    候选太少或全部零分时回退到全局检索，避免过滤过狠。
    """
    if entities:
        cand = [
            i for i, c in enumerate(chunks)
            if any(e in c["title"] for e in entities)
        ]
        if len(cand) >= 2:
            scored = [(i, bm25.score(query, i)) for i in cand]
            scored = [x for x in scored if x[1] > 0]
            if scored:
                scored.sort(key=lambda x: -x[1])
                return scored[:k]
    return bm25.search(query, k)


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS.read_text("utf-8").splitlines() if l.strip()]


# (查询, 类型, ground truth —— 命中条件为 breadcrumb 包含该子串)
EVAL: list[tuple[str, str, str]] = [
    ("灵魂能量怎么获得", "概念", "灵魂能量 > 玩家的灵魂能量槽"),
    ("黑暗祭坛怎么激活", "操作", "黑暗祭坛 > 激活"),
    ("我的仪式放上祭品却没有反应", "排错", "黑暗祭坛 > 合成物品"),
    ("聚晶有什么用", "概念", "空法术聚晶"),
    ("红石傀儡怎么召唤", "多跳", "红石傀儡 > 相关进度表"),
    ("巫妖有什么好处", "概念", "巫妖 > 特点"),
]


def hits_at(chunks: list[dict], results, gold: str, k: int) -> int | None:
    for rank, (i, _) in enumerate(results[:k], 1):
        if gold in chunks[i]["breadcrumb"]:
            return rank
    return None


def run_mode(bm25: BM25, chunks, mode: str, k: int):
    from query_rewrite import analyze
    rows = []
    for q, kind, gold in EVAL:
        query = analyze(q)["query"] if mode == "rewrite" else q
        res = bm25.search(query, k)
        rows.append({
            "q": q, "kind": kind, "gold": gold,
            "query_used": query,
            "rank": hits_at(chunks, res, gold, k),
            "top1": chunks[res[0][0]]["breadcrumb"] if res else "—",
        })
    return rows


def build_docs(chunks: list[dict], title_boost: int = 2) -> list[str]:
    """索引文本 = 正文 + 标题/小节名加权重复。

    标题信号原本完全没进索引，导致「巫妖」这种一词多小节的实体区分不开：
    问「巫妖有什么好处」时，『巫妖 > 特点』和『巫妖 > 巫妖之吻』拿到同样的分。
    """
    docs = []
    for c in chunks:
        head = f"{c['title']} {c['heading']}"
        docs.append(c["text"] + " " + head * title_boost)
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["raw", "rewrite"], default="raw")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--no-title-boost", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    boost = 0 if args.no_title_boost else 2
    bm25 = BM25(build_docs(chunks, boost))

    if args.compare:
        raw = run_mode(bm25, chunks, "raw", args.k)
        rw = run_mode(bm25, chunks, "rewrite", args.k)
        print(f"chunk={len(chunks)}  k={args.k}\n")
        print(f"{'查询':<26}{'类型':<6}{'raw':<8}{'rewrite':<9}{'改写后查询'}")
        print("-" * 92)
        for a, b in zip(raw, rw):
            def fmt(r):
                return f"#{r}" if r else "✗"
            print(f"{a['q']:<26}{a['kind']:<6}{fmt(a['rank']):<8}{fmt(b['rank']):<9}{b['query_used'][:34]}")
        print("-" * 92)
        n = len(EVAL)
        r1 = sum(1 for r in raw if r["rank"] == 1)
        w1 = sum(1 for r in rw if r["rank"] == 1)
        rk = sum(1 for r in raw if r["rank"] and r["rank"] <= args.k)
        wk = sum(1 for r in rw if r["rank"] and r["rank"] <= args.k)
        print(f"top1 命中 : raw {r1}/{n}   rewrite {w1}/{n}")
        print(f"top{args.k} 命中  : raw {rk}/{n}   rewrite {wk}/{n}")
        return 0

    if args.eval:
        rows = run_mode(bm25, chunks, args.mode, args.k)
        print(f"mode={args.mode}  chunk={len(chunks)}\n")
        for r in rows:
            mark = f"#{r['rank']}" if r["rank"] else "✗"
            print(f"[{r['kind']:<5}] {r['q']:<24} {mark:<4} {r['top1']}")
        ok = sum(1 for r in rows if r["rank"] and r["rank"] <= args.k)
        print(f"\n命中 {ok}/{len(EVAL)}")
        return 0

    if not args.q:
        ap.error("需要 --q / --eval / --compare")
    from query_rewrite import analyze
    info = analyze(args.q) if args.mode == "rewrite" else {"query": args.q, "intent": "-"}
    print(f"查询: {info['query']}   (意图 {info['intent']})\n")
    for rank, (i, s) in enumerate(bm25.search(info["query"], args.k), 1):
        c = chunks[i]
        print(f"{rank}. [{s:6.2f}] {c['breadcrumb']}")
        print(f"    {c['text'][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
