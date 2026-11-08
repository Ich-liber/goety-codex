"""混合检索：BM25 / 向量 / RRF 融合。

三种引擎共用同一份 chunk 与同一条查询改写管线，便于把收益归因到「引擎」而不是
「顺带改了别的」。RRF（Reciprocal Rank Fusion）只依赖排名、不依赖分数尺度，
因为 BM25 的分数量级和余弦相似度不可比。

用法：
  python scripts/retrieve_hybrid.py --q "傀儡召不出来" --engine rrf
  python scripts/retrieve_hybrid.py --eval --engine all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _fsfix  # noqa: E402,F401  必须在 fastembed / huggingface_hub 之前导入

from retrieve_bm25 import BM25, build_docs, load_chunks  # noqa: E402

VECTORS = ROOT / "kb" / "vectors.npy"
VMETA = ROOT / "kb" / "vectors.json"
CACHE = ROOT / ".model_cache"
EVAL = ROOT / "eval" / "qa_set.jsonl"

_model = None
_vecs = None
_reranker = None
RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"


def get_model():
    global _model
    if _model is None:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from fastembed import TextEmbedding
        meta = json.loads(VMETA.read_text("utf-8"))
        _model = TextEmbedding(model_name=meta["model"], cache_dir=str(CACHE))
    return _model


def get_vectors():
    global _vecs
    if _vecs is None:
        import numpy as np
        _vecs = np.load(VECTORS)
    return _vecs


def get_reranker():
    """cross-encoder 重排器。首次调用会下载模型（int8 量化版约 0.57 GB）。"""
    global _reranker
    if _reranker is None:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=str(CACHE))
    return _reranker


def vector_search(query: str, k: int = 5) -> list[tuple[int, float]]:
    import numpy as np
    vecs = get_vectors()
    q = np.array(list(get_model().query_embed([query]))[0], dtype=np.float32)
    n = np.linalg.norm(q)
    if n:
        q = q / n
    scores = vecs @ q
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def rrf_fuse(lists: list[list[tuple[int, float]]], k: int = 60, top: int = 5) -> list[tuple[int, float]]:
    """score = Σ 1/(k + rank)，只看排名不看原始分数。"""
    acc: dict[int, float] = {}
    for res in lists:
        for rank, (idx, _) in enumerate(res, 1):
            acc[idx] = acc.get(idx, 0.0) + 1.0 / (k + rank)
    ranked = sorted(acc.items(), key=lambda x: -x[1])[:top]
    return [(i, s) for i, s in ranked]


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25(build_docs(chunks, 0))  # 标题加权已验证无效

    def search(self, query: str, engine: str, k: int = 5,
               entities: list[str] | None = None) -> list[tuple[int, float]]:
        entities = entities or []
        # BM25 走实体过滤（词面匹配容易跑到别的主题上）；向量不需要，它自带语义
        bm = self._bm25_filtered(query, entities, k * 4)
        if engine == "bm25":
            return bm[:k]
        if engine == "vector":
            return vector_search(query, k)
        if engine == "rrf":
            return rrf_fuse([bm, vector_search(query, k * 4)], top=k)
        if engine == "rerank":
            return self._rerank(query, k, entities)
        raise ValueError(engine)

    def _rerank(self, query: str, k: int, entities: list[str]):
        """两阶段检索：RRF 召回较宽的候选，再用 cross-encoder 精排。

        语料从 134 涨到 975 之后出现「同实体多页面互相竞争」——比如
        「仆从模式怎么切」被「僵尸猪灵蛮兵仆从 > 行为模式」抢走。
        这类问题靠词面加权解决不了，需要模型真正读一遍 query-doc 对。

        性能：cross-encoder 的耗时与输入 token 数成正比。原先取 20 个候选
        且送全文，单次查询要 40 秒。现在候选降到 12 个、正文截断到 320 字，
        既保住判断所需的信息量，又把延迟压到可接受范围。
        模型不可用时回退到 RRF，保证服务不挂。
        """
        wide = max(k * 3, 12)
        bm = self._bm25_filtered(query, entities, wide)
        cand = rrf_fuse([bm, vector_search(query, wide)], top=wide)
        if len(cand) <= k:
            return cand
        try:
            encoder = get_reranker()
        except Exception:
            return cand[:k]
        idxs = [i for i, _ in cand]
        docs = [
            f"{self.chunks[i]['breadcrumb']}｜{self.chunks[i]['text'][:320]}"
            for i in idxs
        ]
        try:
            scores = list(encoder.rerank(query, docs))
        except Exception:
            return cand[:k]
        ranked = sorted(zip(idxs, scores), key=lambda x: -x[1])[:k]
        return [(i, float(s)) for i, s in ranked]

    def _bm25_filtered(self, query: str, entities: list[str], k: int):
        if entities:
            cand = [i for i, c in enumerate(self.chunks)
                    if any(e in c["title"] for e in entities)]
            if len(cand) >= 2:
                scored = [(i, self.bm25.score(query, i)) for i in cand]
                scored = [x for x in scored if x[1] > 0]
                if scored:
                    scored.sort(key=lambda x: -x[1])
                    return scored[:k]
        return self.bm25.search(query, k)


TYPE_ZH = {
    "concept": "概念", "howto": "操作", "troubleshoot": "排错",
    "variant": "变体", "progress": "进度", "refuse": "拒答",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q")
    ap.add_argument("--engine", choices=["bm25", "vector", "rrf", "rerank", "all"], default="rrf")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    if not VECTORS.exists():
        print("缺少向量索引，先运行: python scripts/embed_index.py")
        return 1

    from query_rewrite import analyze

    chunks = load_chunks()
    rt = Retriever(chunks)

    if args.q:
        info = analyze(args.q)
        print(f"查询: {info['query']}")
        print(f"实体: {info['entities']}   意图: {info['intent']}\n")
        for engine in (["bm25", "vector", "rrf", "rerank"] if args.engine == "all" else [args.engine]):
            print(f"--- {engine} ---")
            for rank, (i, s) in enumerate(rt.search(info["query"], engine, args.k, info["entities"]), 1):
                print(f"  {rank}. [{s:6.3f}] {chunks[i]['breadcrumb']}")
            print()
        return 0

    if not args.eval:
        ap.error("需要 --q 或 --eval")

    qa = [json.loads(l) for l in EVAL.read_text("utf-8").splitlines() if l.strip()]
    scored = [x for x in qa if x["gold"]]
    engines = ["bm25", "vector", "rrf", "rerank"] if args.engine == "all" else [args.engine]

    print(f"chunk={len(chunks)}  评测集={len(scored)} 条（不含拒答）  k={args.k}\n")
    header = f"{'类型':<8}" + "".join(f"{e:<14}" for e in engines)
    print(header)
    print("-" * len(header))

    stats = {e: {"h1": 0, "hk": 0, "rrf": 0.0} for e in engines}
    per_type = {e: {} for e in engines}

    for e in engines:
        for q in scored:
            info = analyze(q["q"])
            res = rt.search(info["query"], e, args.k, info["entities"])
            rank = None
            for r, (i, _) in enumerate(res, 1):
                if any(g in chunks[i]["breadcrumb"] for g in q["gold"]):
                    rank = r
                    break
            t = q["type"]
            per_type[e].setdefault(t, [0, 0])
            per_type[e][t][1] += 1
            if rank == 1:
                stats[e]["h1"] += 1
                per_type[e][t][0] += 1
            if rank and rank <= args.k:
                stats[e]["hk"] += 1
                stats[e]["rrf"] += 1 / rank
        stats[e]["rrf"] /= len(scored)

    for t in ["concept", "howto", "troubleshoot", "variant", "progress"]:
        cells = []
        for e in engines:
            hit, tot = per_type[e].get(t, [0, 0])
            cells.append(f"{hit}/{tot}" if tot else "—")
        print(f"{TYPE_ZH[t]:<8}" + "".join(f"{c:<14}" for c in cells))
    print("-" * len(header))
    row1 = "".join(f"{stats[e]['h1']}/{len(scored):<12}" for e in engines)
    rowk = "".join(f"{stats[e]['hk']}/{len(scored):<12}" for e in engines)
    rowm = "".join(f"{stats[e]['rrf']:<14.3f}" for e in engines)
    print(f"{'Hit@1':<8}{row1}")
    print(f"{'Hit@' + str(args.k):<8}{rowk}")
    print(f"{'MRR':<8}{rowm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
