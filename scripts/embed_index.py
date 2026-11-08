"""用 ONNX 模型给 chunks 建向量索引。

选型说明：本机无 GPU、也没有可用的 embedding API，装 torch 要拉 2.4 GB 的
CUDA wheel（按实测带宽约半小时），所以改用 fastembed（ONNX Runtime），
模型体积差两个数量级，效果对本任务够用。

用法：
  python scripts/embed_index.py
  python scripts/embed_index.py --model BAAI/bge-small-zh-v1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fsfix  # noqa: E402,F401  必须在 fastembed / huggingface_hub 之前导入

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "kb" / "chunks.jsonl"
OUT_VEC = ROOT / "kb" / "vectors.npy"
OUT_META = ROOT / "kb" / "vectors.json"
CACHE = ROOT / ".model_cache"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in CHUNKS.read_text("utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    try:
        import numpy as np
        from fastembed import TextEmbedding
    except ImportError as exc:
        print(f"缺少依赖: {exc}\n先运行: python scripts/_pip.py install --target .pylib fastembed")
        return 1

    chunks = load_chunks()
    # 索引文本 = 正文 + 小节名。标题（物品名）对所有小节相同，加进来没有区分度，
    # 这一点在 BM25 上已经验证过，这里沿用同样的结论。
    texts = [f"{c['heading']}｜{c['text']}" for c in chunks]

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"模型: {args.model}")
    print(f"缓存: {CACHE}")
    print(f"文档: {len(texts)} 条\n")

    t0 = time.time()
    try:
        model = TextEmbedding(model_name=args.model, cache_dir=str(CACHE))
    except Exception as exc:  # noqa: BLE001
        print(f"加载模型失败: {exc}")
        try:
            print("可用模型:", [m["model"] for m in TextEmbedding.list_supported_models()][:20])
        except Exception:
            pass
        return 1

    vecs = np.array(list(model.embed(texts, batch_size=16)), dtype=np.float32)
    # 归一化后内积即余弦相似度
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    np.save(OUT_VEC, vecs)
    OUT_META.write_text(json.dumps({
        "model": args.model,
        "dim": int(vecs.shape[1]),
        "count": int(vecs.shape[0]),
        "normalized": True,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=1), "utf-8")

    print(f"完成: {vecs.shape[0]} × {vecs.shape[1]} 维，用时 {time.time() - t0:.1f}s")
    print(f"输出: {OUT_VEC.name} ({OUT_VEC.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
