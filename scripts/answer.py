"""生成层：检索 -> 拼上下文 -> 调用 LLM -> 带引用的答案。

设计要点
--------
1. 回答严格约束在检索到的资料内，资料不足时必须说不知道，不许外推。
2. 上下文带编号和 breadcrumb，答案里用 [n] 标注来源，可溯源到 MCMod 页面。
3. key 只从环境变量 / .env 读取，不写进代码，也不打印。
4. 无 key 时退化为"模板拼装"：直接列出命中原文与来源，不做生成。

用法：
  python scripts/answer.py --q "变成巫妖有啥好处"
  python scripts/answer.py --q "傀儡召不出来" --k 5
  python scripts/answer.py --q "..." --no-llm      # 只用模板，不调 API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _fsfix  # noqa: E402,F401

ENV_FILE = ROOT / ".env"

SYSTEM_PROMPT = """你是《我的世界》模组「诡厄巫法（Goety）」的中文问答助手，只负责 1.20.1 版本。

回答规则：
1. 只依据【资料】中的内容回答。资料没有写的，不要用你自己的知识补充。
2. 如果资料不足以回答这个问题，直接说「现有资料里没有找到相关内容」，并说明缺什么，不要猜测。
3. 回答要具体、可操作。涉及操作步骤时按顺序说明，涉及前提条件时先讲条件。
4. 每个关键结论后面用 [编号] 标出来源，编号必须是资料里给出的。
5. 用简体中文，像有经验的玩家在讲解，不要整段照抄资料原文。"""


def load_env() -> None:
    """读取项目内 .env（KEY=VALUE 每行一条），不覆盖已有环境变量。"""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_api() -> tuple[str, str, str] | None:
    """返回 (base_url, api_key, model)，没有可用 key 时返回 None。"""
    if key := os.getenv("DEEPSEEK_API_KEY"):
        return (
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            key,
            os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    if key := os.getenv("OPENAI_API_KEY"):
        return (
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            key,
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    return None


def build_context(chunks: list[dict], hits) -> tuple[str, list[dict]]:
    blocks, refs = [], []
    for n, (i, _score) in enumerate(hits, 1):
        c = chunks[i]
        refs.append({"n": n, "breadcrumb": c["breadcrumb"], "url": c["source_url"]})
        blocks.append(
            f"[{n}] {c['breadcrumb']}\n{c['text']}\n来源：{c['source_url']}"
        )
    return "\n\n".join(blocks), refs


def call_llm(base: str, key: str, model: str, question: str, context: str) -> str:
    import requests

    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【资料】\n{context}\n\n【问题】\n{question}"},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def template_answer(question: str, refs: list[dict], chunks: list[dict], hits) -> str:
    """无 key 时的兜底：不做生成，只把命中的原文和来源列出来。"""
    out = [f"（未配置 API key，以下为检索到的原始资料，未经生成）\n"]
    for n, (i, score) in enumerate(hits, 1):
        c = chunks[i]
        out.append(f"[{n}] {c['breadcrumb']}")
        out.append(f"    {c['text'][:300]}{'…' if len(c['text']) > 300 else ''}")
        out.append(f"    来源：{c['source_url']}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--engine", default="rrf", choices=["bm25", "vector", "rrf"])
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    load_env()

    from query_rewrite import analyze
    from retrieve_hybrid import Retriever
    from retrieve_bm25 import load_chunks

    chunks = load_chunks()
    info = analyze(args.q)
    rt = Retriever(chunks)
    hits = rt.search(info["query"], args.engine, args.k, info["entities"])
    if not hits:
        print("没有检索到任何相关内容。这个问题可能超出了当前知识库范围。")
        return 0

    context, refs = build_context(chunks, hits)

    print("=" * 68)
    print(f"问题     : {args.q}")
    print(f"意图     : {info['intent']}   实体: {info['entities'] or '—'}")
    print(f"改写查询 : {info['query']}")
    print(f"检索     : {args.engine}  top{len(hits)}")
    print("=" * 68)

    api = None if args.no_llm else resolve_api()
    if api is None:
        print(template_answer(args.q, refs, chunks, hits))
        if not args.no_llm:
            hint = "未检测到 DEEPSEEK_API_KEY / OPENAI_API_KEY"
            if not ENV_FILE.exists():
                hint += f"（可在 {ENV_FILE.name} 中配置，参见 .env.example）"
            print(f"\n[{hint}]")
        return 0

    base, key, model = api
    print(f"[调用 {model}]\n")
    try:
        ans = call_llm(base, key, model, args.q, context)
    except Exception as exc:  # noqa: BLE001
        print(f"[API 调用失败] {exc}")
        print("\n退回模板答案：\n")
        print(template_answer(args.q, refs, chunks, hits))
        return 1

    print(ans)
    print("\n" + "-" * 68)
    print("来源：")
    for r in refs:
        print(f"  [{r['n']}] {r['breadcrumb']}")
        print(f"      {r['url']}")

    # 引用编号是否都在资料范围内
    used = sorted({int(x) for x in re.findall(r"\[(\d+)\]", ans)})
    bad = [u for u in used if u < 1 or u > len(refs)]
    if bad:
        print(f"\n[警告] 答案引用了不存在的编号 {bad}，可能存在幻觉")
    return 0


if __name__ == "__main__":
    sys.exit(main())
