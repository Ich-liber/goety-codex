"""交互式问答 CLI。

支持多轮对话：
  - 检索用「上一轮问题 + 当前问题」拼接，缓解「那要怎么变呢」这类指代丢失
  - 生成时把最近几轮问答一起给模型，让它理解上下文
  - 显式保留引用来源，可随时开关

用法：
  python scripts/chat.py
  python scripts/chat.py --k 5 --no-sources

命令：
  /help     显示帮助
  /sources  开关来源显示
  /k N      设置检索条数
  /clear    清空对话历史
  /exit     退出
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _fsfix  # noqa: E402,F401

from answer import (  # noqa: E402
    SYSTEM_PROMPT, build_context, load_env, resolve_api,
)
from query_rewrite import analyze  # noqa: E402
from retrieve_bm25 import load_chunks  # noqa: E402
from retrieve_hybrid import Retriever  # noqa: E402

MAX_TURNS = 3  # 带给模型的历史轮数上限

# 指代词：出现这些词才说明问题依赖上文
ANAPHORA = ("那", "这个", "那个", "它", "它们", "怎么变", "然后", "还有", "上面", "刚才", "呢")


def needs_context(q: str, entities: list[str]) -> bool:
    """判断当前问题是否需要拼接上一轮。

    早期实现是「短于 12 字就拼上一轮」，结果把独立的新话题也拼进去了：
    「傀儡召不出来」只有 6 个字，却被拼成「那要怎么变呢 + 傀儡…」，
    检索被上一轮的巫妖内容污染，直接答不出来。
    现在的规则是：有指代词，或者整句没有任何实体，才借助上文。
    """
    if len(q) > 12:
        return False
    if any(w in q for w in ANAPHORA):
        return True
    return not entities


def call_chat(base: str, key: str, model: str, messages: list[dict]) -> str:
    import requests

    r = requests.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.2},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-sources", action="store_true")
    args = ap.parse_args()

    load_env()
    api = resolve_api()
    if api is None:
        print("未检测到 API key。请在项目根目录 .env 中配置 DEEPSEEK_API_KEY。")
        return 1
    base, key, model = api

    print("正在加载检索索引与向量模型…")
    t0 = time.time()
    chunks = load_chunks()
    rt = Retriever(chunks)
    print(f"就绪（{len(chunks)} 个知识片段，{time.time() - t0:.1f}s）\n")

    print("=" * 70)
    print("  诡厄巫法（Goety 1.20.1）中文问答")
    print("  输入问题直接提问；/help 查看命令；/exit 退出")
    print("=" * 70)

    history: list[tuple[str, str]] = []
    show_sources = not args.no_sources
    k = args.k

    while True:
        try:
            q = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue

        if q.startswith("/"):
            cmd = q.lower()
            if cmd in ("/exit", "/quit"):
                print("再见。")
                break
            if cmd == "/help":
                print("  /sources  开关来源显示\n  /k N      设置检索条数\n  /clear    清空历史\n  /exit     退出")
                continue
            if cmd == "/sources":
                show_sources = not show_sources
                print(f"  来源显示：{'开' if show_sources else '关'}")
                continue
            if cmd.startswith("/k"):
                parts = cmd.split()
                if len(parts) == 2 and parts[1].isdigit():
                    k = max(1, min(int(parts[1]), 10))
                    print(f"  检索条数：{k}")
                else:
                    print("  用法：/k 5")
                continue
            if cmd == "/clear":
                history.clear()
                print("  已清空对话历史。")
                continue
            print("  未知命令，/help 查看帮助")
            continue

        # 检索查询：仅在确有指代时才拼上一轮，避免污染独立话题
        info = analyze(q)
        search_q = info["query"]
        used_ctx = False
        if history and needs_context(q, info["entities"]):
            prev_q = history[-1][0]
            search_q = f"{prev_q} {search_q}"
            used_ctx = True

        ts = time.time()
        hits = rt.search(search_q, "rrf", k, info["entities"])
        if not hits:
            print("\n助手 > 没有检索到相关内容，这个问题可能超出当前知识库范围。")
            continue
        context, refs = build_context(chunks, hits)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for pq, pa in history[-MAX_TURNS:]:
            messages.append({"role": "user", "content": pq})
            messages.append({"role": "assistant", "content": pa})
        messages.append({"role": "user", "content": f"【资料】\n{context}\n\n【问题】\n{q}"})

        try:
            ans = call_chat(base, key, model, messages)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[API 错误] {exc}")
            continue

        cost = time.time() - ts
        tag = " +上文" if used_ctx else ""
        print(f"\n助手 > {ans}")
        if show_sources:
            print("\n  ── 来源 ──")
            for r in refs:
                print(f"  [{r['n']}] {r['breadcrumb']}")
        print(f"\n  ({cost:.1f}s · {info['intent']}{tag} · top{k})")

        history.append((q, ans))

    return 0


if __name__ == "__main__":
    sys.exit(main())
