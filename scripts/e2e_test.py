"""端到端验证：5 类问题各一条，输出 markdown 报告。

覆盖概念 / 排错 / 操作 / 变体 / 拒答，其中拒答最能说明幻觉抑制是否生效。

用法：
  python scripts/e2e_test.py
  python scripts/e2e_test.py --k 4 --out docs/e2e_results.md
"""

from __future__ import annotations

import argparse
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
    build_context, call_llm, load_env, resolve_api, template_answer,
)
from query_rewrite import analyze  # noqa: E402
from retrieve_bm25 import load_chunks  # noqa: E402
from retrieve_hybrid import Retriever  # noqa: E402

CASES = [
    ("变成巫妖有啥好处", "概念", "应命中『巫妖 > 特点』"),
    ("傀儡召不出来", "排错", "应命中『红石傀儡 > 最大召唤上限』"),
    ("祭坛怎么开", "操作", "应命中『黑暗祭坛 > 激活』"),
    ("大师死灵仪式和普通的有啥区别", "变体", "应命中死灵仪式或巫妖页"),
    ("机械动力的齿轮怎么做", "拒答", "必须明确说资料里没有"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="docs/e2e_results.md")
    args = ap.parse_args()

    load_env()
    api = resolve_api()
    if api is None:
        print("未检测到 API key，无法跑生成层。可在项目根目录 .env 中配置。")
        return 1
    base, key, model = api

    chunks = load_chunks()
    rt = Retriever(chunks)  # 只加载一次模型
    lines = [
        "# 端到端验证结果",
        "",
        f"- 生成模型：`{model}`",
        f"- 检索：RRF（BM25 + bge-small-zh-v1.5），top{args.k}",
        f"- 知识库：{len(chunks)} 个 chunk（MCMod 28 个页面，Goety 1.20.1）",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for i, (q, kind, expect) in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] {q}")
        info = analyze(q)
        hits = rt.search(info["query"], "rrf", args.k, info["entities"])
        context, refs = build_context(chunks, hits)

        lines += ["---", "", f"## {i}. {q}", "", f"**类型**：{kind}　**预期**：{expect}", ""]
        lines += ["**检索改写**：`" + info["query"] + "`", ""]
        lines += ["**命中片段**：", ""]
        for n, (idx, _s) in enumerate(hits, 1):
            lines.append(f"- [{n}] {chunks[idx]['breadcrumb']}")
        lines.append("")

        try:
            ans = call_llm(base, key, model, q, context)
        except Exception as exc:  # noqa: BLE001
            ans = f"（API 调用失败：{exc}）"
        lines += ["**回答**：", "", ans, ""]
        lines += ["**来源**：", ""]
        for r in refs:
            lines.append(f"- [{r['n']}] {r['breadcrumb']} — {r['url']}")
        lines.append("")

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), "utf-8")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
