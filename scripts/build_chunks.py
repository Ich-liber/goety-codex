"""把 MCMod 解析结果转成检索用的 chunk。

一个 section 就是一个 chunk（MCMod 的小节天然语义完整）。
保留 breadcrumb 与来源信息，供生成层做引用。

用法：
  python scripts/build_chunks.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "kb" / "corpus_mcmod.jsonl"
OUT = ROOT / "kb" / "chunks.jsonl"

# 纯符号或过短的「伪标题」，不该作为独立 chunk 名
RE_JUNK_HEADING = re.compile(r"^[\s\W\d]+$")


def table_to_text(rows: list[list[str]]) -> str:
    """把相关进度表转成可读文本。

    只保留「名称」和「描述」，丢掉「需求 / 类型 / 父进度 / ID / 是否隐藏」等字段名。
    这些元数据字段在 21 个 chunk 里重复出现，是极强的通用词噪声源：
    「刚开局做什么」会被「名称…需求…类型…」这类词吸引到随便一个进度表上。
    """
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    out = []
    for row in body:
        d = dict(zip(header, row))
        name = (d.get("名称") or "").strip()
        desc = (d.get("描述") or "").strip()
        if name and desc:
            out.append(f"{name}：{desc}")
        elif name:
            out.append(name)
    return "；".join(out)


def main() -> int:
    if not SRC.exists():
        print(f"缺少 {SRC}，先运行 parse_mcmod.py --dir raw/mcmod --out kb/corpus_mcmod.jsonl")
        return 1

    items = [json.loads(l) for l in SRC.read_text("utf-8").splitlines() if l.strip()]

    chunks: list[dict] = []
    index_by_key: dict[tuple[str, str, str], int] = {}
    skipped = 0
    deduped = 0
    for it in items:
        # 部分「游戏设定」类页面的标题会带上模组名（"… - 诡厄巫法"），清掉再入索引
        name = re.sub(r"\s*-\s*诡厄巫法\s*$", "", it["name_zh"]).strip()
        cat = it["category"] or "未分类"
        for i, sec in enumerate(it["sections"]):
            heading = (sec.get("heading") or "").strip()
            text = (sec.get("text") or "").strip()
            if sec.get("table"):
                text = table_to_text(sec["table"])

            if not text:
                skipped += 1
                continue
            if RE_JUNK_HEADING.match(heading):
                heading = "补充"

            # MCMod 的变体页会「引用于」同一份资料，正文完全一致，需去重；
            # 但保留来源 id，供「有几种外观」这类聚合问题使用。
            key = (name, heading, text)
            if key in index_by_key:
                chunks[index_by_key[key]]["aliases"].append(it["mcmod_id"])
                deduped += 1
                continue
            index_by_key[key] = len(chunks)

            chunks.append({
                "chunk_id": f"mcmod:{it['mcmod_id']}#{i}",
                "mcmod_id": it["mcmod_id"],
                "aliases": [],
                "item_id": it["item_id"],
                "title": name,
                "heading": heading,
                "breadcrumb": f"{cat} > {name} > {heading}",
                "text": text,
                "tags": it["tags"],
                "source": "mcmod",
                "source_url": it["source_url"],
                "version": "1.20.1",
                "lang": "zh_cn",
            })

    OUT.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )

    print(f"条目 {len(items)} -> chunk {len(chunks)}（去重 {deduped}，跳过空小节 {skipped}）")
    print(f"输出: {OUT}  {OUT.stat().st_size / 1024:.1f} KB")
    lens = [len(c["text"]) for c in chunks]
    print(f"chunk 字数: min={min(lens)} 中位={sorted(lens)[len(lens)//2]} max={max(lens)} 合计={sum(lens):,}")
    heads = {}
    for c in chunks:
        heads[c["heading"]] = heads.get(c["heading"], 0) + 1
    print("高频小节名:", ", ".join(f"{k}×{v}" for k, v in sorted(heads.items(), key=lambda x: -x[1])[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
