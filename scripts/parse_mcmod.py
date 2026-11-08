"""解析 MCMod 物品页 HTML，产出结构化 JSON。

提取字段：
  mcmod_id / item_id（goety:xxx）/ name_zh / name_en / 资料分类 / 最大堆叠
  tags（矿物词典）/ give 命令 / 正文小节 / 提到过的其它物品（内链）

正文切分依据 MCMod 的实际结构：
  <p><span>小节名</span></p>            -> 小节标题
  <p>…</p>                              -> 段落
  <ul><li>…</li></ul>                   -> 列表项
  <legend>…</legend>                    -> 引用说明，跳过

用法：
  python scripts/parse_mcmod.py --file raw/mcmod/935700.html
  python scripts/parse_mcmod.py --dir raw/mcmod --out kb/mcmod_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

RE_TITLE = re.compile(r"^\s*(?P<zh>.+?)\s*\((?P<en>.+?)\)\s*-\s*")
RE_GIVE = re.compile(r"/give\s+@[ps]\s+(?P<id>[a-z0-9_.\-]+:[a-z0-9_./\-]+)")
RE_TAG = re.compile(r"([a-z0-9_.\-]+:[a-z0-9_/.\-]+)\s*\(\d+\)")


def text_of(node) -> str:
    raw = node.get_text(" ", strip=True)
    raw = re.sub(r"\s+", " ", raw)
    # get_text(" ") 会在 <a> 两侧注入空格，中文之间不需要，去掉
    raw = re.sub(r"(?<=[\u4e00-\u9fff\uff00-\uffef])\s+(?=[\u4e00-\u9fff\uff00-\uffef])", "", raw)
    return raw.strip()


def parse_kv_table(soup: BeautifulSoup) -> dict[str, str]:
    """抓「主要名称：」「资料分类：」这类两列表格行。"""
    out: dict[str, str] = {}
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) != 2:
            continue
        key = text_of(cells[0]).rstrip("：:").strip()
        val = text_of(cells[1])
        if key and val:
            out.setdefault(key, val)
    return out


def body_container(soup: BeautifulSoup):
    """正文容器。

    所有页面类型统一用 div.item-content。少数条目是「引用其它页面」的形式，
    正文被再包一层 fieldset.quote_text（其中 legend 是引用声明，需跳过），
    这种情况下下沉一层，保证子元素序列一致。
    """
    box = soup.select_one("div.item-content")
    if box is None:
        return soup.select_one("fieldset.quote_text")
    inner = box.select_one("fieldset.quote_text")
    return inner if inner is not None else box


def parse_sections(soup: BeautifulSoup) -> list[dict]:
    """把正文按小节切开。"""
    box = body_container(soup)
    if box is None:
        return []

    sections: list[dict] = []
    heading = "概述"
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = " ".join(x for x in buf if x).strip()
        if body:
            sections.append({"heading": heading, "text": body})
        buf = []

    for child in box.children:
        name = getattr(child, "name", None)
        if name in (None, "legend"):
            continue

        if name == "p":
            spans = child.find_all("span", recursive=False)
            # 小节标题：p 里只有一个 span，且文本很短
            if len(spans) == 1 and not child.find("b"):
                title = text_of(spans[0])
                if title and len(title) <= 20:
                    flush()
                    heading = title
                    continue
            buf.append(text_of(child))
        elif name in ("ul", "ol"):
            # 有序列表也是正文：MCMod 的「条件」「步骤」类小节常用 <ol>，
            # 早期版本只处理了 <ul>，导致「变成巫妖的条件」整段丢失
            for li in child.find_all("li", recursive=False):
                t = text_of(li)
                if t:
                    buf.append(t if t.endswith(("。", "；", "！", "，")) else t + "。")
        elif name == "table":
            # 相关进度表，暂存为结构化行
            rows = []
            for tr in child.find_all("tr"):
                cells = [text_of(td) for td in tr.find_all(["th", "td"])]
                if cells:  # 包含表头行，后续清洗
                    rows.append(cells)
            if rows:
                sections.append({"heading": "相关进度表", "text": "", "table": rows})

    flush()
    return sections


def parse_mentions(soup: BeautifulSoup, self_id: str) -> list[dict]:
    """正文里指向其它 MCMod 物品的内链，是免费的关系数据。"""
    box = body_container(soup) or soup
    seen: dict[str, str] = {}
    for a in box.find_all("a", href=True):
        if a.find_parent("legend") is not None:
            continue  # legend 是「引用于本页」的自引用，不算提及
        m = re.search(r"/item/(\d+)\.html", a["href"])
        if not m:
            continue
        mid = m.group(1)
        if mid == self_id or mid in seen:
            continue
        label = text_of(a)
        if label:
            seen[mid] = label
    return [{"mcmod_id": k, "text": v} for k, v in seen.items()]


def parse_file(path: Path) -> dict | None:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    mcmod_id = path.stem
    title_raw = (soup.title.string or "").strip() if soup.title else ""
    m = RE_TITLE.match(title_raw)
    name_zh = m.group("zh") if m else ""
    name_en = m.group("en") if m else ""

    kv = parse_kv_table(soup)
    if not name_zh:
        name_zh = kv.get("主要名称", "")
    if not name_en:
        name_en = kv.get("次要名称", "")

    give_raw = ""
    give_node = soup.select_one("div.item-give")
    if give_node is not None:
        give_raw = text_of(give_node)
    item_id = ""
    gm = RE_GIVE.search(give_raw)
    if gm:
        item_id = gm.group("id")

    tags: list[str] = []
    dict_node = soup.select_one("p.item-dict")
    if dict_node is not None:
        tags = RE_TAG.findall(text_of(dict_node))

    if not name_zh and not item_id:
        return None  # 不是有效物品页

    return {
        "mcmod_id": mcmod_id,
        "item_id": item_id,
        "name_zh": name_zh,
        "name_en": name_en,
        "category": kv.get("资料分类", ""),
        "stack_size": kv.get("最大堆叠", ""),
        "tags": tags,
        "sections": parse_sections(soup),
        "mentions": parse_mentions(soup, mcmod_id),
        "source_url": f"https://www.mcmod.cn/item/{mcmod_id}.html",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.file:
        p = Path(args.file)
        paths.append(p if p.is_absolute() else ROOT / p)
    elif args.dir:
        d = Path(args.dir)
        d = d if d.is_absolute() else ROOT / d
        paths = sorted(p for p in d.glob("*.html") if not p.name.startswith("list_"))
    else:
        ap.error("需要 --file 或 --dir")

    records = [r for r in (parse_file(p) for p in paths) if r]

    if args.out:
        out = Path(args.out)
        out = out if out.is_absolute() else ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"已写入 {len(records)} 条 -> {out}")

    print("=" * 62)
    print(f"解析成功 {len(records)}/{len(paths)}")
    with_id = sum(1 for r in records if r["item_id"])
    with_tags = sum(1 for r in records if r["tags"])
    with_sec = sum(1 for r in records if r["sections"])
    n_sec = sum(len(r["sections"]) for r in records)
    n_men = sum(len(r["mentions"]) for r in records)
    n_txt = sum(len(s.get("text", "")) for r in records for s in r["sections"])
    print(f"  有 item_id（可 join jar）: {with_id}")
    print(f"  有 tags                  : {with_tags}")
    print(f"  有小节                   : {with_sec}")
    print(f"  小节总数                 : {n_sec}")
    print(f"  正文总字数               : {n_txt:,}")
    print(f"  提及物品总数             : {n_men}")
    print("=" * 62)

    for r in records[:4]:
        print("=" * 62)
        print(f"{r['name_zh']}  ({r['name_en']})")
        print(f"  item_id   : {r['item_id']}")
        print(f"  分类      : {r['category']}")
        print(f"  tags      : {r['tags']}")
        print(f"  小节      : {[s['heading'] for s in r['sections']]}")
        print(f"  提及物品  : {[m['text'] for m in r['mentions'][:5]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
