"""查询理解：清洗 + 实体识别 + 意图判定 + 症状扩展。

三层处理：
  1. 实体识别   —— 用 MCMod 物品名做词典，最长优先匹配
  2. 查询清洗   —— 剥离疑问词 / 意图词 / 语气词，只留实义词
  3. 症状扩展   —— 把玩家口语（"没反应"）翻译成文档用词（"激活 条件 有效区域"）

第 3 层是领域知识注入，也是排错型查询唯一的救命稻草：用户描述症状时不会
使用文档里的术语，纯词面匹配必然失效。

用法（作为模块）：
  from query_rewrite import analyze
  info = analyze("我的仪式放上祭品却没有反应")
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
INDEX = ROOT / "kb" / "item_index.json"

# 只用多字停用词，且按长度降序（长词优先删，否则「有什么用」会被「有什么」切碎）。
# 刻意不收单字停用词：「得」会把「获得」拆成「获」，「有」会带来大量噪声，
# 而单字本身的 IDF 很低，交给 BM25 自然降权即可。
STOPWORDS = sorted([
    "有什么好处", "有什么用", "为什么会", "为什么", "是什么", "怎么办", "怎么样",
    "有什么", "怎么", "如何", "怎样", "哪些", "哪个", "多少",
    "请问", "一下", "可不可以", "能不能", "是否", "有没有",
    "需要", "可以", "应该", "我的",
], key=len, reverse=True)

# 玩家口语 -> 文档用词。来自实际阅读文档后的归纳，不是针对某条测试查询硬编码。
SYMPTOM_EXPANSION = {
    "没反应": ["激活", "条件", "有效区域", "需求"],
    "没有反应": ["激活", "条件", "有效区域", "需求"],
    "不起作用": ["激活", "条件", "需求"],
    "不生效": ["激活", "条件", "需求"],
    "不工作": ["激活", "条件", "需求"],
    "不能用": ["激活", "条件", "需求"],
    "放上祭品": ["祭坛基座", "物品"],
    "放了祭品": ["祭坛基座", "物品"],
    "不跟着": ["仆从", "模式", "跟随"],
    "不跟随": ["仆从", "模式", "跟随"],
    "乱跑": ["仆从", "模式"],
    "不听指挥": ["仆从", "模式"],
    "打不过": ["打法", "弱点", "攻击"],
    "不出效果": ["条件", "需求"],
    "没效果": ["条件", "需求"],
}

# 口语动作 -> 文档用词。玩家说「装上」，文档写「装备联动」；
# 玩家说「攒」，文档写「获得」。这类差距靠字符匹配永远跨不过去。
VERB_SYNONYMS = {
    "装上": ["装备", "佩戴"],
    "安上": ["装备", "佩戴"],
    "带上": ["装备", "佩戴"],
    "装": ["装备", "佩戴"],
    "弄": ["制作", "获取"],
    "搞": ["制作", "获取"],
    "攒": ["获得", "获取"],
    "叫出来": ["召唤", "生成"],
    "召不出来": ["召唤", "上限"],
    "做不出来": ["条件", "需求"],
    "开": ["激活", "开启"],
    "解锁": ["开启", "前置"],
    "没用": ["无效", "条件"],
    "干嘛": ["功能", "作用"],
    "干啥": ["功能", "作用"],
}

# 游戏进度语境词 -> 文档里实际出现的词。
# 「刚开局做什么」在字面上没有任何词能匹配到「新手工具」这条进度，
# 但玩家的意思就是它。
GENERIC_EXPANSION = {
    "开局": ["新手", "入门", "进度"],
    "前期": ["新手", "入门", "进度"],
    "新手": ["入门", "进度"],
    "起步": ["新手", "入门"],
}

# 少数高频简称。只收指向明确、歧义小的，避免把「图腾」这种泛指词强行绑定。
ENTITY_ALIASES = {
    "笼子": "诅咒之笼",
    "注入器": "诅咒注入器",
    "坩埚": "女巫坩埚",
}


def expand_verbs(q: str) -> list[str]:
    out: list[str] = []
    for k, vs in VERB_SYNONYMS.items():
        if k in q:
            out.extend(vs)
    return out


def expand_generic(q: str) -> list[str]:
    out: list[str] = []
    for k, vs in GENERIC_EXPANSION.items():
        if k in q:
            out.extend(vs)
    return out


def resolve_aliases(q: str) -> list[str]:
    return [v for k, v in ENTITY_ALIASES.items() if k in q]


INTENT_RULES = [
    ("troubleshoot", ["为什么", "没反应", "没有反应", "不起作用", "不生效", "不能用",
                      "失败", "错误", "崩溃", "召不出来", "做不出来", "不出来", "没用"]),
    ("howto", ["怎么", "如何", "怎样", "步骤", "流程"]),
    ("concept", ["是什么", "有什么用", "作用", "好处", "区别", "介绍"]),
    ("multihop", ["需要", "前置", "先", "材料", "准备"]),
]

# 意图 -> 该意图下答案通常落在哪些小节的标题里。
# 只给 concept 配扩展，这是实测结论：concept 问的是「类别属性」（有什么用/什么好处），
# 实体名本身不足以定位到『特点』这类小节；而 howto / multihop 查询里已经自带
# 动作词（获得、召唤、激活），再扩展只会用泛词稀释精确匹配，实测反而掉分。
INTENT_EXPANSION = {
    "concept": ["特点", "作用", "用途", "效果"],
}


def load_entities() -> list[str]:
    """从物品索引里取全部名称，只保留适合做实体匹配的短名。"""
    if not INDEX.exists():
        return []
    data = json.loads(INDEX.read_text("utf-8"))
    names: set[str] = set()
    for rows in data.values():
        for r in rows:
            n = (r.get("name") or "").strip()
            # 过滤掉过长、含标点/冒号的（那是机制说明而非实体名）
            if 2 <= len(n) <= 12 and not re.search(r"[:：&/（）()\s]", n):
                names.add(n)
    return sorted(names, key=len, reverse=True)


_ENTITIES = load_entities()
# 同时收录「去修饰」的尾词，让「祭坛」也能命中「黑暗祭坛」
_TAILS = sorted({n[-2:] for n in _ENTITIES if len(n) > 2}, key=len, reverse=True)


def find_entities(q: str) -> list[str]:
    """最长优先匹配完整名称，再补类别尾词（如「聚晶」这种 128 个物品的共用后缀）。"""
    hits: list[str] = []
    rest = q
    for name in _ENTITIES:
        if name in rest:
            hits.append(name)
            rest = rest.replace(name, "\u3000")
    for tail in _TAILS:
        # 只在没有更具体的同名实体时才补类别词
        if tail in q and not any(tail in h for h in hits):
            hits.append(tail)
    return hits


def detect_intent(q: str, entities: list[str]) -> str:
    for intent, kws in INTENT_RULES:
        if any(k in q for k in kws):
            return intent
    return "multihop" if not entities else "unknown"


def clean(q: str) -> str:
    s = q
    for w in STOPWORDS:
        s = s.replace(w, " ")
    s = re.sub(r"[，。？！、；：,.?!;:]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def expand(q: str) -> list[str]:
    out: list[str] = []
    for k, terms in SYMPTOM_EXPANSION.items():
        if k in q:
            out.extend(terms)
    return out


def analyze(q: str) -> dict:
    entities = find_entities(q)
    entities += [a for a in resolve_aliases(q) if a not in entities]
    intent = detect_intent(q, entities)
    base = clean(q)
    # 清洗后为空（例如只问了「这是什么」）时退回原查询，避免空检索
    core = base or q
    parts = ([core] + [e for e in entities if e not in core]
             + expand(q) + expand_verbs(q) + expand_generic(q)
             + INTENT_EXPANSION.get(intent, []))
    expanded = " ".join(dict.fromkeys(parts))  # 去重保序，避免「巫妖 巫妖」
    return {
        "raw": q,
        "clean": core,
        "entities": entities,
        "intent": intent,
        "expanded_terms": expand(q),
        "query": expanded,
    }


if __name__ == "__main__":
    tests = [
        "灵魂能量怎么获得",
        "黑暗祭坛怎么激活",
        "我的仪式放上祭品却没有反应",
        "聚晶有什么用",
        "红石傀儡怎么召唤",
        "巫妖有什么好处",
    ]
    print(f"实体词典: {len(_ENTITIES)} 个名称\n")
    for t in tests:
        a = analyze(t)
        print("=" * 60)
        print(f"原查询 : {a['raw']}")
        print(f"意图   : {a['intent']}")
        print(f"实体   : {a['entities']}")
        print(f"清洗后 : {a['clean']}")
        print(f"扩展词 : {a['expanded_terms']}")
        print(f"最终   : {a['query']}")
