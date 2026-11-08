"""Goety Codex 后端：SSE 流式问答接口 + 静态页面。

启动：
  python scripts/../run_server.bat
  或 python -m uvicorn server:app --host 127.0.0.1 --port 8077
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import _fsfix  # noqa: E402,F401

from answer import SYSTEM_PROMPT, build_context, load_env, resolve_api  # noqa: E402
from query_rewrite import analyze  # noqa: E402
from retrieve_bm25 import load_chunks  # noqa: E402
from retrieve_hybrid import Retriever  # noqa: E402

STATIC = ROOT / "static"
MAX_TURNS = 3
ANAPHORA = ("那", "这个", "那个", "它", "它们", "怎么变", "然后", "还有", "上面", "刚才", "呢")

load_env()
API = resolve_api()

print("[boot] 载入知识库…", flush=True)
CHUNKS = load_chunks()
print(f"[boot] {len(CHUNKS)} 个片段，加载检索器与向量模型…", flush=True)
RETRIEVER = Retriever(CHUNKS)
print("[boot] 就绪", flush=True)

app = FastAPI(title="Goety Codex")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class Turn(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    question: str
    history: list[Turn] = []
    mode: str = "rerank"   # 62 条评测集上 rerank 明显优于 rrf：Hit@1 38 vs 26
    topk: int = 4


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def needs_context(q: str, entities: list[str]) -> bool:
    if len(q) > 12:
        return False
    if any(w in q for w in ANAPHORA):
        return True
    return not entities


def stream_completion(messages: list[dict]):
    """同步生成器，逐 token 产出。StreamingResponse 会把它放到线程池执行。"""
    import requests

    base, key, model = API
    r = requests.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.2, "stream": True},
        timeout=180,
        stream=True,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
            delta = obj["choices"][0].get("delta", {}).get("content")
        except Exception:
            continue
        if delta:
            yield delta


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    pages = len({c["mcmod_id"] for c in CHUNKS})
    return {
        "chunks": len(CHUNKS),
        "pages": pages,
        "version": "1.20.1",
        "model": API[2] if API else None,
        "chars": sum(len(c["text"]) for c in CHUNKS),
    }


@app.post("/api/chat")
def chat(req: ChatReq):
    def gen():
        t0 = time.time()

        if API is None:
            yield sse("error", {"msg": "服务端未配置 API key"})
            return

        info = analyze(req.question)
        query = info["query"]
        used_ctx = False
        if req.history and needs_context(req.question, info["entities"]):
            prev = [t for t in req.history if t.role == "user"]
            if prev:
                query = f"{prev[-1].content} {query}"
                used_ctx = True

        hits = RETRIEVER.search(query, req.mode, req.topk, info["entities"])
        if not hits:
            yield sse("meta", {
                "intent": info["intent"], "query": query, "used_ctx": used_ctx,
                "entities": info["entities"], "hits": [], "sources": [],
            })
            yield sse("delta", {"t": "没有检索到与这个问题相关的内容，它可能超出了当前知识库的范围。"})
            yield sse("done", {"sources": [], "elapsed": round(time.time() - t0, 1),
                               "refused": True})
            return

        context, refs = build_context(CHUNKS, hits)
        yield sse("meta", {
            "intent": info["intent"],
            "query": query,
            "used_ctx": used_ctx,
            "entities": info["entities"],
            "hits": [
                {"n": n, "breadcrumb": CHUNKS[i]["breadcrumb"]}
                for n, (i, _s) in enumerate(hits, 1)
            ],
            "sources": refs,
        })

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in req.history[-MAX_TURNS * 2:]:
            messages.append({"role": t.role, "content": t.content})
        messages.append({"role": "user", "content": f"【资料】\n{context}\n\n【问题】\n{req.question}"})

        acc = []
        try:
            for tok in stream_completion(messages):
                acc.append(tok)
                yield sse("delta", {"t": tok})
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"msg": str(exc)})
            return

        text = "".join(acc)
        refused = any(k in text[:40] for k in ("没有找到", "没有相关内容", "资料里没有", "未找到"))
        yield sse("done", {
            "sources": refs,
            "elapsed": round(time.time() - t0, 1),
            "refused": refused,
        })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8077, log_level="info")
