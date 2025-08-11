#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agentic Chat Server (SCHH) — Markdown + SSE + heartbeat
- Forces temperature=1 for GPT-5 mini and similar models
- Adds a Markdown-oriented system prompt
- Streams as proper SSE (data: {...}\n\n) with an initial heartbeat
"""

import os, re, json, time, secrets, functools, logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from threading import RLock

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages.ai import AIMessageChunk
from pinecone import Pinecone

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("schh-rag")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5-mini")
RERANK_MODEL  = os.getenv("RERANK_MODEL",  "gpt-5-nano")  # if you use LLM rerank
DEFAULT_INDEX = os.getenv("DEFAULT_INDEX", "schh")
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",")]
FORCED_TEMP   = 1

SYSTEM_PROMPT = (
    "You are the SCHH community assistant. "
    "Use the provided context to help answer the user's question. Only use general world knowledge if the context is insufficient. "
    "Answer clearly and concisely using well-structured **Markdown**: headings, lists, tables, and code blocks when useful. "
    "Use short paragraphs and preserve line breaks. "
    "Where applicable, include inline bracketed citations like [1], [2] that map to the sources the app shows. "
    "If unsure, say you’re unsure briefly."
)

embeddings = OpenAIEmbeddings()
pc = Pinecone()
memory = MemorySaver()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def sse(data: Dict[str, Any]) -> str:
    """Encode an SSE event with a JSON payload."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@functools.lru_cache(maxsize=2048)
def get_text_cached(chunk_id: str, index_name: str) -> Optional[str]:
    """Fetch a chunk's raw text from Pinecone by id."""
    try:
        idx = pc.Index(index_name)
        res = idx.query(id=chunk_id, top_k=1, include_values=False, include_metadata=True)
        matches = res.get("matches", [])
        if matches:
            return (matches[0].get("metadata") or {}).get("text")
    except Exception as e:
        log.warning(f"get_text_cached({chunk_id}) error: {e}")
    return None

def compress_snippets(snippets: List[str], max_chars: int = 3500) -> str:
    seen, out, total = set(), [], 0
    for s in snippets:
        s = re.sub(r"\s+", " ", s or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        total += len(s)
        if total >= max_chars:
            break
    return "\n\n".join(out)

# -----------------------------------------------------------------------------
# Agent builder (cached per (model,index))
# -----------------------------------------------------------------------------
_build_lock = RLock()

@functools.lru_cache(maxsize=16)
def build_agent(model: str, index_name: str):
    log.info(f"[build_agent] model={model} index={index_name}")

    # Main LLM
    if model.startswith(("gpt-", "o")):
        llm = ChatOpenAI(model=model, temperature=FORCED_TEMP)
    else:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set but Anthropic model requested")
        llm = ChatAnthropic(model_name=model)  # Anthropic ignores 'temperature' here

    # If you later add an LLM-based reranker, keep temperature=1 there too:
    # rerank_llm = ChatOpenAI(model=RERANK_MODEL, temperature=FORCED_TEMP)

    # Vector store
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index, embeddings, "text")

    # ----- Tools -----
    @tool
    def current_date() -> str:
        """Use this to get the current date (YYYY-MM-DD)."""
        return datetime.today().strftime("%Y-%m-%d")

    @tool
    def schh_weather() -> str:
        """Get a short weather summary for the Sun City Hilton Head region (best-effort)."""
        from langchain_community.tools.tavily_search import TavilySearchResults
        try:
            tv = TavilySearchResults(max_results=3, search_depth="advanced", include_answer=True)
            res = tv.invoke({"query": "Weather forecast Sun City Hilton Head SC"})
            return json.dumps(res, ensure_ascii=False)[:4000]
        except Exception as e:
            return f"Weather lookup failed: {e}"

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve relevant knowledge. Returns serialized text and artifacts (docs)."""
        retrieved = vector_store.similarity_search(query, k=40)

        combined = []
        seen_docids = set()
        for doc in retrieved:
            base_id = (doc.id or "").rsplit(":", 1)[0]
            if base_id in seen_docids:
                continue
            seen_docids.add(base_id)

            ctx = {}
            if getattr(doc, "page_content", None):
                ctx[doc.id] = doc.page_content

            meta = getattr(doc, "metadata", {}) or {}
            parent_id = meta.get("parent_id")
            if parent_id:
                t = get_text_cached(parent_id, index_name)
                if t: ctx[parent_id] = t

            for sib_id in meta.get("sib_ids", []) or []:
                t = get_text_cached(sib_id, index_name)
                if t: ctx[sib_id] = t

            doc.page_content = compress_snippets([ctx[k] for k in sorted(ctx)])
            combined.append(doc)

        serialized = "\n||\n".join(
            f"Source: {json.dumps(getattr(d, 'metadata', {}) or {})}\n"
            f"Content: {getattr(d, 'page_content', '')}"
            for d in combined[:6]  # keep top few by initial similarity
        )
        return serialized, combined[:6]

    tools = [current_date, schh_weather, retrieve]
    agent = create_react_agent(llm.bind(temperature=FORCED_TEMP), tools, checkpointer=memory)
    log.info(f"[build_agent] constructed llm model={model} temp={FORCED_TEMP}")
    return agent

def get_agent(model: str, index_name: str):
    with _build_lock:
        return build_agent(model, index_name)

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI()

class CacheControlStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.mount("/static", CacheControlStaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"message": "SCHH RAG Chat"})

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -----------------------------------------------------------------------------
# Chat logic
# -----------------------------------------------------------------------------
def build_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

async def stream_response(agent, messages, config):
    """SSE stream: yields a heartbeat first, then incremental content, then final sources."""
    sources: List[Dict[str, Any]] = []
    # Heartbeat so the client shows activity immediately
    yield sse({"type": "content", "delta": ""})

    try:
        async for msg, meta in agent.astream({"messages": messages}, config, stream_mode="messages"):
            if msg.content:
                if meta.get("langgraph_node") == "agent":
                    if isinstance(msg, AIMessageChunk):
                        if isinstance(msg.content, str):
                            delta = msg.content
                        else:
                            # LC can emit list chunks
                            delta = "".join([rec.get("text", "") for rec in msg.content if isinstance(rec, dict)])
                        if delta:
                            yield sse({"type": "content", "delta": delta})
                else:
                    # Tool outputs; try to collect source metadata if present
                    try:
                        docs = (msg.content or "").split("\n||\n")
                        meta_strs = [re.sub(r"^Source:\s*", "", d.split("\n")[0]) for d in docs if d.strip()]
                        for m in meta_strs:
                            rec = json.loads(m)
                            src = {k: v for k, v in rec.items() if k in ("id", "title", "source", "chunk_id")}
                            if "id" in src:
                                src_id = src["id"].rsplit(":", 1)[0]
                                src["id"] = src_id
                                if not any(s.get("doc_id") == src_id for s in sources):
                                    sources.append({
                                        "doc_id": src_id,
                                        "title": src.get("title") or src_id,
                                        "uri":   src.get("source"),
                                        "chunk": src.get("chunk_id")
                                    })
                    except Exception as e:
                        log.debug(f"source parse error: {e}")
    except Exception as e:
        yield sse({"type": "error", "message": str(e)})
    # Finalize with sources
    yield sse({"type": "final", "sources": sources})

def non_stream_response(agent, messages, config, model: str, index: str) -> JSONResponse:
    t0 = time.time()
    result = agent.invoke({"messages": messages}, config)
    latency = int((time.time() - t0) * 1000)
    answer = result["messages"][-1].content if isinstance(result, dict) else ""

    # Best-effort source extraction (same logic as stream)
    sources: List[Dict[str, Any]] = []
    inter = result.get("messages", []) if isinstance(result, dict) else []
    for m in inter:
        try:
            docs = (getattr(m, "content", "") or "").split("\n||\n")
            meta_strs = [re.sub(r"^Source:\s*", "", d.split("\n")[0]) for d in docs if d.strip()]
            for ms in meta_strs:
                rec = json.loads(ms)
                src_id = (rec.get("id") or "").rsplit(":", 1)[0]
                if src_id and not any(s.get("doc_id") == src_id for s in sources):
                    sources.append({
                        "doc_id": src_id,
                        "title": rec.get("title") or src_id,
                        "uri":   rec.get("source"),
                        "chunk": rec.get("chunk_id")
                    })
        except Exception:
            pass

    return JSONResponse({
        "answer": answer,
        "sources": sources,
        "model": model,
        "index": index,
        "latency_ms": latency
    })

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/chat")
@app.post("/chat/")
async def chat_post(request: Request):
    payload = await request.json()
    prompt = payload.get("prompt", "")
    model  = payload.get("model", DEFAULT_MODEL)
    index  = payload.get("index", DEFAULT_INDEX)
    stream = bool(payload.get("stream", False))
    sessionid = payload.get("sessionid", secrets.token_hex(4))

    agent = get_agent(model, index)
    config = {"configurable": {"thread_id": sessionid}}
    messages = build_messages(prompt)

    if stream:
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        return StreamingResponse(stream_response(agent, messages, config), headers=headers)
    else:
        return non_stream_response(agent, messages, config, model, index)

# Optional GET route kept for compatibility
@app.get("/chat/{prompt}")
async def chat_get(prompt: str, request: Request,
                   sessionid: Optional[str] = None,
                   model: Optional[str] = None,
                   index: Optional[str] = None,
                   stream: Optional[bool] = False):
    agent = get_agent(model or DEFAULT_MODEL, index or DEFAULT_INDEX)
    config = {"configurable": {"thread_id": sessionid or secrets.token_hex(4)}}
    messages = build_messages(prompt)
    if stream:
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        return StreamingResponse(stream_response(agent, messages, config), headers=headers)
    else:
        return non_stream_response(agent, messages, config, model or DEFAULT_MODEL, index or DEFAULT_INDEX)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        build_agent.cache_clear()
    except Exception:
        pass
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)