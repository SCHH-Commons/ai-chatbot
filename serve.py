#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agentic Chat Server (SCHH) — Markdown + SSE + heartbeat + hybrid retrieval + namespaces
- Hybrid Pinecone query (dense + sparse via BM25 encoder)
- Markdown-oriented system prompt
- SSE streaming with initial heartbeat
- Session memory via LangGraph MemorySaver (thread_id=sessionid)
- Query rewrite, sentence-aware compression, strict LLM rerank
- Temperature forced to 1
"""

import os, re, json, time, secrets, functools, logging, math
from typing import Optional, List, Dict, Any
from datetime import datetime
from threading import RLock

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages.ai import AIMessageChunk

from pinecone import Pinecone

# Optional: hybrid sparse encoder (BM25)
try:
    import joblib
    from pinecone_text.sparse import BM25Encoder
except Exception:  # keep server running even if package missing
    joblib = None
    BM25Encoder = None  # type: ignore

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("schh-rag")

DEFAULT_MODEL      = os.getenv("DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_INDEX      = os.getenv("DEFAULT_INDEX", "schh")
DEFAULT_NAMESPACE  = os.getenv("DEFAULT_NAMESPACE", 'schh_v2') 
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-small")

ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",")]
FORCED_TEMP   = 1
HYBRID_ALPHA  = float(os.getenv("HYBRID_ALPHA", "0.5"))
BM25_ENCODER_PATH = os.getenv("BM25_ENCODER_PATH", "./bm25_encoder_schh.joblib")

SYSTEM_PROMPT = (
    "You are the SCHH community assistant.\n"
    "Use the provided context to answer. Only use general world knowledge if the context is insufficient.\n"
    "Reply in clear, well-structured **Markdown** (headings, lists, tables, code blocks when useful).\n"
    "Use short paragraphs and preserve line breaks. Add inline bracketed citations like [1], [2] that map to sources.\n"
    "If unsure, say so briefly."
)

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
pc = Pinecone()
memory = MemorySaver()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def sse(data: Dict[str, Any]) -> str:
    """Encode an SSE event with a JSON payload."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@functools.lru_cache(maxsize=2048)
def get_text_cached(chunk_id: str, index_name: str, namespace: Optional[str]) -> Optional[str]:
    """Fetch a chunk's raw text from Pinecone by id (namespace-aware)."""
    try:
        idx = pc.Index(index_name)
        res = idx.query(
            id=chunk_id,
            top_k=1,
            include_values=False,
            include_metadata=True,
            namespace=namespace
        )
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        if matches:
            md = matches[0].get("metadata") if isinstance(matches[0], dict) else getattr(matches[0], "metadata", {})
            return (md or {}).get("text")
    except Exception as e:
        log.warning(f"get_text_cached({chunk_id}) error: {e}")
    return None

# --- sentence-aware utilities ---
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts, out, in_code = text.split("\n"), [], False
    buf = []
    for line in parts:
        if line.strip().startswith("```"):
            in_code = not in_code
        if in_code:
            buf.append(line)
        else:
            if buf:
                out.append("\n".join(buf)); buf = []
            sents = _SENT_RE.split(line.strip())
            for s in sents:
                if s:
                    out.append(s.strip())
    if buf:
        out.append("\n".join(buf))
    return [s for s in out if s.strip()]

def sentaware_compress(texts: List[str], max_chars: int = 3500) -> str:
    """Sentence-aware dedupe + trim. Preserves code fences & bullets better than flat whitespace trim."""
    seen = set(); out = []; total = 0
    for t in texts:
        if "```" in (t or ""):
            s = t.strip()
            if s and s not in seen:
                seen.add(s); out.append(s); total += len(s)
                if total >= max_chars: break
            continue
        for s in split_sentences(t or ""):
            s = re.sub(r"\s+", " ", s).strip()
            if not s or s in seen: continue
            seen.add(s); out.append(s); total += len(s)
            if total >= max_chars: break
        if total >= max_chars: break
    return "\n".join(out)

def tokenish_overlap(a: str, b: str) -> float:
    """Cheap lexical overlap (proxy for MMR diversity & scoring)."""
    A = set(re.findall(r"[A-Za-z0-9]{3,}", a.lower()))
    B = set(re.findall(r"[A-Za-z0-9]{3,}", b.lower()))
    if not A or not B: return 0.0
    return len(A & B) / math.sqrt(len(A) * len(B))

# Hybrid encoder (lazy load)
_bm25: Optional["BM25Encoder"] = None  # type: ignore
def get_bm25():
    global _bm25
    if _bm25 is not None:
        return _bm25
    if joblib is None or BM25Encoder is None:
        return None
    if not os.path.exists(BM25_ENCODER_PATH):
        log.info("[hybrid] BM25 encoder not found; running dense-only.")
        return None
    try:
        _bm25 = joblib.load(BM25_ENCODER_PATH)
        log.info(f"[hybrid] BM25 encoder loaded from {BM25_ENCODER_PATH}")
        return _bm25
    except Exception as e:
        log.warning(f"[hybrid] failed to load BM25 encoder: {e}")
        return None

# -----------------------------------------------------------------------------
# Agent builder (cached per (model,index,namespace))
# -----------------------------------------------------------------------------
_build_lock = RLock()

@functools.lru_cache(maxsize=24)
def build_agent(model: str, index_name: str, namespace: Optional[str]):
    log.info(f"[build_agent] model={model} index={index_name} ns={namespace}")

    # Main LLM
    if model.startswith(("gpt-", "o")):
        llm = ChatOpenAI(model=model, temperature=FORCED_TEMP)
    else:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set but Anthropic model requested")
        llm = ChatAnthropic(model_name=model)  # Anthropic path

    # ----- Tools -----
    @tool
    def current_date() -> str:
        """Use this to get the current date (YYYY-MM-DD)."""
        return datetime.today().strftime("%Y-%m-%d")

    @tool
    def schh_weather() -> str:
        """Get a short weather summary for the Sun City Hilton Head region (best-effort)."""
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tv = TavilySearchResults(max_results=3, search_depth="advanced", include_answer=True)
            res = tv.invoke({"query": "Weather forecast Sun City Hilton Head SC"})
            return json.dumps(res, ensure_ascii=False)[:4000]
        except Exception as e:
            return f"Weather lookup failed: {e}"

    # Lightweight query rewriter (same main LLM; temp=1 deterministic)
    @tool
    def rewrite_query(q: str) -> str:
        """Rewrite/expand a user query with synonyms/variants/abbreviations for retrieval. Keep it under 30 words."""
        try:
            prompt = (
                "Rewrite this for retrieval: add synonyms, expand abbreviations, and include likely variants. "
                "Return one line, <30 words, no explanations.\n\nQuery: " + q
            )
            resp = llm.invoke(prompt)
            return getattr(resp, "content", "")[:400]
        except Exception:
            return q

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """
        Hybrid Pinecone retrieval:
          - dense embed of query
          - sparse BM25 vector of query (if encoder available)
          - single Pinecone query with alpha blend (namespace-aware)
          - group by base doc id → cap chunks per doc
          - sentence-aware compression
          - strict LLM rerank (indices only) to top N docs
        Returns serialized text for the LLM + artifacts for sources.
        """
        # 1) Encode query (dense + optional sparse)
        dense_q = embeddings.embed_query(query)
        bm25 = get_bm25()
        sparse_q = bm25.encode_queries([query])[0] if bm25 else None

        # 2) Hybrid query to Pinecone (single round trip)
        pine = pc.Index(index_name)
        try:
            resp = pine.query(
                vector=dense_q,
                sparse_vector=sparse_q,   # None → dense-only
                top_k=50,
                include_metadata=True,
                alpha=HYBRID_ALPHA,
                namespace=namespace,
            )
            matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
        except Exception as e:
            log.warning(f"[retrieve] pinecone query failed: {e}")
            return "", []

        # 3) Wrap matches into light doc objects
        docs = []
        for m in matches or []:
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", "")
            mmeta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
            content = (mmeta or {}).get("text", "")
            d = type("Doc", (), {})()
            d.id = mid
            d.metadata = mmeta or {}
            d.page_content = content
            docs.append(d)

        # 4) Group by base doc id; keep up to 3 chunks/doc
        by_doc: Dict[str, List] = {}
        for d in docs:
            base_id = (d.id or "").rsplit(":", 1)[0]
            if not base_id:
                continue
            by_doc.setdefault(base_id, [])
            if len(by_doc[base_id]) < 3:
                by_doc[base_id].append(d)

        # 5) Merge + sentence-aware compression (add parent/siblings where available)
        merged_docs = []
        for base_id, chunks in by_doc.items():
            texts = []
            for doc in chunks:
                if getattr(doc, "page_content", None):
                    texts.append(doc.page_content)
                meta = getattr(doc, "metadata", {}) or {}
                parent_id = meta.get("parent_id")
                if parent_id:
                    t = get_text_cached(parent_id, index_name, namespace)
                    if t:
                        texts.append(t)
                for sib_id in meta.get("sib_ids", []) or []:
                    t = get_text_cached(sib_id, index_name, namespace)
                    if t:
                        texts.append(t)
            merged = sentaware_compress(texts, max_chars=3500)
            if chunks:
                d0 = chunks[0]
                d0.page_content = merged
                merged_docs.append(d0)

        if not merged_docs:
            return "", []

        # 6) Strict LLM rerank to top N (indices only)
        def _llm_rerank(q: str, docs: List, top_k: int = 6) -> List:
            try:
                items = []
                for i, d in enumerate(docs[:30]):
                    title = (getattr(d, "metadata", {}) or {}).get("title") or (d.id or "")[:80]
                    snippet = (getattr(d, "page_content", "") or "")[:800]
                    items.append(f"[{i}] {title}\n{snippet}")
                prompt = (
                    "Rank these passages by relevance to the query. Return a JSON array of indices only.\n\n"
                    f"Query: {q}\n\nPassages:\n" + "\n\n".join(items) + "\n\nIndices:"
                )
                resp = llm.invoke(prompt)  # reuse main llm (temp=1)
                idx = json.loads(getattr(resp, "content", "[]"))
                ranked = [docs[i] for i in idx if isinstance(i, int) and 0 <= i < len(docs)]
                return ranked[:top_k] if ranked else docs[:top_k]
            except Exception:
                return docs[:6]

        ranked = _llm_rerank(query, merged_docs, top_k=6)

        # 7) Serialize for the agent; keep artifacts for sources
        serialized = "\n||\n".join(
            f"Source: {json.dumps(getattr(d, 'metadata', {}) or {})}\n"
            f"Content: {getattr(d, 'page_content', '')}"
            for d in ranked
        )
        return serialized, ranked

    tools = [current_date, schh_weather, rewrite_query, retrieve]
    agent = create_react_agent(llm.bind(temperature=FORCED_TEMP), tools, checkpointer=memory)
    log.info(f"[build_agent] constructed llm model={model} temp={FORCED_TEMP}")
    return agent

def get_agent(model: str, index_name: str, namespace: Optional[str]):
    with _build_lock:
        return build_agent(model, index_name, namespace)

def retrieve_context(query: str, index_name: str, namespace: Optional[str]) -> Dict[str, Any]:
    dense_q = embeddings.embed_query(query)
    bm25 = get_bm25()
    sparse_q = bm25.encode_queries([query])[0] if bm25 else None

    pine = pc.Index(index_name)
    resp = pine.query(
        vector=dense_q,
        sparse_vector=sparse_q,
        top_k=50,
        include_metadata=True,
        alpha=HYBRID_ALPHA,
        namespace=namespace,
    )
    matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])

    # Wrap matches
    docs = []
    for m in matches or []:
        mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", "")
        md  = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        txt = (md or {}).get("text", "")
        d = type("Doc", (), {})()
        d.id, d.metadata, d.page_content = mid, md, txt
        docs.append(d)

    # Group per base doc, cap 3 chunks/doc
    by_doc = {}
    for d in docs:
        base_id = (d.id or "").rsplit(":", 1)[0]
        if not base_id: 
            continue
        by_doc.setdefault(base_id, [])
        if len(by_doc[base_id]) < 3:
            by_doc[base_id].append(d)

    merged_docs = []
    for base_id, chunks in by_doc.items():
        texts = []
        for doc in chunks:
            if getattr(doc, "page_content", None):
                texts.append(doc.page_content)
            meta = getattr(doc, "metadata", {}) or {}
            pid = meta.get("parent_id")
            if pid:
                t = get_text_cached(pid, index_name, namespace)
                if t: texts.append(t)
            for sid in meta.get("sib_ids", []) or []:
                t = get_text_cached(sid, index_name, namespace)
                if t: texts.append(t)
        merged = sentaware_compress(texts, max_chars=3500)
        if chunks:
            d0 = chunks[0]
            d0.page_content = merged
            merged_docs.append(d0)

    # If nothing found, return empty
    if not merged_docs:
        return {"context_md": "", "sources": []}

    # Simple rerank: keep first 6 (you can keep your LLM rerank if you want)
    top = merged_docs[:6]

    # Build Markdown context + sources list
    ctx_parts, sources = [], []
    for i, d in enumerate(top, 1):
        title = (getattr(d, "metadata", {}) or {}).get("title") or (d.id or "")
        uri   = (getattr(d, "metadata", {}) or {}).get("source")
        ctx_parts.append(f"### [{i}] {title}\n{getattr(d, 'page_content', '')}")
        base_id = (d.id or "").rsplit(":", 1)[0]
        if not any(s.get("doc_id")==base_id for s in sources):
            sources.append({"doc_id": base_id, "title": title, "uri": uri})
    context_md = "\n\n".join(ctx_parts)

    return {"context_md": context_md, "sources": sources}

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

def non_stream_response(agent, messages, config, model: str, index: str, namespace: Optional[str]) -> JSONResponse:
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
        "namespace": namespace,
        "latency_ms": latency
    })

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/chat")
@app.post("/chat/")
async def chat_post(request: Request):
    payload   = await request.json()
    prompt    = payload.get("prompt", "")
    model     = payload.get("model", DEFAULT_MODEL)
    index     = payload.get("index", DEFAULT_INDEX)
    namespace = payload.get("namespace", DEFAULT_NAMESPACE)
    stream    = bool(payload.get("stream", False))
    sessionid = payload.get("sessionid", secrets.token_hex(4))

    # Force retrieval first
    ctx = retrieve_context(prompt, index, namespace)
    context_md = ctx["context_md"]
    sources = ctx["sources"]

    # If no context, we can still answer, but say we found nothing
    preface = (
        "No relevant documents were found in the knowledge base. "
        "If you still answer, be explicit that you're using general knowledge."
        if not context_md else
        "Use ONLY the context below where possible. Cite with [1], [2] matching the section headers."
    )

    system = SYSTEM_PROMPT
    user   = f"{preface}\n\n# User query\n{prompt}\n\n# Context\n{context_md}\n"

    # Call the model directly (no need to rely on tools anymore)
    if model.startswith(("gpt-", "o")):
        llm = ChatOpenAI(model=model, temperature=FORCED_TEMP)
    else:
        llm = ChatAnthropic(model_name=model)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    if stream:
        async def gen():
            # initial heartbeat
            yield sse({"type": "content", "delta": ""})
            async for chunk in llm.astream(messages):
                text = getattr(chunk, "content", None)
                if isinstance(text, list):
                    text = "".join([seg.get("text","") for seg in text if isinstance(seg, dict)])
                if text:
                    yield sse({"type": "content", "delta": text})
            yield sse({"type": "final", "sources": sources})
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        return StreamingResponse(gen(), headers=headers)

    # non-stream
    resp = llm.invoke(messages)
    answer = getattr(resp, "content", "")
    return JSONResponse({"answer": answer, "sources": sources, "model": model, "index": index, "namespace": namespace})
# Optional GET route kept for compatibility
@app.get("/chat/{prompt}")
async def chat_get(prompt: str, request: Request,
                   sessionid: Optional[str] = None,
                   model: Optional[str] = None,
                   index: Optional[str] = None,
                   namespace: Optional[str] = None,
                   stream: Optional[bool] = False):
    model = model or DEFAULT_MODEL
    index = index or DEFAULT_INDEX
    namespace = namespace or DEFAULT_NAMESPACE
    agent = get_agent(model, index, namespace)
    config = {"configurable": {"thread_id": sessionid or secrets.token_hex(4)}}
    messages = build_messages(prompt)
    if stream:
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        return StreamingResponse(stream_response(agent, messages, config), headers=headers)
    else:
        return non_stream_response(agent, messages, config, model, index, namespace)

@app.get("/debug_search")
def debug_search(q: str, index: Optional[str] = None, namespace: Optional[str] = None, k: int = 5):
    idx_name = index or DEFAULT_INDEX
    ns = namespace or DEFAULT_NAMESPACE
    try:
        dq = embeddings.embed_query(q)
        bm = get_bm25()
        sv = bm.encode_queries([q])[0] if bm else None
        resp = pc.Index(idx_name).query(
            vector=dq, sparse_vector=sv, top_k=int(k),
            include_metadata=True, alpha=HYBRID_ALPHA, namespace=ns
        )
        matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
        out = []
        for m in matches:
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", "")
            md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
            out.append({
                "id": mid,
                "title": md.get("title"),
                "has_text": bool(md.get("text")),
                "text_preview": (md.get("text") or "")[:160],
            })
        return {"index": idx_name, "namespace": ns, "alpha": HYBRID_ALPHA, "matches": out}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/diag")
def diag(index: Optional[str] = None, namespace: Optional[str] = None):
    idx_name = index or DEFAULT_INDEX
    ns = namespace or DEFAULT_NAMESPACE
    try:
        stats = pc.Index(idx_name).describe_index_stats()
        # Cherry-pick only JSON-safe parts
        out = {
            "dimension": stats.get("dimension"),
            "namespaces": stats.get("namespaces", {}),
            "total_vector_count": stats.get("total_vector_count"),
        }
        return JSONResponse({"index": idx_name, "namespace": ns, "stats": out})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
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