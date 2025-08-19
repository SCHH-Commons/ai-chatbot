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

from collections import deque
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
from pinecone.core.openapi.shared.exceptions import PineconeApiException

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
DEFAULT_NAMESPACE  = os.getenv("DEFAULT_NAMESPACE", 'schh') 
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

# Personality / formatting knobs
TONE = os.getenv("TONE", "warm, succinct, community-friendly, helpful but not chatty")
VOICE_RULES = os.getenv("VOICE_RULES", (
    "Be friendly and clear. Use plain language. Avoid fluff. "
    "Use a light touch of personality (one short friendly line or emoji when appropriate), "
    "but keep the focus on helpful content."
))
STYLE_RULES = os.getenv("STYLE_RULES", (
    "Always format in Markdown with:\n"
    "- A short **TL;DR** (1–2 lines) at the top.\n"
    "- Then **Key points** as bullets (2–6 items).\n"
    "- If procedural, add a **Steps** list.\n"
    "- Use callouts: **Note:**, **Tip:**, **Warning:** as bold labels.\n"
    "- Use tables when comparing options.\n"
    "- Keep line length reasonable; add blank lines between sections.\n"
))

SPEED_MODE = os.getenv("SPEED_MODE", "1") == "1"  # default fast
RETR_TOP_K = int(os.getenv("RETR_TOP_K", "30" if SPEED_MODE else "50"))
DOCS_MAX   = int(os.getenv("DOCS_MAX", "4"  if SPEED_MODE else "6"))
CHUNKS_PER_DOC = int(os.getenv("CHUNKS_PER_DOC", "2" if SPEED_MODE else "3"))
MERGED_MAX_CHARS = int(os.getenv("MERGED_MAX_CHARS", "2500" if SPEED_MODE else "3500"))
USE_GAP_EXTRACTOR = os.getenv("USE_GAP_EXTRACTOR", "0" if SPEED_MODE else "1") == "1"
SIBS_MAX = int(os.getenv("SIBS_MAX", "2"))           # siblings per hit
MAX_EXTRAS = int(os.getenv("MAX_EXTRAS", "300"))     # global cap per request

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
pc = Pinecone()
memory = MemorySaver()

# --- simple in‑process conversation store ---
SESSIONS: dict[str, deque[dict]] = {}
SESSIONS_LOCK = RLock()
MAX_TURNS = 8  # number of prior user/assistant turns to keep per session

def get_history(sessionid: str) -> deque[dict]:
    with SESSIONS_LOCK:
        if sessionid not in SESSIONS:
            SESSIONS[sessionid] = deque(maxlen=2*MAX_TURNS)  # each turn ~2 msgs
        return SESSIONS[sessionid]

def append_msg(sessionid: str, role: str, content: str) -> None:
    with SESSIONS_LOCK:
        get_history(sessionid).append({"role": role, "content": content})

def build_messages_with_history(sessionid: str, system_prompt: str, current_user_msg: str) -> list[dict]:
    """
    Returns: [system] + prior history (user/assistant pairs) + [current user]
    We only inject the RAG context into *current_user_msg* to avoid polluting history.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    prior = list(get_history(sessionid))
    # Optionally clamp in case someone changed MAX_TURNS
    if len(prior) > 2*MAX_TURNS:
        prior = prior[-2*MAX_TURNS:]
    msgs.extend(prior)
    msgs.append({"role": "user", "content": current_user_msg})
    return msgs

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

def fetch_texts_batch(index_name: str, namespace: Optional[str], ids: list[str]) -> dict[str, str]:
    """Fetch metadata.text for ids using small batches; degrade gracefully on errors."""
    out: dict[str, str] = {}
    if not ids:
        return out
    idx = pc.Index(index_name)

    # conservative batch size: 100 ids to avoid URL bloat (fetch is GET under the hood)
    batch = 100
    i = 0
    while i < len(ids):
        subset = ids[i:i+batch]
        try:
            res = idx.fetch(ids=subset, namespace=namespace)
            vectors = (res or {}).get("vectors", {}) if isinstance(res, dict) else getattr(res, "vectors", {})
            for vid, v in (vectors or {}).items():
                md = v.get("metadata") if isinstance(v, dict) else getattr(v, "metadata", {}) or {}
                txt = (md or {}).get("text")
                if txt:
                    out[vid] = txt
            i += batch
        except PineconeApiException as e:
            # If we hit 414 or similar, halve the batch and retry; if too small, give up gracefully
            if getattr(e, "status", None) == 414 and batch > 10:
                batch //= 2
                continue
            # Non-recoverable or still failing with tiny batches — return what we have
            break
        except Exception:
            # Don’t fail the whole request if fetch struggles; just return partials
            break
    return out

# -----------------------------------------------------------------------------
# Agent builder (cached per (model,index,namespace))
# -----------------------------------------------------------------------------
_build_lock = RLock()

@functools.lru_cache(maxsize=24)
def build_agent(model: str, index_name: str, namespace: Optional[str]):
    log.info(f"[build_agent] model={model} index={index_name} ns={namespace}")

    # Main LLM
    if model.startswith(("gpt-", "o")):
        llm = ChatOpenAI(model=model, temperature=FORCED_TEMP, max_tokens=600)  # tune 400–800

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
    # 1) Encode
    dense_q = embeddings.embed_query(query)
    bm25 = get_bm25()
    sparse_q = bm25.encode_queries([query])[0] if bm25 else None

    # 2) Hybrid query
    pine = pc.Index(index_name)
    resp = pine.query(
        vector=dense_q,
        sparse_vector=sparse_q,
        top_k=RETR_TOP_K,
        include_metadata=True,
        alpha=HYBRID_ALPHA,
        namespace=namespace,
    )
    matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])

    # 3) Wrap + group chunks by base doc, cap CHUNKS_PER_DOC
    docs = []
    for m in matches or []:
        mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", "")
        md  = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        txt = (md or {}).get("text", "")
        d = type("Doc", (), {})()
        d.id, d.metadata, d.page_content = mid, md, txt
        docs.append(d)

    by_doc: Dict[str, List] = {}
    for d in docs:
        base_id = (d.id or "").rsplit(":", 1)[0]
        if not base_id:
            continue
        by_doc.setdefault(base_id, [])
        if len(by_doc[base_id]) < CHUNKS_PER_DOC:
            by_doc[base_id].append(d)

    # 4) Batch collect parent/sibling ids (LIMITED)
    need_ids = []
    for chunks in by_doc.values():
        for doc in chunks:
            md = getattr(doc, "metadata", {}) or {}
            pid = md.get("parent_id")
            if pid:
                need_ids.append(pid)
            sibs = md.get("sib_ids", []) or []
            if SIBS_MAX > 0 and sibs:
                need_ids.extend(sibs[:SIBS_MAX])  # only take a few
    # de-dupe and cap
    need_ids = list(dict.fromkeys(need_ids))[:MAX_EXTRAS]
    
    try:
        extras = fetch_texts_batch(index_name, namespace, need_ids)
    except Exception:
        extras = {}

    # 5) Merge + sentence-aware compress
    merged_docs = []
    for base_id, chunks in by_doc.items():
        texts = []
        for doc in chunks:
            if getattr(doc, "page_content", None):
                texts.append(doc.page_content)
            md = getattr(doc, "metadata", {}) or {}
            pid = md.get("parent_id")
            if pid and pid in extras:
                texts.append(extras[pid])
            for sid in md.get("sib_ids", []) or []:
                if sid in extras:
                    texts.append(extras[sid])
        merged = sentaware_compress(texts, max_chars=MERGED_MAX_CHARS)
        if chunks:
            d0 = chunks[0]
            d0.page_content = merged
            merged_docs.append(d0)

    if not merged_docs:
        return {"context_md": "", "sources": []}

    # 6) MMR‑lite doc selection (no LLM re‑rank)
    selected = []
    seen = set()
    for d in merged_docs:
        # Favor diversity by skipping near-duplicates
        keep = True
        for s in selected:
            if tokenish_overlap(d.page_content, s.page_content) > 0.7:
                keep = False
                break
        if keep:
            selected.append(d)
        if len(selected) >= DOCS_MAX:
            break

    top = selected or merged_docs[:DOCS_MAX]

    # 7) Build Markdown context + sources
    ctx_parts, sources = [], []
    for i, d in enumerate(top, 1):
        meta  = getattr(d, "metadata", {}) or {}
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        title = ' '.join(meta.get("title") or (d.id or []))
        uri   = meta.get("source")
        title = ''.join(meta.get("header_path", [])) or title
        ctx_parts.append(f"### [{i}] {title}\n{getattr(d, 'page_content', '')}")
        base = (d.id or "").rsplit(":", 1)[0]
        if not any(s.get("doc_id")==base for s in sources):
            sources.append({"doc_id": base, "title": title, "uri": uri})
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

def extract_gaps_for_general(prompt: str, context_md: str, llm, max_chars: int = 2500) -> list[str]:
    """Ask the LLM to list missing needs so general guidance can target gaps."""
    gap_prompt = (
        "List 3–6 short items the answer would need that are NOT present in the provided context. "
        "Return a JSON array of short noun-phrases or questions. Do not repeat items covered by the context.\n\n"
        f"User query:\n{prompt}\n\nContext (truncated):\n{context_md[:max_chars]}"
    )
    try:
        resp = llm.invoke([{"role":"user","content":gap_prompt}])
        items = json.loads(getattr(resp, "content", "[]"))
        if isinstance(items, list):
            return [str(x)[:120] for x in items][:6]
    except Exception:
        pass
    return []

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

    # 1) Force retrieval first
    ctx = retrieve_context(prompt, index, namespace)
    context_md, sources = ctx["context_md"], ctx["sources"]

    # 2) Choose LLM
    if model.startswith(("gpt-", "o")):
        llm = ChatOpenAI(model=model, temperature=1)
    else:
        llm = ChatAnthropic(model_name=model)

    # 3) Optional: gap targeting for (General) items
    gaps = extract_gaps_for_general(prompt, context_md, llm) if (context_md and USE_GAP_EXTRACTOR) else []
    
    answering_rules = """
    # Answering Rules (Blend Mode)
    1) The SCHH/context is the source of truth for local rules, procedures, contacts, and exceptions.
    2) You MAY add general best-practice/background items only to fill gaps. Prefix each such line with "(General)".
    3) If general guidance conflicts with SCHH/context, omit it. If relevant, say: "SCHH guidance overrides general guidance."
    4) Cite ONLY SCHH/context items using bracketed numbers [1], [2] that map to the provided context sections.
    5) Do NOT include a separate 'Sources' or 'References' section; the app will render sources from metadata.
    6) If the SCHH/context lacks key info, say what's missing, then add targeted (General) items.
    7) Do NOT invent local names, phone numbers, or rules; include those only if present in SCHH/context.
    8) Structure: SCHH-specific guidance (with citations) first, then "(General) additions)" (no citations).
    """

    tone_rules = f"# Tone\nWrite with a {TONE} tone. {VOICE_RULES}"
    style_rules = f"# Formatting\n{STYLE_RULES}"

    gap_hint = f"\nFocus (General) additions on these unmet needs: {', '.join(gaps)}\n" if gaps else ""
    preface = (
        "No relevant SCHH/context was found. If you answer, explicitly note you're using general knowledge only."
        if not context_md else
        "Use the SCHH/context where available. Add (General) items only for gaps."
)

    # 4) Build the *current* user message with RAG context
    user_msg_current = (
        f"{tone_rules}\n{style_rules}\n"
        f"{answering_rules}\n"
        f"{gap_hint}"
        f"# User query\n{prompt}\n\n"
        f"# SCHH/Context Sections\n{context_md}\n"
        f"# Output format\n"
        f"- Start with **TL;DR** (1–2 lines)\n"
        f"- Then **Key points** (bullets)\n"
        f"- Add **Steps** if procedural\n"
        f"- Use inline citations like [1], [2] where relevant\n"
        f"- Do **not** include a 'Sources' section (the app shows sources separately)\n"
    )

    # 5) Build messages with *prior* history for this sessionid
    messages = build_messages_with_history(sessionid, SYSTEM_PROMPT, user_msg_current)

    # 6) Append the raw user prompt (not the big injected one) to history
    append_msg(sessionid, "user", prompt)

    if stream:
        async def gen():
            # initial heartbeat
            # yield sse({"type": "content", "delta": ""})
            final_text_parts = []
            async for chunk in llm.astream(messages):
                text = getattr(chunk, "content", None)
                if isinstance(text, list):
                    text = "".join([seg.get("text","") for seg in text if isinstance(seg, dict)])
                if text:
                    final_text_parts.append(text)
                    yield sse({"type": "content", "delta": text})
            final_text = "".join(final_text_parts)
            # store assistant reply to session history
            append_msg(sessionid, "assistant", final_text)
            yield sse({"type": "final", "sources": sources})
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        return StreamingResponse(gen(), headers=headers)

    # non-stream
    resp = llm.invoke(messages)
    answer = getattr(resp, "content", "")
    append_msg(sessionid, "assistant", answer)
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