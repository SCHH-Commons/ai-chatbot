#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agentic Chat Server (SCHH) — Markdown + SSE + heartbeat + hybrid retrieval + namespaces
- Hybrid Pinecone query (dense + optional sparse via BM25 encoder)
- Markdown-oriented system prompt & formatting rules
- SSE streaming with heartbeat, robust token event handling, guaranteed tail
- Session memory via LangGraph MemorySaver (thread_id=sessionid)
- Query rewrite, sentence-aware compression, strict LLM rerank
- OpenAI minis: force temperature=1.0; Anthropic uses FORCED_TEMP
- Configurable output caps and timeouts
"""

import os, re, json, time, secrets, functools, logging, math, asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from threading import RLock
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk

from pinecone import Pinecone
from pinecone.core.openapi.shared.exceptions import PineconeApiException

# Optional: hybrid sparse encoder (BM25)
try:
    import joblib
    from pinecone_text.sparse import BM25Encoder
except Exception:
    joblib = None
    BM25Encoder = None  # type: ignore

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("schh-rag")

DEFAULT_MODEL      = os.getenv("DEFAULT_MODEL", "gpt-5-mini")
DEFAULT_INDEX      = os.getenv("DEFAULT_INDEX", "schh")
DEFAULT_NAMESPACE  = os.getenv("DEFAULT_NAMESPACE", "schh")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "text-embedding-3-small")

ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",")]

# Temps / caps / timeouts
FORCED_TEMP        = float(os.getenv("FORCED_TEMP", "0.6"))   # used by Anthropic only
MAX_OUTPUT_TOKENS  = int(os.getenv("MAX_OUTPUT_TOKENS", "2400"))
REQUEST_TIMEOUT    = int(os.getenv("REQUEST_TIMEOUT", "120"))  # seconds

# Retrieval knobs
HYBRID_ALPHA       = float(os.getenv("HYBRID_ALPHA", "0.5"))
BM25_ENCODER_PATH  = os.getenv("BM25_ENCODER_PATH", "./bm25_encoder_schh.joblib")
SPEED_MODE         = os.getenv("SPEED_MODE", "1") == "1"
RETR_TOP_K         = int(os.getenv("RETR_TOP_K", "30" if SPEED_MODE else "50"))
DOCS_MAX           = int(os.getenv("DOCS_MAX", "4"  if SPEED_MODE else "6"))
CHUNKS_PER_DOC     = int(os.getenv("CHUNKS_PER_DOC", "2" if SPEED_MODE else "3"))
MERGED_MAX_CHARS   = int(os.getenv("MERGED_MAX_CHARS", "2500" if SPEED_MODE else "3500"))
USE_GAP_EXTRACTOR  = os.getenv("USE_GAP_EXTRACTOR", "0" if SPEED_MODE else "1") == "1"
SIBS_MAX           = int(os.getenv("SIBS_MAX", "2"))
MAX_EXTRAS         = int(os.getenv("MAX_EXTRAS", "300"))

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
    "- A short 1–2 lines summary at the top.\n"
    "- Then **Key points** as bullets (2–6 items).\n"
    "- If procedural, add a **Steps** list.\n"
    "- Use callouts: **Note:**, **Tip:**, **Warning:** as bold labels.\n"
    "- Use tables when comparing options.\n"
    "- Keep line length reasonable; add blank lines between sections.\n"
))

# Globals
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
pc = Pinecone()
memory = MemorySaver()

# --- in-process conversation store ---
SESSIONS: Dict[str, deque[dict]] = {}
SESSIONS_LOCK = RLock()
MAX_TURNS = 8

def get_history(sessionid: str) -> deque[dict]:
    with SESSIONS_LOCK:
        if sessionid not in SESSIONS:
            SESSIONS[sessionid] = deque(maxlen=2*MAX_TURNS)
        return SESSIONS[sessionid]

def append_msg(sessionid: str, role: str, content: str) -> None:
    with SESSIONS_LOCK:
        get_history(sessionid).append({"role": role, "content": content})

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@functools.lru_cache(maxsize=2048)
def get_text_cached(chunk_id: str, index_name: str, namespace: Optional[str]) -> Optional[str]:
    try:
        idx = pc.Index(index_name)
        res = idx.fetch(ids=[chunk_id], namespace=namespace)
        vectors = (res or {}).get("vectors", {}) or {}
        v = vectors.get(chunk_id) or {}
        md = v.get("metadata") or {}
        return md.get("text")
    except Exception as e:
        log.warning(f"get_text_cached({chunk_id}) error: {e}")
    return None

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
    A = set(re.findall(r"[A-Za-z0-9]{3,}", a.lower()))
    B = set(re.findall(r"[A-Za-z0-9]{3,}", b.lower()))
    if not A or not B: return 0.0
    return len(A & B) / math.sqrt(len(A) * len(B))

# Text extraction helpers (for non-stream path robustness)
def _to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    content = getattr(x, "content", None)
    if isinstance(content, list):
        return "".join(seg.get("text", "") for seg in content if isinstance(seg, dict))
    if isinstance(content, str):
        return content
    if isinstance(x, dict):
        for k in ("content", "text", "output", "answer"):
            v = x.get(k)
            if isinstance(v, str):
                return v
            if hasattr(v, "content"):
                return getattr(v, "content", "") or ""
    return ""

def _extract_answer_from_result(res) -> str:
    if res is None:
        return ""
    if isinstance(res, dict):
        msgs = res.get("messages")
        if isinstance(msgs, list) and msgs:
            return _to_text(msgs[-1]) or ""
        out = res.get("output") or res.get("response") or res.get("final") or res.get("answer")
        if out:
            return _to_text(out)
        return ""
    if isinstance(res, list) and res:
        return _to_text(res[-1]) or ""
    return _to_text(res)

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
    out: dict[str, str] = {}
    if not ids:
        return out
    idx = pc.Index(index_name)
    batch = 100
    i = 0
    while i < len(ids):
        subset = ids[i:i+batch]
        try:
            res = idx.fetch(ids=subset, namespace=namespace)
            vectors = (res or {}).get("vectors", {}) or {}
            for vid, v in vectors.items():
                md = v.get("metadata") or {}
                txt = md.get("text")
                if txt:
                    out[vid] = txt
            i += batch
        except PineconeApiException:
            if batch > 10:
                batch //= 2
                continue
            break
        except Exception:
            break
    return out

# -----------------------------------------------------------------------------
# LLM factory (OpenAI temp=1.0; Anthropic uses FORCED_TEMP)
# -----------------------------------------------------------------------------
def make_llm(model: str, max_tokens: Optional[int] = None):
    mt = max_tokens or MAX_OUTPUT_TOKENS
    if model.startswith(("gpt-", "o")):  # OpenAI family
        log.info(f"[make_llm] OpenAI model={model} -> temperature=1.0, streaming=True, max_tokens={mt}")
        return ChatOpenAI(
            model=model,
            temperature=1.0,
            streaming=True,
            max_tokens=mt,
            timeout=REQUEST_TIMEOUT,
            max_retries=2,
        )
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set but Anthropic model requested")
    log.info(f"[make_llm] Anthropic model={model} -> temperature={FORCED_TEMP}, streaming=True, max_tokens={mt}")
    return ChatAnthropic(
        model_name=model,
        temperature=FORCED_TEMP,
        streaming=True,
        max_tokens=mt,
        timeout=REQUEST_TIMEOUT,
        max_retries=2,
    )

# -----------------------------------------------------------------------------
# Retrieval pipeline (server-side authoritative)
# -----------------------------------------------------------------------------
def retrieve_context(query: str, index_name: str, namespace: Optional[str]) -> Dict[str, Any]:
    log.info(f"[retrieve_context] q={query!r} index={index_name} ns={namespace}")
    dense_q = embeddings.embed_query(query)
    bm25 = get_bm25()
    sparse_q = bm25.encode_queries([query])[0] if bm25 else None

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

    need_ids = []
    for chunks in by_doc.values():
        for doc in chunks:
            md = getattr(doc, "metadata", {}) or {}
            pid = md.get("parent_id")
            if pid:
                need_ids.append(pid)
            sibs = md.get("sib_ids", []) or []
            if SIBS_MAX > 0 and sibs:
                need_ids.extend(sibs[:SIBS_MAX])
    need_ids = list(dict.fromkeys(need_ids))[:MAX_EXTRAS]
    try:
        extras = fetch_texts_batch(index_name, namespace, need_ids)
    except Exception:
        extras = {}

    merged_docs = []
    for _, chunks in by_doc.items():
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

    # MMR-lite selection
    selected = []
    for d in merged_docs:
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

    # Build Markdown context + sources
    ctx_parts, sources = [], []
    for i, d in enumerate(top, 1):
        meta  = getattr(d, "metadata", {}) or {}
        raw_title = (meta.get("title") or d.id or "")
        cleaned   = raw_title.split("-")[-1].replace(".md","").replace("_"," ")
        title     = (cleaned.strip() or raw_title)
        uri   = meta.get("source")
        ctx_parts.append(f"### [{i}] {getattr(d, 'page_content', '')}")
        base = (d.id or "").rsplit(":", 1)[0]
        if not any(s.get("doc_id")==base for s in sources):
            sources.append({"doc_id": base, "title": title, "uri": uri})
    context_md = "\n\n".join(ctx_parts)
    return {"context_md": context_md, "sources": sources}

# -----------------------------------------------------------------------------
# Agent builder (cached per (model,index,namespace,include_retrieve,max_tokens))
# -----------------------------------------------------------------------------
_build_lock = RLock()

@functools.lru_cache(maxsize=32)
def build_agent(model: str, index_name: str, namespace: Optional[str],
                include_retrieve: bool, max_tokens: int):
    log.info(f"[build_agent] model={model} index={index_name} ns={namespace} include_retrieve={include_retrieve} max_tokens={max_tokens}")
    llm = make_llm(model, max_tokens=max_tokens)

    @tool
    def current_date() -> str:
        """Get current date (YYYY-MM-DD)."""
        return datetime.today().strftime("%Y-%m-%d")

    @tool
    def schh_weather() -> str:
        """Best-effort weather for Sun City Hilton Head."""
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            tv = TavilySearchResults(max_results=3, search_depth="advanced", include_answer=True)
            res = tv.invoke({"query": "Weather forecast Sun City Hilton Head SC"})
            return json.dumps(res, ensure_ascii=False)[:4000]
        except Exception as e:
            return f"Weather lookup failed: {e}"

    @tool
    def rewrite_query(q: str) -> str:
        """Rewrite/expand a user query for retrieval; <30 words, one line."""
        try:
            prompt = (
                "Rewrite this for retrieval: add synonyms, expand abbreviations, and include likely variants. "
                "Return one line, <30 words, no explanations.\n\nQuery: " + q
            )
            resp = llm.invoke(prompt)
            return getattr(resp, "content", "")[:400]
        except Exception:
            return q

    tools = [current_date, schh_weather, rewrite_query]

    if include_retrieve:
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Hybrid Pinecone retrieval (agent tool variant)."""
            dense_q = embeddings.embed_query(query)
            bm25 = get_bm25()
            sparse_q = bm25.encode_queries([query])[0] if bm25 else None

            pine = pc.Index(index_name)
            try:
                resp = pine.query(
                    vector=dense_q,
                    sparse_vector=sparse_q,
                    top_k=50,
                    include_metadata=True,
                    alpha=HYBRID_ALPHA,
                    namespace=namespace,
                )
                matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
            except Exception as e:
                log.warning(f"[retrieve] pinecone query failed: {e}")
                return "", []

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

            by_doc: Dict[str, List] = {}
            for d in docs:
                base_id = (d.id or "").rsplit(":", 1)[0]
                if not base_id: continue
                by_doc.setdefault(base_id, [])
                if len(by_doc[base_id]) < 3:
                    by_doc[base_id].append(d)

            merged_docs = []
            for _, chunks in by_doc.items():
                texts = []
                for doc in chunks:
                    if getattr(doc, "page_content", None):
                        texts.append(doc.page_content)
                    meta = getattr(doc, "metadata", {}) or {}
                    parent_id = meta.get("parent_id")
                    if parent_id:
                        t = get_text_cached(parent_id, index_name, namespace)
                        if t: texts.append(t)
                    for sib_id in meta.get("sib_ids", []) or []:
                        t = get_text_cached(sib_id, index_name, namespace)
                        if t: texts.append(t)
                merged = sentaware_compress(texts, max_chars=3500)
                if chunks:
                    d0 = chunks[0]
                    d0.page_content = merged
                    merged_docs.append(d0)

            if not merged_docs:
                return "", []

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
                    resp = llm.invoke(prompt)
                    idx = json.loads(getattr(resp, "content", "[]"))
                    ranked = [docs[i] for i in idx if isinstance(i, int) and 0 <= i < len(docs)]
                    return ranked[:top_k] if ranked else docs[:top_k]
                except Exception:
                    return docs[:6]

            ranked = _llm_rerank(query, merged_docs, top_k=6)
            serialized = "\n||\n".join(
                f"Source: {json.dumps(getattr(d, 'metadata', {}) or {})}\n"
                f"Content: {getattr(d, 'page_content', '')}"
                for d in ranked
            )
            return serialized, ranked

        tools.append(retrieve)

    agent = create_react_agent(llm, tools, checkpointer=memory)
    log.info(f"[build_agent] constructed llm model={model} (OpenAI temp=1.0; Anthropic temp={FORCED_TEMP})")
    return agent

def get_agent(model: str, index_name: str, namespace: Optional[str],
              include_retrieve: bool, max_tokens: int):
    with _build_lock:
        return build_agent(model, index_name, namespace, include_retrieve, max_tokens)

# -----------------------------------------------------------------------------
# Message builders (BaseMessage objects)
# -----------------------------------------------------------------------------
def build_messages(user_prompt: str):
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

def build_messages_with_history(sessionid: str, system_prompt: str, current_user_msg: str):
    msgs = [SystemMessage(content=system_prompt)]
    prior = list(get_history(sessionid))
    if len(prior) > 2 * MAX_TURNS:
        prior = prior[-2 * MAX_TURNS:]
    for m in prior:
        role = m.get("role"); content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=current_user_msg))
    return msgs

# -----------------------------------------------------------------------------
# Streaming (event-level + fallbacks, guaranteed tail)
# -----------------------------------------------------------------------------
async def stream_response(agent,
                          messages,
                          config,
                          base_sources: Optional[List[Dict[str, Any]]] = None,
                          sessionid: Optional[str] = None,
                          model: Optional[str] = None):
    sources: List[Dict[str, Any]] = list(base_sources or [])
    final_text_parts: List[str] = []
    yielded_any = False

    # heartbeat
    yield sse({"type": "content", "delta": ""})

    # Primary: event-level streaming
    try:
        async for ev in agent.astream_events({"messages": messages}, config):
            et = ev.get("event"); data = ev.get("data") or {}

            if et == "on_chat_model_stream":
                chunk = data.get("chunk")
                delta = getattr(chunk, "content", "") if chunk else ""
                if isinstance(delta, list):
                    delta = "".join(seg.get("text", "") for seg in delta if isinstance(seg, dict))
                if delta:
                    yielded_any = True
                    final_text_parts.append(delta)
                    yield sse({"type": "content", "delta": delta})

            elif et == "on_llm_stream":
                chunk = data.get("chunk")
                delta = getattr(chunk, "content", "") if hasattr(chunk, "content") else (chunk or "")
                if isinstance(delta, list):
                    delta = "".join(seg.get("text", "") for seg in delta if isinstance(seg, dict))
                if delta:
                    yielded_any = True
                    final_text_parts.append(delta)
                    yield sse({"type": "content", "delta": delta})

            elif et in ("on_chat_model_end", "on_chain_end"):
                text_candidates: List[str] = []
                gens = data.get("generations")
                if gens and isinstance(gens, list):
                    for g in gens:
                        seq = g if isinstance(g, list) else [g]
                        for gg in seq:
                            if isinstance(gg, dict):
                                if gg.get("text"):
                                    text_candidates.append(gg["text"])
                                elif "message" in gg and getattr(gg["message"], "content", ""):
                                    text_candidates.append(gg["message"].content)
                out = data.get("output") or data.get("response")
                if hasattr(out, "content"):
                    text_candidates.append(getattr(out, "content", "") or "")
                elif isinstance(out, dict):
                    for k in ("content", "output", "text"):
                        if out.get(k):
                            text_candidates.append(str(out[k]))
                final_tail = "".join(tc for tc in text_candidates if tc)
                if final_tail:
                    yielded_any = True
                    final_text_parts.append(final_tail)
                    yield sse({"type": "content", "delta": final_tail})

            elif et == "on_tool_end":
                output = data.get("output") or ""
                try:
                    docs = str(output).split("\n||\n")
                    meta_strs = [re.sub(r"^Source:\s*", "", d.split("\n")[0]) for d in docs if d.strip()]
                    for m in meta_strs:
                        rec = json.loads(m)
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
    except Exception as e:
        yield sse({"type": "error", "message": str(e)})

    # Fallback 1: message-level streaming
    if not yielded_any:
        try:
            async for msg, meta in agent.astream({"messages": messages}, config, stream_mode="messages"):
                if isinstance(msg, AIMessageChunk):
                    delta = msg.content if isinstance(msg.content, str) else "".join(
                        seg.get("text", "") for seg in (msg.content or []) if isinstance(seg, dict)
                    )
                    if delta:
                        yielded_any = True
                        final_text_parts.append(delta)
                        yield sse({"type": "content", "delta": delta})
        except Exception:
            pass

    # Fallback 2: raw LLM streaming (bypass agent)
    if not yielded_any and model:
        try:
            raw_llm = make_llm(model, max_tokens=MAX_OUTPUT_TOKENS)
            async for chunk in raw_llm.astream(messages):
                text = getattr(chunk, "content", "")
                if isinstance(text, list):
                    text = "".join(seg.get("text", "") for seg in text if isinstance(seg, dict))
                if text:
                    yielded_any = True
                    final_text_parts.append(text)
                    yield sse({"type": "content", "delta": text})
        except Exception:
            pass

    # Fallback 3: one-shot invoke
    if not yielded_any:
        try:
            res = agent.invoke({"messages": messages}, config)
            text = _extract_answer_from_result(res)
            if text:
                final_text_parts.append(text)
                yield sse({"type": "content", "delta": text})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    # persist
    if sessionid:
        try:
            append_msg(sessionid, "assistant", "".join(final_text_parts))
        except Exception:
            pass

    yield sse({"type": "final", "sources": sources})

# -----------------------------------------------------------------------------
# Non-stream response (robust extraction + fallback)
# -----------------------------------------------------------------------------
def non_stream_response(agent, messages, config, model: str, index: str, namespace: Optional[str],
                        base_sources: Optional[List[Dict[str, Any]]] = None) -> JSONResponse:
    t0 = time.time()
    try:
        result = agent.invoke({"messages": messages}, config)
    except Exception as e:
        log.warning(f"[non_stream] agent.invoke failed: {e}; falling back to raw LLM")
        raw = make_llm(model)
        result = raw.invoke(messages)

    answer = _extract_answer_from_result(result)

    if not answer.strip():
        log.info("[non_stream] empty answer from agent; falling back to raw LLM invoke")
        raw = make_llm(model)
        try:
            raw_res = raw.invoke(messages)
            answer = _to_text(raw_res) or answer
        except Exception as e:
            log.warning(f"[non_stream] raw LLM invoke failed: {e}")

    latency = int((time.time() - t0) * 1000)

    # merge sources (precomputed + any tool-derived)
    sources: List[Dict[str, Any]] = list(base_sources or [])
    try:
        inter = result.get("messages", []) if isinstance(result, dict) else []
        for m in inter:
            docs = (_to_text(m) or "").split("\n||\n")
            meta_strs = [re.sub(r"^Source:\s*", "", d.split("\n")[0]) for d in docs if d.strip()]
            for ms in meta_strs:
                try:
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
                    continue
    except Exception:
        pass

    return JSONResponse({
        "answer": answer or "",
        "sources": sources,
        "model": model,
        "index": index,
        "namespace": namespace,
        "latency_ms": latency
    })

# -----------------------------------------------------------------------------
# FastAPI app and routes
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

# SSE test to isolate proxy/buffering issues
@app.get("/sse_test")
async def sse_test():
    async def gen():
        for _ in range(1000):  # ~500KB total
            yield sse({"type":"content","delta":"."*500})
            await asyncio.sleep(0.005)
        yield sse({"type":"final","sources":[]})
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), headers=headers)

# Debug endpoints
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
        out = {
            "dimension": stats.get("dimension"),
            "namespaces": stats.get("namespaces", {}),
            "total_vector_count": stats.get("total_vector_count"),
        }
        return JSONResponse({"index": idx_name, "namespace": ns, "stats": out})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
# Chat endpoints
# -----------------------------------------------------------------------------
def build_user_msg_with_context(prompt: str, context_md: str, gaps: List[str]) -> str:
    answering_rules = """
# Answering Rules (Blend Mode)
1) The SCHH/context is the source of truth for local rules, procedures, contacts, and exceptions.
2) You MAY add general best-practice/background items only to fill gaps. Prefix each such line with "(General)".
3) If general guidance conflicts with SCHH/context, omit it. If relevant, say: "SCHH guidance overrides general guidance."
4) Cite ONLY SCHH/context items using bracketed numbers [1], [2] that map to the provided context sections.
5) Do NOT include a separate 'Sources' section; the app will render sources from metadata.
6) If the SCHH/context lacks key info, say what's missing, then add targeted (General) items.
7) Do NOT invent local names, phone numbers, or rules; include those only if present in SCHH/context.
8) Structure: SCHH-specific guidance (with citations) first, then "(General) additions)" (no citations).
9) Tools available: current_date, schh_weather, rewrite_query. Do NOT perform retrieval; the SCHH/Context is authoritative for this answer.
"""
    tone_rules = f"# Tone\nWrite with a {TONE} tone. {VOICE_RULES}"
    style_rules = f"# Formatting\n{STYLE_RULES}"
    gap_hint = f"\nFocus (General) additions on these unmet needs: {', '.join(gaps)}\n" if gaps else ""
    return (
        f"{tone_rules}\n{style_rules}\n{answering_rules}\n{gap_hint}"
        f"# User query\n{prompt}\n\n"
        f"# SCHH/Context Sections\n{context_md}\n"
        f"# Output format\n"
        f"- Start with 1–2 lines short summary (don't use a shot summary title)\n"
        f"- Then **Key points** (bullets)\n"
        f"- Add **Steps** if procedural\n"
        f"- Use inline citations like [1], [2] where relevant\n"
        f"- Do **not** include a 'Sources' section (the app shows sources separately)\n"
    )

def extract_gaps_for_general(prompt: str, context_md: str, llm, max_chars: int = 2500) -> list[str]:
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
    mt        = int(payload.get("max_tokens", MAX_OUTPUT_TOKENS))
    mt        = max(256, min(mt, 4000))

    # Retrieval (authoritative)
    ctx = retrieve_context(prompt, index, namespace)
    context_md, sources = ctx["context_md"], ctx["sources"]

    # Gap extraction (small cap ok)
    llm_for_gaps = make_llm(model, max_tokens=512)
    gaps = extract_gaps_for_general(prompt, context_md, llm_for_gaps) if (context_md and USE_GAP_EXTRACTOR) else []

    user_msg_current = build_user_msg_with_context(prompt, context_md, gaps)

    messages = build_messages_with_history(sessionid, SYSTEM_PROMPT, user_msg_current)
    append_msg(sessionid, "user", prompt)

    # Agent WITHOUT retrieve (POST uses server-side context)
    agent = get_agent(model, index, namespace, include_retrieve=False, max_tokens=mt)
    config = {"configurable": {"thread_id": sessionid}}

    if stream:
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            stream_response(agent, messages, config, base_sources=sources, sessionid=sessionid, model=model),
            headers=headers
        )

    # non-stream path
    return non_stream_response(agent, messages, config, model, index, namespace, base_sources=sources)

# GET route (agent WITH retrieve tool)
@app.get("/chat/{prompt}")
async def chat_get(prompt: str, request: Request,
                   sessionid: Optional[str] = None,
                   model: Optional[str] = None,
                   index: Optional[str] = None,
                   namespace: Optional[str] = None,
                   stream: Optional[bool] = False,
                   max_tokens: Optional[int] = None):
    model = model or DEFAULT_MODEL
    index = index or DEFAULT_INDEX
    namespace = namespace or DEFAULT_NAMESPACE
    mt = max_tokens or MAX_OUTPUT_TOKENS

    agent = get_agent(model, index, namespace, include_retrieve=True, max_tokens=mt)
    config = {"configurable": {"thread_id": sessionid or secrets.token_hex(4)}}
    messages = build_messages(prompt)

    if stream:
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            stream_response(agent, messages, config, base_sources=[], sessionid=sessionid or None, model=model),
            headers=headers
        )
    else:
        return non_stream_response(agent, messages, config, model, index, namespace, base_sources=[])

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        build_agent.cache_clear()
    except Exception:
        pass
    import uvicorn
    # In production, prefer running via process manager and avoid reload to prevent stream cuts.
    uvicorn.run('serve:app', host="0.0.0.0", port=8080, reload=False)
