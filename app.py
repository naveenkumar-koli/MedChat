"""
app.py — RAG Chatbot API Server (FastAPI + Ollama + FAISS HNSW)
================================================================
100% Open-Source Stack:
  • Embedding:  sentence-transformers/all-MiniLM-L6-v2  (Apache 2.0)
  • Vector DB:  FAISS HNSW                              (MIT)
  • Reranker:   cross-encoder/ms-marco-MiniLM-L-6-v2   (Apache 2.0)
  • LLM:        Ollama → llama3.2:3b                    (Meta Llama 3 Community)

Pipeline:
  Query → Embed → HNSW Retrieve (top-10)
        → Cross-Encoder Rerank (top-5)
        → Llama 3.2 3B via Ollama (streaming)
        → Return answer + citations + timing

Endpoints:
  GET  /                     Chat UI (Jinja2)
  POST /api/ask              Main RAG query
  POST /api/ingest           Upload & ingest a new PDF
  GET  /api/metrics          Latency / eval stats
  GET  /api/documents        List indexed PDFs
  GET  /api/health           Health check + Ollama status
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from collections import deque
from statistics import median, quantiles
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import faiss
import ollama as ollama_client
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_INDEX_PATH  = "vectorstore/db_faiss_hnsw.index"
FAISS_LEGACY_PATH = "vectorstore/db_faiss"
METADATA_PATH     = "vectorstore/metadata.json"
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOP_K_RETRIEVE    = 6
TOP_K_RERANK      = 3
HNSW_EF_SEARCH    = 64
DATA_PATH         = Path("data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics_lock = threading.Lock()
_latencies    = deque(maxlen=1000)
_query_count  = 0
_error_count  = 0

def record_latency(lat: float):
    global _query_count
    with _metrics_lock:
        _latencies.append(lat)
        _query_count += 1

def get_metrics() -> dict:
    with _metrics_lock:
        lats = list(_latencies)
    if not lats:
        return {"query_count": _query_count, "error_count": _error_count,
                "latency": {}, "note": "No queries yet"}
    sl = sorted(lats)
    p95 = quantiles(sl, n=20)[18] if len(sl) >= 20 else max(sl)
    return {
        "query_count": _query_count,
        "error_count": _error_count,
        "latency": {
            "p50_s":  round(median(sl), 3),
            "p95_s":  round(p95, 3),
            "min_s":  round(min(sl), 3),
            "max_s":  round(max(sl), 3),
            "mean_s": round(sum(sl) / len(sl), 3),
        },
        "llm": OLLAMA_MODEL,
        "target_latency_s": "2–5",
        "within_target": p95 <= 5.0,
    }

# ── RAG Engine ────────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self):
        log.info("Initializing RAG Engine (Ollama + FAISS HNSW)...")

        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        log.info(f"Loading cross-encoder reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        self.index, self.metadata = self._load_index()
        self._verify_ollama()
        log.info(f"RAG Engine ready — {self.index.ntotal} vectors, LLM: {OLLAMA_MODEL}")

    def _verify_ollama(self):
        """Check Ollama is running and model is available."""
        try:
            client = ollama_client.Client(host=OLLAMA_HOST)
            models = [m.model for m in client.list().models]
            if OLLAMA_MODEL not in models and not any(OLLAMA_MODEL in m for m in models):
                log.warning(f"Model '{OLLAMA_MODEL}' not found in Ollama. Pulling now...")
                client.pull(OLLAMA_MODEL)
                log.info(f"✅ Pulled {OLLAMA_MODEL}")
            else:
                log.info(f"✅ Ollama model ready: {OLLAMA_MODEL}")
        except Exception as e:
            log.error(f"⚠️  Ollama not reachable at {OLLAMA_HOST}: {e}")
            log.error("Start Ollama with: ollama serve   then:  ollama pull llama3.2:3b")

    def _load_index(self):
        if Path(FAISS_INDEX_PATH).exists() and Path(METADATA_PATH).exists():
            log.info(f"Loading HNSW index: {FAISS_INDEX_PATH}")
            index = faiss.read_index(FAISS_INDEX_PATH)
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = HNSW_EF_SEARCH
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return index, metadata

        if Path(FAISS_LEGACY_PATH).exists():
            log.warning("HNSW index not found — loading legacy flat FAISS index")
            from langchain_community.vectorstores import FAISS as LCFaiss
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            lc_emb = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            lc_db = LCFaiss.load_local(
                FAISS_LEGACY_PATH, lc_emb, allow_dangerous_deserialization=True
            )
            self._lc_db = lc_db
            self._legacy_mode = True
            return lc_db.index, []

        raise FileNotFoundError(
            "No vector index found. Run `python ingest.py` first."
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        qvec = self.embed_query(query)
        if getattr(self, "_legacy_mode", False):
            docs = self._lc_db.similarity_search_with_score(query, k=top_k)
            return [{
                "text": d.page_content,
                "metadata": {
                    "filename": d.metadata.get("source", "unknown.pdf"),
                    "page_number": d.metadata.get("page", "?"),
                    "pdf_id": "legacy", "chunk_index": 0,
                    "language": "en", "bbox": None,
                },
                "ann_score": float(s),
            } for d, s in docs]

        scores, indices = self.index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            results.append({
                "text": m.get("text", ""),
                "metadata": {
                    "filename":    m.get("filename", "unknown.pdf"),
                    "page_number": m.get("page_number", "?"),
                    "pdf_id":      m.get("pdf_id", ""),
                    "chunk_index": m.get("chunk_index", 0),
                    "token_count": m.get("token_count", 0),
                    "language":    m.get("language", "en"),
                    "bbox":        m.get("bbox", None),
                },
                "ann_score": float(score),
            })
        return results

    def rerank(self, query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            m = chunk["metadata"]
            # Truncate each chunk to ~200 tokens worth of chars for CPU speed
            text = chunk["text"][:800]
            context_parts.append(
                f"[{i}] {m['filename']} p.{m['page_number']}:\n{text}"
            )
        context = "\n\n".join(context_parts)
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a medical AI assistant. Answer using ONLY the document excerpts below. Cite sources as [filename, p.X].
<|eot_id|><|start_header_id|>user<|end_header_id|>
DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

Give a concise answer with citations. End with 2 follow-up questions and the disclaimer.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def generate(self, query: str) -> dict:
        t0 = time.time()

        # Stage 1: Retrieve
        candidates = self.retrieve(query, top_k=TOP_K_RETRIEVE)
        t_retrieve = round(time.time() - t0, 3)

        # Stage 2: Rerank
        top_chunks = self.rerank(query, candidates, top_k=TOP_K_RERANK)
        t_rerank = round(time.time() - t0 - t_retrieve, 3)

        # Stage 3: Generate via Ollama
        prompt = self.build_prompt(query, top_chunks)
        client = ollama_client.Client(host=OLLAMA_HOST)
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.2,
                "num_predict": 512,
                "num_ctx": 2048,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["<|eot_id|>"],
            }
        )
        answer = response.response.strip()
        t_generate = round(time.time() - t0 - t_retrieve - t_rerank, 3)
        total = round(time.time() - t0, 3)
        record_latency(total)

        citations = [{
            "filename":     c["metadata"]["filename"],
            "page":         c["metadata"]["page_number"],
            "ann_score":    round(c.get("ann_score", 0), 4),
            "rerank_score": round(c.get("rerank_score", 0), 4),
            "preview":      c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        } for c in top_chunks]

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": [{
                "rank":         i + 1,
                "text":         c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                "filename":     c["metadata"]["filename"],
                "page":         c["metadata"]["page_number"],
                "ann_score":    round(c.get("ann_score", 0), 4),
                "rerank_score": round(c.get("rerank_score", 0), 4),
            } for i, c in enumerate(top_chunks)],
            "timing": {
                "retrieve_s": t_retrieve,
                "rerank_s":   t_rerank,
                "generate_s": t_generate,
                "total_s":    total,
            },
            "model": OLLAMA_MODEL,
        }

    def get_indexed_documents(self) -> list[dict]:
        if not self.metadata:
            return []
        seen: dict[str, dict] = {}
        for m in self.metadata:
            fn = m.get("filename", "unknown.pdf")
            if fn not in seen:
                seen[fn] = {"filename": fn, "pdf_id": m.get("pdf_id", ""), "chunks": 0, "pages": set()}
            seen[fn]["chunks"] += 1
            seen[fn]["pages"].add(m.get("page_number", 0))
        return sorted([{
            "filename": fn,
            "pdf_id": info["pdf_id"],
            "total_chunks": info["chunks"],
            "total_pages": len(info["pages"]),
        } for fn, info in seen.items()], key=lambda x: x["filename"])


# ── FastAPI App ───────────────────────────────────────────────────────────────
rag: Optional[RAGEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGEngine()
    yield

app = FastAPI(
    title="MedRAG API",
    description="100% Open-Source RAG Chatbot — FAISS HNSW + Cross-Encoder + Llama 3.2 3B via Ollama",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Pydantic Models ───────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    timing: dict
    model: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Starlette 1.2.x: request must be keyword arg, not inside context dict
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/ask")
async def ask(req: AskRequest):
    global _error_count
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="No question provided")
    try:
        result = rag.generate(req.question)
        return JSONResponse(result)
    except Exception as e:
        _error_count += 1
        log.exception("Error in /api/ask")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def metrics():
    return JSONResponse(get_metrics())

@app.get("/api/documents")
async def documents():
    return JSONResponse({"documents": rag.get_indexed_documents()})

@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    """Streaming endpoint — returns SSE tokens as Ollama generates them."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="No question provided")

    def generate_sse():
        global _error_count
        t0 = time.time()
        try:
            # Stage 1 & 2: Retrieve + Rerank (fast)
            candidates = rag.retrieve(req.question, top_k=TOP_K_RETRIEVE)
            top_chunks = rag.rerank(req.question, candidates, top_k=TOP_K_RERANK)
            t_retrieve_rerank = round(time.time() - t0, 3)

            # Send citations immediately (before generation starts)
            citations = [{
                "filename":     c["metadata"]["filename"],
                "page":         c["metadata"]["page_number"],
                "ann_score":    round(c.get("ann_score", 0), 4),
                "rerank_score": round(c.get("rerank_score", 0), 4),
                "preview":      c["text"][:150],
            } for c in top_chunks]

            chunks_data = [{
                "rank": i + 1,
                "filename": c["metadata"]["filename"],
                "page": c["metadata"]["page_number"],
                "ann_score": round(c.get("ann_score", 0), 4),
                "rerank_score": round(c.get("rerank_score", 0), 4),
                "text": c["text"][:250],
            } for i, c in enumerate(top_chunks)]

            yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'chunks': chunks_data, 'retrieve_rerank_s': t_retrieve_rerank})}\n\n"

            # Stage 3: Stream Ollama tokens
            prompt = rag.build_prompt(req.question, top_chunks)
            client = ollama_client.Client(host=OLLAMA_HOST)
            full_answer = ""
            t_gen_start = time.time()

            for chunk in client.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": 0.2,
                    "num_predict": 512,
                    "num_ctx": 2048,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "stop": ["<|eot_id|>"],
                }
            ):
                token = chunk.response
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            total = round(time.time() - t0, 3)
            generate_s = round(time.time() - t_gen_start, 3)
            record_latency(total)

            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'timing': {'retrieve_rerank_s': t_retrieve_rerank, 'generate_s': generate_s, 'total_s': total}, 'model': OLLAMA_MODEL})}\n\n"

        except Exception as e:
            _error_count += 1
            log.exception("Streaming error")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/health")
async def health():
    ollama_ok = False
    try:
        client = ollama_client.Client(host=OLLAMA_HOST)
        models = [m.model for m in client.list().models]
        ollama_ok = any(OLLAMA_MODEL in m for m in models)
    except Exception:
        pass
    return JSONResponse({
        "status": "ok",
        "ollama": ollama_ok,
        "model": OLLAMA_MODEL,
        "vectors_indexed": rag.index.ntotal if rag else 0,
    })

@app.post("/api/ingest")
async def ingest(pdf: UploadFile = File(...)):
    if not pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    save_path = DATA_PATH / pdf.filename
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    content = await pdf.read()
    with open(save_path, "wb") as f:
        f.write(content)
    try:
        from ingest import run_ingestion
        t0 = time.time()
        run_ingestion([save_path], append=True)
        elapsed = round(time.time() - t0, 1)
        rag.index, rag.metadata = rag._load_index()
        return JSONResponse({
            "status": "success",
            "filename": pdf.filename,
            "elapsed_s": elapsed,
            "total_vectors": rag.index.ntotal,
        })
    except Exception as e:
        log.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)