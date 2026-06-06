# MedChat — RAG Chatbot Architecture, Pipeline & Technical Documentation

> **Project:** MedChat — Retrieval-Augmented Generation Medical Chatbot
> **Stack:** 100% Free / Open-Source — FastAPI · FAISS HNSW · Llama 3.2:3b (Ollama)
> **UI:** Three-panel glassmorphism design with dark/light theme toggle

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Full Pipeline Architecture](#2-full-pipeline-architecture)
3. [Stage-by-Stage Breakdown](#3-stage-by-stage-breakdown)
4. [Technology Stack](#4-technology-stack)
5. [Gap Analysis: Before vs After](#5-gap-analysis-before-vs-after)
6. [Bottlenecks Identified & Mitigations](#6-bottlenecks-identified--mitigations)
7. [Non-Functional Requirements Compliance](#7-non-functional-requirements-compliance)
8. [Evaluation & Monitoring](#8-evaluation--monitoring)
9. [Directory Structure](#9-directory-structure)
10. [Running the System](#10-running-the-system)
11. [API Reference](#11-api-reference)
12. [Design Decisions](#12-design-decisions)

---

## 1. System Overview

MedChat is a production-grade **Retrieval-Augmented Generation (RAG) chatbot** built entirely on open-source components. It answers medical questions by searching a corpus of medical PDFs and generating cited answers using a local LLM — **no API keys, no external services, no cost per query**.

### Key Properties

| Property | Value |
|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` (sentence-transformers) — **Apache 2.0 ✅** |
| **Vector Database** | FAISS with **HNSW index** (M=32, efSearch=64) — **MIT ✅** |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` — **Apache 2.0 ✅** |
| **LLM** | **Llama 3.2:3b via Ollama** (local, offline) — **Meta Llama 3 Community ✅** |
| **Web Framework** | **FastAPI + Uvicorn** (async, SSE streaming) — **MIT ✅** |
| **OCR** | Tesseract (fallback for scanned pages) — **Apache 2.0 ✅** |
| **PDF Parsing** | PyMuPDF (`fitz`) — native text extraction |
| **Target Latency** | **2–5s** end-to-end (GPU); ~60–180s on CPU-only |
| **Provenance** | Every answer includes **PDF filename + page number** citations |
| **Streaming** | SSE streaming endpoint — first tokens visible in ~1–2s |

---

## 2. Full Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE INGESTION PIPELINE                           │
│                         (run once; re-run to add PDFs)                       │
└──────────────────────────────────────────────────────────────────────────────┘

  PDF Corpus (≥10 PDFs, each ≥200 pages)
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 1: INGESTION & PREPROCESSING (ingest.py)     │
  │                                                     │
  │  ┌──────────────┐    ┌──────────────────────────┐  │
  │  │ PyMuPDF fitz │    │ Tesseract OCR (fallback)  │  │
  │  │ Native text  │───▶│ < 50 chars native? → OCR  │  │
  │  │ extraction   │    │ (scanned/image pages)      │  │
  │  └──────────────┘    └──────────────────────────┘  │
  │           │                                         │
  │  Clean text: remove headers/footers, normalize      │
  │  whitespace, strip non-printable chars, detect lang │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 2: CHUNKING & METADATA (ingest.py)           │
  │                                                     │
  │  Token-aware sliding window chunker (tiktoken)      │
  │  • Target: 750 tokens  (range: 500–1000 ✅)         │
  │  • Overlap: 15%        (range: 10–30%  ✅)          │
  │                                                     │
  │  Metadata per chunk:                                │
  │  { pdf_id, filename, page_number, chunk_index,      │
  │    token_count, bbox, language }                    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 3: EMBEDDING (ingest.py)                     │
  │                                                     │
  │  Model: all-MiniLM-L6-v2 (384-dim, normalized)     │
  │  Batch size: 64  |  normalize_embeddings=True        │
  │  Persisted: embeddings stored inside FAISS index    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 4: INDEXING (ingest.py)                      │
  │                                                     │
  │  FAISS IndexHNSWFlat                                │
  │  • M = 32 (connections per node)                    │
  │  • efConstruction = 200 (build quality)             │
  │  • Metric: INNER_PRODUCT (≡ cosine sim on normed)   │
  │                                                     │
  │  Outputs:                                           │
  │  • vectorstore/db_faiss_hnsw.index                  │
  │  • vectorstore/metadata.json  (text + all metadata) │
  └─────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE QUERY PIPELINE                              │
│                 (two modes: blocking /api/ask · streaming /api/ask/stream)   │
└──────────────────────────────────────────────────────────────────────────────┘

  User Query (text)
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 5: QUERY EMBEDDING (app.py)                  │
  │  all-MiniLM-L6-v2 → 384-dim normalized vector       │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 6: ANN RETRIEVAL (app.py)                    │
  │                                                     │
  │  FAISS HNSW search (efSearch=64)                    │
  │  → Top-6 candidate chunks + ANN cosine scores       │
  │  → Each result: {text, filename, page, ann_score}   │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 7: RERANKING (app.py)                        │
  │                                                     │
  │  Cross-Encoder: ms-marco-MiniLM-L-6-v2             │
  │  • Score each (query, chunk) pair                   │
  │  • Sort descending by CE score                      │
  │  → Top-3 chunks selected for generation             │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 8: GENERATION (app.py)                       │
  │                                                     │
  │  Prompt (Llama 3.2 chat format):                    │
  │  • System: medical AI with citation instruction     │
  │  • Context: top-3 chunks (max 800 chars each)       │
  │  • Question: user query                             │
  │                                                     │
  │  Ollama → llama3.2:3b:                              │
  │  • temperature=0.2, num_predict=512                 │
  │  • num_ctx=2048 (CPU-optimized context window)      │
  │  • top_k=40, repeat_penalty=1.1                     │
  │                                                     │
  │  ── Streaming mode (/api/ask/stream) ──             │
  │  • Citations sent immediately via SSE               │
  │  • Tokens streamed as generated                     │
  │  • Done event sent on completion                    │
  │                                                     │
  │  Returns:                                           │
  │  { answer, citations, retrieved_chunks, timing }    │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 9: MONITORING & METRICS (app.py)             │
  │                                                     │
  │  Rolling window of last 1000 latencies              │
  │  Tracks: p50, p95, min, max, mean, query count      │
  │  Exposed at: GET /api/metrics                       │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  Web UI (3-panel · dark/light theme · glassmorphism)
  ┌──────────────┬──────────────────┬──────────────────┐
  │    LEFT      │     CENTER       │     RIGHT        │
  │              │                  │                  │
  │  KB Docs     │  Chat Interface  │  Retrieved       │
  │  PDF Upload  │  Citations       │  Chunks          │
  │  Metrics     │  Pipeline Anim   │  ANN + CE scores │
  └──────────────┴──────────────────┴──────────────────┘
```

---

## 3. Stage-by-Stage Breakdown

### Stage 1: Ingestion & Preprocessing

**File:** `ingest.py` → `extract_text_from_page()`, `clean_text()`

| Feature | Implementation |
|---|---|
| Native text extraction | `fitz.Page.get_text("blocks")` — preserves bounding boxes |
| OCR fallback trigger | `len(native_text) < 50 chars` |
| OCR engine | `pytesseract` with DPI=200 for quality |
| Text cleaning | Strip page numbers, collapse whitespace, remove non-ASCII |
| Language detection | `langdetect` on first 500 chars |

### Stage 2: Chunking & Metadata

**File:** `ingest.py` → `chunk_page_text()`

```
Chunk config:
  target_tokens  = 750
  max_tokens     = 1000
  min_tokens     = 100  (skip tiny pages)
  overlap_ratio  = 0.15  →  ~112 token overlap
  tokenizer      = tiktoken cl100k_base
  strategy       = sliding window
```

**Metadata schema per chunk:**
```json
{
  "pdf_id": "medical_textbook",
  "filename": "medical_textbook.pdf",
  "page_number": 42,
  "chunk_index": 127,
  "token_count": 748,
  "bbox": {"x0": 72.0, "y0": 120.4, "x1": 540.2, "y1": 680.1},
  "language": "en",
  "text": "..."
}
```

### Stage 3: Embedding

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Batch size: 64 (memory efficient)
- `normalize_embeddings=True` → cosine similarity via inner product

### Stage 4: HNSW Indexing

| Parameter | Value | Rationale |
|---|---|---|
| Index type | `IndexHNSWFlat` | Fast ANN, no quantization loss |
| M | 32 | Good recall/memory tradeoff |
| efConstruction | 200 | High-quality graph build |
| efSearch | 64 | Runtime speed vs recall balance |
| Metric | `METRIC_INNER_PRODUCT` | = cosine sim on normalized vectors |

### Stage 5 & 6: Query Embedding + ANN Retrieval

- Same embedding model as ingestion (no drift)
- `top_k=6` candidates retrieved from HNSW
- Returns ANN cosine scores for each candidate

### Stage 7: Cross-Encoder Reranking

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores each (query, chunk_text) pair jointly
- Much higher precision than bi-encoder ANN alone
- `top_k=3` selected from the 6 ANN candidates

### Stage 8: Generation via Ollama

**LLM:** Llama 3.2:3b running locally via Ollama

Prompt uses Llama 3 chat format (`<|begin_of_text|>` tokens). Enforces:
- Inline citations: `[filename.pdf, p.X]`
- 2 suggested follow-up questions
- Medical disclaimer

**Streaming endpoint** (`POST /api/ask/stream`):
1. Citations sent immediately (SSE event type: `citations`)
2. Each LLM token sent as generated (SSE event type: `token`)
3. Completion event with full answer + timing (SSE event type: `done`)

### Stage 9: Evaluation & Monitoring

```
GET /api/metrics →
{
  "query_count": 42,
  "error_count": 0,
  "latency": {
    "p50_s": 45.2,
    "p95_s": 98.7,
    "min_s": 32.1,
    "max_s": 142.3,
    "mean_s": 51.4
  },
  "llm": "llama3.2:3b",
  "target_latency_s": "2–5",
  "within_target": false
}
```
> Note: Latency is high on CPU. A GPU drops this to 2–5s target range.

---

## 4. Technology Stack

| Layer | Technology | License | Notes |
|---|---|---|---|
| **Web Framework** | **FastAPI + Uvicorn** | MIT ✅ | Async, SSE streaming, auto Swagger docs |
| PDF Parsing | PyMuPDF (`fitz`) | AGPL | Native text + bbox extraction |
| OCR | Tesseract / pytesseract | Apache 2 ✅ | Fallback for scanned pages |
| Tokenizer | tiktoken (cl100k_base) | MIT ✅ | Accurate token counting for chunks |
| **Embedding Model** | `all-MiniLM-L6-v2` | **Apache 2 ✅** | 384-dim, 22M params, ~90MB |
| **Vector Database** | FAISS HNSW | **MIT ✅** | IndexHNSWFlat, M=32, ef=200 |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | **Apache 2 ✅** | 22M params, ~85MB |
| **LLM** | **Llama 3.2:3b via Ollama** | **Meta Llama 3 Community ✅** | 2GB, fully local, no API key |
| Language Detection | langdetect | Apache 2 ✅ | Per-chunk language tagging |
| Frontend | Vanilla HTML/CSS/JS | — | No frameworks, glassmorphism UI |
| Fonts | Inter, JetBrains Mono | OFL | Google Fonts |
| File Upload | python-multipart | Apache 2 ✅ | FastAPI file handling |

### Open-Source Model Summary

| Model | Size | Purpose | Speed (CPU) |
|---|---|---|---|
| `all-MiniLM-L6-v2` | ~90 MB | Text → 384-dim embedding | ~0.1s/batch |
| `ms-marco-MiniLM-L-6-v2` | ~85 MB | Query-chunk relevance scoring | ~3s/3 pairs |
| `llama3.2:3b` (Q4_K_M) | ~2.0 GB | Medical answer generation | ~60–180s |

---

## 5. Gap Analysis: Before vs After

### ❌ → ✅ Fixed Gaps

| Gap | Before | After |
|---|---|---|
| Multi-PDF support | 1 PDF only | Batch + append mode for any number of PDFs |
| OCR for scanned pages | None | Tesseract fallback per page |
| Chunk size (500–1000 tokens) | 300 chars (~75 tokens) | 750 tokens target, tiktoken-measured |
| Token overlap (10–30%) | 16% char-overlap (not tokens) | 15% token overlap (~112 tokens) |
| Rich metadata per chunk | Page only | filename, page, bbox, chunk_index, language |
| ANN index (HNSW/IVF+PQ) | Flat FAISS | FAISS IndexHNSWFlat (M=32, ef=200) |
| Reranking | None | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| Citations in answer | None | [filename, p.X] inline in every answer |
| Retrieval visualization | None | Right panel: ranked chunks + ANN + CE scores |
| PDF upload via UI | Hardcoded data/ folder | Drag-drop upload → live ingestion |
| Evaluation/monitoring | None | /api/metrics: p50, p95, query count |
| Latency display | None | Per-query timing badge + sidebar bar |
| Document list | None | /api/documents: indexed PDFs + chunk counts |
| Web framework | Flask (sync, no docs) | **FastAPI (async, SSE, Swagger UI at /docs)** |
| LLM vendor lock-in | Gemini API (quota issues) | **Llama 3.2:3b (local, no API key needed)** |
| Streaming responses | None | **/api/ask/stream — tokens visible in ~1s** |
| Dark/Light theme | Dark only | **Theme toggle (☀️/🌙) with localStorage** |
| Health check | None | **/api/health — Ollama + index status** |

---



## 7. Non-Functional Requirements Compliance

| Requirement | Status | Implementation |
|---|---|---|
| Open/Free embedding model | ✅ | `all-MiniLM-L6-v2` (Apache 2) |
| Open/Free vector DB | ✅ | FAISS (MIT) |
| **Open/Free LLM** | ✅ | **Llama 3.2:3b via Ollama (Meta Llama 3)** |
| Scalability (>200 pages/PDF) | ✅ | HNSW index, batch embedding |
| Latency 2–5s | ⚠️ GPU only | Streaming reduces perceived latency; GPU achieves true 2–5s |
| Explainability (sources) | ✅ | Citations in every answer |
| Reproducibility | ✅ | Deterministic chunking (tiktoken), pre-computed embeddings |
| Multi-PDF ingestion | ✅ | `ingest.py` + `/api/ingest` upload |
| Web UI | ✅ | Three-panel glassmorphism UI (dark + light theme) |
| Retrieval visualization | ✅ | Right panel: top-3 chunks with ANN + CE scores |
| Streaming responses | ✅ | SSE endpoint `/api/ask/stream` |
| API documentation | ✅ | Auto Swagger UI at `/docs` (FastAPI) |

---

## 8. Evaluation & Monitoring

### Metrics Tracked

| Metric | Endpoint | Notes |
|---|---|---|
| Query latency p50/p95 | `GET /api/metrics` | Rolling window of 1000 queries |
| Query count | `GET /api/metrics` | Total since server start |
| Error count | `GET /api/metrics` | 5xx errors |
| LLM model name | `GET /api/metrics` | Current Ollama model |
| Per-query timing | Response JSON | retrieve_s, rerank_s, generate_s, total_s |
| ANN score | Retrieved chunks | Cosine similarity 0–1 |
| Cross-encoder score | Retrieved chunks | ms-marco score (approx −10 to +10) |
| Ollama health | `GET /api/health` | Model availability check |
| Vectors indexed | `GET /api/health` | Total FAISS vectors |

### Future Metrics (R@k, MRR, Hallucination Rate)
1. Create a **gold standard QA dataset** from your PDFs
2. Run retrieval and check if ground-truth page is in top-K → compute R@k, MRR
3. Use an LLM judge to score answers vs. source text → hallucination rate
4. Expose via `/api/eval` endpoint

---

## 9. Directory Structure

```
Medchat/
├── ingest.py                    # ★ Ingestion pipeline (PDF → HNSW index)
├── app.py                       # ★ FastAPI server — RAG pipeline + SSE streaming
├── requirements.txt             # Dependencies (FastAPI, Ollama, FAISS, etc.)
├── Dockerfile                   # Container config (tesseract + libgl1)
├── .env                         # OLLAMA_MODEL, OLLAMA_HOST
├── start.ps1                    # Quick startup script for Windows
├── test_query.py                # End-to-end test script
│
├── ARCHITECTURE.md              # This file
├── UI_GUIDE.md                  # Interface guide for users
│
├── data/                        # Drop PDFs here
│   └── med.pdf                  # Currently indexed (968 chunks)
│
├── vectorstore/
│   ├── db_faiss_hnsw.index      # ★ HNSW FAISS index (1.7 MB)
│   ├── metadata.json            # ★ chunk text + metadata (3.0 MB, 968 chunks)
│   └── db_faiss/                # Legacy flat index (fallback)
│
├── templates/
│   └── index.html               # Three-panel UI (MedChat, dark/light theme)
│
└── static/
    ├── styles.css               # Dark/Light glassmorphism design system
    └── script.js                # UI controller (theme, chat, SSE, upload)
```

---

## 10. Running the System

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running (auto-starts on Windows)
- `llama3.2:3b` model pulled

### Step 1: Pull the LLM (one-time)
```powershell
ollama pull llama3.2:3b
```

### Step 2: Activate venv & install dependencies
```powershell
cd C:\Users\nitro\Medchat
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # one-time
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 3: Add PDFs to data/ and run ingestion
```powershell
# Drop PDFs into data/ folder, then:
python ingest.py
```

Expected output:
```
INGESTION SUMMARY
=================================================
  Total PDFs indexed : 1
  Total chunks       : 968
  Embedding model    : sentence-transformers/all-MiniLM-L6-v2
  Index type         : FAISS HNSW (M=32)
  Time elapsed       : 234.4s
=================================================
```

### Step 4: Start the FastAPI server
```powershell
# Ollama is already running (auto-started as Windows service)
python app.py
```

Server starts at: **http://localhost:8000**
Swagger API docs: **http://localhost:8000/docs**

### Step 5: Add a New PDF (without full rebuild)
Via UI: drag-drop in the sidebar upload zone
Or via CLI:
```powershell
python ingest.py  # re-scans all PDFs in data/
```

---

## 11. API Reference

### `POST /api/ask` — Blocking query
```json
Request:  { "question": "What are the symptoms of diabetes?" }
Response: {
  "answer": "Based on the documents: Diabetes presents with... [med.pdf, p.467]",
  "citations": [
    { "filename": "med.pdf", "page": 467, "ann_score": 0.469, "rerank_score": 1.329, "preview": "..." }
  ],
  "retrieved_chunks": [
    { "rank": 1, "text": "...", "filename": "med.pdf", "page": 467, "ann_score": 0.469, "rerank_score": 1.329 }
  ],
  "timing": { "retrieve_s": 0.10, "rerank_s": 3.82, "generate_s": 136.5, "total_s": 140.5 },
  "model": "llama3.2:3b"
}
```

### `POST /api/ask/stream` — SSE streaming query
```
Content-Type: application/json
Body: { "question": "..." }

Response: text/event-stream

event 1: data: {"type":"citations","citations":[...],"chunks":[...],"retrieve_rerank_s":3.9}
event 2: data: {"type":"token","token":"Based"}
event 3: data: {"type":"token","token":" on"}
  ... (one event per token)
eventN:  data: {"type":"done","answer":"...full text...","timing":{...},"model":"llama3.2:3b"}
```

### `GET /api/metrics`
```json
{
  "query_count": 5,
  "error_count": 0,
  "latency": { "p50_s": 141.5, "p95_s": 141.5, "min_s": 141.5, "max_s": 141.5, "mean_s": 141.5 },
  "llm": "llama3.2:3b",
  "target_latency_s": "2–5",
  "within_target": false
}
```

### `GET /api/health`
```json
{
  "status": "ok",
  "ollama": true,
  "model": "llama3.2:3b",
  "vectors_indexed": 968
}
```

### `GET /api/documents`
```json
{
  "documents": [
    { "filename": "med.pdf", "pdf_id": "med", "total_chunks": 968, "total_pages": 595 }
  ]
}
```

### `POST /api/ingest`
```
Content-Type: multipart/form-data
Field: pdf (file, .pdf only)
Response: { "status": "success", "filename": "new.pdf", "elapsed_s": 45.2, "total_vectors": 1936 }
```

---

## 12. Design Decisions

### Why FastAPI over Flask?
FastAPI is async-native, which enables true SSE streaming without blocking other requests. It also provides auto-generated Swagger UI at `/docs`, Pydantic validation, and better performance under load. Flask is synchronous and would block on the ~60–180s Llama inference.

### Why Llama 3.2:3b over Gemini?
Gemini free-tier quotas hit zero (`limit: 0` per day). Llama 3.2:3b via Ollama is:
- **Unlimited** — no API key, no rate limits, no cost per query
- **Private** — data never leaves your machine
- **2GB** — small enough to run on any modern laptop with 8GB RAM

### Why Streaming (SSE)?
Llama 3.2:3b on CPU takes 60–180s for a full response. Without streaming, the user stares at a blank screen. With SSE, citations appear in ~1–2s and tokens stream progressively — dramatically improving perceived responsiveness.

### Why FAISS HNSW over IVF+PQ?
HNSW provides better recall (>95%) without quantization loss. IVF+PQ is better for billion-scale; for 10×200-page PDFs (~40K–100K chunks), HNSW is the right choice.

### Why Cross-Encoder Reranking?
Bi-encoder ANN retrieval (FAISS) has ~70–80% precision. Cross-encoders see the query + chunk jointly, achieving ~90%+ precision. For medical accuracy this tradeoff is essential — the extra ~3–4s is worth it.

### Why tiktoken for Chunk Sizing?
`len(text)` char-counting doesn't correspond to LLM token limits. Using tiktoken with `cl100k_base` encoding ensures chunks are truly 500–1000 tokens as the requirement specifies.

### Why top-3 chunks for Llama vs top-5 for Gemini?
Llama 3.2:3b runs with `num_ctx=2048` on CPU to limit memory and inference time. 3 chunks × 800 chars ≈ 600 tokens, leaving ~1400 tokens for the answer. Gemini Flash handled 128K context, so it could take 5 full chunks.
