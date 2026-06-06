# MedChat — Interface Guide

> MedChat is an AI-powered medical knowledge assistant. It answers your questions by searching through medical PDF documents and generating cited answers using an open-source AI model (Llama 3.2).

---

## 🗺️ Interface Overview

The interface is divided into **three panels** plus a **top navigation bar**.

```
┌──────────────────────────────────────────────────────────────────┐
│                        TOP BAR                                   │
├──────────────────┬───────────────────────┬───────────────────────┤
│                  │                       │                       │
│   LEFT PANEL     │    CENTER PANEL       │    RIGHT PANEL        │
│  Knowledge Base  │       Chat            │  Retrieved Chunks     │
│                  │                       │                       │
└──────────────────┴───────────────────────┴───────────────────────┘
```

---

## 1. 🔝 Top Bar

The top bar is always visible and shows the system status at a glance.

### Left — Logo
- **MedChat** logo with the AI Knowledge Assistant subtitle.

### Center — Pipeline Badges
These three badges animate in real-time while a query is being processed:

| Badge | What it means |
|---|---|
| 🔍 **Retrieve** | The system is searching the FAISS HNSW vector index to find the top 6 most similar text chunks to your question |
| ↕️ **Rerank** | A cross-encoder AI model re-scores and re-orders the retrieved chunks for maximum relevance |
| ⚡ **Generate** | Llama 3.2:3b is generating the final answer using the top 3 chunks as context |

When active, each badge glows blue. When done, they turn green with a checkmark.

### Right — Toolbar Buttons

| Button | Function |
|---|---|
| 📊 Latency pill | Shows the p95 (95th percentile) response time of the last query |
| ☀️/🌙 Theme | Toggle between **dark mode** (default) and **light mode** |
| 📤 Upload | Toggle the PDF upload section in the left panel |
| 📁 Docs | Shows how many PDFs are indexed in the knowledge base |

---

## 2. 📚 Left Panel — Knowledge Base

### Document List
- Shows every PDF that has been indexed and is searchable.
- For each PDF: filename, number of pages, and number of text chunks.
- Click the **refresh** (↻) button to reload the list.

### Upload Zone
- **Drag & drop** any medical PDF here to add it to the knowledge base.
- Or click **Browse File** to pick from your computer.
- Supports both: native PDFs with embedded text, and scanned PDFs (uses OCR).
- After upload, the PDF is automatically chunked, embedded, and added to the HNSW index.

### Evaluation Metrics
Live performance metrics updated after every query:

| Metric | Meaning |
|---|---|
| **p50 Latency** | Median (50th percentile) response time — "typical" speed |
| **p95 Latency** | 95th percentile — worst-case speed for most queries |
| **Queries** | Total number of questions asked this session |
| **Target** | The system's target response window (2–5 seconds) |
| **Latency bar** | Visual bar below the grid — fills red if over target, green if within |

---

## 3. 💬 Center Panel — Chat

This is the main conversation area.

### Welcome Screen
When no question has been asked yet, you see:
- A welcome message explaining what MedChat does.
- **3 starter question buttons** — click any to instantly send that question.

### Chat Messages

**Your question** appears on the right side with a blue bubble.

**AI Answer** appears on the left with:
- 📝 The main answer text with inline citations like `[med.pdf, p.467]`
- 🏷️ **Citation badges** — colored pills at the bottom showing which pages were used, with ANN and CE scores
- ⏱️ **Timing bar** — shows how long each pipeline stage took: Retrieve → Rerank → Generate → Total

### Pipeline Status Bar (while loading)
Below the chat, three stages animate in sequence:
1. **Retrieving chunks…** — FAISS search running
2. **Reranking…** — cross-encoder scoring
3. **Generating answer…** — Llama 3.2 writing the response

### Input Bar
- **Text area** — type your medical question here (up to 2,000 characters)
- 🎤 **Microphone** — voice input (uses browser's Web Speech API)
- **Ask button** — sends your question

---

## 4. 🔎 Right Panel — Retrieved Chunks

Shows exactly which document passages the AI used to answer your question.

Each chunk card shows:
- **Document name** and **page number** (e.g., `med.pdf p.467`)
- **Rank number** (1 = most relevant after reranking)
- **Text preview** of the actual passage
- **Two score bars:**
  - 🔵 **ANN** (Approximate Nearest Neighbor) — cosine similarity from FAISS vector search
  - 🟢 **CE** (Cross-Encoder) — reranker score, more accurate relevance signal

> **Higher CE score = more relevant to your question.** The AI uses the top 3 chunks by CE score to generate the answer.

---

## 5. 🔄 How a Query Works (End-to-End)

```
You type a question
        │
        ▼
[1] EMBED — Your question → 384-dim vector (all-MiniLM-L6-v2)
        │
        ▼
[2] RETRIEVE — FAISS HNSW searches 968 vectors → top 6 chunks
        │
        ▼
[3] RERANK — Cross-encoder scores each (question, chunk) pair → top 3
        │
        ▼
[4] PROMPT — Build context from top 3 chunks + your question
        │
        ▼
[5] GENERATE — Llama 3.2:3b via Ollama generates cited answer
        │
        ▼
[6] DISPLAY — Answer + citations + chunk cards + timing shown
```

---

## 6. 🎨 Dark / Light Mode

Click the **☀️ sun icon** in the top-right to switch to **light mode**.
Click the **🌙 moon icon** to switch back to **dark mode**.

Your preference is saved automatically and remembered on the next visit.

---

## 7. 📌 Key Terms Explained

| Term | Simple explanation |
|---|---|
| **RAG** | Retrieval-Augmented Generation — AI looks up facts before answering |
| **FAISS HNSW** | A fast vector search engine that finds similar text passages |
| **Chunk** | A ~750-word passage extracted from a PDF |
| **Embedding** | A list of 384 numbers that represents the meaning of text |
| **Cross-Encoder** | A small AI model that scores how well a chunk matches a question |
| **Llama 3.2:3b** | An open-source AI language model running locally via Ollama |
| **p50 / p95** | Percentile latency stats — p95 means 95% of queries are faster than this |
| **ANN score** | Cosine similarity from vector search (0 to 1) |
| **CE score** | Cross-encoder relevance score (can be negative to positive) |
| **OCR** | Optical Character Recognition — reads text from scanned/image PDFs |

---

## 8. ⚠️ Important Notes

> [!IMPORTANT]
> MedChat provides AI-generated summaries from documents. It is **not a substitute for professional medical advice**. Always consult a licensed healthcare provider for medical decisions.

> [!NOTE]
> **First query is slow** — Llama 3.2:3b (2GB model) loads into memory on first use. Subsequent queries are faster.

> [!TIP]
> For faster responses, add a GPU to your system. Llama 3.2:3b runs ~10x faster on GPU than CPU.
