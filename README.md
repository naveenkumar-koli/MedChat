# 🩺 MedChat — Open-Source Medical RAG Chatbot

MedChat is a premium, 100% open-source Retrieval-Augmented Generation (RAG) chatbot designed for accurate medical document Q&A. It leverages a modern local AI stack featuring FastAPI, FAISS HNSW, Sentence-Transformers, a Cross-Encoder Reranker, and Llama 3.2 via Ollama.

---

## 🚀 Key Features

*   **Three-Panel Premium UI:**
    *   *Left:* Document management (refresh documents, upload PDFs) & active metrics (P50/P95 latency).
    *   *Center:* Modern chatbot interface featuring full markdown support, dynamic inline citations, follow-up questions, and theme toggling (Light/Dark mode).
    *   *Right:* Interactive retrieval visualization showing retrieved chunks ranked by cross-encoder score with score bar overlays.
*   **Pipeline Animator:** Visualizes active stages of the search in real-time (`Retrieve` ➔ `Rerank` ➔ `Generate`).
*   **State-of-the-Art RAG Pipeline:**
    *   **Text Extraction:** Native text parsing using PyMuPDF (`fitz`) with a fallback to **Tesseract OCR** for scanned PDFs.
    *   **Vector Database:** FAISS with HNSW index representation (`IndexHNSWFlat`) for sub-millisecond retrieval.
    *   **Semantic Reranker:** Re-evaluates retrieval candidates using the `cross-encoder/ms-marco-MiniLM-L-6-v2` model.
    *   **Local LLM:** Fully offline response generation using `llama3.2:3b` via Ollama.
*   **Live Latency Metrics:** Automatically calculates running P50 and P95 latency stats displayed directly in the dashboard UI.

---

## 🛠️ Architecture

```
                       ┌─────────────────────────┐
                       │   PDF Documents (data/) │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Ingestion (ingest.py)   │
                       │ ├─ PyMuPDF / OCR        │
                       │ ├─ Token Chunking       │
                       │ └─ FAISS HNSW Embed     │
                       └────────────┬────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        RAG Engine (app.py)                             │
│                                                                        │
│   Query  ──➔  HNSW Retrieve  ──➔  Rerank (Cross-Encoder)  ──➔  Prompt  │
│                     │                       │                     │    │
│                     ▼                       ▼                     ▼    │
│              (FAISS HNSW)             (MiniLM-L-6)            (Ollama) │
└────────────────────────────────────────────────────────────────────────┘
```

For a detailed walkthrough, see [ARCHITECTURE.md](file:///c:/Users/nitro/Medchat/ARCHITECTURE.md).

---

## 💻 Prerequisites

Ensure you have the following installed on your machine:
*   [Python 3.10+](https://www.python.org/)
*   [Ollama](https://ollama.com/) (running locally)
*   *Optional:* [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (if you intend to ingest scanned PDFs)

---

## ⚙️ Setup & Installation

### 1. Configure Ollama
Make sure Ollama is running, and pull the required Llama model:
```powershell
ollama pull llama3.2:3b
```

### 2. Set Up Virtual Environment & Dependencies
Initialize and activate your virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:
```powershell
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory:
```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
```

---

## 📂 Ingesting Documents

1. Place your target medical PDF documents in the `data/` directory.
2. Run the ingestion pipeline to build the FAISS HNSW index:
```powershell
python ingest.py
```
This generates the index files inside the `vectorstore/` directory.

---

## 🏃 Running the Chatbot

Start the FastAPI application:
```powershell
python app.py
```

Open your browser and navigate to:
*   **Web Interface:** [http://localhost:8000](http://localhost:8000)
*   **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Repository Structure

*   [app.py](file:///c:/Users/nitro/Medchat/app.py): FastAPI application server, endpoints, and LLM prompt builder.
*   [ingest.py](file:///c:/Users/nitro/Medchat/ingest.py): Automated text extraction, OCR, token chunking, and FAISS vector database builder.
*   [templates/index.html](file:///c:/Users/nitro/Medchat/templates/index.html): Custom Jinja2 main page layout.
*   [static/styles.css](file:///c:/Users/nitro/Medchat/static/styles.css): Premium UI styling sheet.
*   [static/script.js](file:///c:/Users/nitro/Medchat/static/script.js): Frontend controller managing web-sockets/requests, voice typing, and citations.
*   [ARCHITECTURE.md](file:///c:/Users/nitro/Medchat/ARCHITECTURE.md): Core system architecture design doc.
*   [UI_GUIDE.md](file:///c:/Users/nitro/Medchat/UI_GUIDE.md): Guide detailing user-interface components.

---

## 🛡️ License

This project is open-source under the MIT License.
