"""
ingest.py — Full RAG Ingestion Pipeline
========================================
Implements all 4 ingestion stages per requirement doc:
  1. PDF text extraction (PyMuPDF native + Tesseract OCR fallback)
  2. Text cleaning / normalization
  3. Token-aware chunking with rich metadata (500–1000 tokens, 15% overlap)
  4. Embedding + FAISS HNSW index build & persist

Usage:
    python ingest.py                   # process all PDFs in data/
    python ingest.py --pdf data/x.pdf  # process a single PDF (append mode)
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import tiktoken
import numpy as np
from tqdm import tqdm
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image
import io

# Optional OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = Path("data")
VECTORSTORE_PATH = Path("vectorstore")
METADATA_PATH = VECTORSTORE_PATH / "metadata.json"
FAISS_INDEX_PATH = str(VECTORSTORE_PATH / "db_faiss_hnsw.index")
FAISS_LEGACY_PATH = str(VECTORSTORE_PATH / "db_faiss")  # kept for backward compat

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "cl100k_base"   # tiktoken encoder for token counting

CHUNK_TARGET_TOKENS = 750         # target chunk size (between 500–1000 tokens)
CHUNK_MAX_TOKENS    = 1000
CHUNK_MIN_TOKENS    = 100
OVERLAP_RATIO       = 0.15        # 15% overlap (within 10–30% range)
BATCH_SIZE          = 64          # embedding batch size

HNSW_M              = 32          # HNSW connections per node
HNSW_EF_CONSTRUCTION = 200        # HNSW build-time search depth
HNSW_EF_SEARCH      = 64         # runtime search depth

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Tokenizer ────────────────────────────────────────────────────────────────
_tokenizer = tiktoken.get_encoding(TOKENIZER_MODEL)

def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))

# ── PDF Extraction ────────────────────────────────────────────────────────────
def extract_text_from_page(page: fitz.Page, pdf_filename: str, page_num: int) -> dict:
    """
    Extract text from a single page.
    Falls back to OCR if native text is sparse (< 50 chars).
    Returns dict with text and bounding-box info.
    """
    blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,block_type)
    native_text = " ".join(
        b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()
    )
    native_text = clean_text(native_text)

    bbox_list = [
        {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}
        for b in blocks if b[6] == 0 and b[4].strip()
    ]
    first_bbox = bbox_list[0] if bbox_list else None

    # OCR fallback for scanned / image pages
    if len(native_text) < 50 and OCR_AVAILABLE:
        try:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img, lang="eng")
            ocr_text = clean_text(ocr_text)
            log.debug(f"OCR fallback — {pdf_filename} p.{page_num}: {len(ocr_text)} chars")
            return {
                "text": ocr_text,
                "source": "ocr",
                "bbox": first_bbox,
            }
        except Exception as e:
            log.warning(f"OCR failed on {pdf_filename} p.{page_num}: {e}")

    return {
        "text": native_text,
        "source": "native",
        "bbox": first_bbox,
    }

def clean_text(text: str) -> str:
    """Normalize, remove headers/footers artifacts, collapse whitespace."""
    # Remove page numbers at start/end of line (e.g. "\n  42\n")
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def detect_language(text: str) -> str:
    try:
        return detect(text[:500]) if text.strip() else "unknown"
    except LangDetectException:
        return "unknown"

# ── Token-Aware Chunking ──────────────────────────────────────────────────────
def chunk_page_text(
    text: str,
    pdf_id: str,
    filename: str,
    page_num: int,
    bbox: Optional[dict],
    language: str,
    chunk_index_start: int = 0,
) -> list[dict]:
    """
    Split page text into token-aware chunks with overlap.
    Each chunk carries full metadata per requirement spec.
    """
    tokens = _tokenizer.encode(text)
    total = len(tokens)

    if total == 0:
        return []

    # If page fits in one chunk
    if total <= CHUNK_MAX_TOKENS:
        if total < CHUNK_MIN_TOKENS:
            return []   # skip very short pages
        return [{
            "text": text,
            "metadata": {
                "pdf_id": pdf_id,
                "filename": filename,
                "page_number": page_num,
                "chunk_index": chunk_index_start,
                "token_count": total,
                "bbox": bbox,
                "language": language,
            }
        }]

    # Sliding-window split
    overlap_tokens = int(CHUNK_TARGET_TOKENS * OVERLAP_RATIO)
    step = CHUNK_TARGET_TOKENS - overlap_tokens
    chunks = []
    idx = chunk_index_start
    pos = 0

    while pos < total:
        end = min(pos + CHUNK_TARGET_TOKENS, total)
        window_tokens = tokens[pos:end]
        chunk_text = _tokenizer.decode(window_tokens).strip()

        if len(chunk_text) >= 30:   # sanity filter
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "pdf_id": pdf_id,
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": idx,
                    "token_count": len(window_tokens),
                    "bbox": bbox,
                    "language": language,
                }
            })
            idx += 1

        if end == total:
            break
        pos += step

    return chunks

# ── Embedding ─────────────────────────────────────────────────────────────────
def load_embedding_model() -> SentenceTransformer:
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model

def embed_chunks(model: SentenceTransformer, chunks: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)

# ── FAISS HNSW Index ──────────────────────────────────────────────────────────
def build_hnsw_index(dim: int) -> faiss.IndexHNSWFlat:
    """Build FAISS HNSW flat index (inner-product for normalized embeddings = cosine sim)."""
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index

def load_existing_metadata() -> list[dict]:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_index_and_metadata(index: faiss.Index, all_chunks: list[dict]):
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    metadata = [c["metadata"] | {"text": c["text"]} for c in all_chunks]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info(f"Saved HNSW index → {FAISS_INDEX_PATH}")
    log.info(f"Saved metadata   → {METADATA_PATH} ({len(metadata)} chunks)")

# ── Main Ingestion ────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: Path, existing_filenames: set[str] | None = None) -> list[dict]:
    """Process a single PDF → returns list of chunk dicts."""
    filename = pdf_path.name

    if existing_filenames and filename in existing_filenames:
        log.info(f"Skipping already-indexed: {filename}")
        return []

    log.info(f"Processing: {filename}")
    pdf_id = pdf_path.stem.lower().replace(" ", "_")
    chunks = []
    chunk_idx = 0

    with fitz.open(str(pdf_path)) as doc:
        num_pages = doc.page_count
        log.info(f"  Pages: {num_pages}")

        for page_num in tqdm(range(num_pages), desc=f"  {filename}", leave=False):
            page = doc[page_num]
            result = extract_text_from_page(page, filename, page_num + 1)
            text = result["text"]

            if not text:
                continue

            lang = detect_language(text)
            page_chunks = chunk_page_text(
                text=text,
                pdf_id=pdf_id,
                filename=filename,
                page_num=page_num + 1,
                bbox=result["bbox"],
                language=lang,
                chunk_index_start=chunk_idx,
            )
            chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

    log.info(f"  → {len(chunks)} chunks from {filename}")
    return chunks

def run_ingestion(pdf_paths: list[Path], append: bool = False):
    """Full pipeline: extract → chunk → embed → index → save."""
    t0 = time.time()

    # Load existing state if appending
    existing_chunks = []
    existing_filenames: set[str] = set()
    existing_index = None

    if append and METADATA_PATH.exists():
        existing_chunks_meta = load_existing_metadata()
        existing_filenames = {m["filename"] for m in existing_chunks_meta}
        existing_chunks = [
            {"text": m.pop("text"), "metadata": m}
            for m in existing_chunks_meta
        ]
        if Path(FAISS_INDEX_PATH).exists():
            existing_index = faiss.read_index(FAISS_INDEX_PATH)
        log.info(f"Append mode: {len(existing_chunks)} existing chunks, {len(existing_filenames)} PDFs")

    # Ingest all PDFs
    new_chunks = []
    for pdf_path in pdf_paths:
        new_chunks.extend(ingest_pdf(pdf_path, existing_filenames if append else None))

    if not new_chunks:
        log.warning("No new chunks produced. Exiting.")
        return

    all_chunks = existing_chunks + new_chunks

    # Embed new chunks
    model = load_embedding_model()
    new_embeddings = embed_chunks(model, new_chunks)
    dim = new_embeddings.shape[1]

    # Build or extend HNSW index
    if existing_index is not None and append:
        index = existing_index
        index.add(new_embeddings)
    else:
        index = build_hnsw_index(dim)
        if existing_chunks:
            # Re-embed existing (rare: only when rebuilding)
            existing_embeddings = embed_chunks(model, existing_chunks)
            index.add(existing_embeddings)
        index.add(new_embeddings)

    log.info(f"HNSW index total vectors: {index.ntotal}")
    save_index_and_metadata(index, all_chunks)

    elapsed = time.time() - t0
    log.info(f"✅ Ingestion complete in {elapsed:.1f}s — {len(all_chunks)} total chunks")

    # Print summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"  Total PDFs indexed : {len({c['metadata']['filename'] for c in all_chunks})}")
    print(f"  Total chunks       : {len(all_chunks)}")
    print(f"  New chunks added   : {len(new_chunks)}")
    print(f"  Embedding model    : {EMBEDDING_MODEL}")
    print(f"  Index type         : FAISS HNSW (M={HNSW_M})")
    print(f"  Time elapsed       : {elapsed:.1f}s")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF (append mode)")
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild")
    args = parser.parse_args()

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            log.error(f"PDF not found: {pdf_path}")
            exit(1)
        run_ingestion([pdf_path], append=not args.rebuild)
    else:
        pdf_files = list(DATA_PATH.glob("*.pdf"))
        if not pdf_files:
            log.error(f"No PDFs found in {DATA_PATH}/")
            exit(1)
        log.info(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")
        run_ingestion(pdf_files, append=not args.rebuild)
