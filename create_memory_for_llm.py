from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import os

# Memory-safe configuration
DATA_PATH = "data/"
CHUNK_SIZE = 300  # Reduced further
CHUNK_OVERLAP = 50
BATCH_SIZE = 20  # Smaller batches

def load_pdf_files(data_path):
    """Load PDFs with error handling"""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
        use_multithreading=False
    )
    return loader.load()

def main():
    print("1. Loading documents...")
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        print("No documents found in data/ directory")
        return
    print(f"   Loaded {len(documents)} pages")

    print("2. Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    print("3. Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("4. Building vector store...")
    # Initialize with first batch
    db = FAISS.from_documents(chunks[:BATCH_SIZE], embeddings)
    
    # Process remaining batches
    for i in tqdm(range(BATCH_SIZE, len(chunks), BATCH_SIZE), desc="Processing"):
        batch = chunks[i:i + BATCH_SIZE]
        db.add_documents(batch)  # CORRECT: No embeddings parameter here
    
    print("5. Saving vector store...")
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Successfully saved {len(chunks)} chunks to {DB_FAISS_PATH}")

if __name__ == "__main__":
    main()