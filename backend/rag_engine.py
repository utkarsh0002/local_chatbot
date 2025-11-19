"""
rag_engine.py
-------------
Handles:
 - PDF/text extraction
 - Encryption & decryption of stored docs
 - Text chunking for embedding
 - Vector embeddings (SentenceTransformer)
 - Local ChromaDB persistent vector store
 - Searching for relevant chunks
 - Clearing & resetting the vector index safely

Notes:
 - ONNX-based Chroma embedding functions are disabled to prevent
   onnxruntime dependency crashes.
 - Chroma is used via the NEW API (PersistentClient).
 - Fully compatible with new Chroma releases.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as ef
import fitz  # PyMuPDF

# ===============================
# Disable Chroma ONNX embedding
# ===============================
# Prevents ONNXRuntime errors on Windows
ef.DefaultEmbeddingFunction = lambda: None

# ===============================
# Paths & config
# ===============================
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
ENCRYPTED_DIR = DATA_DIR / "encrypted_docs"
VECTOR_PERSIST_DIR = DATA_DIR / "vector_store" / "chroma"
KEY_FILE = DATA_DIR / "fernet.key"
COLLECTION_NAME = "docs"

# Ensure directories exist
for p in (DATA_DIR, ENCRYPTED_DIR, VECTOR_PERSIST_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ===============================
# Encryption utilities
# ===============================
def load_or_create_key() -> bytes:
    """Load encryption key from disk or create a new one."""
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    return key

FERNET_KEY = load_or_create_key()
fernet = Fernet(FERNET_KEY)

def encrypt_bytes(b: bytes) -> bytes:
    return fernet.encrypt(b)

def decrypt_bytes(token: bytes) -> bytes:
    return fernet.decrypt(token)

# ===============================
# PDF text extraction
# ===============================
def extract_pdf_text_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# ===============================
# Embedding model
# ===============================
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate vector embeddings for list of texts."""
    return embedder.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    ).tolist()

# ===============================
# ChromaDB initialization
# ===============================
client = PersistentClient(path=str(VECTOR_PERSIST_DIR))

# Create or get collection
existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(COLLECTION_NAME)

# ===============================
# Chunking
# ===============================
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

# ===============================
# Add document to DB
# ===============================
def add_text_document(doc_name: str, text: str) -> None:
    """Encrypt, chunk, embed & index text."""
    # Store encrypted
    (ENCRYPTED_DIR / doc_name).write_bytes(
        encrypt_bytes(text.encode("utf-8"))
    )

    chunks = chunk_text(text)
    if not chunks:
        return

    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]
    embs = embed_texts(chunks)

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metas,
        embeddings=embs
    )

def add_file(file_name: str, content_bytes: bytes) -> None:
    """Add PDF or plain text file."""
    if file_name.lower().endswith(".pdf"):
        text = extract_pdf_text_bytes(content_bytes)
    else:
        text = content_bytes.decode("utf-8", errors="ignore")
    add_text_document(file_name, text)

# ===============================
# Retrieve documents
# ===============================
def list_documents() -> List[str]:
    return [p.name for p in ENCRYPTED_DIR.iterdir()]

def retrieve_relevant(query: str, k: int = 4) -> List[Dict]:
    """Return top-K most relevant chunks."""
    qvec = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    if results and results.get("documents"):
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results.get("distances", [[]])[0],
        ):
            output.append({"document": doc, "meta": meta, "dist": dist})

    return output

def decrypt_document_bytes(name: str) -> Optional[bytes]:
    path = ENCRYPTED_DIR / name
    if not path.exists():
        return None
    return decrypt_bytes(path.read_bytes())

# ===============================
# Clear & reinitialize vector DB
# ===============================
def clear_index():
    """
    Fully reset vector store:
     - Force-stop Chroma
     - Delete vector DB directory
     - Reinitialize client + collection
    """
    global client, collection

    import shutil

    try:
        # Try stopping Chroma cleanly
        try:
            client._system.stop()
        except Exception:
            pass

        # Remove vector store directory
        shutil.rmtree(VECTOR_PERSIST_DIR, ignore_errors=True)

        # Recreate directory
        VECTOR_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        # New client instance
        client = PersistentClient(path=str(VECTOR_PERSIST_DIR))

        # Fresh collection
        collection = client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )

        return True

    except Exception as e:
        return f"Error clearing index: {str(e)}"
