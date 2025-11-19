# =======================
# LOCAL PRIVACY CHATBOT
# Using: Mistral 7B (Ollama)
# Vector DB: Chroma
# Encryption: Fernet
# Terminal UI: Rich
# =======================

import os
from pathlib import Path
import time
import subprocess
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import prompt

from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer
import chromadb


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
ENCRYPTED_DIR = DATA_DIR / "encrypted_docs"
VECTOR_DIR = DATA_DIR / "vector_store"
KEY_FILE = DATA_DIR / "key.key"

MODEL_NAME = "mistral"

console = Console()

# Create folders
DATA_DIR.mkdir(exist_ok=True)
ENCRYPTED_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ENCRYPTION SETUP
# ─────────────────────────────────────────────
def load_or_create_key():
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    return key

FERNET_KEY = load_or_create_key()
fernet = Fernet(FERNET_KEY)


def encrypt(data: bytes) -> bytes:
    return fernet.encrypt(data)

def decrypt(data: bytes) -> bytes:
    return fernet.decrypt(data)


# ─────────────────────────────────────────────
# VECTOR DB SETUP (CHROMA)
# ─────────────────────────────────────────────
console.print("[bold green]Loading sentence-transformer model...[/bold green]")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(VECTOR_DIR))

if "docs" in [c.name for c in client.list_collections()]:
    collection = client.get_collection("docs")
else:
    collection = client.create_collection("docs")


def embed(texts):
    return embedder.encode(texts).tolist()


# ─────────────────────────────────────────────
# DOCUMENT PROCESSING
# ─────────────────────────────────────────────
def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks


def add_document(name, text):
    chunks = chunk_text(text)
    ids = [f"{name}_{i}" for i in range(len(chunks))]
    embeddings = embed(chunks)

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": name}] * len(chunks)
    )


# ─────────────────────────────────────────────
# OLLAMA MODEL CALL
# ─────────────────────────────────────────────
def run_local_model(prompt_text):
    """Runs Mistral locally via Ollama."""
    cmd = ["ollama", "run", MODEL_NAME]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    # Send prompt
    proc.stdin.write(prompt_text)
    proc.stdin.close()

    output = proc.stdout.read()
    return output.strip()


# ─────────────────────────────────────────────
# MAIN CHATBOT LOOP
# ─────────────────────────────────────────────
def main():
    console.print(Panel("[bold cyan]LOCAL PRIVACY CHATBOT (Mistral 7B)[/bold cyan]"))

    while True:
        cmd = prompt("Command (upload/query/list/exit): ").strip().lower()

        if cmd == "upload":
            name = prompt("Enter document name: ")
            console.print("Paste document content. Enter a single line with ONLY '<<END>>' to finish.")
            lines = []
            while True:
                line = input()
                if line.strip() == "<<END>>":
                    break
                lines.append(line)

            text = "\n".join(lines)

            # encrypt file
            encrypted = encrypt(text.encode())
            with open(ENCRYPTED_DIR / name, "wb") as f:
                f.write(encrypted)

            # index vector
            add_document(name, text)

            console.print(f"[green]Document '{name}' added successfully![/green]")

        elif cmd == "list":
            console.print("[yellow]Encrypted documents:[/yellow]")
            for f in ENCRYPTED_DIR.iterdir():
                console.print(f"- {f.name}")

        elif cmd == "query":
            question = prompt("Ask a question: ")

            # retrieve vectors
            results = collection.query(
                query_embeddings=embed([question]),
                n_results=3
            )

            context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

            prompt_text = f"""
You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Answer:
"""

            console.print("[cyan]Thinking...[/cyan]")
            answer = run_local_model(prompt_text)
            console.print(Panel(answer))

        elif cmd == "exit":
            console.print("[bold red]Exiting...[/bold red]")
            break

        else:
            console.print("[red]Invalid command![/red]")


if __name__ == "__main__":
    main()
