"""
app.py
------
Streamlit UI for the Local Privacy Chatbot.

Features:
 - Upload PDF/TXT documents
 - Encrypt and index them locally
 - Query your local LLM (Ollama)
 - View saved docs
 - Clear vector store safely

This app is fully offline and private.
"""
import sys
import os

# -------------------------------------------------------------
# Add project root to Python path so `backend/` modules import
# correctly when running from /web or other directories.
# -------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend import rag_engine, model_runner
import time


# ==============================
# Streamlit Page Configuration
# ==============================
st.set_page_config(page_title="Local Privacy Chatbot", layout="wide")
st.title("🔒 Local Privacy Chatbot — Web UI (Offline)")


# ============================================================
# SIDEBAR — Document Upload, Listing, and Model Selection
# ============================================================
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        accept_multiple_files=False
    )

    if uploaded is not None:
        if st.button("Upload & Index"):
            try:
                bytes_data = uploaded.read()
                rag_engine.add_file(uploaded.name, bytes_data)
                st.success(f"Uploaded and indexed: {uploaded.name}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.header("📂 Indexed Documents")

    # List encrypted documents stored locally
    try:
        docs = rag_engine.list_documents()
    except Exception as e:
        docs = []
        st.error(f"Error listing documents: {e}")

    if docs:
        for d in docs:
            st.write(f"- {d}")

            # Button to decrypt & preview document
            if st.button(f"Decrypt: {d}", key=f"dec_{d}"):
                try:
                    data = rag_engine.decrypt_document_bytes(d)
                    if data is not None:
                        st.code(data.decode("utf-8", errors="ignore")[:4000])
                    else:
                        st.error("Could not decrypt file.")
                except Exception as e:
                    st.error(f"Decrypt failed: {e}")
    else:
        st.write("_No documents indexed yet_")

    st.markdown("---")
    st.write("🤖 Model Selection")

    model_name = st.text_input("Ollama model name", value=model_runner.MODEL)

    if st.button("Set model"):
        model_runner.MODEL = model_name
        st.success(f"Model set to: {model_name}")


# ============================================================
# MAIN AREA — Ask Questions and View Answers
# ============================================================
st.header("💬 Ask a question (uses indexed documents only)")

question = st.text_input("Enter your question here")

col1, col2 = st.columns([2, 1])


# ------------------------------------------------------------
# LEFT COLUMN — QUESTION PROCESSING
# ------------------------------------------------------------
with col1:
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            # Step 1: Retrieve relevant chunks from ChromaDB
            with st.spinner("Retrieving relevant context..."):
                try:
                    results = rag_engine.retrieve_relevant(question, k=4)
                except Exception as e:
                    results = []
                    st.error(f"Error retrieving context: {e}")

            if not results:
                st.info("No relevant context found.")
            else:

                # Build readable context block for the model
                context_display = "\n\n---\n\n".join(
                    [f"[src={r['meta']['source']}] {r['document']}"
                     for r in results]
                )

                # Build LLM prompt
                prompt_text = (
                    "You are a helpful assistant. Use ONLY the context below to answer the question. "
                    "If the answer is not in the context, say you don't know.\n\n"
                    f"Context:\n{context_display}\n\n"
                    f"Question:\n{question}\n\nAnswer:"
                )

                # Display context to the user
                st.subheader("📚 Context (top chunks)")
                for rr in results:
                    dist = rr.get("dist")
                    dist_fmt = f"{dist:.4f}" if isinstance(dist, (float, int)) else "N/A"
                    st.write(f"**Source:** {rr['meta']['source']} — dist={dist_fmt}")
                    st.write(rr["document"])

                # Step 2: Query the local Ollama model
                st.subheader("🤖 Model Answer")
                start = time.time()

                answer = model_runner.run_ollama_query(
                    prompt_text,
                    model_name=model_runner.MODEL,
                    timeout=300
                )

                elapsed = time.time() - start

                # Handle streaming / errors
                if isinstance(answer, str) and (
                    answer.startswith("[Error]") or answer.startswith("[Model error]")
                ):
                    st.error(answer)
                else:
                    st.write(answer)

                st.caption(f"Response time: {elapsed:.2f}s")


# ------------------------------------------------------------
# RIGHT COLUMN — Maintenance Tools
# ------------------------------------------------------------
with col2:
    st.subheader("🧹 Quick Tools")

    # Clear vector DB and encrypted documents
    if st.button("Clear indexed docs (DELETE)"):
        import shutil
        import traceback

        try:
            # Try to stop ChromaDB (best effort)
            client = getattr(rag_engine, "client", None)
            if client:
                try:
                    client.stop()
                except:
                    pass

                try:
                    client._system.stop()
                except:
                    pass

            # Delete folders
            shutil.rmtree(rag_engine.ENCRYPTED_DIR, ignore_errors=True)
            shutil.rmtree(rag_engine.VECTOR_PERSIST_DIR, ignore_errors=True)

            # Recreate clean folders
            rag_engine.ENCRYPTED_DIR.mkdir(parents=True, exist_ok=True)
            rag_engine.VECTOR_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

            st.success("Cleared all documents and vector DB. Restart app!")
        except Exception as e:
            st.error(f"Failed: {e}\n\n{traceback.format_exc()}")

    st.write("Tip: Upload PDFs, then ask questions. Everything stays local.")
