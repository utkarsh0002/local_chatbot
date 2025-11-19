
# 🔒 Local Privacy Chatbot (Offline RAG System)

A fully **local**, **privacy-preserving**, **offline** Retrieval-Augmented-Generation (RAG) system.

This app allows you to:

- Upload **PDF / TXT** files  
- Store them **encrypted at rest**  
- Create embeddings + vector search using **ChromaDB**  
- Ask questions answered using **local LLMs via Ollama**  
- Keep **all data 100% offline**  

---

## 🚀 Features

- 🧠 Local RAG (Retrieval Augmented Generation)  
- 🔐 Encrypted document storage (Fernet AES-128)  
- 📄 PDF extraction via PyMuPDF  
- 🧩 Chunking + embeddings via SentenceTransformer  
- 🗃 Persistent vector DB using Chroma  
- 🤖 Offline LLM inference with Ollama  
- 🌐 Streamlit Web UI  
- 💾 All processing happens on your machine  

---

## 📦 Installation

### 1. Clone repository  
*(convert comment lines to bash code blocks later)*

 ```bash
 git clone https://github.com/YOUR_USERNAME/local_privacy_chatbot.git
 cd local_privacy_chatbot
 ```

---

### 2. Create virtual environment

 ```bash
 python -m venv venv
 ```

### 3. Activate environment

**Windows**

 ```bash
 venv\Scripts\activate
 ```

**Linux/Mac**

 ```bash
 source venv/bin/activate
 ```

---

### 4. Install dependencies

 ```bash
 pip install -r requirements.txt
 ```

---

## 🤖 Install Ollama (Required)

Download from:  
https://ollama.com/download

Pull a model:

 ```bash
 ollama pull mistral
 ```

---

## ▶ Run the Web App

 ```bash
 streamlit run web/app.py
 ```

The UI will open at:

http://localhost:8501

---

## 📁 Project Structure

local_privacy_chatbot/  
│  
├── backend/  
│ ├── rag_engine.py # Encryption, embeddings, vector DB (Chroma)  
│ └── model_runner.py # Ollama LLM interface  
│  
├── web/    
│ └── app.py # Streamlit UI  
│  
├── data/  
│ ├── encrypted_docs/ # Encrypted original documents  
│ └── vector_store/ # ChromaDB persistent store  
│  
├── requirements.txt # Python dependencies  
├── README.md # Project documentation  
└── .gitignore # Git ignored files  


---

## 🧹 Clearing Indexed Data

Inside the UI → **Clear indexed docs (DELETE)**  

Deletes:

- Encrypted docs  
- Vector database  
- Recreates clean directories  

Restart the app after clearing.

---

## 🛡 Security Notes

- All uploaded documents stored **only locally**
- Stored files are **encrypted** using Fernet (AES-128)
- No external services or cloud inference
- Safe for personal notes or sensitive PDFs

---

## ⭐ Future Improvements

If requested, I can extend the repo with:

- Multi-file uploads  
- Real-time progress indicators  
- Search interface  
- Export decrypted documents  
- New embedding models  
- Multi-modal (images + text) support  
- Docker container version  
