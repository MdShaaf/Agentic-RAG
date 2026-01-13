# Agentic-RAG
# AGENTIC RAG – Multi-Document Reader

AGENTIC RAG is an **agent-driven Retrieval-Augmented Generation (RAG)** system designed to read and query multiple document types such as **PDF, CSV, DOCX, and TXT**.  
It enables users to retrieve **guidelines, rules, standards, and regulations** from structured and unstructured documents using a Large Language Model (LLM).

This project focuses on **accuracy, grounded responses, and controlled generation**, making it suitable for domains like **construction guidelines, technical standards, and regulatory documents**.

---

## 🔹 Key Features

- Supports **multiple document formats** (PDF, CSV, DOCX, TXT)
- **Agentic RAG workflow** with iterative reasoning
- Query rewriting for improved retrieval
- Vector-based semantic search using FAISS
- Re-ranking of retrieved chunks for better relevance
- Low-temperature LLM responses for factual consistency
- Modular and extensible architecture
- Simple Streamlit-based interface (optional)

---

## 🔹 Use Cases

- Querying **construction guidelines**
- Retrieving **rules and regulations**
- Technical document analysis
- Internal knowledge base assistant
- Policy and standards interpretation

---

## 🔹 Architecture Overview

```
User Query
   ↓
Query Rewrite (LLM)
   ↓
Vector Retrieval (FAISS)
   ↓
Similarity Re-ranking
   ↓
Agent Decision Loop
   ↓
Final Answer (LLM)
```

---

## 🔹 Tech Stack

### Language & Runtime
- **Python** ≥ 3.13

### LLM
- **Ollama**
- Model: `llama3.1`
- Temperature: `0.1`

### Embeddings
- `sentence-transformers`

### Vector Store
- FAISS

## 🔹 Installation

1. Clone the repository
2. Install dependencies:
```bash
   uv pip compile pyproject.toml -o requirements.txt
   uv pip install -r requirements.txt
```
3. Ensure Ollama is running locally with llama3.1 model

## 🔹 Dependencies

Defined in `pyproject.toml`:

```toml
[project]
name = "my-agentic-rag"
version = "0.1.0"
description = "Documents reader with LLM integration"
requires-python = ">=3.13"
dependencies = [
    "langchain",
    "langchain-community",
    "langchain-text-splitters",
    "sentence-transformers",
    "faiss-cpu",
    "streamlit",
    "pypdf",
    "python-docx",
    "pandas",
    "tqdm",
]
```

---

## 🔹 Project Structure

```
.
├── src/
│   ├── document_reader.py
│   ├── Chunking.py
│   ├── Embedding.py
│   ├── similarity_reranker.py
│   ├── querry_rewrite.py
│   ├── Agentic_RAG.py
│
├── main.py
├── Streamlit.py
├── pyproject.toml
└── README.md
```

---

## 🔹 How It Works

1. Documents are loaded and split into semantic chunks  
2. Chunks are embedded using sentence transformers  
3. Embeddings are stored in FAISS  
4. User queries are rewritten to improve recall  
5. Relevant chunks are retrieved and re-ranked  
6. The agent may retry retrieval if context quality is low  
7. Final response is generated using the LLM  

---

## 🔹 Running the Project

### CLI Mode
```bash
python main.py
```

### Streamlit UI
```bash
streamlit run Stramlit.py
```

Ensure Ollama is running locally before starting the application.

---

## 🔹 Limitations

- Performance depends on document chunking quality
- Large document sets may increase retrieval latency
- No persistent long-term memory across sessions
- Not optimized for production-scale deployments
- Hallucinations are reduced but not fully eliminated

---

## 🔹 Future Improvements

- Graph-based RAG
- Persistent memory across sessions
- Improved reranking models
- Multi-document reasoning
- API deployment

---

## 🔹 License

MIT License

---

## 🔹 Author

**Mohammed Shafeeq**

---

## 🔹 Disclaimer

This project is intended for educational and experimental purposes and is not a certified compliance or legal advisory system.
