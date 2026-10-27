from src.document_reader import load_all_documents
from src.Chunking import chunk_documents
from src.Embedding import build_or_load_faiss
from src.similarity_reranker import similarity_reranker
from src.Agentic_RAG import agentic_rag
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
files_path = BASE_DIR / "PDF_Samples"

def main():
    # Load and prepare documents
    documents = load_all_documents(files_path)
    chunked_docs = chunk_documents(documents)
    embed_store = build_or_load_faiss(chunked_docs)
    reranked_docs = similarity_reranker(chunked_docs, embed_store, "Explain the time series analysis", "Hybrid Search")
    context = []
    for id, doc in enumerate(reranked_docs):
        context.append(doc)

    # context = "\n\n".join(context)
    # Ask question directly - agentic_rag handles everything
    question = "Explain the time series analysis"
    final_answer = agentic_rag(question, context)
    
    print("Final Answer:", final_answer)

if __name__ == "__main__":
    main()