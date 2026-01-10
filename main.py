from src.document_reader import load_all_documents
from src.Chunking import chunk_documents
from src.Embedding import build_or_load_faiss
from src.similarity_reranker import similarity_reranker
from src.querry_rewrite import rewrite_query
from src.Agentic_RAG import agentic_rag
import os
from pathlib import Path
path= Path(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\PDF_Video Reader\PDF_Samples")

def main():
    # Load and prepare documents
    documents = load_all_documents(path)
    chunked_docs = chunk_documents(documents)
    embed_store = build_or_load_faiss(chunked_docs)
    
    # Ask question directly - agentic_rag handles everything
    question = "Explain the time series analysis"
    final_answer = agentic_rag(question, embed_store)
    
    print("Final Answer:", final_answer)

if __name__ == "__main__":
    main()