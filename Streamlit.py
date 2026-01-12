import streamlit as st
from pathlib import Path

from src.document_reader import load_all_documents
from src.Chunking import chunk_documents
from src.Embedding import build_or_load_faiss
from src.Agentic_RAG import agentic_rag

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Agentic RAG PDF Reader",
    layout="wide"
)

PDF_PATH = Path(
    r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\PDF_Video Reader\PDF_Samples"
)


# LOAD RESOURCES ONCE
@st.cache_resource(show_spinner=True)
def load_rag_resources():
    documents = load_all_documents(PDF_PATH)
    chunked_docs = chunk_documents(documents)
    embed_store = build_or_load_faiss(chunked_docs)
    return embed_store

# ----------------------------
# UI
# ----------------------------
st.title("📄 Agentic RAG – PDF Reader")

st.markdown("Ask questions from your documents using Agentic RAG")

embed_store = load_rag_resources()

question = st.text_input(
    "Ask a question",
    placeholder="Explain time series analysis..."
)

if st.button("Ask"):
    if question.strip():
        with st.spinner("Searching..."):
            answer = agentic_rag(question, embed_store)

        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
