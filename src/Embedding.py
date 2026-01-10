
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import os
from langchain_community.vectorstores import FAISS, Chroma
import logging

logger = logging.getLogger(__name__)
class HFEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True
        ).tolist()
    


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
        trust_remote_code=True)


# ---------------- FAISS Handler ----------------
def build_or_load_faiss(chunked_documents, persist_dir="faiss_minilm"):
    embeddings = HFEmbeddings(embedding_model)

    if not os.path.exists(persist_dir):
        try:
            os.makedirs(persist_dir)
            logger.info("Creating new FAISS vectorstore...")
            vectorstore = FAISS.from_documents(
            documents=chunked_documents,
            embedding=embeddings)
            vectorstore.save_local(persist_dir)
        except Exception as e:
            logger.error(f"Failed to create FAISS vectorstore: {e}")
            raise e

    else:

        try:
            logger.info("Loading existing FAISS vectorstore...")
            vectorstore = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Failed to load FAISS vectorstore: {e}")
            raise e

        # ADD NEW DOCUMENTS
        vectorstore.add_documents(chunked_documents)
        vectorstore.save_local(persist_dir)

    return vectorstore