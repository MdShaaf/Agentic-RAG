
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR/"Cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
class HFEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts,normalize_embeddings=True,show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text,normalize_embeddings=True).tolist()
    


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2")


# ---------------- FAISS Handler ----------------
def build_or_load_faiss(chunked_documents):
    embeddings = HFEmbeddings(embedding_model)

    # rebuild if the folder doesn't exist OR if our fingerprint says data changed
    try:
        logger.info("Creating new FAISS vectorstore for the uploaded PDFs...")
        vectorstore = FAISS.from_documents(documents=chunked_documents,embedding=embeddings)

    except Exception as e:
        logger.error(f"Failed while embedding the uploaded PDFs {e}")
        raise e

    return vectorstore

if __name__ == "__main__":
    # Example usage
    print("Function Not configured to work indepently, please follow App.py for more details")