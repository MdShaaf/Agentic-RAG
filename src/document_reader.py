from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader, UnstructuredCSVLoader, Docx2txtLoader
import os
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from tqdm import tqdm
import logging

print("document_reader.py is being executed!")

##Logger settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

###Creating Hash for files
def folder_fingerprint(folder_path):
    hasher = hashlib.sha256()

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            stat = os.stat(file_path)
            hasher.update(file.encode())
            hasher.update(str(stat.st_size).encode())
            hasher.update(str(stat.st_mtime).encode())

    return hasher.hexdigest()


import json

FINGERPRINT_FILE = "folder_fingerprint.json"

def load_old_fingerprint():
    if os.path.exists(FINGERPRINT_FILE):
        with open(FINGERPRINT_FILE, "r") as f:
            return json.load(f).get("fingerprint")
    return None

def save_fingerprint(fp):
    with open(FINGERPRINT_FILE, "w") as f:
        json.dump({"fingerprint": fp}, f)


def load_all_documents(path):
    """
    Load all documents from the given path with caching support.
    Checks if folder contents changed and uses cache if unchanged.
    """
    print("load_all_documents function is being called!")
    
    DOC_CACHE_FILE = "cached_documents.pkl"
    
    # Check if folder contents changed
    new_folder_fingerprint = folder_fingerprint(path)
    old_folder_fingerprint = load_old_fingerprint()
    
    if new_folder_fingerprint != old_folder_fingerprint or not os.path.exists(DOC_CACHE_FILE):
        logger.info("Folder contents changed. Reloading documents...")
        
        # Load all documents
        documents = []
        
        # Load PDFs
        pdf_files = list(path.rglob("**.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files.")
        for pdf in tqdm(pdf_files, desc="Loading PDF files"):
            try:
                loader = PyPDFLoader(str(pdf))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf}: {e}")

        # Load TXT files
        txt_files = list(path.rglob("**.txt"))
        logger.info(f"Found {len(txt_files)} TXT files.")
        for txt_file in tqdm(txt_files, desc="Loading TXT files"):
            try:
                loader = TextLoader(str(txt_file))
                loaded = loader.load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load TXT {txt_file}: {e}")

        # Load DOCX files
        docx_files = list(path.rglob("**.docx"))
        logger.info(f"Found {len(docx_files)} DOCX files")
        for docx_file in tqdm(docx_files, desc="Loading DOCX files"):
            try:
                loader = UnstructuredWordDocumentLoader(str(docx_file))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load DOCX {docx_file}: {e}")

        # Load CSV files
        csv_files = list(path.rglob("**.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            try:
                loader = UnstructuredCSVLoader(str(csv_file))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load CSV {csv_file}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        
        # Save to cache
        logger.info("Saving documents to cache...")
        with open(DOC_CACHE_FILE, "wb") as f:
            pickle.dump(documents, f)
        
        save_fingerprint(new_folder_fingerprint)
        
    else:
        # Load from cache
        logger.info("Folder contents unchanged. Loading documents from cache...")
        with open(DOC_CACHE_FILE, "rb") as f:
            documents = pickle.load(f)
        
        logger.info(f"Total documents loaded: {len(documents)}")
    
    return documents


# ✅ Only run this code when the script is executed directly
if __name__ == "__main__":
    # For testing/demo purposes
    path = Path(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\PDF_Video Reader\PDF_Samples")
    documents = load_all_documents(path)
    print(f"Demo: Loaded {len(documents)} documents")