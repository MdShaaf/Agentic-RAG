from langchain_text_splitters import RecursiveCharacterTextSplitter
from copy import deepcopy

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_docs = []

    for parent_doc_id, doc in enumerate(documents):
        chunks = splitter.split_documents([doc])

        for chunk_id, chunk in enumerate(chunks):
            #  Important: copy metadata so chunks don't share references
            chunk.metadata = deepcopy(chunk.metadata)

            # Add chunk-level metadata
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "parent_doc_id": parent_doc_id,
            })

            chunked_docs.append(chunk)

    return chunked_docs
