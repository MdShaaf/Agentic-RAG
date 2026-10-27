from sentence_transformers import CrossEncoder
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

##function for similarity reranking


reranker = CrossEncoder("BAAI/bge-reranker-base")
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Reranker model loaded successfully.")
logger.info("Defining similarity_reranker function.")
def similarity_reranker(chunked_docs,vectorstore,query,search_type):
    try:
        # Creating BM25 before the branches so Hybrid can use it
        bm25_retriever = BM25Retriever.from_documents(chunked_docs)
        bm25_retriever.k = 10

        if vectorstore is None:
            raise ValueError("Vectorstore is not initialized.")
          
        
        elif search_type=="Semantic Search":
            logger.info("Starting semantic search and reranking.")
            reranked_results = vectorstore.similarity_search(query=query, k=10)
            pairs = [(query, doc.page_content) for doc in reranked_results]
            scores = reranker.predict(pairs)
            reranked_docs = sorted(
                zip(reranked_results, scores),
                key=lambda x: x[1],
                reverse=True)

        elif search_type=="Keyword Search":
            logger.info("Starting keyword search and reranking.")
            reranked_results = bm25_retriever.invoke(query)
            pairs = [(query, doc.page_content) for doc in reranked_results]
            scores = reranker.predict(pairs)
            reranked_docs = sorted(
                zip(reranked_results, scores),
                key=lambda x: x[1],
                reverse=True)

        elif search_type=="Hybrid Search":
            logger.info("Starting hybrid search and reranking.")
            ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore.as_retriever(search_kwargs={"k": 10}), bm25_retriever],weights=[0.5, 0.5])
            reranked_results = ensemble_retriever.invoke(query)
            pairs = [(query, doc.page_content) for doc in reranked_results]
            scores = reranker.predict(pairs)
            reranked_docs = sorted(
                zip(reranked_results, scores),
                key=lambda x: x[1],
                reverse=True)

        return reranked_docs
    
    except Exception as e:
        logger.error(f"Error during similarity search and reranking: {e}")
        raise e

