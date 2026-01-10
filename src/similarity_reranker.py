from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
import logging
logger = logging.getLogger(__name__)
logger.info("Reranker model loaded successfully.")
logger.info("Defining similarity_reranker function.")
def similarity_reranker(vectorstore,query):
    try:
        logger.info("Starting similarity search and reranking.")
        reranked_results = vectorstore.similarity_search(query=query, k=10)
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




