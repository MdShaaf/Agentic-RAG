import logging
from src.similarity_reranker import similarity_reranker
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

llm = Ollama(
    model="llama3.1",
    temperature=0.1
)

MAX_ATTEMPTS = 3
MAX_CONTEXT_CHARS = 6000

def agentic_rag(question: str, vectorstore):
    """
    Agentic RAG: Iteratively retrieves, rewrites queries, and generates answers
    until sufficient context is found or max attempts reached.
    """
    state = {
        "original_question": question,
        "current_query": question,
        "context": "",
        "attempts": 0
    }

    while state["attempts"] < MAX_ATTEMPTS:
        state["attempts"] += 1
        logger.info(f"Agentic RAG attempt {state['attempts']}/{MAX_ATTEMPTS}")

        # 1️⃣ RETRIEVE relevant documents
        logger.debug(f"Searching with query: {state['current_query']}")
        docs = similarity_reranker(
            vectorstore=vectorstore,
            query=state['current_query']
        )

        # 2️⃣ BUILD context from retrieved docs
        # Handle if docs are tuples (doc, score) or just docs
        if docs and isinstance(docs[0], tuple):
            # Extract documents from (doc, score) tuples
            documents = [doc for doc, score in docs]
        else:
            documents = docs
        
        new_context = "\n\n".join(doc.page_content for doc in documents)
        state["context"] += "\n\n" + new_context
        state["context"] = state["context"][-MAX_CONTEXT_CHARS:]  # Keep recent context
        
        logger.debug(f"Context length: {len(state['context'])} chars")

        # 3️⃣ CHECK if we have enough info to answer
        check_prompt = f"""You are an expert assistant analyzing if you have enough information to answer a question.

Original Question: {state['original_question']}

Available Context:
{state['context']}

Task: Determine if the context contains sufficient information to answer the question comprehensively.

If YES (you can answer): Provide the complete answer.
If NO (need more info): Respond with EXACTLY this text: NEED_MORE_INFO

Your response:"""

        response = llm.invoke(check_prompt).strip()

        # 4️⃣ If we have an answer, return it
        if "NEED_MORE_INFO" not in response:
            logger.info(f"✅ Answer found on attempt {state['attempts']}")
            return response

        # 5️⃣ Otherwise, REWRITE the query for better retrieval
        logger.info("Context insufficient. Rewriting query for next attempt...")
        
        rewrite_prompt = f"""You are a query optimization expert. Rewrite the following question to improve document retrieval.

Original Question: {state['original_question']}

Previously Retrieved Context (didn't fully answer the question):
{state['context'][:500]}...

Task: Generate a NEW search query that:
- Focuses on missing information
- Uses different keywords/synonyms
- Is more specific or explores different angles
- Is optimized for semantic search

Output ONLY the rewritten query, nothing else:"""

        state['current_query'] = llm.invoke(rewrite_prompt).strip()
        logger.debug(f"Rewritten query: {state['current_query']}")

    # Max attempts reached
    logger.warning(f"❌ Max attempts ({MAX_ATTEMPTS}) reached without complete answer")
    return f"I searched through the documents {MAX_ATTEMPTS} times but couldn't find enough information to fully answer: '{question}'. Here's what I found:\n\n{state['context'][:1000]}"

