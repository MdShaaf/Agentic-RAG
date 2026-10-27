import logging
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

llm = Ollama(
    model="llama3.2:1b",
    temperature=0.5
)


def run_rag_agent(question: str, context: str) -> str:
    """Generate an answer using the provided question and context."""

    prompt = f"""
    Answer the question using only the provided context.
    If the context does not contain the answer, say so.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    logger.info("Answer generated successfully.")
    return response.strip()

if __name__ == "__main__":
    # Example usage
    question = "What is the main topic of the document?"
    context = "This document discusses the impact of climate change on global agriculture."
    answer = run_rag_agent(question, context)
    print("Answer:", answer)