from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3.1",
    temperature=0.1
)

##querry re-wrting and retrieval function
def rewrite_query(question, context):
    prompt = f"""
    You are a query rewriting assistant.

    Original question:
    {question}

    Existing context (may be empty):
    {context}

    Rewrite the question into a concise, precise search query
    that would work well for semantic document retrieval.
    Do NOT answer the question.
    Return ONLY the rewritten query.
    """

    rewritten = llm.invoke(prompt).strip()
    return rewritten
