rag_context_header = """
RETRIEVED_DOCUMENTS:
------
Use the documents below, retrieved specifically for this query, as supporting material if relevant:
"""

basic_rag_prompt = """
You are an assistant who answers questions using the documents provided.

## Default language of conversation: "{language}"
## Tone with the user: "{voice_tune}"

{rag_context}

Previous conversation:
{chat_history}

Rules:
- Use only the retrieved documents and the conversation above.
- Cite the source file of each fact, e.g. (source: handbook.pdf).
- If the documents do not contain the answer, say you could not find it. Do not guess.

Question: {question}
Answer:
"""


def render_rag_context(docs) -> str:
    """Format retrieved chunks with their source so the LLM can cite them."""
    if not docs:
        return rag_context_header + "\n(no documents found)"
    blocks = [
        f"[{i}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    ]
    return rag_context_header + "\n" + "\n\n".join(blocks)


def render_history(history: list[dict], max_turns: int = 6) -> str:
    turns = history[-max_turns:]
    if not turns:
        return "(none)"
    return "\n".join(f"{'User' if t['role'] == 'human' else 'Assistant'}: {t['content']}" for t in turns)
