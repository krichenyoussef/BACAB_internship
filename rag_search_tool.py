from langchain.tools import tool
from rag_indexer import search

@tool
def rag_search_tool(query: str) -> str:
    """Search indexed documents using semantic similarity"""
    results = search(query)
    if not results:
        return "Aucun résultat trouvé."
    return "\n".join(results)
