# src/agents/chatbot.py

from src.utils.decision import should_use_rag
from src.services.llm_service import get_llm_response
from src.services.rag_service import get_rag_response

def handle_question(question: str, confidence_threshold: int):
    """
    Applique une stratégie intelligente pour choisir LLM ou RAG.
    """
    use_rag = should_use_rag(question, confidence_threshold)

    if use_rag:
        answer = get_rag_response(question)
        mode = "RAG"
    else:
        answer = get_llm_response(question)
        mode = "LLM"

    return answer, mode
