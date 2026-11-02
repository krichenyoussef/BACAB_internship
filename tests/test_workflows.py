from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.connectors.graph_cove_rag import ChatbotGraphOrchestrator_RAG, extract_tagged_sections
from src.connectors.graph_react_judge import ChatbotGraphOrchestrator_ReActJudge
from tests.conftest import FakeVectorStore

CONTEXT = "Every full-time employee receives 25 days of paid annual leave per year."


def test_extract_tagged_sections():
    text = "<relevant_section> a </relevant_section> noise <relevant_section>b</relevant_section>"
    assert extract_tagged_sections(text) == ["a", "b"]


def test_cove_returns_baseline_when_no_verification_needed():
    llm = FakeListChatModel(
        responses=[
            "<relevant_section>Employees receive 25 days of paid leave.</relevant_section>",  # tag relevance
            "You get 25 days of paid leave per year.",  # baseline answer
            "NO VERIFICATION NEEDED",  # verification questions
        ]
    )
    state = ChatbotGraphOrchestrator_RAG(llm).run("How many leave days?", rag_context=CONTEXT)
    assert state["final_response"] == "You get 25 days of paid leave per year."
    assert state["is_verified"] is True


def test_cove_revises_answer_when_verification_fails():
    llm = FakeListChatModel(
        responses=[
            "<relevant_section>Employees receive 25 days of paid leave.</relevant_section>",
            "You get 20 days of paid leave.",  # wrong baseline
            "1. Does the context say 20 days?",  # verification question
            "No. The context says 25 days.",  # verification answer
            "NO",  # quality check fails
            "You get 25 days of paid leave per year.",  # revised answer
        ]
    )
    state = ChatbotGraphOrchestrator_RAG(llm).run("How many leave days?", rag_context=CONTEXT)
    assert state["is_verified"] is False
    assert state["final_response"] == "You get 25 days of paid leave per year."


def test_react_judge_scores_grounded_answer():
    react_llm = FakeListChatModel(
        responses=[
            "Thought: I should search the handbook.\nAction: similarity_search\nAction Input: annual leave days",
            "Thought: I now know the final answer\nFinal Answer: 25 days per year (source: company_handbook.md).",
        ]
    )
    reranker = FakeListChatModel(responses=["[1] (source: company_handbook.md)\n" + CONTEXT])
    judge = FakeListChatModel(responses=['{"score": 9, "remarks": "Grounded in the handbook."}'])
    store = FakeVectorStore()

    orchestrator = ChatbotGraphOrchestrator_ReActJudge(store, react_llm, judge, reranker_llm=reranker)
    state = orchestrator.run("How many leave days do I get?")

    assert store.queries == ["annual leave days"]
    assert state["response"].startswith("25 days per year")
    assert "25 days" in state["rag_context"]  # judge saw the retrieved context
    assert state["score"] == 9


def test_react_judge_survives_unparseable_judge_output():
    react_llm = FakeListChatModel(responses=["Thought: I can answer.\nFinal Answer: I could not find it."])
    judge = FakeListChatModel(responses=["not json"])
    state = ChatbotGraphOrchestrator_ReActJudge(FakeVectorStore(), react_llm, judge).run("Unknown question?")
    assert state["score"] is None
